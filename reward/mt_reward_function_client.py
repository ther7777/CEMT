"""Batch reward client for verl.

This module exposes reward functions that can be referenced from verl GRPO configs.
It orchestrates multiple reward components (BLEU / XCOMET / Kiwi / CoT) and provides
basic output-format validation for CoT-style generations.
"""
import os
import re
import requests
import logging
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import sacrebleu
import numpy as np
import math
import fcntl

# Logging / service endpoints.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Service URLs are injected via environment variables.
XCOMET_SERVER_URL = os.environ.get("XCOMET_SERVER_URL")
KIWI_SERVER_URL = os.environ.get("KIWI_SERVER_URL")
COT_EVALUATOR_SERVER_URL = os.environ.get("COT_EVALUATOR_SERVER_URL")

logger.info(f"[reward-client] XCOMET server: {XCOMET_SERVER_URL or 'not set'}")
logger.info(f"[reward-client] Kiwi server: {KIWI_SERVER_URL or 'not set'}")
logger.info(f"[reward-client] CoT evaluator server: {COT_EVALUATOR_SERVER_URL or 'not set'}")


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def check_think_format_with_details(think_content: str, required_tags: List[str] = None) -> Tuple[bool, str]:
    if not think_content or not think_content.strip(): 
        return False, "empty <think> content"
    
    # If required_tags is not provided, validate against the default full tag set.
    if required_tags is None:
        target_tags = [
            "holistic_semantics_pragmatics_analysis", 
            "argument_predicate_analysis",
            "syntactic_structure_analysis", 
            "translation_strategy_formulation"
        ]
    else:
        target_tags = required_tags
    
    missing_tags = []
    empty_tags = []
    
    for tag in target_tags:
    # f-string matches the specific tag name.
        match = re.search(f'<{tag}>(.*?)</{tag}>', think_content, re.DOTALL)
        if not match:
            missing_tags.append(tag)
        elif not match.group(1).strip():
            empty_tags.append(tag)
    
    if missing_tags:
        return False, f"missing tags: {', '.join(missing_tags)}"
    if empty_tags:
        return False, f"empty tag content: {', '.join(empty_tags)}"
    
    return True, "ok"

def check_think_format(think_content: str) -> bool:
    """Return True if the <think> block contains all required non-empty XML tags."""
    valid, _ = check_think_format_with_details(think_content)
    return valid

def _calculate_dynamic_cot_weight(cot_score: float, T: float, k: float, w_max: float, w_min: float) -> float:
    """Compute a soft-gated weight for the CoT reward from its score via sigmoid."""
    return w_min + (w_max - w_min) * (1.0 / (1.0 + math.exp(k * (cot_score - T))))



def _prepare_and_validate_batch(solution_strs: List[str], ground_truths: List[str], data_sources: List[str], extra_infos: List[Dict[str, Any]], required_tags: List[str] = None) -> Tuple:
    """
    [V5 - 最终合并版] 融合了分层惩罚与feature_report提取逻辑。
    """
    num_samples = len(solution_strs)
    final_scores_placeholder = np.full(num_samples, 0.0) 
    
    # 理由1：恢复分层惩罚机制，为不同格式错误提供精细的负奖励信号。
    PENALTIES = {
        'no_think_tag': -1.0,
        'empty_think': -0.8,
        'missing_tags': -0.6,
        'empty_tag_content': -0.4
    }

    data = {
        'indices': [], 'sources': [], 'truths': [],
        'cot_analyses': [], 'translations': [],
        'feature_reports': [] # 新增字段
    }
    
    error_stats = {
        'no_think_tag': 0, 'empty_think': 0, 'missing_tags': 0,
        'empty_tag_content': 0, 'valid': 0
    }
    
    for i, text in enumerate(solution_strs):
        think_content, translation = parse_think_and_translation(text)
        
        if think_content is None:
            error_stats['no_think_tag'] += 1
            final_scores_placeholder[i] = PENALTIES['no_think_tag']
            continue
            
        is_valid, error_msg = check_think_format_with_details(think_content, required_tags=required_tags)
        
        if is_valid:
            # 理由2：集成 feature_report 的提取与最高优先级日志记录。
            feature_report = extra_infos[i].get('feature_report')
            if feature_report:
                data['feature_reports'].append(feature_report)
            else:
                # 最高优先级日志指令实现
                logger.critical(f"CRITICAL: 样本索引 {i} 的 extra_info 中缺少 'feature_report'！将使用默认值继续。")
                default_report = {"feature_code": [0, 0, 0], "feature_fragments": {"dnt": [], "pragmatic": []}}
                data['feature_reports'].append(default_report)

            # 将通过验证的样本信息添加到data字典
            error_stats['valid'] += 1
            data['indices'].append(i)
            data['truths'].append(ground_truths[i])
            data['sources'].append(extra_infos[i].get('source', data_sources[i]))
            data['cot_analyses'].append(think_content)
            data['translations'].append(translation)
        else:
            # 恢复分层惩罚的应用逻辑
            if "think内容为空" in error_msg:
                error_stats['empty_think'] += 1
                final_scores_placeholder[i] = PENALTIES['empty_think']
            elif "缺失标签" in error_msg:
                error_stats['missing_tags'] += 1
                final_scores_placeholder[i] = PENALTIES['missing_tags']
            elif "空标签内容" in error_msg:
                error_stats['empty_tag_content'] += 1
                final_scores_placeholder[i] = PENALTIES['empty_tag_content']
            else: # 备用情况
                final_scores_placeholder[i] = -1.0

    # 恢复详细的错误统计日志
    logger.info(f"预处理与验证完成。有效样本: {error_stats['valid']}/{num_samples} ({error_stats['valid']/num_samples*100:.1f}%)")
    
    total_errors = num_samples - error_stats['valid']
    if total_errors > 0:
        logger.warning(f"格式错误统计 (共{total_errors}个样本获得分层惩罚):")
        if error_stats['no_think_tag'] > 0:
            logger.warning(f"  • 无<think>标签: {error_stats['no_think_tag']} 个样本 (惩罚: {PENALTIES['no_think_tag']})")
        if error_stats['empty_think'] > 0:
            logger.warning(f"  • 空<think>内容: {error_stats['empty_think']} 个样本 (惩罚: {PENALTIES['empty_think']})")
        if error_stats['missing_tags'] > 0:
            logger.warning(f"  • 缺失必需标签: {error_stats['missing_tags']} 个样本 (惩罚: {PENALTIES['missing_tags']})")
        if error_stats['empty_tag_content'] > 0:
            logger.warning(f"  • 标签内容为空: {error_stats['empty_tag_content']} 个样本 (惩罚: {PENALTIES['empty_tag_content']})")
    
    if error_stats['valid'] == 0:
        logger.error("没有任何样本通过格式验证！所有样本都将获得惩罚分数。")
        return final_scores_placeholder, False, None
    
    return final_scores_placeholder, True, data
    
def parse_think_and_translation(text: str) -> Tuple[Optional[str], str]:
    """[核心工具] 从模型输出中解析<think>内容和真实翻译。"""
    if not isinstance(text, str): return None, ""
    match = re.search(r'<think>(.*?)</think>\s*(.*)', text, re.DOTALL)
    return (match.group(1).strip(), match.group(2).strip()) if match else (None, text.strip())



# ==============================================================================
# 2. 奖励组件计算函数 (Reward Component Calculation Functions)
# ==============================================================================

def _compute_bleu_component_batch(
    ground_truths: List[str], 
    translations: List[str], 
    extra_infos_for_valid: List[Dict[str, Any]] # <--- 新增参数
) -> Optional[np.ndarray]:
    """[组件] 本地批量计算BLEU分数，并根据语言对动态选择分词器。"""
    if len(ground_truths) != len(extra_infos_for_valid):
        logger.error("[组件-BLEU] 样本数量与extra_infos数量不匹配！")
        return None
        
    try:
        scores = []
        for i in range(len(translations)):
            trans = translations[i]
            ref = ground_truths[i]
            sample_info = extra_infos_for_valid[i]
            
            # 从extra_info中获取语言对信息
            pair = sample_info.get('pair', '') # 假设 'pair' 存在于 extra_infos
            if not pair:
                # 作为后备，尝试从data_source解析
                data_source_parts = sample_info.get('data_source', '').split('_')
                if len(data_source_parts) >= 2:
                    pair = f"{data_source_parts[-2]}-{data_source_parts[-1]}"

            # 精确判断目标语言
            target_lang = pair.split('-')[1] if '-' in pair else ''
            tokenizer_name = 'zh' if target_lang == 'zh' else '13a'
            
            # 使用带分词器的sentence_bleu
            score = sacrebleu.sentence_bleu(trans, [ref], tokenize=tokenizer_name).score / 100.0
            scores.append(score)
            
        final_scores = np.array(scores)
        logger.info(f"[组件-BLEU] 计算完成 (样本数: {len(final_scores)}).")
        return final_scores
        
    except Exception as e:
        logger.error(f"[组件-BLEU] 计算时发生未知错误: {e}", exc_info=True)
        return None

def _compute_kiwi_component_batch(sources: List[str], translations: List[str], timeout: int) -> Optional[np.ndarray]:
    """[组件] 远程批量计算CometKiwi分数。"""
    if not KIWI_SERVER_URL: return None
    try:
        payload = {"sources": sources, "mts": translations}
        response = requests.post(KIWI_SERVER_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        scores = response.json().get("scores")
        if scores and len(scores) == len(sources):
            logger.info(f"[组件-Kiwi] 获取成功 (样本数: {len(scores)}).")
            return np.array(scores)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"[组件-Kiwi] 请求失败: {e}")
        return None

def _compute_xcomet_component_batch(sources: List[str], translations: List[str], references: List[str], timeout: int) -> Optional[np.ndarray]:
    """[组件] 远程批量计算XCOMET分数。"""
    if not XCOMET_SERVER_URL: return None
    try:
        payload = {"sources": sources, "mts": translations, "references": references}
        response = requests.post(XCOMET_SERVER_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        scores = response.json().get("scores")
        if scores and len(scores) == len(sources):
            logger.info(f"[组件-XCOMET] 获取成功 (样本数: {len(scores)}).")
            return np.array(scores)
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"[组件-XCOMET] 请求失败: {e}")
        return None


def _compute_cot_component_batch(valid_data: Dict, num_processes: int, timeout: int) -> Optional[np.ndarray]:
    """
    [修改] 此函数被修改以从valid_data字典中读取所有必需数据,
           并发送包含feature_reports的新API负载。
    """
    if not COT_EVALUATOR_SERVER_URL: 
        logger.error("[组件-CoT] 服务URL未配置！")
        return None
    try:
        # 直接从 valid_data 构建 payload，包含了新的 feature_reports 字段
        payload = {
            "sources": valid_data['sources'], 
            "cot_analyses": valid_data['cot_analyses'], 
            "final_translations": valid_data['translations'],
            "feature_reports": valid_data['feature_reports'], # <-- 核心修改点
            "num_processes": num_processes
        }
        
        response = requests.post(COT_EVALUATOR_SERVER_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        process_rewards = result.get("process_rewards")
        eval_details = result.get("evaluation_details", [])

        if process_rewards and len(process_rewards) == len(valid_data['sources']):
            scores = np.array(process_rewards, dtype=float)
            error_count = sum(1 for detail in eval_details if isinstance(detail, dict) and "error" in detail)
            if error_count > 0:
                logger.warning(f"[组件-CoT] {error_count}个样本评估因格式或解析错误，奖励已置为0。")
            logger.info(f"[组件-CoT] 获取成功 (样本数: {len(scores)}).")
            return scores
        
        logger.error(f"[组件-CoT] 服务端返回的奖励数量与请求数量不匹配。")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[组件-CoT] 请求失败: {e}")
        return None


def log_component_scores(extra_infos: List[Dict[str, Any]], indices: List[int], **kwargs):
    """[V3.3 新增] 统一的日志记录辅助函数"""
    for component_name, scores in kwargs.items():
        if scores is not None:
            for i, score in enumerate(scores):
                original_index = indices[i]
                extra_infos[original_index][f'reward/{component_name}_score'] = score



def compute_bleu_xcomet_kiwi_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器] 重构后的 BLEU+XCOMET+Kiwi 奖励函数。"""
    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: return final_scores.tolist()
    extra_infos_for_valid = [extra_infos[i] for i in valid_data['indices']]

    timeout = kwargs.get('request_timeout', 1800)
    if not XCOMET_SERVER_URL or not KIWI_SERVER_URL: logger.error("XCOMET或Kiwi服务URL未配置！"); return final_scores.tolist()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_xcomet = executor.submit(_compute_xcomet_component_batch, valid_data['sources'], valid_data['translations'], valid_data['truths'], timeout)
        future_kiwi = executor.submit(_compute_kiwi_component_batch, valid_data['sources'], valid_data['translations'], timeout)
        
        bleu_scores = _compute_bleu_component_batch(
            valid_data['truths'], 
            valid_data['translations'],
            extra_infos_for_valid  # <--- 传递新参数
        )
        xcomet_scores, kiwi_scores = future_xcomet.result(), future_kiwi.result()

    failed = [name for name, result in [("BLEU", bleu_scores), ("XCOMET", xcomet_scores), ("Kiwi", kiwi_scores)] if result is None]
    if failed:
        logger.error(f"奖励组件失败: {', '.join(failed)}。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    final_scores[valid_data['indices']] = bleu_scores + xcomet_scores + kiwi_scores
    log_component_scores(extra_infos, valid_data['indices'], 
                         bleu=bleu_scores, xcomet=xcomet_scores, 
                         kiwi=kiwi_scores)
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 BLEU+XCOMET+Kiwi 奖励。")
    return final_scores.tolist()

def compute_bleu_kiwi_score_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器] 重构后的 BLEU+Kiwi 奖励函数。"""
    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: return final_scores.tolist()
    extra_infos_for_valid = [extra_infos[i] for i in valid_data['indices']]

    timeout = kwargs.get('request_timeout', 1800)
    if not KIWI_SERVER_URL: logger.error("Kiwi服务URL未配置！"); return final_scores.tolist()

    kiwi_scores = _compute_kiwi_component_batch(valid_data['sources'], valid_data['translations'], timeout)
    bleu_scores = _compute_bleu_component_batch(
        valid_data['truths'], 
        valid_data['translations'],
        extra_infos_for_valid  # <--- 传递新参数
    )

    failed = [name for name, result in [("BLEU", bleu_scores), ("Kiwi", kiwi_scores)] if result is None]
    if failed:
        logger.error(f"奖励组件失败: {', '.join(failed)}。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    final_scores[valid_data['indices']] = bleu_scores + kiwi_scores
    log_component_scores(extra_infos, valid_data['indices'], 
                        bleu=bleu_scores, kiwi=kiwi_scores)
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 BLEU+Kiwi 奖励。")
    return final_scores.tolist()

def compute_bleu_xcomet_kiwi_cot_score_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器] 编排BLEU, XCOMET, Kiwi, CoT四个组件的复合奖励函数。"""
    # [关键修改] 从 kwargs 提取 Shell 脚本传进来的参数
    # 如果没传，默认为 None (即全量检查)
    custom_tags = kwargs.get('required_tags', None)

    final_scores, is_valid, valid_data = _prepare_and_validate_batch(
            solution_strs, ground_truths, data_sources, extra_infos, 
            required_tags=custom_tags  # <--- 传进去
        )
    if not is_valid: return final_scores.tolist()
    extra_infos_for_valid = [extra_infos[i] for i in valid_data['indices']]

    # --- 配置提取与验证 ---
    #weights = kwargs.get('weights', {'bleu': 0.15, 'xcomet': 0.225, 'kiwi': 0.225, 'cot': 0.4})
    weights = kwargs.get('weights', {'bleu': 0.3, 'xcomet': 0.3, 'kiwi': 0.3, 'cot': 0.1})
    timeout = kwargs.get('request_timeout', 1800)
    cot_num_processes = kwargs.get('cot_num_processes', 240)
    logger.info("="*40)
    logger.info(f"🚀 [Reward Weights Active Configuration]")
    logger.info(f"   ► XCOMET : {weights.get('xcomet', 0)}")
    logger.info(f"   ► Kiwi   : {weights.get('kiwi', 0)}")
    logger.info(f"   ► CoT    : {weights.get('cot', 0)}")
    logger.info(f"   ► BLEU   : {weights.get('bleu', 0)}")
    logger.info("="*40)
    if abs(sum(weights.values()) - 1.0) > 0.01: logger.warning(f"权重总和不为1.0 (当前: {sum(weights.values()):.2f})")
    if weights.get('cot', 0) > 0 and not COT_EVALUATOR_SERVER_URL: logger.error("CoT权重>0但URL未配置！"); return final_scores.tolist()
    if weights.get('kiwi', 0) > 0 and not KIWI_SERVER_URL: logger.error("Kiwi权重>0但URL未配置！"); return final_scores.tolist()
    if weights.get('xcomet', 0) > 0 and not XCOMET_SERVER_URL: logger.error("XCOMET权重>0但URL未配置！"); return final_scores.tolist()

    # --- 并行执行组件 ---
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_xcomet = executor.submit(_compute_xcomet_component_batch, valid_data['sources'], valid_data['translations'], valid_data['truths'], timeout) if weights.get('xcomet', 0) > 0 else None
        future_kiwi = executor.submit(_compute_kiwi_component_batch, valid_data['sources'], valid_data['translations'], timeout) if weights.get('kiwi', 0) > 0 else None
        future_cot = executor.submit(_compute_cot_component_batch, valid_data, cot_num_processes, timeout) if weights.get('cot', 0) > 0 else None
        
        bleu_scores = _compute_bleu_component_batch(
            valid_data['truths'], 
            valid_data['translations'],
            extra_infos_for_valid  # <--- 在if的函数调用中加入新参数
        ) if weights.get('bleu', 0) > 0 else np.zeros(len(valid_data['indices']))
        xcomet_scores = future_xcomet.result() if future_xcomet else np.zeros(len(valid_data['indices']))
        kiwi_scores = future_kiwi.result() if future_kiwi else np.zeros(len(valid_data['indices']))
        cot_scores = future_cot.result() if future_cot else np.zeros(len(valid_data['indices']))

    # --- 结果校验与合并 ---
    failed = [name for name, (res, w) in [("BLEU", (bleu_scores, 'bleu')), ("XCOMET", (xcomet_scores, 'xcomet')), ("Kiwi", (kiwi_scores, 'kiwi')), ("CoT", (cot_scores, 'cot'))] if weights.get(w, 0) > 0 and res is None]
    if failed:
        logger.error(f"奖励组件失败: {', '.join(failed)}。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    weighted_scores = (weights.get('bleu', 0) * bleu_scores +
                       weights.get('xcomet', 0) * xcomet_scores +
                       weights.get('kiwi', 0) * kiwi_scores +
                       weights.get('cot', 0) * cot_scores)

    final_scores[valid_data['indices']] = weighted_scores
    log_component_scores(extra_infos, valid_data['indices'], 
                         bleu=bleu_scores, xcomet=xcomet_scores, 
                         kiwi=kiwi_scores, cot=cot_scores)
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 BLEU+XCOMET+Kiwi+CoT 复合奖励。")
    return final_scores.tolist()

def compute_bleu_cot_score_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器] BLEU+CoT 组合奖励函数。"""
    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: return final_scores.tolist()
    extra_infos_for_valid = [extra_infos[i] for i in valid_data['indices']]
    
    # --- 配置提取与验证 ---
    weights = kwargs.get('weights', {'bleu': 0.5, 'cot': 0.5})  # 默认1:1权重
    timeout = kwargs.get('request_timeout', 1800)
    cot_num_processes = kwargs.get('cot_num_processes', 240)
    
    if abs(sum(weights.values()) - 1.0) > 0.01: 
        logger.warning(f"权重总和不为1.0 (当前: {sum(weights.values()):.2f})")
    if weights.get('cot', 0) > 0 and not COT_EVALUATOR_SERVER_URL: 
        logger.error("CoT权重>0但URL未配置！")
        return final_scores.tolist()

    # --- 并行执行组件 ---
    with ThreadPoolExecutor(max_workers=1) as executor:
        # CoT组件异步执行
        future_cot = executor.submit(_compute_cot_component_batch, valid_data, cot_num_processes, timeout) if weights.get('cot', 0) > 0 else None
        
        # BLEU组件同步执行（本地计算，速度快）
        bleu_scores = _compute_bleu_component_batch(
            valid_data['truths'], 
            valid_data['translations'],
            extra_infos_for_valid  # <--- 在if的函数调用中加入新参数
        ) if weights.get('bleu', 0) > 0 else np.zeros(len(valid_data['indices']))
        
        # 获取CoT结果
        cot_scores = future_cot.result() if future_cot else np.zeros(len(valid_data['indices']))

    # --- 结果校验与合并 ---
    failed = [name for name, (res, w) in [("BLEU", (bleu_scores, 'bleu')), ("CoT", (cot_scores, 'cot'))] if weights.get(w, 0) > 0 and res is None]
    if failed:
        logger.error(f"奖励组件失败: {', '.join(failed)}。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    # --- 加权求和 ---
    weighted_scores = (weights.get('bleu', 0) * bleu_scores + 
                       weights.get('cot', 0) * cot_scores)

    final_scores[valid_data['indices']] = weighted_scores
    
    # --- 日志记录 ---
    log_component_scores(extra_infos, valid_data['indices'], 
                        bleu=bleu_scores, cot=cot_scores)
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 BLEU+CoT 组合奖励。")
    return final_scores.tolist()

def compute_bleu_score_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器-消融] 仅计算BLEU分数。"""
    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: return final_scores.tolist()
    extra_infos_for_valid = [extra_infos[i] for i in valid_data['indices']]

    bleu_scores = _compute_bleu_component_batch(
        valid_data['truths'], 
        valid_data['translations'],
        extra_infos_for_valid  # <--- 传递新参数
    )
    if bleu_scores is None:
        logger.error("奖励组件失败: BLEU。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    final_scores[valid_data['indices']] = bleu_scores
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 BLEU 单项奖励。")
    return final_scores.tolist()

def compute_kiwi_score_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器-消融] 仅计算CometKiwi分数。"""
    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: return final_scores.tolist()

    timeout = kwargs.get('request_timeout', 1800)
    if not KIWI_SERVER_URL: logger.error("Kiwi服务URL未配置！"); return final_scores.tolist()

    kiwi_scores = _compute_kiwi_component_batch(valid_data['sources'], valid_data['translations'], timeout)

    if kiwi_scores is None:
        logger.error("奖励组件失败: Kiwi。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    final_scores[valid_data['indices']] = kiwi_scores
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 Kiwi 单项奖励。")
    return final_scores.tolist()

def compute_cot_score_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器-消融] 仅计算CoT过程奖励分数。"""
    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: return final_scores.tolist()

    timeout = kwargs.get('request_timeout', 1800)
    cot_num_processes = kwargs.get('cot_num_processes', 240)
    if not COT_EVALUATOR_SERVER_URL: logger.error("CoT服务URL未配置！"); return final_scores.tolist()

    cot_scores = _compute_cot_component_batch(valid_data, cot_num_processes, timeout)

    if cot_scores is None:
        logger.error("奖励组件失败: CoT。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    final_scores[valid_data['indices']] = cot_scores
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 CoT 单项奖励。")
    return final_scores.tolist()

def compute_bleu_xcomet_kiwi_cot_score_batch_soft_gated(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:

    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: return final_scores.tolist()
    extra_infos_for_valid = [extra_infos[i] for i in valid_data['indices']]

    # --- 配置提取与验证 ---
    weights = kwargs.get('weights', {'bleu': 0.333, 'xcomet': 0.333, 'kiwi': 0.334})
    timeout = kwargs.get('request_timeout', 1800)
    cot_num_processes = kwargs.get('cot_num_processes', 64)
    
    # 提取软门控超参数
    T = kwargs.get('cot_threshold', 0.6)
    k = kwargs.get('cot_steepness', 20)
    w_max = kwargs.get('cot_max_weight', 0.1)
    w_min = kwargs.get('cot_min_weight', 0)
    
    logger.info(f"[奖励-软门控] 动态权重参数已加载 (来自配置):")
    logger.info(f"  • T (cot_threshold): {T} (脚本传入, 默认: 0.8)")
    logger.info(f"  • k (cot_steepness): {k} (脚本传入, 默认: 20)")
    logger.info(f"  • w_max (cot_max_weight): {w_max} (脚本传入, 默认: 0.5)")
    logger.info(f"  • w_min (cot_min_weight): {w_min} (脚本传入, 默认: 0)")

    # URL 检查
    if not COT_EVALUATOR_SERVER_URL or not KIWI_SERVER_URL or not XCOMET_SERVER_URL:
        logger.error("一个或多个服务URL未配置 (CoT, Kiwi, XCOMET)！")
        return final_scores.tolist()

    # --- 并行执行组件 (此部分逻辑不变) ---
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_xcomet = executor.submit(_compute_xcomet_component_batch, valid_data['sources'], valid_data['translations'], valid_data['truths'], timeout)
        future_kiwi = executor.submit(_compute_kiwi_component_batch, valid_data['sources'], valid_data['translations'], timeout)
        future_cot = executor.submit(_compute_cot_component_batch, valid_data, cot_num_processes, timeout)
        
        bleu_scores = _compute_bleu_component_batch(
            valid_data['truths'], 
            valid_data['translations'],
            extra_infos_for_valid  # <--- 在if的函数调用中加入新参数
        ) if weights.get('bleu', 0) > 0 else np.zeros(len(valid_data['indices']))
        
        xcomet_scores = future_xcomet.result()
        kiwi_scores = future_kiwi.result()
        cot_scores = future_cot.result()

    # --- 结果校验 (此部分逻辑不变) ---
    failed = [name for name, res in [("BLEU", bleu_scores), ("XCOMET", xcomet_scores), ("Kiwi", kiwi_scores), ("CoT", cot_scores)] if res is None]
    if failed:
        logger.error(f"奖励组件失败: {', '.join(failed)}。所有有效样本都将获得惩罚分数-1.0。")
        # 注意：这里返回的是一个预填充了惩罚分数的数组
        return final_scores.tolist()

    # --- [核心修改] 逐样本动态加权 ---
    
    # 1. 归一化结果指标(Outcome)的权重
    outcome_weights = {key: value for key, value in weights.items() if key != 'cot'}
    total_outcome_weight = sum(outcome_weights.values())
    if total_outcome_weight > 0:
        outcome_weights = {key: value / total_outcome_weight for key, value in outcome_weights.items()}
    else:
        logger.warning("结果指标权重总和为0，R_outcome将始终为0。")

    # 2. 循环计算每个样本的最终分数
    num_valid_samples = len(valid_data['indices'])
    dynamic_weighted_scores = np.zeros(num_valid_samples)

    for i in range(num_valid_samples):
        # a. 获取当前样本的CoT分数
        s = cot_scores[i]
        
        # b. 计算动态权重
        w_cot = _calculate_dynamic_cot_weight(s, T, k, w_max, w_min)
        
        # c. 计算当前样本的结果分 R_outcome
        r_outcome_i = (outcome_weights.get('bleu', 0) * bleu_scores[i] +
                       outcome_weights.get('xcomet', 0) * xcomet_scores[i] +
                       outcome_weights.get('kiwi', 0) * kiwi_scores[i])
        
        # d. 应用最终公式
        total_score_i = w_cot * s + (1 - w_cot) * r_outcome_i
        dynamic_weighted_scores[i] = total_score_i
        
    # --- 分数更新与日志记录 ---
    final_scores[valid_data['indices']] = dynamic_weighted_scores
    
    log_component_scores(extra_infos, valid_data['indices'], 
                         bleu=bleu_scores, xcomet=xcomet_scores, 
                         kiwi=kiwi_scores, cot=cot_scores)
    
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的动态软门控复合奖励。")
    return final_scores.tolist()


def get_and_increment_step_counter(counter_file_path: str) -> int:
    """
    [新增] 健壮的函数，用于读取、返回并递增持久化的步骤计数器。
    使用文件锁来确保进程安全。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(counter_file_path), exist_ok=True)
    
    # 'a+'模式：如果文件不存在则创建
    with open(counter_file_path, 'a+') as f:
        # 对文件加锁，防止多个进程同时读写
        fcntl.flock(f, fcntl.LOCK_EX)
        
        f.seek(0) # 指针移到文件开头以读取
        content = f.read().strip()
        step = int(content) if content.isdigit() else 0
        
        f.seek(0) # 指针移回文件开头
        f.truncate() # 清空文件
        f.write(str(step + 1)) # 写入新值
        
        fcntl.flock(f, fcntl.LOCK_UN) # 解锁文件
        return step

def compute_alternating_reward_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs) -> List[float]:
    """
    [新增] 编排器：实现过程奖励和结果奖励的交替执行，并支持断点续训。
    所有与状态文件相关的配置逻辑都内聚在此函数内部。
    
    [V2 - 用户需求]
    - 过程奖励 = 仅 CoT 奖励 (compute_cot_score_batch)
    - 结果奖励 = BLEU + XCOMET + Kiwi (compute_bleu_xcomet_kiwi_batch)
    - 两种路径都必须执行格式检查。
    """
    
    # 1. 在函数执行时，才从环境变量中读取 SAVE_DIR
    # (与您的 shell 脚本中的 `export SAVE_DIR=...` 对应)
    SAVE_DIR = os.environ.get("SAVE_DIR") 

    if not SAVE_DIR:
        error_msg = ("CRITICAL ERROR: 'compute_alternating_reward_batch' 需要 'SAVE_DIR' 环境变量, 但未设置! "
                     "请在您的主训练脚本中 `export SAVE_DIR=...`。")
        logger.critical(error_msg)
        raise ValueError(error_msg)

    # 2. 动态构建计数器文件的绝对路径
    counter_file_path = os.path.join(SAVE_DIR, 'reward_step_counter.txt')

    # 3. 获取并更新持久化的全局Step
    #    此文件已由您的主训练脚本在启动时进行了校准
    current_step = get_and_increment_step_counter(counter_file_path)
    
    # 4. 从kwargs获取交替周期配置
    cycle_length = kwargs.get('cycle_length', 10)
    process_reward_steps = kwargs.get('process_reward_steps', 2) # 周期中用于过程奖励的step数
    
    # 5. 决定当前step使用哪种奖励模式
    current_cycle_step = current_step % cycle_length
    
    # 6. 按需计算和打印日志
    logger.info("-" * 50)
    logger.info(f"[交替奖励] 全局Step: {current_step}, 周期内Step: {current_cycle_step}/{cycle_length-1}")
    logger.info(f"[交替奖励] 状态文件: {counter_file_path}")
    
    reward_list = []
    # 判断是执行周期的前部分（结果奖励）还是后部分（过程奖励）
    if current_cycle_step < (cycle_length - process_reward_steps):
        # 周期前段：执行结果奖励
        logger.info(f">>> 模式: 结果奖励 (BLEU + XCOMET + Kiwi)")
        
        # --- [核心修改] ---
        # 调用 `compute_bleu_xcomet_kiwi_batch`
        # 这个函数内部已经包含了 _prepare_and_validate_batch() 格式检查
        reward_list = compute_bleu_xcomet_kiwi_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
        # --- [修改结束] ---
        
        logger.info(f"    计算得到的结果奖励 (前5个样本): {[f'{r:.4f}' for r in reward_list[:5]]}")
    else:
        # 周期后段：执行过程奖励
        logger.info(f">>> 模式: 过程奖励 (CoT Only)")
        
        # --- [保持不变] ---
        # 调用 `compute_cot_score_batch`
        # 这个函数内部也已经包含了 _prepare_and_validate_batch() 格式检查
        reward_list = compute_cot_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)
        # --- [保持不变] ---

        logger.info(f"    计算得到的过程奖励 (前5个样本): {[f'{r:.4f}' for r in reward_list[:5]]}")
    
    logger.info("-" * 50)
    return reward_list

def compute_bleu_kiwi_cot_score_batch(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], **kwargs) -> List[float]:
    """[编排器 V-BKC] 编排BLEU, Kiwi, CoT三个组件的复合奖励函数。
    
    此版本基于 V3.2 客户端重写，
    1. 移除了 XCOMET。
    2. 采用 B:K:C = 4.5:4.5:1 的默认归一化权重。
    3. 兼容新版 _compute_bleu_component_batch (传递 extra_infos_for_valid)。
    """
    final_scores, is_valid, valid_data = _prepare_and_validate_batch(solution_strs, ground_truths, data_sources, extra_infos)
    if not is_valid: 
        return final_scores.tolist()
    
    # [新增] 必须为新版BLEU函数准备 extra_infos
    extra_infos_for_valid = [extra_infos[i] for i in valid_data['indices']]

    # --- 配置提取与验证 ---
    # [修改] 采用 4.5:4.5:1 的动态归一化权重
    default_ratio = {'bleu': 4.5, 'kiwi': 4.5, 'cot': 1.0}
    weights = kwargs.get('weights')
    if weights is None:
        total = sum(default_ratio.values())
        weights = {k: v / total for k, v in default_ratio.items()}
        logger.info(f"使用默认权重 B:K:C = 4.5:4.5:1 (归一化为: {weights['bleu']:.2f}:{weights['kiwi']:.2f}:{weights['cot']:.2f})")
    else:
        weights = dict(weights) # 确保 weights 是可修改的字典

    timeout = kwargs.get('request_timeout', 1800)
    cot_num_processes = kwargs.get('cot_num_processes', 240)
    
    if abs(sum(weights.values()) - 1.0) > 0.01:
        logger.warning(f"权重总和不为1.0 (当前: {sum(weights.values()):.2f})")
    if weights.get('cot', 0) > 0 and not COT_EVALUATOR_SERVER_URL:
        logger.error("CoT权重>0但URL未配置！")
        return final_scores.tolist()
    if weights.get('kiwi', 0) > 0 and not KIWI_SERVER_URL:
        logger.error("Kiwi权重>0但URL未配置！")
        return final_scores.tolist()
    # [移除] XCOMET URL 检查

    # --- 并行执行组件 ---
    # [修改] 移除了 future_xcomet，max_workers=3 -> 2
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_kiwi = executor.submit(_compute_kiwi_component_batch, valid_data['sources'], valid_data['translations'], timeout) if weights.get('kiwi', 0) > 0 else None
        future_cot = executor.submit(_compute_cot_component_batch, valid_data, cot_num_processes, timeout) if weights.get('cot', 0) > 0 else None

        # [修改] 调用新版 _compute_bleu_component_batch，传入 extra_infos_for_valid
        bleu_scores = _compute_bleu_component_batch(
            valid_data['truths'], 
            valid_data['translations'],
            extra_infos_for_valid 
        ) if weights.get('bleu', 0) > 0 else np.zeros(len(valid_data['indices']))
        
        # [移除] xcomet_scores
        kiwi_scores = future_kiwi.result() if future_kiwi else np.zeros(len(valid_data['indices']))
        cot_scores = future_cot.result() if future_cot else np.zeros(len(valid_data['indices']))

    # --- 结果校验与合并 ---
    # [修改] 移除了 XCOMET 的失败检查
    failed = [name for name, (res, key) in [("BLEU", (bleu_scores, 'bleu')), ("Kiwi", (kiwi_scores, 'kiwi')), ("CoT", (cot_scores, 'cot'))] if weights.get(key, 0) > 0 and res is None]
    if failed:
        logger.error(f"奖励组件失败: {', '.join(failed)}。所有有效样本都将获得惩罚分数-1.0。")
        return final_scores.tolist()

    # [修改] 移除了 xcomet_scores 的加权
    weighted_scores = (weights.get('bleu', 0) * bleu_scores +
                       weights.get('kiwi', 0) * kiwi_scores +
                       weights.get('cot', 0) * cot_scores)

    final_scores[valid_data['indices']] = weighted_scores
    
    # [修改] 移除了 xcomet 的日志记录
    log_component_scores(extra_infos, valid_data['indices'], 
                         bleu=bleu_scores, kiwi=kiwi_scores, cot=cot_scores)
    
    logger.info(f"成功计算了 {len(valid_data['indices'])} 个样本的 BLEU+Kiwi+CoT 复合奖励。")
    return final_scores.tolist()