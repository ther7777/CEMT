"""Pick the highest-scoring CoT candidate per sample and emit SFT-ready JSONL."""
import os
import json
import argparse
import logging
from tqdm import tqdm

def setup_logging(log_filename: str):
    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    logger = logging.getLogger(); logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(); ch.setFormatter(formatter); logger.addHandler(ch)
    fh = logging.FileHandler(log_filepath, mode='a', encoding='utf-8'); fh.setFormatter(formatter); logger.addHandler(fh)
    return log_filepath

def main():
    """主函数，负责读取、择优、格式化和报告。"""
    parser = argparse.ArgumentParser(description="从已评分的CoT候选中择优筛选，生成最终SFT数据。")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="由evaluate_cot_candidates.py生成的、已包含分数的jsonl文件。")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="用于保存最终筛选出的SFT样本的jsonl文件。")
    parser.add_argument("--score_threshold", type=float, default=0.0, 
                        help="一个候选的r_process_score必须高于此阈值才会被选中。")
    args = parser.parse_args()

    log_filepath = setup_logging(f"select_best_{os.path.basename(args.input_file)}")
    logging.info("="*80)
    logging.info(f"--- CoT择优筛选任务启动 (V1.1) ---")
    logging.info(f"  [+] 输入文件: {args.input_file}")
    logging.info(f"  [>] 输出文件: {args.output_file}")
    logging.info(f"  [!] 质量阈值: r_process_score >= {args.score_threshold}")
    logging.info("="*80)

    if not os.path.exists(args.input_file):
        logging.error(f"输入文件不存在: {args.input_file}"); return

    discarded_sample_info = []

    total_samples_in = 0
    selected_samples_out = 0

    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile, \
             open(args.output_file, 'w', encoding='utf-8') as outfile:
            
            num_lines = sum(1 for line in open(args.input_file, 'r', encoding='utf-8'))
            infile.seek(0)

            for line in tqdm(infile, total=num_lines, desc="择优筛选"):
                total_samples_in += 1
                sample_id_for_log = f"line_{total_samples_in}"
                
                try:
                    sample = json.loads(line)
                    sample_id_for_log = sample.get('sample_id', sample_id_for_log)
                    candidates = sample.get('cot_candidates', [])

                    best_candidate = None
                    max_score = -1.0

                    for cand in candidates:
                        score = cand.get('r_process_score')
                        if score is not None and score > max_score:
                            max_score = score
                            best_candidate = cand
                    
                    if best_candidate and max_score >= args.score_threshold:
                        final_record = {
                            'simple_id': sample.get('simple_id'),
                            'sample_id': sample.get('sample_id'),
                            'pair': sample.get('pair'),
                            'source': sample.get('source'),
                            'target': sample.get('target'),
                            'feature_report':sample.get('feature_report'),
                            'COT_Inf': best_candidate.get('generated_cot'),
                            'COT_Translation': sample.get('target')
                        }
                        outfile.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                        selected_samples_out += 1
                    else:
                        reason = "Score too low" if best_candidate else "No valid candidates"
                        discarded_sample_info.append({"id": sample_id_for_log, "max_score": max_score, "reason": reason})

                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    discarded_sample_info.append({"id": sample_id_for_log, "max_score": "N/A", "reason": f"Format Error: {e}"})
                    continue
    
    except Exception as e:
        logging.error(f"处理文件时发生严重错误: {e}", exc_info=True)
        return

    discarded_samples_count = len(discarded_sample_info)
    logging.info("\n" + "="*80)
    logging.info("--- CoT择优筛选任务统计报告 ---")
    logging.info(f"  总共读取样本数: {total_samples_in}")
    logging.info("-" * 40)
    logging.info(f"  成功筛选出的高质量样本数: {selected_samples_out}")
    logging.info(f"  被丢弃的样本数: {discarded_samples_count}")
    pass_rate = (selected_samples_out / total_samples_in * 100) if total_samples_in > 0 else 0
    logging.info(f"  通过率: {pass_rate:.2f}%")
    
    if discarded_sample_info:
        logging.warning("--- 以下为被丢弃的样本列表 ---")
        for info in discarded_sample_info[:100]:
            max_score = info['max_score']
            score_str = f"{max_score:.4f}" if isinstance(max_score, (int, float)) else str(max_score)
            logging.warning(f"  - Sample ID: {info['id']}, Max Score: {score_str}, Reason: {info['reason']}")
        if len(discarded_sample_info) > 100:
            logging.warning(f"  ... (还有 {len(discarded_sample_info) - 100} 个未打印) ...")
            
    logging.info("-" * 40)
    logging.info(f"  最终SFT数据集已生成: {args.output_file}")
    logging.info("="*80)

if __name__ == "__main__":
    main()