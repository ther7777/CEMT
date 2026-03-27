"""CoT evaluator service.

This FastAPI server scores CoT outputs via a local OpenAI-compatible endpoint (e.g. vLLM)
and computes a V3 process reward using a configurable addition + penalty scheme.

Notes:
    - The scoring dimensions and penalty map are loaded from a JSON config.
    - A separate prompt template file is loaded based on the config.
"""
import uvicorn
import logging
import argparse
import json
import time
import re
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
from multiprocessing import Pool

# Loaded from the external JSON config at startup.
SERVER_CONFIG = {}

def setup_logging(service_name: str):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{service_name}_evaluator.log"
    log_filepath = os.path.join(log_dir, log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    return log_filepath

# Inference / retry settings.
INFERENCE_PARAMS = {"temperature": 0.2, "top_p": 0.95, "max_tokens": 1000}
MAX_RETRIES = 3
RETRY_DELAY = 3

VLLM_API_ENDPOINT = None
app = FastAPI(title="CoT Evaluator Server V2 (Configurable)")
logger = logging.getLogger("CoTEvaluatorServerV2")

def call_vllm_local_api(prompt: str) -> str | None:
    client = openai.OpenAI(base_url=VLLM_API_ENDPOINT, api_key="<YOUR_API_KEY>", timeout=600.0)
    prompt_length = len(prompt)
    logger.info(f"Prompt length: {prompt_length} chars")
    if prompt_length > 6000:
        logger.warning(f"Prompt is long ({prompt_length} chars); may trigger 400 errors")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="llm",
                messages=[{"role": "user", "content": prompt}],
                **INFERENCE_PARAMS
            )
            raw_content = response.choices[0].message.content

            # Keep raw response logging lightweight to avoid bloating logs.
            logger.debug(f"Raw response (first 500 chars): {raw_content[:500]}")

            if "</think>\n\n" in raw_content:
                main_content = raw_content.split("</think>\n\n", 1)[1]
            else:
                main_content = raw_content
            preliminary_content = main_content.strip()
            json_block_pattern = r'^```(?:json)?\s*\n?(.*?)\n?```\s*$'
            match = re.search(json_block_pattern, preliminary_content, re.DOTALL | re.MULTILINE)
            if match:
                final_content = match.group(1).strip()
            else:
                final_content = preliminary_content
            
            # Best-effort JSON validation for debugging.
            try:
                test_json = json.loads(final_content)
                logger.debug(f"JSON validated. Keys: {list(test_json.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON validation failed: {e}")
                lines = final_content.split('\n')
                if e.lineno <= len(lines):
                    error_line = lines[e.lineno - 1] if e.lineno > 0 else ""
                    logger.error(f"Error line: '{error_line}'")

            return final_content
        except Exception as e:
            logger.error(f"vLLM API call failed ({attempt + 1}/{MAX_RETRIES}): {e}")
            
            # Special-case 400 errors for faster diagnosis.
            if "400" in str(e) or "Bad Request" in str(e):
                logger.error("Detected 400 Bad Request. Possible causes: long prompt, max_tokens, model name, request format.")
                logger.error(f"Prompt head (200 chars): {prompt[:200]}")
                logger.error(f"Prompt tail (200 chars): {prompt[-200:]}")

            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
    return None

def parse_scores_robustly_dynamic(response_text: str) -> Dict[str, int] | None:
    """
    Parse scores using regex patterns derived from the config keys.

    Returns a dict mapping key -> score so ablation setups can change dimensions
    without touching code.
    """
    if not isinstance(response_text, str): return None
    
    mapping = SERVER_CONFIG.get("score_mapping", {})
    # Collect all keys that should be present in the response (plan + execution).
    plan_keys = mapping.get("plan_keys", [])
    exec_key = mapping.get("execution_key")
    
    target_keys = plan_keys + ([exec_key] if exec_key else [])
    
    extracted_scores = {}
    missing_keys = []

    for key in target_keys:
    # Dynamic regex: find '"<key>" ... "score": <digit>'.
    # Use re.escape to safely handle special characters in keys.
        pattern = rf'"{re.escape(key)}"\s*:\s*\{{[^}}]*"score"\s*:\s*([0-5])'
        
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            extracted_scores[key] = int(match.group(1))
        else:
            missing_keys.append(key)
    
    # Return partial results and let the reward function decide defaults.
    if extracted_scores:
        if missing_keys:
            logger.warning(f"Regex parsing partially succeeded; missing keys: {missing_keys}")
        else:
            logger.info(f"Regex parsing succeeded: {extracted_scores}")
        return extracted_scores
    
    return None

def calculate_process_reward(scores_dict: Dict[str, int]) -> float:
    """
    Compute the V3 process reward via an addition + penalty scheme.
    """
    try:
        mapping = SERVER_CONFIG.get("score_mapping", {})
        penalty_cfg = SERVER_CONFIG.get("penalty_map", {})

        # 1) Compute R_plan (aggregate planning score).
        plan_keys = mapping.get("plan_keys", [])
        
        valid_plan_scores = []
        for key in plan_keys:
            if key in scores_dict:
                valid_plan_scores.append(scores_dict[key])
            else:
                logger.warning(f"Missing plan key '{key}' in scores; defaulting to 1")
                valid_plan_scores.append(1)
        
        if valid_plan_scores:
            # Normalize: (s - 1) / 4.0
            normalized_plan = [(s - 1) / 4.0 for s in valid_plan_scores]
            # Arithmetic mean
            r_plan = sum(normalized_plan) / len(normalized_plan)
        else:
            r_plan = 0.0
            
        # 2) Compute execution penalty.
        exec_key = mapping.get("execution_key")
        s6 = scores_dict.get(exec_key, 1)  # default worst

        # Lookup table (config keys are strings: "5", "4", ...)
        p = penalty_cfg.get(str(s6), -1.5)

        # 3) Final process reward.
        r_process = max(0.0, r_plan + p)  # clamp to non-negative (tune if needed)
        
        logger.info(
            f"Calc V3: plan_avg={r_plan:.3f} (n={len(valid_plan_scores)}), penalty={p} (exec_score={s6}) -> final={r_process:.3f}"
        )
        return r_process
        
    except Exception as e:
        logger.error(f"Failed to compute process reward: {e}", exc_info=True)
        return 0.0

def process_single_evaluation(data_tuple: tuple) -> tuple:
    # task tuple includes the prompt template content
    i, source, cot_analysis, final_translation, feature_report, template_content = data_tuple
    
    feature_report_str = json.dumps(feature_report, ensure_ascii=False, indent=2)
    try:
        evaluation_prompt = template_content.format(
            source_text=source,
            cot_analysis=cot_analysis,
            final_translation=final_translation,
            feature_report=feature_report_str
        )
        
        evaluation_response = call_vllm_local_api(evaluation_prompt)
        
        if not evaluation_response:
            logger.warning(f"Sample {i} evaluation failed: empty API response")
            return i, 0.0, {"error": "vLLM API call failed"}
        
        scores_dict = parse_scores_robustly_dynamic(evaluation_response)
        
        if scores_dict:
            process_reward = calculate_process_reward(scores_dict)
            logger.info(f"Sample {i} evaluated. Process reward: {process_reward:.4f}")
            return i, process_reward, {"scores": scores_dict}
        else:
            logger.error(f"Sample {i}: failed to parse scores via regex.")
            return i, 0.0, {"error": "regex parsing failed", "raw_response": evaluation_response[:200]}
            
    except Exception as e:
        logger.error(f"Sample {i}: unexpected error: {e}", exc_info=True)
        return i, 0.0, {"error": f"unexpected error: {str(e)}"}

# API request model.
class CoTEvaluationRequest(BaseModel):
    sources: List[str]
    cot_analyses: List[str]
    final_translations: List[str]
    feature_reports: List[Dict] 
    num_processes: int = 64

@app.post("/evaluate_cot", response_model=Dict)
async def evaluate_cot(request: CoTEvaluationRequest):
    try:
        num_samples = len(request.sources)
        logger.info(f"Received request: {num_samples} samples, {request.num_processes} processes")
        
        # Load the in-memory template loaded during startup.
        current_template = SERVER_CONFIG.get("template", "")
        if not current_template:
            raise HTTPException(status_code=500, detail="Configuration Error: Prompt template not loaded.")

        evaluation_tasks = [
            (i, src, cot, trans, report, current_template) 
            for i, (src, cot, trans, report) in enumerate(
                zip(request.sources, request.cot_analyses, request.final_translations, request.feature_reports)
            )
        ]
        
        process_rewards = [0.0] * num_samples
        evaluation_details = [{}] * num_samples
        
        with Pool(processes=request.num_processes) as pool:
            results = pool.map(process_single_evaluation, evaluation_tasks)
            for i, process_reward, evaluation_detail in results:
                process_rewards[i] = process_reward
                evaluation_details[i] = evaluation_detail
                
        logger.info(f"Finished CoT evaluation for {num_samples} samples")
        return {"process_rewards": process_rewards, "evaluation_details": evaluation_details}
    except Exception as e:
        logger.error(f"CoT evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CoT evaluation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def main():
    global VLLM_API_ENDPOINT, SERVER_CONFIG
    parser = argparse.ArgumentParser(description="CoT Evaluator Server V2 (Configurable)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--vllm-port", type=int, default=8080)
    parser.add_argument("--config-file", type=str, required=True, help="Path to evaluator config JSON")
    args = parser.parse_args()
    
    VLLM_API_ENDPOINT = f"http://localhost:{args.vllm_port}/v1"
    
    # 1) Load config JSON
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    
    with open(args.config_file, 'r', encoding='utf-8') as f:
        SERVER_CONFIG = json.load(f)
    
    # 2) Load the prompt template file referenced by the config
    prompt_path = SERVER_CONFIG.get("prompt_file")
    if not prompt_path or not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file specified in config not found: {prompt_path}")
        
    with open(prompt_path, 'r', encoding='utf-8') as f:
        SERVER_CONFIG["template"] = f.read()

    log_filepath = setup_logging("cot_evaluator_server_A_v2")
    logger.info(f"Starting CoT evaluator server on {args.host}:{args.port}")
    logger.info(f"Config file: {args.config_file}")
    logger.info("Scoring logic: V3 (dynamic parsing + addition-penalty)")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()