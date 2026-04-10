# CEMT

This work is developed at **China Telecom AI Technology Co., Ltd.**

[English](README.md) | [中文](README_ZH.md)

This repository contains the code and data for CEMT.

## Environment Setup

Please follow the steps below to configure the environment:

1. **Install `verl` (HybridFlow)**:
   This project is built on the high-performance reinforcement learning framework [verl (HybridFlow)](https://github.com/volcengine/verl). Please refer to its official documentation for installation:

   ```bash
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install -e .
   ```

2. **Install `vllm`**:
   Inference and serving require `vllm >= 0.8.5`.

   ```bash
   pip install "vllm>=0.8.5"
   ```

3. **Other dependencies**:

   ```bash
   pip install unbabel-comet fastapi "uvicorn[standard]" psutil omegaconf tensordict
   ```

4. **Download evaluation models**:
   You need to download the following models for evaluation:
   * **XCOMET-XL**
   * **CometKiwi-23-XL**

## Data Preparation

* **Raw data**: Located in `data/raw`
* **Test data**: Located in `data/test/main_test`
* **Training data**: Processed training data is located in `data/train`

## Training

* **SFT (Supervised Fine-Tuning)**:
  ```bash
  bash train/sft_cemt.sh
  ```

* **GRPO (Group Relative Policy Optimization)**:
  This project uses [Qwen2.5](https://github.com/QwenLM/Qwen2.5) as the base model.
  ```bash
  bash train/grpo_cemt.sh
  ```

## Inference and Evaluation

Evaluation scripts are located in the `inference/` directory.

```bash
bash inference/inference_eval_cemt.sh
```

## Distributed GRPO Training Guide

If you want to deploy GRPO reward models, to avoid complex resource configuration processes, it is recommended to deploy Comet and LLM models on different machines.

The following is a guide for deploying reward models and running distributed training.

### 1. Deploy Various Reward Model Services

We will start three independent reward services on different servers.

#### 1.1 Deploy XCOMET Service on [COMET Reward Node A]

**Node A Terminal:**

1. Enter the project directory:
   ```bash
   cd /path/to/CEMT
   ```

2. Install dependencies:
   ```bash
   pip install fastapi "uvicorn[standard]" unbabel-comet psutil
   ```

3. Get and record the local IP address:
   ```bash
   export NODE_A_IP=$(hostname -I | awk '{print $1}')
   echo "✅ XCOMET service started. Please note this IP address: $NODE_A_IP"
   # Example output: <YOUR_NODE_IP>
   ```

4. Start the service (recommended to use `tmux`):
   ```bash
   tmux new -s xcomet_server
   python3 reward/reward_server.py --model-name "XCOMET" --port 8001
   ```

#### 1.2 Deploy CometKiwi Service on [COMET Reward Node B]

**Node B Terminal:**

1. Enter the project directory:
   ```bash
   cd /path/to/CEMT
   ```

2. Install dependencies:
   ```bash
   pip install fastapi "uvicorn[standard]" unbabel-comet psutil
   ```

3. Get and record the local IP address:
   ```bash
   export NODE_B_IP=$(hostname -I | awk '{print $1}')
   echo "✅ CometKiwi service started. Please note this IP address: $NODE_B_IP"
   # Example output: <YOUR_NODE_IP>
   ```

4. Start the service (recommended to use `tmux`):
   ```bash
   tmux new -s kiwi_server
   python3 reward/reward_server.py --model-name "CometKiwi" --port 8002
   ```

#### 1.3 Deploy CoT Evaluation Service on [CoT Evaluation Node Group] (LLM as Judge)

This service requires deploying the Qwen3-235B model. This experiment uses 2*8 A100 40G to deploy Qwen3-235B as the server.

**Step 1.3.1 - Prepare and Start Ray Head on [CoT Evaluation Master Node]**

**CoT Evaluation Master Node Terminal:**

1. Set the model path environment variable:
   ```bash
   export MODEL_PATH="/path/to/models/Qwen3-235B-Instruct"
   ```

2. Automatically get the local IP as the Ray Head address:
   ```bash
   export HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
   echo "CoT Evaluation Master Node IP is: $HEAD_ADDRESS"
   ```

3. Get and record the local IP address (this is the final service IP):
   ```bash
   export NODE_COT_IP=$HEAD_ADDRESS
   echo "✅ CoT Evaluation service will be deployed here. Please note this IP address: $NODE_COT_IP"
   ```

4. Start the Ray Head service:
   ```bash
   ray start --head --port=6379 --dashboard-host='0.0.0.0'
   ```

**Step 1.3.2 - Join Ray Cluster on [CoT Evaluation Worker Nodes]**

**CoT Evaluation Worker Node Terminal:**

Use the master node IP address obtained in the previous step to replace `<CoT Evaluation Master Node IP>`.

```bash
ray start --address="<CoT Evaluation Master Node IP>:6379"
```

After success, you can access `http://<CoT Evaluation Master Node IP>:8265` in the browser on the master node to view the Ray Dashboard. You should see 2 nodes.

**Step 1.3.3 - Return to [CoT Evaluation Master Node] to Start VLLM and Evaluation Service**

**CoT Evaluation Master Node Terminal:**

Ensure the `MODEL_PATH` environment variable is still valid.

5. Start the vLLM service (recommended to use `tmux`):
   ```bash
   tmux -u new -s vllm_server
   # Note: This command will automatically utilize GPU resources across the entire Ray cluster
   # Used for judging
   vllm serve ${MODEL_PATH} \
   --dtype auto \
   --api-key <YOUR_API_KEY> \
   --port 8080 \
   --gpu-memory-utilization 0.9 \
   --tensor-parallel-size 8 \
   --pipeline-parallel-size 2 \
   --served-model-name llm \
   --trust-remote-code \
   --max-model-len 6000 \
   --max-num-seqs 64 
   ```

6. Start the CoT Evaluator service:
   ```bash
   tmux -u new -s cot_server
   cd /path/to/CEMT
   python3 reward/cot_evaluator_server.py --port 8003 --vllm-port 8080 --config-file prompts/judge/config_full_cemt.json
   ```

### 2. Configure and Start Training on [Main Training Node]

Now that all reward services are online, we return to the main training node to start GRPO.

#### 2.1 Configure Environment Variables

Fill in the three IP addresses recorded in the previous steps into the shell and execute.

**Main Training Node Terminal:**

1. Fill in your recorded IP addresses:
   ```bash
   export XCOMET_SERVER_URL="http://<Node A IP>:8001/predict"
   export KIWI_SERVER_URL="http://<Node B IP>:8002/predict"
   export COT_EVALUATOR_SERVER_URL="http://<CoT Evaluation Master Node IP>:8003/evaluate_cot"
   ```

2. Verify environment variables:
   ```bash
   echo "XCOMET URL: $XCOMET_SERVER_URL"
   echo "KIWI URL: $KIWI_SERVER_URL"
   echo "COT URL: $COT_EVALUATOR_SERVER_URL"
   ```

#### 2.2 Start Training

**Main Training Node Terminal:**

1. Enter the `verl` framework directory:
   ```bash
   cd /path/to/verl
   ```

2. Install `unbabel-comet`:
   ```bash
   pip install unbabel-comet
   ```

3. Start training (recommended to use `tmux`):
   ```bash
   tmux -u new -s grpo

   # Replace </path/to/your/script.sh> with your actual startup script path
   bash </path/to/your/script.sh>
   ```

### 3. Post-Training Model Merging

After training is complete, use verl's model merger tool to convert FSDP weights to directly usable HuggingFace format.

**Main Training Node Terminal:**

1. Install dependencies:
   ```bash
   pip install omegaconf tensordict
   ```

2. Execute the merge command (please modify the paths to your actual model output paths):
   ```bash
   python -m verl.model_merger merge \
     --backend fsdp \
     --local_dir /path/to/CEMT/models/<your_experiment_name>/global_step_XXX/actor \
     --target_dir /path/to/CEMT/models/<your_experiment_name>/global_step_XXX/actor/huggingface
   ```

## Citation

If you find this repository helpful for your research, please consider citing CEMT:

```bibtex
@inproceedings{shi2026cemt,
    title={CEMT: Chain-of-Thought Enhanced Machine Translation},
    author={Lingling Shi and Haoyu Jin and Ruiyu Fang and Shuangyong Song and Jinsong Su and Yongxiang Li and Xuelong Li},
    booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
    year={2026},
    publisher={Association for Computational Linguistics},
}
```

## Acknowledgements

Thanks to [verl (HybridFlow)](https://github.com/volcengine/verl) for providing the flexible and efficient RLHF framework support.

Thanks to the COMET series of evaluation metrics and their implementation work for providing important tools for machine translation quality assessment.

Thanks to the [Qwen](https://github.com/QwenLM) series of models and their open-source ecosystem for providing the base models for this work.
