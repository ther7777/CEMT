# CEMT

[English](README.md) | [中文](README_ZH.md)

本仓库包含 CEMT 的代码和数据。

## 环境配置

请按照以下步骤配置环境：

1. **安装 `verl` (HybridFlow)**：
   本项目基于高性能强化学习框架 [verl (HybridFlow)](https://github.com/volcengine/verl) 开发。请参考其官方文档进行安装：

   ```bash
   git clone https://github.com/volcengine/verl.git
   cd verl
   pip install -e .
   ```

2. **安装 `vllm`**：
   推理与服务需要安装 `vllm >= 0.8.5`。

   ```bash
   pip install "vllm>=0.8.5"
   ```

3. **其他依赖**：

   ```bash
   pip install unbabel-comet fastapi "uvicorn[standard]" psutil omegaconf tensordict
   ```

4. **下载评估模型**：
   你需要下载以下模型用于评估：
   * **XCOMET-XL**
   * **CometKiwi-23-XL**

## 数据准备

* **原始数据**: 位于 `data/raw`
* **测试数据**：位于 `data/test/main_test`
* **训练数据**：处理好的训练数据位于 `data/train`

## 训练

* **SFT (监督微调)**：
  ```bash
  bash train/sft_cemt.sh
  ```

* **GRPO (Group Relative Policy Optimization)**：
  本项目使用 [Qwen2.5](https://github.com/QwenLM/Qwen2.5) 作为基座模型。
  ```bash
  bash train/grpo_cemt.sh
  ```

## 推理与评估

评估脚本位于 `inference/` 目录下。

```bash
bash inference/inference_eval_cemt.sh
```

## 分布式 GRPO 训练指南

如果你想部署 GRPO 奖励模型，为了避免复杂的配置资源过程，建议在不同的机器上部署 Comet 和 LLM 模型。

以下是部署奖励模型和运行分布式训练的手册。

### 1. 部署各类奖励模型服务

我们将会在不同的服务器上启动三个独立的奖励服务。

#### 1.1 在【COMET 奖励节点 A】上部署 XCOMET 服务

**节点 A 终端：**

1. 进入项目目录：
   ```bash
   cd /path/to/CEMT
   ```

2. 安装依赖：
   ```bash
   pip install fastapi "uvicorn[standard]" unbabel-comet psutil
   ```

3. 获取并记录本机 IP 地址：
   ```bash
   export NODE_A_IP=$(hostname -I | awk '{print $1}')
   echo "✅ XCOMET 服务已启动。请记下此 IP 地址: $NODE_A_IP"
   # 示例输出: <YOUR_NODE_IP>
   ```

4. 启动服务 (建议使用 `tmux`)：
   ```bash
   tmux new -s xcomet_server
   python3 reward/reward_server.py --model-name "XCOMET" --port 8001
   ```

#### 1.2 在【COMET 奖励节点 B】上部署 CometKiwi 服务

**节点 B 终端：**

1. 进入项目目录：
   ```bash
   cd /path/to/CEMT
   ```

2. 安装依赖：
   ```bash
   pip install fastapi "uvicorn[standard]" unbabel-comet psutil
   ```

3. 获取并记录本机 IP 地址：
   ```bash
   export NODE_B_IP=$(hostname -I | awk '{print $1}')
   echo "✅ CometKiwi 服务已启动。请记下此 IP 地址: $NODE_B_IP"
   # 示例输出: <YOUR_NODE_IP>
   ```

4. 启动服务 (建议使用 `tmux`)：
   ```bash
   tmux new -s kiwi_server
   python3 reward/reward_server.py --model-name "CometKiwi" --port 8002
   ```

#### 1.3 在【CoT 评估节点组】上部署 CoT 评估服务 (LLM as Judge)

此服务需要部署 Qwen3-235B 模型，本实验采用 2*8 A100 40G 部署 Qwen3-235B 作为服务端。

**步骤 1.3.1 - 在【CoT 评估主节点】上准备并启动 Ray Head**

**CoT 评估主节点终端：**

1. 设置模型路径环境变量：
   ```bash
   export MODEL_PATH="/path/to/models/Qwen3-235B-Instruct"
   ```

2. 自动获取本机 IP 作为 Ray Head 地址：
   ```bash
   export HEAD_ADDRESS=$(hostname -I | awk '{print $1}')
   echo "CoT 评估主节点 IP 为: $HEAD_ADDRESS"
   ```

3. 获取并记录本机 IP 地址 (这是服务最终的 IP)：
   ```bash
   export NODE_COT_IP=$HEAD_ADDRESS
   echo "✅ CoT 评估服务将部署于此。请记下此 IP 地址: $NODE_COT_IP"
   ```

4. 启动 Ray Head 服务：
   ```bash
   ray start --head --port=6379 --dashboard-host='0.0.0.0'
   ```

**步骤 1.3.2 - 在【CoT 评估工作节点】上加入 Ray 集群**

**CoT 评估工作节点终端：**

使用上一步获取到的主节点 IP 地址替换 `<CoT 评估主节点 IP>`。

```bash
ray start --address="<CoT 评估主节点 IP>:6379"
```

成功后，您可以在主节点浏览器访问 `http://<CoT 评估主节点 IP>:8265` 查看 Ray Dashboard，应能看到 2 个节点。

**步骤 1.3.3 - 回到【CoT 评估主节点】上启动 VLLM 和评估服务**

**CoT 评估主节点终端：**

确保 `MODEL_PATH` 环境变量仍然有效。

5. 启动 vLLM 服务 (建议使用 `tmux`)：
   ```bash
   tmux -u new -s vllm_server
   # 注意：此命令将自动利用整个 Ray 集群的 GPU 资源
   # 用于裁判
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

6. 启动 CoT 评估器服务：
   ```bash
   tmux -u new -s cot_server
   cd /path/to/CEMT
   python3 reward/cot_evaluator_server.py --port 8003 --vllm-port 8080 --config-file prompts/judge/config_full_cemt.json
   ```

### 2. 在【主训练节点】上配置并启动训练

现在所有奖励服务都已在线，我们回到主训练节点来启动 GRPO。

#### 2.1 配置环境变量

将上一步中记录的三个 IP 地址填入 Shell 并执行。

**主训练节点终端：**

1. 填入你记录的 IP 地址：
   ```bash
   export XCOMET_SERVER_URL="http://<节点 A 的 IP>:8001/predict"
   export KIWI_SERVER_URL="http://<节点 B 的 IP>:8002/predict"
   export COT_EVALUATOR_SERVER_URL="http://<CoT 评估主节点的 IP>:8003/evaluate_cot"
   ```

2. 验证环境变量：
   ```bash
   echo "XCOMET URL: $XCOMET_SERVER_URL"
   echo "KIWI URL: $KIWI_SERVER_URL"
   echo "COT URL: $COT_EVALUATOR_SERVER_URL"
   ```

#### 2.2 启动训练

**主训练节点终端：**

1. 进入 `verl` 框架目录：
   ```bash
   cd /path/to/verl
   ```

2. 安装 `unbabel-comet`：
   ```bash
   pip install unbabel-comet
   ```

3. 启动训练 (建议使用 `tmux`)：
   ```bash
   tmux -u new -s grpo

   # 将 </path/to/your/script.sh> 替换为你的实际启动脚本路径
   bash </path/to/your/script.sh>
   ```

### 3. 训练后模型合并

训练完成后，使用 `verl` 的模型合并工具将 FSDP 权重转换为可直接使用的 HuggingFace 格式。

**主训练节点终端：**

1. 安装依赖：
   ```bash
   pip install omegaconf tensordict
   ```

2. 执行合并命令 (请将路径修改为你的实际模型输出路径)：
   ```bash
   python -m verl.model_merger merge \
     --backend fsdp \
     --local_dir /path/to/CEMT/models/<your_experiment_name>/global_step_XXX/actor \
     --target_dir /path/to/CEMT/models/<your_experiment_name>/global_step_XXX/actor/huggingface
   ```

## 引用

如果你觉得本仓库对你的研究有帮助，欢迎引用 CEMT。

## 致谢

感谢 [verl (HybridFlow)](https://github.com/volcengine/verl) 提供的灵活高效的 RLHF 框架支持。

感谢 COMET 系列评测指标及其实现工作为机器翻译质量评估提供的重要工具支持。

感谢 [Qwen](https://github.com/QwenLM) 系列模型及其开源生态为本工作提供的基础模型。
