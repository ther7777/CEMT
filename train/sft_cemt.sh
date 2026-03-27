#!/usr/bin/env bash
# 用途：启动 CEMT 的 SFT 训练。
# 输入：基础模型路径、SFT 训练集 `data/train/sft_cemt_data.parquet`、验证集 `data/train/sft_val_100.parquet`。
# 输出：训练检查点与日志目录 `path/to/checkpoints/${EXP_NAME}`。
# 运行示例：bash train/sft_cemt.sh
set -e

MODEL_PATH="path/to/base/model"
EXP_NAME="path/to/experiment"
DATA_FILE="data/train/sft_cemt_data.parquet"
VAL_FILE="data/train/sft_val_100.parquet"
OUTPUT_DIR="path/to/checkpoints/${EXP_NAME}"

NUM_GPUS=8
MICRO_BATCH_SIZE=1
TOTAL_TRAINING_STEPS=200
SAVE_FREQ=66
TEST_FREQ=10

mkdir -p "${OUTPUT_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" -m verl.trainer.fsdp_sft_trainer \
    model.partial_pretrain="${MODEL_PATH}" \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    model.lora_rank=0 \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    \
    data.train_files="${DATA_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    data.max_length=6000 \
    data.multiturn.enable=false \
    \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.lr_scheduler=cosine \
    \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.project_name="path/to/project" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.total_epochs=3 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.max_ckpt_to_keep=6 \
    trainer.resume_mode=auto \
    trainer.logger='["console","tensorboard"]' \
    trainer.checkpoint.save_contents='[model,optimizer,extra,hf_model]'  
