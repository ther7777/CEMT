#!/usr/bin/env bash
# 用途：启动 CEMT 的 GRPO 训练，并连接外部奖励服务。
# 输入：SFT 检查点、GRPO 训练集 `data/train/grpo_cemt_data.parquet`、验证集 `data/train/grpo_val.parquet`、奖励服务地址。
# 输出：GRPO 检查点、TensorBoard 日志与 `full_run.log`。
# 运行示例：bash train/grpo_cemt.sh
set -ex

# 奖励服务地址可直接填写，也可通过环境变量覆盖。
export XCOMET_SERVER_URL="${XCOMET_SERVER_URL:-http://host:8001/predict}"
export KIWI_SERVER_URL="${KIWI_SERVER_URL:-http://host:8002/predict}"
export COT_EVALUATOR_SERVER_URL="${COT_EVALUATOR_SERVER_URL:-http://host:8003/evaluate_cot}"

export SFT_MODEL_PATH="path/to/cemt/sft/checkpoint"
export TRAIN_DATA_PATH="data/train/grpo_cemt_data.parquet"
export VALID_DATA_PATH="data/train/grpo_val.parquet"
export PROJECT_NAME="path/to/project"
export EXPERIMENT_NAME="path/to/experiment"
export SAVE_DIR="path/to/checkpoints/${EXPERIMENT_NAME}"
export TENSORBOARD_DIR="path/to/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}/$(date +%Y%m%d-%H%M%S)"

echo "Cleaning old Hydra cache..."
rm -rf multirun/ outputs/
mkdir -p ${SAVE_DIR}

echo "Starting verl GRPO training on 8 GPUs..."
python3 -m verl.trainer.main_ppo \
    \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VALID_DATA_PATH} \
    data.shuffle=True \
    data.max_prompt_length=1400 \
    data.max_response_length=2000 \
    data.train_batch_size=64 \
    hydra.run.dir="${SAVE_DIR}" \
    trainer.default_local_dir=${SAVE_DIR} \
    actor_rollout_ref.model.path="${SFT_MODEL_PATH}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    custom_reward_function.path="path/to/CEMT/reward/mt_reward_function_client.py" \
    custom_reward_function.name="compute_bleu_xcomet_kiwi_cot_score_batch" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    algorithm.adv_estimator=grpo \
    critic.enable=False \
    critic.model.enable_gradient_checkpointing=True \
    reward_model.enable=False \
    reward_model.reward_manager="batch" \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.optim.total_training_steps=700 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.optim.total_training_steps=700 \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_num_batched_tokens=6000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.logger='["console","tensorboard"]' 2>&1 | tee "${SAVE_DIR}/full_run.log"
