#!/usr/bin/env bash
# Run InfiniteWeb GUI agent GRPO training with verl.
#
# Prerequisites:
#   1. Install Playwright:
#        pip install playwright fastapi uvicorn
#        playwright install chromium
#
#   2. Start browser worker pool (one process per port):
#        DATASET_DIR=/path/to/InfiniteWeb-Dataset
#        for i in $(seq 1 64); do
#          python examples/wb_agent/env_server/web_env_server.py \
#            --port $((6000+i)) \
#            --dataset-dir "$DATASET_DIR" &
#        done
#        # Wait for workers to start
#        sleep 10
#
#   3. Prepare training data (run once; reads dataset directly, no server needed):
#        python examples/wb_agent/data_preprocess/prepare_wb_agent_data.py \
#          --dataset-dir /path/to/InfiniteWeb-Dataset \
#          --output-dir ~/data/wb_agent_verl
#
#   4. Set MODEL_PATH to your local Qwen3-VL checkpoint or HF model ID.
#
# Usage:
#   bash examples/wb_agent/run_wb_agent_grpo.sh
#   # override model:
#   MODEL_PATH=/path/to/model bash examples/wb_agent/run_wb_agent_grpo.sh

set -x

ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/wb_agent/config"
INTERACTION_CONFIG_PATH="$PROJECT_DIR/examples/wb_agent/config/interaction_config/wb_agent_interaction_config.yaml"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-7B-Instruct}"
DATA_DIR="${DATA_DIR:-$HOME/data/wb_agent_verl}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
N_GPUS="${N_GPUS:-8}"
OFFLOAD="${OFFLOAD:-False}"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='wb_agent_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=25 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$INTERACTION_CONFIG_PATH" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    reward.custom_reward_function.path="$PROJECT_DIR/examples/wb_agent/wb_agent_reward.py" \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='wb_agent_grpo' \
    trainer.experiment_name="qwen3vl_wb_agent_grpo_n8" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    "$@"
