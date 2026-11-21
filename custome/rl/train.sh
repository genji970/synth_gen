
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/rl_config.sh"

MODE="${1:?Missing sft|grpo}"
TASK="${2:?Missing task}"
MODEL="${3:-$RL_MODEL}"

export RL_TRAIN_MODEL_PATH="$MODEL"

if (( RL_GPUS_PER_NODE % RL_ROLLOUT_TP != 0 )); then
    echo "ROLLOUT_TP must divide GPUs/node"
    exit 1
fi

if (( (RL_GRPO_BATCH * RL_ROLLOUT_N) % RL_GRPO_MINI != 0 )); then
    echo "Invalid GRPO mini-batch"
    exit 1
fi

if [[ "$MODE" == "sft" ]]; then
    torchrun \
        --nnodes="$RL_NNODES" \
        --nproc_per_node="$RL_GPUS_PER_NODE" \
        --node_rank="$RL_NODE_RANK" \
        --master_addr="$RL_MASTER_ADDR" \
        --master_port="$RL_MASTER_PORT" \
        -m verl.trainer.sft_trainer \
        engine=fsdp \
        engine.strategy="$RL_FSDP_STRATEGY" \
        engine.fsdp_size="$RL_FSDP_SIZE" \
        engine.param_offload="$RL_PARAM_OFFLOAD" \
        engine.optimizer_offload="$RL_OPTIM_OFFLOAD" \
        model.path="$MODEL" \
        model.enable_gradient_checkpointing="$RL_GRAD_CKPT" \
        model.use_remove_padding=true \
        data.train_files="$RL_DATA/${TASK}_train.parquet" \
        data.val_files="$RL_DATA/${TASK}_eval.parquet" \
        data.messages_key=messages \
        data.train_batch_size="$RL_SFT_BATCH" \
        data.max_length="$RL_SFT_MAX_LENGTH" \
        data.truncation="$RL_DATA_TRUNCATION" \
        data.micro_batch_size_per_gpu="$RL_SFT_MICRO" \
        optim.lr="$RL_SFT_LR" \
        trainer.total_epochs="$RL_SFT_EPOCHS" \
        trainer.test_freq="$RL_SFT_TEST_FREQ" \
        trainer.save_freq="$RL_SFT_SAVE_FREQ" \
        trainer.logger='["console"]' \
        trainer.resume_mode=disable \
        trainer.n_gpus_per_node="$RL_GPUS_PER_NODE" \
        trainer.nnodes="$RL_NNODES" \
        trainer.default_local_dir="$RL_CKPT/sft_$TASK"

elif [[ "$MODE" == "grpo" ]]; then
    [[ "$TASK" == "code" ]] || exit 1

    python -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=false \
        data.train_files="$RL_DATA/code_rl_train.parquet" \
        data.val_files="$RL_DATA/code_rl_eval.parquet" \
        data.shuffle=false \
        data.truncation="$RL_DATA_TRUNCATION" \
        data.train_batch_size="$RL_GRPO_BATCH" \
        data.max_prompt_length="$RL_PROMPT_LEN" \
        data.max_response_length="$RL_RESPONSE_LEN" \
        data.val_max_samples="$RL_GRPO_VAL_MAX" \
        actor_rollout_ref.model.path="$MODEL" \
        actor_rollout_ref.model.use_remove_padding=true \
        actor_rollout_ref.model.enable_gradient_checkpointing="$RL_GRAD_CKPT" \
        actor_rollout_ref.actor.strategy="$RL_FSDP_STRATEGY" \
        actor_rollout_ref.ref.strategy="$RL_FSDP_STRATEGY" \
        actor_rollout_ref.actor.fsdp_config.param_offload="$RL_PARAM_OFFLOAD" \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload="$RL_OPTIM_OFFLOAD" \
        actor_rollout_ref.ref.fsdp_config.param_offload="$RL_PARAM_OFFLOAD" \
        actor_rollout_ref.actor.optim.lr="$RL_GRPO_LR" \
        actor_rollout_ref.actor.ppo_mini_batch_size="$RL_GRPO_MINI" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$RL_GRPO_MICRO" \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.kl_loss_coef="$RL_KL" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n="$RL_ROLLOUT_N" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$RL_ROLLOUT_TP" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$RL_ROLLOUT_MEM" \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        reward.custom_reward_function.path="$ROOT/reward.py" \
        reward.custom_reward_function.name=compute_score \
        reward.reward_manager.name=naive \
        reward.num_workers=1 \
        trainer.n_gpus_per_node="$RL_GPUS_PER_NODE" \
        trainer.nnodes="$RL_NNODES" \
        trainer.total_epochs="$RL_GRPO_EPOCHS" \
        trainer.test_freq="$RL_GRPO_TEST_FREQ" \
        trainer.save_freq="$RL_GRPO_SAVE_FREQ" \
        trainer.logger='["console"]' \
        trainer.default_local_dir="$RL_CKPT/grpo_code" \
        trainer.v1.trainer_mode=sync
else
    echo "Unknown mode: $MODE"
    exit 1
fi
