#!/usr/bin/env bash
# CANONICAL training configuration. Edit this file to configure all model usage.
# Install with ./install.sh to preserve secrets/Podman config in an existing checkout.
# If using an existing synth_gen repo, do not overwrite its config.sh with unzip -o.
# Personal credentials remain external: export HF_TOKEN and SYNTHGEN_REMOTE_TOKEN,
# or keep them in your preexisting config.sh outside the marked block.

# BEGIN CONTINUAL_RL_TRAINING_CONFIG
# This block belongs in custome/config.sh. Configure TRAINING ONLY HERE.
# Edit these values; rl_config.sh derives paths but never overwrites training settings.
# Exactly ONE selectable base model for SFT, GRPO, synth_gen and eval. Change RL_BASE_MODEL below to switch families.
export RL_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Sequential SFT tasks; MUST end with code for the Code-SFT PCA bootstrap.
export RL_SFT_TASKS="math social code"
export RL_SFT_STEPS_MATH=50
export RL_SFT_STEPS_SOCIAL=50
export RL_SFT_STEPS_CODE=50
# Upper limit for dataset epochs. The actual step limits above govern stopping.
export RL_SFT_EPOCHS=1000
export RL_GRPO_TASK="code"
export RL_EVAL_TASKS="math social code"
export RL_TRACKS="baseline ours"
# Separate step limits. Each track starts from the same Code SFT checkpoint.
export RL_GRPO_STEPS_BASELINE=50
export RL_GRPO_STEPS_OURS=50

# Real-time terminal progress; independent of slow model downloads or generation.
# Set RL_PROGRESS_ENABLED=0 to silence our progress log (VERL output remains).
export RL_PROGRESS_ENABLED=1
export RL_PROGRESS_INTERVAL_SECS=15
export RL_PROGRESS_CHILD_OUTPUT=1
export RL_PROGRESS_EVAL_EVERY=10
export RL_PROGRESS_REWARD_EVERY=10

# Experiment ID and Hugging Face private models / generated data.
export RL_RUN_ID="${RL_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
export HF_TOKEN="${HF_TOKEN:-}" # Set with export HF_TOKEN='hf_...' if desired
export RL_HF_TOKEN="${RL_HF_TOKEN:-${HF_TOKEN}}"
export RL_HF_OWNER="${RL_HF_OWNER:-}"
export RL_HF_MODEL_PREFIX="continual-fsdp"
export RL_HF_DATA_PREFIX="continual-code"
# Optional: existing PRIVATE HF dataset repo to append to, e.g. username/my-dataset.
# Empty: automatically create once, then reuse the saved repo across runs.
# All runs/tracks use separate destination folders inside the SAME repo.
export RL_HF_DATA_REPO=""
# Every N NEW JSONL records per stream triggers an asynchronous private upload.
# Remaining (<N) records are uploaded when that track finishes.
export RL_UPLOAD_EVERY=25
export RL_UPLOAD_POLL_SECS=2
export RL_HF_MAX_RETRIES=10
export RL_MODEL_SAVE_EVERY=1

# Distributed training / common model policy.
export RL_RANDOM_SEED=42
# Optionally set CUDA_VISIBLE_DEVICES here to restrict the GPUs used by
# SFT/GRPO/synth_gen. When unset, CUDA detects all GPUs available to this job.
# export CUDA_VISIBLE_DEVICES="0,1"
# 'auto' resolves to the number of CUDA devices visible after the mask.
export RL_GPUS_PER_NODE=auto
if [[ "$RL_GPUS_PER_NODE" == "auto" ]]; then
  RL_GPUS_PER_NODE="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)" || {
    echo 'ERROR: Cannot detect CUDA devices. Activate the Torch virtualenv.' >&2
    return 1 2>/dev/null || exit 1
  }
  export RL_GPUS_PER_NODE
fi
export RL_NNODES=1
export RL_NODE_RANK="${RL_NODE_RANK:-0}"
export RL_MASTER_ADDR="${RL_MASTER_ADDR:-127.0.0.1}"
export RL_MASTER_PORT=29500
export RL_FSDP_STRATEGY=fsdp2
export RL_FSDP_SIZE=-1
export RL_PARAM_OFFLOAD=false
export RL_OPTIM_OFFLOAD=false
export RL_GRAD_CKPT=true

# SFT optimizer settings (same settings, task-specific step limits above).
export RL_SFT_LR=2e-5
export RL_SFT_BATCH=32
export RL_SFT_MICRO=1
export RL_SFT_MAX_LENGTH=2048
export RL_SFT_TEST_EVERY=1

# GRPO optimizer settings (track-specific step limits above).
export RL_GRPO_BATCH=8
export RL_GRPO_MINI=16
export RL_GRPO_MICRO=1
export RL_GRPO_LR=1e-6
export RL_ROLLOUT_N=4
export RL_ROLLOUT_TP=1
export RL_ROLLOUT_MEM=0.45
export RL_PROMPT_LEN=1024
export RL_RESPONSE_LEN=1024
export RL_KL=0.001
export RL_GRPO_VAL_MAX=64
export RL_EVAL_EVERY=1
export RL_EVAL_LIMIT=0

# Actor snapshot is reused for synth_gen; never load a separate generation model.
# CUDA devices are assigned AUTOMATICALLY from CUDA_VISIBLE_DEVICES.
# No RL_SYNTH_DEVICE / RL_SYNTH_WORKER_DEVICES overrides or CPU fallback.
export RL_POLICY_SNAPSHOT_EVERY=1
export RL_SYNTH_MAX_POLICY_LAG_STEPS=2
export RL_POLICY_KEEP_SNAPSHOTS=2
export RL_POLICY_POLL_SECS=2
export RL_POLICY_WAIT_TIMEOUT=3600
export RL_SYNTH_NEW_TOKENS=768
# Reject generated questions containing Python solutions/tests; regenerate up to N attempts.
export RL_SYNTH_PROBLEM_RETRIES=4
# Number of attempts to generate a valid independent pytest for one prediction.
export RL_SYNTH_TEST_RETRIES=3
export RL_SYNTH_TEMP=0.7
export RL_SYNTH_TOP_P=0.95
export RL_SYNTH_TEST_SAMPLES=4
export RL_SYNTH_PROCESSES=4
export RL_SYNTH_BATCH_SIZE=4
# PyTorch allocator budget allocated to ALL synth processes on EACH visible GPU.
# With 1 GPU and 4 workers: each receives 0.90/4 = 22.5% of that GPU.
# This is a memory cap, not a guarantee that 4 model replicas will fit.
export RL_SYNTH_GPU_MEMORY_FRACTION=0.90
export RL_SYNTH_START_METHOD=spawn
export RL_SYNTH_VERBOSE=0
export RL_SYNTH_QUEUE_SIZE=4
export RL_VERIFY_CONCURRENCY=2
export RL_MAX_ATTEMPTS_PER_STEP=1200
export RL_ALLOWED_PIP="numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 sympy==1.13.3 networkx==3.4.2"
export RL_CHECK_COVERAGE=0

# Prior-subtask PCA: a configurable fraction of UNIQUE actually accessed SFT
# training examples per task (not the full unused benchmark dataset).
# Samples are recorded during each task's SFT and fitted together using the
# SAME frozen Code-SFT encoder before GRPO (shared feature coordinates).
export RL_PCA_REFERENCE_FRACTION=0.10
export RL_PCA_MIN_REFERENCE_SAMPLES=4
export RL_PCA_EMBED_BATCH_SIZE=8
# 0: compare against PREVIOUS learned subtasks only, not current Code task.
export RL_PCA_INCLUDE_CURRENT_TASK=0
# any: at least one prior task passes all 1D/2D/3D thresholds; all: every task.
export RL_PCA_MATCH_POLICY=any

# Prior-task PCA and reward policies.
export RL_FEATURE_DEVICE="cuda:0"
export RL_SERVICE_CUDA_VISIBLE_DEVICES="${RL_SERVICE_CUDA_VISIBLE_DEVICES:-}"
export RL_MAX_FEATURE_TOKENS=256
export RL_THRESHOLD_D1=1.5
export RL_THRESHOLD_D2=1.5
export RL_THRESHOLD_D3=1.5
export RL_REQUIRED_PASS=1
export RL_REQUIRE_SYNTH_VERIFIER=1
export RL_WEIGHT_D1=1
export RL_WEIGHT_D2=1
export RL_WEIGHT_D3=1
export RL_WEIGHT_EXEC=2
export RL_WEIGHT_ERR=1
export RL_VISUALIZE=1
export RL_VIS_EVERY=1

# Optional per-step evaluation, reward service, VERL checkout.
export RL_FULL_EVAL_EVERY=0
export RL_EVAL_CUDA_VISIBLE_DEVICES="${RL_EVAL_CUDA_VISIBLE_DEVICES:-}"
export RL_EVAL_TP=1
export RL_PER_STEP_EVAL_LIMIT=64
export RL_SERVICE_PORT="${RL_SERVICE_PORT:-18865}"
export RL_SERVICE_BIND="${RL_SERVICE_BIND:-127.0.0.1}"
export RL_SERVICE_TOKEN="${RL_SERVICE_TOKEN:-}"
export SYNTHGEN_USE_REMOTE_PODMAN=1
export RL_VERL_REPO="${RL_VERL_REPO:-/root/verl}"
# Ensure the legacy synth_gen default-LangChain setting cannot silently choose
# Together/Llama (the RL adapter explicitly loads RL_BASE_MODEL / its checkpoints).
# To run the independent original synth_gen demo with Together, set LLM manually
# in a separate shell. This pipeline intentionally has only ONE model family.
unset LLM
unset RL_SYNTH_MODEL
# END CONTINUAL_RL_TRAINING_CONFIG
