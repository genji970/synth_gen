#!/usr/bin/env bash
# RunPod: source custome/config.sh && source custome/rl/rl_config.sh
export RL_MODEL="Qwen/Qwen2.5-7B-Instruct"
export RL_GPUS_PER_NODE=4 RL_NNODES=1 RL_NODE_RANK=0
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export RL_MASTER_ADDR="127.0.0.1" RL_MASTER_PORT=29500
export RL_FSDP_STRATEGY="fsdp2" RL_FSDP_SIZE=-1
export RL_PARAM_OFFLOAD=false RL_OPTIM_OFFLOAD=false RL_GRAD_CKPT=true
export RL_ROOT="/root/synth_gen/custome/rl"
export RL_DATA="$RL_ROOT/data" RL_OUT="$RL_ROOT/output" RL_CKPT="$RL_ROOT/checkpoints"
export RL_FEATURE_MODEL="$RL_CKPT/hf_after_code"  # frozen code-SFT policy snapshot; may require large CPU RAM
export RL_SFT_BATCH=32 RL_SFT_MICRO=1 RL_SFT_LR=2e-5 RL_SFT_EPOCHS=1
export RL_SFT_MAX_LENGTH=2048 RL_DATA_TRUNCATION=right
export RL_GRPO_BATCH=8 RL_GRPO_MINI=16 RL_GRPO_MICRO=1
export RL_GRPO_LR=1e-6 RL_GRPO_EPOCHS=1 RL_ROLLOUT_N=4
export RL_GRPO_VAL_MAX=64 RL_SFT_SAVE_FREQ=after_each_epoch RL_SFT_TEST_FREQ=after_each_epoch
export RL_GRPO_SAVE_FREQ=1 RL_GRPO_TEST_FREQ=1
export RL_ROLLOUT_TP=1 RL_ROLLOUT_MEM=0.5
export RL_PROMPT_LEN=1024 RL_RESPONSE_LEN=1024 RL_KL=0.001
export RL_WEIGHT_D1=1 RL_WEIGHT_D2=1 RL_WEIGHT_D3=1
export RL_WEIGHT_EXEC=2 RL_WEIGHT_ERR=1
export RL_MAX_FEATURE_TOKENS=512 RL_MAX_NEW_TOKENS=768
export RL_CHECK_COVERAGE=1
export RL_GEN_TEMPERATURE=0.7 RL_GEN_TOP_P=0.95
export RL_EVAL_LIMIT=0  # 0=full official evaluation; nonzero=debug subsample
export RL_VISUALIZE=1 RL_VIS_EVERY=1
export RL_CODE_SEEDS="Fibonacci with a twist.|Graph shortest paths|Dynamic programming"
export RL_SYNTHGEN_SAMPLES_PER_SEED=1
export RL_WEB_SEARCH=1 RL_WEB_RESULTS=3  # optional DuckDuckGo seed enrichment
export RL_ALLOWED_PIP="numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 sympy==1.13.3 networkx==3.4.2 requests==2.32.3"
export RL_UPLOAD_EVERY=25  # accumulated NEW samples per upload
export RL_HF_REPO_PREFIX="synthgen-continual-code" # random private name per run
export RL_HF_OWNER=""    # empty=logged-in HF username
export RL_HF_TOKEN="${HF_TOKEN:-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}" # login/token via HF_TOKEN, never commit secrets
mkdir -p "$RL_DATA" "$RL_OUT" "$RL_CKPT"
export PYTHONPATH="/root/synth_gen:${PYTHONPATH:-}"
