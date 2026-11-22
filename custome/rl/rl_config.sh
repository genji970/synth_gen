#!/usr/bin/env bash
# Runtime-derived paths only. ALL model, task, steps, GPUs, synth, and HF
# settings come from custome/config.sh. NO duplicate training defaults here.
_rl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_rl_top="$(cd "${_rl_dir}/.." && pwd)"
if [[ -z "${RL_BASE_MODEL:-}" || -z "${RL_SFT_TASKS:-}" || -z "${RL_GRPO_STEPS_OURS:-}" ]]; then
  echo "[config error] First run: source custome/config.sh (after applying config.sh.rl_additions)." >&2
  return 1 2>/dev/null || exit 1
fi
export RL_ROOT="${_rl_dir}"
export RL_DATA="${_rl_dir}/data"
export RL_CKPT="${_rl_dir}/checkpoints/${RL_RUN_ID:?Set RL_RUN_ID in custome/config.sh}"
export RL_OUT="${_rl_dir}/output/${RL_RUN_ID}"
# Derived aliases; NEVER edit/override the canonical model in a second file.
export RL_MODEL="${RL_BASE_MODEL}"
export RL_FEATURE_MODEL="${RL_CKPT}/hf_after_code"
export RL_SERVICE_URL="http://${RL_SERVICE_BIND:?}:${RL_SERVICE_PORT:?}"
export PYTHONPATH="$(cd "${_rl_top}/.." && pwd):${PYTHONPATH:-}"
unset _rl_dir _rl_top
mkdir -p "$RL_ROOT" "$RL_DATA" "$RL_CKPT" "$RL_OUT"
