#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 기존 Podman / synth_gen 환경 설정
source /root/synth_gen/custome/config.sh

# Continual learning 전용 설정
source "$ROOT/rl_config.sh"

# 기존 synth_gen 기본 LLM 선택 비활성화
unset LLM

# 원격 Podman 활성화
export SYNTHGEN_USE_REMOTE_PODMAN=1

if [[ -f /root/.synthgen_remote_token ]]; then
    export SYNTHGEN_REMOTE_TOKEN="$(
        cat /root/.synthgen_remote_token
    )"
fi