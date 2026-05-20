#!/usr/bin/env bash
# Shared environment defaults for local and RunPod execution.
#
# RunPod mounts persistent network storage at /workspace. Anything cached under
# /root or /tmp may disappear when the Pod is terminated, so every script sources
# this file before importing Python libraries. Operators can still override any
# path by exporting the variable before running a script.

env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
env_v2_root="$(cd "${env_script_dir}/.." && pwd -P)"

if [[ -n "${RUNPOD_PERSISTENT_ROOT:-}" ]]; then
    env_persistent_root="$RUNPOD_PERSISTENT_ROOT"
elif [[ -d "/workspace" && -w "/workspace" ]]; then
    env_persistent_root="/workspace"
else
    env_persistent_root="$env_v2_root"
fi

export HATESPEECH_CACHE_ROOT="${HATESPEECH_CACHE_ROOT:-${env_persistent_root}/.cache/hatespeech-v2}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HATESPEECH_CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${HATESPEECH_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${HATESPEECH_CACHE_ROOT}/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${HATESPEECH_CACHE_ROOT}/matplotlib}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HATESPEECH_CACHE_ROOT}/pip}"
export WANDB_DIR="${WANDB_DIR:-${HATESPEECH_CACHE_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${HATESPEECH_CACHE_ROOT}/wandb-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
    "$HATESPEECH_CACHE_ROOT" \
    "$XDG_CACHE_HOME" \
    "$HF_HOME" \
    "$HF_HUB_CACHE" \
    "$TRANSFORMERS_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$TORCH_HOME" \
    "$MPLCONFIGDIR" \
    "$PIP_CACHE_DIR" \
    "$WANDB_DIR" \
    "$WANDB_CACHE_DIR"

unset env_script_dir env_v2_root env_persistent_root
