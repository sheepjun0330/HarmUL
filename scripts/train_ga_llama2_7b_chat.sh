#!/usr/bin/env bash
set -euo pipefail

# Full training script for GradAscent on the local llama2 harmbench JSON dataset.
# Uses 2 GPUs + ZeRO-3 by default. GA fits more reliably with paged_adamw_32bit.

MODEL="${MODEL:-Llama-2-7b-chat-hf}"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-chat-hf}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${HF_HOME:-$HOME/.cache/huggingface}/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots}"
if [[ -n "${MODEL_PATH:-}" ]]; then
  RESOLVED_MODEL_PATH="${MODEL_PATH}"
elif [[ -d "${SNAPSHOT_DIR}" ]] && first_snapshot="$(find "${SNAPSHOT_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)" && [[ -n "${first_snapshot}" ]]; then
  RESOLVED_MODEL_PATH="${first_snapshot}"
else
  RESOLVED_MODEL_PATH="${MODEL_ID}"
fi

DATA_DIR="${DATA_DIR:-data/unlearn/llama2_7b_chat_harmbench}"
TASK_PREFIX="${TASK_PREFIX:-json_llama2_7b_chat}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/default_config.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
OPTIM="${OPTIM:-paged_adamw_32bit}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-0}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"
POST_EVAL="${POST_EVAL:-0}"
POST_EVAL_EXPERIMENT="${POST_EVAL_EXPERIMENT:-configs/experiment/eval/json_unlearn/llama2_7b_chat_harmbench.yaml}"
POST_EVAL_MODEL_PATH_TEMPLATE="${POST_EVAL_MODEL_PATH_TEMPLATE:-saves/unlearn/{task_name}}"
WANDB="${WANDB:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-sheepjun}"
WANDB_PROJECT="${WANDB_PROJECT:-HarmUL}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cmd=(
  uv run python src/unlearn.py
  --methods GA
  --data-dir "${DATA_DIR}"
  --model "${MODEL}"
  --model-path "${RESOLVED_MODEL_PATH}"
  --task-prefix "${TASK_PREFIX}"
  --accelerate-config "${ACCELERATE_CONFIG}"
  --cuda-visible-devices "${CUDA_DEVICES}"
  trainer.args.optim="${OPTIM}"
)

if [[ "${ENABLE_EVAL}" == "1" ]]; then
  cmd+=(--enable-eval)
fi
if [[ "${POST_EVAL}" == "1" ]]; then
  cmd+=(--post-eval --post-eval-experiment "${POST_EVAL_EXPERIMENT}")
  if [[ -n "${POST_EVAL_MODEL_PATH_TEMPLATE}" ]]; then
    cmd+=(--post-eval-model-path-template "${POST_EVAL_MODEL_PATH_TEMPLATE}")
  fi
fi
if [[ "${WANDB}" == "1" ]]; then
  cmd+=(--wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  cmd+=(--num-train-epochs "${NUM_TRAIN_EPOCHS}")
fi
if [[ -n "${LEARNING_RATE}" ]]; then
  cmd+=(--learning-rate "${LEARNING_RATE}")
fi
cmd+=(--per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}")
cmd+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
cmd+=(trainer.args.save_strategy="${SAVE_STRATEGY}")
if [[ "${SAVE_STRATEGY}" == "steps" && -n "${SAVE_STEPS}" ]]; then
  cmd+=(trainer.args.save_steps="${SAVE_STEPS}")
fi
if [[ "${SAVE_STRATEGY}" != "no" && -n "${SAVE_TOTAL_LIMIT}" ]]; then
  cmd+=(trainer.args.save_total_limit="${SAVE_TOTAL_LIMIT}")
fi
if [[ "${SAVE_ONLY_MODEL}" == "1" ]]; then
  cmd+=(trainer.args.save_only_model=True)
else
  cmd+=(trainer.args.save_only_model=False)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  cmd+=(trainer.args.gradient_checkpointing=True)
fi

cmd+=("$@")

exec env HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" "${cmd[@]}"
