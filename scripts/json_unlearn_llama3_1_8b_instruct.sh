#!/usr/bin/env bash
set -euo pipefail

# Run unlearning methods on local JSONL datasets for Llama-3.1-8B-Instruct.
# You can override defaults with env vars, and pass extra Hydra overrides as args.
#
# Examples:
#   uv run bash scripts/json_unlearn_llama3_1_8b_instruct.sh
#   METHODS="GA,GD,NPO" uv run bash scripts/json_unlearn_llama3_1_8b_instruct.sh --dry-run

MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
METHODS="${METHODS:-GradAscent,GradDiff,NPO,SimNPO,RMU,DPO}"
DATA_DIR="${DATA_DIR:-data/unlearn/llama3_1_8b_instruct_jailbreak_log}"
TASK_PREFIX="${TASK_PREFIX:-json_llama3_1_8b_instruct}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate/default_config.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

# Optional knobs (leave empty to use trainer/config defaults)
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"
POST_EVAL="${POST_EVAL:-0}"
POST_EVAL_EXPERIMENT="${POST_EVAL_EXPERIMENT:-}"
POST_EVAL_MODEL_PATH_TEMPLATE="${POST_EVAL_MODEL_PATH_TEMPLATE:-saves/unlearn/{task_name}}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  _python="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  _python="python"
elif command -v python3 >/dev/null 2>&1; then
  _python="python3"
else
  echo "Error: neither 'python' nor 'python3' was found in PATH." >&2
  exit 127
fi

cmd=(
  "${_python}" src/unlearn.py
  --methods "${METHODS}"
  --data-dir "${DATA_DIR}"
  --model "${MODEL}"
  --model-path "${MODEL_PATH}"
  --task-prefix "${TASK_PREFIX}"
  --accelerate-config "${ACCELERATE_CONFIG}"
  --cuda-visible-devices "${CUDA_DEVICES}"
)

if [[ "${ENABLE_EVAL}" == "1" ]]; then
  cmd+=(--enable-eval)
fi
if [[ "${POST_EVAL}" == "1" ]]; then
  cmd+=(--post-eval)
  if [[ -n "${POST_EVAL_EXPERIMENT}" ]]; then
    cmd+=(--post-eval-experiment "${POST_EVAL_EXPERIMENT}")
  fi
  if [[ -n "${POST_EVAL_MODEL_PATH_TEMPLATE}" ]]; then
    cmd+=(--post-eval-model-path-template "${POST_EVAL_MODEL_PATH_TEMPLATE}")
  fi
fi
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  cmd+=(--num-train-epochs "${NUM_TRAIN_EPOCHS}")
fi
if [[ -n "${LEARNING_RATE}" ]]; then
  cmd+=(--learning-rate "${LEARNING_RATE}")
fi
if [[ -n "${PER_DEVICE_TRAIN_BATCH_SIZE}" ]]; then
  cmd+=(--per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}")
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  cmd+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
fi

# Pass through any extra CLI args (e.g., --dry-run or custom Hydra overrides)
cmd+=("$@")

echo "Running JSON unlearning methods: ${METHODS}"
echo "Model config: ${MODEL}"
echo "Data dir: ${DATA_DIR}"
if [[ "${POST_EVAL}" == "1" ]]; then
  echo "Post-eval: enabled (${POST_EVAL_EXPERIMENT:-default eval.yaml})"
fi

exec "${cmd[@]}"
