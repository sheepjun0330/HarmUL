#!/usr/bin/env bash
set -euo pipefail

# Build a masked unlearning dataset from a local jailbreak JSON file.
# Defaults target the current Llama-2 Harmbench-style raw file.
#
# Examples:
#   uv run bash scripts/preprocess_masked_jailbreak_data.sh
#   INPUTS="data/raw/jailbreak_log_meta-llama_Llama-3_1-8B-Instruct.json" \
#     OUT_DIR="data/unlearn/llama3_1_8b_instruct_jailbreak_log_masked" \
#     uv run bash scripts/preprocess_masked_jailbreak_data.sh

INPUTS="${INPUTS:-data/raw/Jailbreak-R1-attack_llama2_7b_chat_Harmbench.json}"
OUT_DIR="${OUT_DIR:-data/unlearn/llama2_7b_chat_harmbench_masked}"
MASK_FORGET_FIELDS="${MASK_FORGET_FIELDS:-answer}"
MASK_TOKEN="${MASK_TOKEN:-[MASKED_HARMFUL_CONTENT]}"
EVAL_RATIO="${EVAL_RATIO:-0.2}"
SEED="${SEED:-42}"
MAX_FORGET="${MAX_FORGET:-}"
MAX_RETAIN="${MAX_RETAIN:-}"
REFUSAL_TEXT="${REFUSAL_TEXT:-}"

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

read -r -a input_args <<< "${INPUTS}"

cmd=(
  "${_python}" data/preprocess.py
  --inputs "${input_args[@]}"
  --out_dir "${OUT_DIR}"
  --eval_ratio "${EVAL_RATIO}"
  --seed "${SEED}"
  --mask_forget_fields "${MASK_FORGET_FIELDS}"
  --mask_token "${MASK_TOKEN}"
)

if [[ -n "${MAX_FORGET}" ]]; then
  cmd+=(--max_forget "${MAX_FORGET}")
fi
if [[ -n "${MAX_RETAIN}" ]]; then
  cmd+=(--max_retain "${MAX_RETAIN}")
fi
if [[ -n "${REFUSAL_TEXT}" ]]; then
  cmd+=(--refusal_text "${REFUSAL_TEXT}")
fi

cmd+=("$@")

echo "Building masked jailbreak dataset"
echo "Inputs: ${INPUTS}"
echo "Out dir: ${OUT_DIR}"
echo "Masked forget fields: ${MASK_FORGET_FIELDS}"

exec "${cmd[@]}"
