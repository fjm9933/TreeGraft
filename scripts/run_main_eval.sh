#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MODEL_GROUP="${MODEL_GROUP:-all}"
BENCHES=(${BENCHES:-gsm8k alpaca humaneval qa mt_bench cnndm})
QUESTION_BEGIN="${QUESTION_BEGIN:-0}"
QUESTION_END="${QUESTION_END:-80}"
TOTAL_TOKEN="${TOTAL_TOKEN:-63}"
MAX_NEW_TOKEN="${MAX_NEW_TOKEN:-256}"
NUM_CHOICES="${NUM_CHOICES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:-10}"
DEPTH="${DEPTH:-5}"
LOOKBACK_LAYERS="${LOOKBACK_LAYERS:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/main_eval}"
TREEGRAFT_SCHEDULER="${TREEGRAFT_SCHEDULER:-treegraft_scheduler}"
COMBINE_LAYERS="${COMBINE_LAYERS:-[]}"
SCHEDULER_BUDGET_B0="${SCHEDULER_BUDGET_B0:-5}"

if [[ "${TREEGRAFT_SCHEDULER}" == "treegraft_scheduler" ]]; then
  SCHEDULER_FAMILY="treegraft_scheduler"
  ACTIVE_COMBINE_LAYERS="[]"
elif [[ "${TREEGRAFT_SCHEDULER}" == "none" ]]; then
  SCHEDULER_FAMILY="none"
  ACTIVE_COMBINE_LAYERS="${COMBINE_LAYERS}"
else
  echo "Unsupported TREEGRAFT_SCHEDULER=${TREEGRAFT_SCHEDULER}; use treegraft_scheduler or none." >&2
  exit 1
fi

sanitize_id() {
  local raw="$1"
  printf '%s' "$raw" \
    | tr '[:upper:]' '[:lower:]' \
    | sed 's#[ /]#_#g; s#[^a-z0-9._-]#-#g; s#--*#-#g; s#^[-_.]*##; s#[-_.]*$##'
}

require_model_path() {
  local env_name="$1"
  local value="${!env_name:-}"
  if [[ -z "${value}" ]]; then
    echo "Missing required model path: export ${env_name}=/path/to/checkpoint" >&2
    exit 1
  fi
  printf '%s' "${value}"
}

run_pair() {
  local group="$1"
  local base_model_name="$2"
  local base_model_env="$3"
  local middle_model_name="$4"
  local middle_model_env="$5"
  local conv_template="$6"
  local base_model_path
  local middle_model_path

  base_model_path="$(require_model_path "${base_model_env}")"
  middle_model_path="$(require_model_path "${middle_model_env}")"

  local pair_id
  pair_id="$(sanitize_id "${base_model_name}__${middle_model_name}")"

  echo "========================================"
  echo "TreeGraft main evaluation"
  echo "group=${group}"
  echo "pair=${pair_id}"
  echo "scheduler=${SCHEDULER_FAMILY}"
  echo "combine_layers=${ACTIVE_COMBINE_LAYERS}"
  echo "benchmarks=${BENCHES[*]}"
  echo "output=${OUTPUT_ROOT}/${pair_id}"
  echo "========================================"

  for bench in "${BENCHES[@]}"; do
    local bench_out="${OUTPUT_ROOT}/${pair_id}/${bench}"
    local model_id="${pair_id}-main_eval-${bench}"
    local answer_path="${bench_out}/${model_id}.jsonl"
    local scheduler_path="${bench_out}/${model_id}_scheduler.txt"
    mkdir -p "${bench_out}"

    python -u -m graft.evaluation.gen_combine_tree_answer \
      --base-model-path "${base_model_path}" \
      --middle-model-path "${middle_model_path}" \
      --conv-template "${conv_template}" \
      --bench-name "${bench}" \
      --model-id "${model_id}" \
      --answer-file "${answer_path}" \
      --scheduler-trace-output "${scheduler_path}" \
      --combine-layers "${ACTIVE_COMBINE_LAYERS}" \
      --scheduler-family "${SCHEDULER_FAMILY}" \
      --scheduler-pair-id "${pair_id}" \
      --scheduler-budget-b0 "${SCHEDULER_BUDGET_B0}" \
      --top-k "${TOP_K}" \
      --depth "${DEPTH}" \
      --total-token "${TOTAL_TOKEN}" \
      --max-new-token "${MAX_NEW_TOKEN}" \
      --num-choices "${NUM_CHOICES}" \
      --temperature "${TEMPERATURE}" \
      --question-begin "${QUESTION_BEGIN}" \
      --question-end "${QUESTION_END}" \
      --reselect-frontier-lookback-layers "${LOOKBACK_LAYERS}"
  done
}

run_llama() {
  run_pair "llama" "Llama-3.3-70B-Instruct" "LLAMA_70B_PATH" "Llama-3.2-1B-Instruct" "LLAMA_1B_PATH" "llama-3-chat"
  run_pair "llama" "Llama-3.3-70B-Instruct" "LLAMA_70B_PATH" "Llama-3.2-3B-Instruct" "LLAMA_3B_PATH" "llama-3-chat"
  run_pair "llama" "Llama-3.3-70B-Instruct" "LLAMA_70B_PATH" "Llama-3.1-8B-Instruct" "LLAMA_8B_PATH" "llama-3-chat"
  run_pair "llama" "Llama-3.1-8B-Instruct" "LLAMA_8B_PATH" "Llama-3.2-1B-Instruct" "LLAMA_1B_PATH" "llama-3-chat"
  run_pair "llama" "Llama-3.1-8B-Instruct" "LLAMA_8B_PATH" "Llama-3.2-3B-Instruct" "LLAMA_3B_PATH" "llama-3-chat"
}

run_qwen() {
  run_pair "qwen" "Qwen3-32B" "QWEN_32B_PATH" "Qwen3-0.6B" "QWEN_0P6B_PATH" "qwen3"
  run_pair "qwen" "Qwen3-32B" "QWEN_32B_PATH" "Qwen3-1.7B" "QWEN_1P7B_PATH" "qwen3"
  run_pair "qwen" "Qwen3-32B" "QWEN_32B_PATH" "Qwen3-8B" "QWEN_8B_PATH" "qwen3"
  run_pair "qwen" "Qwen3-32B" "QWEN_32B_PATH" "Qwen3-4B" "QWEN_4B_PATH" "qwen3"
  run_pair "qwen" "Qwen3-8B" "QWEN_8B_PATH" "Qwen3-0.6B" "QWEN_0P6B_PATH" "qwen3"
}

case "${MODEL_GROUP}" in
  llama)
    run_llama
    ;;
  qwen)
    run_qwen
    ;;
  all)
    run_llama
    run_qwen
    ;;
  *)
    echo "Unsupported MODEL_GROUP=${MODEL_GROUP}; use llama, qwen, or all." >&2
    exit 1
    ;;
esac

echo "Done. Results are under ${OUTPUT_ROOT}."
