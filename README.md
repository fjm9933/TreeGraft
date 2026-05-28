# TreeGraft

This repository contains the evaluation code for TreeGraft, an adaptive multi-drafter grafting method for tree-based speculative decoding.

The released package includes the TreeGraft shared-tree runtime, the value-guided online scheduler, benchmark question files, and a single script for running the main evaluation preset.

## Package Layout

```text
package root/
  README.md
  requirements.txt
  graft/
    combine_tree/          # TreeGraft tree construction and scheduler runtime
    ngram_tree/            # N-gram small drafter
    model/                 # KV-cache model wrappers
    evaluation/            # Evaluation entrypoint
    data/                  # Benchmark question files
  scripts/
    run_main_eval.sh       # Single evaluation entrypoint
  outputs/
```

## Requirements

Install dependencies in a Python environment with CUDA-enabled PyTorch:

```bash
pip install -r requirements.txt
```

The code expects local access to the evaluated model checkpoints.

## Model Paths

Set model checkpoint paths with environment variables:

```bash
export LLAMA_70B_PATH=/path/to/Llama-3.3-70B-Instruct
export LLAMA_8B_PATH=/path/to/Llama-3.1-8B-Instruct
export LLAMA_3B_PATH=/path/to/Llama-3.2-3B-Instruct
export LLAMA_1B_PATH=/path/to/Llama-3.2-1B-Instruct

export QWEN_32B_PATH=/path/to/Qwen3-32B
export QWEN_8B_PATH=/path/to/Qwen3-8B
export QWEN_4B_PATH=/path/to/Qwen3-4B
export QWEN_1P7B_PATH=/path/to/Qwen3-1.7B
export QWEN_0P6B_PATH=/path/to/Qwen3-0.6B
```

Only the paths required by the selected model group need to be set.
Llama checkpoints should use the Instruct variants; Qwen checkpoints use the base Qwen3 names shown above.

## Quick Smoke Test

Run a small subset first:

```bash
QUESTION_END=2 BENCHES="gsm8k" MODEL_GROUP=qwen bash scripts/run_main_eval.sh
```

This evaluates two GSM8K examples for the Qwen model group.

## Main Evaluation

Run all configured model pairs and benchmarks:

```bash
MODEL_GROUP=all bash scripts/run_main_eval.sh
```

Available `MODEL_GROUP` values:

```text
llama
qwen
all
```

By default, the script evaluates the six benchmarks used in the paper:

```text
gsm8k alpaca humaneval qa mt_bench cnndm
```

You can override the benchmark list:

```bash
BENCHES="gsm8k cnndm" MODEL_GROUP=llama bash scripts/run_main_eval.sh
```

## Common Options

The script is configured through environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_GROUP` | `all` | Which model family to run: `llama`, `qwen`, or `all` |
| `BENCHES` | `gsm8k alpaca humaneval qa mt_bench cnndm` | Benchmarks to evaluate |
| `QUESTION_BEGIN` | `0` | First example index |
| `QUESTION_END` | `80` | End example index |
| `OUTPUT_ROOT` | `outputs/main_eval` | Output directory |
| `MAX_NEW_TOKEN` | `256` | Maximum generated tokens |
| `NUM_CHOICES` | `1` | Number of choices per example |
| `TEMPERATURE` | `0.0` | Decoding temperature |
| `TOTAL_TOKEN` | `63` | Verification tree budget |
| `TOP_K` | `10` | Expansion frontier size |
| `DEPTH` | `5` | Maximum drafting depth |
| `LOOKBACK_LAYERS` | `5` | Frontier reselection lookback |
| `TREEGRAFT_SCHEDULER` | `treegraft_scheduler` | Scheduler mode: `treegraft_scheduler` or `none` |
| `SCHEDULER_BUDGET_B0` | `5` | Maximum middle-drafter calls per decoding step when the scheduler is enabled |
| `COMBINE_LAYERS` | `[]` | Fixed middle-drafter schedule when the scheduler is disabled |

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 QUESTION_BEGIN=0 QUESTION_END=80 MODEL_GROUP=all \
  bash scripts/run_main_eval.sh
```

## Outputs

Results are written under:

```text
outputs/main_eval/
```

Each benchmark run produces:

```text
<model_pair>/<benchmark>/<model_id>.jsonl
<model_pair>/<benchmark>/<model_id>_scheduler.txt
```

The JSONL file contains generated answers and runtime statistics. The scheduler file records scheduler decisions and timing information.

## Scheduler

The default evaluation uses TreeGraft's value-guided online scheduler. At each drafting step, the scheduler decides whether to call the middle drafter or continue with the low-cost n-gram drafter.
