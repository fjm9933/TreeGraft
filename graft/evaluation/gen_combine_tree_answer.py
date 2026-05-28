"""TreeGraft combine-tree evaluation entrypoint."""

import argparse
import json
import os
from pathlib import Path
import time

import shortuuid
import torch
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

try:
    from ..combine_tree import CombineDraftGenerator, CombinePropagator
except Exception:
    from graft.combine_tree import CombineDraftGenerator, CombinePropagator


def default_system_message(conv_template):
    if conv_template in ["llama-2-chat", "llama-3-chat"]:
        return (
            "You are a helpful, respectful and honest assistant. Always answer "
            "as helpfully as possible, while being safe. Your answers should "
            "not include harmful, unethical, racist, sexist, toxic, dangerous, "
            "or illegal content. If a question does not make sense, explain "
            "why instead of answering incorrectly."
        )
    if conv_template == "vicuna":
        return "You are a helpful, respectful and honest assistant."
    return "You are a helpful assistant."


def build_prompt(tokenizer, messages, conv_template):
    if conv_template in ["llama-3-chat", "mixtral"]:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, None

    conv = get_conversation_template(conv_template)
    if messages and messages[0]["role"] == "system":
        conv.system_message = messages[0]["content"]
        user_messages = messages[1:]
    else:
        user_messages = messages

    for msg in user_messages:
        if msg["role"] == "user":
            conv.append_message(conv.roles[0], msg["content"])
        elif msg["role"] == "assistant":
            conv.append_message(conv.roles[1], msg["content"])
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    if conv_template == "llama-2-chat":
        prompt += " "
    return prompt, conv


def decode_output(tokenizer, conv, output_ids):
    if conv is not None and conv.stop_token_ids:
        stop_positions = [
            i for i, tok_id in enumerate(output_ids)
            if tok_id in conv.stop_token_ids
        ]
        if stop_positions:
            output_ids = output_ids[:stop_positions[0]]

    output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    if conv is not None and conv.stop_str and output.find(conv.stop_str) > 0:
        output = output[:output.find(conv.stop_str)]

    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for item in special_token:
                output = output.replace(item, "")
        elif special_token is not None:
            output = output.replace(special_token, "")
    return output.strip()


def parse_layers(value):
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("--combine-layers must be a JSON integer array") from exc
    if not isinstance(parsed, list):
        raise ValueError("--combine-layers must be a JSON integer array")

    layers = []
    for item in parsed:
        layer = int(item)
        if layer <= 0:
            raise ValueError("--combine-layers only accepts positive fixed layer ids")
        layers.append(layer)
    return layers


def mean_elapsed_ms(records):
    values = []
    for record in list(records or []):
        elapsed = float(record.get("elapsed_ms", 0.0) or 0.0)
        if elapsed > 0.0:
            values.append(elapsed)
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def aggregate_layer_elapsed_records(records, depth, elapsed_key, count_key):
    total_ms = [0.0] * int(max(0, depth))
    call_count = [0] * int(max(0, depth))

    for record in list(records or []):
        try:
            layer_idx = int(record.get("layer_idx", 0))
        except Exception:
            continue
        if layer_idx <= 0 or layer_idx > int(depth):
            continue

        slot = int(layer_idx) - 1
        total_ms[slot] += float(record.get(elapsed_key, 0.0) or 0.0)
        call_count[slot] += int(record.get(count_key, 0) or 0)

    avg_ms = [
        (float(total_ms[idx]) / float(call_count[idx]))
        if call_count[idx] > 0 else 0.0
        for idx in range(len(total_ms))
    ]
    return {
        "call_count": [int(x) for x in call_count],
        "total_ms": [float(x) for x in total_ms],
        "avg_ms": [float(x) for x in avg_ms],
    }


def build_pair_id(args):
    pair_id = str(args.scheduler_pair_id or "").strip()
    if pair_id:
        return pair_id
    base_name = Path(str(args.base_model_path)).name.strip().lower()
    middle_name = Path(str(args.middle_model_path)).name.strip().lower()
    return f"{base_name}__{middle_name}" if middle_name else base_name


def scheduler_state(draft_generator):
    scheduler = getattr(draft_generator, "scheduler", None)
    if scheduler is None:
        return None
    return {
        "family": getattr(scheduler, "family", None),
        "enabled": bool(getattr(scheduler, "enabled", False)),
        "online_active": bool(getattr(scheduler, "online_active", False)),
        "budget_b0": getattr(scheduler, "budget_b0", None),
        "t_tgt_ms": getattr(scheduler, "T_tgt", None),
        "t_mid_ms": getattr(scheduler, "T_mid", None),
        "prompt_len_tokens": getattr(scheduler, "prompt_len_tokens", None),
    }


def build_runtime(args, pair_id):
    draft_generator = CombineDraftGenerator(
        draft_strategy="combine",
        scheduler_family=args.scheduler_family,
        scheduler_budget_b0=args.scheduler_budget_b0,
        scheduler_pair_id=pair_id,
        scheduler_task_id=args.bench_name,
        total_tokens=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        max_matching_ngram_size=args.max_matching_ngram_size,
        stop_on_relax_zero=True,
        legacy_total_token_stop=False,
        reselect_frontier_by_middle_score=True,
        reselect_frontier_lookback_layers=args.reselect_frontier_lookback_layers,
        combine_layers=args.combine_layers,
        middle_model_path=args.middle_model_path,
        middle_max_length=args.max_length,
        device_map="auto",
        dtype=torch.float16,
    )
    propagator = CombinePropagator(
        base_model_path=args.base_model_path,
        draft_generator=draft_generator,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        update_prefill_with_accepted=False,
        device_map="auto",
        dtype=torch.float16,
    )
    return propagator, draft_generator


def seed_messages(conv_template):
    return [
        {"role": "system", "content": default_system_message(conv_template)}
    ]


def generate_turn(
    propagator,
    draft_generator,
    tokenizer,
    conv_template,
    messages,
    args,
    pair_id,
    collect_timing=False,
):
    prompt, conv = build_prompt(tokenizer, messages, conv_template)
    add_special_tokens = conv_template not in ["llama-3-chat", "mixtral"]
    input_ids = tokenizer(
        [prompt],
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    ).input_ids
    input_len = input_ids.shape[1]

    scheduler = getattr(draft_generator, "scheduler", None)
    if scheduler is not None and getattr(scheduler, "enabled", False):
        draft_generator.set_scheduler_prompt_len_tokens(int(input_len))
        draft_generator.set_scheduler_pair_id(pair_id)
        draft_generator.set_scheduler_task_id(args.bench_name)

    device = next(propagator.base_model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.time()
    output_ids, new_token, step_idx = propagator.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=float(args.temperature),
        max_new_tokens=args.max_new_token,
        is_llama3=(conv_template == "llama-3-chat"),
        log=True,
        collect_time_stats=bool(collect_timing),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    output_ids = output_ids[0][input_len:]
    output = decode_output(tokenizer, conv, output_ids)
    return output, int(new_token), int(step_idx + 1), float(time.time() - start_time)


def warmup_scheduler(propagator, draft_generator, tokenizer, questions, args, pair_id):
    scheduler = getattr(draft_generator, "scheduler", None)
    if scheduler is None or not getattr(scheduler, "needs_warmup_rounds", False):
        return

    forced_layers = list(range(1, int(args.depth) + 1))
    draft_generator.set_scheduler_force_warmup_mode(True)
    draft_generator.set_scheduler_force_warmup_collection(
        target_timing=True,
        middle_timing=True,
    )
    draft_generator.set_scheduler_forced_layers(forced_layers)

    last_target_ms = None
    last_middle_ms = None
    try:
        warmup_question = questions[0]
        for _ in range(3):
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            messages = seed_messages(args.conv_template)
            target_records = []
            middle_records = []
            for turn in warmup_question["turns"]:
                messages.append({"role": "user", "content": turn})
                output, _, _, _ = generate_turn(
                    propagator,
                    draft_generator,
                    tokenizer,
                    args.conv_template,
                    messages,
                    args,
                    pair_id,
                    collect_timing=False,
                )
                messages.append({"role": "assistant", "content": output})
                target_records.extend(
                    getattr(propagator, "last_target_forward_records", [])
                )
                middle_records.extend(
                    getattr(propagator, "last_middle_forward_records", [])
                )
            last_target_ms = mean_elapsed_ms(target_records)
            last_middle_ms = mean_elapsed_ms(middle_records)
    finally:
        draft_generator.set_scheduler_force_warmup_mode(False)
        draft_generator.set_scheduler_force_warmup_collection(
            target_timing=False,
            middle_timing=False,
        )
        draft_generator.set_scheduler_forced_layers(None)

    draft_generator.finalize_scheduler_after_warmup(
        A_t=None,
        T_tgt=last_target_ms,
        T_mid=last_middle_ms,
    )
    scheduler = draft_generator.scheduler
    print(
        f"[scheduler:{scheduler.family}] "
        f"T_tgt={float(scheduler.T_tgt):.4f} ms | "
        f"T_mid={float(scheduler.T_mid):.4f} ms | "
        f"B0={scheduler.budget_b0}"
    )


def trace_output_path(args, answer_file):
    if args.scheduler_trace_output:
        return os.path.expanduser(args.scheduler_trace_output)
    return None


def write_scheduler_trace_header(fout, args, model_id, pair_id, draft_generator):
    scheduler = getattr(draft_generator, "scheduler", None)
    state = scheduler_state(draft_generator) or {}
    fout.write("TreeGraft Scheduler Trace\n")
    fout.write("=" * 80 + "\n")
    fout.write(f"model_id={model_id}\n")
    fout.write(f"bench_name={args.bench_name}\n")
    fout.write(f"pair_id={pair_id}\n")
    fout.write(f"depth={args.depth}\n")
    fout.write(f"top_k={args.top_k}\n")
    fout.write(f"total_token={args.total_token}\n")
    fout.write(f"combine_layers={json.dumps(args.combine_layers)}\n")
    fout.write(f"scheduler_family={getattr(scheduler, 'family', args.scheduler_family)}\n")
    fout.write(f"scheduler_budget_b0={state.get('budget_b0')}\n")
    fout.write(f"scheduler_t_tgt_ms={state.get('t_tgt_ms')}\n")
    fout.write(f"scheduler_t_mid_ms={state.get('t_mid_ms')}\n")
    fout.write("=" * 80 + "\n")


def compact_layer_record(record):
    keys = [
        "layer_idx",
        "action",
        "scheduler_call_count",
        "scheduler_elapsed_ms",
        "ngram_call_count",
        "ngram_elapsed_ms",
        "middle_call_count",
        "middle_elapsed_ms",
        "frontier_size_before",
        "frontier_size_after",
        "tree_nodes_before",
        "tree_node_count_after",
        "pre_reselect_candidate_size",
        "pre_reselect_depth_delta",
        "pre_reselect_score_delta",
        "collapse_debt",
    ]
    return {key: record.get(key) for key in keys if key in record}


def write_scheduler_choice_trace(
    fout,
    question_id,
    choice_index,
    choice,
    turn_layer_records,
    state_before,
    state_after,
):
    trace_record = {
        "type": "choice",
        "question_id": question_id,
        "choice_index": int(choice_index),
        "total_steps": int(choice["total_steps"]),
        "total_new_tokens": int(sum(choice["new_tokens"])),
        "avg_accept_len": float(choice["avg_accept_len"]),
        "scheduler_state_before": state_before,
        "scheduler_state_after": state_after,
        "sample_scheduler_layer_call_count": choice[
            "sample_scheduler_layer_call_count"
        ],
        "sample_scheduler_layer_total_ms": choice[
            "sample_scheduler_layer_total_ms"
        ],
        "sample_ngram_layer_call_count": choice["sample_ngram_layer_call_count"],
        "sample_ngram_layer_total_ms": choice["sample_ngram_layer_total_ms"],
        "turn_layers": [
            [compact_layer_record(record) for record in layer_records]
            for layer_records in turn_layer_records
        ],
    }
    fout.write(json.dumps(trace_record, ensure_ascii=True) + "\n")
    fout.flush()


def run_eval(args):
    questions = load_questions(
        args.question_file,
        args.question_begin,
        args.question_end,
    )
    if not questions:
        raise ValueError("No questions were loaded for evaluation")

    pair_id = build_pair_id(args)
    propagator, draft_generator = build_runtime(args, pair_id)
    tokenizer = propagator.tokenizer

    print("Check model eval state:", not propagator.base_model.training)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    warmup_scheduler(propagator, draft_generator, tokenizer, questions, args, pair_id)

    answer_file = os.path.expanduser(args.answer_file)
    answer_dir = os.path.dirname(answer_file)
    if answer_dir:
        os.makedirs(answer_dir, exist_ok=True)

    trace_fout = None
    trace_path = trace_output_path(args, answer_file)
    scheduler = getattr(draft_generator, "scheduler", None)
    scheduler_enabled = bool(scheduler is not None and getattr(scheduler, "enabled", False))
    if trace_path and scheduler_enabled:
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        trace_fout = open(trace_path, "w", encoding="utf-8")
        write_scheduler_trace_header(
            trace_fout,
            args,
            args.model_id,
            pair_id,
            draft_generator,
        )
        print(f"Scheduler trace -> {trace_path}")

    try:
        for question in tqdm(questions):
            choices = []
            for choice_index in range(int(args.num_choices)):
                torch.manual_seed(choice_index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(choice_index)

                messages = seed_messages(args.conv_template)
                turns = []
                new_tokens = []
                wall_time = []
                steps = []
                correction_layer_freq = [0] * int(max(0, args.depth))
                sample_actual_layer_records = []
                turn_layer_records = []
                state_before = scheduler_state(draft_generator)

                for user_turn in question["turns"]:
                    messages.append({"role": "user", "content": user_turn})
                    output, new_token, step_count, elapsed = generate_turn(
                        propagator,
                        draft_generator,
                        tokenizer,
                        args.conv_template,
                        messages,
                        args,
                        pair_id,
                        collect_timing=(trace_fout is not None),
                    )
                    turns.append(output)
                    new_tokens.append(new_token)
                    wall_time.append(elapsed)
                    steps.append(step_count)

                    turn_freq = list(
                        getattr(propagator, "last_sample_correction_layer_freq", [])
                    )
                    for layer_idx, count in enumerate(
                        turn_freq[:len(correction_layer_freq)]
                    ):
                        correction_layer_freq[layer_idx] += int(count)

                    layer_records = [
                        dict(record)
                        for record in getattr(
                            propagator,
                            "last_actual_layer_records",
                            [],
                        )
                    ]
                    turn_layer_records.append(layer_records)
                    sample_actual_layer_records.extend(layer_records)
                    messages.append({"role": "assistant", "content": output})

                total_steps = sum(steps)
                total_new = sum(new_tokens)
                scheduler_layer_stats = aggregate_layer_elapsed_records(
                    sample_actual_layer_records,
                    args.depth,
                    elapsed_key="scheduler_elapsed_ms",
                    count_key="scheduler_call_count",
                )
                ngram_layer_stats = aggregate_layer_elapsed_records(
                    sample_actual_layer_records,
                    args.depth,
                    elapsed_key="ngram_elapsed_ms",
                    count_key="ngram_call_count",
                )

                choice = {
                    "index": choice_index,
                    "turns": turns,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "total_steps": total_steps,
                    "avg_accept_len": (
                        float(total_new) / float(total_steps)
                        if total_steps > 0 else 0.0
                    ),
                    "middle_correction_layer_freq": correction_layer_freq,
                    "sample_scheduler_layer_call_count": scheduler_layer_stats[
                        "call_count"
                    ],
                    "sample_scheduler_layer_total_ms": scheduler_layer_stats[
                        "total_ms"
                    ],
                    "sample_scheduler_layer_avg_ms": scheduler_layer_stats[
                        "avg_ms"
                    ],
                    "sample_scheduler_total_ms": float(
                        sum(scheduler_layer_stats["total_ms"])
                    ),
                    "sample_ngram_layer_call_count": ngram_layer_stats[
                        "call_count"
                    ],
                    "sample_ngram_layer_total_ms": ngram_layer_stats["total_ms"],
                    "sample_ngram_layer_avg_ms": ngram_layer_stats["avg_ms"],
                    "sample_ngram_total_ms": float(
                        sum(ngram_layer_stats["total_ms"])
                    ),
                }
                choices.append(choice)

                if trace_fout is not None:
                    write_scheduler_choice_trace(
                        trace_fout,
                        question_id=question["question_id"],
                        choice_index=choice_index,
                        choice=choice,
                        turn_layer_records=turn_layer_records,
                        state_before=state_before,
                        state_after=scheduler_state(draft_generator),
                    )

            answer = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": args.model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            with open(answer_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(answer, ensure_ascii=True) + "\n")
    finally:
        if trace_fout is not None:
            trace_fout.close()


def reorg_answer_file(answer_file):
    answers = {}
    answer_file = os.path.expanduser(answer_file)
    with open(answer_file, "r", encoding="utf-8") as fin:
        for line in fin:
            answers[json.loads(line)["question_id"]] = line
    with open(answer_file, "w", encoding="utf-8") as fout:
        for question_id in sorted(answers.keys()):
            fout.write(answers[question_id])


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run TreeGraft evaluation.")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--middle-model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument(
        "--conv-template",
        default="vicuna",
        choices=[
            "vicuna",
            "llama-2-chat",
            "llama-3-chat",
            "qwen",
            "qwen3",
            "mixtral",
        ],
    )
    parser.add_argument("--model-id", default="treegraft")
    parser.add_argument("--bench-name", default="mt_bench")
    parser.add_argument("--question-file", default=None)
    parser.add_argument("--question-begin", type=int, default=None)
    parser.add_argument("--question-end", type=int, default=None)
    parser.add_argument("--answer-file", default=None)
    parser.add_argument("--scheduler-trace-output", default=None)
    parser.add_argument("--max-new-token", type=int, default=512)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2200)
    parser.add_argument("--combine-layers", default="[]")
    parser.add_argument(
        "--scheduler-family",
        default="none",
        choices=["none", "treegraft_scheduler"],
    )
    parser.add_argument("--scheduler-pair-id", default="")
    parser.add_argument("--scheduler-budget-b0", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--total-token", type=int, default=63)
    parser.add_argument("--max-matching-ngram-size", type=int, default=5)
    parser.add_argument("--reselect-frontier-lookback-layers", type=int, default=0)
    return parser


def validate_args(args):
    args.combine_layers = parse_layers(args.combine_layers)
    numeric_fields = [
        "max_new_token",
        "max_length",
        "num_choices",
        "top_k",
        "depth",
        "total_token",
        "max_matching_ngram_size",
    ]
    for field in numeric_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be > 0")
    if float(args.temperature) < 0.0:
        raise ValueError("--temperature must be >= 0")
    if args.reselect_frontier_lookback_layers < 0:
        raise ValueError("--reselect-frontier-lookback-layers must be >= 0")
    if args.scheduler_budget_b0 is not None and args.scheduler_budget_b0 < 0:
        raise ValueError("--scheduler-budget-b0 must be >= 0")

    root = Path(__file__).resolve().parents[1]
    if args.question_file is None:
        args.question_file = str(root / "data" / args.bench_name / "question.jsonl")
    if args.answer_file is None:
        args.answer_file = f"{args.bench_name}/{args.model_id}.jsonl"
    return args


def main():
    args = validate_args(build_arg_parser().parse_args())
    for key, value in vars(args).items():
        print(f"{key}={value}")
    print(f"Output to {args.answer_file}")
    run_eval(args)
    reorg_answer_file(args.answer_file)


if __name__ == "__main__":
    main()
