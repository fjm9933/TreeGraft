"""N-gram small-drafter propagation runtime."""
import torch
import time
from transformers import AutoTokenizer

from graft.model.utils import (
    reset_tree_mode,
    evaluate_posterior,
    prepare_logits_processor,
)
from graft.model.kv_cache import initialize_past_key_values


class NgramPropagator:
    """Propagation loop backed by an n-gram draft generator."""

    def __init__(
        self,
        base_model_path: str,
        draft_generator,
        tokenizer_path: str = None,
        max_length: int = 2048,
        update_prefill_with_accepted: bool = False,
        device_map: str = "auto",
        dtype=torch.float16,
    ):
        """Create a propagation loop for the target model and n-gram drafter."""

        from graft.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
        from graft.model.modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
        from graft.model.modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
        from graft.model.modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
        from transformers import AutoConfig

        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]

        if Type == 'LlamaForCausalLM':
            self.base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        elif Type == 'Qwen2ForCausalLM':
            self.base_model = KVQwen2ForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        elif Type == 'Qwen3ForCausalLM':
            self.base_model = KVQwen3ForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        else:
            self.base_model = KVMixtralForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )

        self.base_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or base_model_path,
            use_fast=False
        )

        self.draft_generator = draft_generator
        self.max_length = max_length
        self.update_prefill_with_accepted = update_prefill_with_accepted

        self.past_key_values = None
        self.past_key_values_data = None
        self.current_length_data = None

        self.time_stats = self._create_empty_time_stats()
        self.last_sample_correction_layer_freq = []
        self.last_middle_forward_records = []
        self.last_target_forward_records = []
        self.last_actual_layer_records = []
        self.last_actual_call_records = []
        self.last_step_diagnostics = []

    def _create_empty_time_stats(self):
        """Create an empty timing-statistics dictionary."""
        return {
            "stage1_draft_forward": [],
            "stage1_init_tree": [],
            "stage1_total": [],
            "stage1_propose_total": [],
            "stage1_find_anchor": [],
            "stage1_get_anchor_candidates": [],
            "stage1_relax": [],
            "stage1_layer1": [],
            "stage1_expand_parents": [],
            "stage1_global_topk": [],
            "stage1_update_paths_mask": [],
            "stage1_budget_prune": [],
            "stage1_postprocess": [],

            "stage2_medium_forward": [],
            "stage2_override_scores": [],
            "stage2_reselect_frontier": [],
            "stage2_expand_correction": [],
            "stage2_prune_correction": [],
            "stage2_trim_new_leaves": [],
            "stage2_total": [],

            # 3: Target
            "stage3_set_tree_mask": [],
            "stage3_prepare_position": [],
            "stage3_target_forward": [],
            "stage3_get_logits": [],
            "stage3_prepare_candidates": [],
            "stage3_evaluate_posterior": [],
            "stage3_update_input_ids": [],
            "stage3_update_kv_cache": [],
            "stage3_sample_bonus": [],
            "stage3_total": [],
        }

    def reset_time_stats(self):
        self.time_stats = self._create_empty_time_stats()

    def get_time_stats_summary(self):
        summary = {}
        for key, times in self.time_stats.items():
            if times:
                summary[key] = {
                    "total_ms": sum(times) * 1000,
                    "avg_ms": (sum(times) / len(times)) * 1000,
                    "count": len(times),
                }
            else:
                summary[key] = {"total_ms": 0, "avg_ms": 0, "count": 0}
        return summary

    def profile_target_forward_linear_curve(
        self,
        token_counts,
        repeats: int = 3,
        token_id: int = 1,
    ):


        repeats = max(1, int(repeats))
        token_id = int(token_id)
        if hasattr(self, "past_key_values") and self.past_key_values is not None:
            past_key_values = self.past_key_values
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=self.max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        device = next(self.base_model.parameters()).device
        records = []
        for token_count in list(token_counts or []):
            node_count = max(1, int(token_count))
            node_parents = [-1] + [i for i in range(node_count - 1)]
            tree_mask = torch.eye(node_count, dtype=torch.bool)
            tree_mask[:, 0] = True
            for i in range(1, node_count):
                p = int(node_parents[i])
                if p >= 0:
                    tree_mask[i].add_(tree_mask[p])
            tree_position_ids = (torch.sum(tree_mask, dim=1) - 1).to(device)
            tree_mask = tree_mask.float().to(device)[None, None]
            draft_tokens = torch.full(
                (1, node_count),
                token_id,
                dtype=torch.long,
                device=device,
            )
            position_ids = tree_position_ids.unsqueeze(0)

            for _ in range(repeats):
                current_length_data.zero_()
                self.base_model.model.tree_mask = tree_mask
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = self.base_model.model(
                        input_ids=draft_tokens,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                    )
                    _ = self.base_model.lm_head(outputs[0])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                records.append(
                    {
                        "step_idx": None,
                        "tree_nodes": int(node_count),
                        "forward_tokens": int(node_count),
                        "elapsed_ms": float(elapsed_ms),
                        "source": "synthetic_linear",
                    }
                )
        self.base_model.model.tree_mask = None
        current_length_data.zero_()
        return records

    def _extract_topk_logprobs(self, logits: torch.Tensor, width: int):
        if logits.dim() == 3:
            logits = logits[0]
        width = max(1, min(width, logits.shape[-1]))
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        top = torch.topk(logprobs, width, dim=-1)
        return top.indices, top.values

    def _record_draft_profile(self):
        profile = getattr(self.draft_generator, "last_profile_stats", None)
        if not profile:
            return {"stage2_total": 0.0}

        mapping = {
            "propose_total": "stage1_propose_total",
            "find_anchor": "stage1_find_anchor",
            "get_anchor_candidates": "stage1_get_anchor_candidates",
            "relax": "stage1_relax",
            "layer1": "stage1_layer1",
            "expand_parents": "stage1_expand_parents",
            "global_topk": "stage1_global_topk",
            "update_paths_mask": "stage1_update_paths_mask",
            "budget_prune": "stage1_budget_prune",
            "postprocess": "stage1_postprocess",
        }
        for src_key, dst_key in mapping.items():
            self.time_stats[dst_key].append(float(profile.get(src_key, 0.0)))
        stage2_mapping = {
            "medium_forward": "stage2_medium_forward",
            "correction_override_scores": "stage2_override_scores",
            "correction_reselect_frontier": "stage2_reselect_frontier",
            "correction_expand": "stage2_expand_correction",
            "correction_prune": "stage2_prune_correction",
            "correction_trim": "stage2_trim_new_leaves",
        }
        stage2_sum = 0.0
        for src_key, dst_key in stage2_mapping.items():
            v = float(profile.get(src_key, 0.0))
            self.time_stats[dst_key].append(v)
            stage2_sum += v

        stage2_total = float(profile.get("correction_total", stage2_sum))
        # correction_total
        if stage2_total < stage2_sum:
            stage2_total = stage2_sum
        return {"stage2_total": stage2_total}

    def _after_target_tree_forward(
        self,
        input_ids,
        draft_tokens,
        retrieve_indices,
        tree_logits,
        step_idx: int,
        collect_time_stats: bool = False,
    ):
        """Hook for subclasses that want to consume the verified draft tree."""
        return {}

    def _print_tree(
        self,
        title,
        draft_tokens,
        retrieve_indices,
        node_match_ngram=None,
        correction_loops=None,
    ):


        if draft_tokens.dim() > 1:
             draft_tokens = draft_tokens[0]
        if node_match_ngram is not None and torch.is_tensor(node_match_ngram):
            node_match_ngram = node_match_ngram.detach().cpu().tolist()
        elif node_match_ngram is not None:
            node_match_ngram = list(node_match_ngram)

        # Build Adjacency List
        # retrieve_indices: [num_paths, max_depth]
        # Paths are indices into draft_tokens

        adj = {}
        # Root is always index 0
        node_depth = {0: 0}

        num_paths = retrieve_indices.shape[0]
        max_depth = retrieve_indices.shape[1]
        corr_loops = []
        if correction_loops is not None:
            try:
                corr_loops = [int(x) for x in correction_loops]
            except Exception:
                corr_loops = []
        print(
            f"[NGRAM DEBUG] {title} total_nodes={draft_tokens.shape[0]}, "
            f"leaves={num_paths}, max_depth={max_depth}, corr_loops={corr_loops}"
        )

        # For visualization, we need parent->child map
        # Retrieve indices format: [RootIdx=0, ChildLayer1, ChildLayer2, ...]

        # Note: topK_genrate returns paths starting from Root?
        # Let's verify standard EAGLE outputs.
        # Usually indices are: [0, idx1, idx2...]

        # Let's just traverse all paths
        for p in range(num_paths):
            path = retrieve_indices[p].tolist()
            # path should be [0, idx_layer1, idx_layer2...]
            # if 0 is included.
            # If not included, we assume parent of path[0] is 0 if depth 0?

            # EAGLE conventions: path indices are into draft_tokens tensor.
            # -1 is padding

            for d in range(len(path) - 1):
                parent = path[d]
                child = path[d+1]
                if parent == -1 or child == -1: break

                if parent not in adj: adj[parent] = []
                if child not in adj[parent]: adj[parent].append(child)
                node_depth[child] = d + 1

        # Recursive Print
        def _recursive_print(node_idx, prefix="", is_last=True, is_root=True):
            token_id = draft_tokens[node_idx].item()
            token_str = self.tokenizer.decode([token_id], errors="replace")
            # Cleaning string for display
            token_str = token_str.replace('\n', '\\n').replace('\r', '\\r')

            # Connector
            if is_root:
                connector = ""
                new_prefix = ""
            else:
                connector = "    " if is_last else "    "
                new_prefix = prefix + ("    " if is_last else "    ")

            layer = node_depth.get(node_idx, '?')
            if node_match_ngram is not None and node_idx < len(node_match_ngram):
                match_ngram_size = node_match_ngram[node_idx]
            else:
                match_ngram_size = -1
            if isinstance(match_ngram_size, int) and match_ngram_size >= 0:
                layer_info = f"Layer {layer}, match_ngram={match_ngram_size}"
            else:
                layer_info = f"Layer {layer}, match_ngram=N/A"
            print(f"{prefix}{connector}{token_str} ({layer_info})")

            if node_idx in adj:
                children = adj[node_idx]
                # Sort children by index just to be deterministic
                children.sort()
                for i, child in enumerate(children):
                    _recursive_print(child, new_prefix, i == len(children)-1, False)

        _recursive_print(0)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_decode_steps=None,
        is_llama3=False,
        log=False,
        collect_time_stats=False,
        collect_step_diagnostics=False,
        debug=False,
    ):


        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()

        # ngram/prefill
        self.draft_generator.reset_kv()
        self.last_sample_correction_layer_freq = []
        self.last_middle_forward_records = []
        self.last_target_forward_records = []
        self.last_actual_layer_records = []
        self.last_actual_call_records = []
        self.last_step_diagnostics = []
        correction_layer_counter = {}
        step_diagnostics = []

        def _accumulate_correction_layers():
            loops = getattr(self.draft_generator, "last_middle_correction_loops", [])
            if loops is None:
                return
            for x in loops:
                try:
                    layer_idx = int(x)
                except Exception:
                    continue
                if layer_idx <= 0:
                    continue
                correction_layer_counter[layer_idx] = correction_layer_counter.get(layer_idx, 0) + 1

        if hasattr(self, "past_key_values") and self.past_key_values is not None:
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=self.max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        # input_ids
        device = next(self.base_model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        self.base_model.model.tree_mask = None
        self.base_model.model.tree_mode = None

        # Prefill: base model forward
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
            )
            orig = self.base_model.lm_head(outputs[0])

        prefill_top_tokens, prefill_top_logprobs = self._extract_topk_logprobs(
            orig[0], self.draft_generator.max_topk
        )
        self.draft_generator.set_prefill_cache(
            reference_tokens=input_ids[0],
            top_tokens=prefill_top_tokens,
            top_logprobs=prefill_top_logprobs,
            prompt_tokens=input_ids[0],
        )

        # token
        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            sample_token = torch.multinomial(probabilities, 1)
        else:
            sample_token = torch.argmax(orig[:, -1])
            sample_token = sample_token[None, None]

        # root token = sample_token
        input_ids_with_token = torch.cat((input_ids, sample_token.to(input_ids.device)), dim=1)
        if collect_time_stats:
            torch.cuda.synchronize()
            t_init_tree_start = time.perf_counter()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.draft_generator.topK_genrate(
            None,  # hidden_states - ngram draft
            input_ids_with_token,
            self.base_model.lm_head,  # head - ngram draft
            logits_processor,
            collect_time_stats=collect_time_stats,
        )
        current_tree_middle_records = [
            dict(x) for x in getattr(self.draft_generator, "middle_forward_records", [])
        ]
        current_tree_actual_layer_records = [
            dict(x) for x in getattr(self.draft_generator, "last_actual_layer_records", [])
        ]
        current_tree_actual_call_records = [
            dict(x) for x in getattr(self.draft_generator, "last_actual_call_records", [])
        ]
        current_tree_correction_loops = list(
            getattr(self.draft_generator, "last_middle_correction_loops", [])
        )
        if collect_time_stats:
            torch.cuda.synchronize()
            t_init_tree_end = time.perf_counter()
            init_elapsed = t_init_tree_end - t_init_tree_start
            profile_stats = self._record_draft_profile()
            stage2_cost = float(profile_stats.get("stage2_total", 0.0))
            stage2_cost = max(0.0, min(stage2_cost, init_elapsed))
            self.time_stats["stage1_init_tree"].append(max(0.0, init_elapsed - stage2_cost))
            self.time_stats["stage2_total"].append(stage2_cost)

        # Ensure tensors on correct device
        draft_tokens = draft_tokens.to(input_ids.device)
        retrieve_indices = retrieve_indices.to(input_ids.device)
        tree_mask = tree_mask.to(input_ids.device)
        tree_position_ids = tree_position_ids.to(input_ids.device)
        _accumulate_correction_layers()

        debug_tree_limit = 10
        debug_tree_printed = 0
        debug_limit_notified = False
        if debug:
            if debug_tree_printed < debug_tree_limit:
                print("[NGRAM DEBUG] Draft Tree (init)")
                self._print_tree(
                    "Draft",
                    draft_tokens,
                    retrieve_indices,
                    node_match_ngram=getattr(self.draft_generator, "last_node_match_ngram", None),
                    correction_loops=getattr(self.draft_generator, "last_middle_correction_loops", []),
                )
                debug_tree_printed += 1

        new_token = 0
        reserve_tokens = int(self.draft_generator.total_tokens)
        if hasattr(self.draft_generator, "get_kv_tree_token_reserve"):
            reserve_tokens = int(self.draft_generator.get_kv_tree_token_reserve())
        max_length = self.max_length - reserve_tokens - 10

        for idx in range(max_length):
            if collect_time_stats:
                torch.cuda.synchronize()
                t_stage3_start = time.perf_counter()

            if collect_time_stats:
                torch.cuda.synchronize()
                t_mask_start = time.perf_counter()

            self.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_mask_end = time.perf_counter()
                self.time_stats["stage3_set_tree_mask"].append(t_mask_end - t_mask_start)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_pos_start = time.perf_counter()

            position_ids = tree_position_ids + input_ids.shape[1]
            if position_ids is not None and position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_pos_end = time.perf_counter()
                self.time_stats["stage3_prepare_position"].append(t_pos_end - t_pos_start)

            # 3.3 Target forward
            collect_target_timing = bool(
                collect_time_stats
                or getattr(self.draft_generator, "should_collect_scheduler_target_timing", lambda: False)()
            )
            if collect_target_timing:
                torch.cuda.synchronize()
                t_forward_start = time.perf_counter()

            with torch.inference_mode():
                outputs = self.base_model.model(
                    input_ids=draft_tokens,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                tree_logits = self.base_model.lm_head(outputs[0])

            if collect_target_timing:
                torch.cuda.synchronize()
                t_forward_end = time.perf_counter()
                target_forward_elapsed = t_forward_end - t_forward_start
            else:
                target_forward_elapsed = 0.0

            if collect_time_stats:
                self.time_stats["stage3_target_forward"].append(target_forward_elapsed)
            if collect_target_timing:
                self.last_target_forward_records.append(
                    {
                        "step_idx": int(idx),
                        "tree_nodes": int(draft_tokens.shape[1]),
                        "forward_tokens": int(draft_tokens.shape[1]),
                        "elapsed_ms": float(target_forward_elapsed * 1000.0),
                    }
                )
            self._after_target_tree_forward(
                input_ids=input_ids,
                draft_tokens=draft_tokens,
                retrieve_indices=retrieve_indices,
                tree_logits=tree_logits,
                step_idx=int(idx),
                collect_time_stats=collect_time_stats,
            )

            if collect_time_stats:
                torch.cuda.synchronize()
                t_logits_start = time.perf_counter()

            retrieve_indices = retrieve_indices.to(tree_logits.device)
            logits = tree_logits[0, retrieve_indices]

            if collect_time_stats:
                torch.cuda.synchronize()
                t_logits_end = time.perf_counter()
                self.time_stats["stage3_get_logits"].append(t_logits_end - t_logits_start)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_cand_start = time.perf_counter()

            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            retrieve_indices_input = retrieve_indices.to(draft_tokens.device)
            candidates = draft_tokens[0, retrieve_indices_input]

            if collect_time_stats:
                torch.cuda.synchronize()
                t_cand_end = time.perf_counter()
                self.time_stats["stage3_prepare_candidates"].append(t_cand_end - t_cand_start)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_eval_start = time.perf_counter()

            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accept_length_int = int(accept_length.item()) if torch.is_tensor(accept_length) else int(accept_length)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_eval_end = time.perf_counter()
                self.time_stats["stage3_evaluate_posterior"].append(t_eval_end - t_eval_start)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_input_start = time.perf_counter()

            prev_input_len = input_ids.shape[1]
            select_indices = (
                retrieve_indices[best_candidate, : accept_length_int + 1] + prev_input_len
            )

            input_ids = torch.cat(
                [input_ids, candidates[None, best_candidate, : accept_length_int + 1].to(input_ids.device)], dim=-1
            )

            if collect_time_stats:
                torch.cuda.synchronize()
                t_input_end = time.perf_counter()
                self.time_stats["stage3_update_input_ids"].append(t_input_end - t_input_start)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_kv_start = time.perf_counter()

            for kv_data in past_key_values_data:
                tgt = kv_data[..., select_indices.to(kv_data.device), :]
                dst = kv_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
                dst.copy_(tgt, non_blocking=True)

            current_length_data.fill_(prev_input_len + tgt.shape[-2])

            if collect_time_stats:
                torch.cuda.synchronize()
                t_kv_end = time.perf_counter()
                self.time_stats["stage3_update_kv_cache"].append(t_kv_end - t_kv_start)

            if collect_time_stats:
                torch.cuda.synchronize()
                t_sample_start = time.perf_counter()

            if logits_processor is not None:
                sample_token = torch.multinomial(sample_p, 1)
                sample_token = sample_token[None]
            else:
                sample_token = torch.argmax(sample_p)
                sample_token = sample_token[None, None]

            if collect_time_stats:
                torch.cuda.synchronize()
                t_sample_end = time.perf_counter()
                self.time_stats["stage3_sample_bonus"].append(t_sample_end - t_sample_start)

            if collect_step_diagnostics:
                accept_tokens = int(accept_length_int + 1)
                tree_node_count = int(tree_logits.shape[1])
                retrieve_path_count = int(candidates.shape[0]) if candidates.dim() >= 2 else 0
                retrieve_path_max_len = int(candidates.shape[1]) if candidates.dim() >= 2 else 0
                if retrieve_path_count > 0 and retrieve_path_max_len > 0:
                    retrieve_path_mean_len = float(
                        (candidates >= 0).sum(dim=1).float().mean().item()
                    )
                else:
                    retrieve_path_mean_len = 0.0
                root_logprobs = torch.log_softmax(logits[0, 0].float(), dim=-1)
                root_probs = torch.exp(root_logprobs)
                root_top = torch.topk(root_probs, k=min(2, int(root_probs.shape[-1])), dim=-1)
                root_top1_prob = float(root_top.values[0].item()) if root_top.values.numel() > 0 else 0.0
                root_top2_prob = float(root_top.values[1].item()) if root_top.values.numel() > 1 else 0.0
                root_margin = float(root_top1_prob - root_top2_prob)
                root_entropy = float((-(root_probs * root_logprobs).sum()).item())
                sample_p_safe = sample_p.float().clamp_min(1e-12)
                sample_p_entropy = float((-(sample_p_safe * torch.log(sample_p_safe)).sum()).item())
                accepted_tokens_vec = candidates[best_candidate, :accept_tokens].to(logits.device)
                accepted_logprobs = torch.log_softmax(
                    logits[best_candidate, :accept_tokens].float(), dim=-1
                )
                accepted_path_token_logprobs = accepted_logprobs.gather(
                    -1, accepted_tokens_vec.unsqueeze(-1)
                ).squeeze(-1)
                accepted_path_logprob_sum = float(accepted_path_token_logprobs.sum().item())
                accepted_path_logprob_mean = float(accepted_path_token_logprobs.mean().item())
                step_diagnostics.append(
                    {
                        "step_idx": int(idx),
                        "tree_nodes": int(draft_tokens.shape[1]),
                        "tree_node_count": int(tree_node_count),
                        "retrieve_path_count": int(retrieve_path_count),
                        "retrieve_path_max_len": int(retrieve_path_max_len),
                        "retrieve_path_mean_len": float(retrieve_path_mean_len),
                        "accept_tokens": int(accept_tokens),
                        "accepted_path_len": int(accept_tokens),
                        "accepted_path_len_ratio": (
                            float(accept_tokens) / float(retrieve_path_max_len)
                            if retrieve_path_max_len > 0 else 0.0
                        ),
                        "accepted_path_index": int(best_candidate),
                        "input_len_before": int(prev_input_len),
                        "input_len_after": int(input_ids.shape[1]),
                        "new_tokens_cum_after": int(new_token + accept_tokens),
                        "target_forward_ms": float(target_forward_elapsed * 1000.0),
                        "root_top1_prob": float(root_top1_prob),
                        "root_top2_prob": float(root_top2_prob),
                        "root_margin": float(root_margin),
                        "root_entropy": float(root_entropy),
                        "sample_p_entropy": float(sample_p_entropy),
                        "accepted_path_logprob_sum": float(accepted_path_logprob_sum),
                        "accepted_path_logprob_mean": float(accepted_path_logprob_mean),
                        "middle_call_count": int(len(current_tree_middle_records)),
                        "middle_forward_total_ms": float(
                            sum(float(x.get("elapsed_ms", 0.0)) for x in current_tree_middle_records)
                        ),
                        "middle_call_layers": [
                            int(x) for x in current_tree_correction_loops
                        ],
                        "actual_layers": [dict(x) for x in current_tree_actual_layer_records],
                        "actual_calls": [dict(x) for x in current_tree_actual_call_records],
                        "middle_records": [dict(x) for x in current_tree_middle_records],
                    }
                )

            if hasattr(self.draft_generator, "observe_scheduler_step"):
                self.draft_generator.observe_scheduler_step(
                    accept_tokens=int(accept_length_int + 1),
                    target_forward_ms=float(target_forward_elapsed * 1000.0),
                    middle_records=current_tree_middle_records,
                )

            if self.update_prefill_with_accepted and accept_length_int >= 0:
                accepted_tokens = candidates[best_candidate, : accept_length_int + 1].to(input_ids.device)
                accepted_logits = logits[best_candidate, : accept_length_int + 1]
                acc_top_tokens, acc_top_logprobs = self._extract_topk_logprobs(
                    accepted_logits, self.draft_generator.max_topk
                )
                self.draft_generator.append_prefill_cache(
                    new_tokens=accepted_tokens,
                    new_top_tokens=acc_top_tokens,
                    new_top_logprobs=acc_top_logprobs,
                )

            if collect_time_stats:
                torch.cuda.synchronize()
                t_stage3_end = time.perf_counter()
                self.time_stats["stage3_total"].append(t_stage3_end - t_stage3_start)

            new_token += accept_length_int + 1

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
            if is_llama3 and stop_token_id in input_ids[0, input_len:]:
                break
            if new_token >= max_new_tokens:
                break
            if max_decode_steps is not None and (idx + 1) >= int(max_decode_steps):
                break

            if collect_time_stats:
                torch.cuda.synchronize()
                t_stage1_start = time.perf_counter()

            input_ids_with_token = torch.cat((input_ids, sample_token.to(input_ids.device)), dim=1)
            middle_record_start = len(getattr(self.draft_generator, "middle_forward_records", []))
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.draft_generator.topK_genrate(
                None,  # hidden_states - ngram draft
                input_ids_with_token,
                self.base_model.lm_head,  # head - ngram draft
                logits_processor,
                collect_time_stats=collect_time_stats,
            )
            current_tree_middle_records = [
                dict(x)
                for x in getattr(self.draft_generator, "middle_forward_records", [])[middle_record_start:]
            ]
            current_tree_actual_layer_records = [
                dict(x) for x in getattr(self.draft_generator, "last_actual_layer_records", [])
            ]
            current_tree_actual_call_records = [
                dict(x) for x in getattr(self.draft_generator, "last_actual_call_records", [])
            ]
            current_tree_correction_loops = list(
                getattr(self.draft_generator, "last_middle_correction_loops", [])
            )
            _accumulate_correction_layers()
            if debug:
                if debug_tree_printed < debug_tree_limit:
                    print(f"[NGRAM DEBUG] Draft Tree (step-{idx + 1})")
                    self._print_tree(
                        "Draft",
                        draft_tokens,
                        retrieve_indices,
                        node_match_ngram=getattr(self.draft_generator, "last_node_match_ngram", None),
                        correction_loops=getattr(self.draft_generator, "last_middle_correction_loops", []),
                    )
                    debug_tree_printed += 1
                elif not debug_limit_notified:
                    print(f"[NGRAM DEBUG] tree print limit reached ({debug_tree_limit})")
                    debug_limit_notified = True

            if collect_time_stats:
                torch.cuda.synchronize()
                t_stage1_end = time.perf_counter()
                stage1_elapsed = t_stage1_end - t_stage1_start
                profile_stats = self._record_draft_profile()
                stage2_cost = float(profile_stats.get("stage2_total", 0.0))
                stage2_cost = max(0.0, min(stage2_cost, stage1_elapsed))
                stage1_elapsed_wo_correction = max(0.0, stage1_elapsed - stage2_cost)
                self.time_stats["stage1_draft_forward"].append(stage1_elapsed_wo_correction)
                self.time_stats["stage1_total"].append(stage1_elapsed_wo_correction)
                self.time_stats["stage2_total"].append(stage2_cost)

        depth_cfg = max(0, int(getattr(self.draft_generator, "depth", 0)))
        observed_max_layer = max(correction_layer_counter.keys()) if correction_layer_counter else 0
        hist_len = max(depth_cfg, observed_max_layer)
        self.last_sample_correction_layer_freq = [
            int(correction_layer_counter.get(layer, 0))
            for layer in range(1, hist_len + 1)
        ]
        if debug and (
            bool(getattr(self.draft_generator, "combine_enabled", False))
            or any(self.last_sample_correction_layer_freq)
        ):
            print(
                f"[NGRAM DEBUG] sample_correction_layer_freq="
                f"{self.last_sample_correction_layer_freq}"
            )
        self.last_middle_forward_records = list(
            getattr(self.draft_generator, "middle_forward_records", [])
        )
        self.last_actual_layer_records = list(
            getattr(self.draft_generator, "all_actual_layer_records", [])
        )
        self.last_actual_call_records = list(
            getattr(self.draft_generator, "all_actual_call_records", [])
        )
        self.last_step_diagnostics = list(step_diagnostics)

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx + 1
