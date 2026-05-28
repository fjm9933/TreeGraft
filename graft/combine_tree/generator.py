"""TreeGraft combine-tree draft generator."""


import math
import time
from typing import Iterable, List, Optional

import torch

from graft.model.kv_cache import initialize_past_key_values
from graft.ngram_tree.generator import NgramDraftGenerator
from .scheduler import TreeGraftScheduler


class CombineDraftGenerator(NgramDraftGenerator):
    """Combine n-gram drafting with middle-drafter correction."""

    def __init__(
        self,
        *args,
        draft_strategy: str = "combine",
        combine_layers: List[int] = None,
        trim_nonparent_new_leaves_after_combine: bool = False,
        reselect_frontier_by_middle_score: bool = True,
        reselect_frontier_lookback_layers: int = 0,
        middle_model_path: str = None,
        middle_tokenizer_path: str = None,
        middle_max_length: int = 2048,
        middle_device_map: str = "auto",
        middle_dtype=torch.float16,
        scheduler_family: str = "none",
        scheduler_budget_b0: Optional[int] = None,
        scheduler_pair_id: str = "",
        scheduler_task_id: str = "",
        **kwargs,
    ):
        requested_strategy = (draft_strategy or "combine").lower()
        # prefill_topw/direct_match combine
        base_strategy = "direct_match" if requested_strategy == "combine" else requested_strategy
        super().__init__(*args, draft_strategy=base_strategy, **kwargs)

        self.requested_draft_strategy = requested_strategy
        self.combine_enabled = requested_strategy == "combine"
        self.base_stop_on_relax_zero = bool(self.stop_on_relax_zero)

        self.combine_layers = list(combine_layers or [])
        self.combine_layer_set = set()
        self.combine_event_mode = False
        self.combine_event_max = 0
        self._parse_combine_layers(self.combine_layers)
        self.trim_nonparent_new_leaves_after_combine = bool(trim_nonparent_new_leaves_after_combine)
        self.reselect_frontier_by_middle_score = bool(reselect_frontier_by_middle_score)
        try:
            self.reselect_frontier_lookback_layers = max(0, int(reselect_frontier_lookback_layers))
        except Exception:
            self.reselect_frontier_lookback_layers = 0

        if self.combine_event_mode:
            self.stop_on_relax_zero = True

        self.middle_model_path = middle_model_path
        self.middle_tokenizer_path = middle_tokenizer_path or middle_model_path
        self.middle_max_length = int(middle_max_length)
        self.middle_device_map = middle_device_map
        self.middle_dtype = middle_dtype
        self.combine_skip_source = "direct_match"
        self.scheduler_family = str(scheduler_family or "none")
        if self.scheduler_family != "none" and not self.combine_enabled:
            raise ValueError("scheduler_family requires draft_strategy=combine")
        if self.scheduler_family != "none" and self.combine_event_mode:
            raise ValueError("online scheduler does not support event-mode combine_layers=[-1,k]")

        self.middle_model = None
        self.middle_past_key_values = None
        self.middle_past_key_values_data = None
        self.middle_current_length_data = None
        self.middle_synced_ctx_len = 0
        self.middle_forward_records = []
        self.last_actual_layer_records = []
        self.all_actual_layer_records = []
        self.last_actual_call_records = []
        self.all_actual_call_records = []
        self._active_combine_layer = None
        if self.scheduler_family != "none":
            self.scheduler = TreeGraftScheduler(
                family=self.scheduler_family,
                budget_b0=scheduler_budget_b0,
                pair_id=str(scheduler_pair_id or ""),
                task_id=str(scheduler_task_id or ""),
                top_k=int(self.top_k),
                depth=int(self.depth),
            )
        else:
            self.scheduler = None
        self.scheduler_forced_layer_set = None
        self.scheduler_force_warmup_mode = False
        self.scheduler_force_warmup_target_timing = False
        self.scheduler_force_warmup_middle_timing = False

        if self.combine_enabled and not self.middle_model_path:
            raise ValueError("combine strategy requires middle_model_path")

    def set_combine_layers(self, combine_layers: List[int]):
        self.combine_layers = list(combine_layers or [])
        self._parse_combine_layers(self.combine_layers)
        self.stop_on_relax_zero = bool(self.base_stop_on_relax_zero)
        if self.combine_event_mode:
            self.stop_on_relax_zero = True

    def _create_empty_profile(self):
        profile = super()._create_empty_profile()
        profile.update(
            {
                "correction_total": 0.0,
                "correction_override_scores": 0.0,
                "correction_reselect_frontier": 0.0,
                "correction_expand": 0.0,
                "correction_prune": 0.0,
                "correction_trim": 0.0,
            }
        )
        return profile

    def _parse_combine_layers(self, layers: Iterable[int]):
        vals = []
        for x in list(layers or []):
            try:
                vals.append(int(x))
            except Exception:
                continue
        if len(vals) >= 2 and vals[0] == -1:
            self.combine_event_mode = True
            self.combine_event_max = max(0, int(vals[1]))
            self.combine_layer_set = set()
        else:
            self.combine_event_mode = False
            self.combine_event_max = 0
            self.combine_layer_set = {v for v in vals if v > 0}

    def _reselect_frontier_on_layer_scores(self, frontier, node_paths, node_parents, node_cum_scores):
        if not frontier:
            return []
        total = len(node_cum_scores)
        if total <= 0:
            return []
        depths = self._compute_node_depths(
            node_paths=node_paths,
            node_parents=node_parents,
            total=total,
        )
        layer_nodes, _ = self._collect_reselect_candidate_nodes(frontier, depths)
        if not layer_nodes:
            return [int(x) for x in frontier if 0 <= int(x) < total]
        return self._select_topk_nodes_by_score(layer_nodes, node_cum_scores)

    def _compute_node_depths(self, node_paths, node_parents, total):
        total = int(max(0, int(total)))
        if total <= 0:
            return []
        if node_paths is not None and len(node_paths) == total:
            return [max(0, int(len(path) - 1)) for path in node_paths]
        depths = [0] * total
        for i in range(1, total):
            p = int(node_parents[i]) if i < len(node_parents) else -1
            depths[i] = (depths[p] + 1) if (0 <= p < total) else 0
        return depths

    def _collect_reselect_candidate_nodes(self, frontier, depths):
        if not frontier or not depths:
            return [], set()
        total = len(depths)
        frontier_depth_set = set()
        for idx in frontier:
            ii = int(idx)
            if 0 <= ii < total:
                frontier_depth_set.add(int(depths[ii]))
        if not frontier_depth_set:
            return [], set()

        candidate_depth_set = set(frontier_depth_set)
        lookback = int(max(0, int(self.reselect_frontier_lookback_layers)))
        if lookback > 0:
            expanded_depth_set = set()
            for d in frontier_depth_set:
                dd = int(d)
                low = max(0, dd - lookback)
                for x in range(low, dd + 1):
                    expanded_depth_set.add(int(x))
            candidate_depth_set = expanded_depth_set

        layer_nodes = [i for i, d in enumerate(depths) if int(d) in candidate_depth_set]
        return layer_nodes, frontier_depth_set

    def _select_topk_nodes_by_score(self, nodes, node_cum_scores):
        if not nodes:
            return []
        ranked = [int(i) for i in nodes]
        ranked.sort(key=lambda i: (-float(node_cum_scores[i]), int(i)))
        keep = max(1, min(int(self.top_k), len(ranked)))
        return [int(i) for i in ranked[:keep]]

    def reset_kv(self):
        super().reset_kv()
        if self.middle_current_length_data is not None:
            self.middle_current_length_data.zero_()
        if self.middle_model is not None and hasattr(self.middle_model, "model"):
            self.middle_model.model.tree_mask = None
            self.middle_model.model.tree_mode = None
        self.middle_synced_ctx_len = 0
        self.middle_forward_records = []
        self.last_actual_layer_records = []
        self.all_actual_layer_records = []
        self.last_actual_call_records = []
        self.all_actual_call_records = []
        self._active_combine_layer = None

    def start_scheduler_warmup(self):
        return

    def activate_scheduler_online(self):
        return

    def suspend_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.suspend()

    def set_scheduler_forced_layers(self, layers: Optional[List[int]]):
        if self.scheduler is None:
            return
        if layers is None:
            self.scheduler_forced_layer_set = None
            return
        forced = set()
        for x in list(layers or []):
            try:
                v = int(x)
            except Exception:
                continue
            if v > 0:
                forced.add(v)
        self.scheduler_forced_layer_set = forced

    def set_scheduler_force_warmup_collection(
        self,
        target_timing: bool = False,
        middle_timing: bool = False,
    ):
        self.scheduler_force_warmup_target_timing = bool(target_timing)
        self.scheduler_force_warmup_middle_timing = bool(middle_timing)

    def set_scheduler_force_warmup_mode(self, enabled: bool = False):
        self.scheduler_force_warmup_mode = bool(enabled)

    def initialize_scheduler_phase0(
        self,
        A_t: float,
        T_tgt: float,
        T_mid: float,
        middle_curve: Optional[List[dict]] = None,
    ):
        if self.scheduler is not None:
            self.scheduler.initialize_phase0(
                A_t=A_t,
                T_tgt=T_tgt,
                T_mid=T_mid,
                middle_curve=middle_curve,
            )

    def finalize_scheduler_after_warmup(
        self,
        A_t: Optional[float],
        T_tgt: Optional[float] = None,
        T_mid: Optional[float] = None,
    ):
        if self.scheduler is not None:
            self.scheduler.finalize_after_warmup(
                A_t=A_t,
                T_tgt=T_tgt,
                T_mid=T_mid,
            )

    def set_scheduler_prompt_len_tokens(self, prompt_len_tokens: Optional[float]):
        if self.scheduler is None:
            return
        setter = getattr(self.scheduler, "set_prompt_len_tokens", None)
        if callable(setter):
            setter(prompt_len_tokens)

    def set_scheduler_pair_id(self, pair_id: Optional[str]):
        if self.scheduler is None:
            return
        setter = getattr(self.scheduler, "set_pair_id", None)
        if callable(setter):
            setter(pair_id)

    def set_scheduler_task_id(self, task_id: Optional[str]):
        if self.scheduler is None:
            return
        setter = getattr(self.scheduler, "set_task_id", None)
        if callable(setter):
            setter(task_id)

    def observe_scheduler_step(
        self,
        accept_tokens: int,
        target_forward_ms: float,
        middle_records,
    ):
        if self.scheduler is not None:
            self.scheduler.observe_step(
                accept_tokens=accept_tokens,
                target_forward_ms=target_forward_ms,
                middle_records=middle_records,
            )

    def should_collect_scheduler_target_timing(self) -> bool:
        if bool(getattr(self, "scheduler_force_warmup_target_timing", False)):
            return True
        return bool(self.scheduler is not None and self.scheduler.should_collect_warmup_target_timing())

    def should_collect_scheduler_middle_timing(self) -> bool:
        if bool(getattr(self, "scheduler_force_warmup_middle_timing", False)):
            return True
        return bool(self.scheduler is not None and self.scheduler.should_collect_warmup_middle_timing())

    def _load_kv_causal_lm(self, model_path, dtype, device_map):
        from transformers import AutoConfig
        from graft.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
        from graft.model.modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
        from graft.model.modeling_qwen3_kv import Qwen3ForCausalLM as KVQwen3ForCausalLM
        from graft.model.modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM

        model_type = AutoConfig.from_pretrained(model_path).architectures[0]
        if model_type == "LlamaForCausalLM":
            model = KVLlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        elif model_type == "Qwen2ForCausalLM":
            model = KVQwen2ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        elif model_type == "Qwen3ForCausalLM":
            model = KVQwen3ForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        else:
            model = KVMixtralForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
            )
        model.eval()
        return model

    def _check_aux_vocab_compat(self):
        return

    def _ensure_middle_model(self):
        if self.middle_model is not None:
            return
        if not self.middle_model_path:
            raise ValueError("middle_model_path is required for combine strategy")
        self.middle_model = self._load_kv_causal_lm(
            self.middle_model_path,
            dtype=self.middle_dtype,
            device_map=self.middle_device_map,
        )
        self._check_aux_vocab_compat()

    def _ensure_middle_cache(self):
        self._ensure_middle_model()
        if self.middle_past_key_values is not None:
            return
        (
            self.middle_past_key_values,
            self.middle_past_key_values_data,
            self.middle_current_length_data,
        ) = initialize_past_key_values(self.middle_model, max_length=self.middle_max_length)
        self.middle_synced_ctx_len = 0

    def _set_middle_kv_length(self, kv_len: int):
        if self.middle_current_length_data is not None:
            self.middle_current_length_data.fill_(int(kv_len))
        if self.middle_past_key_values is not None:
            for layer_kv in self.middle_past_key_values:
                if not layer_kv:
                    continue
                for kv in layer_kv:
                    if kv is not None and hasattr(kv, "current_length"):
                        kv.current_length.fill_(int(kv_len))

    def _build_tree_mask_and_position(self, node_tokens, node_parents, device):
        node_num = len(node_tokens)
        tree_mask = torch.eye(node_num, dtype=torch.bool)
        tree_mask[:, 0] = True
        for i in range(1, node_num):
            p = int(node_parents[i])
            if p >= 0:
                tree_mask[i].add_(tree_mask[p])
        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        tree_tokens = torch.tensor(node_tokens, dtype=torch.long, device=device).unsqueeze(0)
        tree_mask = tree_mask.float().to(device)[None, None]
        tree_position_ids = tree_position_ids.to(device)
        return tree_tokens, tree_mask, tree_position_ids

    def profile_middle_forward_linear_curve(
        self,
        token_counts,
        repeats: int = 3,
        prefix_len: int = 0,
        token_id: int = 1,
    ):


        self._ensure_middle_cache()
        repeats = max(1, int(repeats))
        prefix_len = max(0, int(prefix_len))
        token_id = int(token_id)
        model = self.middle_model
        model_device = next(model.parameters()).device
        records = []

        for token_count in list(token_counts or []):
            node_count = max(1, int(token_count))
            node_tokens = [token_id] * node_count
            node_parents = [-1] + [i for i in range(node_count - 1)]
            context_prefix_tokens = [token_id] * prefix_len
            for _ in range(repeats):
                self.middle_synced_ctx_len = 0
                self._set_middle_kv_length(0)
                if model_device.type == "cuda":
                    torch.cuda.synchronize(model_device)
                t_start = time.perf_counter()
                self._run_middle_tree_forward(
                    context_prefix_tokens=context_prefix_tokens,
                    node_tokens=node_tokens,
                    node_parents=node_parents,
                    collect_time_stats=False,
                    profile=None,
                )
                if model_device.type == "cuda":
                    torch.cuda.synchronize(model_device)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                records.append(
                    {
                        "layer_idx": None,
                        "tree_nodes": int(node_count),
                        "prefix_len": int(prefix_len),
                        "forward_tokens": int(prefix_len + node_count),
                        "context_len": int(prefix_len),
                        "elapsed_ms": float(elapsed_ms),
                        "source": "synthetic_linear",
                    }
                )
        self.middle_synced_ctx_len = 0
        self._set_middle_kv_length(0)
        return records

    def _run_middle_tree_forward(
        self,
        context_prefix_tokens,
        node_tokens,
        node_parents,
        collect_time_stats=False,
        profile=None,
    ):


        record_scheduler_timing = bool(
            (not collect_time_stats) and self.should_collect_scheduler_middle_timing()
        )
        t_medium_start = time.perf_counter() if ((collect_time_stats and profile is not None) or record_scheduler_timing) else None
        self._ensure_middle_cache()
        model = self.middle_model
        model_device = next(model.parameters()).device
        if t_medium_start is not None and model_device.type == "cuda":
            torch.cuda.synchronize(model_device)

        tree_tokens, tree_mask, tree_position_ids = self._build_tree_mask_and_position(
            node_tokens=node_tokens,
            node_parents=node_parents,
            device=model_device,
        )
        model.model.tree_mask = tree_mask

        current_ctx_len = len(context_prefix_tokens)
        if current_ctx_len < int(self.middle_synced_ctx_len):
            self.middle_synced_ctx_len = 0
            self._set_middle_kv_length(0)

        fused_new_tokens = context_prefix_tokens[self.middle_synced_ctx_len:current_ctx_len]
        prefix_len = len(fused_new_tokens)
        base_ctx_len = current_ctx_len - prefix_len
        if base_ctx_len < 0:
            raise ValueError(
                f"Invalid base_ctx_len={base_ctx_len}, current_ctx_len={current_ctx_len}, prefix_len={prefix_len}"
            )
        self._set_middle_kv_length(base_ctx_len)

        if prefix_len > 0:
            prefix_ids = torch.tensor(
                fused_new_tokens, dtype=torch.long, device=model_device
            ).unsqueeze(0)
            forward_input_ids = torch.cat((prefix_ids, tree_tokens), dim=1)
            prefix_pos = torch.arange(
                base_ctx_len,
                base_ctx_len + prefix_len,
                device=model_device,
                dtype=torch.long,
            ).unsqueeze(0)
            draft_pos = tree_position_ids + current_ctx_len
            if draft_pos.dim() == 1:
                draft_pos = draft_pos.unsqueeze(0)
            position_ids = torch.cat((prefix_pos, draft_pos), dim=1)
        else:
            forward_input_ids = tree_tokens
            position_ids = tree_position_ids + current_ctx_len
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        outputs = model.model(
            input_ids=forward_input_ids,
            past_key_values=self.middle_past_key_values,
            position_ids=position_ids,
        )
        hidden_states = outputs[0]
        if prefix_len > 0:
            hidden_states = hidden_states[:, prefix_len:, :]
        tree_logits = model.lm_head(hidden_states)
        if t_medium_start is not None:
            if model_device.type == "cuda":
                torch.cuda.synchronize(model_device)
            medium_cost = time.perf_counter() - t_medium_start
            if collect_time_stats and profile is not None:
                profile["medium_forward"] += medium_cost
                profile["correction_total"] += medium_cost
            self.middle_forward_records.append(
                {
                    "layer_idx": (
                        int(self._active_combine_layer)
                        if self._active_combine_layer is not None
                        else None
                    ),
                    "tree_nodes": int(len(node_tokens)),
                    "prefix_len": int(prefix_len),
                    "forward_tokens": int(prefix_len + len(node_tokens)),
                    "context_len": int(current_ctx_len),
                    "elapsed_ms": float(medium_cost * 1000.0),
                }
            )
        self.middle_synced_ctx_len = current_ctx_len
        model.model.tree_mask = None
        return tree_logits

    def _override_scores_with_logits(self, tree_logits, node_tokens, node_parents, node_cum_scores):
        if len(node_tokens) == 0:
            return
        logprobs = torch.log_softmax(tree_logits[0].float(), dim=-1)
        node_cum_scores[0] = 0.0
        for idx in range(1, len(node_tokens)):
            parent_idx = int(node_parents[idx])
            tok = int(node_tokens[idx])
            if parent_idx < 0 or parent_idx >= logprobs.shape[0]:
                continue
            lp = float(logprobs[parent_idx, tok].item())
            pscore = float(node_cum_scores[parent_idx])
            if pscore == float("-inf"):
                node_cum_scores[idx] = float("-inf")
            else:
                node_cum_scores[idx] = pscore + lp

    def _override_scores_with_middle(
        self,
        tree_logits,
        node_tokens,
        node_parents,
        node_cum_scores,
    ):


        self._override_scores_with_logits(
            tree_logits=tree_logits,
            node_tokens=node_tokens,
            node_parents=node_parents,
            node_cum_scores=node_cum_scores,
        )

    def _collect_parent_suffixes(
        self,
        parent_idx: int,
        suffix_len: int,
        context_tokens,
        context_prefix_tokens,
        node_paths,
        node_children,
        ngram_index,
    ):


        suffix_len = max(1, int(suffix_len))
        parent_path = node_paths[int(parent_idx)]
        window_src = list(context_prefix_tokens) + list(parent_path)
        max_n = min(int(self.max_matching_ngram_size), len(window_src))

        existing_children = set(int(t) for t in node_children[int(parent_idx)].keys())
        coverage = set(existing_children)
        suffix_info = {}
        child_match_len_local = {}
        child_recent_pos_local = {}
        hit_relax_zero = False

        for n in range(max_n, 0, -1):
            query = tuple(int(x) for x in window_src[-n:])
            starts = ngram_index.get(n, {}).get(query, [])
            for start in starts:
                suffix_start = int(start) + int(n)
                suffix_end = suffix_start + suffix_len
                if suffix_end > len(context_tokens):
                    continue
                suffix = tuple(int(x) for x in context_tokens[suffix_start:suffix_end])
                if len(suffix) != suffix_len:
                    continue
                child_tok = int(suffix[0])
                child_pos = int(suffix_start)
                prev_mlen = child_match_len_local.get(child_tok, -1)
                if int(n) > int(prev_mlen):
                    child_match_len_local[child_tok] = int(n)
                prev_pos = child_recent_pos_local.get(child_tok, -1)
                if int(child_pos) > int(prev_pos):
                    child_recent_pos_local[child_tok] = int(child_pos)
                coverage.add(child_tok)
                if suffix in suffix_info:
                    continue
                if self._path_exists(parent_idx, suffix, node_children):
                    continue
                child_lp = self._estimate_token_logprob_from_cache(suffix_start - 1, suffix[0])
                suffix_info[suffix] = (int(n), float(child_lp), int(child_pos))

            if len(coverage) >= int(self.top_k):
                break

        if len(coverage) < int(self.top_k):
            hit_relax_zero = True
            if self.stop_on_relax_zero:
                return suffix_info, child_match_len_local, child_recent_pos_local, hit_relax_zero
            for tok in self._get_relax_token_order(context_tokens):
                tok = int(tok)
                if tok in coverage:
                    continue
                suffix, last_pos = self._build_relax_suffix(tok, context_tokens, suffix_len)
                if suffix in suffix_info:
                    continue
                if self._path_exists(parent_idx, suffix, node_children):
                    continue
                child_pos = int(last_pos) if last_pos is not None else -1
                prev_mlen = child_match_len_local.get(int(suffix[0]), -1)
                if 0 > int(prev_mlen):
                    child_match_len_local[int(suffix[0])] = 0
                prev_pos = child_recent_pos_local.get(int(suffix[0]), -1)
                if int(child_pos) > int(prev_pos):
                    child_recent_pos_local[int(suffix[0])] = int(child_pos)
                child_lp = self._estimate_token_logprob_from_cache(
                    (None if last_pos is None else int(last_pos) - 1),
                    int(suffix[0]),
                )
                suffix_info[suffix] = (0, float(child_lp), int(child_pos))
                coverage.add(int(suffix[0]))
                if len(coverage) >= int(self.top_k):
                    break

        return suffix_info, child_match_len_local, child_recent_pos_local, hit_relax_zero

    def _expand_layer_direct_match(
        self,
        frontier,
        suffix_len,
        context_tokens,
        context_prefix_tokens,
        ngram_index,
        max_nodes,
        node_tokens,
        node_parents,
        node_children,
        node_paths,
        node_match_len,
        node_recent_pos,
        node_cum_scores,
        collect_time_stats: bool = False,
        profile: dict = None,
    ):
        layer_child_candidates = []
        terminated_parents = 0
        for parent_idx in frontier:
            parent_score = float(node_cum_scores[parent_idx])
            suffix_info, child_match_len_local, child_recent_pos_local, hit_relax_zero = self._collect_parent_suffixes(
                parent_idx=parent_idx,
                suffix_len=suffix_len,
                context_tokens=context_tokens,
                context_prefix_tokens=context_prefix_tokens,
                node_paths=node_paths,
                node_children=node_children,
                ngram_index=ngram_index,
            )

            ordered_suffixes = sorted(
                suffix_info.items(),
                key=lambda kv: (-int(kv[1][0]), -int(kv[1][2]), kv[0]),
            )
            for suffix, (match_len_local, child_lp, child_pos_local) in ordered_suffixes:
                self._add_suffix_path(
                    parent_idx=parent_idx,
                    suffix_tokens=suffix,
                    match_len=int(match_len_local),
                    child_logprob=float(child_lp),
                    child_pos=int(child_pos_local),
                    max_nodes=max_nodes,
                    node_tokens=node_tokens,
                    node_parents=node_parents,
                    node_children=node_children,
                    node_paths=node_paths,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                    node_cum_scores=node_cum_scores,
                    source_parent_score=parent_score,
                )

            if self.stop_on_relax_zero and hit_relax_zero:
                terminated_parents += 1
                continue

            for child_tok, child_idx in node_children[parent_idx].items():
                child_tok = int(child_tok)
                mlen_local = int(child_match_len_local.get(child_tok, 0))
                pos_local = int(child_recent_pos_local.get(child_tok, node_recent_pos[child_idx]))
                node_match_len[child_idx] = max(int(node_match_len[child_idx]), int(mlen_local))
                node_recent_pos[child_idx] = max(int(node_recent_pos[child_idx]), int(pos_local))
                layer_child_candidates.append(
                    (
                        int(child_idx),
                        int(max(0, mlen_local)),
                        float(node_cum_scores[child_idx]),
                    )
                )

        return layer_child_candidates, terminated_parents

    def _quantile(self, values, q: float):
        nums = sorted(float(x) for x in list(values or []))
        if not nums:
            return 0.0
        if len(nums) == 1:
            return float(nums[0])
        q = max(0.0, min(1.0, float(q)))
        pos = q * float(len(nums) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(nums) - 1)
        frac = pos - float(lo)
        return float(nums[lo] * (1.0 - frac) + nums[hi] * frac)

    def _summarize_frontier_state(
        self,
        frontier,
        node_paths,
        node_cum_scores,
        node_children=None,
        node_match_len=None,
        node_recent_pos=None,
    ):
        if not frontier:
            return {
                "frontier_size": 0,
                "frontier_mean_depth": 0.0,
                "frontier_max_depth": 0.0,
                "frontier_mean_score": 0.0,
                "frontier_score_p50": 0.0,
                "frontier_score_p90": 0.0,
                "frontier_finite_score_ratio": 0.0,
                "frontier_leaf_ratio": 0.0,
                "parent_match_len_mean": 0.0,
                "parent_match_len_p90": 0.0,
                "parent_recent_pos_mean": 0.0,
                "parent_recent_pos_min": 0.0,
                "parent_recent_pos_max": 0.0,
                "parent_recent_pos_span": 0.0,
            }
        total = len(node_cum_scores)
        depths = []
        scores = []
        leaf_flags = []
        match_lens = []
        recent_positions = []
        for idx in frontier:
            ii = int(idx)
            if ii < 0 or ii >= total:
                continue
            if node_paths is not None and ii < len(node_paths):
                depths.append(float(max(0, int(len(node_paths[ii]) - 1))))
            else:
                depths.append(0.0)
            scores.append(float(node_cum_scores[ii]))
            if node_children is not None and ii < len(node_children):
                leaf_flags.append(1.0 if len(node_children[ii]) == 0 else 0.0)
            if node_match_len is not None and ii < len(node_match_len):
                match_lens.append(float(node_match_len[ii]))
            if node_recent_pos is not None and ii < len(node_recent_pos):
                recent_positions.append(float(node_recent_pos[ii]))
        count = len(scores)
        if count <= 0:
            return {
                "frontier_size": 0,
                "frontier_mean_depth": 0.0,
                "frontier_max_depth": 0.0,
                "frontier_mean_score": 0.0,
                "frontier_score_p50": 0.0,
                "frontier_score_p90": 0.0,
                "frontier_finite_score_ratio": 0.0,
                "frontier_leaf_ratio": 0.0,
                "parent_match_len_mean": 0.0,
                "parent_match_len_p90": 0.0,
                "parent_recent_pos_mean": 0.0,
                "parent_recent_pos_min": 0.0,
                "parent_recent_pos_max": 0.0,
                "parent_recent_pos_span": 0.0,
            }
        finite_scores = [x for x in scores if math.isfinite(float(x))]
        finite_count = len(finite_scores)
        mean_score = float(sum(finite_scores) / float(finite_count)) if finite_count > 0 else 0.0
        mean_match = float(sum(match_lens) / float(len(match_lens))) if match_lens else 0.0
        recent_mean = (
            float(sum(recent_positions) / float(len(recent_positions)))
            if recent_positions else 0.0
        )
        recent_min = float(min(recent_positions)) if recent_positions else 0.0
        recent_max = float(max(recent_positions)) if recent_positions else 0.0
        return {
            "frontier_size": int(count),
            "frontier_mean_depth": float(sum(depths) / float(count)),
            "frontier_max_depth": float(max(depths)) if depths else 0.0,
            "frontier_mean_score": float(mean_score),
            "frontier_score_p50": self._quantile(finite_scores, 0.50),
            "frontier_score_p90": self._quantile(finite_scores, 0.90),
            "frontier_finite_score_ratio": float(finite_count) / float(count),
            "frontier_leaf_ratio": (
                float(sum(leaf_flags) / float(len(leaf_flags))) if leaf_flags else 0.0
            ),
            "parent_match_len_mean": float(mean_match),
            "parent_match_len_p90": self._quantile(match_lens, 0.90),
            "parent_recent_pos_mean": float(recent_mean),
            "parent_recent_pos_min": float(recent_min),
            "parent_recent_pos_max": float(recent_max),
            "parent_recent_pos_span": float(max(0.0, recent_max - recent_min)),
        }

    def _summarize_reselect_transition(self, frontier_before, frontier_after, depths, frontier_depth_set):
        before_set = {int(x) for x in list(frontier_before or [])}
        after_list = [int(x) for x in list(frontier_after or [])]
        after_count = len(after_list)
        if after_count <= 0:
            return {
                "overlap_ratio": 0.0,
                "from_past_layer_ratio": 0.0,
            }
        overlap = sum(1 for x in after_list if x in before_set)
        from_past = 0
        for idx in after_list:
            if 0 <= idx < len(depths) and int(depths[idx]) not in frontier_depth_set:
                from_past += 1
        return {
            "overlap_ratio": float(overlap) / float(after_count),
            "from_past_layer_ratio": float(from_past) / float(after_count),
        }

    def _build_scheduler_record(
        self,
        layer_idx,
        depth,
        frontier,
        node_paths,
        node_parents,
        node_cum_scores,
        node_children,
        node_match_len,
        node_recent_pos,
    ):
        before_summary = self._summarize_frontier_state(
            frontier=frontier,
            node_paths=node_paths,
            node_cum_scores=node_cum_scores,
            node_children=node_children,
            node_match_len=node_match_len,
            node_recent_pos=node_recent_pos,
        )
        total = len(node_cum_scores)
        depths = self._compute_node_depths(
            node_paths=node_paths,
            node_parents=node_parents,
            total=total,
        )
        candidate_nodes, frontier_depth_set = self._collect_reselect_candidate_nodes(
            frontier=frontier,
            depths=depths,
        )
        pre_selected_frontier = self._select_topk_nodes_by_score(candidate_nodes, node_cum_scores)
        pre_frontier_summary = self._summarize_frontier_state(
            frontier=pre_selected_frontier,
            node_paths=node_paths,
            node_cum_scores=node_cum_scores,
            node_children=node_children,
            node_match_len=node_match_len,
            node_recent_pos=node_recent_pos,
        )
        pre_transition = self._summarize_reselect_transition(
            frontier_before=frontier,
            frontier_after=pre_selected_frontier,
            depths=depths,
            frontier_depth_set=frontier_depth_set,
        )
        pre_reselect_depth_delta = float(
            pre_frontier_summary["frontier_mean_depth"] - before_summary["frontier_mean_depth"]
        )
        return {
            "layer_idx": int(layer_idx),
            "remaining_layers": int(max(0, int(depth) - int(layer_idx) + 1)),
            "frontier_size": int(before_summary["frontier_size"]),
            "tree_nodes_before": int(total),
            "depth_before": float(before_summary["frontier_mean_depth"]),
            "frontier_before": float(before_summary["frontier_size"]),
            "score_before": float(before_summary["frontier_mean_score"]),
            "leaf_ratio_before": float(before_summary["frontier_leaf_ratio"]),
            "score_before_p50": float(before_summary["frontier_score_p50"]),
            "score_before_p90": float(before_summary["frontier_score_p90"]),
            "pre_reselect_candidate_size": float(len(candidate_nodes)),
            "pre_reselect_cands": float(len(candidate_nodes)),
            "pre_reselect_frontier_mean_depth": float(pre_frontier_summary["frontier_mean_depth"]),
            "pre_reselect_frontier_mean_score": float(pre_frontier_summary["frontier_mean_score"]),
            "pre_reselect_frontier_finite_score_ratio": float(
                pre_frontier_summary["frontier_finite_score_ratio"]
            ),
            "pre_reselect_depth_delta": float(pre_reselect_depth_delta),
            "pre_reselect_score_delta": float(
                pre_frontier_summary["frontier_mean_score"] - before_summary["frontier_mean_score"]
            ),
            "pre_reselect_overlap_ratio": float(pre_transition["overlap_ratio"]),
            "pre_reselect_from_past_layer_ratio": float(pre_transition["from_past_layer_ratio"]),
            "collapse_debt": float(max(0.0, -pre_reselect_depth_delta)),
            "scheduler_observation_only": True,
        }

    def _expand_layer_combine(
        self,
        frontier,
        context_prefix_tokens,
        max_nodes,
        node_tokens,
        node_parents,
        node_children,
        node_paths,
        node_match_len,
        node_recent_pos,
        node_cum_scores,
        collect_time_stats=False,
        profile=None,
        policy_record=None,
    ):
        if not frontier:
            return [], []

        tree_logits = self._run_middle_tree_forward(
            context_prefix_tokens=context_prefix_tokens,
            node_tokens=node_tokens,
            node_parents=node_parents,
            collect_time_stats=collect_time_stats,
            profile=profile,
        )
        t_override_start = (
            time.perf_counter() if (collect_time_stats and profile is not None) else None
        )
        self._override_scores_with_middle(
            tree_logits=tree_logits,
            node_tokens=node_tokens,
            node_parents=node_parents,
            node_cum_scores=node_cum_scores,
        )
        if t_override_start is not None:
            delta = time.perf_counter() - t_override_start
            profile["correction_override_scores"] += delta
            profile["correction_total"] += delta

        effective_frontier = [int(x) for x in frontier]
        before_summary = None
        depths = None
        frontier_depth_set = set()
        if policy_record is not None:
            before_summary = self._summarize_frontier_state(
                frontier=effective_frontier,
                node_paths=node_paths,
                node_cum_scores=node_cum_scores,
            )
            depths = self._compute_node_depths(
                node_paths=node_paths,
                node_parents=node_parents,
                total=len(node_cum_scores),
            )
            _, frontier_depth_set = self._collect_reselect_candidate_nodes(
                frontier=effective_frontier,
                depths=depths,
            )
        if self.reselect_frontier_by_middle_score:
            t_reselect_start = (
                time.perf_counter() if (collect_time_stats and profile is not None) else None
            )
            effective_frontier = self._reselect_frontier_on_layer_scores(
                frontier=effective_frontier,
                node_paths=node_paths,
                node_parents=node_parents,
                node_cum_scores=node_cum_scores,
            )
            if t_reselect_start is not None:
                delta = time.perf_counter() - t_reselect_start
                profile["correction_reselect_frontier"] += delta
                profile["correction_total"] += delta
        if policy_record is not None:
            after_summary = self._summarize_frontier_state(
                frontier=effective_frontier,
                node_paths=node_paths,
                node_cum_scores=node_cum_scores,
            )
            post_transition = self._summarize_reselect_transition(
                frontier_before=frontier,
                frontier_after=effective_frontier,
                depths=depths,
                frontier_depth_set=frontier_depth_set,
            )
            policy_record.update(
                {
                    "reselect_frontier_size_before": int(before_summary["frontier_size"]),
                    "reselect_frontier_size_after": int(after_summary["frontier_size"]),
                    "reselect_frontier_mean_depth_before": float(before_summary["frontier_mean_depth"]),
                    "reselect_frontier_mean_depth_after": float(after_summary["frontier_mean_depth"]),
                    "reselect_frontier_mean_score_before": float(before_summary["frontier_mean_score"]),
                    "reselect_frontier_mean_score_after": float(after_summary["frontier_mean_score"]),
                    "reselect_frontier_finite_score_ratio_before": float(
                        before_summary["frontier_finite_score_ratio"]
                    ),
                    "reselect_frontier_finite_score_ratio_after": float(
                        after_summary["frontier_finite_score_ratio"]
                    ),
                    "reselect_depth_delta": float(
                        after_summary["frontier_mean_depth"] - before_summary["frontier_mean_depth"]
                    ),
                    "reselect_score_delta": float(
                        after_summary["frontier_mean_score"] - before_summary["frontier_mean_score"]
                    ),
                    "reselect_overlap_ratio": float(post_transition["overlap_ratio"]),
                    "reselect_from_past_layer_ratio": float(
                        post_transition["from_past_layer_ratio"]
                    ),
                }
            )
        if not effective_frontier:
            return [], []

        layer_child_candidates = []
        newly_created_leaf_indices = []
        t_expand_start = time.perf_counter() if (collect_time_stats and profile is not None) else None
        for parent_idx in effective_frontier:
            parent_idx = int(parent_idx)
            parent_score = float(node_cum_scores[parent_idx])
            parent_recent = int(node_recent_pos[parent_idx]) if parent_idx < len(node_recent_pos) else -1

            parent_lp = torch.log_softmax(tree_logits[0, parent_idx].float(), dim=-1)
            take = min(int(self.top_k), int(parent_lp.shape[-1]))
            top = torch.topk(parent_lp, take, dim=-1)
            selected_child_indices = []

            for tok, lp in zip(top.indices.tolist(), top.values.tolist()):
                tok = int(tok)
                lp = float(lp)
                child_idx = node_children[parent_idx].get(tok)
                if child_idx is None:
                    if len(node_tokens) >= int(max_nodes):
                        break
                    child_idx = len(node_tokens)
                    node_tokens.append(tok)
                    node_parents.append(parent_idx)
                    node_children.append({})
                    node_paths.append(node_paths[parent_idx] + [tok])
                    node_match_len.append(0)
                    node_recent_pos.append(parent_recent)
                    node_cum_scores.append(parent_score + lp)
                    node_children[parent_idx][tok] = child_idx
                    newly_created_leaf_indices.append(int(child_idx))
                else:
                    child_idx = int(child_idx)
                    node_match_len[child_idx] = max(int(node_match_len[child_idx]), 0)
                    node_recent_pos[child_idx] = max(int(node_recent_pos[child_idx]), int(parent_recent))
                    node_cum_scores[child_idx] = parent_score + lp
                selected_child_indices.append(int(child_idx))

            # top-k
            seen_local = set()
            for child_idx in selected_child_indices:
                if child_idx in seen_local:
                    continue
                seen_local.add(child_idx)
                score_local = float(node_cum_scores[int(child_idx)])
                layer_child_candidates.append((int(child_idx), 0, score_local))
        if t_expand_start is not None:
            delta = time.perf_counter() - t_expand_start
            profile["correction_expand"] += delta
            profile["correction_total"] += delta

        return layer_child_candidates, newly_created_leaf_indices

    def _dedup_and_sort_child_candidates(self, layer_child_candidates):
        dedup = {}
        for child_idx, mlen, score_local in layer_child_candidates:
            prev = dedup.get(child_idx)
            if prev is None or (mlen, score_local) > (prev[0], prev[1]):
                dedup[child_idx] = (int(mlen), float(score_local))
        child_candidates = [(idx, v[0], v[1]) for idx, v in dedup.items()]
        child_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))
        return child_candidates

    def _apply_layer_prune(self, child_candidates, layer_idx: int, force_topk: bool = False):
        if force_topk:
            if len(child_candidates) > int(self.top_k):
                return child_candidates[: int(self.top_k)]
            return child_candidates
        if layer_idx >= 2 and len(child_candidates) > int(self.top_k):
            return child_candidates[: int(self.top_k)]
        return child_candidates

    def _trim_new_leaves_not_in_frontier(
        self,
        new_leaf_indices,
        frontier,
        node_tokens,
        node_parents,
        node_children,
        node_paths,
        node_match_len,
        node_recent_pos,
        node_cum_scores,
    ):


        if not new_leaf_indices:
            return (
                node_tokens,
                node_parents,
                node_children,
                node_paths,
                node_match_len,
                node_recent_pos,
                node_cum_scores,
                frontier,
            )

        keep_frontier = set(int(x) for x in frontier)
        remove_set = {int(x) for x in new_leaf_indices if int(x) not in keep_frontier and int(x) > 0}
        if not remove_set:
            return (
                node_tokens,
                node_parents,
                node_children,
                node_paths,
                node_match_len,
                node_recent_pos,
                node_cum_scores,
                frontier,
            )

        final_remove = set()
        for idx in remove_set:
            if 0 <= int(idx) < len(node_children) and len(node_children[int(idx)]) == 0:
                final_remove.add(int(idx))
        if not final_remove:
            return (
                node_tokens,
                node_parents,
                node_children,
                node_paths,
                node_match_len,
                node_recent_pos,
                node_cum_scores,
                frontier,
            )

        old_n = len(node_tokens)
        keep_old = [i for i in range(old_n) if i not in final_remove]
        old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_old)}

        new_tokens = [node_tokens[i] for i in keep_old]
        new_paths = [node_paths[i] for i in keep_old]
        new_match = [node_match_len[i] for i in keep_old]
        new_recent = [node_recent_pos[i] for i in keep_old]
        new_scores = [node_cum_scores[i] for i in keep_old]

        new_parents = []
        for old_idx in keep_old:
            old_parent = int(node_parents[old_idx])
            if old_parent < 0:
                new_parents.append(-1)
            else:
                new_parents.append(int(old2new[old_parent]))

        rebuilt_children = [dict() for _ in range(len(new_tokens))]
        for child_idx in range(1, len(new_tokens)):
            p = int(new_parents[child_idx])
            tok = int(new_tokens[child_idx])
            rebuilt_children[p][tok] = int(child_idx)

        new_frontier = [int(old2new[idx]) for idx in frontier if int(idx) in old2new]
        return (
            new_tokens,
            new_parents,
            rebuilt_children,
            new_paths,
            new_match,
            new_recent,
            new_scores,
            new_frontier,
        )

    def _collect_leaf_frontier(self, node_children):
        leaves = []
        for idx in range(0, len(node_children)):
            if len(node_children[idx]) == 0:
                leaves.append(int(idx))
        return leaves

    @torch.no_grad()
    def _topK_genrate_combine(self, input_ids, logits_processor, collect_time_stats=False):
        if self.use_adaptive_config:
            raise ValueError("combine currently does not support adaptive layer_config")

        input_ids_device = input_ids.device
        context_tokens = input_ids[0].detach().cpu().tolist()
        if not context_tokens:
            raise RuntimeError("input_ids is empty")

        depth = int(max(1, self.depth))
        top_k = int(max(1, self.top_k))
        final_max_nodes = int(self.total_tokens) + 1
        if self.legacy_total_token_stop:
            max_nodes = final_max_nodes
        else:
            max_nodes = max(
                final_max_nodes,
                1 + top_k + depth * top_k * top_k,
            )
        root_token = int(context_tokens[-1])
        context_prefix_tokens = context_tokens[:-1]
        ngram_index = self._build_context_ngram_index(context_tokens, self.max_matching_ngram_size)

        node_tokens = [root_token]
        node_parents = [-1]
        node_children = [dict()]
        node_paths = [[root_token]]
        node_match_len = [-1]
        node_recent_pos = [len(context_tokens) - 1]
        node_cum_scores = [0.0]
        correction_loops = []
        actual_layer_records = []
        actual_call_records = []

        frontier = [0]
        profile = self._create_empty_profile() if collect_time_stats else None
        base_suffix_len = int(max(1, self.max_matching_ngram_size))
        scheduler_active = bool(
            self.scheduler is not None
            and bool(getattr(self.scheduler, "enabled", False))
            and (
                bool(getattr(self.scheduler, "online_active", False))
                or bool(getattr(self, "scheduler_force_warmup_mode", False))
            )
        )
        scheduler_is_ready = bool(
            scheduler_active and getattr(self.scheduler, "controller_kind", "legacy") == "scheduler"
        )
        if self.scheduler is not None:
            self.scheduler.reset_decode_step_state()

        def _apply_first_layer_fallback(layer_idx, terminated_parents, frontier_now):
            if (
                self.stop_on_relax_zero
                and (not self.combine_event_mode)
                and layer_idx == 1
                and terminated_parents == len(frontier_now)
                and len(node_children[0]) == 0
            ):
                freq_order = self._get_relax_token_order(context_tokens)
                fallback_tok = int(freq_order[0]) if freq_order else int(root_token)
                fallback_pos = -1
                for i in range(len(context_tokens) - 1, -1, -1):
                    if int(context_tokens[i]) == fallback_tok:
                        fallback_pos = int(i)
                        break
                self._add_suffix_path(
                    parent_idx=0,
                    suffix_tokens=(fallback_tok,),
                    match_len=0,
                    child_logprob=0.0,
                    child_pos=int(fallback_pos),
                    max_nodes=max_nodes,
                    node_tokens=node_tokens,
                    node_parents=node_parents,
                    node_children=node_children,
                    node_paths=node_paths,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                    node_cum_scores=node_cum_scores,
                    source_parent_score=0.0,
                )
                return True
            return False

        if not self.combine_event_mode:
            for layer_idx in range(1, depth + 1):
                if not frontier:
                    break
                suffix_len = max(1, base_suffix_len - (layer_idx - 1))
                preview_record = None
                if scheduler_is_ready and self.scheduler_forced_layer_set is None:
                    preview_record = self._build_scheduler_record(
                        layer_idx=layer_idx,
                        depth=depth,
                        frontier=frontier,
                        node_paths=node_paths,
                        node_parents=node_parents,
                        node_cum_scores=node_cum_scores,
                        node_children=node_children,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                    )
                    preview_record.update(
                        self.scheduler.build_decision(
                            preview_record,
                            depth,
                        )
                    )

                if scheduler_active:
                    if self.scheduler_forced_layer_set is not None:
                        do_combine = bool(layer_idx in self.scheduler_forced_layer_set)
                    else:
                        do_combine = bool(
                            preview_record is not None
                            and bool(preview_record.get("scheduler_should_call", False))
                        )
                else:
                    do_combine = layer_idx in self.combine_layer_set
                if preview_record is not None:
                    preview_record["scheduled_call"] = bool(do_combine)
                if do_combine and self.scheduler is not None:
                    self.scheduler.note_decode_step_call(layer_idx=layer_idx)

                layer_frontier_before = list(frontier)
                scheduler_elapsed_ms = (
                    float(preview_record.get("scheduler_elapsed_ms", preview_record.get("scheduler_elapsed_ms", 0.0)))
                    if preview_record is not None
                    else 0.0
                )
                scheduler_call_count = int(1 if (scheduler_active and preview_record is not None) else 0)
                ngram_elapsed_ms = 0.0
                ngram_call_count = 0
                before_summary = self._summarize_frontier_state(
                    frontier=layer_frontier_before,
                    node_paths=node_paths,
                    node_cum_scores=node_cum_scores,
                    node_children=node_children,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                )
                current_tree_record = self._build_scheduler_record(
                    layer_idx=layer_idx,
                    depth=depth,
                    frontier=layer_frontier_before,
                    node_paths=node_paths,
                    node_parents=node_parents,
                    node_cum_scores=node_cum_scores,
                    node_children=node_children,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                )
                layer_actual_record = {
                    "layer_idx": int(layer_idx),
                    "action": "call" if do_combine else "skip",
                    "remaining_layers": int(max(0, depth - layer_idx + 1)),
                    "frontier_size_before": int(before_summary["frontier_size"]),
                    "frontier_mean_depth_before": float(before_summary["frontier_mean_depth"]),
                    "frontier_max_depth_before": float(before_summary["frontier_max_depth"]),
                    "frontier_mean_score_before": float(before_summary["frontier_mean_score"]),
                    "frontier_score_p50_before": float(before_summary["frontier_score_p50"]),
                    "frontier_score_p90_before": float(before_summary["frontier_score_p90"]),
                    "frontier_finite_score_ratio_before": float(before_summary["frontier_finite_score_ratio"]),
                    "frontier_leaf_ratio_before": float(before_summary["frontier_leaf_ratio"]),
                    "parent_match_len_mean_before": float(before_summary["parent_match_len_mean"]),
                    "parent_match_len_p90_before": float(before_summary["parent_match_len_p90"]),
                    "parent_recent_pos_mean_before": float(before_summary["parent_recent_pos_mean"]),
                    "parent_recent_pos_span_before": float(before_summary["parent_recent_pos_span"]),
                    "tree_nodes_before": int(current_tree_record["tree_nodes_before"]),
                    "pre_reselect_candidate_size": float(
                        current_tree_record["pre_reselect_candidate_size"]
                    ),
                    "pre_reselect_cands": float(current_tree_record["pre_reselect_cands"]),
                    "pre_reselect_frontier_mean_depth": float(
                        current_tree_record["pre_reselect_frontier_mean_depth"]
                    ),
                    "pre_reselect_frontier_mean_score": float(
                        current_tree_record["pre_reselect_frontier_mean_score"]
                    ),
                    "pre_reselect_frontier_finite_score_ratio": float(
                        current_tree_record["pre_reselect_frontier_finite_score_ratio"]
                    ),
                    "pre_reselect_depth_delta": float(
                        current_tree_record["pre_reselect_depth_delta"]
                    ),
                    "pre_reselect_score_delta": float(
                        current_tree_record["pre_reselect_score_delta"]
                    ),
                    "pre_reselect_overlap_ratio": float(
                        current_tree_record["pre_reselect_overlap_ratio"]
                    ),
                    "pre_reselect_from_past_layer_ratio": float(
                        current_tree_record["pre_reselect_from_past_layer_ratio"]
                    ),
                    "collapse_debt": float(current_tree_record["collapse_debt"]),
                    "scheduler_call_count": int(scheduler_call_count),
                    "scheduler_elapsed_ms": float(scheduler_elapsed_ms),
                    "ngram_call_count": int(ngram_call_count),
                    "ngram_elapsed_ms": float(ngram_elapsed_ms),
                    "skip_source": (
                        "call" if do_combine else str(self.combine_skip_source)
                    ),
                }
                call_observer_record = {
                    "layer_idx": int(layer_idx),
                } if do_combine else None
                middle_record_start = len(getattr(self, "middle_forward_records", []))

                t_expand_start = time.perf_counter()
                if do_combine:
                    correction_loops.append(int(layer_idx))
                    self._active_combine_layer = int(layer_idx)
                    layer_child_candidates, new_leaf_indices = self._expand_layer_combine(
                        frontier=frontier,
                        context_prefix_tokens=context_prefix_tokens,
                        max_nodes=max_nodes,
                        node_tokens=node_tokens,
                        node_parents=node_parents,
                        node_children=node_children,
                        node_paths=node_paths,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                        node_cum_scores=node_cum_scores,
                        collect_time_stats=collect_time_stats,
                        profile=profile,
                        policy_record=call_observer_record,
                    )
                    self._active_combine_layer = None
                    terminated_parents_in_layer = 0
                else:
                    new_leaf_indices = []
                    layer_child_candidates, terminated_parents_in_layer = self._expand_layer_direct_match(
                        frontier=frontier,
                        suffix_len=suffix_len,
                        context_tokens=context_tokens,
                        context_prefix_tokens=context_prefix_tokens,
                        ngram_index=ngram_index,
                        max_nodes=max_nodes,
                        node_tokens=node_tokens,
                        node_parents=node_parents,
                        node_children=node_children,
                        node_paths=node_paths,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                        node_cum_scores=node_cum_scores,
                        collect_time_stats=collect_time_stats,
                        profile=profile,
                    )
                expand_cost = time.perf_counter() - t_expand_start
                if collect_time_stats and profile is not None:
                    if do_combine:
                        pass
                    else:
                        profile["expand_parents"] += expand_cost
                        profile["propose_total"] += expand_cost
                        if layer_idx == 1:
                            profile["layer1"] += expand_cost
                if not do_combine:
                    ngram_elapsed_ms = float(expand_cost * 1000.0)
                    ngram_call_count = 1

                if _apply_first_layer_fallback(layer_idx, terminated_parents_in_layer, frontier):
                    break

                t_prune_start = time.perf_counter() if collect_time_stats else None
                child_candidates = self._dedup_and_sort_child_candidates(layer_child_candidates)
                dedup_candidate_count = int(len(child_candidates))
                child_candidates = self._apply_layer_prune(
                    child_candidates,
                    layer_idx=layer_idx,
                    force_topk=False,
                )
                if collect_time_stats and profile is not None:
                    prune_cost = time.perf_counter() - t_prune_start
                    if do_combine:
                        profile["correction_prune"] += prune_cost
                        profile["correction_total"] += prune_cost
                    else:
                        profile["global_topk"] += prune_cost

                frontier = [c[0] for c in child_candidates]
                if (
                    self.trim_nonparent_new_leaves_after_combine
                    and do_combine
                    and (layer_idx + 1) <= depth
                    and (
                        scheduler_active
                        or ((layer_idx + 1) not in self.combine_layer_set)
                    )
                    and len(node_tokens) < max_nodes
                ):
                    t_trim_start = (
                        time.perf_counter() if (collect_time_stats and profile is not None) else None
                    )
                    (
                        node_tokens,
                        node_parents,
                        node_children,
                        node_paths,
                        node_match_len,
                        node_recent_pos,
                        node_cum_scores,
                        frontier,
                    ) = self._trim_new_leaves_not_in_frontier(
                        new_leaf_indices=new_leaf_indices,
                        frontier=frontier,
                        node_tokens=node_tokens,
                        node_parents=node_parents,
                        node_children=node_children,
                        node_paths=node_paths,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                        node_cum_scores=node_cum_scores,
                    )
                    if t_trim_start is not None:
                        trim_cost = time.perf_counter() - t_trim_start
                        profile["correction_trim"] += trim_cost
                        profile["correction_total"] += trim_cost
                if len(node_tokens) >= max_nodes:
                    pass

                after_summary = self._summarize_frontier_state(
                    frontier=frontier,
                    node_paths=node_paths,
                    node_cum_scores=node_cum_scores,
                    node_children=node_children,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                )
                layer_actual_record.update(
                    {
                        "raw_child_candidate_count": int(len(layer_child_candidates)),
                        "dedup_child_candidate_count": int(dedup_candidate_count),
                        "pruned_child_candidate_count": int(len(frontier)),
                        "terminated_parent_count": int(terminated_parents_in_layer),
                        "terminated_parent_ratio": (
                            float(terminated_parents_in_layer) / float(len(layer_frontier_before))
                            if len(layer_frontier_before) > 0 else 0.0
                        ),
                        "new_leaf_count": int(len(new_leaf_indices)),
                        "frontier_size_after": int(after_summary["frontier_size"]),
                        "frontier_mean_depth_after": float(after_summary["frontier_mean_depth"]),
                        "frontier_max_depth_after": float(after_summary["frontier_max_depth"]),
                        "frontier_mean_score_after": float(after_summary["frontier_mean_score"]),
                        "frontier_score_p50_after": float(after_summary["frontier_score_p50"]),
                        "frontier_score_p90_after": float(after_summary["frontier_score_p90"]),
                        "frontier_finite_score_ratio_after": float(after_summary["frontier_finite_score_ratio"]),
                        "frontier_leaf_ratio_after": float(after_summary["frontier_leaf_ratio"]),
                        "tree_node_count_after": int(len(node_tokens)),
                        "trim_new_leaf_count": int(
                            max(
                                0,
                                len(new_leaf_indices)
                                - len([x for x in new_leaf_indices if x in set(frontier)]),
                            )
                        ),
                    }
                )
                layer_actual_record["trimmed_leaf_ratio"] = (
                    float(layer_actual_record["trim_new_leaf_count"]) / float(layer_actual_record["new_leaf_count"])
                    if float(layer_actual_record["new_leaf_count"]) > 0.0 else 0.0
                )
                layer_actual_record["ngram_call_count"] = int(ngram_call_count)
                layer_actual_record["ngram_elapsed_ms"] = float(ngram_elapsed_ms)
                if do_combine:
                    layer_middle_records = [
                        dict(x)
                        for x in getattr(self, "middle_forward_records", [])[middle_record_start:]
                    ]
                    layer_actual_record["middle_call_count"] = int(len(layer_middle_records))
                    layer_actual_record["middle_elapsed_ms"] = float(
                        sum(float(x.get("elapsed_ms", 0.0)) for x in layer_middle_records)
                    )
                    if layer_middle_records:
                        last_middle = layer_middle_records[-1]
                        layer_actual_record["middle_prefix_len"] = int(last_middle.get("prefix_len", 0))
                        layer_actual_record["middle_tree_nodes"] = int(last_middle.get("tree_nodes", 0))
                        layer_actual_record["middle_forward_tokens"] = int(last_middle.get("forward_tokens", 0))
                    if call_observer_record is not None:
                        layer_actual_record.update(
                            {
                                "reselect_frontier_size_before": int(call_observer_record.get("reselect_frontier_size_before", 0)),
                                "reselect_frontier_size_after": int(call_observer_record.get("reselect_frontier_size_after", 0)),
                                "reselect_frontier_mean_depth_before": float(call_observer_record.get("reselect_frontier_mean_depth_before", 0.0)),
                                "reselect_frontier_mean_depth_after": float(call_observer_record.get("reselect_frontier_mean_depth_after", 0.0)),
                                "reselect_frontier_mean_score_before": float(call_observer_record.get("reselect_frontier_mean_score_before", 0.0)),
                                "reselect_frontier_mean_score_after": float(call_observer_record.get("reselect_frontier_mean_score_after", 0.0)),
                                "reselect_frontier_finite_score_ratio_before": float(
                                    call_observer_record.get(
                                        "reselect_frontier_finite_score_ratio_before", 0.0
                                    )
                                ),
                                "reselect_frontier_finite_score_ratio_after": float(
                                    call_observer_record.get(
                                        "reselect_frontier_finite_score_ratio_after", 0.0
                                    )
                                ),
                                "reselect_depth_delta": float(call_observer_record.get("reselect_depth_delta", 0.0)),
                                "reselect_score_delta": float(call_observer_record.get("reselect_score_delta", 0.0)),
                                "reselect_overlap_ratio": float(call_observer_record.get("reselect_overlap_ratio", 0.0)),
                                "reselect_from_past_layer_ratio": float(call_observer_record.get("reselect_from_past_layer_ratio", 0.0)),
                            }
                        )
                        call_record = dict(layer_actual_record)
                        actual_call_records.append(call_record)
                actual_layer_records.append(layer_actual_record)
                if len(node_tokens) >= max_nodes:
                    break
        else:
            # combine_layers=[-1,k]
            correction_count = 0
            pending_correction = False
            pending_event_frontier = []

            for layer_idx in range(1, depth + 1):
                if len(node_tokens) >= max_nodes:
                    break

                # combine
                do_correction = False
                correction_frontier = []
                if pending_correction and correction_count < int(self.combine_event_max):
                    do_correction = True
                    correction_frontier = list(pending_event_frontier)
                elif len(frontier) == 0 and correction_count < int(self.combine_event_max):
                    do_correction = True
                    correction_frontier = list(pending_event_frontier) if pending_event_frontier else list(frontier)

                if do_correction:
                    correction_loops.append(int(layer_idx))
                    t_prepare_frontier_start = (
                        time.perf_counter() if (collect_time_stats and profile is not None) else None
                    )
                    if not correction_frontier:
                        correction_frontier = list(frontier)
                    if not correction_frontier:
                        correction_frontier = self._collect_leaf_frontier(node_children)
                    if t_prepare_frontier_start is not None:
                        prepare_frontier_cost = time.perf_counter() - t_prepare_frontier_start
                        profile["correction_reselect_frontier"] += prepare_frontier_cost
                        profile["correction_total"] += prepare_frontier_cost
                    if not correction_frontier:
                        break

                    layer_frontier_before = list(correction_frontier)
                    before_summary = self._summarize_frontier_state(
                        frontier=layer_frontier_before,
                        node_paths=node_paths,
                        node_cum_scores=node_cum_scores,
                        node_children=node_children,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                    )
                    current_tree_record = self._build_scheduler_record(
                        layer_idx=layer_idx,
                        depth=depth,
                        frontier=layer_frontier_before,
                        node_paths=node_paths,
                        node_parents=node_parents,
                        node_cum_scores=node_cum_scores,
                        node_children=node_children,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                    )
                    layer_actual_record = {
                        "layer_idx": int(layer_idx),
                        "action": "call",
                        "remaining_layers": int(max(0, depth - layer_idx + 1)),
                        "frontier_size_before": int(before_summary["frontier_size"]),
                        "frontier_mean_depth_before": float(before_summary["frontier_mean_depth"]),
                        "frontier_max_depth_before": float(before_summary["frontier_max_depth"]),
                        "frontier_mean_score_before": float(before_summary["frontier_mean_score"]),
                        "frontier_score_p50_before": float(before_summary["frontier_score_p50"]),
                        "frontier_score_p90_before": float(before_summary["frontier_score_p90"]),
                        "frontier_finite_score_ratio_before": float(before_summary["frontier_finite_score_ratio"]),
                        "frontier_leaf_ratio_before": float(before_summary["frontier_leaf_ratio"]),
                        "parent_match_len_mean_before": float(before_summary["parent_match_len_mean"]),
                        "parent_match_len_p90_before": float(before_summary["parent_match_len_p90"]),
                        "parent_recent_pos_mean_before": float(before_summary["parent_recent_pos_mean"]),
                        "parent_recent_pos_span_before": float(before_summary["parent_recent_pos_span"]),
                        "tree_nodes_before": int(current_tree_record["tree_nodes_before"]),
                        "pre_reselect_candidate_size": float(
                            current_tree_record["pre_reselect_candidate_size"]
                        ),
                        "pre_reselect_cands": float(current_tree_record["pre_reselect_cands"]),
                        "pre_reselect_frontier_mean_depth": float(
                            current_tree_record["pre_reselect_frontier_mean_depth"]
                        ),
                        "pre_reselect_frontier_mean_score": float(
                            current_tree_record["pre_reselect_frontier_mean_score"]
                        ),
                        "pre_reselect_frontier_finite_score_ratio": float(
                            current_tree_record["pre_reselect_frontier_finite_score_ratio"]
                        ),
                        "pre_reselect_depth_delta": float(
                            current_tree_record["pre_reselect_depth_delta"]
                        ),
                        "pre_reselect_score_delta": float(
                            current_tree_record["pre_reselect_score_delta"]
                        ),
                        "pre_reselect_overlap_ratio": float(
                            current_tree_record["pre_reselect_overlap_ratio"]
                        ),
                        "pre_reselect_from_past_layer_ratio": float(
                            current_tree_record["pre_reselect_from_past_layer_ratio"]
                        ),
                        "collapse_debt": float(current_tree_record["collapse_debt"]),
                        "scheduler_call_count": 0,
                        "scheduler_elapsed_ms": 0.0,
                        "ngram_call_count": 0,
                        "ngram_elapsed_ms": 0.0,
                    }
                    call_observer_record = {"layer_idx": int(layer_idx)}
                    middle_record_start = len(getattr(self, "middle_forward_records", []))

                    t_expand_start = time.perf_counter() if collect_time_stats else None
                    self._active_combine_layer = int(layer_idx)
                    layer_child_candidates, new_leaf_indices = self._expand_layer_combine(
                        frontier=correction_frontier,
                        context_prefix_tokens=context_prefix_tokens,
                        max_nodes=max_nodes,
                        node_tokens=node_tokens,
                        node_parents=node_parents,
                        node_children=node_children,
                        node_paths=node_paths,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                        node_cum_scores=node_cum_scores,
                        collect_time_stats=collect_time_stats,
                        profile=profile,
                        policy_record=call_observer_record,
                    )
                    self._active_combine_layer = None
                    if collect_time_stats and profile is not None:
                        expand_cost = time.perf_counter() - t_expand_start
                        pass

                    t_prune_start = time.perf_counter() if collect_time_stats else None
                    child_candidates = self._dedup_and_sort_child_candidates(layer_child_candidates)
                    dedup_candidate_count = int(len(child_candidates))
                    child_candidates = self._apply_layer_prune(
                        child_candidates,
                        layer_idx=layer_idx,
                        force_topk=False,
                    )
                    if collect_time_stats and profile is not None:
                        prune_cost = time.perf_counter() - t_prune_start
                        profile["correction_prune"] += prune_cost
                        profile["correction_total"] += prune_cost

                    frontier = [c[0] for c in child_candidates]
                    if (
                        self.trim_nonparent_new_leaves_after_combine
                        and (layer_idx + 1) <= depth
                        and len(frontier) > 0
                        and len(node_tokens) < max_nodes
                    ):
                        t_trim_start = (
                            time.perf_counter() if (collect_time_stats and profile is not None) else None
                        )
                        (
                            node_tokens,
                            node_parents,
                            node_children,
                            node_paths,
                            node_match_len,
                            node_recent_pos,
                            node_cum_scores,
                            frontier,
                        ) = self._trim_new_leaves_not_in_frontier(
                            new_leaf_indices=new_leaf_indices,
                            frontier=frontier,
                            node_tokens=node_tokens,
                            node_parents=node_parents,
                            node_children=node_children,
                            node_paths=node_paths,
                            node_match_len=node_match_len,
                            node_recent_pos=node_recent_pos,
                            node_cum_scores=node_cum_scores,
                        )
                        if t_trim_start is not None:
                            trim_cost = time.perf_counter() - t_trim_start
                            profile["correction_trim"] += trim_cost
                            profile["correction_total"] += trim_cost
                    after_summary = self._summarize_frontier_state(
                        frontier=frontier,
                        node_paths=node_paths,
                        node_cum_scores=node_cum_scores,
                        node_children=node_children,
                        node_match_len=node_match_len,
                        node_recent_pos=node_recent_pos,
                    )
                    layer_actual_record.update(
                        {
                            "raw_child_candidate_count": int(len(layer_child_candidates)),
                            "dedup_child_candidate_count": int(dedup_candidate_count),
                            "pruned_child_candidate_count": int(len(frontier)),
                            "terminated_parent_count": 0,
                            "terminated_parent_ratio": 0.0,
                            "new_leaf_count": int(len(new_leaf_indices)),
                            "frontier_size_after": int(after_summary["frontier_size"]),
                            "frontier_mean_depth_after": float(after_summary["frontier_mean_depth"]),
                            "frontier_max_depth_after": float(after_summary["frontier_max_depth"]),
                            "frontier_mean_score_after": float(after_summary["frontier_mean_score"]),
                            "frontier_score_p50_after": float(after_summary["frontier_score_p50"]),
                            "frontier_score_p90_after": float(after_summary["frontier_score_p90"]),
                            "frontier_finite_score_ratio_after": float(after_summary["frontier_finite_score_ratio"]),
                            "frontier_leaf_ratio_after": float(after_summary["frontier_leaf_ratio"]),
                            "tree_node_count_after": int(len(node_tokens)),
                            "trim_new_leaf_count": int(max(0, len(new_leaf_indices) - len([x for x in new_leaf_indices if x in set(frontier)]))),
                        }
                    )
                    layer_actual_record["trimmed_leaf_ratio"] = (
                        float(layer_actual_record["trim_new_leaf_count"]) / float(layer_actual_record["new_leaf_count"])
                        if float(layer_actual_record["new_leaf_count"]) > 0.0 else 0.0
                    )
                    layer_middle_records = [
                        dict(x)
                        for x in getattr(self, "middle_forward_records", [])[middle_record_start:]
                    ]
                    layer_actual_record["middle_call_count"] = int(len(layer_middle_records))
                    layer_actual_record["middle_elapsed_ms"] = float(
                        sum(float(x.get("elapsed_ms", 0.0)) for x in layer_middle_records)
                    )
                    if layer_middle_records:
                        last_middle = layer_middle_records[-1]
                        layer_actual_record["middle_prefix_len"] = int(last_middle.get("prefix_len", 0))
                        layer_actual_record["middle_tree_nodes"] = int(last_middle.get("tree_nodes", 0))
                        layer_actual_record["middle_forward_tokens"] = int(last_middle.get("forward_tokens", 0))
                    layer_actual_record.update(
                        {
                            "reselect_frontier_size_before": int(call_observer_record.get("reselect_frontier_size_before", 0)),
                            "reselect_frontier_size_after": int(call_observer_record.get("reselect_frontier_size_after", 0)),
                            "reselect_frontier_mean_depth_before": float(call_observer_record.get("reselect_frontier_mean_depth_before", 0.0)),
                            "reselect_frontier_mean_depth_after": float(call_observer_record.get("reselect_frontier_mean_depth_after", 0.0)),
                            "reselect_frontier_mean_score_before": float(call_observer_record.get("reselect_frontier_mean_score_before", 0.0)),
                            "reselect_frontier_mean_score_after": float(call_observer_record.get("reselect_frontier_mean_score_after", 0.0)),
                            "reselect_frontier_finite_score_ratio_before": float(
                                call_observer_record.get(
                                    "reselect_frontier_finite_score_ratio_before", 0.0
                                )
                            ),
                            "reselect_frontier_finite_score_ratio_after": float(
                                call_observer_record.get(
                                    "reselect_frontier_finite_score_ratio_after", 0.0
                                )
                            ),
                            "reselect_depth_delta": float(call_observer_record.get("reselect_depth_delta", 0.0)),
                            "reselect_score_delta": float(call_observer_record.get("reselect_score_delta", 0.0)),
                            "reselect_overlap_ratio": float(call_observer_record.get("reselect_overlap_ratio", 0.0)),
                            "reselect_from_past_layer_ratio": float(call_observer_record.get("reselect_from_past_layer_ratio", 0.0)),
                        }
                    )
                    actual_layer_records.append(layer_actual_record)
                    actual_call_records.append(dict(layer_actual_record))
                    pending_correction = False
                    pending_event_frontier = []
                    correction_count += 1
                    continue

                # normal direct
                if not frontier:
                    break

                current_parents = list(frontier)
                suffix_len = max(1, base_suffix_len - (layer_idx - 1))
                before_summary = self._summarize_frontier_state(
                    frontier=current_parents,
                    node_paths=node_paths,
                    node_cum_scores=node_cum_scores,
                    node_children=node_children,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                )
                current_tree_record = self._build_scheduler_record(
                    layer_idx=layer_idx,
                    depth=depth,
                    frontier=current_parents,
                    node_paths=node_paths,
                    node_parents=node_parents,
                    node_cum_scores=node_cum_scores,
                    node_children=node_children,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                )
                t_expand_start = time.perf_counter()
                layer_child_candidates, _ = self._expand_layer_direct_match(
                    frontier=frontier,
                    suffix_len=suffix_len,
                    context_tokens=context_tokens,
                    context_prefix_tokens=context_prefix_tokens,
                    ngram_index=ngram_index,
                    max_nodes=max_nodes,
                    node_tokens=node_tokens,
                    node_parents=node_parents,
                    node_children=node_children,
                    node_paths=node_paths,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                    node_cum_scores=node_cum_scores,
                    collect_time_stats=collect_time_stats,
                    profile=profile,
                )
                ngram_elapsed_ms = 0.0
                ngram_call_count = 1
                expand_cost = time.perf_counter() - t_expand_start
                ngram_elapsed_ms = float(expand_cost * 1000.0)
                if collect_time_stats and profile is not None:
                    profile["expand_parents"] += expand_cost
                    profile["propose_total"] += expand_cost
                    if layer_idx == 1:
                        profile["layer1"] += expand_cost

                t_prune_start = time.perf_counter() if collect_time_stats else None
                child_candidates = self._dedup_and_sort_child_candidates(layer_child_candidates)
                dedup_candidate_count = int(len(child_candidates))
                child_candidates = self._apply_layer_prune(
                    child_candidates,
                    layer_idx=layer_idx,
                    force_topk=False,
                )
                if collect_time_stats and profile is not None:
                    profile["global_topk"] += time.perf_counter() - t_prune_start

                frontier = [c[0] for c in child_candidates]
                if not frontier and correction_count < int(self.combine_event_max):
                    pending_correction = True
                    pending_event_frontier = current_parents
                else:
                    pending_correction = False
                    pending_event_frontier = []
                after_summary = self._summarize_frontier_state(
                    frontier=frontier,
                    node_paths=node_paths,
                    node_cum_scores=node_cum_scores,
                    node_children=node_children,
                    node_match_len=node_match_len,
                    node_recent_pos=node_recent_pos,
                )
                actual_layer_records.append(
                    {
                        "layer_idx": int(layer_idx),
                        "action": "skip",
                        "remaining_layers": int(max(0, depth - layer_idx + 1)),
                        "frontier_size_before": int(before_summary["frontier_size"]),
                        "frontier_mean_depth_before": float(before_summary["frontier_mean_depth"]),
                        "frontier_max_depth_before": float(before_summary["frontier_max_depth"]),
                        "frontier_mean_score_before": float(before_summary["frontier_mean_score"]),
                        "frontier_score_p50_before": float(before_summary["frontier_score_p50"]),
                        "frontier_score_p90_before": float(before_summary["frontier_score_p90"]),
                        "frontier_finite_score_ratio_before": float(before_summary["frontier_finite_score_ratio"]),
                        "frontier_leaf_ratio_before": float(before_summary["frontier_leaf_ratio"]),
                        "parent_match_len_mean_before": float(before_summary["parent_match_len_mean"]),
                        "parent_match_len_p90_before": float(before_summary["parent_match_len_p90"]),
                        "parent_recent_pos_mean_before": float(before_summary["parent_recent_pos_mean"]),
                        "parent_recent_pos_span_before": float(before_summary["parent_recent_pos_span"]),
                        "tree_nodes_before": int(current_tree_record["tree_nodes_before"]),
                        "pre_reselect_candidate_size": float(
                            current_tree_record["pre_reselect_candidate_size"]
                        ),
                        "pre_reselect_cands": float(current_tree_record["pre_reselect_cands"]),
                        "pre_reselect_frontier_mean_depth": float(
                            current_tree_record["pre_reselect_frontier_mean_depth"]
                        ),
                        "pre_reselect_frontier_mean_score": float(
                            current_tree_record["pre_reselect_frontier_mean_score"]
                        ),
                        "pre_reselect_frontier_finite_score_ratio": float(
                            current_tree_record["pre_reselect_frontier_finite_score_ratio"]
                        ),
                        "pre_reselect_depth_delta": float(
                            current_tree_record["pre_reselect_depth_delta"]
                        ),
                        "pre_reselect_score_delta": float(
                            current_tree_record["pre_reselect_score_delta"]
                        ),
                        "pre_reselect_overlap_ratio": float(
                            current_tree_record["pre_reselect_overlap_ratio"]
                        ),
                        "pre_reselect_from_past_layer_ratio": float(
                            current_tree_record["pre_reselect_from_past_layer_ratio"]
                        ),
                        "collapse_debt": float(current_tree_record["collapse_debt"]),
                        "scheduler_call_count": 0,
                        "scheduler_elapsed_ms": 0.0,
                        "ngram_call_count": int(ngram_call_count),
                        "ngram_elapsed_ms": float(ngram_elapsed_ms),
                        "raw_child_candidate_count": int(len(layer_child_candidates)),
                        "dedup_child_candidate_count": int(dedup_candidate_count),
                        "pruned_child_candidate_count": int(len(frontier)),
                        "terminated_parent_count": 0,
                        "terminated_parent_ratio": 0.0,
                        "new_leaf_count": 0,
                        "frontier_size_after": int(after_summary["frontier_size"]),
                        "frontier_mean_depth_after": float(after_summary["frontier_mean_depth"]),
                        "frontier_max_depth_after": float(after_summary["frontier_max_depth"]),
                        "frontier_mean_score_after": float(after_summary["frontier_mean_score"]),
                        "frontier_score_p50_after": float(after_summary["frontier_score_p50"]),
                        "frontier_score_p90_after": float(after_summary["frontier_score_p90"]),
                        "frontier_finite_score_ratio_after": float(after_summary["frontier_finite_score_ratio"]),
                        "frontier_leaf_ratio_after": float(after_summary["frontier_leaf_ratio"]),
                        "tree_node_count_after": int(len(node_tokens)),
                        "trim_new_leaf_count": 0,
                        "trimmed_leaf_ratio": 0.0,
                    }
                )

        t_post_start = time.perf_counter() if collect_time_stats else None
        outputs = self._build_tree_outputs(
            node_tokens=node_tokens,
            node_parents=node_parents,
            node_children=node_children,
            node_match_len=node_match_len,
            node_cum_scores=node_cum_scores,
            device=input_ids_device,
            logits_processor=logits_processor,
            collect_time_stats=collect_time_stats,
            profile=profile,
        )
        if collect_time_stats and profile is not None:
            profile["postprocess"] += time.perf_counter() - t_post_start
            self.last_profile_stats = profile
        else:
            self.last_profile_stats = None
        self.last_middle_correction_loops = correction_loops
        self.last_actual_layer_records = actual_layer_records
        self.all_actual_layer_records.extend(actual_layer_records)
        self.last_actual_call_records = actual_call_records
        self.all_actual_call_records.extend(actual_call_records)
        return outputs

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor, collect_time_stats=False):
        if self.combine_enabled:
            return self._topK_genrate_combine(
                input_ids=input_ids,
                logits_processor=logits_processor,
                collect_time_stats=collect_time_stats,
            )
        self.last_middle_correction_loops = []
        self.last_actual_layer_records = []
        self.all_actual_layer_records = []
        self.last_actual_call_records = []
        self.all_actual_call_records = []
        return super().topK_genrate(
            hidden_states=hidden_states,
            input_ids=input_ids,
            head=head,
            logits_processor=logits_processor,
            collect_time_stats=collect_time_stats,
        )
