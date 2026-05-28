"""N-gram small-drafter tree generator."""
import torch
import time
from collections import Counter


class NgramDraftGenerator:
    """Training-free n-gram small drafter with an EAGLE-compatible tree API."""

    def __init__(
        self,
        draft_model_path: str = None,
        draft_strategy: str = "prefill_topw",
        total_tokens: int = 60,
        depth: int = 5,
        top_k: int = 10,
        use_adaptive_config: bool = False,
        layer_config: dict = None,
        max_matching_ngram_size: int = 3,
        stop_on_relax_zero: bool = True,
        prune_to_total_tokens: bool = True,
        legacy_total_token_stop: bool = False,
        device_map: str = "auto",
        dtype=torch.float16,
    ):
        """Create an n-gram draft generator."""

        self.use_adaptive_config = use_adaptive_config

        if use_adaptive_config:
            if layer_config is None:
                layer_config = {}
                layer_config["1"] = [0, top_k]
                for i in range(2, depth + 2):
                    layer_config[str(i)] = [top_k, top_k]

            self.layer_config = layer_config
            self.depth = len(layer_config) - 1

            max_candidates = max(cfg[0] for cfg in layer_config.values())
            max_global = max(cfg[1] for cfg in layer_config.values())
            self.max_topk = max(max_candidates, max_global)
        else:
            self.layer_config = None
            self.depth = depth
            self.max_topk = top_k

        self.total_tokens = total_tokens
        self.top_k = top_k
        self.max_matching_ngram_size = max_matching_ngram_size
        self.stop_on_relax_zero = bool(stop_on_relax_zero)
        self.prune_to_total_tokens = bool(prune_to_total_tokens)
        self.legacy_total_token_stop = bool(legacy_total_token_stop)
        self.draft_strategy = (draft_strategy or "prefill_topw").lower()

        self.lm_head = None

        self.tree_mask_init = torch.eye(self.max_topk, dtype=torch.float32)[None, None]
        self.position_ids = torch.zeros(self.max_topk, dtype=torch.long)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

        # ngram/prefill
        self.prefill_reference_tokens = None
        self.prefill_reference_list = None
        self.prefill_top_tokens = None
        self.prefill_top_logprobs = None
        self.prompt_tokens = None
        self.ngram_last_pos = None  # Dict[n, Dict[tuple(token_ids), last_start]]
        self.relax_anchor_pos = None
        self.last_profile_stats = None
        self.last_node_match_ngram = None
        self.last_middle_correction_loops = []

    def get_kv_tree_token_reserve(self) -> int:
        reserve = int(max(0, int(self.total_tokens)))
        if self.legacy_total_token_stop:
            return reserve
        if self.prune_to_total_tokens:
            return reserve

        strategy = (self.draft_strategy or "").lower()
        combine_enabled = bool(getattr(self, "combine_enabled", False))
        if strategy == "direct_match" or combine_enabled:
            depth = int(max(1, int(self.depth)))
            top_k = int(max(1, int(self.top_k)))
            max_nodes = max(
                int(self.total_tokens) + 1,
                1 + top_k + depth * top_k * top_k,
            )
            reserve = max(reserve, int(max_nodes - 1))
        return reserve

    def _propose_by_strategy(
        self,
        sequence_tokens,
        width: int,
        device: torch.device = None,
        collect_time_stats: bool = False,
        profile: dict = None,
    ):


        if self.draft_strategy == "prefill_topw":
            return self._propose_candidates(
                sequence_tokens,
                width,
                device=device,
                collect_time_stats=collect_time_stats,
                profile=profile,
            )
        raise ValueError(
            f"Unsupported draft_strategy={self.draft_strategy!r}. "
            "Proposer path only supports 'prefill_topw'. "
            "Use topK_genrate() direct_match branch for direct matching."
        )

    def reset_kv(self):
        self.prefill_reference_tokens = None
        self.prefill_reference_list = None
        self.prefill_top_tokens = None
        self.prefill_top_logprobs = None
        self.prompt_tokens = None
        self.ngram_last_pos = None
        self.relax_anchor_pos = None
        self.tree_mask = None
        self.last_node_match_ngram = None
        self.last_middle_correction_loops = []

    def set_prefill_cache(
        self,
        reference_tokens: torch.Tensor,
        top_tokens: torch.Tensor,
        top_logprobs: torch.Tensor,
        prompt_tokens: torch.Tensor = None,
    ):


        if reference_tokens.dim() != 1:
            reference_tokens = reference_tokens.view(-1)
        if top_tokens.dim() != 2 or top_logprobs.dim() != 2:
            raise ValueError("top_tokens/top_logprobs must be 2D tensors [N, W]")
        if top_tokens.shape != top_logprobs.shape:
            raise ValueError("top_tokens and top_logprobs must have the same shape")
        if reference_tokens.shape[0] != top_tokens.shape[0]:
            raise ValueError("reference_tokens length must match top_tokens first dimension")

        self.prefill_reference_tokens = reference_tokens.detach().clone()
        self.prefill_reference_list = self.prefill_reference_tokens.detach().cpu().tolist()
        self.prefill_top_tokens = top_tokens.detach().clone()
        self.prefill_top_logprobs = top_logprobs.detach().clone().float()
        self.prompt_tokens = (
            prompt_tokens.detach().clone().view(-1)
            if prompt_tokens is not None
            else self.prefill_reference_tokens.clone()
        )
        self._rebuild_ngram_index()
        self.relax_anchor_pos = self._select_relax_anchor()

    def append_prefill_cache(
        self,
        new_tokens: torch.Tensor,
        new_top_tokens: torch.Tensor,
        new_top_logprobs: torch.Tensor,
    ):

        if new_tokens is None or new_top_tokens is None or new_top_logprobs is None:
            return

        if new_tokens.dim() != 1:
            new_tokens = new_tokens.view(-1)
        if new_top_tokens.dim() != 2 or new_top_logprobs.dim() != 2:
            raise ValueError("new_top_tokens/new_top_logprobs must be 2D tensors [M, W]")
        if new_top_tokens.shape != new_top_logprobs.shape:
            raise ValueError("new_top_tokens and new_top_logprobs must have the same shape")
        if new_tokens.shape[0] != new_top_tokens.shape[0]:
            raise ValueError("new_tokens length must match new_top_tokens first dimension")

        if self.prefill_reference_tokens is None:
            self.set_prefill_cache(
                new_tokens,
                new_top_tokens,
                new_top_logprobs,
                prompt_tokens=new_tokens,
            )
            return

        device = self.prefill_reference_tokens.device
        old_len = int(self.prefill_reference_tokens.shape[0])
        self.prefill_reference_tokens = torch.cat(
            (self.prefill_reference_tokens, new_tokens.to(device)), dim=0
        )
        if self.prefill_reference_list is None:
            self.prefill_reference_list = self.prefill_reference_tokens.detach().cpu().tolist()
            self._rebuild_ngram_index()
        else:
            self.prefill_reference_list.extend(new_tokens.detach().cpu().tolist())
            self._append_ngram_index(old_len)
        self.prefill_top_tokens = torch.cat(
            (self.prefill_top_tokens, new_top_tokens.to(self.prefill_top_tokens.device)), dim=0
        )
        self.prefill_top_logprobs = torch.cat(
            (self.prefill_top_logprobs, new_top_logprobs.to(self.prefill_top_logprobs.device).float()), dim=0
        )

    def reset(self):
        self.tree_mask = None

    def _create_empty_profile(self):
        return {
            "propose_total": 0.0,
            "find_anchor": 0.0,
            "get_anchor_candidates": 0.0,
            "relax": 0.0,
            "layer1": 0.0,
            "expand_parents": 0.0,
            "global_topk": 0.0,
            "update_paths_mask": 0.0,
            "budget_prune": 0.0,
            "medium_forward": 0.0,
            "postprocess": 0.0,
        }

    def _rebuild_ngram_index(self):
        if self.prefill_reference_list is None:
            self.ngram_last_pos = None
            return
        ref = self.prefill_reference_list
        ref_len = len(ref)
        if ref_len == 0:
            self.ngram_last_pos = {}
            return

        max_n = min(int(self.max_matching_ngram_size), ref_len)
        index = {n: {} for n in range(1, max_n + 1)}
        for n in range(1, max_n + 1):
            d = index[n]
            end = ref_len - n + 1
            for start in range(end):
                d[tuple(ref[start:start + n])] = start
        self.ngram_last_pos = index

    def _append_ngram_index(self, old_len: int):
        if self.prefill_reference_list is None:
            self.ngram_last_pos = None
            return
        ref = self.prefill_reference_list
        new_len = len(ref)
        if new_len == 0:
            self.ngram_last_pos = {}
            return
        if self.ngram_last_pos is None:
            self._rebuild_ngram_index()
            return

        max_n = min(int(self.max_matching_ngram_size), new_len)
        for n in range(1, max_n + 1):
            d = self.ngram_last_pos.setdefault(n, {})
            start_from = max(0, int(old_len) - n + 1)
            end = new_len - n + 1
            for start in range(start_from, end):
                d[tuple(ref[start:start + n])] = start

    def _select_relax_anchor(self):
        if self.prompt_tokens is None or self.prompt_tokens.numel() == 0:
            return None
        if self.prefill_top_tokens is None or self.prefill_top_tokens.shape[0] == 0:
            return None

        prompt_len = min(self.prompt_tokens.numel(), self.prefill_top_tokens.shape[0])
        if prompt_len <= 0:
            return None

        prompt_tokens = self.prompt_tokens[:prompt_len]
        uniq, counts = torch.unique(prompt_tokens, return_counts=True)
        if uniq.numel() == 0:
            return 0

        most_freq_token = uniq[torch.argmax(counts)]
        positions = (prompt_tokens == most_freq_token).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            return int(prompt_len - 1)
        return int(positions[-1].item())

    def _find_anchor(self, sequence_tokens, ngram_size: int):
        if self.prefill_reference_tokens is None or self.prefill_reference_tokens.numel() == 0:
            return None, 0
        if self.prefill_reference_list is None:
            return None, 0

        if torch.is_tensor(sequence_tokens):
            seq_list = sequence_tokens.view(-1).detach().cpu().tolist()
        elif isinstance(sequence_tokens, list):
            seq_list = sequence_tokens
        else:
            seq_list = list(sequence_tokens)

        ref_len = len(self.prefill_reference_list)
        n = min(int(ngram_size), len(seq_list), ref_len)
        if n <= 0:
            return None, 0

        d = None if self.ngram_last_pos is None else self.ngram_last_pos.get(n, None)
        if d is None:
            return None, 0

        start = d.get(tuple(seq_list[-n:]), None)
        if start is None:
            return None, 0

        anchor_pos = start + n - 1
        if anchor_pos < self.prefill_top_tokens.shape[0]:
            return anchor_pos, n

        return None, 0

    def _get_anchor_candidates(self, anchor_pos: int, width: int, device: torch.device):
        if width <= 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
            )

        if self.prefill_top_tokens is None or self.prefill_top_tokens.shape[0] == 0:
            return (
                torch.zeros((width,), dtype=torch.long, device=device),
                torch.zeros((width,), dtype=torch.float32, device=device),
            )

        anchor_pos = max(0, min(int(anchor_pos), self.prefill_top_tokens.shape[0] - 1))
        avail = self.prefill_top_tokens.shape[1]
        take = min(width, avail)

        tokens = self.prefill_top_tokens[anchor_pos, :take].to(device)
        logprobs = self.prefill_top_logprobs[anchor_pos, :take].to(device)

        if take < width:
            pad = width - take
            tokens = torch.cat((tokens, tokens[:1].repeat(pad)), dim=0)
            logprobs = torch.cat((logprobs, logprobs[:1].repeat(pad)), dim=0)

        return tokens, logprobs

    def _propose_candidates(
        self,
        sequence_tokens,
        width: int,
        device: torch.device = None,
        collect_time_stats: bool = False,
        profile: dict = None,
    ):


        t_propose_start = time.perf_counter() if collect_time_stats else None
        if device is None:
            if torch.is_tensor(sequence_tokens):
                device = sequence_tokens.device
            elif self.prefill_top_tokens is not None:
                device = self.prefill_top_tokens.device
            else:
                device = torch.device("cpu")
        if width <= 0:
            return (
                torch.empty((0,), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float32, device=device),
                None,
                0,
            )

        if torch.is_tensor(sequence_tokens):
            seq_len = int(sequence_tokens.numel())
        else:
            seq_len = len(sequence_tokens)
        local_ngram_size = min(self.max_matching_ngram_size, seq_len)

        while local_ngram_size > 0:
            if collect_time_stats:
                t_find_start = time.perf_counter()
            anchor_pos, match_len = self._find_anchor(sequence_tokens, local_ngram_size)
            if collect_time_stats and profile is not None:
                profile["find_anchor"] += time.perf_counter() - t_find_start
            if anchor_pos is not None:
                if collect_time_stats:
                    t_get_start = time.perf_counter()
                tokens, logprobs = self._get_anchor_candidates(anchor_pos, width, device)
                if collect_time_stats and profile is not None:
                    profile["get_anchor_candidates"] += time.perf_counter() - t_get_start
                    profile["propose_total"] += time.perf_counter() - t_propose_start
                return tokens, logprobs, anchor_pos, match_len
            local_ngram_size -= 1

        if self.relax_anchor_pos is None:
            self.relax_anchor_pos = self._select_relax_anchor()

        if self.relax_anchor_pos is not None:
            if collect_time_stats:
                t_relax_start = time.perf_counter()
            tokens, logprobs = self._get_anchor_candidates(
                self.relax_anchor_pos, width, device
            )
            if collect_time_stats and profile is not None:
                profile["relax"] += time.perf_counter() - t_relax_start
                profile["propose_total"] += time.perf_counter() - t_propose_start
            return tokens, logprobs, self.relax_anchor_pos, 0

        if torch.is_tensor(sequence_tokens):
            token_id = int(sequence_tokens.view(-1)[-1].item()) if sequence_tokens.numel() > 0 else 0
        else:
            token_id = int(sequence_tokens[-1]) if len(sequence_tokens) > 0 else 0
        tokens = torch.full((width,), token_id, dtype=torch.long, device=device)
        logprobs = torch.zeros((width,), dtype=torch.float32, device=device)
        if collect_time_stats and profile is not None:
            profile["relax"] += time.perf_counter() - t_propose_start
            profile["propose_total"] += time.perf_counter() - t_propose_start
        return tokens, logprobs, None, 0

    def _build_context_ngram_index(self, context_tokens, max_n: int):
        index = {}
        n_max = min(max(1, int(max_n)), len(context_tokens))
        for n in range(1, n_max + 1):
            d = {}
            end = len(context_tokens) - n + 1
            for start in range(end):
                key = tuple(context_tokens[start:start + n])
                d.setdefault(key, []).append(start)
            index[n] = d
        return index

    def _get_relax_token_order(self, context_tokens):
        if not context_tokens:
            return []

        counter = Counter(context_tokens)
        last_pos = {}
        for idx, tok in enumerate(context_tokens):
            last_pos[tok] = idx

        order = sorted(counter.keys(), key=lambda t: (-counter[t], -last_pos[t], int(t)))
        seen = set(order)

        if self.prefill_top_tokens is not None and self.prefill_top_tokens.shape[0] > 0:
            if self.relax_anchor_pos is None:
                self.relax_anchor_pos = self._select_relax_anchor()
            if self.relax_anchor_pos is not None:
                anchor = max(0, min(int(self.relax_anchor_pos), int(self.prefill_top_tokens.shape[0]) - 1))
                extra = self.prefill_top_tokens[anchor].detach().cpu().tolist()
                for tok in extra:
                    if tok not in seen:
                        seen.add(tok)
                        order.append(int(tok))
        return order

    def _build_relax_suffix(self, token_id: int, context_tokens, suffix_len: int):
        suffix_len = max(1, int(suffix_len))
        last_pos = None
        for i in range(len(context_tokens) - 1, -1, -1):
            if int(context_tokens[i]) == int(token_id):
                last_pos = i
                break

        if last_pos is None:
            return tuple([int(token_id)] * suffix_len), None

        chunk = list(context_tokens[last_pos:last_pos + suffix_len])
        if len(chunk) < suffix_len:
            chunk.extend([int(token_id)] * (suffix_len - len(chunk)))
        return tuple(int(x) for x in chunk), last_pos

    def _estimate_token_logprob_from_cache(self, context_pos: int, token_id: int):
        if self.prefill_top_tokens is None or self.prefill_top_logprobs is None:
            return 0.0
        if context_pos is None or context_pos < 0:
            return 0.0
        if context_pos >= int(self.prefill_top_tokens.shape[0]):
            return 0.0

        row_tokens = self.prefill_top_tokens[int(context_pos)]
        row_logprobs = self.prefill_top_logprobs[int(context_pos)].float()
        hit = (row_tokens == int(token_id)).nonzero(as_tuple=True)[0]
        if hit.numel() > 0:
            return float(row_logprobs[int(hit[0])].item())
        if row_logprobs.numel() > 0:
            return float(row_logprobs[-1].item()) - 5.0
        return 0.0

    def _path_exists(self, parent_idx: int, suffix_tokens, node_children):
        node = int(parent_idx)
        for tok in suffix_tokens:
            child = node_children[node].get(int(tok))
            if child is None:
                return False
            node = child
        return True

    def _add_suffix_path(
        self,
        parent_idx: int,
        suffix_tokens,
        match_len: int,
        child_logprob: float,
        child_pos: int,
        max_nodes: int,
        node_tokens,
        node_parents,
        node_children,
        node_paths,
        node_match_len,
        node_recent_pos,
        node_cum_scores,
        source_parent_score: float,
    ):


        if not suffix_tokens:
            return None

        node = int(parent_idx)
        direct_child = None
        tokens_to_expand = tuple(int(x) for x in suffix_tokens)

        for i, tok in enumerate(tokens_to_expand):
            tok = int(tok)
            child = node_children[node].get(tok)
            if child is None:
                if len(node_tokens) >= int(max_nodes):
                    break
                child = len(node_tokens)
                node_tokens.append(tok)
                node_parents.append(node)
                node_children.append({})
                node_paths.append(node_paths[node] + [tok])
                node_match_len.append(-1)
                node_recent_pos.append(-1)
                node_cum_scores.append(float("-inf"))
                node_children[node][tok] = child

            node = child
            if i == 0:
                direct_child = node
                node_match_len[node] = max(int(node_match_len[node]), int(match_len))
                node_recent_pos[node] = max(int(node_recent_pos[node]), int(child_pos))
                cand_score = float(source_parent_score) + float(child_logprob)
                if node_cum_scores[node] == float("-inf"):
                    node_cum_scores[node] = cand_score
                else:
                    node_cum_scores[node] = max(float(node_cum_scores[node]), cand_score)
        return direct_child

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
        parent_path = node_paths[int(parent_idx)]  # [root, ..., parent]
        window_src = list(context_prefix_tokens) + list(parent_path)
        max_n = min(int(self.max_matching_ngram_size), len(window_src))

        existing_children = set(int(t) for t in node_children[int(parent_idx)].keys())
        coverage = set(existing_children)
        suffix_info = {}  # suffix -> (match_len, child_logprob, child_pos)
        child_match_len_local = {}
        child_recent_pos_local = {}
        hit_relax_zero = False

        for n in range(max_n, 0, -1):
            query = tuple(window_src[-n:])
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

        # n=0 relax
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

    def _build_tree_outputs(
        self,
        node_tokens,
        node_parents,
        node_children,
        node_match_len,
        device: torch.device,
        logits_processor,
        node_cum_scores=None,
        collect_time_stats: bool = False,
        profile: dict = None,
    ):

        final_max_nodes = int(self.total_tokens) + 1
        if (
            (not self.legacy_total_token_stop)
            and self.prune_to_total_tokens
            and len(node_tokens) > final_max_nodes
        ):
            t_budget_start = time.perf_counter() if (collect_time_stats and profile is not None) else None
            (
                node_tokens,
                node_parents,
                node_children,
                node_match_len,
                node_cum_scores,
            ) = self._prune_tree_to_budget(
                node_tokens=node_tokens,
                node_parents=node_parents,
                node_children=node_children,
                node_match_len=node_match_len,
                node_cum_scores=node_cum_scores,
                max_nodes=final_max_nodes,
            )
            if t_budget_start is not None:
                profile["budget_prune"] += time.perf_counter() - t_budget_start
        node_num = len(node_tokens)
        if node_num <= 0:
            raise RuntimeError("direct_match produced an empty tree")

        tree_mask = torch.eye(node_num).bool()
        tree_mask[:, 0] = True
        for i in range(1, node_num):
            p = int(node_parents[i])
            if p >= 0:
                tree_mask[i].add_(tree_mask[p])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1
        leaves = [i for i in range(node_num) if len(node_children[i]) == 0]
        if not leaves:
            leaves = [0]

        max_depth = int(tree_position_ids.max().item()) + 1
        retrieve_indices = []
        for leaf in leaves:
            path = []
            cur = int(leaf)
            while cur >= 0:
                path.append(cur)
                cur = int(node_parents[cur]) if cur < len(node_parents) else -1
            path.reverse()
            if len(path) < max_depth:
                path = path + ([-1] * (max_depth - len(path)))
            retrieve_indices.append(path)

        if logits_processor is not None:
            maxitem = node_num + 5

            def custom_sort(lst):
                return [x if x >= 0 else maxitem for x in lst]

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=device)
        draft_tokens = torch.tensor(node_tokens, dtype=torch.long, device=device).unsqueeze(0)
        tree_mask = tree_mask.float().to(device)[None, None]
        self.tree_mask = tree_mask
        tree_position_ids = tree_position_ids.to(device)
        self.last_node_match_ngram = torch.tensor(node_match_len, dtype=torch.long)
        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    def _prune_tree_to_budget(
        self,
        node_tokens,
        node_parents,
        node_children,
        node_match_len,
        node_cum_scores,
        max_nodes: int,
    ):


        old_n = len(node_tokens)
        max_nodes = max(1, int(max_nodes))
        if old_n <= max_nodes:
            return node_tokens, node_parents, node_children, node_match_len, node_cum_scores

        if node_cum_scores is None or len(node_cum_scores) != old_n:
            scores = [0.0] + [float("-inf")] * (old_n - 1)
        else:
            scores = [float(x) for x in node_cum_scores]
            scores[0] = 0.0

        depths = [0] * old_n
        for idx in range(1, old_n):
            p = int(node_parents[idx])
            if 0 <= p < old_n:
                depths[idx] = depths[p] + 1

        ranked_nodes = sorted(
            range(1, old_n),
            key=lambda i: (-float(scores[i]), int(depths[i]), int(i)),
        )

        keep = {0}
        for idx in ranked_nodes:
            if idx in keep:
                continue
            chain = []
            cur = int(idx)
            valid = True
            while cur not in keep:
                if cur < 0 or cur >= old_n:
                    valid = False
                    break
                chain.append(cur)
                parent = int(node_parents[cur])
                if parent == cur:
                    valid = False
                    break
                cur = parent
            if not valid:
                continue
            if len(keep) + len(chain) > max_nodes:
                continue
            keep.update(chain)
            if len(keep) >= max_nodes:
                break

        keep_old = sorted(keep)
        old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_old)}

        new_tokens = [node_tokens[i] for i in keep_old]
        new_match = [node_match_len[i] for i in keep_old]
        new_scores = [scores[i] for i in keep_old]

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

        return new_tokens, new_parents, rebuilt_children, new_match, new_scores

    @torch.no_grad()
    def _topK_genrate_direct_match(self, input_ids, logits_processor, collect_time_stats=False):
        if self.use_adaptive_config:
            raise ValueError("direct_match currently does not support adaptive layer_config")

        input_ids_device = input_ids.device
        context_tokens = input_ids[0].detach().cpu().tolist()
        if not context_tokens:
            raise RuntimeError("input_ids is empty")

        final_max_nodes = int(self.total_tokens) + 1
        depth = int(max(1, self.depth))
        top_k = int(max(1, self.top_k))
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
        node_cum_scores = [0.0]  # root score

        frontier = [0]
        profile = self._create_empty_profile() if collect_time_stats else None

        base_suffix_len = int(max(1, self.max_matching_ngram_size))
        for layer_idx in range(1, depth + 1):
            if not frontier:
                break

            suffix_len = max(1, base_suffix_len - (layer_idx - 1))
            layer_child_candidates = []  # (child_idx, match_len_local, score_local)
            terminated_parents_in_layer = 0

            t_layer_expand_start = time.perf_counter() if collect_time_stats else None
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
                    terminated_parents_in_layer += 1
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

            if collect_time_stats and profile is not None:
                expand_cost = time.perf_counter() - t_layer_expand_start
                profile["expand_parents"] += expand_cost
                profile["propose_total"] += expand_cost
                if layer_idx == 1:
                    profile["layer1"] += expand_cost

            # stop_on_relax_zero
            # root
            if (
                self.stop_on_relax_zero
                and layer_idx == 1
                and terminated_parents_in_layer == len(frontier)
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
                break

            dedup = {}
            for child_idx, mlen, score_local in layer_child_candidates:
                prev = dedup.get(child_idx)
                if prev is None or (mlen, score_local) > (prev[0], prev[1]):
                    dedup[child_idx] = (mlen, score_local)
            child_candidates = [(idx, v[0], v[1]) for idx, v in dedup.items()]
            child_candidates.sort(key=lambda x: (-x[1], -x[2], x[0]))

            t_prune_start = time.perf_counter() if collect_time_stats else None
            if layer_idx >= 2 and len(child_candidates) > top_k:
                child_candidates = child_candidates[:top_k]
            if collect_time_stats and profile is not None:
                profile["global_topk"] += time.perf_counter() - t_prune_start

            frontier = [c[0] for c in child_candidates]

            if len(node_tokens) >= max_nodes:
                break

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
        return outputs

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor, collect_time_stats=False):
        if self.draft_strategy == "direct_match":
            return self._topK_genrate_direct_match(
                input_ids=input_ids,
                logits_processor=logits_processor,
                collect_time_stats=collect_time_stats,
            )
        if self.draft_strategy != "prefill_topw":
            raise ValueError(
                f"Unsupported draft_strategy={self.draft_strategy!r}. "
                "Expected one of {'prefill_topw', 'direct_match'}."
            )

        input_ids_device = input_ids.device
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []
        match_ngram_list = []

        prefix_tokens = input_ids[0]
        prefix_tokens_list = prefix_tokens.detach().cpu().tolist()
        self.reset()  # tree_mask
        profile = self._create_empty_profile() if collect_time_stats else None

        if self.use_adaptive_config:
            layer1_topk = self.layer_config["1"][1]
        else:
            layer1_topk = top_k

        if collect_time_stats:
            t_layer1_start = time.perf_counter()
        layer1_tokens, layer1_logprobs, _, layer1_match_len = self._propose_by_strategy(
            prefix_tokens_list,
            layer1_topk,
            device=input_ids_device,
            collect_time_stats=collect_time_stats,
            profile=profile,
        )
        if collect_time_stats and profile is not None:
            profile["layer1"] += time.perf_counter() - t_layer1_start
        topk_index = layer1_tokens[None]
        topk_p = layer1_logprobs[None]

        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))

        ss_token.append(topk_index)
        layer1_match = torch.full(
            (1, layer1_topk), int(layer1_match_len), dtype=torch.long, device=input_ids_device
        )
        match_ngram_list.append(layer1_match)
        input_ids = topk_index

        current_paths = [[int(tok)] for tok in topk_index[0].tolist()]

        tree_mask = self.tree_mask_init[:, :, :layer1_topk, :layer1_topk].to(input_ids_device)
        topk_cs_index = torch.arange(layer1_topk, device=input_ids_device)

        for i in range(depth):
            if self.use_adaptive_config:
                layer_key = str(i + 2)
                candidates_per_node = self.layer_config[layer_key][0]
                global_topk = self.layer_config[layer_key][1]
            else:
                candidates_per_node = top_k
                global_topk = top_k

            self.tree_mask = tree_mask

            current_num_nodes = input_ids.shape[1]

            if self.use_adaptive_config:
                if i == 0:
                    scores_layer_start = 0  # Layer 1 starts at 0
                else:
                    scores_layer_start = self.layer_config["1"][1]
                    for j in range(i):
                        prev_layer_topk = self.layer_config[str(j + 1)][1]
                        candidates_per_node_prev = self.layer_config[str(j + 2)][0]
                        scores_layer_start += prev_layer_topk * candidates_per_node_prev

                parent_idx_in_prev_layer = topk_cs_index // max(candidates_per_node, 1)
                parents = scores_layer_start + parent_idx_in_prev_layer + 1  # +1 for 1-indexing
            else:
                bias1 = top_k if i > 0 else 0
                bias2 = max(0, i - 1)
                bias = 1 + top_k ** 2 * bias2 + bias1
                parents = (topk_cs_index + bias)
            parents_list.append(parents)

            topk_index = torch.zeros(
                (current_num_nodes, candidates_per_node), dtype=torch.long, device=input_ids_device
            )
            topk_p = torch.zeros(
                (current_num_nodes, candidates_per_node), dtype=torch.float32, device=input_ids_device
            )
            topk_match_ngram = torch.zeros(
                (current_num_nodes, candidates_per_node), dtype=torch.long, device=input_ids_device
            )

            if collect_time_stats:
                t_expand_start = time.perf_counter()
            for parent_idx in range(current_num_nodes):
                parent_path = current_paths[parent_idx]
                if parent_path:
                    parent_sequence = prefix_tokens_list + parent_path
                else:
                    parent_sequence = prefix_tokens_list

                cand_tokens, cand_logprobs, _, match_len = self._propose_by_strategy(
                    parent_sequence,
                    candidates_per_node,
                    device=input_ids_device,
                    collect_time_stats=collect_time_stats,
                    profile=profile,
                )
                topk_index[parent_idx] = cand_tokens
                topk_p[parent_idx] = cand_logprobs
                topk_match_ngram[parent_idx].fill_(int(match_len))
            if collect_time_stats and profile is not None:
                profile["expand_parents"] += time.perf_counter() - t_expand_start

            if collect_time_stats:
                t_topk_start = time.perf_counter()
            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), global_topk, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // max(candidates_per_node, 1)

            input_ids = topk_index.view(-1)[topk_cs_index][None]
            if collect_time_stats and profile is not None:
                profile["global_topk"] += time.perf_counter() - t_topk_start

            ss_token.append(topk_index)
            match_ngram_list.append(topk_match_ngram)
            scores_list.append(cu_scores)

            if collect_time_stats:
                t_update_start = time.perf_counter()
            flat_tokens = topk_index.view(-1)
            next_paths = []
            for sel in range(global_topk):
                pid = int(out_ids[sel].item())
                child_token = int(flat_tokens[topk_cs_index[sel]].item())
                next_paths.append(current_paths[pid] + [child_token])
            current_paths = next_paths

            # Ensure indices are aligned
            out_ids = out_ids.to(tree_mask.device)
            new_mask_chunk = self.tree_mask_init[:, :, :global_topk, :global_topk].to(tree_mask.device)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], new_mask_chunk), dim=3)
            if collect_time_stats and profile is not None:
                profile["update_paths_mask"] += time.perf_counter() - t_update_start

        if collect_time_stats:
            t_post_start = time.perf_counter()
        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        match_ngram_flat = torch.cat(match_ngram_list, dim=0).view(-1)

        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        draft_match_ngram = match_ngram_flat[top_scores_index]
        root_ngram = torch.full((1,), -1, dtype=torch.long, device=draft_match_ngram.device)
        draft_match_ngram = torch.cat((root_ngram, draft_match_ngram), dim=0)

        # draft_parents
        if self.use_adaptive_config:

            # scores_list
            scores_ranges = []
            scores_ranges.append((0, self.layer_config["1"][1]))  # Layer 1
            for j in range(depth):
                layer_key = str(j + 2)
                prev_layer_topk = self.layer_config[str(j + 1)][1]
                candidates_per_node = self.layer_config[layer_key][0]
                start = scores_ranges[-1][1]
                end = start + prev_layer_topk * candidates_per_node
                scores_ranges.append((start, end))

            draft_parents = torch.zeros(len(top_scores_index), dtype=torch.long, device=top_scores_index.device)

            for idx, score_idx in enumerate(top_scores_index):
                score_idx_val = score_idx.item()

                layer_idx = 0
                for j, (start, end) in enumerate(scores_ranges):
                    if start <= score_idx_val < end:
                        layer_idx = j
                        break

                if layer_idx == 0:
                    draft_parents[idx] = 0
                else:
                    offset_in_layer = score_idx_val - scores_ranges[layer_idx][0]
                    candidates_per_node = self.layer_config[str(layer_idx + 1)][0]
                    parent_idx_in_prev_layer = offset_in_layer // candidates_per_node
                    parent_idx_in_scores = scores_ranges[layer_idx - 1][0] + parent_idx_in_prev_layer
                    draft_parents[idx] = parent_idx_in_scores + 1  # +1 for 1-indexing
        else:
            # top_k
            draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()

        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth_val = position_ids_list[i]
                for j in reversed(range(depth_val + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids.to(input_ids_device)
        self.last_node_match_ngram = draft_match_ngram.detach().cpu()

        if collect_time_stats and profile is not None:
            profile["postprocess"] += time.perf_counter() - t_post_start
            self.last_profile_stats = profile
        else:
            self.last_profile_stats = None

        return draft_tokens.to(input_ids_device), retrieve_indices.to(input_ids_device), tree_mask.to(input_ids_device), tree_position_ids
