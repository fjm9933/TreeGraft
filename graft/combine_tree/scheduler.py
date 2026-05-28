import math
import time
from typing import Dict, Iterable, List, Optional

from .scheduler_runtime import predict_treegraft_scheduler_margin


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


class TreeGraftScheduler:
    """TreeGraft value-guided online scheduler."""

    controller_kind = "scheduler"
    needs_phase0_timing = False
    needs_phase0_baseline_probe = False
    needs_warmup_rounds = True

    def __init__(
        self,
        family: str = "treegraft_scheduler",
        score_mode: str = "tree_state",
        budget_b0: Optional[int] = None,
        top_k: int = 10,
        depth: int = 5,
        pair_id: str = "",
        task_id: str = "",
        **__,
    ):
        family = str(family or "treegraft_scheduler")
        if family != "treegraft_scheduler":
            raise ValueError(f"Unsupported scheduler family={family}")
        self.family = family
        self.score_mode = str(score_mode or "tree_state")
        self.budget_b0 = None if budget_b0 is None else max(0, int(budget_b0))
        self.top_k = max(1, int(top_k))
        self.depth = max(1, int(depth))
        self.pair_id = str(pair_id or "")
        self.task_id = str(task_id or "")
        self.enabled = True
        self.online_active = False
        self.A_t = None
        self.T_tgt = None
        self.T_mid = None
        self.warmup_round3_A_t = None
        self.warmup_round3_T_tgt = None
        self.warmup_round3_T_mid = None
        self.prompt_len_tokens = 0.0
        self.decode_step_call_count = 0
        self.called_layers: List[int] = []
        self.seen_scores: List[float] = []

    def set_prompt_len_tokens(self, prompt_len_tokens: Optional[float]):
        self.prompt_len_tokens = max(0.0, _safe_float(prompt_len_tokens, 0.0))

    def set_pair_id(self, pair_id: Optional[str]):
        self.pair_id = str(pair_id or "")

    def set_task_id(self, task_id: Optional[str]):
        self.task_id = str(task_id or "")

    def suspend(self):
        self.enabled = False
        self.online_active = False

    def is_ready(self) -> bool:
        return bool(
            self.enabled
            and self.online_active
            and self.T_tgt is not None
            and self.T_mid is not None
            and self.budget_b0 is not None
        )

    def should_collect_warmup_target_timing(self) -> bool:
        return bool(self.enabled and not self.is_ready())

    def should_collect_warmup_middle_timing(self) -> bool:
        return bool(self.enabled and not self.is_ready())

    def should_force_probe_for_tree(self) -> bool:
        return False

    def should_force_probe_layer(self, layer_idx: int) -> bool:
        del layer_idx
        return False

    def initialize_phase0(self, *_, **__):
        return

    def finalize_after_warmup(
        self,
        A_t: Optional[float],
        T_tgt: Optional[float] = None,
        T_mid: Optional[float] = None,
    ):
        del A_t
        if T_tgt is None or T_mid is None:
            raise RuntimeError("TreeGraft scheduler warmup failed to collect target/middle timing")
        self.T_tgt = max(1e-8, float(T_tgt))
        self.T_mid = max(1e-8, float(T_mid))
        self.warmup_round3_T_tgt = float(self.T_tgt)
        self.warmup_round3_T_mid = float(self.T_mid)
        if self.budget_b0 is None:
            self.budget_b0 = max(0, int(math.floor(float(self.T_tgt) / float(self.T_mid))))
        self.online_active = True
        self.reset_decode_step_state()

    def observe_step(self, *_, **__):
        return

    def reset_decode_step_state(self):
        self.decode_step_call_count = 0
        self.called_layers = []
        self.seen_scores = []

    def note_decode_step_call(self, layer_idx: Optional[int] = None):
        self.decode_step_call_count += 1
        if layer_idx is None:
            return
        try:
            layer_val = int(layer_idx)
        except Exception:
            return
        if layer_val > 0 and layer_val not in self.called_layers:
            self.called_layers.append(layer_val)
            self.called_layers.sort()

    def _remaining_budget(self) -> int:
        return max(0, int(self.budget_b0 or 0) - int(self.decode_step_call_count))

    def _tree_state(self, preview_record: Dict) -> List[float]:
        return [
            max(0.0, _safe_float(preview_record.get("depth_before", 0.0))),
            _clamp01(_safe_float(preview_record.get("leaf_ratio_before", 0.0))),
            max(0.0, _safe_float(preview_record.get("collapse_debt", 0.0))),
            _clamp01(
                _safe_float(
                    preview_record.get(
                        "pre_reselect_past_ratio",
                        preview_record.get("pre_reselect_from_past_layer_ratio", 0.0),
                    )
                )
            ),
            _clamp01(
                _safe_float(
                    preview_record.get(
                        "pre_reselect_overlap",
                        preview_record.get("pre_reselect_overlap_ratio", 1.0),
                    )
                )
            ),
            _safe_float(
                preview_record.get(
                    "pre_reselect_score",
                    preview_record.get("pre_reselect_frontier_mean_score", 0.0),
                )
            ),
        ]

    def build_decision(
        self,
        preview_record: Dict,
        depth: int,
    ) -> Dict:
        start = time.perf_counter()
        remaining_budget = self._remaining_budget()
        if not self.is_ready():
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {
                "scheduler_should_call": False,
                "scheduler_mode": "inactive",
                "scheduler_u": None,
                "scheduler_family": self.family,
                "scheduler_score_mode": self.score_mode,
                "scheduler_budget_b0": self.budget_b0,
                "scheduler_remaining_budget": remaining_budget,
                "scheduler_elapsed_ms": float(elapsed_ms),
            }

        layer_idx = int(preview_record.get("layer_idx", 0))
        total_depth = max(1, int(depth or self.depth))
        margin = predict_treegraft_scheduler_margin(
            prompt_len_tokens=float(self.prompt_len_tokens),
            t_tgt_ms=float(self.T_tgt or 0.0),
            t_mid_ms=float(self.T_mid or 0.0),
            depth=total_depth,
            layer_idx=layer_idx,
            called_layers=self.called_layers,
            tree_state=self._tree_state(preview_record),
        )
        should_call = bool(remaining_budget > 0 and float(margin) > 0.0)
        self.seen_scores.append(float(margin))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "scheduler_mode": "online",
            "scheduler_u": float(margin),
            "scheduler_t_mid_ms": float(self.T_mid) if self.T_mid is not None else None,
            "scheduler_t_tgt_ms": float(self.T_tgt) if self.T_tgt is not None else None,
            "scheduler_decode_step_call_count": int(self.decode_step_call_count),
            "scheduler_should_call": bool(should_call),
            "scheduler_family": self.family,
            "scheduler_score_mode": self.score_mode,
            "scheduler_budget_b0": int(self.budget_b0 or 0),
            "scheduler_remaining_budget": int(remaining_budget),
            "scheduler_called_layers": list(self.called_layers),
            "scheduler_elapsed_ms": float(elapsed_ms),
        }
