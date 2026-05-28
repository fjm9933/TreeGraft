import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

from .scheduler_artifacts import SCHEDULER_ARTIFACTS


FEATURE_DEPTH = int(SCHEDULER_ARTIFACTS["metadata"]["depth"])


def _relu(vec: Sequence[float]) -> List[float]:
    return [float(v) if float(v) > 0.0 else 0.0 for v in vec]


def _dense_forward(x: Sequence[float], layer: Dict[str, List]) -> List[float]:
    out: List[float] = []
    for row, bias in zip(layer["weight"], layer["bias"]):
        total = float(bias)
        for weight, value in zip(row, x):
            total += float(weight) * float(value)
        out.append(float(total))
    return out


def _zscore(x: Sequence[float], mean: Sequence[float], std: Sequence[float]) -> List[float]:
    return [
        (float(value) - float(mu)) / max(float(sigma), 1e-12)
        for value, mu, sigma in zip(x, mean, std)
    ]


def _context_features(
    *,
    prompt_len_tokens: float,
    t_tgt_ms: float,
    t_mid_ms: float,
) -> List[float]:
    safe_t_tgt = max(float(t_tgt_ms), 1e-6)
    safe_t_mid = max(float(t_mid_ms), 1e-6)
    return [
        math.log1p(max(float(prompt_len_tokens), 0.0)),
        safe_t_tgt,
        safe_t_mid,
        math.log(safe_t_tgt / safe_t_mid),
    ]


def _prefix_features(
    *,
    depth: int,
    layer_idx: int,
    called_layers: Iterable[int],
) -> List[float]:
    call_set = {int(x) for x in called_layers if 1 <= int(x) <= int(depth)}
    row = [
        float(layer_idx) / float(max(1, int(depth))),
        float(len(call_set)) / float(max(1, int(depth))),
    ]
    for bit in range(1, FEATURE_DEPTH + 1):
        row.append(1.0 if bit in call_set else 0.0)
    return row


def _state_row(
    *,
    prompt_len_tokens: float,
    t_tgt_ms: float,
    t_mid_ms: float,
    depth: int,
    layer_idx: int,
    called_layers: Iterable[int],
    tree_state: Sequence[float],
) -> List[float]:
    row: List[float] = []
    row.extend(
        _context_features(
            prompt_len_tokens=prompt_len_tokens,
            t_tgt_ms=t_tgt_ms,
            t_mid_ms=t_mid_ms,
        )
    )
    row.extend(_prefix_features(depth=depth, layer_idx=layer_idx, called_layers=called_layers))
    row.extend(float(x) for x in tree_state)
    return row


@dataclass(frozen=True)
class ScalarMLP:
    x_mean: Tuple[float, ...]
    x_std: Tuple[float, ...]
    y_mean: float
    y_std: float
    layers: Tuple[Dict[str, List], ...]

    def predict(self, row: Sequence[float]) -> float:
        xz = _zscore(row, self.x_mean, self.x_std)
        h = _relu(_dense_forward(xz, self.layers[0]))
        h = _relu(_dense_forward(h, self.layers[1]))
        yz = _dense_forward(h, self.layers[2])[0]
        return float(yz) * float(self.y_std) + float(self.y_mean)


@lru_cache(maxsize=1)
def _scheduler_model() -> ScalarMLP:
    artifact = SCHEDULER_ARTIFACTS["scheduler"]
    return ScalarMLP(
        x_mean=tuple(float(x) for x in artifact["x_mean"]),
        x_std=tuple(float(x) for x in artifact["x_std"]),
        y_mean=float(artifact["y_mean"]),
        y_std=float(artifact["y_std"]),
        layers=tuple(artifact["layers"]),
    )


def predict_treegraft_scheduler_margin(
    *,
    prompt_len_tokens: float,
    t_tgt_ms: float,
    t_mid_ms: float,
    depth: int,
    layer_idx: int,
    called_layers: Iterable[int],
    tree_state: Sequence[float],
) -> float:
    row = _state_row(
        prompt_len_tokens=prompt_len_tokens,
        t_tgt_ms=t_tgt_ms,
        t_mid_ms=t_mid_ms,
        depth=depth,
        layer_idx=layer_idx,
        called_layers=called_layers,
        tree_state=tree_state,
    )
    return float(_scheduler_model().predict(row))
