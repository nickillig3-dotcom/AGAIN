from __future__ import annotations

"""
Features_MarketStructure.py
===========================

Purpose
-------
Market Structure features (OHLCV-only) implemented in a *strictly causal* way.

Non-negotiable principles
-------------------------
- OHLCV only: No orderbook, no external data.
- No lookahead: anything that needs future bars (pivots) is emitted only when confirmed.
- Deterministic definitions: tie-handling is explicit and stable.
- Warmup policy:
  - price-level features (pivot values, last pivot levels) use math.nan until defined
  - event features (break/sweep/bos/choch/swing events) use 0.0/1.0 (no NaNs)
  - state features (trend_state) use -1.0/0.0/+1.0, with 0.0 meaning "unknown/neutral"
- Caching: computed exactly once per series and parameter set.

Key idea: Confirmed pivots without lookahead
--------------------------------------------
A pivot at index p needs 'right' future bars to confirm.
We therefore emit the pivot at the *confirmation bar* i = p + right (not at p).

Pivot high (unique max) at p is confirmed on i when:
  high[p] is strictly greater than all highs in window [p-left .. p+right]
  and the max is unique (ties => no pivot)

Likewise pivot low (unique min).

This is causal: at bar i we have seen all bars up to i, so the window is fully known.

Feature API
-----------
store = MarketStructureStore(series)
ph = store.feature("pivot_high_conf", left=2, right=2)
trend = store.feature("trend_state", left=2, right=2, tol_bps=0.0)
bos = store.feature("bos_up", left=2, right=2, tol_bps=0.0)

Feature naming
--------------
Pivot / levels:
- pivot_high_conf, pivot_low_conf
- last_pivot_high, last_pivot_low
- bars_since_pivot_high, bars_since_pivot_low

Swing classification (events at confirmation bars):
- swing_high_higher, swing_high_lower, equal_highs
- swing_low_higher, swing_low_lower, equal_lows

Break / structure events:
- break_up_close, break_down_close
- bos_up, bos_down
- choch_up, choch_down
- sweep_high, sweep_low
"""

from dataclasses import dataclass
import inspect
import math
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, ValidationError, require


__all__ = [
    "MarketStructureStore",
    "STRUCTURE_REGISTRY",
]


_NAN = float("nan")


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _isnan(x: float) -> bool:
    return math.isnan(float(x))


def _freeze_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if isinstance(v, str):
        return str(v)
    try:
        return str(v)
    except Exception:
        return repr(v)


def _check_int(name: str, v: Any) -> int:
    try:
        x = int(v)
    except Exception as e:
        raise ValidationError(f"{name} must be int-castable, got {v!r}") from e
    require(x > 0, f"{name} must be > 0, got {x}")
    return x


def _check_nonneg_float(name: str, v: Any) -> float:
    try:
        x = float(v)
    except Exception as e:
        raise ValidationError(f"{name} must be float-castable, got {v!r}") from e
    require(_finite(x) and x >= 0.0, f"{name} must be finite and >= 0, got {x!r}")
    return x


def _tol_frac(tol_bps: float) -> float:
    tb = _check_nonneg_float("tol_bps", tol_bps)
    return tb / 10_000.0


# -----------------------------
# Registry
# -----------------------------
@dataclass(frozen=True, slots=True)
class _FeatureDef:
    name: str
    fn: Callable[..., Any]
    param_defaults: Dict[str, Any]
    required_params: Tuple[str, ...]
    allowed_params: Tuple[str, ...]


STRUCTURE_REGISTRY: Dict[str, _FeatureDef] = {}


def _register_feature(name: str, fn: Callable[..., Any]) -> None:
    nm = str(name).strip().lower()
    require(nm, "feature name must be non-empty")
    require(nm not in STRUCTURE_REGISTRY, f"feature {nm!r} already registered")

    sig = inspect.signature(fn)
    defaults: Dict[str, Any] = {}
    required: List[str] = []
    allowed: List[str] = []

    for pname, p in sig.parameters.items():
        if pname == "store":
            continue
        allowed.append(pname)
        if p.default is inspect._empty:
            required.append(pname)
        else:
            defaults[pname] = p.default

    STRUCTURE_REGISTRY[nm] = _FeatureDef(
        name=nm,
        fn=fn,
        param_defaults=defaults,
        required_params=tuple(required),
        allowed_params=tuple(allowed),
    )


# -----------------------------
# Store
# -----------------------------
class MarketStructureStore:
    """
    Caches base OHLCV arrays and derived market structure features.
    """

    def __init__(self, series: OhlcvSeries) -> None:
        require(series is not None, "series must not be None")
        self.series = series

        self._base_cache: Dict[str, List[float]] = {}
        self._vec_cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[float]] = {}
        self._bundle_cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}

    @property
    def n(self) -> int:
        return len(self.series.ts_ms)

    def base(self, name: str) -> List[float]:
        nm = str(name).strip().lower()
        require(nm, "base name must be non-empty")
        if nm in self._base_cache:
            return self._base_cache[nm]

        s = self.series
        if nm == "open":
            out = [float(v) for v in s.open]
        elif nm == "high":
            out = [float(v) for v in s.high]
        elif nm == "low":
            out = [float(v) for v in s.low]
        elif nm == "close":
            out = [float(v) for v in s.close]
        elif nm == "volume":
            out = [float(v) for v in s.volume]
        else:
            raise ValidationError(f"Unknown base series {name!r}. Supported: open/high/low/close/volume")

        require(len(out) == self.n, f"base({nm}) length mismatch")
        self._base_cache[nm] = out
        return out

    def feature(self, name: str, **params: Any) -> List[float]:
        nm = str(name).strip().lower()
        require(nm in STRUCTURE_REGISTRY, f"Unknown structure feature {name!r}. Available: {sorted(STRUCTURE_REGISTRY.keys())}")
        fdef = STRUCTURE_REGISTRY[nm]

        # Validate param names
        for k in params.keys():
            require(k in fdef.allowed_params, f"Feature {nm!r} does not accept param {k!r}. Allowed: {fdef.allowed_params}")

        # Canonical params: fill defaults
        canonical: Dict[str, Any] = dict(fdef.param_defaults)
        canonical.update(params)

        for req in fdef.required_params:
            require(req in canonical, f"Feature {nm!r} missing required param {req!r}")

        key_items = tuple(sorted((k, _freeze_value(canonical[k])) for k in canonical.keys()))
        key = (nm, key_items)

        if key in self._vec_cache:
            return self._vec_cache[key]

        out_any = fdef.fn(store=self, **canonical)  # type: ignore[arg-type]
        require(isinstance(out_any, list), f"Feature {nm!r} must return List[float], got {type(out_any).__name__}")
        out: List[float] = out_any

        require(len(out) == self.n, f"Feature {nm!r} produced length {len(out)}, expected {self.n}")
        self._vec_cache[key] = out
        return out


# -----------------------------
# Internal bundles
# -----------------------------
@dataclass(slots=True)
class _PivotsBundle:
    left: int
    right: int

    pivot_high_conf: List[float]       # value at confirm bar, else nan
    pivot_low_conf: List[float]        # value at confirm bar, else nan
    pivot_high_index: List[int]        # pivot index p at confirm bar, else -1
    pivot_low_index: List[int]         # pivot index p at confirm bar, else -1

    last_pivot_high: List[float]       # last confirmed pivot high value up to bar i
    last_pivot_low: List[float]        # last confirmed pivot low value up to bar i
    last_pivot_high_index: List[int]   # pivot index of last confirmed pivot high (p), else -1
    last_pivot_low_index: List[int]    # pivot index of last confirmed pivot low (p), else -1

    bars_since_pivot_high: List[float]  # bars since last pivot HIGH confirmation bar; nan if none
    bars_since_pivot_low: List[float]   # bars since last pivot LOW confirmation bar; nan if none


@dataclass(slots=True)
class _StructureBundle:
    left: int
    right: int
    tol_bps: float

    swing_high_higher: List[float]
    swing_high_lower: List[float]
    equal_highs: List[float]

    swing_low_higher: List[float]
    swing_low_lower: List[float]
    equal_lows: List[float]

    trend_state: List[float]  # -1/0/+1

    break_up_close: List[float]
    break_down_close: List[float]

    bos_up: List[float]
    bos_down: List[float]
    choch_up: List[float]
    choch_down: List[float]

    sweep_high: List[float]
    sweep_low: List[float]


def _get_pivots(store: MarketStructureStore, *, left: int, right: int) -> _PivotsBundle:
    l = _check_int("left", left)
    r = _check_int("right", right)

    key = ("pivots", (("left", l), ("right", r)))
    if key in store._bundle_cache:
        return store._bundle_cache[key]

    hi = store.base("high")
    lo = store.base("low")
    n = store.n

    ph_conf = [_NAN] * n
    pl_conf = [_NAN] * n
    ph_pidx = [-1] * n
    pl_pidx = [-1] * n

    # Sliding window length for centered pivot at p confirmed at i = p + right
    w = l + r + 1
    dq_max: deque[int] = deque()  # indices, values non-increasing (keep duplicates)
    dq_min: deque[int] = deque()  # indices, values non-decreasing (keep duplicates)

    for i in range(n):
        # --- max deque for highs ---
        h_i = hi[i]
        # keep duplicates by popping only strictly smaller
        while dq_max and float(hi[dq_max[-1]]) < float(h_i):
            dq_max.pop()
        dq_max.append(i)

        # --- min deque for lows ---
        l_i = lo[i]
        while dq_min and float(lo[dq_min[-1]]) > float(l_i):
            dq_min.pop()
        dq_min.append(i)

        start = i - w + 1
        # remove out-of-window
        while dq_max and dq_max[0] < start:
            dq_max.popleft()
        while dq_min and dq_min[0] < start:
            dq_min.popleft()

        # candidate pivot index for confirmation at i
        p = i - r
        if p < 0:
            continue
        if p - l < 0:
            continue
        # At this point, the full window [p-l .. p+r] is within [0..i] and start==p-l

        # Pivot high: max index equals p and unique within window
        if dq_max and dq_max[0] == p:
            if len(dq_max) == 1 or float(hi[dq_max[1]]) < float(hi[p]):
                ph_conf[i] = float(hi[p])
                ph_pidx[i] = int(p)

        # Pivot low: min index equals p and unique within window
        if dq_min and dq_min[0] == p:
            if len(dq_min) == 1 or float(lo[dq_min[1]]) > float(lo[p]):
                pl_conf[i] = float(lo[p])
                pl_pidx[i] = int(p)

    # last pivot levels + bars since confirmation
    last_ph = [_NAN] * n
    last_pl = [_NAN] * n
    last_ph_idx = [-1] * n
    last_pl_idx = [-1] * n
    bsince_ph = [_NAN] * n
    bsince_pl = [_NAN] * n

    last_ph_val = _NAN
    last_pl_val = _NAN
    last_ph_pidx = -1
    last_pl_pidx = -1
    last_ph_conf_i: Optional[int] = None
    last_pl_conf_i: Optional[int] = None

    for i in range(n):
        if ph_pidx[i] != -1:
            last_ph_val = float(ph_conf[i])
            last_ph_pidx = int(ph_pidx[i])
            last_ph_conf_i = i
            bsince_ph[i] = 0.0
        else:
            if last_ph_conf_i is None:
                bsince_ph[i] = _NAN
            else:
                bsince_ph[i] = float(i - last_ph_conf_i)

        if pl_pidx[i] != -1:
            last_pl_val = float(pl_conf[i])
            last_pl_pidx = int(pl_pidx[i])
            last_pl_conf_i = i
            bsince_pl[i] = 0.0
        else:
            if last_pl_conf_i is None:
                bsince_pl[i] = _NAN
            else:
                bsince_pl[i] = float(i - last_pl_conf_i)

        last_ph[i] = last_ph_val
        last_pl[i] = last_pl_val
        last_ph_idx[i] = last_ph_pidx
        last_pl_idx[i] = last_pl_pidx

    bundle = _PivotsBundle(
        left=l,
        right=r,
        pivot_high_conf=ph_conf,
        pivot_low_conf=pl_conf,
        pivot_high_index=ph_pidx,
        pivot_low_index=pl_pidx,
        last_pivot_high=last_ph,
        last_pivot_low=last_pl,
        last_pivot_high_index=last_ph_idx,
        last_pivot_low_index=last_pl_idx,
        bars_since_pivot_high=bsince_ph,
        bars_since_pivot_low=bsince_pl,
    )

    store._bundle_cache[key] = bundle
    return bundle


def _get_structure(store: MarketStructureStore, *, left: int, right: int, tol_bps: float) -> _StructureBundle:
    l = _check_int("left", left)
    r = _check_int("right", right)
    tb = _check_nonneg_float("tol_bps", tol_bps)
    tol = _tol_frac(tb)

    key = ("structure", (("left", l), ("right", r), ("tol_bps", float(tb))))
    if key in store._bundle_cache:
        return store._bundle_cache[key]

    piv = _get_pivots(store, left=l, right=r)
    hi = store.base("high")
    lo = store.base("low")
    cl = store.base("close")
    n = store.n

    # events default to 0.0
    hh = [0.0] * n
    lh = [0.0] * n
    eh = [0.0] * n

    hl_ = [0.0] * n
    ll_ = [0.0] * n
    el = [0.0] * n

    trend = [0.0] * n

    br_up = [0.0] * n
    br_dn = [0.0] * n

    bos_up = [0.0] * n
    bos_dn = [0.0] * n
    choch_up = [0.0] * n
    choch_dn = [0.0] * n

    sw_hi = [0.0] * n
    sw_lo = [0.0] * n

    # swing classification + trend from last two highs/lows
    prev_ph: Optional[float] = None
    prev_pl: Optional[float] = None

    ph1: Optional[float] = None
    ph2: Optional[float] = None
    pl1: Optional[float] = None
    pl2: Optional[float] = None

    for i in range(n):
        # update swings when new confirmed pivots arrive
        if piv.pivot_high_index[i] != -1:
            cur = float(piv.pivot_high_conf[i])
            if prev_ph is not None and _finite(prev_ph) and _finite(cur) and prev_ph != 0.0:
                upper = prev_ph * (1.0 + tol)
                lower = prev_ph * (1.0 - tol)
                if cur > upper:
                    hh[i] = 1.0
                elif cur < lower:
                    lh[i] = 1.0
                else:
                    eh[i] = 1.0
            prev_ph = cur
            # last two pivot highs
            ph1, ph2 = ph2, cur

        if piv.pivot_low_index[i] != -1:
            cur = float(piv.pivot_low_conf[i])
            if prev_pl is not None and _finite(prev_pl) and _finite(cur) and prev_pl != 0.0:
                upper = prev_pl * (1.0 + tol)
                lower = prev_pl * (1.0 - tol)
                if cur > upper:
                    hl_[i] = 1.0
                elif cur < lower:
                    ll_[i] = 1.0
                else:
                    el[i] = 1.0
            prev_pl = cur
            # last two pivot lows
            pl1, pl2 = pl2, cur

        # trend state uses last two highs and last two lows
        if (
            ph1 is not None and ph2 is not None and pl1 is not None and pl2 is not None
            and _finite(ph1) and _finite(ph2) and _finite(pl1) and _finite(pl2)
            and ph1 != 0.0 and pl1 != 0.0
        ):
            ph_up = ph2 > ph1 * (1.0 + tol)
            ph_dn = ph2 < ph1 * (1.0 - tol)
            pl_up = pl2 > pl1 * (1.0 + tol)
            pl_dn = pl2 < pl1 * (1.0 - tol)

            if ph_up and pl_up:
                trend[i] = 1.0
            elif ph_dn and pl_dn:
                trend[i] = -1.0
            else:
                trend[i] = 0.0
        else:
            trend[i] = 0.0

    # breaks & sweeps from last pivot levels
    for i in range(n):
        lph = piv.last_pivot_high[i]
        lpl = piv.last_pivot_low[i]

        if _finite(lph):
            lvl = float(lph) * (1.0 + tol)
            if float(cl[i]) > lvl:
                br_up[i] = 1.0
            if float(hi[i]) >= lvl and float(cl[i]) < lvl:
                sw_hi[i] = 1.0

        if _finite(lpl):
            lvl = float(lpl) * (1.0 - tol)
            if float(cl[i]) < lvl:
                br_dn[i] = 1.0
            if float(lo[i]) <= lvl and float(cl[i]) > lvl:
                sw_lo[i] = 1.0

    # classify breaks into BOS / CHOCH using trend_state
    for i in range(n):
        t = float(trend[i])
        if br_up[i] == 1.0:
            if t > 0.0:
                bos_up[i] = 1.0
            elif t < 0.0:
                choch_up[i] = 1.0

        if br_dn[i] == 1.0:
            if t < 0.0:
                bos_dn[i] = 1.0
            elif t > 0.0:
                choch_dn[i] = 1.0

    bundle = _StructureBundle(
        left=l,
        right=r,
        tol_bps=float(tb),

        swing_high_higher=hh,
        swing_high_lower=lh,
        equal_highs=eh,

        swing_low_higher=hl_,
        swing_low_lower=ll_,
        equal_lows=el,

        trend_state=trend,

        break_up_close=br_up,
        break_down_close=br_dn,

        bos_up=bos_up,
        bos_down=bos_dn,
        choch_up=choch_up,
        choch_down=choch_dn,

        sweep_high=sw_hi,
        sweep_low=sw_lo,
    )

    store._bundle_cache[key] = bundle
    return bundle


# -----------------------------
# Feature functions (registry targets)
# -----------------------------
def _feat_pivot_high_conf(store: MarketStructureStore, *, left: int = 2, right: int = 2) -> List[float]:
    return _get_pivots(store, left=left, right=right).pivot_high_conf


def _feat_pivot_low_conf(store: MarketStructureStore, *, left: int = 2, right: int = 2) -> List[float]:
    return _get_pivots(store, left=left, right=right).pivot_low_conf


def _feat_last_pivot_high(store: MarketStructureStore, *, left: int = 2, right: int = 2) -> List[float]:
    return _get_pivots(store, left=left, right=right).last_pivot_high


def _feat_last_pivot_low(store: MarketStructureStore, *, left: int = 2, right: int = 2) -> List[float]:
    return _get_pivots(store, left=left, right=right).last_pivot_low


def _feat_bars_since_pivot_high(store: MarketStructureStore, *, left: int = 2, right: int = 2) -> List[float]:
    return _get_pivots(store, left=left, right=right).bars_since_pivot_high


def _feat_bars_since_pivot_low(store: MarketStructureStore, *, left: int = 2, right: int = 2) -> List[float]:
    return _get_pivots(store, left=left, right=right).bars_since_pivot_low


def _feat_swing_high_higher(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).swing_high_higher


def _feat_swing_high_lower(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).swing_high_lower


def _feat_equal_highs(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).equal_highs


def _feat_swing_low_higher(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).swing_low_higher


def _feat_swing_low_lower(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).swing_low_lower


def _feat_equal_lows(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).equal_lows


def _feat_trend_state(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).trend_state


def _feat_break_up_close(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).break_up_close


def _feat_break_down_close(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).break_down_close


def _feat_bos_up(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).bos_up


def _feat_bos_down(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).bos_down


def _feat_choch_up(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).choch_up


def _feat_choch_down(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).choch_down


def _feat_sweep_high(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).sweep_high


def _feat_sweep_low(store: MarketStructureStore, *, left: int = 2, right: int = 2, tol_bps: float = 0.0) -> List[float]:
    return _get_structure(store, left=left, right=right, tol_bps=tol_bps).sweep_low


# Register all features
_register_feature("pivot_high_conf", _feat_pivot_high_conf)
_register_feature("pivot_low_conf", _feat_pivot_low_conf)
_register_feature("last_pivot_high", _feat_last_pivot_high)
_register_feature("last_pivot_low", _feat_last_pivot_low)
_register_feature("bars_since_pivot_high", _feat_bars_since_pivot_high)
_register_feature("bars_since_pivot_low", _feat_bars_since_pivot_low)

_register_feature("swing_high_higher", _feat_swing_high_higher)
_register_feature("swing_high_lower", _feat_swing_high_lower)
_register_feature("equal_highs", _feat_equal_highs)

_register_feature("swing_low_higher", _feat_swing_low_higher)
_register_feature("swing_low_lower", _feat_swing_low_lower)
_register_feature("equal_lows", _feat_equal_lows)

_register_feature("trend_state", _feat_trend_state)

_register_feature("break_up_close", _feat_break_up_close)
_register_feature("break_down_close", _feat_break_down_close)

_register_feature("bos_up", _feat_bos_up)
_register_feature("bos_down", _feat_bos_down)
_register_feature("choch_up", _feat_choch_up)
_register_feature("choch_down", _feat_choch_down)

_register_feature("sweep_high", _feat_sweep_high)
_register_feature("sweep_low", _feat_sweep_low)


# -----------------------------
# Self-test
# -----------------------------
def _self_test() -> None:
    def _expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    # Deterministic series engineered for pivots and events with left=1,right=1
    # Indices 0..6
    highs = [10.0, 11.0, 12.0, 11.0, 12.5, 12.2, 13.0]
    lows =  [ 9.0, 10.0, 11.0, 10.0, 11.5, 11.2, 12.0]
    closes = [9.5, 10.5, 11.5, 10.8, 11.8, 12.0, 12.8]
    opens = closes[:]  # keep it simple

    series = OhlcvSeries(
        ts_ms=[1, 2, 3, 4, 5, 6, 7],
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        volume=[1.0] * 7,
        symbol="TEST",
        timeframe="1m",
    )

    store = MarketStructureStore(series)

    ph = store.feature("pivot_high_conf", left=1, right=1)
    pl = store.feature("pivot_low_conf", left=1, right=1)

    # No-lookahead: pivot at p=2 (high=12) confirmed at i=3, not at i=2
    _expect(_isnan(ph[2]), "pivot_high_conf must NOT appear at pivot index (no-lookahead)")
    _expect(_finite(ph[3]) and abs(ph[3] - 12.0) < 1e-12, f"pivot_high_conf[3] expected 12 got {ph[3]}")
    # Second pivot high at p=4 (12.5) confirmed at i=5
    _expect(_finite(ph[5]) and abs(ph[5] - 12.5) < 1e-12, f"pivot_high_conf[5] expected 12.5 got {ph[5]}")

    # Pivot low at p=3 (low=10) confirmed at i=4
    _expect(_finite(pl[4]) and abs(pl[4] - 10.0) < 1e-12, f"pivot_low_conf[4] expected 10 got {pl[4]}")
    # Pivot low at p=5 (11.2) confirmed at i=6
    _expect(_finite(pl[6]) and abs(pl[6] - 11.2) < 1e-12, f"pivot_low_conf[6] expected 11.2 got {pl[6]}")

    # Cache check: exact same call returns same list object
    ph2 = store.feature("pivot_high_conf", left=1, right=1)
    _expect(ph2 is ph, "cache failed: pivot_high_conf with same params should be same object")

    last_ph = store.feature("last_pivot_high", left=1, right=1)
    last_pl = store.feature("last_pivot_low", left=1, right=1)

    # After confirmation at i=3, last pivot high is 12 through i=4
    _expect(_finite(last_ph[4]) and abs(last_ph[4] - 12.0) < 1e-12, f"last_pivot_high[4] expected 12 got {last_ph[4]}")
    # After confirmation at i=5, last pivot high becomes 12.5
    _expect(_finite(last_ph[6]) and abs(last_ph[6] - 12.5) < 1e-12, f"last_pivot_high[6] expected 12.5 got {last_ph[6]}")

    # Sweep above 12 at bar i=4: high=12.5 pierces, close=11.8 back below 12
    sweep_high = store.feature("sweep_high", left=1, right=1, tol_bps=0.0)
    _expect(sweep_high[4] == 1.0, "Expected sweep_high at index 4")

    # Break above 12.5 at bar i=6: close=12.8 > last_pivot_high(12.5)
    break_up = store.feature("break_up_close", left=1, right=1, tol_bps=0.0)
    _expect(break_up[6] == 1.0, "Expected break_up_close at index 6")

    # Swing classification: second pivot high > first => swing_high_higher at i=5
    shh = store.feature("swing_high_higher", left=1, right=1, tol_bps=0.0)
    _expect(shh[5] == 1.0, "Expected swing_high_higher at index 5")

    # Swing lows: second pivot low > first => swing_low_higher at i=6
    slh = store.feature("swing_low_higher", left=1, right=1, tol_bps=0.0)
    _expect(slh[6] == 1.0, "Expected swing_low_higher at index 6")

    # Trend becomes bullish only once we have two highs and two lows (here at i=6)
    trend = store.feature("trend_state", left=1, right=1, tol_bps=0.0)
    _expect(trend[5] == 0.0, "Trend should still be neutral before second low is confirmed")
    _expect(trend[6] == 1.0, "Trend should be bullish at index 6")

    # BOS up requires trend bullish and break_up_close => at i=6
    bos = store.feature("bos_up", left=1, right=1, tol_bps=0.0)
    _expect(bos[6] == 1.0, "Expected bos_up at index 6")

    # Equality tolerance test (equal highs within tol_bps)
    highs2 = [10.0, 11.0, 12.0, 11.0, 12.0004, 11.0]  # pivot highs at p=2 and p=4
    lows2 =  [ 9.0, 10.0, 11.0, 10.0, 11.0,    10.5]
    closes2 = [9.5, 10.5, 11.5, 10.8, 11.8, 10.9]

    series2 = OhlcvSeries(
        ts_ms=[1, 2, 3, 4, 5, 6],
        open=closes2[:],
        high=highs2,
        low=lows2,
        close=closes2,
        volume=[1.0] * 6,
        symbol="TEST",
        timeframe="1m",
    )
    store2 = MarketStructureStore(series2)

    eqh = store2.feature("equal_highs", left=1, right=1, tol_bps=10.0)  # 10 bps tolerance
    shh2 = store2.feature("swing_high_higher", left=1, right=1, tol_bps=10.0)

    # second pivot high confirmed at i=5; within tolerance => equal_highs=1, swing_high_higher=0
    _expect(eqh[5] == 1.0, "Expected equal_highs at confirmation index 5 with tol_bps")
    _expect(shh2[5] == 0.0, "Expected no swing_high_higher when highs are equal within tolerance")

    print("Features_MarketStructure self-test: OK")


if __name__ == "__main__":
    _self_test()
