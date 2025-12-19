from __future__ import annotations

"""
Regime_Classifier.py
====================

Purpose
-------
Kausal (no-lookahead) Regime-Klassifikation nur aus OHLCV:
- Trend Direction (trend_dir) in {-1,0,+1}
- Volatility Regime (vol_regime) in {-1,0,+1}
- Volume Regime (volume_regime) in {-1,0,+1}
- Combined phase_id in {0..26} mapping (trend, vol, volume)

Design goals
------------
- Robust & konservativ: unklar / warmup -> 0 (neutral) statt raten.
- Schnell & cachebar: einmal pro Serie + Parameter-Set.
- Mining-friendly: Registry -> systematische Parameter-Sweeps.

Wichtiger Realismus-Hinweis (Trend)
-----------------------------------
MarketStructure-Trend (Pivot-basiert) kann in monotonicen/impulsiven Phasen "stale" werden,
weil keine neuen bestätigten Pivots auftauchen. DI/ADX ist in solchen Phasen oft zuverlässiger.
Daher ist die Trend-Direction-Logik:

  1) ADX >= threshold ist Voraussetzung für "Trend" (sonst 0)
  2) Direction bevorzugt DI (+DI vs -DI), wenn Separation >= di_min_sep
  3) MarketStructure trend_state ist Fallback, wenn DI nicht eindeutig ist

Damit ist das Regime live-robuster und vermeidet, dass staler Structure-Trend DI/ADX aushebelt.
"""

from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable, Dict, List, Sequence, Tuple

from Core_Types import OhlcvSeries, ValidationError, require
from Features_Indicators import FeatureStore, zscore
from Features_MarketStructure import MarketStructureStore


__all__ = [
    "RegimeStore",
    "REGIME_REGISTRY",
    "segments_from_labels",
    "mask_equals",
]


_NAN = float("nan")


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _isnan(x: float) -> bool:
    return math.isnan(float(x))


def _as_int(x: Any, name: str) -> int:
    try:
        v = int(x)
    except Exception as e:
        raise ValidationError(f"{name} must be int-castable, got {x!r}") from e
    return v


def _as_float(x: Any, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise ValidationError(f"{name} must be float-castable, got {x!r}") from e
    require(_finite(v), f"{name} must be finite, got {v!r}")
    return v


def _check_pos_int(x: Any, name: str) -> int:
    v = _as_int(x, name)
    require(v > 0, f"{name} must be > 0, got {v}")
    return v


def _check_nonneg_float(x: Any, name: str) -> float:
    v = _as_float(x, name)
    require(v >= 0.0, f"{name} must be >= 0, got {v}")
    return v


def _sign3(x: float) -> int:
    """
    Convert float to {-1,0,+1} with a small deadzone.
    """
    xf = float(x)
    if xf > 0.5:
        return 1
    if xf < -0.5:
        return -1
    return 0


def _freeze_value(v: Any) -> Any:
    """
    Cache key normalizer (stable + hashable).
    """
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


def _label_eq(a: float, b: float) -> bool:
    if _isnan(a) and _isnan(b):
        return True
    return float(a) == float(b)


@dataclass(frozen=True, slots=True)
class _RegimeDef:
    name: str
    fn: Callable[..., Any]
    param_defaults: Dict[str, Any]
    required_params: Tuple[str, ...]
    allowed_params: Tuple[str, ...]


REGIME_REGISTRY: Dict[str, _RegimeDef] = {}


def _register_regime(name: str, fn: Callable[..., Any]) -> None:
    nm = str(name).strip().lower()
    require(nm, "regime name must be non-empty")
    require(nm not in REGIME_REGISTRY, f"regime {nm!r} already registered")

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

    REGIME_REGISTRY[nm] = _RegimeDef(
        name=nm,
        fn=fn,
        param_defaults=defaults,
        required_params=tuple(required),
        allowed_params=tuple(allowed),
    )


class RegimeStore:
    """
    Caches regimes for a given OHLCV series.

    Reuses:
      - FeatureStore: ATR/ADX/DI, and base series
      - MarketStructureStore: trend_state (as fallback)
    """

    def __init__(self, series: OhlcvSeries) -> None:
        require(series is not None, "series must not be None")
        self.series = series
        self.ind = FeatureStore(series)
        self.ms = MarketStructureStore(series)

        self._vec_cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], List[float]] = {}
        self._bundle_cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Any] = {}

    @property
    def n(self) -> int:
        return len(self.series.ts_ms)

    def regime(self, name: str, **params: Any) -> List[float]:
        nm = str(name).strip().lower()
        require(nm in REGIME_REGISTRY, f"Unknown regime {name!r}. Available: {sorted(REGIME_REGISTRY.keys())}")
        rdef = REGIME_REGISTRY[nm]

        for k in params.keys():
            require(k in rdef.allowed_params, f"Regime {nm!r} does not accept param {k!r}. Allowed: {rdef.allowed_params}")

        canonical: Dict[str, Any] = dict(rdef.param_defaults)
        canonical.update(params)

        for req in rdef.required_params:
            require(req in canonical, f"Regime {nm!r} missing required param {req!r}")

        key_items = tuple(sorted((k, _freeze_value(canonical[k])) for k in canonical.keys()))
        key = (nm, key_items)

        if key in self._vec_cache:
            return self._vec_cache[key]

        out_any = rdef.fn(store=self, **canonical)  # type: ignore[arg-type]
        require(isinstance(out_any, list), f"Regime {nm!r} must return List[float], got {type(out_any).__name__}")
        out: List[float] = out_any
        require(len(out) == self.n, f"Regime {nm!r} produced length {len(out)}, expected {self.n}")

        self._vec_cache[key] = out
        return out


# -----------------------------
# Regimes
# -----------------------------
def _regime_trend_dir(
    store: RegimeStore,
    *,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    di_min_sep: float = 0.0,
    ms_left: int = 2,
    ms_right: int = 2,
    tol_bps: float = 0.0,
) -> List[float]:
    """
    Trend direction in {-1,0,+1}.

    Conservative trend gating:
    - Requires ADX >= adx_threshold.

    Direction logic (DI-first, MS fallback):
    - If DI is available and abs(+DI - -DI) >= di_min_sep:
        direction = sign(+DI - -DI)
      else:
        direction = sign(MarketStructure trend_state)

    Rationale:
    - MarketStructure pivot-trend can become stale in monotonic impulses (no new pivots),
      while DI/ADX still captures direction. DI-first prevents stale MS from overriding.
    """
    p = _check_pos_int(adx_period, "adx_period")
    thr = _check_nonneg_float(adx_threshold, "adx_threshold")
    sep = _check_nonneg_float(di_min_sep, "di_min_sep")
    l = _check_pos_int(ms_left, "ms_left")
    r = _check_pos_int(ms_right, "ms_right")
    tb = _check_nonneg_float(tol_bps, "tol_bps")

    n = store.n
    out = [0.0] * n

    adx = store.ind.indicator("adx", period=p)
    pdi = store.ind.indicator("plus_di", period=p)
    mdi = store.ind.indicator("minus_di", period=p)

    ms_trend = store.ms.feature("trend_state", left=l, right=r, tol_bps=tb)

    for i in range(n):
        a = adx[i]
        if not _finite(a) or float(a) < thr:
            continue

        # DI direction if strong enough
        dir_di = 0
        if _finite(pdi[i]) and _finite(mdi[i]):
            diff = float(pdi[i]) - float(mdi[i])
            if abs(diff) >= sep:
                dir_di = 1 if diff > 0.0 else -1

        # Market structure as fallback
        dir_ms = _sign3(ms_trend[i]) if _finite(ms_trend[i]) else 0

        out[i] = float(dir_di if dir_di != 0 else dir_ms)

    return out


def _regime_vol_regime(
    store: RegimeStore,
    *,
    atr_period: int = 14,
    z_period: int = 200,
    z_threshold: float = 1.0,
) -> List[float]:
    """
    Volatility regime in {-1,0,+1} based on zscore(ATR/Close).
    """
    ap = _check_pos_int(atr_period, "atr_period")
    zp = _check_pos_int(z_period, "z_period")
    zt = _check_nonneg_float(z_threshold, "z_threshold")

    n = store.n
    out = [0.0] * n

    atr = store.ind.indicator("atr", period=ap)
    cl = store.ind.base("close")

    atr_pct = [_NAN] * n
    for i in range(n):
        a = atr[i]
        c = cl[i]
        if _finite(a) and _finite(c) and float(c) > 0.0:
            atr_pct[i] = float(a) / float(c)

    z = zscore(atr_pct, zp)
    for i in range(n):
        zi = z[i]
        if not _finite(zi):
            continue
        if float(zi) > zt:
            out[i] = 1.0
        elif float(zi) < -zt:
            out[i] = -1.0
        else:
            out[i] = 0.0

    return out


def _regime_volume_regime(
    store: RegimeStore,
    *,
    z_period: int = 200,
    z_threshold: float = 1.0,
) -> List[float]:
    """
    Volume regime in {-1,0,+1} based on zscore(volume).
    """
    zp = _check_pos_int(z_period, "z_period")
    zt = _check_nonneg_float(z_threshold, "z_threshold")

    n = store.n
    out = [0.0] * n
    vol = store.ind.base("volume")

    z = zscore(vol, zp)
    for i in range(n):
        zi = z[i]
        if not _finite(zi):
            continue
        if float(zi) > zt:
            out[i] = 1.0
        elif float(zi) < -zt:
            out[i] = -1.0
        else:
            out[i] = 0.0

    return out


def _regime_phase_id(
    store: RegimeStore,
    *,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    di_min_sep: float = 0.0,
    ms_left: int = 2,
    ms_right: int = 2,
    tol_bps: float = 0.0,
    vol_atr_period: int = 14,
    vol_z_period: int = 200,
    vol_z_threshold: float = 1.0,
    volume_z_period: int = 200,
    volume_z_threshold: float = 1.0,
) -> List[float]:
    """
    Combined phase id in 0..26:
      t = trend + 1  (0..2)
      v = vol + 1    (0..2)
      m = volume + 1 (0..2)
      phase = t*9 + v*3 + m
    """
    n = store.n
    trend = store.regime(
        "trend_dir",
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        di_min_sep=di_min_sep,
        ms_left=ms_left,
        ms_right=ms_right,
        tol_bps=tol_bps,
    )
    vol = store.regime(
        "vol_regime",
        atr_period=vol_atr_period,
        z_period=vol_z_period,
        z_threshold=vol_z_threshold,
    )
    volu = store.regime(
        "volume_regime",
        z_period=volume_z_period,
        z_threshold=volume_z_threshold,
    )

    out = [0.0] * n
    for i in range(n):
        t = _sign3(trend[i])
        v = _sign3(vol[i])
        m = _sign3(volu[i])
        out[i] = float((t + 1) * 9 + (v + 1) * 3 + (m + 1))
    return out


# Register regimes
_register_regime("trend_dir", _regime_trend_dir)
_register_regime("vol_regime", _regime_vol_regime)
_register_regime("volume_regime", _regime_volume_regime)
_register_regime("phase_id", _regime_phase_id)


# -----------------------------
# Utilities
# -----------------------------
def segments_from_labels(labels: Sequence[float]) -> List[Tuple[int, int, float]]:
    """
    Convert label series into contiguous segments:
      returns list of (start_idx, end_idx_exclusive, label)
    NaN labels are grouped together.
    """
    require(labels is not None, "labels must not be None")
    n = len(labels)
    if n == 0:
        return []
    segs: List[Tuple[int, int, float]] = []
    start = 0
    prev = float(labels[0])
    for i in range(1, n):
        cur = float(labels[i])
        if not _label_eq(cur, prev):
            segs.append((start, i, prev))
            start = i
            prev = cur
    segs.append((start, n, prev))
    return segs


def mask_equals(labels: Sequence[float], value: float) -> List[float]:
    """
    0/1 mask where labels == value (NaN-safe).
    """
    require(labels is not None, "labels must not be None")
    v = float(value)
    out = [0.0] * len(labels)
    for i, x in enumerate(labels):
        if _label_eq(float(x), v):
            out[i] = 1.0
    return out


# -----------------------------
# Self-test
# -----------------------------
def _self_test() -> None:
    def _expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    # Deterministic synthetic series with mild variability (avoids zero-std edge cases)
    n0, n1, n2 = 10, 10, 10
    n = n0 + n1 + n2

    close: List[float] = []
    noise = [0.0, 0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.04, -0.04, 0.0]
    for i in range(n0):
        close.append(100.0 + noise[i])
    for i in range(n1):
        close.append(100.0 + float(i) + 0.05 * ((i % 3) - 1))
    for i in range(n2):
        close.append(110.0 - float(i) + 0.05 * ((i % 3) - 1))

    open_ = close[:]
    high: List[float] = []
    low: List[float] = []
    volume: List[float] = []

    for i in range(n):
        c = close[i]
        if i < n0:
            rng = 0.20 + 0.02 * ((i % 3) - 1)  # 0.18/0.20/0.22
            volume.append(100.0 + 10.0 * ((i % 3) - 1))
        else:
            rng = 1.00 + 0.10 * ((i % 3) - 1)  # 0.90/1.00/1.10
            volume.append(500.0 + 20.0 * ((i % 4) - 1.5))  # mild variation
        high.append(c + rng)
        low.append(c - rng)

    series = OhlcvSeries(
        ts_ms=list(range(1, n + 1)),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        symbol="TEST",
        timeframe="1m",
    )

    store = RegimeStore(series)

    trend = store.regime(
        "trend_dir",
        adx_period=3,
        adx_threshold=40.0,
        di_min_sep=0.0,
        ms_left=1,
        ms_right=1,
        tol_bps=0.0,
    )
    trend2 = store.regime(
        "trend_dir",
        adx_period=3,
        adx_threshold=40.0,
        di_min_sep=0.0,
        ms_left=1,
        ms_right=1,
        tol_bps=0.0,
    )
    _expect(trend2 is trend, "RegimeStore cache failed for trend_dir")
    _expect(len(trend) == n, "trend_dir length mismatch")

    # end of range should be neutral with strict ADX threshold
    _expect(trend[n0 - 1] == 0.0, "Expected neutral trend at end of range segment")

    up_slice = trend[n0 + 5: n0 + n1]          # late uptrend
    dn_slice = trend[n0 + n1 + 5: n]           # late downtrend
    _expect(sum(1 for x in up_slice if x == 1.0) >= 3, f"Expected mostly +1 in late uptrend, got {up_slice}")
    _expect(sum(1 for x in dn_slice if x == -1.0) >= 3, f"Expected mostly -1 in late downtrend, got {dn_slice}")

    volreg = store.regime("vol_regime", atr_period=3, z_period=5, z_threshold=0.5)
    _expect(len(volreg) == n, "vol_regime length mismatch")
    _expect(sum(1 for x in volreg[n0: n0 + 6] if x == 1.0) >= 1, "Expected at least one high-vol detection near transition")

    volu = store.regime("volume_regime", z_period=5, z_threshold=0.5)
    _expect(len(volu) == n, "volume_regime length mismatch")
    _expect(sum(1 for x in volu[n0: n0 + 6] if x == 1.0) >= 1, "Expected at least one high-volume detection near transition")

    phase = store.regime(
        "phase_id",
        adx_period=3,
        adx_threshold=40.0,
        di_min_sep=0.0,
        ms_left=1,
        ms_right=1,
        tol_bps=0.0,
        vol_atr_period=3,
        vol_z_period=5,
        vol_z_threshold=0.5,
        volume_z_period=5,
        volume_z_threshold=0.5,
    )
    _expect(len(phase) == n, "phase_id length mismatch")
    for x in phase:
        _expect(0.0 <= float(x) <= 26.0, f"phase_id out of range: {x}")
        _expect(abs(float(x) - round(float(x))) < 1e-9, f"phase_id not integer-like: {x}")

    segs = segments_from_labels(trend)
    _expect(len(segs) > 0, "segments_from_labels returned empty")
    m = mask_equals(trend, 1.0)
    _expect(len(m) == n and sum(m) >= 1.0, "mask_equals failed for trend=+1")

    print("Regime_Classifier self-test: OK")


if __name__ == "__main__":
    _self_test()
