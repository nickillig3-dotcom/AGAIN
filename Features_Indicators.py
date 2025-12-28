from __future__ import annotations

"""
Features_Indicators.py
======================

Purpose
-------
A strict, OHLCV-only indicator/feature library with caching via FeatureStore.

Non-negotiable principles
-------------------------
- OHLCV only: everything must be derived from Open/High/Low/Close/Volume.
- No lookahead: indicator values at bar i may only depend on bars <= i.
- Warmup policy: undefined values are math.nan (never silently 0).
- Strict input checks and deterministic outputs.
- Caching: the same indicator (same parameters) is computed exactly once per series.

How to use
----------
store = FeatureStore(series)
rsi14 = store.indicator("rsi", period=14, source="close")
sma20 = store.indicator("sma", period=20, source="close")
bb_u  = store.indicator("bb_upper", period=20, mult=2.0, source="close")

Base series:
close = store.base("close")
hlc3  = store.base("hlc3")

Indicator registry
------------------
The store exposes an indicator registry so strategy mining can systematically
enumerate a large strategy space without re-computation.

Important note about NaNs
-------------------------
- Rolling functions propagate NaNs: a rolling window yields a value only if
  ALL values in the window are finite. Otherwise output is nan.
- This makes mining robust and avoids accidental use of undefined warmup values.
"""

from dataclasses import dataclass
import inspect
import math
from collections import deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, ValidationError, require


__all__ = [
    "FeatureStore",
    "INDICATOR_REGISTRY",
    # low-level vector functions (usable directly)
    "sma",
    "ema",
    "rma",
    "wma",
    "rolling_std",
    "zscore",
    "true_range",
    "atr",
    "rsi",
    "roc",
    "momentum",
    "stoch_k",
    "stoch_d",
    "obv",
    "mfi",
    "cmf",
    "vwma",
    "vwap_cum",
    "plus_di",
    "minus_di",
    "adx",
]


# -----------------------------
# Utilities
# -----------------------------
_NAN = float("nan")


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _isnan(x: float) -> bool:
    return math.isnan(float(x))


def _as_float(x: Any) -> float:
    return float(x)


def _ensure_len_match(out: Sequence[float], n: int, name: str) -> None:
    require(len(out) == n, f"{name} produced length {len(out)}, expected {n}")


def _check_period(period: int, *, name: str = "period") -> int:
    try:
        p = int(period)
    except Exception as e:
        raise ValidationError(f"{name} must be int-castable, got {period!r}") from e
    require(p > 0, f"{name} must be > 0, got {p}")
    return p


def _freeze_value(v: Any) -> Any:
    # Keep cache keys stable and hashable.
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
    # Fallback: stable stringification
    try:
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        return str(v)
    except Exception:
        return repr(v)


# -----------------------------
# Rolling helpers with NaN propagation
# -----------------------------
def _rolling_sum_count(x: Sequence[float], period: int) -> Tuple[List[float], List[int]]:
    """
    Returns (rolling_sum, rolling_count) where count is number of finite values in window.
    A window is "valid" iff count == period.
    """
    p = _check_period(period)
    n = len(x)
    out_sum = [_NAN] * n
    out_cnt = [0] * n

    s = 0.0
    cnt = 0

    for i in range(n):
        xi = x[i]
        if _finite(xi):
            s += float(xi)
            cnt += 1

        if i >= p:
            xold = x[i - p]
            if _finite(xold):
                s -= float(xold)
                cnt -= 1

        out_sum[i] = s
        out_cnt[i] = cnt

    return out_sum, out_cnt


def _rolling_sumsq_count(x: Sequence[float], period: int) -> Tuple[List[float], List[int]]:
    p = _check_period(period)
    n = len(x)
    out_sumsq = [_NAN] * n
    out_cnt = [0] * n

    s2 = 0.0
    cnt = 0

    for i in range(n):
        xi = x[i]
        if _finite(xi):
            xf = float(xi)
            s2 += xf * xf
            cnt += 1

        if i >= p:
            xold = x[i - p]
            if _finite(xold):
                xo = float(xold)
                s2 -= xo * xo
                cnt -= 1

        out_sumsq[i] = s2
        out_cnt[i] = cnt

    return out_sumsq, out_cnt


# -----------------------------
# Vector indicators (low-level)
# -----------------------------
def sma(x: Sequence[float], period: int) -> List[float]:
    """Simple moving average with NaN propagation."""
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n

    s, cnts = _rolling_sum_count(x, p)

    for i in range(n):
        if i >= p - 1 and cnts[i] == p:
            out[i] = float(s[i]) / float(p)

    return out


def ema(x: Sequence[float], period: int) -> List[float]:
    """
    Exponential moving average (classic alpha=2/(p+1)),
    seeded with SMA(p) at the first valid window.
    """
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n
    if n == 0:
        return out

    alpha = 2.0 / (p + 1.0)

    # Find first fully valid window using rolling sum+count
    s, cnts = _rolling_sum_count(x, p)
    seed_idx = None
    for i in range(n):
        if i >= p - 1 and cnts[i] == p:
            seed_idx = i
            out[i] = float(s[i]) / float(p)
            break

    if seed_idx is None:
        return out

    prev = out[seed_idx]
    for i in range(seed_idx + 1, n):
        xi = x[i]
        if not _finite(xi) or not _finite(prev):
            out[i] = _NAN
            prev = _NAN
            continue
        prev = alpha * float(xi) + (1.0 - alpha) * float(prev)
        out[i] = prev

    return out


def rma(x: Sequence[float], period: int) -> List[float]:
    """
    Wilder's RMA (a.k.a. SMMA):
      rma[i] = (rma[i-1]*(p-1) + x[i]) / p
    seeded with SMA(p) at the first valid window.
    """
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n
    if n == 0:
        return out

    s, cnts = _rolling_sum_count(x, p)
    seed_idx = None
    for i in range(n):
        if i >= p - 1 and cnts[i] == p:
            seed_idx = i
            out[i] = float(s[i]) / float(p)
            break

    if seed_idx is None:
        return out

    prev = out[seed_idx]
    for i in range(seed_idx + 1, n):
        xi = x[i]
        if not _finite(xi) or not _finite(prev):
            out[i] = _NAN
            prev = _NAN
            continue
        prev = (float(prev) * (p - 1.0) + float(xi)) / float(p)
        out[i] = prev

    return out


def wma(x: Sequence[float], period: int) -> List[float]:
    """
    Weighted moving average with weights 1..p (linear), NaN propagation.
    Computed in O(n) after initial seed per valid region.
    """
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n
    if n == 0:
        return out

    denom = (p * (p + 1)) / 2.0

    # rolling sum and count for validity
    s, cnts = _rolling_sum_count(x, p)

    wsum: Optional[float] = None
    prev_sum: Optional[float] = None
    prev_valid = False

    for i in range(n):
        if i < p - 1 or cnts[i] != p:
            out[i] = _NAN
            wsum = None
            prev_sum = None
            prev_valid = False
            continue

        # current window [i-p+1 .. i] is fully finite
        if (not prev_valid) or wsum is None or prev_sum is None:
            # seed: compute weighted sum from scratch
            w = 0.0
            start = i - p + 1
            for k in range(p):
                w += (k + 1) * float(x[start + k])
            wsum = w
            prev_sum = float(s[i])  # sum of current window
        else:
            # incremental update:
            # W' = (W - S_old) + p*x_new, where S_old is previous window sum.
            x_new = float(x[i])
            wsum = float(wsum) - float(prev_sum) + float(p) * x_new
            prev_sum = float(s[i])

        out[i] = float(wsum) / denom
        prev_valid = True

    return out


def rolling_std(x: Sequence[float], period: int) -> List[float]:
    """Rolling standard deviation, NaN propagation."""
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n

    s, cnts = _rolling_sum_count(x, p)
    s2, cnt2 = _rolling_sumsq_count(x, p)

    for i in range(n):
        if i >= p - 1 and cnts[i] == p and cnt2[i] == p:
            mean = float(s[i]) / float(p)
            var = (float(s2[i]) / float(p)) - (mean * mean)
            if var < 0.0 and var > -1e-18:
                var = 0.0
            out[i] = math.sqrt(var) if var >= 0.0 else _NAN

    return out


def zscore(x: Sequence[float], period: int) -> List[float]:
    """Rolling z-score: (x - mean) / std, NaN propagation."""
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n

    mu = sma(x, p)
    sd = rolling_std(x, p)

    for i in range(n):
        xi = x[i]
        if _finite(xi) and _finite(mu[i]) and _finite(sd[i]) and float(sd[i]) > 0.0:
            out[i] = (float(xi) - float(mu[i])) / float(sd[i])

    return out


def _rolling_max(x: Sequence[float], period: int) -> List[float]:
    """Rolling max using a monotonic deque. Requires finite inputs."""
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n
    dq: deque[int] = deque()

    for i in range(n):
        xi = x[i]
        require(_finite(xi), f"rolling_max requires finite inputs; got x[{i}]={xi!r}")

        while dq and float(x[dq[-1]]) <= float(xi):
            dq.pop()
        dq.append(i)

        # remove out of window
        if dq[0] <= i - p:
            dq.popleft()

        if i >= p - 1:
            out[i] = float(x[dq[0]])

    return out


def _rolling_min(x: Sequence[float], period: int) -> List[float]:
    """Rolling min using a monotonic deque. Requires finite inputs."""
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n
    dq: deque[int] = deque()

    for i in range(n):
        xi = x[i]
        require(_finite(xi), f"rolling_min requires finite inputs; got x[{i}]={xi!r}")

        while dq and float(x[dq[-1]]) >= float(xi):
            dq.pop()
        dq.append(i)

        if dq[0] <= i - p:
            dq.popleft()

        if i >= p - 1:
            out[i] = float(x[dq[0]])

    return out


def true_range(high: Sequence[float], low: Sequence[float], close: Sequence[float]) -> List[float]:
    """True Range (TR). TR[0]=high-low. For i>0: max(h-l, abs(h-prev_close), abs(l-prev_close))."""
    n = len(close)
    require(len(high) == n and len(low) == n, "true_range: high/low/close length mismatch")
    out = [_NAN] * n
    if n == 0:
        return out

    h0 = float(high[0]); l0 = float(low[0]); c0 = float(close[0])
    require(_finite(h0) and _finite(l0) and _finite(c0), "true_range: non-finite input at index 0")
    out[0] = abs(h0 - l0)

    for i in range(1, n):
        hi = float(high[i]); lo = float(low[i]); pc = float(close[i - 1])
        require(_finite(hi) and _finite(lo) and _finite(pc), f"true_range: non-finite input at index {i}")
        out[i] = max(abs(hi - lo), abs(hi - pc), abs(lo - pc))

    return out


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> List[float]:
    """Average True Range using Wilder RMA on TR."""
    tr = true_range(high, low, close)
    return rma(tr, period)


def roc(x: Sequence[float], period: int) -> List[float]:
    """
    Rate of change in percent:
      ROC[i] = 100 * (x[i] / x[i-p] - 1)
    """
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n
    for i in range(n):
        if i < p:
            continue
        xi = x[i]
        x0 = x[i - p]
        if _finite(xi) and _finite(x0) and float(x0) != 0.0:
            out[i] = 100.0 * (float(xi) / float(x0) - 1.0)
    return out


def momentum(x: Sequence[float], period: int) -> List[float]:
    """Momentum: x[i] - x[i-p]."""
    p = _check_period(period)
    n = len(x)
    out = [_NAN] * n
    for i in range(n):
        if i < p:
            continue
        xi = x[i]
        x0 = x[i - p]
        if _finite(xi) and _finite(x0):
            out[i] = float(xi) - float(x0)
    return out


def rsi(close: Sequence[float], period: int) -> List[float]:
    """RSI using Wilder smoothing. Warmup -> nan; after that [0..100]."""
    p = _check_period(period)
    n = len(close)
    out = [_NAN] * n
    if n == 0:
        return out

    gains = [_NAN] * n
    losses = [_NAN] * n

    # delta starts at i=1; i=0 is undefined -> keep nan so warmup behaves strictly
    for i in range(1, n):
        c0 = close[i - 1]
        c1 = close[i]
        if not (_finite(c0) and _finite(c1)):
            gains[i] = _NAN
            losses[i] = _NAN
            continue
        d = float(c1) - float(c0)
        gains[i] = d if d > 0.0 else 0.0
        losses[i] = (-d) if d < 0.0 else 0.0

    avg_gain = rma(gains, p)
    avg_loss = rma(losses, p)

    for i in range(n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if not (_finite(ag) and _finite(al)):
            continue
        if float(al) == 0.0 and float(ag) == 0.0:
            out[i] = 50.0
        elif float(al) == 0.0:
            out[i] = 100.0
        elif float(ag) == 0.0:
            out[i] = 0.0
        else:
            rs = float(ag) / float(al)
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


def stoch_k(high: Sequence[float], low: Sequence[float], close: Sequence[float], k_period: int) -> List[float]:
    """
    Stochastic %K:
      %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
    """
    kp = _check_period(k_period, name="k_period")
    n = len(close)
    require(len(high) == n and len(low) == n, "stoch_k: high/low/close length mismatch")
    out = [_NAN] * n
    if n == 0:
        return out

    hh = _rolling_max(high, kp)
    ll = _rolling_min(low, kp)

    for i in range(n):
        if i < kp - 1:
            continue
        if not (_finite(hh[i]) and _finite(ll[i]) and _finite(close[i])):
            continue
        denom = float(hh[i]) - float(ll[i])
        if denom <= 0.0:
            out[i] = 0.0
        else:
            out[i] = 100.0 * (float(close[i]) - float(ll[i])) / denom

    return out


def stoch_d(high: Sequence[float], low: Sequence[float], close: Sequence[float], k_period: int, d_period: int) -> List[float]:
    """Stochastic %D = SMA(%K, d_period)."""
    dp = _check_period(d_period, name="d_period")
    k = stoch_k(high, low, close, k_period)
    return sma(k, dp)


def obv(close: Sequence[float], volume: Sequence[float]) -> List[float]:
    """On-Balance Volume (OBV). OBV[0]=0."""
    n = len(close)
    require(len(volume) == n, "obv: close/volume length mismatch")
    out = [_NAN] * n
    if n == 0:
        return out

    out[0] = 0.0
    for i in range(1, n):
        c0 = close[i - 1]
        c1 = close[i]
        v1 = volume[i]
        if not (_finite(c0) and _finite(c1) and _finite(v1)):
            out[i] = _NAN
            continue
        prev = out[i - 1]
        if not _finite(prev):
            out[i] = _NAN
            continue
        if float(c1) > float(c0):
            out[i] = float(prev) + float(v1)
        elif float(c1) < float(c0):
            out[i] = float(prev) - float(v1)
        else:
            out[i] = float(prev)
    return out


def mfi(high: Sequence[float], low: Sequence[float], close: Sequence[float], volume: Sequence[float], period: int) -> List[float]:
    """
    Money Flow Index (MFI), period-based, outputs 0..100.
    Uses typical price (hlc3) and raw money flow tp*volume.
    """
    p = _check_period(period)
    n = len(close)
    require(len(high) == n and len(low) == n and len(volume) == n, "mfi: length mismatch")
    out = [_NAN] * n
    if n == 0:
        return out

    tp = [_NAN] * n
    rmf = [_NAN] * n
    for i in range(n):
        hi = high[i]; lo = low[i]; cl = close[i]; vol = volume[i]
        if _finite(hi) and _finite(lo) and _finite(cl) and _finite(vol):
            tp[i] = (float(hi) + float(lo) + float(cl)) / 3.0
            rmf[i] = float(tp[i]) * float(vol)

    pos = [_NAN] * n
    neg = [_NAN] * n
    # i=0 undefined direction -> nan
    for i in range(1, n):
        if not (_finite(tp[i]) and _finite(tp[i - 1]) and _finite(rmf[i])):
            continue
        if float(tp[i]) > float(tp[i - 1]):
            pos[i] = float(rmf[i])
            neg[i] = 0.0
        elif float(tp[i]) < float(tp[i - 1]):
            pos[i] = 0.0
            neg[i] = float(rmf[i])
        else:
            pos[i] = 0.0
            neg[i] = 0.0

    pos_sum, pos_cnt = _rolling_sum_count(pos, p)
    neg_sum, neg_cnt = _rolling_sum_count(neg, p)

    for i in range(n):
        if i < p:
            continue
        if pos_cnt[i] != p or neg_cnt[i] != p:
            continue
        ps = float(pos_sum[i])
        ns = float(neg_sum[i])
        if ns == 0.0 and ps == 0.0:
            out[i] = 50.0
        elif ns == 0.0:
            out[i] = 100.0
        elif ps == 0.0:
            out[i] = 0.0
        else:
            mr = ps / ns
            out[i] = 100.0 - (100.0 / (1.0 + mr))

    return out


def cmf(high: Sequence[float], low: Sequence[float], close: Sequence[float], volume: Sequence[float], period: int) -> List[float]:
    """
    Chaikin Money Flow (CMF):
      CMF = sum(MFV, p) / sum(volume, p)
    Where MFV = MFM * volume and
      MFM = ((close-low) - (high-close)) / (high-low)
    """
    p = _check_period(period)
    n = len(close)
    require(len(high) == n and len(low) == n and len(volume) == n, "cmf: length mismatch")
    out = [_NAN] * n
    if n == 0:
        return out

    mfv = [_NAN] * n
    vol = [_NAN] * n

    for i in range(n):
        hi = high[i]; lo = low[i]; cl = close[i]; v = volume[i]
        if not (_finite(hi) and _finite(lo) and _finite(cl) and _finite(v)):
            continue
        hi_f, lo_f, cl_f, v_f = float(hi), float(lo), float(cl), float(v)
        denom = hi_f - lo_f
        if denom == 0.0:
            mfm = 0.0
        else:
            mfm = ((cl_f - lo_f) - (hi_f - cl_f)) / denom
        mfv[i] = mfm * v_f
        vol[i] = v_f

    mfv_sum, mfv_cnt = _rolling_sum_count(mfv, p)
    vol_sum, vol_cnt = _rolling_sum_count(vol, p)

    for i in range(n):
        if i < p - 1:
            continue
        if mfv_cnt[i] != p or vol_cnt[i] != p:
            continue
        vs = float(vol_sum[i])
        if vs == 0.0:
            out[i] = _NAN
        else:
            out[i] = float(mfv_sum[i]) / vs

    return out


def vwma(price: Sequence[float], volume: Sequence[float], period: int) -> List[float]:
    """Volume-weighted moving average over a rolling window."""
    p = _check_period(period)
    n = len(price)
    require(len(volume) == n, "vwma: price/volume length mismatch")
    out = [_NAN] * n

    pv = [_NAN] * n
    vv = [_NAN] * n
    for i in range(n):
        pi = price[i]
        vi = volume[i]
        if _finite(pi) and _finite(vi):
            pv[i] = float(pi) * float(vi)
            vv[i] = float(vi)

    pv_sum, pv_cnt = _rolling_sum_count(pv, p)
    v_sum, v_cnt = _rolling_sum_count(vv, p)

    for i in range(n):
        if i < p - 1:
            continue
        if pv_cnt[i] != p or v_cnt[i] != p:
            continue
        denom = float(v_sum[i])
        if denom == 0.0:
            out[i] = _NAN
        else:
            out[i] = float(pv_sum[i]) / denom

    return out


def vwap_cum(price: Sequence[float], volume: Sequence[float]) -> List[float]:
    """
    Cumulative VWAP (sessionless):
      vwap[i] = sum(price*volume up to i) / sum(volume up to i)
    """
    n = len(price)
    require(len(volume) == n, "vwap_cum: price/volume length mismatch")
    out = [_NAN] * n
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        pi = price[i]
        vi = volume[i]
        if not (_finite(pi) and _finite(vi)):
            out[i] = _NAN
            continue
        cum_pv += float(pi) * float(vi)
        cum_v += float(vi)
        out[i] = (cum_pv / cum_v) if cum_v > 0.0 else _NAN
    return out


def _dmi_bundle(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute (+DI, -DI, ADX) bundle using Wilder smoothing.
    """
    p = _check_period(period)
    n = len(close)
    require(len(high) == n and len(low) == n, "_dmi_bundle: length mismatch")

    # TR, +DM, -DM
    tr = [_NAN] * n
    pdm = [_NAN] * n
    mdm = [_NAN] * n

    # i=0 undefined for DM; TR[0]=high-low
    h0 = float(high[0]); l0 = float(low[0]); c0 = float(close[0])
    require(_finite(h0) and _finite(l0) and _finite(c0), "_dmi_bundle: non-finite index 0")
    tr[0] = abs(h0 - l0)

    for i in range(1, n):
        hi = float(high[i]); lo = float(low[i]); pc = float(close[i - 1])
        hi_1 = float(high[i - 1]); lo_1 = float(low[i - 1])

        require(_finite(hi) and _finite(lo) and _finite(pc) and _finite(hi_1) and _finite(lo_1), f"_dmi_bundle: non-finite at {i}")

        up = hi - hi_1
        down = lo_1 - lo

        pdm[i] = up if (up > down and up > 0.0) else 0.0
        mdm[i] = down if (down > up and down > 0.0) else 0.0
        tr[i] = max(abs(hi - lo), abs(hi - pc), abs(lo - pc))

    sm_tr = rma(tr, p)
    sm_pdm = rma(pdm, p)
    sm_mdm = rma(mdm, p)

    plus_di_out = [_NAN] * n
    minus_di_out = [_NAN] * n
    dx = [_NAN] * n

    for i in range(n):
        if not (_finite(sm_tr[i]) and float(sm_tr[i]) > 0.0 and _finite(sm_pdm[i]) and _finite(sm_mdm[i])):
            continue
        pdi = 100.0 * float(sm_pdm[i]) / float(sm_tr[i])
        mdi = 100.0 * float(sm_mdm[i]) / float(sm_tr[i])
        plus_di_out[i] = pdi
        minus_di_out[i] = mdi

        denom = pdi + mdi
        if denom > 0.0:
            dx[i] = 100.0 * abs(pdi - mdi) / denom

    adx_out = rma(dx, p)
    return plus_di_out, minus_di_out, adx_out


def plus_di(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> List[float]:
    return _dmi_bundle(high, low, close, period)[0]


def minus_di(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> List[float]:
    return _dmi_bundle(high, low, close, period)[1]


def adx(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> List[float]:
    return _dmi_bundle(high, low, close, period)[2]


# -----------------------------
# FeatureStore + Registry
# -----------------------------
IndicatorFn = Callable[..., List[float]]


@dataclass(frozen=True, slots=True)
class _IndicatorDef:
    name: str
    fn: Callable[..., Any]
    param_defaults: Dict[str, Any]
    required_params: Tuple[str, ...]
    allowed_params: Tuple[str, ...]


INDICATOR_REGISTRY: Dict[str, _IndicatorDef] = {}


def _register_indicator(name: str, fn: Callable[..., Any]) -> None:
    nm = str(name).strip().lower()
    require(nm, "indicator name must be non-empty")
    require(nm not in INDICATOR_REGISTRY, f"indicator {nm!r} already registered")

    sig = inspect.signature(fn)
    param_defaults: Dict[str, Any] = {}
    required: List[str] = []
    allowed: List[str] = []

    for pname, p in sig.parameters.items():
        if pname == "store":
            continue
        allowed.append(pname)
        if p.default is inspect._empty:
            required.append(pname)
        else:
            param_defaults[pname] = p.default

    INDICATOR_REGISTRY[nm] = _IndicatorDef(
        name=nm,
        fn=fn,
        param_defaults=param_defaults,
        required_params=tuple(required),
        allowed_params=tuple(allowed),
    )


class FeatureStore:
    """
    Caches base series and computed indicators for a given OHLCV series.
    """

    def __init__(self, series: OhlcvSeries) -> None:
        require(series is not None, "series must not be None")
        # series is expected to be validated earlier; we still rely on its contract.
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
        n = len(s.ts_ms)

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
        elif nm == "hl2":
            hi = self.base("high")
            lo = self.base("low")
            out = [(hi[i] + lo[i]) / 2.0 for i in range(n)]
        elif nm == "hlc3":
            hi = self.base("high")
            lo = self.base("low")
            cl = self.base("close")
            out = [(hi[i] + lo[i] + cl[i]) / 3.0 for i in range(n)]
        elif nm == "ohlc4":
            op = self.base("open")
            hi = self.base("high")
            lo = self.base("low")
            cl = self.base("close")
            out = [(op[i] + hi[i] + lo[i] + cl[i]) / 4.0 for i in range(n)]
        else:
            raise ValidationError(f"Unknown base series {name!r}. Supported: open/high/low/close/volume/hl2/hlc3/ohlc4")

        _ensure_len_match(out, n, f"base({nm})")
        self._base_cache[nm] = out
        return out

    def indicator(self, name: str, **params: Any) -> List[float]:
        nm = str(name).strip().lower()
        require(nm in INDICATOR_REGISTRY, f"Unknown indicator {name!r}. Available: {sorted(INDICATOR_REGISTRY.keys())}")
        idef = INDICATOR_REGISTRY[nm]

        # Validate params
        for k in params.keys():
            require(k in idef.allowed_params, f"Indicator {nm!r} does not accept param {k!r}. Allowed: {idef.allowed_params}")

        # Canonicalize: fill defaults
        canonical: Dict[str, Any] = dict(idef.param_defaults)
        canonical.update(params)

        for req in idef.required_params:
            require(req in canonical, f"Indicator {nm!r} missing required param {req!r}")

        key_items = tuple(sorted((k, _freeze_value(canonical[k])) for k in canonical.keys()))
        key = (nm, key_items)

        if key in self._vec_cache:
            return self._vec_cache[key]

        out_any = idef.fn(store=self, **canonical)  # type: ignore[arg-type]
        require(isinstance(out_any, list), f"Indicator {nm!r} must return a List[float], got {type(out_any).__name__}")
        out: List[float] = out_any

        _ensure_len_match(out, self.n, f"indicator({nm})")
        self._vec_cache[key] = out
        return out

    # Bundle caching (multi-output indicators)
    def _bundle(self, bundle_name: str, **params: Any) -> Any:
        bnm = str(bundle_name).strip().lower()
        key_items = tuple(sorted((k, _freeze_value(v)) for k, v in params.items()))
        key = (bnm, key_items)
        if key in self._bundle_cache:
            return self._bundle_cache[key]
        raise ValidationError(f"Bundle {bundle_name!r} not computed. Use a bundle-computing indicator function.")


# -----------------------------
# Registry-backed indicator functions (store-aware)
# -----------------------------
def _ind_returns(store: FeatureStore) -> List[float]:
    cl = store.base("close")
    n = store.n
    out = [_NAN] * n
    if n == 0:
        return out
    for i in range(1, n):
        c0 = cl[i - 1]
        c1 = cl[i]
        if _finite(c0) and _finite(c1) and c0 != 0.0:
            out[i] = (c1 / c0) - 1.0
    return out


def _ind_log_returns(store: FeatureStore) -> List[float]:
    cl = store.base("close")
    n = store.n
    out = [_NAN] * n
    if n == 0:
        return out
    for i in range(1, n):
        c0 = cl[i - 1]
        c1 = cl[i]
        if _finite(c0) and _finite(c1) and c0 > 0.0 and c1 > 0.0:
            out[i] = math.log(c1 / c0)
    return out


def _ind_sma(store: FeatureStore, *, source: str = "close", period: int = 14) -> List[float]:
    x = store.base(source)
    return sma(x, period)


def _ind_ema(store: FeatureStore, *, source: str = "close", period: int = 14) -> List[float]:
    x = store.base(source)
    return ema(x, period)


def _ind_rma(store: FeatureStore, *, source: str = "close", period: int = 14) -> List[float]:
    x = store.base(source)
    return rma(x, period)


def _ind_wma(store: FeatureStore, *, source: str = "close", period: int = 14) -> List[float]:
    x = store.base(source)
    return wma(x, period)


def _ind_std(store: FeatureStore, *, source: str = "close", period: int = 14) -> List[float]:
    x = store.base(source)
    return rolling_std(x, period)


def _ind_zscore(store: FeatureStore, *, source: str = "close", period: int = 14) -> List[float]:
    x = store.base(source)
    return zscore(x, period)


def _bb_bundle(store: FeatureStore, *, source: str = "close", period: int = 20, mult: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    x = store.base(source)
    mid = sma(x, period)
    sd = rolling_std(x, period)
    n = store.n
    upper = [_NAN] * n
    lower = [_NAN] * n
    for i in range(n):
        if _finite(mid[i]) and _finite(sd[i]):
            upper[i] = float(mid[i]) + float(mult) * float(sd[i])
            lower[i] = float(mid[i]) - float(mult) * float(sd[i])
    return mid, upper, lower


def _ind_bb_mid(store: FeatureStore, *, source: str = "close", period: int = 20, mult: float = 2.0) -> List[float]:
    bkey = ("bb", tuple(sorted((("source", source), ("period", int(period)), ("mult", float(mult))))))
    if bkey in store._bundle_cache:
        return store._bundle_cache[bkey][0]
    bundle = _bb_bundle(store, source=source, period=period, mult=mult)
    store._bundle_cache[bkey] = bundle
    return bundle[0]


def _ind_bb_upper(store: FeatureStore, *, source: str = "close", period: int = 20, mult: float = 2.0) -> List[float]:
    bkey = ("bb", tuple(sorted((("source", source), ("period", int(period)), ("mult", float(mult))))))
    if bkey in store._bundle_cache:
        return store._bundle_cache[bkey][1]
    bundle = _bb_bundle(store, source=source, period=period, mult=mult)
    store._bundle_cache[bkey] = bundle
    return bundle[1]


def _ind_bb_lower(store: FeatureStore, *, source: str = "close", period: int = 20, mult: float = 2.0) -> List[float]:
    bkey = ("bb", tuple(sorted((("source", source), ("period", int(period)), ("mult", float(mult))))))
    if bkey in store._bundle_cache:
        return store._bundle_cache[bkey][2]
    bundle = _bb_bundle(store, source=source, period=period, mult=mult)
    store._bundle_cache[bkey] = bundle
    return bundle[2]


def _ind_donchian_high(store: FeatureStore, *, period: int = 20) -> List[float]:
    hi = store.base("high")
    return _rolling_max(hi, period)


def _ind_donchian_low(store: FeatureStore, *, period: int = 20) -> List[float]:
    lo = store.base("low")
    return _rolling_min(lo, period)


def _ind_true_range(store: FeatureStore) -> List[float]:
    hi = store.base("high")
    lo = store.base("low")
    cl = store.base("close")
    return true_range(hi, lo, cl)


def _ind_atr(store: FeatureStore, *, period: int = 14) -> List[float]:
    hi = store.base("high")
    lo = store.base("low")
    cl = store.base("close")
    return atr(hi, lo, cl, period)


def _ind_rsi(store: FeatureStore, *, period: int = 14, source: str = "close") -> List[float]:
    x = store.base(source)
    return rsi(x, period)


def _ind_roc(store: FeatureStore, *, period: int = 12, source: str = "close") -> List[float]:
    x = store.base(source)
    return roc(x, period)


def _ind_momentum(store: FeatureStore, *, period: int = 10, source: str = "close") -> List[float]:
    x = store.base(source)
    return momentum(x, period)


def _stoch_bundle(store: FeatureStore, *, k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
    hi = store.base("high")
    lo = store.base("low")
    cl = store.base("close")
    k = stoch_k(hi, lo, cl, k_period)
    d = sma(k, d_period)
    return k, d


def _ind_stoch_k(store: FeatureStore, *, k_period: int = 14, d_period: int = 3) -> List[float]:
    bkey = ("stoch", tuple(sorted((("k_period", int(k_period)), ("d_period", int(d_period))))))
    if bkey in store._bundle_cache:
        return store._bundle_cache[bkey][0]
    bundle = _stoch_bundle(store, k_period=k_period, d_period=d_period)
    store._bundle_cache[bkey] = bundle
    return bundle[0]


def _ind_stoch_d(store: FeatureStore, *, k_period: int = 14, d_period: int = 3) -> List[float]:
    bkey = ("stoch", tuple(sorted((("k_period", int(k_period)), ("d_period", int(d_period))))))
    if bkey in store._bundle_cache:
        return store._bundle_cache[bkey][1]
    bundle = _stoch_bundle(store, k_period=k_period, d_period=d_period)
    store._bundle_cache[bkey] = bundle
    return bundle[1]


def _ind_obv(store: FeatureStore) -> List[float]:
    cl = store.base("close")
    vol = store.base("volume")
    return obv(cl, vol)


def _ind_mfi(store: FeatureStore, *, period: int = 14) -> List[float]:
    hi = store.base("high")
    lo = store.base("low")
    cl = store.base("close")
    vol = store.base("volume")
    return mfi(hi, lo, cl, vol, period)


def _ind_cmf(store: FeatureStore, *, period: int = 20) -> List[float]:
    hi = store.base("high")
    lo = store.base("low")
    cl = store.base("close")
    vol = store.base("volume")
    return cmf(hi, lo, cl, vol, period)


def _ind_vwma(store: FeatureStore, *, period: int = 20, source: str = "close") -> List[float]:
    x = store.base(source)
    vol = store.base("volume")
    return vwma(x, vol, period)


def _ind_vwap_cum(store: FeatureStore, *, source: str = "hlc3") -> List[float]:
    x = store.base(source)
    vol = store.base("volume")
    return vwap_cum(x, vol)


def _dmi_cached(store: FeatureStore, *, period: int = 14) -> Tuple[List[float], List[float], List[float]]:
    bkey = ("dmi", tuple(sorted((("period", int(period)),))))
    if bkey in store._bundle_cache:
        return store._bundle_cache[bkey]
    hi = store.base("high")
    lo = store.base("low")
    cl = store.base("close")
    bundle = _dmi_bundle(hi, lo, cl, period)
    store._bundle_cache[bkey] = bundle
    return bundle


def _ind_plus_di(store: FeatureStore, *, period: int = 14) -> List[float]:
    return _dmi_cached(store, period=period)[0]


def _ind_minus_di(store: FeatureStore, *, period: int = 14) -> List[float]:
    return _dmi_cached(store, period=period)[1]


def _ind_adx(store: FeatureStore, *, period: int = 14) -> List[float]:
    return _dmi_cached(store, period=period)[2]
# -----------------------------
# MACD (bundle)
# -----------------------------
def _macd_cached(
    store: FeatureStore,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[List[float], List[float], List[float]]:
    """Cached MACD bundle: (line, signal, histogram)."""

    fp = _check_period(fast_period, name="fast_period")
    sp = _check_period(slow_period, name="slow_period")
    sg = _check_period(signal_period, name="signal_period")
    require(fp < sp, f"fast_period must be < slow_period, got fast={fp}, slow={sp}")

    bkey = (
        "macd",
        tuple(sorted((
            ("source", str(source)),
            ("fast_period", int(fp)),
            ("slow_period", int(sp)),
            ("signal_period", int(sg)),
        ))),
    )
    if bkey in store._bundle_cache:
        return store._bundle_cache[bkey]

    x = store.base(source)
    fast = ema(x, fp)
    slow = ema(x, sp)

    n = store.n
    line = [_NAN] * n
    for i in range(n):
        if _finite(fast[i]) and _finite(slow[i]):
            line[i] = float(fast[i]) - float(slow[i])

    signal = ema(line, sg)

    hist = [_NAN] * n
    for i in range(n):
        if _finite(line[i]) and _finite(signal[i]):
            hist[i] = float(line[i]) - float(signal[i])

    bundle = (line, signal, hist)
    store._bundle_cache[bkey] = bundle
    return bundle


def _ind_macd_line(
    store: FeatureStore,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> List[float]:
    return _macd_cached(
        store,
        source=source,
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
    )[0]


def _ind_macd_signal(
    store: FeatureStore,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> List[float]:
    return _macd_cached(
        store,
        source=source,
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
    )[1]


def _ind_macd_hist(
    store: FeatureStore,
    *,
    source: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> List[float]:
    return _macd_cached(
        store,
        source=source,
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period,
    )[2]


# Register indicators
_register_indicator("returns", _ind_returns)
_register_indicator("log_returns", _ind_log_returns)

_register_indicator("sma", _ind_sma)
_register_indicator("ema", _ind_ema)
_register_indicator("rma", _ind_rma)
_register_indicator("wma", _ind_wma)

_register_indicator("std", _ind_std)
_register_indicator("zscore", _ind_zscore)

_register_indicator("bb_mid", _ind_bb_mid)
_register_indicator("bb_upper", _ind_bb_upper)
_register_indicator("bb_lower", _ind_bb_lower)

_register_indicator("donchian_high", _ind_donchian_high)
_register_indicator("donchian_low", _ind_donchian_low)

_register_indicator("true_range", _ind_true_range)
_register_indicator("atr", _ind_atr)

_register_indicator("rsi", _ind_rsi)
_register_indicator("roc", _ind_roc)
_register_indicator("momentum", _ind_momentum)
_register_indicator("macd_line", _ind_macd_line)
_register_indicator("macd_signal", _ind_macd_signal)
_register_indicator("macd_hist", _ind_macd_hist)

_register_indicator("stoch_k", _ind_stoch_k)
_register_indicator("stoch_d", _ind_stoch_d)

_register_indicator("obv", _ind_obv)
_register_indicator("mfi", _ind_mfi)
_register_indicator("cmf", _ind_cmf)
_register_indicator("vwma", _ind_vwma)
_register_indicator("vwap_cum", _ind_vwap_cum)

_register_indicator("plus_di", _ind_plus_di)
_register_indicator("minus_di", _ind_minus_di)
_register_indicator("adx", _ind_adx)


# -----------------------------
# Minimal self-test
# -----------------------------
def _self_test() -> None:
    def _expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    # Synthetic, deterministic OHLCV
    # close: 1,2,3,4,5
    cl = [1.0, 2.0, 3.0, 4.0, 5.0]
    series = OhlcvSeries(
        ts_ms=[1, 2, 3, 4, 5],
        open=[1.0, 2.0, 3.0, 4.0, 5.0],
        high=[2.0, 3.0, 4.0, 5.0, 6.0],
        low=[0.0, 1.0, 2.0, 3.0, 4.0],
        close=cl,
        volume=[10.0, 10.0, 10.0, 10.0, 10.0],
        symbol="TEST",
        timeframe="1m",
    )

    store = FeatureStore(series)

    # SMA(3) on [1,2,3,4,5] => [nan,nan,2,3,4]
    sma3 = store.indicator("sma", source="close", period=3)
    _expect(_isnan(sma3[0]) and _isnan(sma3[1]), "SMA warmup expected nan")
    _expect(abs(sma3[2] - 2.0) < 1e-12, f"SMA[2] expected 2.0 got {sma3[2]}")
    _expect(abs(sma3[4] - 4.0) < 1e-12, f"SMA[4] expected 4.0 got {sma3[4]}")

    # Cache: exact same call returns same object reference
    sma3_again = store.indicator("sma", source="close", period=3)
    _expect(sma3_again is sma3, "FeatureStore cache failed: expected same object for same indicator+params")

    # EMA(3) with SMA seed on linear series will match [nan,nan,2,3,4]
    ema3 = store.indicator("ema", source="close", period=3)
    _expect(_isnan(ema3[0]) and _isnan(ema3[1]), "EMA warmup expected nan")
    _expect(abs(ema3[2] - 2.0) < 1e-12, f"EMA[2] expected 2.0 got {ema3[2]}")
    _expect(abs(ema3[4] - 4.0) < 1e-12, f"EMA[4] expected 4.0 got {ema3[4]}")

    # RSI on strictly increasing should hit 100 after warmup (period=3 -> last index should be 100)
    rsi3 = store.indicator("rsi", source="close", period=3)
    _expect(_finite(rsi3[4]) and abs(rsi3[4] - 100.0) < 1e-9, f"RSI expected ~100 got {rsi3[4]}")

    # ATR on this synthetic series: high-low is constant 2; no gaps; TR constant 2 => ATR constant 2 after warmup.
    atr3 = store.indicator("atr", period=3)
    _expect(_finite(atr3[2]) and abs(atr3[2] - 2.0) < 1e-12, f"ATR[2] expected 2 got {atr3[2]}")
    _expect(_finite(atr3[4]) and abs(atr3[4] - 2.0) < 1e-12, f"ATR[4] expected 2 got {atr3[4]}")

    # Bollinger mid equals SMA, upper/lower finite after warmup
    bbm = store.indicator("bb_mid", source="close", period=3, mult=2.0)
    bbu = store.indicator("bb_upper", source="close", period=3, mult=2.0)
    bbl = store.indicator("bb_lower", source="close", period=3, mult=2.0)
    _expect(_isnan(bbm[1]) and _isnan(bbu[1]) and _isnan(bbl[1]), "BB warmup expected nan")
    _expect(_finite(bbm[4]) and _finite(bbu[4]) and _finite(bbl[4]), "BB expected finite after warmup")

    # ADX on very short series may legitimately remain NaN (strict warmup policy).
    adx_short = store.indicator("adx", period=3)
    _expect(len(adx_short) == store.n, "ADX length mismatch on short series")

    # DMI/ADX finite test on a longer deterministic series
    n2 = 20
    close2 = [100.0 + float(i) for i in range(n2)]
    series2 = OhlcvSeries(
        ts_ms=list(range(1, n2 + 1)),
        open=close2[:],
        high=[c + 1.0 for c in close2],
        low=[c - 1.0 for c in close2],
        close=close2[:],
        volume=[10.0] * n2,
        symbol="TEST",
        timeframe="1m",
    )
    store2 = FeatureStore(series2)

    adx5 = store2.indicator("adx", period=5)
    pdi5 = store2.indicator("plus_di", period=5)
    mdi5 = store2.indicator("minus_di", period=5)

    _expect(len(adx5) == n2 and len(pdi5) == n2 and len(mdi5) == n2, "DMI bundle length mismatch")

    _expect(_finite(adx5[-1]) and 0.0 <= adx5[-1] <= 100.0, f"ADX expected finite 0..100, got {adx5[-1]}")
    _expect(_finite(pdi5[-1]) and 0.0 <= pdi5[-1] <= 100.0, f"+DI expected finite 0..100, got {pdi5[-1]}")
    _expect(_finite(mdi5[-1]) and 0.0 <= mdi5[-1] <= 100.0, f"-DI expected finite 0..100, got {mdi5[-1]}")

    # Unknown indicator should raise
    try:
        store.indicator("does_not_exist")
        raise AssertionError("Expected ValidationError for unknown indicator")
    except ValidationError:
        pass

    print("Features_Indicators self-test: OK")


if __name__ == "__main__":
    _self_test()
