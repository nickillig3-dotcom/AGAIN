from __future__ import annotations

"""
Mine_Run.py
===========

Runner:
- loads OHLCV (CSV > Binance(Data_Binance); no synthetic fallback)
- builds StrategySpace preset (small/medium/large)
- auto-configures WalkForward splits (or uses CLI overrides)
- runs the StrategyMiner
- prints summary and exports JSON

Fixes vs previous:
- Binance download_ohlcv max limit (e.g. 1500): auto chunk/paginate when --limit > max_limit
- "small" preset is now a realistic quick preset (not the unit-test tiny grid)
- Scoring min thresholds can be AUTO (default) based on fold count to avoid rejecting everything
"""

import argparse
import csv
import inspect
import math
import os
import json
import time
import hashlib
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, ValidationError, require
from Strategy_Space import StrategySpaceConfig, ComplexityBudget, StrategySpace, StrategySpec, strategy_spec_from_canonical
from Miner_Search import (
    WalkForwardConfig,
    SearchConfig,
    ScoringConfig,
    StrategyMiner,
    save_results_json,
    make_walkforward_splits,
)


# -----------------------------
# Time helpers
# -----------------------------
def _parse_timeframe_minutes(tf: str) -> int:
    s = str(tf).strip().lower()
    if not s:
        return 0
    if s.isdigit():
        return int(s)

    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        else:
            unit += ch
    if not num:
        return 0
    n = int(num)
    if unit in ("m", "min", "mins", "minute", "minutes", ""):
        return n
    if unit in ("h", "hr", "hrs", "hour", "hours"):
        return n * 60
    if unit in ("d", "day", "days"):
        return n * 1440
    if unit in ("w", "week", "weeks"):
        return n * 10080
    return 0


def _tf_ms(timeframe: str) -> int:
    m = _parse_timeframe_minutes(timeframe)
    return (m if m > 0 else 1) * 60_000


def _now_ms() -> int:
    return int(time.time() * 1000)
def _looks_like_epoch_ms(ts_ms: Sequence[int]) -> bool:
    """Heuristic: distinguish real epoch-ms candles from synthetic/index-like timestamps."""
    if not ts_ms:
        return False
    # 2000-01-01 in epoch ms
    return int(ts_ms[0]) >= 946_684_800_000 or int(ts_ms[-1]) >= 946_684_800_000
def _recommended_min_bars(timeframe: str) -> int:
    """
    Conservative minimum history length for "money-grade" mining.

    Why:
    - Mining large strategy spaces is extremely sensitive to non-stationarity and chance.
    - With too few bars, walk-forward + holdout gates become noisy and selection bias dominates.
    - Perp futures also have funding/volatility regimes that need time to show up.

    This function is only used for SpiritedReminder defaults; you can always override via --limit.
    """
    tf = str(timeframe).strip()
    # Explicit map for common crypto mining TFs (1m..1h). Values are bars.
    table = {
        "1m": 20_000,   # ~14 days
        "3m": 12_000,   # ~25 days
        "5m": 6_000,    # ~21 days
        "15m": 3_000,   # ~31 days
        "30m": 2_000,   # ~41 days
        "1h": 2_000,    # ~83 days
    }
    if tf in table:
        return int(table[tf])

    # Fallback: target ~21 days of bars
    mins = _parse_timeframe_minutes(tf)
    if mins <= 0:
        return 5_000
    bars_per_day = int(max(1, round(1440.0 / float(mins))))
    return int(max(500, bars_per_day * 21))

def _sanitize_series_for_backtest(series: OhlcvSeries, *, timeframe: str, source_label: str = "") -> OhlcvSeries:
    """
    Fail-fast data hygiene.

    Why:
    - Strategy mining is extremely sensitive to silent data issues.
    - Binance often returns the *currently forming* (not-yet-closed) candle as the last bar.
      If you keep it, your dataset becomes non-deterministic across runs.
    - Missing bars (gaps) break execution realism for OHLC backtests.

    Rules:
    - If timestamps look like epoch-ms, enforce uniform timeframe spacing.
    - Drop the last candle if it appears to still be open.
    """
    require(series is not None, "series must not be None")
    series.validate()

    ts = [int(x) for x in series.ts_ms]
    if not _looks_like_epoch_ms(ts):
        # Synthetic/offline series uses index-like timestamps; skip epoch-based checks.
        return series

    tf_ms = int(_tf_ms(timeframe))
    require(tf_ms > 0, "timeframe must resolve to >0 ms")

    # 1) Drop last candle if it appears incomplete.
    now_ms = _now_ms()
    if len(ts) >= 1:
        last_ts = int(ts[-1])
        if last_ts + tf_ms > now_ms:
            # last bar has not closed yet
            series = series.slice(stop=len(ts) - 1)
            ts = [int(x) for x in series.ts_ms]
            print(
                "INFO: Dropped last candle (appears incomplete). "
                f"last_ts={last_ts} tf_ms={tf_ms} now_ms={now_ms} source={source_label}"
            )

    # 2) Strict spacing: every bar must be exactly tf_ms apart.
    bad = []
    for i in range(1, len(ts)):
        dt = int(ts[i]) - int(ts[i - 1])
        if dt != tf_ms:
            bad.append((i - 1, ts[i - 1], ts[i], dt))
            if len(bad) >= 5:
                break

    if bad:
        details = "; ".join(f"i={j}->{j+1} dt={dt} (t0={t0}, t1={t1})" for j, t0, t1, dt in bad)
        raise ValidationError(
            "Non-uniform candle spacing detected. "
            f"Expected dt={tf_ms}ms for timeframe={timeframe!r}. "
            f"First mismatches: {details}. "
            "This usually means missing bars or mixed timeframes in the dataset."
        )

    # 3) Light alignment warning (do not fail): Binance candles are typically aligned to tf.
    #    If this triggers, it can indicate resampling or timezone offsets in CSV data.
    mis = 0
    for t in ts[: min(50, len(ts))]:
        if (int(t) % tf_ms) != 0:
            mis += 1
            if mis >= 3:
                break
    if mis:
        print(
            "WARNING: ts_ms values are not aligned to timeframe boundaries (ts % tf_ms != 0). "
            "If this is not intentional, you may be backtesting resampled / misaligned candles."
        )

    return series

# -----------------------------
# Synthetic OHLCV (offline)
# -----------------------------
def _enforce_ohlc_invariants(o: float, h: float, l: float, c: float) -> Tuple[float, float, float, float]:
    """
    Enforce strict OHLC invariants required by Core_Types.OhlcvSeries:
      high >= max(open, close)
      low  <= min(open, close)
      low <= high
    """
    eps = 1e-9
    oo = float(o)
    cc = float(c)
    hh = max(float(h), oo, cc) + eps
    ll = min(float(l), oo, cc) - eps
    if ll > hh:
        hh = ll + eps
    return oo, hh, ll, cc


def _make_synthetic_series(n: int, symbol: str, timeframe: str) -> OhlcvSeries:
    """
    Deterministic synthetic OHLCV series designed to trigger a variety of signals.
    Drift is scaled by n to avoid insane trend magnitude for large n (e.g. 5000 bars).
    """
    require(n >= 200, "synthetic n must be >= 200")

    ts = list(range(1, n + 1))
    open_: List[float] = []
    high: List[float] = []
    low: List[float] = []
    close: List[float] = []
    vol: List[float] = []

    # total drift magnitude across half the series (kept moderate)
    drift_total = 10.0
    rate_up = drift_total / float(max(1, n // 2))
    rate_dn = drift_total / float(max(1, n // 2))

    for i in range(n):
        base = 100.0 + 2.2 * math.sin(2.0 * math.pi * i / 50.0) + 0.7 * math.sin(2.0 * math.pi * i / 13.0)

        if i < n // 2:
            drift = rate_up * i
        else:
            drift = drift_total - rate_dn * (i - n // 2)

        c = base + drift
        close.append(float(c))

        o = close[i - 1] if i > 0 else c
        open_.append(float(o))

        wick = 0.7 + 0.35 * math.sin(2.0 * math.pi * i / 17.0)
        extra = 0.8 if (i % 47 == 0) else 0.0  # occasional sweep-like wick

        h0 = c + wick + 0.12 * math.sin(2.0 * math.pi * i / 9.0)
        l0 = c - wick - extra - 0.12 * math.sin(2.0 * math.pi * i / 11.0)

        oo, hh, ll, cc = _enforce_ohlc_invariants(o, h0, l0, c)
        high.append(hh)
        low.append(ll)

        v = 220.0 + 70.0 * math.sin(2.0 * math.pi * i / 30.0) + (50.0 if (i % 60 < 5) else 0.0)
        vol.append(max(1.0, float(v)))

    return OhlcvSeries(
        ts_ms=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=vol,
        symbol=str(symbol),
        timeframe=str(timeframe),
    )


# -----------------------------
# CSV Loader
# -----------------------------
def _load_series_csv(path: str, *, symbol: str, timeframe: str) -> OhlcvSeries:
    p = str(path).strip()
    require(p, "csv path must be non-empty")
    require(os.path.exists(p), f"csv file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        rows = list(csv.reader(f))

    require(len(rows) > 0, "CSV is empty")

    if has_header:
        header = [h.strip().lower() for h in rows[0]]
        data_rows = rows[1:]

        def idx(*names: str) -> int:
            for nm in names:
                if nm.lower() in header:
                    return header.index(nm.lower())
            return -1

        i_t = idx("ts_ms", "timestamp", "time", "t")
        i_o = idx("open", "o")
        i_h = idx("high", "h")
        i_l = idx("low", "l")
        i_c = idx("close", "c")
        i_v = idx("volume", "v")
        require(all(i >= 0 for i in (i_t, i_o, i_h, i_l, i_c, i_v)), f"CSV header missing required columns: {header}")

        ts_ms: List[int] = []
        op: List[float] = []
        hi: List[float] = []
        lo: List[float] = []
        cl: List[float] = []
        vo: List[float] = []
        for r in data_rows:
            if len(r) < max(i_t, i_o, i_h, i_l, i_c, i_v) + 1:
                continue
            ts_ms.append(int(float(r[i_t])))
            op.append(float(r[i_o])); hi.append(float(r[i_h])); lo.append(float(r[i_l])); cl.append(float(r[i_c])); vo.append(float(r[i_v]))
        require(len(ts_ms) > 0, "CSV had no parsable data rows")
        return OhlcvSeries(ts_ms=ts_ms, open=op, high=hi, low=lo, close=cl, volume=vo, symbol=symbol, timeframe=timeframe)

    # no header
    ts_ms: List[int] = []
    op: List[float] = []
    hi: List[float] = []
    lo: List[float] = []
    cl: List[float] = []
    vo: List[float] = []
    for r in rows:
        if len(r) < 6:
            continue
        ts_ms.append(int(float(r[0])))
        op.append(float(r[1])); hi.append(float(r[2])); lo.append(float(r[3])); cl.append(float(r[4])); vo.append(float(r[5]))
    require(len(ts_ms) > 0, "CSV had no parsable rows")
    return OhlcvSeries(ts_ms=ts_ms, open=op, high=hi, low=lo, close=cl, volume=vo, symbol=symbol, timeframe=timeframe)


# -----------------------------
# Binance Loader (best-effort + chunking)
# -----------------------------
def _call_with_signature(fn: Any, kwargs_pool: Dict[str, Any]) -> Any:
    """
    Call fn by matching its signature params against kwargs_pool.
    Skips None values to avoid "start_ms must be int" errors.
    """
    sig = inspect.signature(fn)
    kwargs = {}
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in kwargs_pool:
            val = kwargs_pool[p.name]
            if val is None:
                continue
            kwargs[p.name] = val
    return fn(**kwargs)


def _coerce_to_ohlcv_series(obj: Any, *, symbol: str, timeframe: str) -> OhlcvSeries:
    """
    Convert common candle structures to OhlcvSeries.
    """
    if isinstance(obj, OhlcvSeries):
        return obj

    if isinstance(obj, dict):
        keys = {str(k).lower(): k for k in obj.keys()}
        req = ("ts_ms", "open", "high", "low", "close", "volume")
        if all(k in keys for k in req):
            return OhlcvSeries(
                ts_ms=[int(x) for x in obj[keys["ts_ms"]]],
                open=[float(x) for x in obj[keys["open"]]],
                high=[float(x) for x in obj[keys["high"]]],
                low=[float(x) for x in obj[keys["low"]]],
                close=[float(x) for x in obj[keys["close"]]],
                volume=[float(x) for x in obj[keys["volume"]]],
                symbol=str(obj.get("symbol", symbol)),
                timeframe=str(obj.get("timeframe", timeframe)),
            )
        for ck in ("candles", "ohlcv", "klines", "data"):
            if ck in keys:
                return _coerce_to_ohlcv_series(obj[keys[ck]], symbol=symbol, timeframe=timeframe)

    if isinstance(obj, (list, tuple)):
        xs = list(obj)
        require(len(xs) > 0, "Empty OHLCV data returned")

        first = xs[0]
        # list-of-lists klines
        if isinstance(first, (list, tuple)) and len(first) >= 6:
            ts_ms: List[int] = []
            op: List[float] = []
            hi: List[float] = []
            lo: List[float] = []
            cl: List[float] = []
            vo: List[float] = []
            for row in xs:
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    continue
                ts_ms.append(int(float(row[0])))
                op.append(float(row[1])); hi.append(float(row[2])); lo.append(float(row[3])); cl.append(float(row[4])); vo.append(float(row[5]))
            require(len(ts_ms) > 0, "Could not parse list-candle OHLCV data")
            return OhlcvSeries(ts_ms=ts_ms, open=op, high=hi, low=lo, close=cl, volume=vo, symbol=symbol, timeframe=timeframe)

    raise ValidationError(f"Unsupported OHLCV data type returned: {type(obj).__name__}")


def _merge_series(parts: List[OhlcvSeries], *, symbol: str, timeframe: str, take_last: int) -> OhlcvSeries:
    """
    Merge multiple OhlcvSeries chunks, sort by ts_ms, deduplicate, and take last N bars.
    """
    ts: List[int] = []
    op: List[float] = []
    hi: List[float] = []
    lo: List[float] = []
    cl: List[float] = []
    vo: List[float] = []

    for s in parts:
        ts.extend([int(x) for x in s.ts_ms])
        op.extend([float(x) for x in s.open])
        hi.extend([float(x) for x in s.high])
        lo.extend([float(x) for x in s.low])
        cl.extend([float(x) for x in s.close])
        vo.extend([float(x) for x in s.volume])

    require(len(ts) > 0, "no data to merge")

    rows = list(zip(ts, op, hi, lo, cl, vo))
    rows.sort(key=lambda r: r[0])

    # dedup by timestamp (keep last occurrence)
    dedup: List[Tuple[int, float, float, float, float, float]] = []
    last_t = None
    for r in rows:
        if last_t is None or r[0] != last_t:
            dedup.append(r)
            last_t = r[0]
        else:
            dedup[-1] = r  # overwrite last

    if take_last > 0 and len(dedup) > take_last:
        dedup = dedup[-take_last:]

    ts2 = [r[0] for r in dedup]
    op2 = [r[1] for r in dedup]
    hi2 = [r[2] for r in dedup]
    lo2 = [r[3] for r in dedup]
    cl2 = [r[4] for r in dedup]
    vo2 = [r[5] for r in dedup]

    return OhlcvSeries(ts_ms=ts2, open=op2, high=hi2, low=lo2, close=cl2, volume=vo2, symbol=symbol, timeframe=timeframe)


def _load_series_binance_best_effort(symbol: str, timeframe: str, limit: int) -> OhlcvSeries:
    """
    Preferred path:
      Data_Binance.download_ohlcv(...) -> klines
      Data_Binance.parse_klines_to_series(klines, ...) -> OhlcvSeries

    Handles max per-request limit (commonly 1500) by chunking/pagination.
    """
    try:
        import Data_Binance as DB  # type: ignore
    except Exception as e:
        raise ValidationError(f"Could not import Data_Binance: {e}") from e

    if not (hasattr(DB, "download_ohlcv") and callable(getattr(DB, "download_ohlcv"))):
        raise ValidationError("Data_Binance.download_ohlcv not found/callable")

    dl = getattr(DB, "download_ohlcv")

    parse_fn = getattr(DB, "parse_klines_to_series", None)
    can_parse = callable(parse_fn)

    # common hard cap for Binance klines limits in wrappers
    MAX_LIMIT = 1500
    total = int(limit)
    require(total > 0, "limit must be > 0")

    end_ms = _now_ms()
    step_ms = _tf_ms(timeframe)

    parts: List[OhlcvSeries] = []
    remaining = total

    # fetch newest -> older chunks
    while remaining > 0:
        req_lim = min(MAX_LIMIT, remaining)
        start_ms = end_ms - req_lim * step_ms
        if start_ms < 0:
            start_ms = 0

        kwargs_pool = {
            "symbol": symbol,
            "pair": symbol,
            "market": symbol,
            "timeframe": timeframe,
            "tf": timeframe,
            "interval": timeframe,
            "limit": int(req_lim),
            "n": int(req_lim),
            "count": int(req_lim),
            "max_bars": int(req_lim),
            "start_ms": int(start_ms),
            "end_ms": int(end_ms),
            "since": int(start_ms),
            "until": int(end_ms),
        }

        res = _call_with_signature(dl, kwargs_pool)

        # If res already is OhlcvSeries/dict/list -> coerce
        chunk_series: Optional[OhlcvSeries] = None

        # If klines require parsing, do it
        if can_parse:
            try:
                kwargs_pool2 = dict(kwargs_pool)
                kwargs_pool2["klines"] = res
                kwargs_pool2["data"] = res
                kwargs_pool2["candles"] = res
                parsed = _call_with_signature(parse_fn, kwargs_pool2)  # type: ignore[arg-type]
                chunk_series = _coerce_to_ohlcv_series(parsed, symbol=symbol, timeframe=timeframe)
            except Exception:
                chunk_series = None

        if chunk_series is None:
            chunk_series = _coerce_to_ohlcv_series(res, symbol=symbol, timeframe=timeframe)

        got = len(chunk_series.ts_ms)
        if got <= 0:
            break

        parts.append(chunk_series)
        remaining -= got

        # move end_ms backward to avoid overlap (use earliest ts in this chunk)
        earliest = int(min(int(x) for x in chunk_series.ts_ms))
        end_ms = earliest - step_ms
        if end_ms <= 0:
            break

        # safety: if API returns fewer than requested, still proceed, but stop if no progress
        if got < req_lim and remaining > 0:
            # still attempt next chunk; but if next would be identical time, break
            pass

    if not parts:
        raise ValidationError("download_ohlcv returned no data")

    merged = _merge_series(parts, symbol=symbol, timeframe=timeframe, take_last=total)
    return merged


# -----------------------------
# Space presets (small/medium/large)
# -----------------------------
def _space_preset(name: str) -> Tuple[StrategySpaceConfig, ComplexityBudget]:
    nm = str(name).strip().lower()
    require(nm in ("small", "medium", "large"), f"space preset must be small|medium|large, got {name!r}")

    # NOTE: "small" is now a realistic quick preset (NOT the unit-test tiny grid).
    if nm == "small":
        cfg = StrategySpaceConfig(
            ms_left_right=((1, 1), (2, 2)),
            ms_tol_bps=(0.0, 10.0),
            within_bars=(0, 3, 5, 10),
            rsi_periods=(7, 14),
            rsi_oversold=(30.0,),
            rsi_overbought=(70.0,),
            rsi_mid=(50.0,),
            ema_fast_periods=(10, 20),
            ema_slow_periods=(50,),
            adx_periods=(14,),
            adx_thresholds=(20.0,),
            bb_periods=(20,),
            bb_mults=(2.0,),
            donchian_periods=(20,),
            volume_z_periods=(200,),
            volume_z_thresholds=(0.5,),
            stop_percent=(0.01, 0.015),
            stop_atr_periods=(14,),
            stop_atr_mults=(2.0,),
            stop_trailing=(False,),
            stop_structure_buffer_bps=(10.0,),
            tp_rr=(1.5, 2.0),
            time_stops=(0, 60, 120),
            cooldowns=(0, 5),
            use_regime_filters=True,
            max_generated=0,
        )
        budget = ComplexityBudget(
            max_entry_steps=3,
            max_total_atomic_conditions=9,
            max_within_bars=30,
            max_complexity=24,
        )
        return cfg, budget

    if nm == "medium":
        cfg = StrategySpaceConfig(
            ms_left_right=((1, 1), (2, 2)),
            ms_tol_bps=(0.0, 5.0, 10.0),
            within_bars=(0, 3, 5, 10, 20),
            rsi_periods=(7, 14, 21),
            rsi_oversold=(20.0, 30.0),
            rsi_overbought=(70.0, 80.0),
            rsi_mid=(50.0,),
            ema_fast_periods=(10, 20, 30),
            ema_slow_periods=(50, 100),
            adx_periods=(14,),
            adx_thresholds=(20.0, 25.0),
            bb_periods=(20,),
            bb_mults=(2.0,),
            donchian_periods=(20, 50),
            volume_z_periods=(200,),
            volume_z_thresholds=(0.5, 1.0),
            stop_percent=(0.005, 0.01, 0.015),
            stop_atr_periods=(14,),
            stop_atr_mults=(1.5, 2.0, 2.5),
            stop_trailing=(False,),
            stop_structure_buffer_bps=(5.0, 10.0, 20.0),
            tp_rr=(1.0, 1.5, 2.0, 3.0),
            time_stops=(0, 60, 120, 240),
            cooldowns=(0, 5, 10),
            use_regime_filters=True,
            max_generated=0,
        )
        budget = ComplexityBudget(
            max_entry_steps=3,
            max_total_atomic_conditions=9,
            max_within_bars=50,
            max_complexity=24,
        )
        return cfg, budget

    # large
    cfg = StrategySpaceConfig(
        ms_left_right=((1, 1), (2, 2), (3, 3)),
        ms_tol_bps=(0.0, 5.0, 10.0, 20.0, 30.0),
        within_bars=(0, 3, 5, 10, 20, 50),
        rsi_periods=(5, 7, 14, 21, 28),
        rsi_oversold=(20.0, 25.0, 30.0, 35.0),
        rsi_overbought=(65.0, 70.0, 75.0, 80.0),
        rsi_mid=(50.0,),
        ema_fast_periods=(5, 10, 20, 30, 40),
        ema_slow_periods=(50, 100, 200),
        adx_periods=(10, 14, 20),
        adx_thresholds=(15.0, 20.0, 25.0, 30.0),
        bb_periods=(20, 30),
        bb_mults=(2.0, 2.5),
        donchian_periods=(20, 50, 100),
        volume_z_periods=(100, 200, 400),
        volume_z_thresholds=(0.5, 1.0, 1.5),
        stop_percent=(0.003, 0.005, 0.008, 0.01, 0.015, 0.02),
        stop_atr_periods=(10, 14),
        stop_atr_mults=(1.0, 1.5, 2.0, 2.5, 3.0),
        stop_trailing=(False, True),
        stop_structure_buffer_bps=(0.0, 5.0, 10.0, 20.0, 30.0),
        tp_rr=(0.8, 1.0, 1.5, 2.0, 3.0, 4.0),
        time_stops=(0, 30, 60, 120, 240, 480),
        cooldowns=(0, 3, 5, 10, 20),
        use_regime_filters=True,
        max_generated=0,
    )
    budget = ComplexityBudget(
        max_entry_steps=3,
        max_total_atomic_conditions=10,
        max_within_bars=80,
        max_complexity=28,
    )
    return cfg, budget


# -----------------------------
# Walk-forward auto
# -----------------------------
def _auto_walkforward(nbars: int) -> WalkForwardConfig:
    n = int(nbars)
    require(n > 0, "nbars must be > 0")

    if n < 800:
        test = max(80, n // 4)
        train = max(test * 2, n // 2)
    else:
        test = max(150, n // 6)
        train = max(test * 3, n // 2)

    step = test
    cfg = WalkForwardConfig(train_bars=train, test_bars=test, step_bars=step, purge_bars=0, embargo_bars=0, anchored=False)
    _ = make_walkforward_splits(n, cfg)
    return cfg
def _auto_walkforward_money(nbars: int) -> WalkForwardConfig:
    """
    Money-mode walk-forward defaults (more OOS signal).

    Key idea:
    The miner does *not* fit a model on the train slice. The "train" part mainly acts as
    indicator warmup / regime stabilization. So we can shrink train windows to get *more*
    non-overlapping OOS test folds, which materially reduces selection bias.

    Compared to _auto_walkforward():
    - smaller train windows (still enough warmup)
    - more test folds by construction
    """
    n = int(nbars)
    require(n > 0, "nbars must be > 0")

    # Keep test windows meaningful but not huge.
    min_test = 150
    max_test = 2000
    test = max(min_test, min(max_test, max(1, n // 10)))

    # Train is mostly warmup; keep a floor and cap to preserve fold count.
    train_min = 400
    train_cap = 2000
    train = max(train_min, min(train_cap, max(1, n // 3)))

    if train + test >= n:
        # Not enough bars for this heuristic; fall back to the older rule.
        return _auto_walkforward(n)

    step = test
    cfg = WalkForwardConfig(train_bars=train, test_bars=test, step_bars=step, purge_bars=0, embargo_bars=0, anchored=False)
    _ = make_walkforward_splits(n, cfg)
    return cfg
def _auto_holdout_bars(
    nbars: int,
    *,
    train_bars: int,
    test_bars: int,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    min_frac: float = 0.10,
    max_frac: float = 0.20,
) -> int:
    """
    Conservative default holdout sizing.

    Why:
    - Mining across large strategy spaces has extreme multiple-testing risk.
      A final untouched "most recent" slice is the cheapest, highest-signal sanity gate.
    - We want the holdout to be *meaningful* (at least a full test window when possible),
      but we must not destroy the ability to form at least one WF fold.

    Rules:
    - desired >= test_bars (comparable length to a test fold)
    - desired >= min_frac * total bars
    - cap at max_frac * total bars
    - additionally cap so at least one fold is possible with start=0

    Returns:
      holdout_bars (>=0). Returns 0 if no feasible holdout exists.
    """
    n = int(nbars)
    require(n > 0, "nbars must be > 0")

    tr = max(1, int(train_bars))
    te = max(1, int(test_bars))
    pg = max(0, int(purge_bars))
    em = max(0, int(embargo_bars))

    require(0.0 < float(min_frac) <= 1.0, "min_frac must be in (0,1]")
    require(0.0 < float(max_frac) <= 1.0, "max_frac must be in (0,1]")
    require(float(min_frac) <= float(max_frac), "min_frac must be <= max_frac")

    # Need at least one fold with start=0:
    # test_end = train_bars + embargo + purge + test_bars
    max_holdout = n - (tr + em + pg + te)
    if max_holdout <= 0:
        return 0

    desired = max(int(te), int(n * float(min_frac)))
    cap = int(n * float(max_frac))
    hb = min(int(max_holdout), int(desired), int(cap))
    return max(0, int(hb))

# -----------------------------
# Printing
# -----------------------------
def _short_regime_filter_str(rf: Any) -> str:
    if rf is None:
        return ""
    try:
        if isinstance(rf, dict) and "all_of" in rf:
            conds = rf["all_of"]
            if not conds:
                return ""
            parts = []
            for c in conds[:3]:
                rn = c.get("regime_name", "?")
                op = c.get("op", "?")
                val = c.get("value", "?")
                parts.append(f"{rn}{op}{val}")
            return " & ".join(parts) + (" ..." if len(conds) > 3 else "")
    except Exception:
        pass
    return ""


def _print_top(report: Any, top_n: int = 10) -> None:
    top = list(report.top_results)[: max(0, int(top_n))]
    if not top:
        dbg = list(getattr(report, "debug_top", ()) or [])[: max(0, int(top_n))]
        if not dbg:
            print("No top results.")
            return
        print("\nTop candidates (DEBUG / rejected):")
        for i, r in enumerate(dbg, 1):
            agg = r.aggregate or {}
            mr = float(agg.get("mean_return_used", agg.get("mean_return", 0.0)) or 0.0)
            mdd = float(agg.get("mean_drawdown", 0.0) or 0.0)
            pf = float(agg.get("mean_profit_factor", 0.0) or 0.0)
            trades = agg.get("fold_trades", [])
            entries = agg.get("fold_entries", [])
            trades_str = ",".join(str(x) for x in trades) if isinstance(trades, list) else str(trades)
            entries_str = ",".join(str(x) for x in entries) if isinstance(entries, list) else str(entries)
            tr_tot = int(agg.get("trades_total", 0) or 0)
            en_tot = int(agg.get("entries_total", 0) or 0)
            damp = float(agg.get("pf_trade_damp", 0.0) or 0.0)
            hold = float(agg.get("mean_hold_bars", 0.0) or 0.0)

            stress = agg.get("stress") if isinstance(agg, dict) else None
            stress_str = ""
            try:
                if isinstance(stress, dict) and float(stress.get("cost_mult", 1.0) or 1.0) > 1.0:
                    sm = float(stress.get("mean_return", 0.0) or 0.0)
                    sw = float(stress.get("worst_fold_return", 0.0) or 0.0)
                    mult = float(stress.get("cost_mult"))
                    stress_str = f" stress(x{mult:g}):ret={sm: .4f} worst={sw: .4f}"
            except Exception:
                stress_str = ""
            # Holdout (if available)
            ho = getattr(r, "holdout_aggregate", None)
            ho_str = ""
            try:
                if isinstance(ho, dict) and ho:
                    ho_ret = float(ho.get("net_return", 0.0) or 0.0)
                    ho_dd = float(ho.get("max_drawdown", 0.0) or 0.0)
                    ho_tr = int(ho.get("trades", 0) or 0)
                    ho_str = f" holdout:ret={ho_ret: .4f} dd={ho_dd: .4f} tr={ho_tr}"
                # Optional: holdout stress (if present)
                hs2 = ho.get("stress", None)
                if isinstance(hs2, dict) and float(hs2.get("cost_mult", 1.0) or 1.0) > 1.0:
                    mult = float(hs2.get("cost_mult"))
                    hs_ret = float(hs2.get("net_return", 0.0) or 0.0)
                    hs_dd = float(hs2.get("max_drawdown", 0.0) or 0.0)
                    hs_tr = int(hs2.get("trades", 0) or 0)
                    ho_str += f" ho_stress(x{mult:g}):ret={hs_ret: .4f} dd={hs_dd: .4f} tr={hs_tr}"
                    rr = hs2.get("return_ratio", None)
                    if rr is not None:
                        try:
                            ho_str += f" rr={float(rr):.2f}"
                        except Exception:
                            pass
                    rr = hs2.get("return_ratio", None)
                    if rr is not None:
                        try:
                            ho_str += f" rr={float(rr):.2f}"
                        except Exception:
                            pass
                # Optional: holdout permutation test (Loop 3)
                pt = ho.get("perm_test", None)
                if isinstance(pt, dict) and pt:
                    pval = pt.get("p_value", None)
                    npt = pt.get("valid", pt.get("trials_used", 0))
                    if pval is not None and int(npt or 0) > 0:
                        ho_str += f" ho_perm:p={float(pval):.3f} n={int(npt)}"     
            except Exception:
                ho_str = ""
            print(
                f"{i:>2}. score={float(r.score):> .6f} id={r.strategy_id} dir={r.direction:<5} "
                f"ret={mr:> .4f} dd={mdd:> .4f} pf={pf:> .2f} trades=[{trades_str}] "
                f"entries=[{entries_str}] en_tot={en_tot} tr_tot={tr_tot} hold={hold:.1f} pf_damp={damp:.2f} "
                f"REJECT={getattr(r, 'reject_reason', '')}"
                + ho_str
                + stress_str
            )
        return

    print("\nTop strategies:")
    for i, r in enumerate(top, 1):
        agg = r.aggregate or {}
        mr = float(agg.get("mean_return_used", agg.get("mean_return", 0.0)) or 0.0)
        mdd = float(agg.get("mean_drawdown", 0.0) or 0.0)
        pf = float(agg.get("mean_profit_factor", 0.0) or 0.0)

        # Optional: cost stress-test (Loop 5)
        stress = agg.get("stress") if isinstance(agg, dict) else None
        stress_str = ""
        try:
            if isinstance(stress, dict) and float(stress.get("cost_mult", 1.0) or 1.0) > 1.0:
                sm = float(stress.get("mean_return", 0.0) or 0.0)
                sw = float(stress.get("worst_fold_return", 0.0) or 0.0)
                sdd = float(stress.get("mean_drawdown", 0.0) or 0.0)
                mult = float(stress.get("cost_mult"))
                stress_str = f" stress(x{mult:g}):ret={sm: .4f} worst={sw: .4f} dd={sdd: .4f}"
                rr = stress.get("mean_return_ratio", None)
                if rr is not None:
                    try:
                        stress_str += f" rr={float(rr):.2f}"
                    except Exception:
                        pass
        except Exception:
            stress_str = ""

        # Optional: final holdout (Loop 4)
        ho = getattr(r, "holdout_aggregate", None)
        ho_str = ""
        try:
            if isinstance(ho, dict) and ho:
                ho_ret = float(ho.get("net_return", 0.0) or 0.0)
                ho_dd = float(ho.get("max_drawdown", 0.0) or 0.0)
                ho_tr = int(ho.get("trades", 0) or 0)
                ho_str = f" holdout:ret={ho_ret: .4f} dd={ho_dd: .4f} tr={ho_tr}"
                hs2 = ho.get("stress", None)
                if isinstance(hs2, dict) and float(hs2.get("cost_mult", 1.0) or 1.0) > 1.0:
                    mult = float(hs2.get("cost_mult", 1.0) or 1.0)
                    hs_ret = float(hs2.get("net_return", 0.0) or 0.0)
                    hs_dd = float(hs2.get("max_drawdown", 0.0) or 0.0)
                    hs_tr = int(hs2.get("trades", 0) or 0)
                    ho_str += f" ho_stress(x{mult:g}):ret={hs_ret: .4f} dd={hs_dd: .4f} tr={hs_tr}"
                    rr = hs2.get("return_ratio", None)
                    if rr is not None:
                        try:
                            ho_str += f" rr={float(rr):.2f}"
                        except Exception:
                            pass
                # Optional: holdout permutation test + param robustness summaries (Loops 3,5)
                pt = ho.get("perm_test", None)
                if isinstance(pt, dict) and pt:
                    pval = pt.get("p_value", None)
                    npt = pt.get("valid", pt.get("trials_used", 0))
                    if pval is not None and int(npt or 0) > 0:
                        ho_str += f" ho_perm:p={float(pval):.3f} n={int(npt)}"
                pr = ho.get("param_robust", None)
                if isinstance(pr, dict) and pr:
                    pr_ratio = pr.get("pass_ratio", None)
                    passed = int(pr.get("passed", 0) or 0)
                    trials = int(pr.get("trials", 0) or 0)
                    if pr_ratio is not None and trials > 0:
                        ho_str += f" ho_pr:{float(pr_ratio):.2f} ({passed}/{trials})"                      
        except Exception:
            ho_str = ""
        trades = agg.get("fold_trades", [])
        trades_str = ",".join(str(x) for x in trades) if isinstance(trades, list) else str(trades)
        rf = _short_regime_filter_str(r.regime_filter)
        # Optional: CSCV per-strategy selection diagnostics (Loop 14)
        cscv_str = ""
        try:
            cscv = agg.get("cscv", None)
            if isinstance(cscv, dict) and cscv:
                sf = cscv.get("selected_fraction", None)
                rk = cscv.get("oos_rank_median", None)
                if sf is not None and rk is not None:
                    cscv_str = f" cscv:sel={float(sf):.2f} oos={float(rk):.2f}"
                elif sf is not None:
                    cscv_str = f" cscv:sel={float(sf):.2f}"
                elif rk is not None:
                    cscv_str = f" cscv:oos={float(rk):.2f}"
        except Exception:
            cscv_str = ""

        print(
            f"{i:>2}. score={r.score:> .6f} id={r.strategy_id} dir={r.direction:<5} "
            f"ret={mr:> .4f} dd={mdd:> .4f} pf={pf:> .2f} trades=[{trades_str}] "
            f"cx={r.complexity:<2} tags={list(r.tags)}"
            + (f" rf={rf}" if rf else "")
            + ho_str
            + stress_str
            + cscv_str
        )
        # Optional: print human-readable strategy DSL (Loop 16).
        dsl = str(getattr(r, 'strategy_dsl', '') or '')
        if dsl:
            dsl_short = dsl.replace('\n', ' ').strip()
            if len(dsl_short) > 220:
                dsl_short = dsl_short[:217] + '...'
            print(f"    dsl: {dsl_short}")
# Replay helpers (load StrategySpec from saved JSON)
# -----------------------------

class _ReplaySpace:
    """Minimal StrategySpace-like wrapper to replay a fixed list of StrategySpec."""

    def __init__(self, specs: Sequence[StrategySpec]) -> None:
        self._specs: List[StrategySpec] = list(specs)

    def iter_strategies(self) -> Sequence[StrategySpec]:
        # StrategyMiner only requires an iterable of StrategySpec.
        return self._specs


def _load_replay_specs_from_results_json(path: str, *, top_n: int = 0, ids: Optional[Sequence[str]] = None) -> List[StrategySpec]:
    """Load StrategySpec(s) from a saved MinerResults_*.json file.

    Selection:
    - If ids is provided: filter by prefix match on strategy_id or strategy_hash.
    - Else: take the first top_n strategies (or all if top_n<=0).

    Uses spec_canonical to reconstruct StrategySpec without ambiguity.
    """

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    top = payload.get("top_results")
    if not isinstance(top, list):
        raise ValidationError("Invalid results JSON: missing 'top_results' list")

    want: List[str] = []
    if ids:
        want = [str(x).strip() for x in ids if str(x).strip()]

    out: List[StrategySpec] = []
    for r in top:
        if not isinstance(r, dict):
            continue

        sid = str(r.get("strategy_id", "") or "")
        sh = str(r.get("strategy_hash", "") or "")

        if want:
            ok = False
            for w in want:
                if sid.startswith(w) or sh.startswith(w):
                    ok = True
                    break
            if not ok:
                continue

        canon = r.get("spec_canonical")
        if canon is None:
            raise ValidationError("Results JSON is missing 'spec_canonical' (cannot replay reliably)")

        spec = strategy_spec_from_canonical(canon)
        out.append(spec)

        if int(top_n) > 0 and len(out) >= int(top_n):
            break

    if not out:
        raise ValidationError("No strategies selected for replay (check --replay_ids/--replay_top)")

    return out


# -----------------------------
# CLI
# -----------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strategy miner on Binance or CSV (engine-only; no synthetic fallback).")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1m")
    p.add_argument("--limit", type=int, default=1500, help="Requested bars (will be chunked if > max per request)")

    p.add_argument("--csv", type=str, default="")
    # Replay: evaluate previously mined strategies (loaded from MinerResults_*.json)
    p.add_argument("--replay_json", type=str, default="", help="Path to MinerResults_*.json. If set, skip StrategySpace generation and replay those strategies.")
    p.add_argument("--replay_top", type=int, default=0, help="How many top strategies to replay from the JSON (0=all).")
    p.add_argument("--replay_ids", type=str, default="", help="Comma-separated strategy_id/strategy_hash prefixes to replay (optional).")

    # NOTE: "tiny" is kept as a backwards-compatible alias for "small".
    p.add_argument("--space", type=str, default="small", choices=["tiny", "small", "medium", "large"])
    p.add_argument("--mode", type=str, default="iterate", choices=["iterate", "sample"])
    p.add_argument("--max_evals", type=int, default=300)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sample_prob", type=float, default=0.15)

    p.add_argument("--backtest", type=str, default="engine", choices=["engine"], help="Backtest backend (engine only; no fallback).")
    # Engine backtester realism knobs (perpetual futures).
    # Only applies when backtest backend is "engine" (or "auto" resolves to engine).
    p.add_argument("--engine_leverage", type=float, default=0.0, help="(engine) Override leverage (0 = default).")
    p.add_argument(
        "--engine_risk_frac",
        type=float,
        default=0.0,
        help="(engine) Override risk_per_trade_fraction (e.g., 0.005 = 0.5 percent of equity). 0 = default.",
    )
    p.add_argument(
        "--engine_max_margin_frac",
        type=float,
        default=0.0,
        help="(engine) Override max_margin_fraction (fraction of equity). 0 = default.",
    )
    p.add_argument(
        "--engine_maintenance_margin_rate",
        type=float,
        default=0.0,
        help="(engine) Override maintenance_margin_rate (0 = default).",
    )
    p.add_argument(
        "--engine_liquidation_fee_rate",
        type=float,
        default=-1.0,
        help="(engine) Override liquidation_fee_rate (-1 = default).",
    )
    p.add_argument(
        "--engine_min_notional",
        type=float,
        default=-1.0,
        help="(engine) Minimum entry order notional in USDT (-1 = default, 0 disables).",
    )
    p.add_argument(
        "--engine_min_qty",
        type=float,
        default=-1.0,
        help="(engine) Minimum entry quantity in base units (-1 = default, 0 disables).",
    )
    p.add_argument(
        "--engine_qty_step",
        type=float,
        default=-1.0,
        help="(engine) Quantity step size (lot size). Floors qty to step at entry (-1 = default, 0 disables).",
    )
    # Miner diversification / variant control (important for undertrading)
    p.add_argument("--entry_key_mode", type=str, default="entry_exit_time_regime",
                    choices=["entry", "entry_exit", "entry_exit_time", "entry_exit_time_regime"])    
    p.add_argument("--max_variants_per_entry", type=int, default=1)
    p.add_argument("--no_diversify_by_entry", action="store_true")
    p.add_argument("--no_diversify_by_family", action="store_true")
    p.add_argument("--max_per_family", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    # Cross-market / cross-timeframe validation (Loop 13)
    p.add_argument(
        "--validate",
        action="append",
        default=[],
        help="Extra validation datasets as SYMBOL:TIMEFRAME (repeatable). Example: --validate ETHUSDT:1m",
    )
    p.add_argument(
        "--validate_top",
        type=int,
        default=10,
        help="Validate the top N mined strategies on each validation dataset.",
    )
    p.add_argument(
        "--validate_min_pass_ratio",
        type=float,
        default=1.0,
        help="Min fraction of validation datasets a strategy must pass (1.0 = all).",
    )
    p.add_argument(
        "--validate_out",
        type=str,
        default="",
        help="Optional output path for validated results (defaults to <out>_validated.json).",
    )
    # Walk-forward overrides
    p.add_argument("--train_bars", type=int, default=0)
    p.add_argument("--test_bars", type=int, default=0)
    p.add_argument("--step_bars", type=int, default=0)
    p.add_argument("--purge_bars", type=int, default=0)
    p.add_argument("--embargo_bars", type=int, default=0)
    p.add_argument("--anchored", action="store_true")
    p.add_argument("--holdout_bars", type=int, default=-1, help="Reserve last N bars as untouched final holdout (-1 auto, 0 disables).")
    # Holdout permutation test (Loop 3; negative control)
    p.add_argument(
        "--holdout_perm_trials",
        type=int,
        default=0,
        help="Holdout permutation trials (circular time-shift of signals inside holdout). 0 disables.",
    )
    p.add_argument(
        "--holdout_perm_max_p",
        type=float,
        default=0.0,
        help="Max empirical p-value allowed for holdout permutation test (0 disables gating).",
    )
    # Holdout latency stress test (execution delay; real-money robustness)
    p.add_argument(
        "--holdout_latency_delay_bars",
        type=int,
        default=0,
        help="Extra bars of signal delay applied inside holdout (simulates latency). 0 disables. Typical: 1 for 1m/5m.",
    )
    p.add_argument(
        "--holdout_latency_return_ratio_min",
        type=float,
        default=0.30,
        help="Require delayed holdout return to retain at least this fraction of base holdout return (when base>0). 0 disables ratio gate.",
    )
    p.add_argument(
        "--no_holdout_latency",
        action="store_true",
        help="Disable holdout latency defaults (useful with --money_mode).",
    )
    # Holdout intrabar adverse-fill stress (STOP overshoot realism)
    p.add_argument(
        "--holdout_adverse_fill_stress_mult",
        type=float,
        default=-1.0,
        help="Override adverse_fill_slip_mult ONLY inside holdout for a stress run. -1 auto (money_mode->1.0 else 0 disables).",
    )
    p.add_argument(
        "--holdout_adverse_fill_return_ratio_min",
        type=float,
        default=0.30,
        help="Require adverse-fill-stressed holdout return to retain at least this fraction of base holdout return (when base>0). 0 disables ratio gate.",
    )
    p.add_argument(
        "--no_holdout_adverse_fill_stress",
        action="store_true",
        help="Disable holdout adverse-fill stress defaults (useful with --money_mode).",
    )
    # Holdout profit concentration gate (single-trade dependence)
    p.add_argument(
        "--holdout_max_top_profit_share",
        type=float,
        default=-1.0,
        help="Max share of holdout gross profits attributable to the single best trade. -1 auto (MoneyMode->0.60 else disables with 1.0).",
    )
    p.add_argument(
        "--no_holdout_profit_conc",
        action="store_true",
        help="Disable holdout profit concentration defaults (useful with --money_mode).",
    )
    # Holdout time-bucket profit concentration (temporal dependence)
    p.add_argument(
        "--holdout_profit_bucket_ms",
        type=int,
        default=-1,
        help="Holdout profit bucket size in ms for time-concentration check. -1 auto (uses funding_period_ms or 28800000).",
    )
    p.add_argument(
        "--holdout_max_top_bucket_profit_share",
        type=float,
        default=-1.0,
        help="Max share of HOLDOUT gross profits contributed by the best time bucket (0..1). -1 auto (money_mode->0.80 else 1.0).",
    )
    p.add_argument(
        "--no_holdout_time_conc",
        action="store_true",
        help="Disable holdout time-concentration defaults (useful with --money_mode).",
    )
    # Holdout volatility-regime profit concentration (market-condition dependence)
    p.add_argument(
        "--holdout_max_top_vol_profit_share",
        type=float,
        default=-1.0,
        help=(
            "Max share of HOLDOUT gross profits contributed by a single volatility regime bucket (vol_regime=-1/0/+1). "
            "-1 auto (money_mode->0.90 else 1.0)."
        ),
    )
    p.add_argument(
        "--no_holdout_vol_regime_conc",
        action="store_true",
        help="Disable holdout volatility-regime concentration defaults (useful with --money_mode).",
    )
    # Holdout segment consistency gate (temporal stability)
    p.add_argument(
        "--holdout_segments",
        type=int,
        default=-1,
        help="Split holdout into N segments and apply consistency gates. -1 auto (money_mode->3 else 0 disables).",
    )
    p.add_argument(
        "--holdout_min_pos_segment_ratio",
        type=float,
        default=-1.0,
        help="Min fraction of holdout segments with non-negative return (>=0). -1 auto (money_mode->0.67 else 0 disables).",
    )
    p.add_argument(
        "--holdout_min_segment_return",
        type=float,
        default=-2.0,
        help="Minimum allowed net return in any holdout segment. -2 auto (money_mode->-0.05 else -1 disables). -1 disables.",
    )
    p.add_argument(
        "--no_holdout_segments",
        action="store_true",
        help="Disable holdout segment-consistency defaults (useful with --money_mode).",
    )
    # Holdout parameter robustness test (Loop 5)
    p.add_argument(
        "--param_robust_trials",
        type=int,
        default=0,
        help="Holdout parameter robustness trials (perturb params and re-run holdout). 0 disables.",
    )
    p.add_argument(
        "--param_robust_jitter",
        type=float,
        default=0.10,
        help="Relative jitter magnitude for robustness (e.g., 0.10 => +/-10%%).",
    )
    p.add_argument(
        "--param_robust_min_pass_ratio",
        type=float,
        default=0.0,
        help="Min fraction of perturbations that must pass (0 disables gating).",
    )
    p.add_argument(
        "--param_robust_return_ratio_min",
        type=float,
        default=0.30,
        help="Each perturbation must retain at least this fraction of base holdout return (when base>0).",
    )
    p.add_argument(
        "--param_robust_trade_ratio_min",
        type=float,
        default=0.50,
        help="Each perturbation must retain at least this fraction of base holdout trades (>=1).",
    )
    p.add_argument(
        "--no_param_robust",
        action="store_true",
        help="Disable parameter-robustness defaults (useful with --money_mode).",
    )
    # Scoring tweaks (AUTO defaults = -1)
    p.add_argument("--fee_bps", type=float, default=4.0, help="Trading fee in bps per side (taker-like).")
    p.add_argument("--spread_bps", type=float, default=2.0, help="Simulated bid/ask spread in bps (applied on each fill).")
    p.add_argument("--slippage_bps", type=float, default=1.0, help="Additional slippage in bps (applied on each fill).")
    p.add_argument(
        "--adverse_fill_slip_mult",
        type=float,
        default=-1.0,
        help="Intrabar STOP/LIQ adverse fill slip. 0=fill at trigger (legacy), 1=fill at bar extreme. -1 auto (money_mode->0.25 else 0).",
    )
    # Funding (perpetual futures)
    # - none: disable funding
    # - constant: constant funding per funding interval (default 8h)
    # - binance: historical Binance funding events (engine backend). Requires epoch-ms timestamps + network access.
    # Alias: const == constant (backwards compatibility)
    p.add_argument(
        "--funding",
        type=str,
        default="none",
        choices=["none", "constant", "binance", "const"],
        help="Funding model: none | constant | binance (alias: const).",
    )
    p.add_argument(
        "--funding_bps",
        type=float,
        default=0.0,
        help="Constant funding rate per funding interval in bps (positive: longs pay shorts). Used when --funding constant/const.",
    )
    p.add_argument(
        "--funding_default_bps",
        type=float,
        default=0.0,
        help="Fallback funding bps (used when --funding binance and fundingTime missing; also legacy fallback for constant if funding_bps=0).",
    )
    p.add_argument(
        "--funding_period_ms",
        type=int,
        default=28_800_000,
        help="Funding interval in milliseconds (default 8h = 28800000).",
    )
    p.add_argument(
        "--funding_use_last_known",
        action="store_true",
        help="For binance funding: if exact timestamp missing, use last known <= timestamp.",
    )
    # Robustness: cost stress-test (recommended)
    p.add_argument(
        "--stress_cost_mult",
        type=float,
        default=2.0,
        help="Cost stress multiplier (fees+spread+slippage). <=1 disables stress test. Typical: 2.0",
    )
    p.add_argument(
        "--stress_min_mean_return",
        type=float,
        default=0.0,
        help="Minimum mean return across folds under stress costs (applied when stress_cost_mult>1).",
    )
    p.add_argument(
        "--stress_min_worst_fold_return",
        type=float,
        default=-0.0,
        help="Minimum worst-fold return under stress costs (applied when stress_cost_mult>1).",
    )
    p.add_argument("--weight_complexity", type=float, default=0.002)
    p.add_argument("--min_entries_total", type=int, default=-1, help="-1 auto")
    p.add_argument("--min_entries_per_fold", type=int, default=-1, help="-1 auto")
    p.add_argument("--min_trades_total", type=int, default=-1, help="-1 auto")
    p.add_argument("--min_trades_per_fold", type=int, default=-1, help="-1 auto")
    p.add_argument("--max_dd_limit", type=float, default=0.50)
    # Perpetual futures safety (engine backend): liquidation is a hard failure.
    # (Liquidations are tracked by Backtest_Engine when maintenance_margin_rate>0.)
    p.add_argument(
        "--max_liquidations_total",
        type=int,
        default=0,
        help="Reject if total liquidations across folds exceeds this (engine backend only). 0 = reject any liquidation.",
    )
    p.add_argument(
        "--max_liquidations_per_fold",
        type=int,
        default=0,
        help="Reject if any single fold exceeds this liquidation count (engine backend only). 0 = reject any liquidation.",
    )
    p.add_argument("--print_top", type=int, default=10)
    p.add_argument("--min_mean_return", type=float, default=0.0)
    p.add_argument("--min_worst_fold_return", type=float, default=-0.05)
    p.add_argument("--min_profit_factor", type=float, default=1.0)
    p.add_argument("--max_turnover_per_1000", type=float, default=500.0)
    # OOS profit concentration gate (avoid "one lucky trade" strategies)
    p.add_argument(
        "--max_top_profit_share",
        type=float,
        default=-1.0,
        help="Max share of OOS gross profits attributable to the single best trade across WF test folds. -1 auto (MoneyMode->0.70 else disables with 1.0).",
    )
    p.add_argument(
        "--no_profit_conc",
        action="store_true",
        help="Disable OOS profit concentration defaults (useful with --money_mode).",
    )
    # Selection-bias / robustness guards (Loop 2)
    p.add_argument(
        "--money_mode",
        action="store_true",
        help="Enable stricter defaults (auto holdout + excess-return scoring + consistency guards).",
    )
    p.add_argument(
        "--spirited_reminder",
        action="store_true",
        help="Ultra-conservative preset: implies --money_mode, avoids synthetic fallback (requires Binance unless --csv), defaults to Binance funding, and auto-runs cross-market validation.",
   )
    p.add_argument(
        "--return_mode",
        type=str,
        default="absolute",
        choices=["absolute", "excess"],
        help="Return scoring mode: absolute net returns or excess vs buy&hold baseline.",
    )
    p.add_argument(
        "--return_trade_damp_ref",
        type=int,
        default=30,
        help="Trades required for full return weight in scoring (penalizes low-trade strategies).",
    )
    p.add_argument(
        "--min_pos_fold_ratio",
        type=float,
        default=-1.0,
        help="-1 auto (money_mode), 0 disables. Fraction of folds with positive (used) return.",
    )
    p.add_argument(
        "--min_t_stat",
        type=float,
        default=0.0,
        help="Optional t-stat threshold on fold returns (used return mode). 0 disables.",
    )
    p.add_argument(
        "--max_p_value_adj",
        type=float,
        default=0.0,
        help="Optional Bonferroni-adjusted p-value threshold (used return mode). 0 disables.",
    )
    p.add_argument(
        "--multiple_testing_trials",
        type=int,
        default=0,
        help="Trials for p-value adjustment (0 => max_evals).",
    )
    p.add_argument(
        "--cscv_min_selected_fraction",
        type=float,
        default=-1.0,
        help="-1 auto (money_mode), 0 disables. CSCV gate: require strategy selected as IS-winner in at least this share of CSCV splits.",
    )
    p.add_argument(
        "--cscv_min_oos_rank_median",
        type=float,
        default=-1.0,
        help="-1 auto (money_mode), 0 disables. CSCV gate: require median OOS rank (0..1, when selected) at least this value.",
    )
    return p.parse_args(list(argv) if argv is not None else None)

def _series_sha1(series: Any) -> str:
    """
    Stable-ish fingerprint of the dataset used for mining.
    Uses (ts_ms, OHLCV) with fixed float formatting to avoid noise.

    Why include OHLCV (not only close):
    - Prevents false "same dataset" fingerprints when only highs/lows changed
      (common when a last, still-forming candle sneaks in).
    - Makes results reproducible and debuggable across runs.
    """
    h = hashlib.sha1()
    try:
        ts = getattr(series, "ts_ms", [])
        op = getattr(series, "open", [])
        hi = getattr(series, "high", [])
        lo = getattr(series, "low", [])
        cl = getattr(series, "close", [])
        vo = getattr(series, "volume", [])
        sym = str(getattr(series, "symbol", "") or "")
        tf = str(getattr(series, "timeframe", "") or "")
        h.update(f"symbol={sym}|timeframe={tf}|n={len(ts)}|".encode("utf-8"))

        for t, o, h0, l0, c0, v0 in zip(ts, op, hi, lo, cl, vo):
            h.update(str(int(t)).encode("utf-8"))
            h.update(b",")
            h.update(
                f"{float(o):.8f},{float(h0):.8f},{float(l0):.8f},{float(c0):.8f},{float(v0):.8f}".encode(
                    "utf-8"
                )
            )
            h.update(b";")
    except Exception:
        # fallback: still return something deterministic-ish
        h.update(repr(series).encode("utf-8"))
    return h.hexdigest()
# -----------------------------
# Validation helpers (Loop 13)
# -----------------------------
def _parse_validate_datasets(items: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Parse --validate entries.

    Accepted formats:
      SYMBOL:TIMEFRAME   (preferred)
      SYMBOL,TIMEFRAME   (fallback)
    """
    out: List[Tuple[str, str]] = []
    for raw in list(items or []):
        s = str(raw).strip()
        if not s:
            continue
        if ':' in s:
            a, b = s.split(':', 1)
        elif ',' in s:
            a, b = s.split(',', 1)
        else:
            raise ValidationError(f"Invalid --validate '{s}'. Use SYMBOL:TIMEFRAME (e.g. ETHUSDT:1m)")
        sym = str(a).strip().upper()
        tf = str(b).strip()
        require(bool(sym) and bool(tf), f"Invalid --validate '{s}'. Use SYMBOL:TIMEFRAME (e.g. ETHUSDT:1m)")
        out.append((sym, tf))

    # De-duplicate while preserving order
    seen: set = set()
    uniq: List[Tuple[str, str]] = []
    for sym, tf in out:
        key = f"{sym}:{tf}"
        if key in seen:
            continue
        seen.add(key)
        uniq.append((sym, tf))
    return uniq


def _select_primary_reject_reason(reject_reasons: Dict[str, int]) -> str:
    if not reject_reasons:
        return ''
    items = sorted(reject_reasons.items(), key=lambda kv: kv[1], reverse=True)
    return str(items[0][0])


def _summarize_candidate_for_validation(r: Any) -> Dict[str, Any]:
    """Extract a small, JSONable summary from a CandidateResult-like object."""
    agg = getattr(r, 'aggregate', None) or {}
    out: Dict[str, Any] = {}
    try:
        out['score'] = float(getattr(r, 'score', 0.0) or 0.0)
    except Exception:
        out['score'] = 0.0
    try:
        out['mean_return_used'] = float(agg.get('mean_return_used', agg.get('mean_return', 0.0)) or 0.0)
        out['worst_fold_return_used'] = float(agg.get('worst_fold_return_used', agg.get('worst_fold_return', 0.0)) or 0.0)
        out['mean_drawdown'] = float(agg.get('mean_drawdown', 0.0) or 0.0)
        out['mean_profit_factor'] = float(agg.get('mean_profit_factor', 0.0) or 0.0)
        out['trades_total'] = int(agg.get('trades_total', 0) or 0)
        out['entries_total'] = int(agg.get('entries_total', 0) or 0)
        out['mean_hold_bars'] = float(agg.get('mean_hold_bars', 0.0) or 0.0)
    except Exception:
        pass

    ho = getattr(r, 'holdout_aggregate', None)
    if isinstance(ho, dict) and ho:
        try:
            out['holdout_return'] = float(ho.get('net_return', 0.0) or 0.0)
            out['holdout_drawdown'] = float(ho.get('max_drawdown', 0.0) or 0.0)
            out['holdout_trades'] = int(ho.get('trades', 0) or 0)
        except Exception:
            pass
        pt = ho.get('perm_test', None)
        if isinstance(pt, dict) and pt:
            try:
                if pt.get('p_value', None) is not None:
                    out['holdout_perm_p'] = float(pt.get('p_value'))
            except Exception:
                pass
    return out
def _spirited_default_validate_sets(*, symbol: str, timeframe: str) -> List[str]:
    """
    Default cross-market validation basket (SpiritedReminder).

    We keep this intentionally small (runtime), but require that any mined edge is not a
    single-market fluke.

    Returns:
      A list of strings in the same format accepted by --validate (SYMBOL:TIMEFRAME).
    """
    sym = str(symbol).strip().upper()
    tf = str(timeframe).strip()

    # Core, high-liquidity perp symbols.
    basket = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

    if sym in basket:
        candidates = [s for s in basket if s != sym]
    else:
        # If mining an alt, validate on majors to test "is this a general edge or a coincidence?"
        candidates = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    out: List[str] = []
    for s in candidates:
        entry = f"{s}:{tf}"
        if entry not in out:
            out.append(entry)
        if len(out) >= 2:
            break
    return out

def _validate_spec_on_series(
    *,
    spec_obj: Any,
    series: OhlcvSeries,
    wf: WalkForwardConfig,
    base_search: SearchConfig,
    scoring: ScoringConfig,
) -> Dict[str, Any]:
    """Replay exactly one StrategySpec on a given series using the miner gates."""
    class _SingleSpecSpace:
        def __init__(self, spec: Any) -> None:
            self._spec = spec
        def iter_strategies(self) -> Any:
            yield self._spec

    # Validation should be deterministic: never sample-drop the single candidate,
    # and don't diversify away the only candidate.
    val_search = replace(
        base_search,
        mode='iterate',
        sample_prob=1.0,
        top_k=1,
        max_variants_per_entry=1,
        diversify_by_entry=False,
        diversify_by_family=False,
    )

    summary: Dict[str, Any] = {'passed': False}
    try:
        miner = StrategyMiner(series, _SingleSpecSpace(spec_obj), wf, val_search, scoring)
        rep = miner.run()
    except Exception as e:
        summary['error'] = f"{type(e).__name__}: {e}"
        return summary

    passed = bool(rep.accepted > 0 and getattr(rep, 'top_results', None))
    summary['passed'] = passed
    summary['reject_reason'] = _select_primary_reject_reason(getattr(rep, 'reject_reasons', {}) or {})
    summary['reject_reasons'] = dict(getattr(rep, 'reject_reasons', {}) or {})

    if passed:
        r0 = rep.top_results[0]
        summary['result'] = _summarize_candidate_for_validation(r0)
    else:
        dbg = list(getattr(rep, 'debug_top', ()) or [])
        if dbg:
            summary['best_reject'] = {
                'reject_reason': str(getattr(dbg[0], 'reject_reason', '') or ''),
                'summary': _summarize_candidate_for_validation(dbg[0]),
            }
    return summary
# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    symbol = str(args.symbol).strip().upper()
    timeframe = str(args.timeframe).strip()
    limit = int(args.limit)
    spirited = bool(getattr(args, "spirited_reminder", False))
    if spirited:
        print("SpiritedReminder: enabled (ultra-conservative mining preset).")
        # SpiritedReminder implies MoneyMode.
        args.money_mode = True
        # CSV mode is allowed; otherwise Binance download requires network access.
        # Increase history length if needed (walk-forward/holdout gates need enough data).
        try:
            rec = _recommended_min_bars(timeframe)
            if int(args.limit) < int(rec) and not str(getattr(args, "csv", "")).strip():
                print(f"SpiritedReminder: increasing --limit from {int(args.limit)} to {int(rec)} bars.")
                args.limit = int(rec)
                limit = int(args.limit)
        except Exception:
            pass
    # ---- Load series ----
    series: Optional[OhlcvSeries] = None
    data_source = ""

    try:
        if str(args.csv).strip():
            series = _load_series_csv(str(args.csv).strip(), symbol=symbol, timeframe=timeframe)
            data_source = f"CSV:{args.csv}"
        else:
            try:
                series = _load_series_binance_best_effort(symbol, timeframe, limit)
                data_source = "BINANCE(Data_Binance)"
            except Exception as e:
                # No automatic synthetic fallback: this miner is for deployable strategies only.
                # No synthetic fallback: provide --csv or fix connectivity.
                raise ValidationError(
                    "Binance load failed and synthetic fallback is disabled. "
                    "Fix connectivity or provide --csv."
                ) from e
    except Exception as e:
        print("ERROR: Failed to load series.")
        print(f"{type(e).__name__}: {e}")
        return 2

    require(series is not None, "internal: series is None after load")
    # ---- Data hygiene (fail-fast) ----
    # - Drop last incomplete candle (common with live Binance downloads)
    # - Enforce uniform timeframe spacing (detect gaps / mixed TF)
    try:
        series = _sanitize_series_for_backtest(series, timeframe=timeframe, source_label=data_source)
    except Exception as e:
        print("ERROR: Series failed hygiene checks.")
        print(f"{type(e).__name__}: {e}")
        return 2

    n = len(series.ts_ms)
    require(n >= 200, f"Need at least 200 bars, got {n}")

    print(f"Data source: {data_source}")
    print(f"Series: symbol={series.symbol} timeframe={series.timeframe} bars={n}")
    # ---- Money mode (Loop 2): bias toward settings you could actually deploy ----
    money_mode = bool(getattr(args, "money_mode", False))
    if money_mode:
        # Force more conservative research defaults.
        try:
            args.return_mode = "excess"
        except Exception:
            pass

        # Auto-holdout if user didn't specify one.
        if int(getattr(args, "holdout_bars", 0) or 0) <= 0:
            auto_hb = max(0, int(round(0.20 * float(n))))
            min_left = 300
            if n - auto_hb < min_left:
                auto_hb = max(0, int(n - min_left))
            if auto_hb >= n:
                auto_hb = 0
            args.holdout_bars = int(auto_hb)
            if auto_hb > 0:
                print(f"MoneyMode: auto holdout_bars={int(auto_hb)} (untouched final segment)")
    # MoneyMode safety: enforce the engine backend (funding/margin/liquidation realism).
    if money_mode:
        bt = str(getattr(args, "backtest", "auto") or "auto").strip().lower()
        if bt == "auto":
            args.backtest = "engine"
            print("MoneyMode: forcing --backtest engine (perp realism: margin/liquidation/funding).")
        elif bt != "engine":
            print("ERROR: --money_mode requires --backtest engine.")
            return 2
    # SpiritedReminder extras: funding realism + cross-market validation defaults.
    if bool(getattr(args, "spirited_reminder", False)):
        # Default to historical Binance funding when running on real Binance candles.
        fm = str(getattr(args, "funding", "none") or "none").strip().lower()
        if fm in ("none", "") and not str(getattr(args, "csv", "")).strip():
            try:
                if _looks_like_epoch_ms(getattr(series, "ts_ms", []) or []):
                    args.funding = "binance"
                    print("SpiritedReminder: defaulting --funding binance (historical funding included).")
                else:
                    print("SpiritedReminder: non-epoch timestamps; leaving funding disabled.")
            except Exception:
                pass

        # If the user didn't provide --validate, auto-validate on a small basket of majors.
        if not str(getattr(args, "csv", "")).strip():
            try:
                if not list(getattr(args, "validate", []) or []):
                    args.validate = _spirited_default_validate_sets(symbol=symbol, timeframe=timeframe)
                    if args.validate:
                        print("SpiritedReminder: auto --validate " + ", ".join(list(args.validate)))
            except Exception:
                pass
    # ---- Engine BacktestConfig overrides (Loop 1) ----
    # Especially important for perp futures: funding is a real cost/rebate.
    engine_overrides: Dict[str, Any] = {}
    funding_mode = str(getattr(args, "funding", "none") or "none").strip().lower()
    if funding_mode == "const":
        funding_mode = "constant"
    if funding_mode not in ("none", "constant", "binance"):
        print(f"WARNING: unknown --funding {funding_mode!r}; treating as 'none'.")
        funding_mode = "none"

    if funding_mode != "none":
        # quick heuristic: real epoch ms timestamps are ~1e12+
        epochish = int(series.ts_ms[0]) > 1_000_000_000_000

        if funding_mode == "constant":
            funding_period_ms = int(getattr(args, "funding_period_ms", 28_800_000) or 28_800_000)
            bps = float(getattr(args, "funding_bps", 0.0) or 0.0)
            if abs(bps) < 1e-12:
                # Backwards compat: old runs used funding_default_bps for constant mode
                bps = float(getattr(args, "funding_default_bps", 0.0) or 0.0)
            fr = float(bps) / 10000.0
            engine_overrides["funding_period_ms"] = int(funding_period_ms)
            engine_overrides["funding_rate_per_period"] = float(fr)
            print(f"Funding(constant): {bps:.4f} bps per interval ({int(funding_period_ms)} ms)")

        elif funding_mode == "binance":
            # Binance funding happens on fixed 8h boundaries.
            engine_overrides["funding_period_ms"] = 28_800_000
            if not epochish:
                msg = (
                    "Series timestamps do not look like epoch ms; cannot fetch Binance funding rates. "
                    "Use real Binance data or disable funding."
                )
                if money_mode:
                    print("ERROR: " + msg)
                    return 2
                print("WARNING: " + msg)
            else:
                try:
                    import Data_Binance as DB

                    start_ms = int(series.ts_ms[0])
                    end_ms = int(series.ts_ms[-1])

                    rates = DB.download_funding_rates(symbol=symbol, start_ms=start_ms, end_ms=end_ms, base_url=DB.BINANCE_USDM_MAINNET)
                    default_rate = float(getattr(args, "funding_default_bps", 0.0)) / 10000.0
                    use_last = bool(getattr(args, "funding_use_last_known", False))
                    fn = DB.build_funding_rate_fn(rates, default_rate=default_rate, use_last_known=use_last)

                    engine_overrides["funding_rate_fn"] = fn
                    engine_overrides["funding_rate_per_period"] = 0.0

                    if rates:
                        vals = [float(r) for _, r in rates]
                        mean = sum(vals) / float(len(vals))
                        mn = min(vals)
                        mx = max(vals)
                        print(
                            f"Funding(Binance): loaded {len(rates)} events; "
                            f"mean={mean*10000:.4f} bps min={mn*10000:.4f} bps max={mx*10000:.4f} bps"
                        )
                    else:
                        print("Funding(Binance): no events returned for range; using default funding only.")

                except Exception as e:
                    msg = f"Failed to load Binance funding rates: {type(e).__name__}: {e}"
                    if money_mode:
                        print("ERROR: " + msg)
                        return 2
                    print("WARNING: " + msg)
    replay_specs: Optional[List[StrategySpec]] = None
    replay_path = str(getattr(args, "replay_json", "") or "").strip()
    if replay_path:
        try:
            rid = str(getattr(args, "replay_ids", "") or "").strip()
            ids = [x.strip() for x in rid.split(",") if x.strip()] if rid else None
            replay_specs = _load_replay_specs_from_results_json(
                replay_path,
                top_n=int(getattr(args, "replay_top", 0) or 0),
                ids=ids,
            )
            # Force pure evaluation behavior (no search-space diversification caps)
            args.mode = "iterate"
            args.max_evals = int(len(replay_specs))
            if int(getattr(args, "top_k", 0) or 0) <= 0:
                args.top_k = int(len(replay_specs))
            else:
                args.top_k = min(int(args.top_k), int(len(replay_specs)))
            args.no_diversify_by_entry = True
            args.no_diversify_by_family = True
            print(f"Replay: loaded {len(replay_specs)} strategies from {replay_path}")
        except Exception as e:
            print(f"ERROR: failed to load --replay_json {replay_path!r}")
            print(f"{type(e).__name__}: {e}")
            return 2

    # ---- Space preset ----
    # CLI sugar: allow --space tiny as alias for small (older docs/examples).
    if str(getattr(args, "space", "small") or "small").strip().lower() == "tiny":
        args.space = "small"

    if replay_specs:
        space = _ReplaySpace(replay_specs)
    else:
        cfg, budget = _space_preset(str(args.space))
        space = StrategySpace(cfg, budget)

    # ---- Walk-forward ----
    # holdout_bars semantics:
    #   -1 => AUTO (recommended for research / real-money conclusions)
    #    0 => disabled (unsafe; higher risk of selecting noise)
    #   >0 => explicit number of bars
    holdout_arg = int(getattr(args, "holdout_bars", -1))

    if int(args.train_bars) > 0 and int(args.test_bars) > 0:
        train_bars = int(args.train_bars)
        test_bars = int(args.test_bars)
        step_bars = int(args.step_bars) if int(args.step_bars) > 0 else int(test_bars)

        hb = int(holdout_arg)
        if hb < 0:
            hb = _auto_holdout_bars(
                n,
                train_bars=train_bars,
                test_bars=test_bars,
                purge_bars=int(args.purge_bars),
                embargo_bars=int(args.embargo_bars),
            )
            if hb <= 0:
                print("ERROR: AUTO holdout could not be allocated with the current train/test sizes.")
                print(
                    "HINT: Increase --limit (more bars), reduce --train_bars/--test_bars, "
                    "or disable holdout via --holdout_bars 0 (unsafe)."
                )
                return 2

        wf = WalkForwardConfig(
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            purge_bars=int(args.purge_bars),
            embargo_bars=int(args.embargo_bars),
            anchored=bool(args.anchored),
            holdout_bars=int(hb),
        )
    else:
        wf0 = _auto_walkforward_money(n) if money_mode else _auto_walkforward(n)

        hb = int(holdout_arg)
        if hb < 0:
            hb = _auto_holdout_bars(
                n,
                train_bars=int(wf0.train_bars),
                test_bars=int(wf0.test_bars),
                purge_bars=int(args.purge_bars),
                embargo_bars=int(args.embargo_bars),
            )
            if hb <= 0:
                print("ERROR: AUTO holdout could not be allocated (dataset too small for holdout + WF folds).")
                print("HINT: Increase --limit (more bars) or disable holdout via --holdout_bars 0 (unsafe).")
                return 2

        # Holdout reduces usable bars. Size AUTO-WF on n_eff so we typically get more folds
        # (less selection bias / overfit).
        n_eff = int(n - hb) if (hb > 0 and hb < n) else int(n)
        wf_auto = _auto_walkforward_money(n_eff) if money_mode else _auto_walkforward(n_eff)

        wf = WalkForwardConfig(
            train_bars=int(wf_auto.train_bars),
            test_bars=int(wf_auto.test_bars),
            step_bars=int(wf_auto.step_bars),
            purge_bars=int(args.purge_bars),
            embargo_bars=int(args.embargo_bars),
            anchored=bool(args.anchored),
            holdout_bars=int(hb),
        )

    try:
        splits = make_walkforward_splits(n, wf)
    except Exception as e:
        print("ERROR: Could not create walk-forward splits.")
        print(f"{type(e).__name__}: {e}")
        print("HINT: Increase --limit, reduce train/test sizes, or disable holdout via --holdout_bars 0 (unsafe).")
        return 2

    folds = len(splits)
    print(
        f"WalkForward: folds={folds} train={wf.train_bars} test={wf.test_bars} step={wf.step_bars} "
        f"purge={wf.purge_bars} embargo={wf.embargo_bars} anchored={wf.anchored} holdout={getattr(wf, 'holdout_bars', 0)}"
    )

    holdout_bars_final = int(getattr(wf, 'holdout_bars', 0) or 0)

    # ---- Scoring AUTO defaults ----
    # AUTO is intentionally conservative: avoid "1-2 trade wonders" that look great by chance.
    # You can still override any minimum explicitly via CLI.
    _base = ScoringConfig()  # use class defaults as lower bounds
    total_test_bars = sum(int(s.test.stop) - int(s.test.start) for s in splits) if splits else 0

    # Heuristic (AUTO):
    # - Avoid "1-2 trade wonders" by requiring a minimum sample size.
    # - Keep it realistic across timeframes by expressing the requirement as "bars per trade".
    #
    # Target: at least ~1 trade per ~300 out-of-sample (test) bars.
    # Example: on 5m, 300 bars ≈ 25 hours ≈ ~1 trade/day.
    BARS_PER_TRADE_TARGET = 300
    auto_trades_total = max(
        int(_base.min_trades_total),
        int(folds) * 2,
        int(total_test_bars // max(1, BARS_PER_TRADE_TARGET)),
    )
    # Per-fold requirement: at least ~25% of the average trades per fold,
    # so folds don't "pass" with only 1 trade in one fold and 0 in another.
    avg_per_fold = int(math.ceil(float(auto_trades_total) / float(max(1, folds))))
    auto_trades_per_fold = max(int(_base.min_trades_per_fold), 1, int(math.ceil(avg_per_fold * 0.25)))

    auto_entries_per_fold = max(int(_base.min_entries_per_fold), auto_trades_per_fold)
    auto_entries_total = max(int(_base.min_entries_total), auto_trades_total)


    min_entries_total = int(args.min_entries_total) if int(args.min_entries_total) >= 0 else int(auto_entries_total)
    min_entries_per_fold = int(args.min_entries_per_fold) if int(args.min_entries_per_fold) >= 0 else int(auto_entries_per_fold)
    min_trades_total = int(args.min_trades_total) if int(args.min_trades_total) >= 0 else int(auto_trades_total)
    min_trades_per_fold = int(args.min_trades_per_fold) if int(args.min_trades_per_fold) >= 0 else int(auto_trades_per_fold)
    # AUTO: fold-positivity ratio (Loop 2)
    # A cheap guard against strategies that only work in a minority of windows.
    try:
        arg_pos = float(getattr(args, "min_pos_fold_ratio", -1.0))
    except Exception:
        arg_pos = -1.0

    if arg_pos < 0.0:
        if bool(getattr(args, "money_mode", False)):
            if folds >= 5:
                min_pos_fold_ratio = 0.6
            elif folds == 4:
                min_pos_fold_ratio = 0.5
            elif folds == 3:
                min_pos_fold_ratio = 2.0 / 3.0
            else:
                min_pos_fold_ratio = 0.0
        else:
            min_pos_fold_ratio = 0.0
    else:
        min_pos_fold_ratio = max(0.0, min(1.0, float(arg_pos)))
    # Loop 15: CSCV stability gate defaults
    # Uses CSCV per-strategy stats computed on the final pool (top_k) with no extra backtests.
    try:
        arg_cscv_sel = float(getattr(args, "cscv_min_selected_fraction", -1.0))
    except Exception:
        arg_cscv_sel = -1.0
    try:
        arg_cscv_oos = float(getattr(args, "cscv_min_oos_rank_median", -1.0))
    except Exception:
        arg_cscv_oos = -1.0

    if arg_cscv_sel < 0.0:
        if bool(getattr(args, "money_mode", False)) and folds >= 4:
            cscv_min_selected_fraction = 0.03
        else:
            cscv_min_selected_fraction = 0.0
    else:
        cscv_min_selected_fraction = max(0.0, min(1.0, float(arg_cscv_sel)))

    if arg_cscv_oos < 0.0:
        if bool(getattr(args, "money_mode", False)) and folds >= 4:
            cscv_min_oos_rank_median = 0.55
        else:
            cscv_min_oos_rank_median = 0.0
    else:
        cscv_min_oos_rank_median = max(0.0, min(1.0, float(arg_cscv_oos)))

    # Note: CSCV needs a pool of strategies (top_k >= 8). If the pool is smaller, the miner will skip the gate.
    if (float(cscv_min_selected_fraction) > 0.0 or float(cscv_min_oos_rank_median) > 0.0) and int(getattr(args, "top_k", 0) or 0) < 8:
        print("WARNING: CSCV gate enabled but --top_k < 8. Increase --top_k so CSCV can be computed/applied.")

    # Loop 3: holdout permutation test defaults (negative control)
    try:
        holdout_perm_trials = int(getattr(args, "holdout_perm_trials", 0) or 0)
    except Exception:
        holdout_perm_trials = 0
    try:
        holdout_perm_max_p = float(getattr(args, "holdout_perm_max_p", 0.0) or 0.0)
    except Exception:
        holdout_perm_max_p = 0.0

    holdout_perm_trials = max(0, int(holdout_perm_trials))
    holdout_perm_max_p = max(0.0, min(1.0, float(holdout_perm_max_p)))

    # MoneyMode: enable the permutation gate by default (only if we actually have a holdout slice)
    if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0:
        if int(holdout_perm_trials) <= 0:
            holdout_perm_trials = 100
        if float(holdout_perm_max_p) <= 0.0:
            holdout_perm_max_p = 0.10
    # Loop 5: holdout parameter robustness (parameter sensitivity test)
    try:
        param_robust_trials = int(getattr(args, "param_robust_trials", 0) or 0)
    except Exception:
        param_robust_trials = 0
    try:
        param_robust_jitter = float(getattr(args, "param_robust_jitter", 0.10) or 0.10)
    except Exception:
        param_robust_jitter = 0.10
    try:
        param_robust_min_pass_ratio = float(getattr(args, "param_robust_min_pass_ratio", 0.0) or 0.0)
    except Exception:
        param_robust_min_pass_ratio = 0.0
    try:
        param_robust_return_ratio_min = float(getattr(args, "param_robust_return_ratio_min", 0.30) or 0.30)
    except Exception:
        param_robust_return_ratio_min = 0.30
    try:
        param_robust_trade_ratio_min = float(getattr(args, "param_robust_trade_ratio_min", 0.50) or 0.50)
    except Exception:
        param_robust_trade_ratio_min = 0.50

    param_robust_trials = max(0, int(param_robust_trials))
    param_robust_jitter = max(0.0, min(1.0, float(param_robust_jitter)))
    param_robust_min_pass_ratio = max(0.0, min(1.0, float(param_robust_min_pass_ratio)))
    param_robust_return_ratio_min = max(0.0, min(1.0, float(param_robust_return_ratio_min)))
    param_robust_trade_ratio_min = max(0.0, min(1.0, float(param_robust_trade_ratio_min)))

    # Explicit opt-out or missing holdout => disable
    if bool(getattr(args, "no_param_robust", False)) or int(holdout_bars_final) <= 0:
        param_robust_trials = 0
        param_robust_min_pass_ratio = 0.0

    # MoneyMode defaults: enable robustness gate by default (only with a holdout)
    if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0 and not bool(getattr(args, "no_param_robust", False)):
        if int(param_robust_trials) <= 0:
            param_robust_trials = 12
        if float(param_robust_min_pass_ratio) <= 0.0:
            param_robust_min_pass_ratio = 0.60
        if float(param_robust_return_ratio_min) <= 0.0:
            param_robust_return_ratio_min = 0.30
        if float(param_robust_trade_ratio_min) <= 0.0:
            param_robust_trade_ratio_min = 0.50
        if float(param_robust_jitter) <= 0.0:
            param_robust_jitter = 0.10
    # Holdout latency stress (execution delay; fragility killer for 1m strategies)
    try:
        holdout_latency_delay_bars = int(getattr(args, "holdout_latency_delay_bars", 0) or 0)
    except Exception:
        holdout_latency_delay_bars = 0
    try:
        holdout_latency_return_ratio_min = float(getattr(args, "holdout_latency_return_ratio_min", 0.30) or 0.30)
    except Exception:
        holdout_latency_return_ratio_min = 0.30

    holdout_latency_delay_bars = max(0, int(holdout_latency_delay_bars))
    holdout_latency_return_ratio_min = max(0.0, min(1.0, float(holdout_latency_return_ratio_min)))

    # Explicit opt-out or missing holdout => disable
    if bool(getattr(args, "no_holdout_latency", False)) or int(holdout_bars_final) <= 0:
        holdout_latency_delay_bars = 0

    # MoneyMode defaults: enable latency gate by default (only with a holdout)
    if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0 and not bool(getattr(args, "no_holdout_latency", False)):
        if int(holdout_latency_delay_bars) <= 0:
            holdout_latency_delay_bars = 1
        if float(holdout_latency_return_ratio_min) <= 0.0:
            holdout_latency_return_ratio_min = 0.30
        if int(holdout_latency_delay_bars) > 0:
            print(f"MoneyMode: holdout_latency_delay_bars={int(holdout_latency_delay_bars)} (execution-delay stress)")
    # Holdout intrabar adverse-fill stress (STOP overshoot realism)
    try:
        holdout_adverse_fill_stress_mult = float(getattr(args, "holdout_adverse_fill_stress_mult", -1.0))
    except Exception:
        holdout_adverse_fill_stress_mult = -1.0
    try:
        holdout_adverse_fill_return_ratio_min = float(getattr(args, "holdout_adverse_fill_return_ratio_min", 0.30) or 0.30)
    except Exception:
        holdout_adverse_fill_return_ratio_min = 0.30

    holdout_adverse_fill_return_ratio_min = max(0.0, min(1.0, float(holdout_adverse_fill_return_ratio_min)))

    if float(holdout_adverse_fill_stress_mult) < 0.0:
        if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0 and not bool(getattr(args, "no_holdout_adverse_fill_stress", False)):
            holdout_adverse_fill_stress_mult = 1.0
        else:
            holdout_adverse_fill_stress_mult = 0.0

    holdout_adverse_fill_stress_mult = max(0.0, min(1.0, float(holdout_adverse_fill_stress_mult)))

    # Explicit opt-out or missing holdout => disable
    if bool(getattr(args, "no_holdout_adverse_fill_stress", False)) or int(holdout_bars_final) <= 0:
        holdout_adverse_fill_stress_mult = 0.0

    # MoneyMode defaults: enable adverse-fill stress by default (only with a holdout)
    if (
        bool(getattr(args, "money_mode", False))
        and int(holdout_bars_final) > 0
        and not bool(getattr(args, "no_holdout_adverse_fill_stress", False))
    ):
        if float(holdout_adverse_fill_stress_mult) <= 0.0:
            holdout_adverse_fill_stress_mult = 1.0
        if float(holdout_adverse_fill_return_ratio_min) <= 0.0:
            holdout_adverse_fill_return_ratio_min = 0.30
        if float(holdout_adverse_fill_stress_mult) > 0.0:
            print(
                f"MoneyMode: holdout_adverse_fill_stress_mult={float(holdout_adverse_fill_stress_mult):g} "
                f"ret_ratio_min={float(holdout_adverse_fill_return_ratio_min):g} (intrabar STOP stress)"
            )
    # OOS profit concentration gate (avoid "one lucky trade" strategies in WF folds)
    try:
        max_top_profit_share = float(getattr(args, "max_top_profit_share", -1.0))
    except Exception:
        max_top_profit_share = -1.0

    if float(max_top_profit_share) < 0.0:
        max_top_profit_share = 0.70 if bool(getattr(args, "money_mode", False)) else 1.0

    max_top_profit_share = max(0.0, min(1.0, float(max_top_profit_share)))

    # Explicit opt-out => disable
    if bool(getattr(args, "no_profit_conc", False)):
        max_top_profit_share = 1.0

    # MoneyMode default: enable profit concentration gate on OOS folds
    if bool(getattr(args, "money_mode", False)) and not bool(getattr(args, "no_profit_conc", False)):
        if float(max_top_profit_share) >= 1.0:
            max_top_profit_share = 0.70
        if float(max_top_profit_share) < 1.0:
            print(f"MoneyMode: max_top_profit_share={float(max_top_profit_share):g} (profit concentration gate on OOS folds)")

    # Holdout profit concentration gate (avoid "one lucky trade" strategies)
    try:
        holdout_max_top_profit_share = float(getattr(args, "holdout_max_top_profit_share", -1.0))
    except Exception:
        holdout_max_top_profit_share = -1.0

    if float(holdout_max_top_profit_share) < 0.0:
        holdout_max_top_profit_share = 0.60 if bool(getattr(args, "money_mode", False)) else 1.0

    holdout_max_top_profit_share = max(0.0, min(1.0, float(holdout_max_top_profit_share)))

    # Explicit opt-out or missing holdout => disable
    if bool(getattr(args, "no_holdout_profit_conc", False)) or int(holdout_bars_final) <= 0:
        holdout_max_top_profit_share = 1.0

    # MoneyMode default: enable profit concentration gate (only with a holdout)
    if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0 and not bool(getattr(args, "no_holdout_profit_conc", False)):
        if float(holdout_max_top_profit_share) >= 1.0:
            holdout_max_top_profit_share = 0.60
        if float(holdout_max_top_profit_share) < 1.0:
            print(f"MoneyMode: holdout_max_top_profit_share={float(holdout_max_top_profit_share):g} (profit concentration gate)")
    # Holdout time-bucket profit concentration (temporal dependence)
    try:
        holdout_profit_bucket_ms = int(getattr(args, "holdout_profit_bucket_ms", -1) or -1)
    except Exception:
        holdout_profit_bucket_ms = -1
    if int(holdout_profit_bucket_ms) <= 0:
        # Auto: align to funding schedule if provided; default to 8h.
        try:
            fb = int(getattr(args, "funding_period_ms", 28_800_000) or 28_800_000)
        except Exception:
            fb = 28_800_000
        holdout_profit_bucket_ms = int(fb) if int(fb) > 0 else 28_800_000
    holdout_profit_bucket_ms = max(1, int(holdout_profit_bucket_ms))

    try:
        holdout_max_top_bucket_profit_share = float(getattr(args, "holdout_max_top_bucket_profit_share", -1.0))
    except Exception:
        holdout_max_top_bucket_profit_share = -1.0

    if float(holdout_max_top_bucket_profit_share) < 0.0:
        if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0 and not bool(getattr(args, "no_holdout_time_conc", False)):
            holdout_max_top_bucket_profit_share = 0.80
        else:
            holdout_max_top_bucket_profit_share = 1.0
    holdout_max_top_bucket_profit_share = max(0.0, min(1.0, float(holdout_max_top_bucket_profit_share)))

    if bool(getattr(args, "no_holdout_time_conc", False)) or int(holdout_bars_final) <= 0:
        holdout_max_top_bucket_profit_share = 1.0

    if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0 and float(holdout_max_top_bucket_profit_share) < 1.0:
        print(
            f"MoneyMode: holdout_max_top_bucket_profit_share={float(holdout_max_top_bucket_profit_share):g} "
            f"bucket_ms={int(holdout_profit_bucket_ms)} (reject time-clustered holdout profits)"
        )
    # Holdout volatility-regime profit concentration (market-condition dependence)
    try:
        holdout_max_top_vol_profit_share = float(getattr(args, "holdout_max_top_vol_profit_share", -1.0))
    except Exception:
        holdout_max_top_vol_profit_share = -1.0

    if float(holdout_max_top_vol_profit_share) < 0.0:
        if (
            bool(getattr(args, "money_mode", False))
            and int(holdout_bars_final) > 0
            and not bool(getattr(args, "no_holdout_vol_regime_conc", False))
        ):
            holdout_max_top_vol_profit_share = 0.90
        else:
            holdout_max_top_vol_profit_share = 1.0

    holdout_max_top_vol_profit_share = max(0.0, min(1.0, float(holdout_max_top_vol_profit_share)))

    if bool(getattr(args, "no_holdout_vol_regime_conc", False)) or int(holdout_bars_final) <= 0:
        holdout_max_top_vol_profit_share = 1.0

    if bool(getattr(args, "money_mode", False)) and int(holdout_bars_final) > 0 and float(holdout_max_top_vol_profit_share) < 1.0:
        print(
            f"MoneyMode: holdout_max_top_vol_profit_share={float(holdout_max_top_vol_profit_share):g} "
            f"(reject vol-regime-clustered holdout profits)"
        )

    # Holdout segment consistency gate (temporal stability)
    try:
        holdout_segment_count = int(getattr(args, "holdout_segments", -1) or -1)
    except Exception:
        holdout_segment_count = -1
    try:
        holdout_min_pos_segment_ratio = float(getattr(args, "holdout_min_pos_segment_ratio", -1.0))
    except Exception:
        holdout_min_pos_segment_ratio = -1.0
    try:
        holdout_min_segment_return = float(getattr(args, "holdout_min_segment_return", -2.0))
    except Exception:
        holdout_min_segment_return = -2.0

    # Explicit opt-out or missing holdout => disable
    if bool(getattr(args, "no_holdout_segments", False)) or int(holdout_bars_final) <= 0:
        holdout_segment_count = 0
        holdout_min_pos_segment_ratio = 0.0
        holdout_min_segment_return = -1.0
    else:
        # Auto defaults
        if int(holdout_segment_count) < 0:
            holdout_segment_count = 3 if bool(getattr(args, "money_mode", False)) else 0
        holdout_segment_count = max(0, int(holdout_segment_count))

        if float(holdout_min_pos_segment_ratio) < 0.0:
            holdout_min_pos_segment_ratio = (2.0 / 3.0) if bool(getattr(args, "money_mode", False)) else 0.0
        holdout_min_pos_segment_ratio = max(0.0, min(1.0, float(holdout_min_pos_segment_ratio)))

        # -2 (or < -1) means "auto"; -1 disables.
        if float(holdout_min_segment_return) < -1.0:
            holdout_min_segment_return = -0.05 if bool(getattr(args, "money_mode", False)) else -1.0
        holdout_min_segment_return = max(-1.0, float(holdout_min_segment_return))

        # If segmentation is disabled, disable the derived gates to avoid confusion.
        if int(holdout_segment_count) <= 1:
            holdout_min_pos_segment_ratio = 0.0
            holdout_min_segment_return = -1.0

        if (
            bool(getattr(args, "money_mode", False))
            and int(holdout_bars_final) > 0
            and int(holdout_segment_count) > 1
            and (float(holdout_min_pos_segment_ratio) > 0.0 or float(holdout_min_segment_return) > -1.0)
        ):
            print(
                f"MoneyMode: holdout_segments={int(holdout_segment_count)} "
                f"min_pos_segment_ratio={float(holdout_min_pos_segment_ratio):g} "
                f"min_segment_return={float(holdout_min_segment_return):g} (reject unstable holdout sub-windows)"
            )

    # Engine realism knobs (account/sizing/margin) -> Backtest_Engine.BacktestConfig overrides:

    lev = float(getattr(args, "engine_leverage", 0.0) or 0.0)
    if lev < 0.0:
        print("ERROR: --engine_leverage must be >= 0.")
        return 2
    if lev > 0.0:
        engine_overrides["leverage"] = float(lev)

    rf = float(getattr(args, "engine_risk_frac", 0.0) or 0.0)
    if rf < 0.0 or rf > 1.0:
        print("ERROR: --engine_risk_frac must be in [0, 1].")
        return 2
    if rf > 0.0:
        engine_overrides["risk_per_trade_fraction"] = float(rf)

    mmf = float(getattr(args, "engine_max_margin_frac", 0.0) or 0.0)
    if mmf < 0.0 or mmf > 1.0:
        print("ERROR: --engine_max_margin_frac must be in [0, 1].")
        return 2
    if mmf > 0.0:
        engine_overrides["max_margin_fraction"] = float(mmf)

    mmr = float(getattr(args, "engine_maintenance_margin_rate", 0.0) or 0.0)
    if mmr < 0.0 or mmr > 1.0:
        print("ERROR: --engine_maintenance_margin_rate must be in [0, 1].")
        return 2
    if mmr > 0.0:
        engine_overrides["maintenance_margin_rate"] = float(mmr)

    lfr = float(getattr(args, "engine_liquidation_fee_rate", -1.0) or -1.0)
    if lfr < -1.0 or lfr > 1.0:
        print("ERROR: --engine_liquidation_fee_rate must be in [-1, 1].")
        return 2
    if lfr >= 0.0:
        engine_overrides["liquidation_fee_rate"] = float(lfr)

    mn = float(getattr(args, "engine_min_notional", -1.0) or -1.0)
    if mn < -1.0:
        print("ERROR: --engine_min_notional must be >= -1.")
        return 2
    if mn >= 0.0:
        engine_overrides["min_notional_usdt"] = float(mn)

    mq = float(getattr(args, "engine_min_qty", -1.0) or -1.0)
    if mq < -1.0:
        print("ERROR: --engine_min_qty must be >= -1.")
        return 2
    if mq >= 0.0:
        engine_overrides["min_qty"] = float(mq)

    qs = float(getattr(args, "engine_qty_step", -1.0) or -1.0)
    if qs < -1.0:
        print("ERROR: --engine_qty_step must be >= -1.")
        return 2
    if qs >= 0.0:
        engine_overrides["qty_step"] = float(qs)
    # Funding config (for reporting / defaults).
    # Backtest_Engine uses engine_overrides above (constant rate or historical Binance funding).
    funding_mode = str(getattr(args, "funding", "none") or "none").strip().lower()
    if funding_mode == "const":
        funding_mode = "constant"
    funding_period_ms = int(getattr(args, "funding_period_ms", 28_800_000) or 28_800_000)
    if funding_mode == "none":
        funding_bps = 0.0
    elif funding_mode == "constant":
        funding_bps = float(getattr(args, "funding_bps", 0.0) or 0.0)
        if abs(funding_bps) < 1e-12:
            funding_bps = float(getattr(args, "funding_default_bps", 0.0) or 0.0)
    elif funding_mode == "binance":
        # Historical funding is handled in engine_overrides via a funding_rate_fn.
        # Keep ScoringConfig funding_bps at 0 to avoid silently using constant fallback rates.
        funding_period_ms = 28_800_000
        funding_bps = 0.0
    else:
        print(f"WARNING: Unknown funding mode '{funding_mode}', defaulting to none.")
        funding_bps = 0.0

    # Intrabar adverse fill slippage for STOP/LIQ triggers (OHLC-only)
    try:
        # NOTE: do not use `or -1.0` here; 0.0 is a valid explicit CLI value.
        _afm_raw = getattr(args, "adverse_fill_slip_mult", -1.0)
        if _afm_raw is None:
            _afm_raw = -1.0
        adverse_fill_slip_mult = float(_afm_raw)
    except Exception:
        adverse_fill_slip_mult = -1.0

    if float(adverse_fill_slip_mult) < 0.0:
        adverse_fill_slip_mult = 0.25 if bool(getattr(args, "money_mode", False)) else 0.0
        if bool(getattr(args, "money_mode", False)) and float(adverse_fill_slip_mult) > 0.0:
            print(f"MoneyMode: adverse_fill_slip_mult={float(adverse_fill_slip_mult):g} (STOP/LIQ overshoot fill)")
    adverse_fill_slip_mult = max(0.0, min(1.0, float(adverse_fill_slip_mult)))
    scoring = ScoringConfig(
        min_entries_total=min_entries_total,
        max_entries_total=20000,
        min_entries_per_fold=min_entries_per_fold,
        min_trades_total=min_trades_total,
        min_trades_per_fold=min_trades_per_fold,
        max_drawdown_limit=float(args.max_dd_limit),
        max_liquidations_total=int(getattr(args, "max_liquidations_total", 0) or 0),
        max_liquidations_per_fold=int(getattr(args, "max_liquidations_per_fold", 0) or 0),
        # Enforce consistent semantics:
        # - if no holdout slice exists (holdout_bars=0), the miner cannot gate on holdout.
        # - if holdout_bars>0, we keep the default 'required' behavior (critical anti-overfit safety).
        holdout_required=bool(holdout_bars_final > 0),
        holdout_stress_required=bool(holdout_bars_final > 0),
        holdout_latency_delay_bars=int(holdout_latency_delay_bars),
        holdout_latency_return_ratio_min=float(holdout_latency_return_ratio_min),
        holdout_adverse_fill_stress_mult=float(holdout_adverse_fill_stress_mult),
        holdout_adverse_fill_return_ratio_min=float(holdout_adverse_fill_return_ratio_min),
        holdout_max_top_profit_share=float(holdout_max_top_profit_share),
        holdout_profit_bucket_ms=int(holdout_profit_bucket_ms),
        holdout_max_top_bucket_profit_share=float(holdout_max_top_bucket_profit_share),
        holdout_max_top_vol_profit_share=float(holdout_max_top_vol_profit_share),
        holdout_segment_count=int(holdout_segment_count),
        holdout_min_pos_segment_ratio=float(holdout_min_pos_segment_ratio),
        holdout_min_segment_return=float(holdout_min_segment_return),
        holdout_perm_trials=int(holdout_perm_trials),
        holdout_perm_max_p=float(holdout_perm_max_p),
        # Loop 5: parameter robustness
        param_robust_trials=int(param_robust_trials),
        param_robust_jitter=float(param_robust_jitter),
        param_robust_min_pass_ratio=float(param_robust_min_pass_ratio),
        param_robust_return_ratio_min=float(param_robust_return_ratio_min),
        param_robust_trade_ratio_min=float(param_robust_trade_ratio_min),
        weight_return=1.0,
        weight_dd=0.7,
        weight_stability=0.5,
        weight_pf=0.1,
        weight_complexity=float(args.weight_complexity),
        weight_turnover=0.05,
        weight_worst_fold=0.25,

        fee_bps=float(args.fee_bps),
        spread_bps=float(args.spread_bps),
        slippage_bps=float(args.slippage_bps),
        adverse_fill_slip_mult=float(adverse_fill_slip_mult),
        funding_bps=float(funding_bps),
        funding_period_ms=int(funding_period_ms),
        stress_cost_mult=float(args.stress_cost_mult),
        stress_min_mean_return=float(args.stress_min_mean_return),
        stress_min_worst_fold_return=float(args.stress_min_worst_fold_return),
        min_mean_return=float(args.min_mean_return),
        min_worst_fold_return=float(args.min_worst_fold_return),
        min_profit_factor=float(args.min_profit_factor),
        max_turnover_per_1000=float(args.max_turnover_per_1000),
        max_top_profit_share=float(max_top_profit_share),
        # Loop 2: selection-bias guards
        return_mode=str(getattr(args, "return_mode", "absolute")),
        return_trade_damp_ref=int(getattr(args, "return_trade_damp_ref", 30) or 30),
        min_pos_fold_ratio=float(min_pos_fold_ratio),
        min_t_stat=float(getattr(args, "min_t_stat", 0.0) or 0.0),
        max_p_value_adj=float(getattr(args, "max_p_value_adj", 0.0) or 0.0),
        multiple_testing_trials=int(getattr(args, "multiple_testing_trials", 0) or 0),
        cscv_min_selected_fraction=float(cscv_min_selected_fraction),
        cscv_min_oos_rank_median=float(cscv_min_oos_rank_median),
    )
    # ---- Search ----
    # NOTE: built late so engine_overrides includes funding + engine_* knobs.
    search = SearchConfig(
        mode=str(args.mode),
        max_evals=int(args.max_evals),
        top_k=int(args.top_k),
        seed=int(args.seed),
        sample_prob=float(args.sample_prob),
        backtest_mode=str(args.backtest),
        max_expr_cache=4000,
        max_regime_cache=1200,
        diversify_by_entry=(not bool(args.no_diversify_by_entry)),
        max_variants_per_entry=max(1, int(args.max_variants_per_entry)),
        diversify_by_family=(not bool(args.no_diversify_by_family)),
        max_per_family=max(0, int(args.max_per_family)),
        entry_key_mode=str(args.entry_key_mode),
        engine_cfg_overrides=dict(engine_overrides),
    )

    print(
        f"Search: mode={search.mode} max_evals={search.max_evals} top_k={search.top_k} seed={search.seed} backtest={search.backtest_mode} "
        f"entry_key={getattr(search,'entry_key_mode','entry_exit_time_regime')} variants/entry={getattr(search,'max_variants_per_entry',1)} "
        f"div_entry={bool(getattr(search,'diversify_by_entry',True))} div_family={bool(getattr(search,'diversify_by_family',True))} max_per_family={int(getattr(search,'max_per_family',0))}\n"
        f"Scoring(min): entries_total>={min_entries_total}, entries_per_fold>={min_entries_per_fold}, "
        f"trades_total>={min_trades_total}, trades_per_fold>={min_trades_per_fold}, max_dd<={scoring.max_drawdown_limit}, "
        f"max_liqs_total<={int(getattr(scoring,'max_liquidations_total',0) or 0)}, max_liqs_per_fold<={int(getattr(scoring,'max_liquidations_per_fold',0) or 0)}, "
        f"pf_transform={getattr(scoring,'pf_transform','log')}\n"
        f"Costs: fee_bps={scoring.fee_bps}, spread_bps={scoring.spread_bps}, slippage_bps={scoring.slippage_bps}"
    )
    if float(getattr(scoring, "max_top_profit_share", 1.0) or 1.0) < 1.0:
        print(
            f"OOS(profit conc): max_top_profit_share<={float(getattr(scoring,'max_top_profit_share',1.0) or 1.0):g}"
        )
    if float(getattr(scoring, "stress_cost_mult", 1.0) or 1.0) > 1.0:
        print(
            f"Stress(cost x{float(scoring.stress_cost_mult):g}): min_mean_return>={float(scoring.stress_min_mean_return):g}, "
            f"min_worst_fold_return>={float(scoring.stress_min_worst_fold_return):g}, "
            f"mean_ret_ratio_min={float(getattr(scoring,'stress_mean_return_ratio_min',0.0) or 0.0):g}, "
            f"worst_ret_ratio_min={float(getattr(scoring,'stress_worst_fold_return_ratio_min',0.0) or 0.0):g}" 
        )
    else:
        print("Stress: disabled (stress_cost_mult<=1)")
    # Holdout gating summary (only relevant when holdout_bars>0)
    if int(getattr(wf, "holdout_bars", 0) or 0) > 0:
        print(
            f"Holdout(gate): required={bool(getattr(scoring,'holdout_required',False))}, "
            f"min_trades>={int(getattr(scoring,'holdout_min_trades',0))}, "
            f"trade_ratio_min={float(getattr(scoring,'holdout_trade_ratio_min',0.0))}, "
            f"min_return>={float(getattr(scoring,'holdout_min_return',0.0))}, "
            f"min_pf>={float(getattr(scoring,'holdout_min_profit_factor',0.0))}, "
            f"max_dd<={float(getattr(scoring,'holdout_max_drawdown_limit',1.0))}"
        )
        if float(getattr(scoring, "holdout_max_top_profit_share", 1.0) or 1.0) < 1.0:
            print(
                f"Holdout(profit conc): max_top_profit_share<={float(getattr(scoring,'holdout_max_top_profit_share',1.0) or 1.0):g}"
            )
        if float(getattr(scoring, "holdout_max_top_bucket_profit_share", 1.0) or 1.0) < 1.0:
            print(
                f"Holdout(time concentration): max_top_bucket_profit_share={float(getattr(scoring,'holdout_max_top_bucket_profit_share',1.0) or 1.0):g} "
                f"bucket_ms={int(getattr(scoring,'holdout_profit_bucket_ms',28800000) or 28800000)}"
            )
        if int(getattr(scoring, "holdout_segment_count", 0) or 0) > 1 and (
            float(getattr(scoring, "holdout_min_pos_segment_ratio", 0.0) or 0.0) > 0.0
            or float(getattr(scoring, "holdout_min_segment_return", -1.0) or -1.0) > -1.0
        ):
            print(
                f"Holdout(segment stability): segments={int(getattr(scoring,'holdout_segment_count',0) or 0)}, "
                f"min_pos_segment_ratio={float(getattr(scoring,'holdout_min_pos_segment_ratio',0.0) or 0.0):g}, "
                f"min_segment_return={float(getattr(scoring,'holdout_min_segment_return',-1.0) or -1.0):g}"
            )
        if float(getattr(scoring, "stress_cost_mult", 1.0) or 1.0) > 1.0 and bool(getattr(scoring, "holdout_stress_required", True)):
            print(
                f"Holdout(stress gate): min_return>={float(getattr(scoring,'holdout_stress_min_return',0.0))}, "
                f"min_pf>={float(getattr(scoring,'holdout_stress_min_profit_factor',0.0))}, "
                f"max_dd<={float(getattr(scoring,'holdout_stress_max_drawdown_limit',0.0)) or float(getattr(scoring,'holdout_max_drawdown_limit',1.0))}, "
                f"ret_ratio_min={float(getattr(scoring,'holdout_stress_return_ratio_min',0.0) or 0.0):g}"
            )
        # Holdout latency gate summary (only when enabled)
        if int(getattr(scoring, "holdout_latency_delay_bars", 0) or 0) > 0:
            print(
                f"Holdout(latency gate): delay_bars={int(getattr(scoring,'holdout_latency_delay_bars',0) or 0)}, "
                f"ret_ratio_min={float(getattr(scoring,'holdout_latency_return_ratio_min',0.0) or 0.0):g}"
            )
        # Holdout adverse-fill stress summary (only when enabled)
        if float(getattr(scoring, "holdout_adverse_fill_stress_mult", 0.0) or 0.0) > max(
            0.0, float(getattr(scoring, "adverse_fill_slip_mult", 0.0) or 0.0)
        ):
            print(
                f"Holdout(adverse fill stress): mult={float(getattr(scoring,'holdout_adverse_fill_stress_mult',0.0) or 0.0):g}, "
                f"ret_ratio_min={float(getattr(scoring,'holdout_adverse_fill_return_ratio_min',0.0) or 0.0):g}"
            )
        # Loop 5: parameter robustness summary (only when enabled)
        if int(getattr(scoring, "param_robust_trials", 0) or 0) > 0:
            print(
                f"Holdout(param robust): trials={int(getattr(scoring,'param_robust_trials',0))}, "
                f"jitter={float(getattr(scoring,'param_robust_jitter',0.0) or 0.0):g}, "
                f"min_pass_ratio={float(getattr(scoring,'param_robust_min_pass_ratio',0.0) or 0.0):g}, "
                f"ret_ratio_min={float(getattr(scoring,'param_robust_return_ratio_min',0.0) or 0.0):g}, "
                f"trade_ratio_min={float(getattr(scoring,'param_robust_trade_ratio_min',0.0) or 0.0):g}"
            )
    # ---- Run miner ----
    miner = StrategyMiner(series, space, wf, search, scoring)
    report = miner.run()

    print("\nRun summary:")
    print(f"Evaluated: {report.evaluated} | Accepted: {report.accepted} | Rejected: {report.rejected} | Folds: {report.folds}")
    # Backtest backend transparency
    try:
        cs = getattr(report, "candidate_stats", None) or {}
        backend = cs.get("backtest_backend", "?")
        counts = cs.get("backtest_backend_counts", {}) or {}
        counts_stress = cs.get("backtest_backend_counts_stress", {}) or {}
        info = cs.get("backtest_backend_info", {}) or {}
        note = ""
        if isinstance(info, dict):
            note = str(info.get("note", "") or "")
        print(f"Backtest backend: {backend} counts={counts} stress_counts={counts_stress}")
        if note:
            print(f"Backend note: {note}")
        # CSCV/PBO selection-bias diagnostic (Loop 14)
        cscv = cs.get("cscv", None)
        if isinstance(cscv, dict) and bool(cscv.get("enabled", False)):
            try:
                pool = int(cscv.get("pool_size", 0) or 0)
                folds = int(cscv.get("folds", 0) or 0)
                is_folds = int(cscv.get("is_folds", 0) or 0)
                combos = int(cscv.get("combinations_used", 0) or 0)
                pbo = float(cscv.get("pbo", 1.0) or 1.0)
                print(f"CSCV/PBO: pool={pool} folds={folds} IS={is_folds} combos={combos} PBO={pbo:.2f}")
            except Exception:
                pass
        # Optional CSCV stability gate (Loop 15)
        cscv_gate = cs.get("cscv_gate", None)
        if isinstance(cscv_gate, dict) and cscv_gate:
            try:
                enabled = bool(cscv_gate.get("enabled", False))
                applied = bool(cscv_gate.get("applied", False))
                sel = float(cscv_gate.get("min_selected_fraction", 0.0) or 0.0)
                oos = float(cscv_gate.get("min_oos_rank_median", 0.0) or 0.0)
                if enabled and applied:
                    pool0 = int(cscv_gate.get("pool_size", 0) or 0)
                    kept = int(cscv_gate.get("kept", 0) or 0)
                    dropped = int(cscv_gate.get("dropped", 0) or 0)
                    print(f"CSCV gate: sel>={sel:g} oos_med>={oos:g} kept={kept}/{pool0} dropped={dropped}")
                elif enabled:
                    reason = str(cscv_gate.get("reason", "") or "")
                    if reason:
                        print(f"CSCV gate: enabled but not applied ({reason}) sel>={sel:g} oos_med>={oos:g}")
                    else:
                        print(f"CSCV gate: enabled but not applied sel>={sel:g} oos_med>={oos:g}")
            except Exception:
                pass
    except Exception:
        pass

    # Optional debug: show the most common rejection reasons (helps tuning scoring filters).
    if getattr(report, "reject_reasons", None):
        items = sorted(report.reject_reasons.items(), key=lambda kv: kv[1], reverse=True)[:8]
        if items:
            print("Top reject reasons:", ", ".join(f"{k}={v}" for k, v in items))
    # Print top results early so you see winners even if JSON export fails.
    if int(getattr(args, "print_top", 0) or 0) > 0:
        _print_top(report, top_n=int(args.print_top))
    # ---- Cross-market validation (Loop 13) ----
    validate_sets: List[Tuple[str, str]] = []
    try:
        validate_sets = _parse_validate_datasets(getattr(args, 'validate', []) or [])
    except Exception as e:
        print('ERROR: invalid --validate specification.')
        print(f'{type(e).__name__}: {e}')
        return 2

    validated_report = None
    validation_overview: Dict[str, Any] = {}

    if validate_sets:
        if str(getattr(args, 'csv', '')).strip():
            print('\nWARNING: --validate is only supported when using Binance data in this runner (no --offline/--csv).')
        elif report.accepted <= 0 or not report.top_results:
            print('\nINFO: --validate specified but there are no accepted strategies to validate.')
        else:
            print('\nValidation (Loop 13): cross-market / cross-timeframe replay')
            labels = [f'{sym}:{tf}' for sym, tf in validate_sets]
            print('Validation datasets:', ', '.join(labels))

            val_series: Dict[str, OhlcvSeries] = {}
            for sym, tf in validate_sets:
                label = f'{sym}:{tf}'
                try:
                    s_val = _load_series_binance_best_effort(sym, tf, limit)
                    s_val = _sanitize_series_for_backtest(s_val, timeframe=tf, source_label=f'BINANCE_VALIDATE:{label}')
                except Exception as e:
                    print(f'ERROR: failed to load validation dataset {label}.')
                    print(f'{type(e).__name__}: {e}')
                    return 2
                val_series[label] = s_val

            top_n = int(getattr(args, 'validate_top', 10) or 10)
            top_n = max(1, min(top_n, len(report.top_results)))
            min_ratio = float(getattr(args, 'validate_min_pass_ratio', 1.0) or 1.0)
            if not math.isfinite(min_ratio):
                min_ratio = 1.0
            min_ratio = max(0.0, min(1.0, min_ratio))

            n_ds = len(val_series)
            required_pass = int(math.ceil(min_ratio * n_ds)) if n_ds > 0 else 0
            if min_ratio > 0.0:
                required_pass = max(1, required_pass)
            required_pass = min(required_pass, n_ds) if n_ds > 0 else 0

            print(f'Validating top {top_n} strategies | require pass >= {required_pass}/{n_ds} datasets (min_pass_ratio={min_ratio:g})')

            validated: List[Any] = []
            annotated: List[Any] = []

            for r in list(report.top_results)[:top_n]:
                spec_obj = getattr(r, 'spec_obj', None)
                sid = str(getattr(r, 'strategy_id', '?') or '?')
                if spec_obj is None:
                    per_ds = {'_internal': {'passed': False, 'error': 'missing spec_obj'}}
                    val_block = {
                        'passed': False,
                        'pass_count': 0,
                        'pass_ratio': 0.0,
                        'required_pass': int(required_pass),
                        'n_datasets': int(n_ds),
                        'datasets': per_ds,
                    }
                    agg2 = dict(getattr(r, 'aggregate', None) or {})
                    agg2['validation'] = val_block
                    r2 = replace(r, aggregate=agg2)
                    annotated.append(r2)
                    print(f'  - {sid} pass 0/{n_ds}: FAIL (missing spec_obj)')
                    continue

                per_ds: Dict[str, Any] = {}
                pass_count = 0
                for label, s_val in val_series.items():
                    v = _validate_spec_on_series(spec_obj=spec_obj, series=s_val, wf=wf, base_search=search, scoring=scoring)
                    per_ds[label] = v
                    if bool(v.get('passed', False)):
                        pass_count += 1

                pass_ratio = (pass_count / n_ds) if n_ds > 0 else 0.0
                passed = bool(pass_count >= required_pass) if n_ds > 0 else True

                val_block = {
                    'passed': passed,
                    'pass_count': int(pass_count),
                    'pass_ratio': float(pass_ratio),
                    'required_pass': int(required_pass),
                    'n_datasets': int(n_ds),
                    'datasets': per_ds,
                }
                agg2 = dict(getattr(r, 'aggregate', None) or {})
                agg2['validation'] = val_block
                r2 = replace(r, aggregate=agg2)
                annotated.append(r2)
                if passed:
                    validated.append(r2)

                print(f"  - {sid} pass {pass_count}/{n_ds}: {'OK' if passed else 'FAIL'}")

            validation_overview = {
                'datasets': list(val_series.keys()),
                'validated_top': int(top_n),
                'min_pass_ratio': float(min_ratio),
                'required_pass': int(required_pass),
                'passed_strategies': int(len(validated)),
            }

            # Attach validation summaries to the main report export (top_n only)
            try:
                report = replace(report, top_results=tuple(annotated) + tuple(report.top_results[top_n:]))
            except Exception:
                pass

            if validated:
                try:
                    validated_report = replace(report, accepted=len(validated), top_results=tuple(validated))
                except Exception:
                    validated_report = None

            print(f'Validation passed: {len(validated)}/{top_n} strategies')
    # ---- Export ----
    out = str(args.out).strip()
    if not out:
        if replay_path:
            stem = os.path.splitext(os.path.basename(replay_path))[0]
            out = f"ReplayResults_{stem}_{symbol}_{timeframe}_bars{n}.json"
        else:
            out = f"MinerResults_{symbol}_{timeframe}_bars{n}_space{args.space}.json"

    try:
        meta = {
            "data_source": data_source,
            "symbol": getattr(series, "symbol", None),
            "timeframe": getattr(series, "timeframe", None),
            "bars": int(len(series)) if hasattr(series, "__len__") else None,
            "ts_start_ms": int(series.ts_ms[0]) if getattr(series, "ts_ms", None) else None,
            "ts_end_ms": int(series.ts_ms[-1]) if getattr(series, "ts_ms", None) else None,
            "series_sha1": _series_sha1(series),
        }
        if validation_overview:
            meta["validation"] = validation_overview
        save_results_json(out, report, meta=meta)
        print(f"\nSaved results: {out}")
    except Exception as e:
        print("WARNING: Could not save JSON results.")
        print(f"{type(e).__name__}: {e}")
    # Save validated results (if any)
    if validated_report is not None:
        out_v = str(getattr(args, 'validate_out', '')).strip()
        if not out_v:
            out_v = (out[:-5] + '_validated.json') if out.lower().endswith('.json') else (out + '_validated.json')
        try:
            meta_v = {
                'data_source': data_source,
                'symbol': getattr(series, 'symbol', None),
                'timeframe': getattr(series, 'timeframe', None),
                'bars': int(len(series)) if hasattr(series, '__len__') else None,
                'ts_start_ms': int(series.ts_ms[0]) if getattr(series, 'ts_ms', None) else None,
                'ts_end_ms': int(series.ts_ms[-1]) if getattr(series, 'ts_ms', None) else None,
                'series_sha1': _series_sha1(series),
                'validation': validation_overview,
            }
            save_results_json(out_v, validated_report, meta=meta_v)
            print(f"Saved validated results: {out_v}")
        except Exception as e:
            print('WARNING: Could not save validated JSON results.')
            print(f"{type(e).__name__}: {e}")
    # If still nothing accepted, give immediate actionable hint
    if report.accepted == 0:
        print("\nHINT: 0 strategies accepted.")

        if any(str(k).startswith("backtest_fail") for k in report.reject_reasons.keys()):
            print("NOTE: Many candidates were rejected due to backtest exceptions (backtest_fail).")
            print("      Fix backtest_fail first; otherwise no strategy can be accepted.")
            print("      After Loop 3, the reject reason will include the exception type (e.g. backtest_fail:ValueError).")

        print("Try one of these:")
        print("  - increase --max_evals (e.g. 2000)")
        print("  - use a larger space preset: --space medium or --space large")
        print("  - relax minimums: --min_trades_total 0 --min_entries_total 0 (or keep AUTO but lower)")
        print("  - for small datasets: reduce WF test size: --train_bars 400 --test_bars 200 (example)")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
