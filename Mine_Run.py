from __future__ import annotations

"""
Mine_Run.py
===========

Runner:
- loads OHLCV (CSV > Binance(Data_Binance) > offline synthetic fallback)
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
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, ValidationError, require
from Strategy_Space import StrategySpaceConfig, ComplexityBudget, StrategySpace
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
        print("No top results.")
        return

    print("\nTop strategies:")
    for i, r in enumerate(top, 1):
        agg = r.aggregate or {}
        mr = float(agg.get("mean_return", 0.0) or 0.0)
        mdd = float(agg.get("mean_drawdown", 0.0) or 0.0)
        pf = float(agg.get("mean_profit_factor", 0.0) or 0.0)
        trades = agg.get("fold_trades", [])
        trades_str = ",".join(str(x) for x in trades) if isinstance(trades, list) else str(trades)
        rf = _short_regime_filter_str(r.regime_filter)

        print(
            f"{i:>2}. score={r.score:> .6f} id={r.strategy_id} dir={r.direction:<5} "
            f"ret={mr:> .4f} dd={mdd:> .4f} pf={pf:> .2f} trades=[{trades_str}] "
            f"cx={r.complexity:<2} tags={list(r.tags)}"
            + (f" rf={rf}" if rf else "")
        )


# -----------------------------
# CLI
# -----------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strategy miner on Binance or offline synthetic data.")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1m")
    p.add_argument("--limit", type=int, default=1500, help="Requested bars (will be chunked if > max per request)")

    p.add_argument("--offline", action="store_true")
    p.add_argument("--require_binance", action="store_true")
    p.add_argument("--csv", type=str, default="")

    p.add_argument("--space", type=str, default="small", choices=["small", "medium", "large"])
    p.add_argument("--mode", type=str, default="iterate", choices=["iterate", "sample"])
    p.add_argument("--max_evals", type=int, default=300)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--sample_prob", type=float, default=0.15)

    p.add_argument("--backtest", type=str, default="auto", choices=["auto", "engine", "simple"])
    p.add_argument("--out", type=str, default="")

    # Walk-forward overrides
    p.add_argument("--train_bars", type=int, default=0)
    p.add_argument("--test_bars", type=int, default=0)
    p.add_argument("--step_bars", type=int, default=0)
    p.add_argument("--purge_bars", type=int, default=0)
    p.add_argument("--embargo_bars", type=int, default=0)
    p.add_argument("--anchored", action="store_true")

    # Scoring tweaks (AUTO defaults = -1)
    p.add_argument("--fee_bps", type=float, default=4.0)
    p.add_argument("--weight_complexity", type=float, default=0.002)
    p.add_argument("--min_entries_total", type=int, default=-1, help="-1 auto")
    p.add_argument("--min_entries_per_fold", type=int, default=-1, help="-1 auto")
    p.add_argument("--min_trades_total", type=int, default=-1, help="-1 auto")
    p.add_argument("--min_trades_per_fold", type=int, default=-1, help="-1 auto")
    p.add_argument("--max_dd_limit", type=float, default=0.50)

    p.add_argument("--print_top", type=int, default=10)

    return p.parse_args(list(argv) if argv is not None else None)


# -----------------------------
# Main
# -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    symbol = str(args.symbol).strip().upper()
    timeframe = str(args.timeframe).strip()
    limit = int(args.limit)

    # ---- Load series ----
    series: Optional[OhlcvSeries] = None
    data_source = ""

    try:
        if str(args.csv).strip():
            series = _load_series_csv(str(args.csv).strip(), symbol=symbol, timeframe=timeframe)
            data_source = f"CSV:{args.csv}"
        elif bool(args.offline):
            # offline length: if user asked big limit, allow up to 5000 for better folds
            n_off = max(600, min(5000, max(600, limit)))
            series = _make_synthetic_series(n_off, symbol=symbol, timeframe=timeframe)
            data_source = "OFFLINE_SYNTHETIC"
        else:
            try:
                series = _load_series_binance_best_effort(symbol, timeframe, limit)
                data_source = "BINANCE(Data_Binance)"
            except Exception as e:
                if bool(args.require_binance):
                    raise
                print("WARNING: Binance load failed, falling back to offline synthetic.")
                print(f"Reason: {type(e).__name__}: {e}")
                series = _make_synthetic_series(800, symbol=symbol, timeframe=timeframe)
                data_source = "OFFLINE_SYNTHETIC(FALLBACK)"
    except Exception as e:
        print("ERROR: Failed to load series.")
        print(f"{type(e).__name__}: {e}")
        return 2

    require(series is not None, "internal: series is None after load")
    n = len(series.ts_ms)
    require(n >= 200, f"Need at least 200 bars, got {n}")

    print(f"Data source: {data_source}")
    print(f"Series: symbol={series.symbol} timeframe={series.timeframe} bars={n}")

    # ---- Space preset ----
    cfg, budget = _space_preset(str(args.space))
    space = StrategySpace(cfg, budget)

    # ---- Walk-forward ----
    if int(args.train_bars) > 0 and int(args.test_bars) > 0:
        wf = WalkForwardConfig(
            train_bars=int(args.train_bars),
            test_bars=int(args.test_bars),
            step_bars=int(args.step_bars) if int(args.step_bars) > 0 else int(args.test_bars),
            purge_bars=int(args.purge_bars),
            embargo_bars=int(args.embargo_bars),
            anchored=bool(args.anchored),
        )
    else:
        wf = _auto_walkforward(n)
        if int(args.purge_bars) != 0 or int(args.embargo_bars) != 0 or bool(args.anchored):
            wf = WalkForwardConfig(
                train_bars=int(wf.train_bars),
                test_bars=int(wf.test_bars),
                step_bars=int(wf.step_bars),
                purge_bars=int(args.purge_bars),
                embargo_bars=int(args.embargo_bars),
                anchored=bool(args.anchored),
            )

    splits = make_walkforward_splits(n, wf)
    folds = len(splits)
    print(
        f"WalkForward: folds={folds} train={wf.train_bars} test={wf.test_bars} step={wf.step_bars} "
        f"purge={wf.purge_bars} embargo={wf.embargo_bars} anchored={wf.anchored}"
    )

    # ---- Search ----
    search = SearchConfig(
        mode=str(args.mode),
        max_evals=int(args.max_evals),
        top_k=int(args.top_k),
        seed=int(args.seed),
        sample_prob=float(args.sample_prob),
        backtest_mode=str(args.backtest),
        max_expr_cache=4000,
        max_regime_cache=1200,
    )

    # ---- Scoring AUTO defaults ----
    min_entries_total = int(args.min_entries_total) if int(args.min_entries_total) >= 0 else max(2, folds)
    min_entries_per_fold = int(args.min_entries_per_fold) if int(args.min_entries_per_fold) >= 0 else 0
    min_trades_total = int(args.min_trades_total) if int(args.min_trades_total) >= 0 else max(2, folds)
    min_trades_per_fold = int(args.min_trades_per_fold) if int(args.min_trades_per_fold) >= 0 else 0

    scoring = ScoringConfig(
        min_entries_total=min_entries_total,
        max_entries_total=20000,
        min_entries_per_fold=min_entries_per_fold,
        min_trades_total=min_trades_total,
        min_trades_per_fold=min_trades_per_fold,
        max_drawdown_limit=float(args.max_dd_limit),

        weight_return=1.0,
        weight_dd=0.7,
        weight_stability=0.5,
        weight_pf=0.1,
        weight_complexity=float(args.weight_complexity),
        weight_turnover=0.05,
        weight_worst_fold=0.25,

        fee_bps=float(args.fee_bps),
    )

    print(
        f"Search: mode={search.mode} max_evals={search.max_evals} top_k={search.top_k} seed={search.seed} backtest={search.backtest_mode}\n"
        f"Scoring(min): entries_total>={min_entries_total}, entries_per_fold>={min_entries_per_fold}, "
        f"trades_total>={min_trades_total}, trades_per_fold>={min_trades_per_fold}, max_dd<={scoring.max_drawdown_limit}"
    )

    # ---- Run miner ----
    miner = StrategyMiner(series, space, wf, search, scoring)
    report = miner.run()

    print("\nRun summary:")
    print(f"Evaluated: {report.evaluated} | Accepted: {report.accepted} | Rejected: {report.rejected} | Folds: {report.folds}")

    _print_top(report, top_n=int(args.print_top))

    # ---- Export ----
    out = str(args.out).strip()
    if not out:
        out = f"MinerResults_{symbol}_{timeframe}_bars{n}_space{args.space}.json"

    try:
        save_results_json(out, report)
        print(f"\nSaved results: {out}")
    except Exception as e:
        print("WARNING: Could not save JSON results.")
        print(f"{type(e).__name__}: {e}")

    # If still nothing accepted, give immediate actionable hint
    if report.accepted == 0:
        print("\nHINT: 0 strategies accepted.")
        print("Try one of these:")
        print("  - increase --max_evals (e.g. 2000)")
        print("  - use a larger space preset: --space medium or --space large")
        print("  - relax minimums: --min_trades_total 0 --min_entries_total 0 (or keep AUTO but lower)")
        print("  - for small datasets: reduce WF test size: --train_bars 400 --test_bars 200 (example)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
