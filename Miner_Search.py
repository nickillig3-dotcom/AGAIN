from __future__ import annotations

"""
Miner_Search.py
===============

Purpose
-------
Strategy mining / search runner built on top of:
- Strategy_Space (StrategySpec generator)
- Signals_Events (conditions/sequences evaluation)
- Regime_Classifier (regime masks)
- Backtest_Engine (optional; auto adapter)
with priorities:
- realism, efficiency, robustness
- no lookahead
- pruning to keep large spaces mineable

Major improvement (Loop 12)
---------------------------
The StrategySpace generator yields strategy variants in a nested order:
  entry_template -> exit -> stop -> tp -> time_stop -> cooldown -> regime

This can cause the miner to spend its entire evaluation budget on the first (huge)
entry family (e.g., sweep-reversal sequences), never reaching other families.

To solve this, we add *diversified candidate selection*:
- cap variants per unique entry signature
- cap evaluations per entry family bucket (trend / mean-reversion / breakout / reversal)

Also adds detailed rejection telemetry to debug "0 accepted".

Self-test:
----------
python Miner_Search.py
"""

from dataclasses import dataclass, asdict
import hashlib
import heapq
import inspect
import json
import math
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, ValidationError, require
from Signals_Events import SignalHub
from Regime_Classifier import RegimeStore
from Strategy_Space import (
    StrategySpace,
    StrategySpaceConfig,
    ComplexityBudget,
    StrategySpec,
    StopSpec,
    TakeProfitSpec,
    RegimeFilterSpec,
    RegimeConditionSpec,
)


# -----------------------------
# Helpers
# -----------------------------
_NAN = float("nan")


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((float(x) - mu) ** 2 for x in xs) / float(len(xs) - 1)
    return math.sqrt(var)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0.0:
        return default
    return float(a) / float(b)


def _to_jsonable(obj: Any) -> Any:
    """
    Turn arbitrary nested dataclasses/tuples into JSON-serializable structures.
    """
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _to_jsonable(asdict(obj))
    return str(obj)


# -----------------------------
# Configs
# -----------------------------
@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """
    Rolling walk-forward with optional purge/embargo.

    train_bars: length of train window (used only for split placement)
    test_bars:  length of test window (evaluation window)
    step_bars:  step between folds
    purge_bars: remove last purge_bars from train AND first purge_bars from test
    embargo_bars: gap between train_end and test_start (extra separation)
    anchored: if True, train_start stays 0 and train_end expands; if False, rolling window
    """
    train_bars: int = 1000
    test_bars: int = 500
    step_bars: int = 500
    purge_bars: int = 0
    embargo_bars: int = 0
    anchored: bool = False

    def __post_init__(self) -> None:
        require(int(self.train_bars) > 0, "train_bars must be > 0")
        require(int(self.test_bars) > 0, "test_bars must be > 0")
        require(int(self.step_bars) > 0, "step_bars must be > 0")
        require(int(self.purge_bars) >= 0, "purge_bars must be >= 0")
        require(int(self.embargo_bars) >= 0, "embargo_bars must be >= 0")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """
    Search procedure settings.

    Diversification controls:
    - diversify_by_entry: cap stop/tp/exit/time variants per unique entry signature
    - max_variants_per_entry: how many variants per unique entry are allowed
    - diversify_by_family: cap evaluations per entry-family bucket (direction-aware)
    - max_per_family: cap per bucket; 0 => auto distribution based on max_evals
    """
    mode: str = "iterate"  # "iterate" or "sample"
    max_evals: int = 1000
    top_k: int = 50
    seed: int = 123
    sample_prob: float = 0.1  # only used in mode="sample"
    backtest_mode: str = "auto"  # "auto" | "engine" | "simple"

    # Caches
    max_expr_cache: int = 4000
    max_regime_cache: int = 1000

    # Diversification (Loop 12)
    diversify_by_entry: bool = True
    max_variants_per_entry: int = 1
    diversify_by_family: bool = True
    max_per_family: int = 0  # 0 => auto

    def __post_init__(self) -> None:
        m = str(self.mode).strip().lower()
        require(m in ("iterate", "sample"), f"mode invalid: {self.mode!r}")
        object.__setattr__(self, "mode", m)

        bt = str(self.backtest_mode).strip().lower()
        require(bt in ("auto", "engine", "simple"), f"backtest_mode invalid: {self.backtest_mode!r}")
        object.__setattr__(self, "backtest_mode", bt)

        require(int(self.max_evals) > 0, "max_evals must be > 0")
        require(int(self.top_k) > 0, "top_k must be > 0")
        require(float(self.sample_prob) > 0.0 and float(self.sample_prob) <= 1.0, "sample_prob must be in (0,1]")

        require(int(self.max_expr_cache) >= 0, "max_expr_cache must be >= 0")
        require(int(self.max_regime_cache) >= 0, "max_regime_cache must be >= 0")

        require(int(self.max_variants_per_entry) >= 1, "max_variants_per_entry must be >= 1")
        require(int(self.max_per_family) >= 0, "max_per_family must be >= 0 (0=auto)")


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """
    Robustness-first constraints & scoring weights.
    All values in "fraction" units where applicable (e.g. 0.2 = 20%).
    """
    # Early filters (cheap, pre-backtest)
    min_entries_total: int = 3
    max_entries_total: int = 20000
    min_entries_per_fold: int = 1

    # Post-backtest constraints
    min_trades_total: int = 3
    min_trades_per_fold: int = 1
    max_drawdown_limit: float = 0.50  # reject if any fold DD exceeds this

    # Score weights (test fold aggregates)
    weight_return: float = 1.0
    weight_dd: float = 0.7
    weight_stability: float = 0.5
    weight_pf: float = 0.1
    weight_complexity: float = 0.01
    weight_turnover: float = 0.05
    weight_worst_fold: float = 0.25

    # Execution assumptions for simple backtester
    fee_bps: float = 4.0  # 0.04% per trade side (simple model uses 2 sides)

    def __post_init__(self) -> None:
        require(int(self.min_entries_total) >= 0, "min_entries_total must be >= 0")
        require(int(self.max_entries_total) > 0, "max_entries_total must be > 0")
        require(int(self.min_entries_per_fold) >= 0, "min_entries_per_fold must be >= 0")
        require(int(self.min_trades_total) >= 0, "min_trades_total must be >= 0")
        require(int(self.min_trades_per_fold) >= 0, "min_trades_per_fold must be >= 0")
        require(float(self.max_drawdown_limit) > 0.0 and float(self.max_drawdown_limit) <= 1.0, "max_drawdown_limit must be in (0,1]")
        require(float(self.fee_bps) >= 0.0, "fee_bps must be >= 0")


# -----------------------------
# Splits
# -----------------------------
@dataclass(frozen=True, slots=True)
class FoldSplit:
    train: slice
    test: slice

    def __post_init__(self) -> None:
        require(self.train.start is not None and self.train.stop is not None, "train slice must be bounded")
        require(self.test.start is not None and self.test.stop is not None, "test slice must be bounded")
        require(int(self.train.start) >= 0 and int(self.train.stop) >= 0, "train indices must be >= 0")
        require(int(self.test.start) >= 0 and int(self.test.stop) >= 0, "test indices must be >= 0")
        require(int(self.train.start) < int(self.train.stop), "train slice empty")
        require(int(self.test.start) < int(self.test.stop), "test slice empty")


def make_walkforward_splits(n_bars: int, cfg: WalkForwardConfig) -> List[FoldSplit]:
    """
    Creates walk-forward folds. Training slices are provided for completeness,
    but the miner primarily evaluates on test slices.

    Purge/embargo:
      train_eff_end = train_end - purge
      test_eff_start = train_end + embargo + purge

    anchored=False:
      train_start = start
      train_end = start + train_bars
    anchored=True:
      train_start = 0
      train_end = start + train_bars
    """
    n = int(n_bars)
    require(n > 0, "n_bars must be > 0")

    train_bars = int(cfg.train_bars)
    test_bars = int(cfg.test_bars)
    step = int(cfg.step_bars)
    purge = int(cfg.purge_bars)
    embargo = int(cfg.embargo_bars)
    anchored = bool(cfg.anchored)

    splits: List[FoldSplit] = []
    start = 0

    while True:
        train_start = 0 if anchored else start
        train_end = (start + train_bars) if anchored else (train_start + train_bars)

        train_eff_end = max(train_start, train_end - purge)
        test_start = train_end + embargo + purge
        test_end = test_start + test_bars

        if test_end > n:
            break
        if train_eff_end <= train_start:
            start += step
            continue

        splits.append(FoldSplit(train=slice(train_start, train_eff_end), test=slice(test_start, test_end)))
        start += step

    require(len(splits) > 0, "No walk-forward splits possible; increase n_bars or reduce train/test sizes.")
    return splits


# -----------------------------
# Metrics / Results
# -----------------------------
@dataclass(frozen=True, slots=True)
class FoldMetrics:
    fold_index: int
    start: int
    end: int

    trades: int
    win_rate: float
    profit_factor: float
    net_return: float          # fraction, e.g. 0.12 means +12%
    max_drawdown: float        # fraction, 0.25 means -25% peak-to-trough
    avg_trade_return: float    # average per trade, fraction
    exposure: float            # fraction of bars in position
    turnover: float            # trades per 1000 bars

    fees_paid: float           # fraction of equity (simple model)
    funding_paid: float        # fraction (simple model -> 0)


@dataclass(frozen=True, slots=True)
class CandidateResult:
    strategy_id: str
    strategy_hash: str
    strategy_name: str
    direction: str
    tags: Tuple[str, ...]
    complexity: int

    score: float
    fold_metrics: Tuple[FoldMetrics, ...]
    aggregate: Dict[str, Any]
    regime_filter: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class MinerReport:
    evaluated: int
    accepted: int
    rejected: int
    folds: int

    reject_reasons: Dict[str, int]
    candidate_stats: Dict[str, Any]

    top_results: Tuple[CandidateResult, ...]
    search_config: Dict[str, Any]
    wf_config: Dict[str, Any]
    scoring_config: Dict[str, Any]


# -----------------------------
# Compiler (Spec -> signals + regime mask)
# -----------------------------
@dataclass(slots=True)
class CompiledStrategy:
    entry: List[float]
    exit: Optional[List[float]]
    entry_mask: Optional[List[float]]  # regime mask applied to entry
    stop: StopSpec
    take_profit: TakeProfitSpec
    time_stop_bars: int
    cooldown_bars: int


class StrategyCompiler:
    """
    Compiles StrategySpec to signals (entry/exit arrays) using a shared SignalHub and RegimeStore.
    Provides caching for expression evaluation to speed up mining.

    Regime filters are applied only to ENTRY by default (exit remains unmasked),
    because you generally want to be able to exit even if regime changes.
    """

    def __init__(self, series: OhlcvSeries, *, max_expr_cache: int = 4000, max_regime_cache: int = 1000) -> None:
        self.series = series
        self.hub = SignalHub(series)
        self.reg = RegimeStore(series)
        self.n = len(series.ts_ms)

        self._expr_cache: Dict[Any, List[float]] = {}
        self._seq_cache: Dict[Any, List[float]] = {}
        self._regime_mask_cache: Dict[Any, List[float]] = {}

        self._max_expr_cache = int(max_expr_cache)
        self._max_regime_cache = int(max_regime_cache)

    # ---- Freeze helpers for stable keys ----
    def _freeze_value(self, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            if math.isnan(v):
                return ("nan",)
            if math.isinf(v):
                return ("inf", 1 if v > 0 else -1)
            return float(round(v, 12))
        if isinstance(v, str):
            return str(v)
        if isinstance(v, (list, tuple)):
            return tuple(self._freeze_value(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((str(k), self._freeze_value(x)) for k, x in v.items()))
        return str(v)

    def _key_feature_ref(self, fr: Any) -> Tuple[Any, ...]:
        return ("FR", fr.kind, fr.name, tuple(sorted((k, self._freeze_value(v)) for k, v in (fr.params or {}).items())), int(fr.shift))

    def _key_expr(self, expr: Any) -> Tuple[Any, ...]:
        from Signals_Events import Condition, AllOf, AnyOf, Not

        if isinstance(expr, Condition):
            rhs = expr.rhs
            if hasattr(rhs, "kind") and hasattr(rhs, "name") and hasattr(rhs, "params"):
                rhs_key = ("RHS_FR", self._key_feature_ref(rhs))
            elif isinstance(rhs, tuple):
                rhs_key = ("RHS_TUP", tuple(self._freeze_value(x) for x in rhs))
            elif rhs is None:
                rhs_key = ("RHS_NONE", None)
            else:
                rhs_key = ("RHS_CONST", self._freeze_value(rhs))
            return ("COND", self._key_feature_ref(expr.lhs), expr.op.value, rhs_key)

        if isinstance(expr, AllOf):
            terms = sorted((self._key_expr(t) for t in expr.terms), key=lambda x: repr(x))
            return ("ALL", tuple(terms))

        if isinstance(expr, AnyOf):
            terms = sorted((self._key_expr(t) for t in expr.terms), key=lambda x: repr(x))
            return ("ANY", tuple(terms))

        if isinstance(expr, Not):
            return ("NOT", self._key_expr(expr.term))

        raise ValidationError(f"Unsupported expr type for key: {type(expr).__name__}")

    def _key_sequence(self, seq: Any) -> Tuple[Any, ...]:
        from Signals_Events import EventSequence, SequenceStep

        require(isinstance(seq, EventSequence), "seq must be EventSequence")
        steps = []
        for st in seq.steps:
            require(isinstance(st, SequenceStep), "seq.steps must contain SequenceStep")
            steps.append(("STEP", self._key_expr(st.expr), int(st.within_bars), int(st.hold_bars)))
        return ("SEQ", tuple(steps))

    # ---- Expression evaluation with caching ----
    def eval_expr(self, expr: Any) -> List[float]:
        key = self._key_expr(expr)
        if key in self._expr_cache:
            return self._expr_cache[key]

        ev = self.hub.eval(expr)
        require(len(ev) == self.n, "expr event length mismatch")

        if self._max_expr_cache == 0:
            return ev
        if len(self._expr_cache) >= self._max_expr_cache:
            self._expr_cache.clear()
        self._expr_cache[key] = ev
        return ev

    def eval_sequence(self, seq: Any) -> List[float]:
        key = self._key_sequence(seq)
        if key in self._seq_cache:
            return self._seq_cache[key]

        ev = self.hub.eval_sequence(seq)
        require(len(ev) == self.n, "sequence event length mismatch")

        if self._max_expr_cache == 0:
            return ev
        if len(self._seq_cache) >= self._max_expr_cache:
            self._seq_cache.clear()
        self._seq_cache[key] = ev
        return ev

    # ---- Regime mask evaluation ----
    def _key_regime_cond(self, c: RegimeConditionSpec) -> Tuple[Any, ...]:
        return (
            c.regime_name,
            c.op,
            float(round(float(c.value), 12)),
            tuple(sorted((str(k), self._freeze_value(v)) for k, v in (c.params or {}).items())),
        )

    def _regime_cond_to_mask(self, c: RegimeConditionSpec) -> List[float]:
        labels = self.reg.regime(c.regime_name, **(c.params or {}))
        n = self.n
        out = [0.0] * n
        v = float(c.value)
        op = str(c.op).upper()

        for i in range(n):
            x = float(labels[i])
            if not _finite(x):
                out[i] = 0.0
                continue

            if op == "EQ":
                ok = (x == v)
            elif op == "NE":
                ok = (x != v)
            elif op == "GT":
                ok = (x > v)
            elif op == "GE":
                ok = (x >= v)
            elif op == "LT":
                ok = (x < v)
            elif op == "LE":
                ok = (x <= v)
            else:
                raise ValidationError(f"Unsupported regime op: {c.op!r}")

            out[i] = 1.0 if ok else 0.0

        return out

    def regime_mask(self, rf: RegimeFilterSpec) -> Optional[List[float]]:
        if rf is None or len(rf.all_of) == 0:
            return None

        key = ("RF", tuple(sorted(self._key_regime_cond(c) for c in rf.all_of)))
        if key in self._regime_mask_cache:
            return self._regime_mask_cache[key]

        mask = [1.0] * self.n
        for cond in rf.all_of:
            m = self._regime_cond_to_mask(cond)
            for i in range(self.n):
                mask[i] = 1.0 if (mask[i] > 0.5 and m[i] > 0.5) else 0.0

        if self._max_regime_cache == 0:
            return mask
        if len(self._regime_mask_cache) >= self._max_regime_cache:
            self._regime_mask_cache.clear()
        self._regime_mask_cache[key] = mask
        return mask

    # ---- Compile a full strategy ----
    def compile(self, spec: StrategySpec) -> CompiledStrategy:
        n = self.n

        if hasattr(spec.entry, "steps"):  # EventSequence
            entry = self.eval_sequence(spec.entry)
        else:
            entry = self.eval_expr(spec.entry)

        exit_: Optional[List[float]]
        if spec.exit is None:
            exit_ = None
        else:
            if hasattr(spec.exit, "steps"):
                exit_ = self.eval_sequence(spec.exit)  # type: ignore[arg-type]
            else:
                exit_ = self.eval_expr(spec.exit)  # type: ignore[arg-type]

        require(len(entry) == n, "entry length mismatch")
        if exit_ is not None:
            require(len(exit_) == n, "exit length mismatch")

        mask = self.regime_mask(spec.regime_filter)
        if mask is not None:
            require(len(mask) == n, "mask length mismatch")
            entry_masked = [0.0] * n
            for i in range(n):
                entry_masked[i] = 1.0 if (entry[i] > 0.5 and mask[i] > 0.5) else 0.0
        else:
            entry_masked = entry

        return CompiledStrategy(
            entry=entry_masked,
            exit=exit_,
            entry_mask=mask,
            stop=spec.stop,
            take_profit=spec.take_profit,
            time_stop_bars=int(spec.time_stop_bars),
            cooldown_bars=int(spec.cooldown_bars),
        )


# -----------------------------
# Backtest evaluation
# -----------------------------
@dataclass(slots=True)
class _Trade:
    entry_i: int
    exit_i: int
    entry_price: float
    exit_price: float
    ret: float  # fraction
    fees: float


@dataclass(slots=True)
class BacktestOutput:
    equity_curve: List[float]
    trades: List[_Trade]
    fees_paid: float
    funding_paid: float


def _simple_backtest(
    series: OhlcvSeries,
    compiled: CompiledStrategy,
    direction: str,
    start: int,
    end: int,
    *,
    fee_bps: float = 4.0,
    aux: Optional[Dict[str, Any]] = None,
) -> BacktestOutput:
    """
    Deterministic simple backtester (fallback). No funding; simple fees.
    See previous version for semantics (unchanged).
    """
    require(0 <= start < end <= len(series.ts_ms), "invalid start/end")
    require(direction in ("long", "short"), "direction must be long/short")

    entry_sig = compiled.entry
    exit_sig = compiled.exit
    stop = compiled.stop
    tp = compiled.take_profit
    time_stop = int(compiled.time_stop_bars)
    cooldown = int(compiled.cooldown_bars)

    o = [float(x) for x in series.open]
    h = [float(x) for x in series.high]
    l = [float(x) for x in series.low]
    c = [float(x) for x in series.close]
    n = len(c)

    fee_rate = float(fee_bps) / 10000.0

    equity_curve = [1.0] * n
    equity_cash = 1.0

    in_pos = False
    entry_i = -1
    entry_price = 0.0
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None
    entry_equity = 1.0
    bars_in_pos = 0
    cooldown_left = 0

    trades: List[_Trade] = []
    fees_paid = 0.0

    def _get_atr(period: int, idx: int) -> Optional[float]:
        if aux is None:
            return None
        atr_map = aux.get("atr", {})
        if period not in atr_map:
            return None
        v = float(atr_map[period][idx])
        return v if _finite(v) else None

    def _get_pivot_level(level_name: str, key_lr: Tuple[int, int], idx: int) -> Optional[float]:
        if aux is None:
            return None
        mp = aux.get(level_name, {})
        if key_lr not in mp:
            return None
        v = float(mp[key_lr][idx])
        return v if _finite(v) else None

    def _set_risk_levels(signal_i: int, entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        sp: Optional[float] = None
        tpv: Optional[float] = None

        if stop.kind == "none":
            sp = None
        elif stop.kind == "percent":
            pct = float(stop.stop_pct)
            sp = entry_px * (1.0 - pct) if direction == "long" else entry_px * (1.0 + pct)
        elif stop.kind == "atr":
            atr_v = _get_atr(int(stop.atr_period), signal_i)
            if atr_v is None or atr_v <= 0.0:
                return None, None
            mult = float(stop.atr_mult)
            sp = entry_px - mult * atr_v if direction == "long" else entry_px + mult * atr_v
        elif stop.kind == "structure":
            key_lr = (int(stop.ms_left), int(stop.ms_right))
            buf = float(stop.buffer_bps) / 10000.0
            if stop.level == "last_pivot_low":
                lvl = _get_pivot_level("last_pivot_low", key_lr, signal_i)
                if lvl is None:
                    return None, None
                sp = float(lvl) * (1.0 - buf)
            else:
                lvl = _get_pivot_level("last_pivot_high", key_lr, signal_i)
                if lvl is None:
                    return None, None
                sp = float(lvl) * (1.0 + buf)
        else:  # pragma: no cover
            return None, None

        # --- take profit ---
        if tp.kind == "none":
            tpv = None
        elif tp.kind == "percent":
            pct = float(tp.tp_pct)
            tpv = entry_px * (1.0 + pct) if direction == "long" else entry_px * (1.0 - pct)
        elif tp.kind == "atr":
            atr_v = _get_atr(int(tp.atr_period), signal_i)
            if atr_v is not None and atr_v > 0.0:
                mult = float(tp.atr_mult)
                tpv = entry_px + mult * atr_v if direction == "long" else entry_px - mult * atr_v
            else:
                tpv = None
        elif tp.kind == "rr":
            rr = float(tp.rr)
            if sp is None:
                tpv = None
            else:
                risk = abs(entry_px - float(sp))
                if risk > 0.0:
                    tpv = entry_px + rr * risk if direction == "long" else entry_px - rr * risk
                else:
                    tpv = None
        else:  # pragma: no cover
            tpv = None

        return sp, tpv

    for i in range(start, end):
        if not in_pos:
            equity_curve[i] = equity_cash
        else:
            ur = (float(c[i]) - entry_price) / entry_price if direction == "long" else (entry_price - float(c[i])) / entry_price
            equity_curve[i] = entry_equity * (1.0 + ur)

        if not in_pos and cooldown_left > 0:
            cooldown_left -= 1

        if not in_pos and cooldown_left == 0 and entry_sig[i] > 0.5:
            px = float(c[i])
            fee = fee_rate
            equity_cash *= (1.0 - fee)
            fees_paid += fee

            entry_equity = equity_cash
            entry_price = px
            entry_i = i
            bars_in_pos = 0

            sp, tpv = _set_risk_levels(signal_i=i, entry_px=px)
            if stop.kind in ("atr", "structure", "percent") and sp is None:
                equity_cash /= (1.0 - fee) if (1.0 - fee) > 0 else equity_cash
                fees_paid -= fee
                entry_i = -1
                entry_price = 0.0
                entry_equity = equity_cash
                continue

            stop_price = sp
            tp_price = tpv
            in_pos = True
            continue

        if in_pos and i > entry_i:
            stop_hit = False
            tp_hit = False

            if stop_price is not None:
                stop_hit = float(l[i]) <= float(stop_price) if direction == "long" else float(h[i]) >= float(stop_price)

            if tp_price is not None:
                tp_hit = float(h[i]) >= float(tp_price) if direction == "long" else float(l[i]) <= float(tp_price)

            exit_reason = None
            exit_px: Optional[float] = None

            if stop_hit:
                exit_reason = "stop"
                exit_px = float(stop_price)
            elif tp_hit:
                exit_reason = "tp"
                exit_px = float(tp_price)

            if exit_reason is None:
                if exit_sig is not None and exit_sig[i] > 0.5:
                    exit_reason = "exit"
                    exit_px = float(c[i])
                elif time_stop > 0 and bars_in_pos >= time_stop:
                    exit_reason = "time"
                    exit_px = float(c[i])

            if in_pos and stop.kind == "atr" and bool(stop.trail):
                atr_v = _get_atr(int(stop.atr_period), i)
                if atr_v is not None and atr_v > 0.0 and stop_price is not None:
                    mult = float(stop.atr_mult)
                    if direction == "long":
                        new_sp = float(c[i]) - mult * atr_v
                        stop_price = max(float(stop_price), new_sp)
                    else:
                        new_sp = float(c[i]) + mult * atr_v
                        stop_price = min(float(stop_price), new_sp)

            bars_in_pos += 1

            if exit_reason is not None and exit_px is not None:
                ret = (exit_px - entry_price) / entry_price if direction == "long" else (entry_price - exit_px) / entry_price
                equity_cash = entry_equity * (1.0 + ret)

                fee = fee_rate
                equity_cash *= (1.0 - fee)
                fees_paid += fee

                trades.append(_Trade(
                    entry_i=entry_i,
                    exit_i=i,
                    entry_price=entry_price,
                    exit_price=exit_px,
                    ret=ret,
                    fees=2.0 * fee_rate,
                ))

                in_pos = False
                entry_i = -1
                entry_price = 0.0
                stop_price = None
                tp_price = None
                entry_equity = equity_cash
                bars_in_pos = 0
                cooldown_left = cooldown
                continue

    if in_pos:
        i = end - 1
        exit_px = float(c[i])
        ret = (exit_px - entry_price) / entry_price if direction == "long" else (entry_price - exit_px) / entry_price
        equity_cash = entry_equity * (1.0 + ret)

        fee = fee_rate
        equity_cash *= (1.0 - fee)
        fees_paid += fee

        trades.append(_Trade(
            entry_i=entry_i,
            exit_i=i,
            entry_price=entry_price,
            exit_price=exit_px,
            ret=ret,
            fees=2.0 * fee_rate,
        ))

    for i in range(end, n):
        equity_curve[i] = equity_curve[i - 1] if i > 0 else 1.0

    return BacktestOutput(
        equity_curve=equity_curve,
        trades=trades,
        fees_paid=float(fees_paid),
        funding_paid=0.0,
    )


def _compute_fold_metrics(fold_index: int, start: int, end: int, out: BacktestOutput) -> FoldMetrics:
    eq = out.equity_curve
    require(0 <= start < end <= len(eq), "invalid metrics window")

    eq_window = eq[start:end]
    eq0 = float(eq_window[0]) if eq_window else 1.0
    eq1 = float(eq_window[-1]) if eq_window else eq0
    net_ret = (eq1 / eq0 - 1.0) if eq0 > 0 else 0.0

    peak = eq0
    max_dd = 0.0
    for v in eq_window:
        fv = float(v)
        if fv > peak:
            peak = fv
        dd = (peak - fv) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    trades = out.trades
    t_in_window = [t for t in trades if (t.entry_i >= start and t.exit_i < end)]
    ntr = len(t_in_window)
    wins = sum(1 for t in t_in_window if float(t.ret) > 0.0)
    win_rate = float(wins) / float(ntr) if ntr > 0 else 0.0

    gross_profit = sum(float(t.ret) for t in t_in_window if float(t.ret) > 0.0)
    gross_loss = -sum(float(t.ret) for t in t_in_window if float(t.ret) < 0.0)
    pf = _safe_div(gross_profit, gross_loss, default=0.0) if ntr > 0 else 0.0

    avg_trade = _mean([float(t.ret) for t in t_in_window]) if ntr > 0 else 0.0

    bars_in_pos = 0
    for t in t_in_window:
        bars_in_pos += max(0, int(t.exit_i) - int(t.entry_i) + 1)
    exposure = float(bars_in_pos) / float(max(1, end - start))
    turnover = (float(ntr) / float(max(1, end - start))) * 1000.0

    return FoldMetrics(
        fold_index=int(fold_index),
        start=int(start),
        end=int(end),
        trades=int(ntr),
        win_rate=float(win_rate),
        profit_factor=float(pf),
        net_return=float(net_ret),
        max_drawdown=float(max_dd),
        avg_trade_return=float(avg_trade),
        exposure=float(exposure),
        turnover=float(turnover),
        fees_paid=float(out.fees_paid),
        funding_paid=float(out.funding_paid),
    )


# -----------------------------
# Backtest Engine adapter (optional)
# -----------------------------
def _try_backtest_engine(series: OhlcvSeries, compiled: CompiledStrategy, direction: str, start: int, end: int) -> Optional[BacktestOutput]:
    """
    Attempts to run project Backtest_Engine if a compatible API is detected.
    Returns BacktestOutput on success, else None.
    """
    try:
        import Backtest_Engine as BE  # type: ignore
    except Exception:
        return None

    candidates: List[Tuple[str, Any]] = []
    for name in ("run_backtest", "backtest", "simulate", "simulate_signals", "run"):
        if hasattr(BE, name):
            candidates.append((name, getattr(BE, name)))

    for name in ("BacktestEngine", "Engine", "Simulator"):
        if hasattr(BE, name):
            candidates.append((name, getattr(BE, name)))

    if not candidates:
        return None

    entry = compiled.entry
    exit_ = compiled.exit

    kwargs_pool = {
        "series": series,
        "ohlcv": series,
        "data": series,
        "entry": entry,
        "entry_signal": entry,
        "entry_signals": entry,
        "exit": exit_,
        "exit_signal": exit_,
        "exit_signals": exit_,
        "direction": direction,
        "side": direction,
        "start": start,
        "end": end,
        "start_idx": start,
        "end_idx": end,
        "stop": compiled.stop,
        "stop_spec": compiled.stop,
        "take_profit": compiled.take_profit,
        "tp": compiled.take_profit,
        "tp_spec": compiled.take_profit,
        "time_stop_bars": compiled.time_stop_bars,
        "cooldown_bars": compiled.cooldown_bars,
    }

    def _call_with_signature(fn: Any) -> Any:
        sig = inspect.signature(fn)
        kwargs = {}
        for p in sig.parameters.values():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if p.name in kwargs_pool:
                kwargs[p.name] = kwargs_pool[p.name]
        return fn(**kwargs)

    for name, obj in candidates:
        try:
            if inspect.isclass(obj):
                inst = obj()  # type: ignore[call-arg]
                for meth in ("run_backtest", "backtest", "simulate", "simulate_signals", "run"):
                    if hasattr(inst, meth):
                        fn = getattr(inst, meth)
                        res = _call_with_signature(fn)
                        parsed = _parse_engine_result(res, end)
                        if parsed is not None:
                            return parsed
            else:
                res = _call_with_signature(obj)
                parsed = _parse_engine_result(res, end)
                if parsed is not None:
                    return parsed
        except Exception:
            continue

    return None


def _parse_engine_result(res: Any, end: int) -> Optional[BacktestOutput]:
    if res is None:
        return None

    if isinstance(res, BacktestOutput):
        return res

    equity = None
    trades = None
    fees_paid = 0.0
    funding_paid = 0.0

    if isinstance(res, dict):
        for k in ("equity_curve", "equity", "equity_series"):
            if k in res:
                equity = res[k]
                break
        for k in ("trades", "trade_list"):
            if k in res:
                trades = res[k]
                break
        fees_paid = float(res.get("fees_paid", 0.0) or 0.0)
        funding_paid = float(res.get("funding_paid", 0.0) or 0.0)
    else:
        for k in ("equity_curve", "equity", "equity_series"):
            if hasattr(res, k):
                equity = getattr(res, k)
                break
        for k in ("trades", "trade_list"):
            if hasattr(res, k):
                trades = getattr(res, k)
                break
        if hasattr(res, "fees_paid"):
            fees_paid = float(getattr(res, "fees_paid") or 0.0)
        if hasattr(res, "funding_paid"):
            funding_paid = float(getattr(res, "funding_paid") or 0.0)

    if equity is None:
        return None

    try:
        eq = [float(x) for x in equity]
        require(len(eq) >= end, "engine equity curve shorter than required end index")
    except Exception:
        return None

    parsed_trades: List[_Trade] = []
    if trades is not None:
        try:
            for t in trades:
                if isinstance(t, dict):
                    ei = int(t.get("entry_i", t.get("entry_idx", -1)))
                    xi = int(t.get("exit_i", t.get("exit_idx", -1)))
                    ep = float(t.get("entry_price", t.get("entry", _NAN)))
                    xp = float(t.get("exit_price", t.get("exit", _NAN)))
                    r = float(t.get("ret", t.get("return", 0.0)))
                    parsed_trades.append(_Trade(ei, xi, ep, xp, r, float(t.get("fees", 0.0) or 0.0)))
                else:
                    ei = int(getattr(t, "entry_i", getattr(t, "entry_idx", -1)))
                    xi = int(getattr(t, "exit_i", getattr(t, "exit_idx", -1)))
                    ep = float(getattr(t, "entry_price", getattr(t, "entry", _NAN)))
                    xp = float(getattr(t, "exit_price", getattr(t, "exit", _NAN)))
                    r = float(getattr(t, "ret", getattr(t, "return", 0.0)))
                    parsed_trades.append(_Trade(ei, xi, ep, xp, r, float(getattr(t, "fees", 0.0) or 0.0)))
        except Exception:
            parsed_trades = []

    return BacktestOutput(
        equity_curve=eq,
        trades=parsed_trades,
        fees_paid=float(fees_paid),
        funding_paid=float(funding_paid),
    )


# -----------------------------
# Miner
# -----------------------------
class StrategyMiner:
    def __init__(
        self,
        series: OhlcvSeries,
        space: StrategySpace,
        wf_cfg: WalkForwardConfig,
        search_cfg: SearchConfig,
        score_cfg: ScoringConfig,
    ) -> None:
        require(series is not None, "series must not be None")
        self.series = series
        self.space = space
        self.wf_cfg = wf_cfg
        self.search_cfg = search_cfg
        self.score_cfg = score_cfg

        self.compiler = StrategyCompiler(
            series,
            max_expr_cache=int(search_cfg.max_expr_cache),
            max_regime_cache=int(search_cfg.max_regime_cache),
        )

        self.splits = make_walkforward_splits(len(series.ts_ms), wf_cfg)

        # auxiliary feature cache for simple backtester
        self._aux: Dict[str, Any] = {
            "atr": {},               # period -> series
            "last_pivot_low": {},    # (l,r) -> series
            "last_pivot_high": {},   # (l,r) -> series
        }

        self._rng = random.Random(int(search_cfg.seed))

        # candidate stats
        self._cand_stats: Dict[str, int] = {
            "space_seen": 0,
            "sampled_in": 0,
            "sampled_out": 0,
            "skipped_entry_cap": 0,
            "skipped_family_cap": 0,
            "yielded": 0,
        }

    # ---- Diversification keys ----
    def _entry_key(self, spec: StrategySpec) -> str:
        """
        Unique key for entry signature (direction + canonical entry AST).
        Ignores stop/tp/exit/time/cooldown/regime.
        """
        ct = spec.canonical_tuple()
        entry_part = None
        # ct looks like: ("StrategySpec", ("direction",...), ("entry", ...), ...)
        for item in ct:
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "entry":
                entry_part = item[1]
                break
        if entry_part is None:
            entry_part = repr(spec.entry)
        s = repr((spec.direction, entry_part)).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    def _family_bucket(self, spec: StrategySpec) -> str:
        """
        Direction-aware family bucket for caps.
        """
        tags = set(str(t).strip().lower() for t in (spec.tags or ()))
        if "mean_reversion" in tags:
            fam = "mean_reversion"
        elif "breakout" in tags:
            fam = "breakout"
        elif ("trend" in tags) or ("continuation" in tags):
            fam = "trend"
        elif ("reversal" in tags) or ("liquidity" in tags):
            fam = "reversal"
        else:
            fam = "other"
        return f"{spec.direction}:{fam}"

    # ---- Candidate iteration ----
    def _iter_candidates(self) -> Iterator[StrategySpec]:
        cfg = self.search_cfg
        mode = cfg.mode
        p = float(cfg.sample_prob)

        div_entry = bool(cfg.diversify_by_entry)
        max_per_entry = int(cfg.max_variants_per_entry)

        div_family = bool(cfg.diversify_by_family)
        max_per_family = int(cfg.max_per_family)

        # auto distribution: 4 main families * 2 directions = 8 buckets
        if div_family and max_per_family == 0:
            buckets = 8
            max_per_family = max(1, int(math.ceil(float(cfg.max_evals) / float(buckets))))
            # small slack for empty buckets
            max_per_family = int(max_per_family)

        entry_counts: Dict[str, int] = {}
        family_counts: Dict[str, int] = {}

        for spec in self.space.iter_strategies():
            self._cand_stats["space_seen"] += 1

            if mode == "sample":
                if self._rng.random() > p:
                    self._cand_stats["sampled_out"] += 1
                    continue
                self._cand_stats["sampled_in"] += 1

            if div_entry:
                ek = self._entry_key(spec)
                c = entry_counts.get(ek, 0)
                if c >= max_per_entry:
                    self._cand_stats["skipped_entry_cap"] += 1
                    continue

            if div_family:
                fb = self._family_bucket(spec)
                fc = family_counts.get(fb, 0)
                if fc >= max_per_family:
                    self._cand_stats["skipped_family_cap"] += 1
                    continue

            # accept
            if div_entry:
                entry_counts[ek] = entry_counts.get(ek, 0) + 1
            if div_family:
                family_counts[fb] = family_counts.get(fb, 0) + 1

            self._cand_stats["yielded"] += 1
            yield spec

    # ---- Entry filters with reason ----
    def _passes_entry_filters(self, entry: Sequence[float]) -> Tuple[bool, str]:
        sc = self.score_cfg

        total_entries = 0
        for split in self.splits:
            s = int(split.test.start)  # type: ignore[arg-type]
            e = int(split.test.stop)   # type: ignore[arg-type]
            cnt = 0
            for i in range(s, e):
                if entry[i] > 0.5:
                    cnt += 1
            if cnt < int(sc.min_entries_per_fold):
                return False, "min_entries_per_fold"
            total_entries += cnt

        if total_entries < int(sc.min_entries_total):
            return False, "min_entries_total"
        if total_entries > int(sc.max_entries_total):
            return False, "max_entries_total"
        return True, "ok"

    def _ensure_aux_for(self, compiled: CompiledStrategy) -> None:
        if compiled.stop.kind == "atr":
            p = int(compiled.stop.atr_period)
            if p not in self._aux["atr"]:
                self._aux["atr"][p] = self.compiler.hub.ind.indicator("atr", period=p)

        if compiled.take_profit.kind == "atr":
            p = int(compiled.take_profit.atr_period)
            if p not in self._aux["atr"]:
                self._aux["atr"][p] = self.compiler.hub.ind.indicator("atr", period=p)

        if compiled.stop.kind == "structure":
            l = int(compiled.stop.ms_left)
            r = int(compiled.stop.ms_right)
            key = (l, r)
            if compiled.stop.level == "last_pivot_low":
                if key not in self._aux["last_pivot_low"]:
                    self._aux["last_pivot_low"][key] = self.compiler.hub.ms.feature("last_pivot_low", left=l, right=r)
            else:
                if key not in self._aux["last_pivot_high"]:
                    self._aux["last_pivot_high"][key] = self.compiler.hub.ms.feature("last_pivot_high", left=l, right=r)

    def _run_backtest(self, compiled: CompiledStrategy, direction: str, start: int, end: int) -> BacktestOutput:
        mode = self.search_cfg.backtest_mode
        self._ensure_aux_for(compiled)

        if mode in ("engine", "auto"):
            engine_out = _try_backtest_engine(self.series, compiled, direction, start, end)
            if engine_out is not None:
                return engine_out
            if mode == "engine":
                raise ValidationError("backtest_mode='engine' but Backtest_Engine adapter could not run.")

        return _simple_backtest(
            self.series,
            compiled,
            direction,
            start,
            end,
            fee_bps=float(self.score_cfg.fee_bps),
            aux=self._aux,
        )

    def _score(self, spec: StrategySpec, folds: Sequence[FoldMetrics]) -> Tuple[float, Dict[str, Any]]:
        sc = self.score_cfg

        rets = [float(f.net_return) for f in folds]
        dds = [float(f.max_drawdown) for f in folds]
        pfs = [float(f.profit_factor) for f in folds]
        turns = [float(f.turnover) for f in folds]

        mean_ret = _mean(rets)
        std_ret = _std(rets)
        mean_dd = _mean(dds)
        mean_pf = _mean(pfs)
        mean_turn = _mean(turns)
        worst_fold = min(rets) if rets else 0.0

        pf_term = math.log(1.0 + mean_pf) if mean_pf > 0.0 else 0.0
        complexity = float(spec.complexity())

        score = 0.0
        score += float(sc.weight_return) * mean_ret
        score -= float(sc.weight_dd) * mean_dd
        score -= float(sc.weight_stability) * std_ret
        score += float(sc.weight_pf) * pf_term
        score -= float(sc.weight_complexity) * complexity
        score -= float(sc.weight_turnover) * (mean_turn / 1000.0)
        score += float(sc.weight_worst_fold) * worst_fold

        agg = {
            "mean_return": mean_ret,
            "std_return": std_ret,
            "worst_fold_return": worst_fold,
            "mean_drawdown": mean_dd,
            "mean_profit_factor": mean_pf,
            "mean_turnover_per_1000": mean_turn,
            "complexity": int(spec.complexity()),
            "fold_returns": rets,
            "fold_drawdowns": dds,
            "fold_trades": [int(f.trades) for f in folds],
        }
        return float(score), agg

    # ---- Main run ----
    def run(self) -> MinerReport:
        cfg = self.search_cfg
        sc = self.score_cfg

        evaluated = 0
        accepted = 0
        rejected = 0

        reject_reasons: Dict[str, int] = {}

        def _rej(reason: str) -> None:
            r = str(reason)
            reject_reasons[r] = int(reject_reasons.get(r, 0)) + 1

        heap: List[Tuple[float, str, CandidateResult]] = []

        for spec in self._iter_candidates():
            if evaluated >= int(cfg.max_evals):
                break

            evaluated += 1

            # ---- compile
            try:
                compiled = self.compiler.compile(spec)
            except Exception:
                rejected += 1
                _rej("compile_fail")
                continue

            # ---- cheap entry filters
            ok_entry, reason = self._passes_entry_filters(compiled.entry)
            if not ok_entry:
                rejected += 1
                _rej(f"entry_filter:{reason}")
                continue

            # ---- evaluate folds
            fold_metrics: List[FoldMetrics] = []
            total_trades = 0
            ok = True
            fail_reason = ""

            for fi, split in enumerate(self.splits):
                start = int(split.test.start)  # type: ignore[arg-type]
                end = int(split.test.stop)     # type: ignore[arg-type]

                try:
                    bt_out = self._run_backtest(compiled, spec.direction, start, end)
                except Exception:
                    ok = False
                    fail_reason = "backtest_fail"
                    break

                fm = _compute_fold_metrics(fi, start, end, bt_out)
                fold_metrics.append(fm)
                total_trades += int(fm.trades)

                if int(fm.trades) < int(sc.min_trades_per_fold):
                    ok = False
                    fail_reason = "min_trades_per_fold"
                    break
                if float(fm.max_drawdown) > float(sc.max_drawdown_limit):
                    ok = False
                    fail_reason = "max_drawdown_fail"
                    break

            if not ok:
                rejected += 1
                _rej(fail_reason or "fold_fail")
                continue

            if total_trades < int(sc.min_trades_total):
                rejected += 1
                _rej("min_trades_total")
                continue

            score, agg = self._score(spec, fold_metrics)
            if not _finite(score):
                rejected += 1
                _rej("nonfinite_score")
                continue

            res = CandidateResult(
                strategy_id=spec.id_str(),
                strategy_hash=spec.hash(),
                strategy_name=spec.name,
                direction=spec.direction,
                tags=spec.tags,
                complexity=spec.complexity(),
                score=float(score),
                fold_metrics=tuple(fold_metrics),
                aggregate=agg,
                regime_filter=_to_jsonable(spec.regime_filter),
            )

            accepted += 1

            if len(heap) < int(cfg.top_k):
                heapq.heappush(heap, (float(res.score), res.strategy_hash, res))
            else:
                if float(res.score) > float(heap[0][0]):
                    heapq.heapreplace(heap, (float(res.score), res.strategy_hash, res))

        top = sorted((x[2] for x in heap), key=lambda r: float(r.score), reverse=True)

        cand_stats = dict(self._cand_stats)
        cand_stats["evaluated"] = int(evaluated)

        return MinerReport(
            evaluated=int(evaluated),
            accepted=int(accepted),
            rejected=int(rejected),
            folds=int(len(self.splits)),
            reject_reasons=reject_reasons,
            candidate_stats=_to_jsonable(cand_stats),
            top_results=tuple(top),
            search_config=_to_jsonable(asdict(self.search_cfg)),
            wf_config=_to_jsonable(asdict(self.wf_cfg)),
            scoring_config=_to_jsonable(asdict(self.score_cfg)),
        )


# -----------------------------
# Export
# -----------------------------
def save_results_json(path: str, report: MinerReport) -> None:
    p = str(path).strip()
    require(p, "path must be non-empty")

    payload = _to_jsonable({
        "evaluated": report.evaluated,
        "accepted": report.accepted,
        "rejected": report.rejected,
        "folds": report.folds,
        "reject_reasons": report.reject_reasons,
        "candidate_stats": report.candidate_stats,
        "search_config": report.search_config,
        "wf_config": report.wf_config,
        "scoring_config": report.scoring_config,
        "top_results": [
            {
                "strategy_id": r.strategy_id,
                "strategy_hash": r.strategy_hash,
                "strategy_name": r.strategy_name,
                "direction": r.direction,
                "tags": list(r.tags),
                "complexity": r.complexity,
                "score": r.score,
                "aggregate": r.aggregate,
                "fold_metrics": [_to_jsonable(f) for f in r.fold_metrics],
                "regime_filter": r.regime_filter,
            }
            for r in report.top_results
        ],
    })

    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# -----------------------------
# Self-test
# -----------------------------
def _make_synthetic_series(n: int = 400) -> OhlcvSeries:
    require(n >= 200, "n must be >= 200 for self-test folds")

    ts = list(range(1, n + 1))
    close: List[float] = []
    high: List[float] = []
    low: List[float] = []
    open_: List[float] = []
    vol: List[float] = []

    for i in range(n):
        base = 100.0 + 2.0 * math.sin(2.0 * math.pi * i / 50.0)
        drift = 0.015 * i if i < n // 2 else 0.015 * (n // 2) - 0.010 * (i - n // 2)

        c = base + drift
        close.append(c)

        o = close[i - 1] if i > 0 else c
        open_.append(o)

        wick = 0.6 + 0.25 * math.sin(2.0 * math.pi * i / 17.0)
        extra = 0.4 if (i % 47 == 0) else 0.0
        hh = max(c + wick, o, c)
        ll = min(c - wick - extra, o, c)
        high.append(hh)
        low.append(ll)

        v = 200.0 + 50.0 * math.sin(2.0 * math.pi * i / 30.0) + (20.0 if (i % 60 < 5) else 0.0)
        vol.append(max(1.0, v))

    return OhlcvSeries(
        ts_ms=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=vol,
        symbol="SYN",
        timeframe="1m",
    )


def _self_test() -> None:
    series = _make_synthetic_series(400)

    cfg = StrategySpaceConfig.small_self_test()
    budget = ComplexityBudget(
        max_entry_steps=3,
        max_total_atomic_conditions=8,
        max_within_bars=20,
        max_complexity=22,
    )
    space = StrategySpace(cfg, budget)

    wf = WalkForwardConfig(
        train_bars=200,
        test_bars=100,
        step_bars=100,
        purge_bars=0,
        embargo_bars=0,
        anchored=False,
    )

    search = SearchConfig(
        mode="iterate",
        max_evals=60,
        top_k=10,
        seed=123,
        backtest_mode="simple",
        max_expr_cache=2000,
        max_regime_cache=300,
        diversify_by_entry=True,
        max_variants_per_entry=1,
        diversify_by_family=True,
        max_per_family=0,  # auto
    )

    scoring = ScoringConfig(
        min_entries_total=1,
        max_entries_total=5000,
        min_entries_per_fold=0,
        min_trades_total=1,
        min_trades_per_fold=0,
        max_drawdown_limit=0.90,
        fee_bps=2.0,
    )

    miner = StrategyMiner(series, space, wf, search, scoring)
    report = miner.run()

    require(report.evaluated > 0, "Expected evaluated > 0")
    require(isinstance(report.reject_reasons, dict), "reject_reasons must be dict")
    require(report.accepted >= 1, f"Expected at least one accepted strategy in self-test, got {report.accepted}")

    # ensure sorted desc
    scores = [r.score for r in report.top_results]
    require(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Top results not sorted by score desc")

    print("Miner_Search self-test: OK")
    print(f"Evaluated: {report.evaluated} | Accepted: {report.accepted} | Rejected: {report.rejected} | Folds: {report.folds}")
    if report.top_results:
        top = report.top_results[0]
        print(f"Top1: id={top.strategy_id} score={top.score:.6f} dir={top.direction} complexity={top.complexity} tags={list(top.tags)}")
    if report.reject_reasons:
        # show top reasons
        items = sorted(report.reject_reasons.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("Top reject reasons:", ", ".join(f"{k}={v}" for k, v in items))


if __name__ == "__main__":
    _self_test()
