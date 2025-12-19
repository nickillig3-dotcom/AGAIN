from __future__ import annotations

"""
Strategy_Space.py
=================

Purpose
-------
Defines a formal, mining-friendly strategy grammar (StrategySpec) and a configurable
Strategy Space generator that combines:

- Indicators (Features_Indicators via Signals_Events FeatureRef kind="indicator")
- Market Structure (Features_MarketStructure via FeatureRef kind="structure")
- Temporal sequences (Domino principle) via EventSequence / SequenceStep
- Risk specs (Stop / TakeProfit / time stop / cooldown)
- Regime filters (using Regime_Classifier by name, stored as specs; no runtime integration here)

This file does NOT backtest or trade.
It only creates structured StrategySpec objects with stable IDs and complexity scores.

Non-negotiable principles
-------------------------
- No lookahead: FeatureRef.shift is always >= 0 (Signals_Events enforces this).
- Deterministic generation order (sorted grids).
- Large strategy space, but controlled via ComplexityBudget and grids.

Note about Regime filters
-------------------------
Signals_Events currently supports FeatureRef.kind in {"base","indicator","structure"}.
So Regime filters are stored as separate specs in StrategySpec.regime_filter.
Later (single-file loop) we can integrate "regime" into Signals_Events or add an adapter.

Run self-test:
--------------
python Strategy_Space.py
"""

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from Core_Types import ValidationError, require
from Signals_Events import (
    Op,
    FeatureRef,
    Condition,
    AllOf,
    AnyOf,
    Not,
    SequenceStep,
    EventSequence,
)

# -----------------------------
# Basic helpers
# -----------------------------
_NAN = float("nan")


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


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


def _check_nonneg_int(x: Any, name: str) -> int:
    v = _as_int(x, name)
    require(v >= 0, f"{name} must be >= 0, got {v}")
    return v


def _check_pos_float(x: Any, name: str) -> float:
    v = _as_float(x, name)
    require(v > 0.0, f"{name} must be > 0, got {v}")
    return v


def _check_nonneg_float(x: Any, name: str) -> float:
    v = _as_float(x, name)
    require(v >= 0.0, f"{name} must be >= 0, got {v}")
    return v


def _round_float(x: float, nd: int = 12) -> float:
    # Stable float normalization for canonicalization.
    return float(round(float(x), nd))


def _freeze_params(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    # stable, hashable, order-independent representation
    items: List[Tuple[str, Any]] = []
    for k, v in (d or {}).items():
        items.append((str(k), _freeze_value(v)))
    return tuple(sorted(items))


def _freeze_value(v: Any) -> Any:
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
        return _round_float(v)
    if isinstance(v, str):
        return str(v)
    if isinstance(v, (tuple, list)):
        return tuple(_freeze_value(x) for x in v)
    if isinstance(v, dict):
        return _freeze_params(v)
    try:
        return str(v)
    except Exception:
        return repr(v)


# -----------------------------
# Risk / Regime Specs
# -----------------------------
@dataclass(frozen=True, slots=True)
class StopSpec:
    """
    Risk stop specification.

    kind:
      - "none"
      - "percent"   : stop_pct (e.g., 0.01 for 1%)
      - "atr"       : atr_period, atr_mult, trail
      - "structure" : ms_left, ms_right, buffer_bps, level ("last_pivot_low"/"last_pivot_high")
                     Level is side-aware; generator sets it.
    """
    kind: str

    # percent
    stop_pct: float = 0.0

    # atr
    atr_period: int = 14
    atr_mult: float = 2.0
    trail: bool = False

    # structure
    ms_left: int = 2
    ms_right: int = 2
    buffer_bps: float = 0.0
    level: str = ""  # "last_pivot_low" or "last_pivot_high"

    def __post_init__(self) -> None:
        k = str(self.kind).strip().lower()
        require(k in ("none", "percent", "atr", "structure"), f"StopSpec.kind invalid: {self.kind!r}")
        object.__setattr__(self, "kind", k)

        if k == "percent":
            sp = _check_pos_float(self.stop_pct, "stop_pct")
            require(sp < 0.50, "stop_pct must be < 0.50 (50%) for sanity")
            object.__setattr__(self, "stop_pct", float(sp))

        elif k == "atr":
            ap = _check_pos_int(self.atr_period, "atr_period")
            am = _check_pos_float(self.atr_mult, "atr_mult")
            object.__setattr__(self, "atr_period", int(ap))
            object.__setattr__(self, "atr_mult", float(am))
            object.__setattr__(self, "trail", bool(self.trail))

        elif k == "structure":
            ml = _check_pos_int(self.ms_left, "ms_left")
            mr = _check_pos_int(self.ms_right, "ms_right")
            bb = _check_nonneg_float(self.buffer_bps, "buffer_bps")
            lvl = str(self.level).strip().lower()
            require(lvl in ("last_pivot_low", "last_pivot_high"), f"StopSpec.level invalid: {self.level!r}")
            object.__setattr__(self, "ms_left", int(ml))
            object.__setattr__(self, "ms_right", int(mr))
            object.__setattr__(self, "buffer_bps", float(bb))
            object.__setattr__(self, "level", lvl)

        # none -> no additional checks


@dataclass(frozen=True, slots=True)
class TakeProfitSpec:
    """
    Take profit specification.

    kind:
      - "none"
      - "rr"       : rr (risk multiple based on stop distance)
      - "percent"  : tp_pct
      - "atr"      : atr_period, atr_mult
    """
    kind: str

    rr: float = 0.0
    tp_pct: float = 0.0
    atr_period: int = 14
    atr_mult: float = 2.0

    def __post_init__(self) -> None:
        k = str(self.kind).strip().lower()
        require(k in ("none", "rr", "percent", "atr"), f"TakeProfitSpec.kind invalid: {self.kind!r}")
        object.__setattr__(self, "kind", k)

        if k == "rr":
            rr = _check_pos_float(self.rr, "rr")
            require(rr <= 10.0, "rr too large for default space (<=10)")
            object.__setattr__(self, "rr", float(rr))

        elif k == "percent":
            tp = _check_pos_float(self.tp_pct, "tp_pct")
            require(tp < 1.0, "tp_pct must be < 1.0 (100%) for sanity")
            object.__setattr__(self, "tp_pct", float(tp))

        elif k == "atr":
            ap = _check_pos_int(self.atr_period, "atr_period")
            am = _check_pos_float(self.atr_mult, "atr_mult")
            object.__setattr__(self, "atr_period", int(ap))
            object.__setattr__(self, "atr_mult", float(am))


@dataclass(frozen=True, slots=True)
class RegimeConditionSpec:
    """
    Regime condition stored as a spec (not evaluated here).

    regime_name:
      - "trend_dir" | "vol_regime" | "volume_regime" | "phase_id" (others later allowed)

    op: one of {"EQ","NE","GT","GE","LT","LE"}
    value: numeric constant
    params: passed to RegimeStore.regime(regime_name, **params) later
    """
    regime_name: str
    op: str
    value: float
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rn = str(self.regime_name).strip().lower()
        require(rn, "RegimeConditionSpec.regime_name must be non-empty")
        op = str(self.op).strip().upper()
        require(op in ("EQ", "NE", "GT", "GE", "LT", "LE"), f"RegimeConditionSpec.op invalid: {self.op!r}")
        v = _as_float(self.value, "value")
        object.__setattr__(self, "regime_name", rn)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "value", float(v))


@dataclass(frozen=True, slots=True)
class RegimeFilterSpec:
    """
    Conjunction of regime conditions (ALL must hold).
    """
    all_of: Tuple[RegimeConditionSpec, ...]

    def __post_init__(self) -> None:
        require(self.all_of is not None, "RegimeFilterSpec.all_of must not be None")
        # Allow empty tuple to represent "no filter"
        object.__setattr__(self, "all_of", tuple(self.all_of))


# -----------------------------
# Strategy Spec
# -----------------------------
ConditionExpr = Union[Condition, AllOf, AnyOf, Not]
EntryLogic = Union[ConditionExpr, EventSequence]
ExitLogic = Optional[Union[ConditionExpr, EventSequence]]


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """
    A strategy definition as data (AST/spec), not executable code.

    direction: "long" or "short"
    entry: ConditionExpr or EventSequence
    exit:  optional ConditionExpr or EventSequence (can be None)
    stop: StopSpec (required; can be "none")
    take_profit: TakeProfitSpec
    time_stop_bars: max holding duration in bars (0 means disabled)
    cooldown_bars: bars to wait after a trade closes before next entry (0 means none)
    regime_filter: optional RegimeFilterSpec (empty -> none)
    tags: tuple of strings describing style/regime intent
    """
    direction: str
    entry: EntryLogic
    exit: ExitLogic
    stop: StopSpec
    take_profit: TakeProfitSpec
    time_stop_bars: int = 0
    cooldown_bars: int = 0
    regime_filter: RegimeFilterSpec = field(default_factory=lambda: RegimeFilterSpec(all_of=tuple()))
    tags: Tuple[str, ...] = tuple()
    name: str = ""

    def __post_init__(self) -> None:
        d = str(self.direction).strip().lower()
        require(d in ("long", "short"), f"StrategySpec.direction invalid: {self.direction!r}")
        object.__setattr__(self, "direction", d)

        ts = _check_nonneg_int(self.time_stop_bars, "time_stop_bars")
        cd = _check_nonneg_int(self.cooldown_bars, "cooldown_bars")
        object.__setattr__(self, "time_stop_bars", int(ts))
        object.__setattr__(self, "cooldown_bars", int(cd))

        require(self.entry is not None, "StrategySpec.entry must not be None")
        require(self.stop is not None and isinstance(self.stop, StopSpec), "StrategySpec.stop must be StopSpec")
        require(self.take_profit is not None and isinstance(self.take_profit, TakeProfitSpec), "StrategySpec.take_profit must be TakeProfitSpec")
        require(self.regime_filter is not None and isinstance(self.regime_filter, RegimeFilterSpec), "StrategySpec.regime_filter must be RegimeFilterSpec")
        object.__setattr__(self, "tags", tuple(str(t) for t in (self.tags or tuple())))

        nm = str(self.name).strip()
        object.__setattr__(self, "name", nm)

    # ---- Canonicalization / IDs ----
    def canonical_tuple(self) -> Tuple[Any, ...]:
        """
        Stable, order-insensitive canonical representation where appropriate.
        """
        return (
            "StrategySpec",
            ("direction", self.direction),
            ("entry", _canon_entry(self.entry)),
            ("exit", _canon_exit(self.exit)),
            ("stop", _canon_stop(self.stop)),
            ("take_profit", _canon_tp(self.take_profit)),
            ("time_stop_bars", int(self.time_stop_bars)),
            ("cooldown_bars", int(self.cooldown_bars)),
            ("regime_filter", _canon_regime_filter(self.regime_filter)),
            ("tags", tuple(sorted(self.tags))),
            ("name", self.name),
        )

    def hash(self) -> str:
        s = repr(self.canonical_tuple()).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    def id_str(self) -> str:
        # short id useful for logs
        return self.hash()[:12]

    # ---- Complexity ----
    def complexity(self) -> int:
        """
        A simple, monotonic complexity score used for pruning/ordering.
        """
        atomic_entry = _count_atomic_in_entry(self.entry)
        atomic_exit = _count_atomic_in_exit(self.exit)
        steps = _count_steps(self.entry)
        score = 0
        score += atomic_entry + atomic_exit
        score += 2 * max(0, steps - 1)  # sequences are more complex
        if self.stop.kind != "none":
            score += 1
        if self.take_profit.kind != "none":
            score += 1
        if len(self.regime_filter.all_of) > 0:
            score += 1 + len(self.regime_filter.all_of) // 2
        if self.time_stop_bars > 0:
            score += 1
        if self.cooldown_bars > 0:
            score += 1
        return int(score)


# -----------------------------
# Canonicalization helpers
# -----------------------------
def _canon_feature_ref(fr: FeatureRef) -> Tuple[Any, ...]:
    return (
        "FeatureRef",
        ("kind", fr.kind),
        ("name", fr.name),
        ("params", _freeze_params(fr.params or {})),
        ("shift", int(fr.shift)),
    )


def _canon_condition(cond: Condition) -> Tuple[Any, ...]:
    rhs = cond.rhs
    if isinstance(rhs, FeatureRef):
        rhs_c = ("FeatureRefRHS", _canon_feature_ref(rhs))
    elif isinstance(rhs, tuple):
        rhs_c = ("TupleRHS", tuple(_freeze_value(x) for x in rhs))
    elif rhs is None:
        rhs_c = ("NoneRHS", None)
    else:
        rhs_c = ("ConstRHS", _freeze_value(rhs))
    return (
        "Condition",
        ("lhs", _canon_feature_ref(cond.lhs)),
        ("op", cond.op.value),
        ("rhs", rhs_c),
    )


def _canon_expr(expr: ConditionExpr) -> Tuple[Any, ...]:
    if isinstance(expr, Condition):
        return _canon_condition(expr)
    if isinstance(expr, AllOf):
        # Order-insensitive: sort terms by repr of canonical form to dedup permutations.
        terms = sorted((_canon_expr(t) for t in expr.terms), key=lambda x: repr(x))
        return ("AllOf", tuple(terms))
    if isinstance(expr, AnyOf):
        terms = sorted((_canon_expr(t) for t in expr.terms), key=lambda x: repr(x))
        return ("AnyOf", tuple(terms))
    if isinstance(expr, Not):
        return ("Not", _canon_expr(expr.term))
    raise ValidationError(f"Unsupported ConditionExpr type: {type(expr).__name__}")


def _canon_sequence(seq: EventSequence) -> Tuple[Any, ...]:
    steps = []
    for st in seq.steps:
        steps.append((
            "Step",
            ("expr", _canon_expr(st.expr) if not isinstance(st.expr, EventSequence) else ("NestedSeqUnsupported",)),
            ("within_bars", int(st.within_bars)),
            ("hold_bars", int(st.hold_bars)),
        ))
    return ("EventSequence", tuple(steps))


def _canon_entry(entry: EntryLogic) -> Tuple[Any, ...]:
    if isinstance(entry, EventSequence):
        return _canon_sequence(entry)
    return _canon_expr(entry)  # ConditionExpr


def _canon_exit(exit_: ExitLogic) -> Any:
    if exit_ is None:
        return None
    if isinstance(exit_, EventSequence):
        return _canon_sequence(exit_)
    return _canon_expr(exit_)


def _canon_stop(stop: StopSpec) -> Tuple[Any, ...]:
    return (
        "StopSpec",
        ("kind", stop.kind),
        ("stop_pct", _round_float(stop.stop_pct)),
        ("atr_period", int(stop.atr_period)),
        ("atr_mult", _round_float(stop.atr_mult)),
        ("trail", bool(stop.trail)),
        ("ms_left", int(stop.ms_left)),
        ("ms_right", int(stop.ms_right)),
        ("buffer_bps", _round_float(stop.buffer_bps)),
        ("level", stop.level),
    )


def _canon_tp(tp: TakeProfitSpec) -> Tuple[Any, ...]:
    return (
        "TakeProfitSpec",
        ("kind", tp.kind),
        ("rr", _round_float(tp.rr)),
        ("tp_pct", _round_float(tp.tp_pct)),
        ("atr_period", int(tp.atr_period)),
        ("atr_mult", _round_float(tp.atr_mult)),
    )


def _canon_regime_filter(rf: RegimeFilterSpec) -> Tuple[Any, ...]:
    # Order-insensitive: sort by repr
    items = []
    for c in rf.all_of:
        items.append((
            "RegimeCond",
            ("regime_name", c.regime_name),
            ("op", c.op),
            ("value", _round_float(c.value)),
            ("params", _freeze_params(c.params or {})),
        ))
    items = sorted(items, key=lambda x: repr(x))
    return ("RegimeFilter", tuple(items))


# -----------------------------
# Complexity helpers
# -----------------------------
def _count_atomic_expr(expr: ConditionExpr) -> int:
    if isinstance(expr, Condition):
        return 1
    if isinstance(expr, AllOf):
        return sum(_count_atomic_expr(t) for t in expr.terms)
    if isinstance(expr, AnyOf):
        return sum(_count_atomic_expr(t) for t in expr.terms)
    if isinstance(expr, Not):
        return _count_atomic_expr(expr.term)
    raise ValidationError(f"Unsupported ConditionExpr type: {type(expr).__name__}")


def _count_atomic_in_entry(entry: EntryLogic) -> int:
    if isinstance(entry, EventSequence):
        return sum(_count_atomic_expr(st.expr) for st in entry.steps)
    return _count_atomic_expr(entry)


def _count_atomic_in_exit(exit_: ExitLogic) -> int:
    if exit_ is None:
        return 0
    if isinstance(exit_, EventSequence):
        return sum(_count_atomic_expr(st.expr) for st in exit_.steps)
    return _count_atomic_expr(exit_)


def _count_steps(entry: EntryLogic) -> int:
    if isinstance(entry, EventSequence):
        return len(entry.steps)
    return 1


# -----------------------------
# Budget + Config
# -----------------------------
@dataclass(frozen=True, slots=True)
class ComplexityBudget:
    """
    Hard constraints to keep the space large but mineable.

    max_entry_steps: maximum number of steps in the entry EventSequence.
    max_total_atomic_conditions: total atomic conditions in entry+exit.
    max_within_bars: maximum allowed within_bars in sequences.
    max_complexity: total complexity score.
    """
    max_entry_steps: int = 3
    max_total_atomic_conditions: int = 8
    max_within_bars: int = 50
    max_complexity: int = 14

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_entry_steps", _check_pos_int(self.max_entry_steps, "max_entry_steps"))
        object.__setattr__(self, "max_total_atomic_conditions", _check_pos_int(self.max_total_atomic_conditions, "max_total_atomic_conditions"))
        object.__setattr__(self, "max_within_bars", _check_nonneg_int(self.max_within_bars, "max_within_bars"))
        object.__setattr__(self, "max_complexity", _check_pos_int(self.max_complexity, "max_complexity"))


@dataclass(frozen=True, slots=True)
class StrategySpaceConfig:
    """
    Parameter grids for strategy generation.

    Keep grids coarse by default for robustness; expand later for "maximal" search.
    """
    # Market structure pivots / events
    ms_left_right: Tuple[Tuple[int, int], ...] = ((1, 1), (2, 2), (3, 3))
    ms_tol_bps: Tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)

    # Sequence timing
    within_bars: Tuple[int, ...] = (0, 3, 5, 10, 20)

    # RSI
    rsi_periods: Tuple[int, ...] = (7, 14, 21)
    rsi_oversold: Tuple[float, ...] = (20.0, 30.0)
    rsi_overbought: Tuple[float, ...] = (70.0, 80.0)
    rsi_mid: Tuple[float, ...] = (50.0,)

    # EMA
    ema_fast_periods: Tuple[int, ...] = (10, 20)
    ema_slow_periods: Tuple[int, ...] = (50, 100)

    # ADX filter (optional in entry)
    adx_periods: Tuple[int, ...] = (14,)
    adx_thresholds: Tuple[float, ...] = (20.0, 25.0)

    # Bollinger
    bb_periods: Tuple[int, ...] = (20,)
    bb_mults: Tuple[float, ...] = (2.0,)

    # Donchian
    donchian_periods: Tuple[int, ...] = (20, 50)

    # Volume zscore
    volume_z_periods: Tuple[int, ...] = (200,)
    volume_z_thresholds: Tuple[float, ...] = (0.5, 1.0)

    # Stops
    stop_percent: Tuple[float, ...] = (0.005, 0.01, 0.015)  # 0.5%, 1%, 1.5%
    stop_atr_periods: Tuple[int, ...] = (14,)
    stop_atr_mults: Tuple[float, ...] = (1.5, 2.0, 2.5)
    stop_trailing: Tuple[bool, ...] = (False,)
    stop_structure_buffer_bps: Tuple[float, ...] = (5.0, 10.0, 20.0)

    # Take profits
    tp_rr: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    tp_percent: Tuple[float, ...] = tuple()  # optional
    tp_atr_periods: Tuple[int, ...] = tuple()
    tp_atr_mults: Tuple[float, ...] = tuple()

    # Trade mgmt
    time_stops: Tuple[int, ...] = (0, 60, 120)
    cooldowns: Tuple[int, ...] = (0, 5, 10)

    # Regime filter templates (specs) - can be expanded later
    use_regime_filters: bool = True
    regime_params_trend: Dict[str, Any] = field(default_factory=lambda: {
        "adx_period": 14,
        "adx_threshold": 20.0,
        "di_min_sep": 0.0,
        "ms_left": 2,
        "ms_right": 2,
        "tol_bps": 0.0,
    })
    regime_params_vol: Dict[str, Any] = field(default_factory=lambda: {
        "atr_period": 14,
        "z_period": 200,
        "z_threshold": 1.0,
    })
    regime_params_volume: Dict[str, Any] = field(default_factory=lambda: {
        "z_period": 200,
        "z_threshold": 1.0,
    })

    # Generation limit (for debug)
    max_generated: int = 0  # 0 means unlimited

    @staticmethod
    def small_self_test() -> "StrategySpaceConfig":
        """
        Smaller grids for deterministic self-test (fast).
        """
        return StrategySpaceConfig(
            ms_left_right=((1, 1),),
            ms_tol_bps=(0.0,),
            within_bars=(0, 3),
            rsi_periods=(3,),
            rsi_oversold=(30.0,),
            rsi_overbought=(70.0,),
            rsi_mid=(50.0,),
            ema_fast_periods=(10,),
            ema_slow_periods=(50,),
            adx_periods=(14,),
            adx_thresholds=(20.0,),
            bb_periods=(20,),
            bb_mults=(2.0,),
            donchian_periods=(20,),
            volume_z_periods=(50,),
            volume_z_thresholds=(0.5,),
            stop_percent=(0.01,),
            stop_atr_periods=(14,),
            stop_atr_mults=(2.0,),
            stop_trailing=(False,),
            stop_structure_buffer_bps=(10.0,),
            tp_rr=(2.0,),
            time_stops=(0, 60),
            cooldowns=(0,),
            use_regime_filters=True,
            max_generated=200,
        )


# -----------------------------
# Strategy Space Generator
# -----------------------------
class StrategySpace:
    """
    Generates StrategySpec objects from templates + parameter grids.
    Deterministic order, deduplicated by canonical hash.
    """

    def __init__(self, cfg: StrategySpaceConfig, budget: ComplexityBudget) -> None:
        require(cfg is not None, "cfg must not be None")
        require(budget is not None, "budget must not be None")
        self.cfg = cfg
        self.budget = budget

    def iter_strategies(self) -> Iterator[StrategySpec]:
        cfg = self.cfg
        budget = self.budget

        seen: set[str] = set()
        emitted = 0
        max_gen = int(cfg.max_generated) if int(cfg.max_generated) > 0 else None

        for direction in ("long", "short"):
            for entry, tags, name in self._iter_entry_templates(direction):
                for exit_ in self._iter_exit_templates(direction, entry_family_tags=tags):
                    for stop in self._iter_stops(direction):
                        for tp in self._iter_take_profits():
                            for ts in sorted(cfg.time_stops):
                                for cd in sorted(cfg.cooldowns):
                                    for rf in self._iter_regime_filters(direction, entry_family_tags=tags):
                                        spec = StrategySpec(
                                            direction=direction,
                                            entry=entry,
                                            exit=exit_,
                                            stop=stop,
                                            take_profit=tp,
                                            time_stop_bars=int(ts),
                                            cooldown_bars=int(cd),
                                            regime_filter=rf,
                                            tags=tuple(tags),
                                            name=name,
                                        )

                                        if not self._within_budget(spec):
                                            continue

                                        h = spec.hash()
                                        if h in seen:
                                            continue
                                        seen.add(h)

                                        yield spec
                                        emitted += 1
                                        if max_gen is not None and emitted >= max_gen:
                                            return

    # ---- Budget checks ----
    def _within_budget(self, spec: StrategySpec) -> bool:
        b = self.budget

        # entry steps limit
        if isinstance(spec.entry, EventSequence):
            if len(spec.entry.steps) > b.max_entry_steps:
                return False
            for j, st in enumerate(spec.entry.steps):
                if j > 0 and int(st.within_bars) > b.max_within_bars:
                    return False

        total_atomic = _count_atomic_in_entry(spec.entry) + _count_atomic_in_exit(spec.exit)
        if total_atomic > b.max_total_atomic_conditions:
            return False

        if spec.complexity() > b.max_complexity:
            return False

        return True

    # ---- Regime filters ----
    def _iter_regime_filters(self, direction: str, entry_family_tags: Tuple[str, ...]) -> Iterator[RegimeFilterSpec]:
        cfg = self.cfg

        # Always include "no filter"
        yield RegimeFilterSpec(all_of=tuple())

        if not cfg.use_regime_filters:
            return

        # Directional trend filters (common and robust)
        if "trend" in entry_family_tags or "breakout" in entry_family_tags:
            # require trend_dir == +1/-1
            val = 1.0 if direction == "long" else -1.0
            yield RegimeFilterSpec(all_of=(
                RegimeConditionSpec("trend_dir", "EQ", val, params=dict(cfg.regime_params_trend)),
            ))

        # Range / compression filters for mean reversion
        if "mean_reversion" in entry_family_tags:
            yield RegimeFilterSpec(all_of=(
                RegimeConditionSpec("trend_dir", "EQ", 0.0, params=dict(cfg.regime_params_trend)),
                RegimeConditionSpec("vol_regime", "EQ", -1.0, params=dict(cfg.regime_params_vol)),
            ))

        # High vol / high volume confirmation for breakouts
        if "breakout" in entry_family_tags:
            yield RegimeFilterSpec(all_of=(
                RegimeConditionSpec("vol_regime", "EQ", 1.0, params=dict(cfg.regime_params_vol)),
            ))

    # ---- Stops / Take profits ----
    def _iter_stops(self, direction: str) -> Iterator[StopSpec]:
        cfg = self.cfg

        # Percent stops
        for sp in sorted(cfg.stop_percent):
            yield StopSpec(kind="percent", stop_pct=float(sp))

        # ATR stops
        for ap in sorted(cfg.stop_atr_periods):
            for am in sorted(cfg.stop_atr_mults):
                for tr in sorted(cfg.stop_trailing):
                    yield StopSpec(kind="atr", atr_period=int(ap), atr_mult=float(am), trail=bool(tr))

        # Structure stops (side-aware pivot level)
        lvl = "last_pivot_low" if direction == "long" else "last_pivot_high"
        for (l, r) in sorted(cfg.ms_left_right):
            for bb in sorted(cfg.stop_structure_buffer_bps):
                yield StopSpec(kind="structure", ms_left=int(l), ms_right=int(r), buffer_bps=float(bb), level=lvl)

        # "none" stop is allowed but discouraged (kept for space completeness)
        yield StopSpec(kind="none")

    def _iter_take_profits(self) -> Iterator[TakeProfitSpec]:
        cfg = self.cfg

        for rr in sorted(cfg.tp_rr):
            yield TakeProfitSpec(kind="rr", rr=float(rr))

        for tp in sorted(cfg.tp_percent):
            yield TakeProfitSpec(kind="percent", tp_pct=float(tp))

        # ATR take profits optional
        for ap in sorted(cfg.tp_atr_periods):
            for am in sorted(cfg.tp_atr_mults):
                yield TakeProfitSpec(kind="atr", atr_period=int(ap), atr_mult=float(am))

        yield TakeProfitSpec(kind="none")

    # ---- Entry templates ----
    def _iter_entry_templates(self, direction: str) -> Iterator[Tuple[EntryLogic, Tuple[str, ...], str]]:
        """
        Returns (entry_logic, tags, name).
        Tags are high-level family descriptors used for regime filter selection.
        """
        cfg = self.cfg
        side = direction

        for (l, r) in sorted(cfg.ms_left_right):
            for tol in sorted(cfg.ms_tol_bps):
                ms_params = {"left": int(l), "right": int(r), "tol_bps": float(tol)}

                # --- 1) Liquidity sweep reversal (sequence) ---
                for rp in sorted(cfg.rsi_periods):
                    if side == "long":
                        sweep_name = "sweep_low"
                        cross_op = Op.CROSS_ABOVE
                        thr_list = cfg.rsi_oversold
                        ema_cross_op = Op.CROSS_ABOVE
                    else:
                        sweep_name = "sweep_high"
                        cross_op = Op.CROSS_BELOW
                        thr_list = cfg.rsi_overbought
                        ema_cross_op = Op.CROSS_BELOW

                    sweep = Condition(
                        lhs=FeatureRef("structure", sweep_name, dict(ms_params), shift=0),
                        op=Op.IS_TRUE,
                        rhs=None,
                    )

                    for thr in sorted(thr_list):
                        rsi_cross = Condition(
                            lhs=FeatureRef("indicator", "rsi", {"period": int(rp), "source": "close"}, shift=0),
                            op=cross_op,
                            rhs=float(thr),
                        )

                        # Optional price confirmation: close crosses EMA(fast)
                        for ema_fast in sorted(cfg.ema_fast_periods):
                            price_cross = Condition(
                                lhs=FeatureRef("base", "close", {}, shift=0),
                                op=ema_cross_op,
                                rhs=FeatureRef("indicator", "ema", {"period": int(ema_fast), "source": "close"}, shift=0),
                            )

                            for w1 in sorted(cfg.within_bars):
                                for w2 in sorted(cfg.within_bars):
                                    seq = EventSequence(steps=[
                                        SequenceStep(expr=sweep, within_bars=0, hold_bars=1),
                                        SequenceStep(expr=rsi_cross, within_bars=int(w1), hold_bars=1),
                                        SequenceStep(expr=price_cross, within_bars=int(w2), hold_bars=1),
                                    ])
                                    name = f"sweep_reversal_{side}_L{l}R{r}_tol{tol}_rsi{rp}_thr{thr}_ema{ema_fast}_w{w1}-{w2}"
                                    tags = ("reversal", "liquidity", "sequence")
                                    yield seq, tags, name

                # --- 2) BOS continuation (sequence) ---
                if side == "long":
                    bos_name = "bos_up"
                    close_cross_op = Op.CROSS_ABOVE
                else:
                    bos_name = "bos_down"
                    close_cross_op = Op.CROSS_BELOW

                bos = Condition(
                    lhs=FeatureRef("structure", bos_name, dict(ms_params), shift=0),
                    op=Op.IS_TRUE,
                    rhs=None,
                )

                # Add a simple trend filter on entry-bar: EMA(fast) > EMA(slow) for long (reverse for short)
                for ema_fast in sorted(cfg.ema_fast_periods):
                    for ema_slow in sorted(cfg.ema_slow_periods):
                        if ema_fast >= ema_slow:
                            continue

                        ema_fast_ref = FeatureRef("indicator", "ema", {"period": int(ema_fast), "source": "close"}, shift=0)
                        ema_slow_ref = FeatureRef("indicator", "ema", {"period": int(ema_slow), "source": "close"}, shift=0)

                        ema_trend = Condition(
                            lhs=ema_fast_ref,
                            op=(Op.GT if side == "long" else Op.LT),
                            rhs=ema_slow_ref,
                        )

                        price_cross = Condition(
                            lhs=FeatureRef("base", "close", {}, shift=0),
                            op=close_cross_op,
                            rhs=ema_fast_ref,
                        )

                        for w in sorted(cfg.within_bars):
                            seq = EventSequence(steps=[
                                SequenceStep(expr=bos, within_bars=0, hold_bars=1),
                                SequenceStep(expr=AllOf([price_cross, ema_trend]), within_bars=int(w), hold_bars=1),
                            ])
                            name = f"bos_cont_{side}_L{l}R{r}_tol{tol}_ema{ema_fast}-{ema_slow}_w{w}"
                            tags = ("trend", "continuation", "sequence")
                            yield seq, tags, name

                # --- 3) Mean reversion (single-step condition) ---
                for bp in sorted(cfg.bb_periods):
                    for mult in sorted(cfg.bb_mults):
                        if side == "long":
                            # close < bb_lower AND rsi < oversold
                            bb_ref = FeatureRef("indicator", "bb_lower", {"period": int(bp), "mult": float(mult), "source": "close"}, shift=0)
                            bb_cond = Condition(lhs=FeatureRef("base", "close", {}, shift=0), op=Op.LT, rhs=bb_ref)
                            thr_list = cfg.rsi_oversold
                            rsi_op = Op.LT
                        else:
                            bb_ref = FeatureRef("indicator", "bb_upper", {"period": int(bp), "mult": float(mult), "source": "close"}, shift=0)
                            bb_cond = Condition(lhs=FeatureRef("base", "close", {}, shift=0), op=Op.GT, rhs=bb_ref)
                            thr_list = cfg.rsi_overbought
                            rsi_op = Op.GT

                        for rp in sorted(cfg.rsi_periods):
                            for thr in sorted(thr_list):
                                rsi_cond = Condition(
                                    lhs=FeatureRef("indicator", "rsi", {"period": int(rp), "source": "close"}, shift=0),
                                    op=rsi_op,
                                    rhs=float(thr),
                                )
                                entry = AllOf([bb_cond, rsi_cond])
                                name = f"bb_mr_{side}_bb{bp}x{mult}_rsi{rp}_thr{thr}"
                                tags = ("mean_reversion", "bands")
                                yield entry, tags, name

                # --- 4) Breakout + volume confirmation (sequence) ---
                if side == "long":
                    br_name = "break_up_close"
                    don_name = "donchian_high"
                    don_op = Op.GT
                else:
                    br_name = "break_down_close"
                    don_name = "donchian_low"
                    don_op = Op.LT

                br = Condition(
                    lhs=FeatureRef("structure", br_name, dict(ms_params), shift=0),
                    op=Op.IS_TRUE,
                    rhs=None,
                )

                for dp in sorted(cfg.donchian_periods):
                    don_ref = FeatureRef("indicator", don_name, {"period": int(dp)}, shift=0)
                    don_break = Condition(
                        lhs=FeatureRef("base", "close", {}, shift=0),
                        op=don_op,
                        rhs=don_ref,
                    )

                    # Volume zscore confirmation
                    for vzp in sorted(cfg.volume_z_periods):
                        vol_z = FeatureRef("indicator", "zscore", {"source": "volume", "period": int(vzp)}, shift=0)
                        for vzthr in sorted(cfg.volume_z_thresholds):
                            vol_cond = Condition(lhs=vol_z, op=Op.GT, rhs=float(vzthr))

                            for w in sorted(cfg.within_bars):
                                seq = EventSequence(steps=[
                                    SequenceStep(expr=AnyOf([br, don_break]), within_bars=0, hold_bars=1),
                                    SequenceStep(expr=vol_cond, within_bars=int(w), hold_bars=1),
                                ])
                                name = f"breakout_{side}_L{l}R{r}_tol{tol}_don{dp}_vz{vzp}thr{vzthr}_w{w}"
                                tags = ("breakout", "sequence")
                                yield seq, tags, name

    # ---- Exit templates ----
    def _iter_exit_templates(self, direction: str, entry_family_tags: Tuple[str, ...]) -> Iterator[ExitLogic]:
        """
        Exit logic templates. It's OK for exit to be None (risk mgmt handles).
        Kept conservative and simple to avoid overfitting.
        """
        cfg = self.cfg
        side = direction

        # Always allow "no rule-based exit" (stop/tp/time stop manage the trade)
        yield None

        # For mean reversion: exit on BB midline cross
        if "mean_reversion" in entry_family_tags:
            for bp in sorted(cfg.bb_periods):
                for mult in sorted(cfg.bb_mults):
                    bb_mid = FeatureRef("indicator", "bb_mid", {"period": int(bp), "mult": float(mult), "source": "close"}, shift=0)
                    if side == "long":
                        yield Condition(lhs=FeatureRef("base", "close", {}, shift=0), op=Op.CROSS_ABOVE, rhs=bb_mid)
                    else:
                        yield Condition(lhs=FeatureRef("base", "close", {}, shift=0), op=Op.CROSS_BELOW, rhs=bb_mid)

        # Generic exit: RSI crosses back through midline
        for rp in sorted(cfg.rsi_periods):
            for mid in sorted(cfg.rsi_mid):
                rsi_ref = FeatureRef("indicator", "rsi", {"period": int(rp), "source": "close"}, shift=0)
                if side == "long":
                    yield Condition(lhs=rsi_ref, op=Op.CROSS_BELOW, rhs=float(mid))
                else:
                    yield Condition(lhs=rsi_ref, op=Op.CROSS_ABOVE, rhs=float(mid))


# -----------------------------
# Self-test
# -----------------------------
def _self_test() -> None:
    def _expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    cfg = StrategySpaceConfig.small_self_test()
    budget = ComplexityBudget(
        max_entry_steps=3,
        max_total_atomic_conditions=8,
        max_within_bars=10,
        max_complexity=20,
    )

    space = StrategySpace(cfg, budget)

    # Deterministic generation check: first few IDs match across runs
    specs1 = list(space.iter_strategies())
    _expect(len(specs1) >= 20, f"Expected at least 20 strategies in small self-test space, got {len(specs1)}")

    specs2 = list(StrategySpace(cfg, budget).iter_strategies())
    _expect(len(specs2) == len(specs1), "Deterministic size mismatch across runs")

    ids1 = [s.hash() for s in specs1[:10]]
    ids2 = [s.hash() for s in specs2[:10]]
    _expect(ids1 == ids2, "Deterministic order/hash mismatch across runs")

    # Hash stability check
    s0 = specs1[0]
    h0a = s0.hash()
    h0b = s0.hash()
    _expect(h0a == h0b, "Strategy hash not stable")

    # Uniqueness check
    uniq = len(set(s.hash() for s in specs1))
    _expect(uniq == len(specs1), "Expected all generated strategies to be deduplicated/unique")

    # Complexity sanity
    for s in specs1[:50]:
        c = s.complexity()
        _expect(isinstance(c, int) and c >= 0, "Complexity must be non-negative int")
        _expect(c <= budget.max_complexity, "Generated strategy violates complexity budget")

    # Canonical tuple must be JSON-serializable (via repr/str; not strict JSON),
    # so we at least ensure it does not contain unserializable objects beyond primitives/tuples.
    ct = s0.canonical_tuple()
    _expect(isinstance(ct, tuple), "canonical_tuple must return tuple")

    print("Strategy_Space self-test: OK")
    print(f"Generated strategies: {len(specs1)} (small_self_test)")


if __name__ == "__main__":
    _self_test()
