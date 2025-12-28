from __future__ import annotations

"""
Signals_Events.py
=================

Purpose
-------
Unifies Indicators and Market Structure into a single, formal "Condition / Event / Sequence" layer.

This is the bridge between:
- continuous features (float arrays, with NaN warmup)
- structure events / levels (0/1 events, trend states, pivot levels)
and the future Strategy Miner:
- systematic enumeration of conditions
- temporal ordering: "A then within X bars B then signal"

Non-negotiable principles
-------------------------
- OHLCV-only (delegated to feature sources).
- No lookahead:
  - FeatureRef.shift must be >= 0 (only past values).
  - CROSS / RISING / FALLING use i-1 (past) only.
- NaN policy:
  - If any required numeric input is NaN/non-finite => Condition is False (0.0).
  - Event arrays are always 0.0 / 1.0.
- Deterministic, strict input validation.

Key abstractions
----------------
- FeatureRef: references a base / indicator / structure feature + parameters + optional shift
- Condition: atomic boolean condition using operators (GT, CROSS_ABOVE, ...)
- ConditionExpr: composite logic (AllOf / AnyOf / Not)
- EventSequence: ordered steps with "within bars" windows (Domino principle)

Typical usage
-------------
hub = SignalHub(series)

cond = Condition(
    lhs=FeatureRef(kind="indicator", name="rsi", params={"period": 14, "source": "close"}),
    op=Op.GT,
    rhs=55.0
)
event = hub.eval(cond)  # List[float] 0/1

seq = EventSequence(steps=[
    SequenceStep(expr=Condition(...), within_bars=0),
    SequenceStep(expr=Condition(...), within_bars=10),
    SequenceStep(expr=Condition(...), within_bars=5),
])
signal = hub.eval_sequence(seq)
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from Core_Types import OhlcvSeries, ValidationError, require
from Features_Indicators import FeatureStore
from Features_MarketStructure import MarketStructureStore


__all__ = [
    "Op",
    "FeatureRef",
    "Condition",
    "AllOf",
    "AnyOf",
    "Not",
    "SequenceStep",
    "EventSequence",
    "SignalHub",
]


_NAN = float("nan")


def _finite(x: float) -> bool:
    return math.isfinite(float(x))

def _as_float(x: Any, name: str = "value") -> float:
    try:
        v = float(x)
    except Exception as e:
        raise ValidationError(f"{name} must be float-castable, got {x!r}") from e
    require(_finite(v), f"{name} must be finite, got {v!r}")
    return v


def _as_int(x: Any, name: str = "value") -> int:
    try:
        v = int(x)
    except Exception as e:
        raise ValidationError(f"{name} must be int-castable, got {x!r}") from e
    return v

# -----------------------------
# Operators
# -----------------------------
class Op(str, Enum):
    # comparisons
    GT = "GT"
    GE = "GE"
    LT = "LT"
    LE = "LE"
    EQ = "EQ"
    NE = "NE"

    BETWEEN = "BETWEEN"     # rhs = (low, high)
    ABS_GT = "ABS_GT"       # rhs = threshold (float)

    # booleans on event-like series
    IS_TRUE = "IS_TRUE"
    IS_FALSE = "IS_FALSE"

    # crosses
    CROSS_ABOVE = "CROSS_ABOVE"
    CROSS_BELOW = "CROSS_BELOW"

    # slopes / monotonic changes
    RISING = "RISING"
    FALLING = "FALLING"


# -----------------------------
# Feature references
# -----------------------------
@dataclass(frozen=True, slots=True)
class FeatureRef:
    """
    Reference to a feature series.

    kind:
      - "base"      => FeatureStore.base(name)
      - "indicator" => FeatureStore.indicator(name, **params)
      - "structure" => MarketStructureStore.feature(name, **params)

    shift:
      - must be >= 0 (only past values)
      - shift=1 means: at bar i, use original[i-1]
    """
    kind: str
    name: str
    params: Dict[str, Any]
    shift: int = 0

    def __post_init__(self) -> None:
        k = str(self.kind).strip().lower()
        n = str(self.name).strip().lower()
        require(k in ("base", "indicator", "structure"), f"FeatureRef.kind must be base/indicator/structure, got {self.kind!r}")
        require(len(n) > 0, "FeatureRef.name must be non-empty")

        s = _as_int(self.shift, "shift")
        require(s >= 0, f"FeatureRef.shift must be >= 0 (no lookahead), got {s}")

        # Normalize (without mutating dict contents)
        object.__setattr__(self, "kind", k)
        object.__setattr__(self, "name", n)
        object.__setattr__(self, "shift", int(s))


# RHS can be constant float/int, FeatureRef, or (low, high) tuple for BETWEEN
RHS = Union[float, int, FeatureRef, Tuple[float, float]]


@dataclass(frozen=True, slots=True)
class Condition:
    """
    Atomic condition: lhs OP rhs -> event series (0/1).

    NaN policy:
      If any needed input is non-finite at bar i => output 0.0 at i.

    Notes:
    - For IS_TRUE / IS_FALSE, rhs is ignored.
    - For RISING / FALLING, rhs is ignored and lhs uses i-1.
    - For CROSS ops, rhs can be FeatureRef or constant.
    """
    lhs: FeatureRef
    op: Op
    rhs: Optional[RHS] = None

    def __post_init__(self) -> None:
        require(isinstance(self.op, Op), f"Condition.op must be Op, got {type(self.op).__name__}")

        if self.op in (Op.BETWEEN,):
            require(isinstance(self.rhs, tuple) and len(self.rhs) == 2, "BETWEEN requires rhs=(low, high)")
            low, high = self.rhs  # type: ignore[misc]
            low_f = _as_float(low, "BETWEEN.low")
            high_f = _as_float(high, "BETWEEN.high")
            require(low_f <= high_f, f"BETWEEN requires low<=high, got low={low_f}, high={high_f}")

        if self.op in (Op.ABS_GT, Op.GT, Op.GE, Op.LT, Op.LE, Op.EQ, Op.NE, Op.CROSS_ABOVE, Op.CROSS_BELOW):
            require(self.rhs is not None, f"{self.op.value} requires rhs")

        if self.op in (Op.IS_TRUE, Op.IS_FALSE, Op.RISING, Op.FALLING):
            # rhs ignored
            return


# -----------------------------
# Composite expressions (logic)
# -----------------------------
ConditionExpr = Union[Condition, "AllOf", "AnyOf", "Not"]


@dataclass(frozen=True, slots=True)
class AllOf:
    terms: Sequence[ConditionExpr]

    def __post_init__(self) -> None:
        require(self.terms is not None and len(self.terms) > 0, "AllOf.terms must be non-empty")


@dataclass(frozen=True, slots=True)
class AnyOf:
    terms: Sequence[ConditionExpr]

    def __post_init__(self) -> None:
        require(self.terms is not None and len(self.terms) > 0, "AnyOf.terms must be non-empty")


@dataclass(frozen=True, slots=True)
class Not:
    term: ConditionExpr


# -----------------------------
# Sequences (Domino principle)
# -----------------------------
@dataclass(frozen=True, slots=True)
class SequenceStep:
    """
    One step in a sequence.

    within_bars:
      For step 0: ignored (can be 0).
      For step j>0: step j must occur within `within_bars` after step j-1.
      Allowed: 0 (same-bar) or more.

    hold_bars:
      Condition must be true for hold_bars consecutive bars.
      hold_bars=1 means "single bar trigger" (default).
    """
    expr: ConditionExpr
    within_bars: int = 0
    hold_bars: int = 1

    def __post_init__(self) -> None:
        w = _as_int(self.within_bars, "within_bars")
        h = _as_int(self.hold_bars, "hold_bars")
        require(w >= 0, f"within_bars must be >= 0, got {w}")
        require(h >= 1, f"hold_bars must be >= 1, got {h}")
        object.__setattr__(self, "within_bars", int(w))
        object.__setattr__(self, "hold_bars", int(h))


@dataclass(frozen=True, slots=True)
class EventSequence:
    steps: Sequence[SequenceStep]

    def __post_init__(self) -> None:
        require(self.steps is not None and len(self.steps) > 0, "EventSequence.steps must be non-empty")


# -----------------------------
# Hub
# -----------------------------
class SignalHub:
    """
    Orchestrates feature retrieval (indicators + structure) and evaluates conditions/sequences.
    """

    def __init__(self, series: OhlcvSeries) -> None:
        require(series is not None, "series must not be None")
        self.series = series

        self.ind = FeatureStore(series)
        self.ms = MarketStructureStore(series)

        # Cache for shifted features to avoid repeated shift work.
        # Key: (kind, name, frozen_params_tuple, shift)
        self._shift_cache: Dict[Tuple[str, str, Tuple[Tuple[str, Any], ...], int], List[float]] = {}

    @property
    def n(self) -> int:
        return len(self.series.ts_ms)

    def get(self, ref: FeatureRef) -> List[float]:
        """
        Retrieve the feature series for a FeatureRef, applying ref.shift (>=0).
        """
        kind = ref.kind
        name = ref.name
        shift = ref.shift

        frozen_params = tuple(sorted((k, _freeze_value(v)) for k, v in (ref.params or {}).items()))
        key = (kind, name, frozen_params, shift)

        if key in self._shift_cache:
            return self._shift_cache[key]

        # Get base series without shift
        if kind == "base":
            base = self.ind.base(name)
        elif kind == "indicator":
            base = self.ind.indicator(name, **(ref.params or {}))
        elif kind == "structure":
            base = self.ms.feature(name, **(ref.params or {}))
        else:  # pragma: no cover (guarded by FeatureRef)
            raise ValidationError(f"Unknown FeatureRef.kind {kind!r}")

        require(len(base) == self.n, "feature length mismatch")

        if shift == 0:
            out = base
        else:
            n = self.n
            out = [_NAN] * n
            for i in range(n):
                j = i - shift
                if j >= 0:
                    out[i] = float(base[j])
            # This is a new list (shifted)
        self._shift_cache[key] = out
        return out

    # ---- Expression evaluation ----
    def eval(self, expr: ConditionExpr) -> List[float]:
        if isinstance(expr, Condition):
            return self.eval_condition(expr)
        if isinstance(expr, AllOf):
            return self._eval_all(expr)
        if isinstance(expr, AnyOf):
            return self._eval_any(expr)
        if isinstance(expr, Not):
            return self._eval_not(expr)
        raise ValidationError(f"Unsupported ConditionExpr type: {type(expr).__name__}")

    def eval_condition(self, cond: Condition) -> List[float]:
        lhs = self.get(cond.lhs)
        n = self.n
        out = [0.0] * n
        op = cond.op

        # Helper to get rhs series or constant
        rhs_is_series = isinstance(cond.rhs, FeatureRef)
        rhs_series: Optional[List[float]] = None
        rhs_const: Optional[float] = None

        if op in (Op.IS_TRUE, Op.IS_FALSE, Op.RISING, Op.FALLING):
            pass
        elif op == Op.BETWEEN:
            # rhs is tuple
            low, high = cond.rhs  # type: ignore[misc]
            rhs_low = _as_float(low, "BETWEEN.low")
            rhs_high = _as_float(high, "BETWEEN.high")
        else:
            if rhs_is_series:
                rhs_series = self.get(cond.rhs)  # type: ignore[arg-type]
            else:
                rhs_const = _as_float(cond.rhs, "rhs")  # type: ignore[arg-type]

        # IS_TRUE / IS_FALSE
        if op == Op.IS_TRUE:
            for i in range(n):
                x = lhs[i]
                if _finite(x) and float(x) > 0.5:
                    out[i] = 1.0
            return out

        if op == Op.IS_FALSE:
            for i in range(n):
                x = lhs[i]
                if _finite(x) and float(x) <= 0.5:
                    out[i] = 1.0
            return out

        # RISING / FALLING
        if op == Op.RISING:
            for i in range(1, n):
                a = lhs[i - 1]
                b = lhs[i]
                if _finite(a) and _finite(b) and float(b) > float(a):
                    out[i] = 1.0
            return out

        if op == Op.FALLING:
            for i in range(1, n):
                a = lhs[i - 1]
                b = lhs[i]
                if _finite(a) and _finite(b) and float(b) < float(a):
                    out[i] = 1.0
            return out

        # BETWEEN
        if op == Op.BETWEEN:
            for i in range(n):
                x = lhs[i]
                if _finite(x) and rhs_low <= float(x) <= rhs_high:
                    out[i] = 1.0
            return out

        # ABS_GT
        if op == Op.ABS_GT:
            thr = float(rhs_const)  # type: ignore[arg-type]
            for i in range(n):
                x = lhs[i]
                if _finite(x) and abs(float(x)) > thr:
                    out[i] = 1.0
            return out

        # comparisons and crosses
        if rhs_is_series:
            rhs = rhs_series  # type: ignore[assignment]
            require(rhs is not None and len(rhs) == n, "rhs series length mismatch")
        else:
            rc = float(rhs_const)  # type: ignore[arg-type]

        if op in (Op.GT, Op.GE, Op.LT, Op.LE, Op.EQ, Op.NE):
            for i in range(n):
                a = lhs[i]
                if not _finite(a):
                    continue
                if rhs_is_series:
                    b = rhs[i]  # type: ignore[index]
                    if not _finite(b):
                        continue
                    bv = float(b)
                else:
                    bv = rc

                av = float(a)
                if op == Op.GT and av > bv:
                    out[i] = 1.0
                elif op == Op.GE and av >= bv:
                    out[i] = 1.0
                elif op == Op.LT and av < bv:
                    out[i] = 1.0
                elif op == Op.LE and av <= bv:
                    out[i] = 1.0
                elif op == Op.EQ and av == bv:
                    out[i] = 1.0
                elif op == Op.NE and av != bv:
                    out[i] = 1.0
            return out

        if op in (Op.CROSS_ABOVE, Op.CROSS_BELOW):
            for i in range(1, n):
                a0 = lhs[i - 1]
                a1 = lhs[i]
                if not (_finite(a0) and _finite(a1)):
                    continue

                if rhs_is_series:
                    b0 = rhs[i - 1]  # type: ignore[index]
                    b1 = rhs[i]      # type: ignore[index]
                    if not (_finite(b0) and _finite(b1)):
                        continue
                    b0v = float(b0); b1v = float(b1)
                else:
                    b0v = rc; b1v = rc

                a0v = float(a0); a1v = float(a1)

                if op == Op.CROSS_ABOVE:
                    if a0v <= b0v and a1v > b1v:
                        out[i] = 1.0
                else:
                    if a0v >= b0v and a1v < b1v:
                        out[i] = 1.0
            return out

        raise ValidationError(f"Unsupported operator {op.value}")

    def _eval_all(self, expr: AllOf) -> List[float]:
        n = self.n
        # Start with all ones; AND in each term
        out = [1.0] * n
        for term in expr.terms:
            ev = self.eval(term)
            require(len(ev) == n, "AllOf term length mismatch")
            for i in range(n):
                out[i] = 1.0 if (out[i] > 0.5 and ev[i] > 0.5) else 0.0
        return out

    def _eval_any(self, expr: AnyOf) -> List[float]:
        n = self.n
        out = [0.0] * n
        for term in expr.terms:
            ev = self.eval(term)
            require(len(ev) == n, "AnyOf term length mismatch")
            for i in range(n):
                out[i] = 1.0 if (out[i] > 0.5 or ev[i] > 0.5) else 0.0
        return out

    def _eval_not(self, expr: Not) -> List[float]:
        ev = self.eval(expr.term)
        n = self.n
        require(len(ev) == n, "Not term length mismatch")
        out = [0.0] * n
        for i in range(n):
            out[i] = 0.0 if ev[i] > 0.5 else 1.0
        return out

    # ---- Sequence evaluation ----
    def eval_sequence(self, seq: EventSequence) -> List[float]:
        require(seq is not None and len(seq.steps) > 0, "seq must be non-empty")
        n = self.n
        k = len(seq.steps)

        # Precompute step event series (with hold handling)
        step_events: List[List[float]] = []
        for idx, step in enumerate(seq.steps):
            ev = self.eval(step.expr)  # 0/1
            require(len(ev) == n, "step event length mismatch")
            if step.hold_bars > 1:
                ev = _apply_hold(ev, step.hold_bars)
            step_events.append(ev)

        signal = [0.0] * n
        INF = 10**18

        # deadline[j] = latest bar index at which step j is allowed to trigger (given previous steps)
        # For j=0: always allowed (INF)
        deadline = [-1] * k
        deadline[0] = INF

        for i in range(n):
            for j in range(k):
                if step_events[j][i] <= 0.5:
                    continue

                if j == 0:
                    # Step 0 always allowed
                    if k == 1:
                        signal[i] = 1.0
                    else:
                        # opens eligibility window for step 1
                        dl = i + seq.steps[1].within_bars
                        if dl > deadline[1]:
                            deadline[1] = dl
                else:
                    # Step j allowed only if within window from step j-1
                    if i <= deadline[j]:
                        if j == k - 1:
                            signal[i] = 1.0
                        else:
                            dl = i + seq.steps[j + 1].within_bars
                            if dl > deadline[j + 1]:
                                deadline[j + 1] = dl

        return signal


def _apply_hold(event: Sequence[float], hold_bars: int) -> List[float]:
    """
    Converts an event series into "hold satisfied at i":
      output[i]=1 iff event is true for hold_bars consecutive bars ending at i.
    """
    h = _as_int(hold_bars, "hold_bars")
    require(h >= 1, "hold_bars must be >= 1")
    n = len(event)
    if h == 1:
        return [1.0 if float(event[i]) > 0.5 else 0.0 for i in range(n)]

    out = [0.0] * n
    streak = 0
    for i in range(n):
        if float(event[i]) > 0.5:
            streak += 1
        else:
            streak = 0
        if streak >= h:
            out[i] = 1.0
    return out


# Helper for freezing parameters (cache key)
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


# -----------------------------
# Self-test
# -----------------------------
def _self_test() -> None:
    def _expect(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    # --- Test 1: NaN warmup handling (RSI period=3 on 5 bars) ---
    series1 = OhlcvSeries(
        ts_ms=[1, 2, 3, 4, 5],
        open=[1, 2, 3, 4, 5],
        high=[2, 3, 4, 5, 6],
        low=[0, 1, 2, 3, 4],
        close=[1, 2, 3, 4, 5],
        volume=[10, 10, 10, 10, 10],
        symbol="TEST",
        timeframe="1m",
    )
    hub1 = SignalHub(series1)

    cond_rsi = Condition(
        lhs=FeatureRef("indicator", "rsi", {"period": 3, "source": "close"}, shift=0),
        op=Op.GT,
        rhs=50.0,
    )
    ev_rsi = hub1.eval(cond_rsi)
    _expect(ev_rsi == [0.0, 0.0, 0.0, 1.0, 1.0], f"RSI warmup condition mismatch: {ev_rsi}")

    # --- Test 2: CROSS_ABOVE close crosses above SMA(3) ---
    # close: 1,1,1,2,3 => SMA3: nan,nan,1,1.333,2 => cross above at i=3
    series2 = OhlcvSeries(
        ts_ms=[1, 2, 3, 4, 5],
        open=[1, 1, 1, 2, 3],
        high=[1.1, 1.1, 1.1, 2.1, 3.1],
        low=[0.9, 0.9, 0.9, 1.9, 2.9],
        close=[1, 1, 1, 2, 3],
        volume=[1, 1, 1, 1, 1],
        symbol="TEST",
        timeframe="1m",
    )
    hub2 = SignalHub(series2)

    cond_cross = Condition(
        lhs=FeatureRef("base", "close", {}, shift=0),
        op=Op.CROSS_ABOVE,
        rhs=FeatureRef("indicator", "sma", {"source": "close", "period": 3}, shift=0),
    )
    ev_cross = hub2.eval(cond_cross)
    _expect(ev_cross == [0.0, 0.0, 0.0, 1.0, 0.0], f"CROSS_ABOVE mismatch: {ev_cross}")

    # --- Test 3 & 4: Structure event + sequence ---
    # Use the deterministic market structure series from Features_MarketStructure self-test.
    highs = [10.0, 11.0, 12.0, 11.0, 12.5, 12.2, 13.0]
    lows =  [ 9.0, 10.0, 11.0, 10.0, 11.5, 11.2, 12.0]
    closes = [9.5, 10.5, 11.5, 10.8, 11.8, 12.0, 12.8]
    series3 = OhlcvSeries(
        ts_ms=[1, 2, 3, 4, 5, 6, 7],
        open=closes[:],
        high=highs,
        low=lows,
        close=closes,
        volume=[1.0] * 7,
        symbol="TEST",
        timeframe="1m",
    )
    hub3 = SignalHub(series3)

    sweep_cond = Condition(
        lhs=FeatureRef("structure", "sweep_high", {"left": 1, "right": 1, "tol_bps": 0.0}, shift=0),
        op=Op.IS_TRUE,
        rhs=None,
    )
    break_cond = Condition(
        lhs=FeatureRef("structure", "break_up_close", {"left": 1, "right": 1, "tol_bps": 0.0}, shift=0),
        op=Op.IS_TRUE,
        rhs=None,
    )

    ev_sweep = hub3.eval(sweep_cond)
    ev_break = hub3.eval(break_cond)

    _expect(ev_sweep[4] == 1.0 and sum(ev_sweep) == 1.0, f"Expected exactly one sweep at i=4, got {ev_sweep}")
    _expect(ev_break[6] == 1.0 and sum(ev_break) == 1.0, f"Expected exactly one break at i=6, got {ev_break}")

    # Sequence: A=sweep_high then within 3 bars B=break_up_close => signal at i=6
    seq = EventSequence(steps=[
        SequenceStep(expr=sweep_cond, within_bars=0, hold_bars=1),
        SequenceStep(expr=break_cond, within_bars=3, hold_bars=1),
    ])
    sig = hub3.eval_sequence(seq)
    _expect(sig[6] == 1.0 and sum(sig) == 1.0, f"Sequence signal mismatch, expected only at i=6, got {sig}")

    # Composite logic smoke test: (break AND NOT sweep) should be true at i=6
    expr = AllOf([break_cond, Not(sweep_cond)])
    ev = hub3.eval(expr)
    _expect(ev[6] == 1.0, "Composite logic failed at i=6")

    print("Signals_Events self-test: OK")


if __name__ == "__main__":
    _self_test()
