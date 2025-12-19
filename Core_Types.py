from __future__ import annotations

"""
Core_Types.py
================

This file defines the *core contract types* for the whole project.

Non‑negotiable design principles
--------------------------------
- Inputs are **OHLCV only** (Open, High, Low, Close, Volume). Any feature must be derived from OHLCV.
- **No lookahead**: later components may only use information available up to the current bar.
- **Realism > Robustness > Efficiency** (and efficiency must never break realism).
- Timestamps are **epoch milliseconds** (int) to match Binance conventions.
- This module has **zero third‑party dependencies**. Everything else builds on it.

Why this matters
----------------
Strategy mining creates huge search spaces. Without strict contracts and validation, silent data issues
(NaNs, out‑of‑order bars, impossible OHLC relations) will produce fake edges and catastrophic overfitting.

Keep this module small, stable, and brutally strict.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import numbers
from typing import Iterator, Optional, Sequence, Type, TypeVar


# -----------------------------
# Type aliases (semantic sugar)
# -----------------------------
TimestampMS = int
Price = float
Quantity = float


__all__ = [
    "TimestampMS",
    "Price",
    "Quantity",
    "ContractError",
    "ValidationError",
    "NotSupportedError",
    "require",
    "Side",
    "OrderType",
    "TimeInForce",
    "Bar",
    "OhlcvSeries",
]


# -----------------------------
# Exceptions
# -----------------------------
class ContractError(Exception):
    """Base class for all contract-related errors in the project."""


class ValidationError(ContractError):
    """Raised when input data violates required invariants."""


class NotSupportedError(ContractError):
    """Raised when a requested feature/path is intentionally unsupported."""


# -----------------------------
# Guardrails
# -----------------------------
def require(condition: bool, msg: str) -> None:
    """Raise ValidationError with a clear message if condition is False."""
    if not condition:
        raise ValidationError(msg)


def _is_integral(x: object) -> bool:
    # Accept numpy integer types too.
    return isinstance(x, numbers.Integral)


def _is_real(x: object) -> bool:
    # Accept numpy floating types too.
    return isinstance(x, numbers.Real)


def _finite(x: float) -> bool:
    # math.isfinite handles Python floats; for numpy scalars it still works.
    return math.isfinite(float(x))


def _ge(a: float, b: float, eps: float) -> bool:
    # a >= b with tolerance
    return float(a) + eps >= float(b)


def _le(a: float, b: float, eps: float) -> bool:
    # a <= b with tolerance
    return float(a) <= float(b) + eps


# Default tolerance to avoid false positives from tiny float rounding errors.
_DEFAULT_EPS: float = 1e-12


# -----------------------------
# Enums (fundamental only)
# -----------------------------
class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    GTC = "GTC"  # Good-Till-Canceled
    IOC = "IOC"  # Immediate-Or-Cancel
    FOK = "FOK"  # Fill-Or-Kill


# -----------------------------
# Core OHLCV types
# -----------------------------
@dataclass(frozen=True, slots=True)
class Bar:
    """
    A single OHLCV bar.

    Invariants (strict):
    - ts_ms > 0
    - all numeric fields are finite
    - volume >= 0
    - high >= max(open, close, low) and low <= min(open, close, high)
    """

    ts_ms: TimestampMS
    open: Price
    high: Price
    low: Price
    close: Price
    volume: float

    def __post_init__(self) -> None:
        # Cast to plain Python types (handles numpy scalars cleanly).
        object.__setattr__(self, "ts_ms", int(self.ts_ms))
        object.__setattr__(self, "open", float(self.open))
        object.__setattr__(self, "high", float(self.high))
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "close", float(self.close))
        object.__setattr__(self, "volume", float(self.volume))

        self._validate()

    def _validate(self, eps: float = _DEFAULT_EPS) -> None:
        require(_is_integral(self.ts_ms), f"Bar.ts_ms must be integral, got {type(self.ts_ms).__name__}")
        require(self.ts_ms > 0, f"Bar.ts_ms must be > 0, got {self.ts_ms}")

        for name in ("open", "high", "low", "close", "volume"):
            val = getattr(self, name)
            require(_is_real(val), f"Bar.{name} must be real, got {type(val).__name__}")
            require(_finite(val), f"Bar.{name} must be finite, got {val!r}")

        require(self.volume >= 0.0, f"Bar.volume must be >= 0, got {self.volume}")

        require(
            _ge(self.high, self.low, eps),
            f"Bar.high must be >= Bar.low (eps={eps}), got high={self.high}, low={self.low}",
        )

        # open/close must lie inside [low, high] (with tolerance)
        max_inside = max(self.open, self.close)
        min_inside = min(self.open, self.close)
        require(
            _ge(self.high, max_inside, eps),
            f"Bar.high must be >= max(open, close), got high={self.high}, open={self.open}, close={self.close}",
        )
        require(
            _le(self.low, min_inside, eps),
            f"Bar.low must be <= min(open, close), got low={self.low}, open={self.open}, close={self.close}",
        )

        # Additionally ensure low <= high relative to open/close (redundant but clearer errors)
        require(
            _le(self.low, self.high, eps),
            f"Bar.low must be <= Bar.high (eps={eps}), got low={self.low}, high={self.high}",
        )


T = TypeVar("T")


@dataclass(slots=True)
class OhlcvSeries:
    """
    Columnar OHLCV container.

    This class is designed to work with:
    - Python lists/tuples
    - array('d') etc.
    - numpy arrays (if the user uses numpy elsewhere)

    Contract:
    - All columns have identical length N > 0 (unless allow_empty=True in validate()).
    - Timestamps are strictly increasing (no duplicates, no gaps assumptions).
    - Each row satisfies the same OHLCV invariants as `Bar`.

    Note:
    - This container *does not copy* your arrays. Treat the passed sequences as read-only.
    - If you mutate underlying sequences after validation, you can break invariants.
    """

    ts_ms: Sequence[TimestampMS]
    open: Sequence[Price]
    high: Sequence[Price]
    low: Sequence[Price]
    close: Sequence[Price]
    volume: Sequence[float]

    symbol: Optional[str] = None
    timeframe: Optional[str] = None  # e.g., "1m", "5m", "1h" (convention only)

    validate_on_init: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        if self.validate_on_init:
            self.validate()

    def __len__(self) -> int:
        return len(self.ts_ms)

    @property
    def n(self) -> int:
        return len(self.ts_ms)

    def validate(self, *, allow_empty: bool = False, eps: float = _DEFAULT_EPS) -> None:
        # Length checks
        lengths = (
            len(self.ts_ms),
            len(self.open),
            len(self.high),
            len(self.low),
            len(self.close),
            len(self.volume),
        )
        require(len(set(lengths)) == 1, f"OhlcvSeries columns must have identical length, got {lengths}")
        n = lengths[0]
        if allow_empty:
            return
        require(n > 0, "OhlcvSeries must not be empty")

        # Timestamp monotonicity and per-row validation
        prev_ts: Optional[int] = None
        for i in range(n):
            ts = self.ts_ms[i]
            require(_is_integral(ts), f"ts_ms[{i}] must be integral, got {type(ts).__name__}")
            ts_i = int(ts)
            require(ts_i > 0, f"ts_ms[{i}] must be > 0, got {ts_i}")
            if prev_ts is not None:
                require(
                    ts_i > prev_ts,
                    f"ts_ms must be strictly increasing; ts_ms[{i-1}]={prev_ts} >= ts_ms[{i}]={ts_i}",
                )
            prev_ts = ts_i

            # Per-row OHLCV numeric checks
            o = self.open[i]
            h = self.high[i]
            l = self.low[i]
            c = self.close[i]
            v = self.volume[i]

            # Types & finiteness
            for name, val in (("open", o), ("high", h), ("low", l), ("close", c), ("volume", v)):
                require(_is_real(val), f"{name}[{i}] must be real, got {type(val).__name__}")
                require(_finite(val), f"{name}[{i}] must be finite, got {val!r}")

            o_f, h_f, l_f, c_f, v_f = float(o), float(h), float(l), float(c), float(v)
            require(v_f >= 0.0, f"volume[{i}] must be >= 0, got {v_f}")

            require(
                _ge(h_f, l_f, eps),
                f"high[{i}] must be >= low[{i}] (eps={eps}), got high={h_f}, low={l_f}",
            )

            max_inside = max(o_f, c_f)
            min_inside = min(o_f, c_f)
            require(
                _ge(h_f, max_inside, eps),
                f"high[{i}] must be >= max(open, close), got high={h_f}, open={o_f}, close={c_f}",
            )
            require(
                _le(l_f, min_inside, eps),
                f"low[{i}] must be <= min(open, close), got low={l_f}, open={o_f}, close={c_f}",
            )

    def get_bar(self, index: int) -> Bar:
        """Materialize a Bar at `index`. Useful for debugging and interoperability."""
        return Bar(
            ts_ms=int(self.ts_ms[index]),
            open=float(self.open[index]),
            high=float(self.high[index]),
            low=float(self.low[index]),
            close=float(self.close[index]),
            volume=float(self.volume[index]),
        )

    def iter_bars(self) -> Iterator[Bar]:
        """Iterate over bars (materializes Bar objects)."""
        for i in range(self.n):
            yield self.get_bar(i)

    def slice(self, start: Optional[int] = None, stop: Optional[int] = None) -> "OhlcvSeries":
        """Return a sliced view/copy of the underlying columns (depends on sequence type)."""
        return OhlcvSeries(
            ts_ms=self.ts_ms[slice(start, stop)],
            open=self.open[slice(start, stop)],
            high=self.high[slice(start, stop)],
            low=self.low[slice(start, stop)],
            close=self.close[slice(start, stop)],
            volume=self.volume[slice(start, stop)],
            symbol=self.symbol,
            timeframe=self.timeframe,
            validate_on_init=self.validate_on_init,
        )

    @classmethod
    def from_bars(
        cls: Type["OhlcvSeries"],
        bars: Sequence[Bar],
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        validate_on_init: bool = True,
    ) -> "OhlcvSeries":
        require(len(bars) > 0, "from_bars requires at least one Bar")
        ts_ms = [b.ts_ms for b in bars]
        o = [b.open for b in bars]
        h = [b.high for b in bars]
        l = [b.low for b in bars]
        c = [b.close for b in bars]
        v = [b.volume for b in bars]
        return cls(
            ts_ms=ts_ms,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            symbol=symbol,
            timeframe=timeframe,
            validate_on_init=validate_on_init,
        )


# -----------------------------
# Minimal self-test
# -----------------------------
def _expect_raises(exc_type: Type[BaseException], fn, *, contains: Optional[str] = None) -> None:
    try:
        fn()
    except exc_type as e:
        if contains is not None:
            require(contains in str(e), f"Expected error message to contain {contains!r}, got {str(e)!r}")
        return
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Expected {exc_type.__name__}, but got {type(e).__name__}: {e}") from e
    raise AssertionError(f"Expected {exc_type.__name__} to be raised, but no exception occurred.")


def _self_test() -> None:
    # Valid Bar
    b = Bar(ts_ms=1700000000000, open=100, high=105, low=99, close=102, volume=123.0)
    require(b.high >= b.low, "self-test sanity check failed")

    # Invalid Bar: high < open
    def _bad_bar():
        Bar(ts_ms=1700000000001, open=100, high=99, low=98, close=99, volume=1)

    _expect_raises(ValidationError, _bad_bar, contains="high")

    # Valid Series
    s = OhlcvSeries(
        ts_ms=[1, 2, 3],
        open=[10, 11, 12],
        high=[11, 12, 13],
        low=[9, 10, 11],
        close=[10.5, 11.5, 12.5],
        volume=[100, 110, 120],
        symbol="TEST",
        timeframe="1m",
    )
    require(len(s) == 3, "series length mismatch")
    require(s.get_bar(0).ts_ms == 1, "get_bar failed")

    # Invalid Series: timestamp not increasing
    def _bad_ts():
        OhlcvSeries(
            ts_ms=[1, 1, 2],
            open=[10, 11, 12],
            high=[11, 12, 13],
            low=[9, 10, 11],
            close=[10, 11, 12],
            volume=[1, 1, 1],
        )

    _expect_raises(ValidationError, _bad_ts, contains="strictly increasing")

    # Invalid Series: mismatched lengths
    def _bad_len():
        OhlcvSeries(
            ts_ms=[1, 2, 3],
            open=[10, 11],
            high=[11, 12, 13],
            low=[9, 10, 11],
            close=[10, 11, 12],
            volume=[1, 1, 1],
        )

    _expect_raises(ValidationError, _bad_len, contains="identical length")

    # from_bars
    s2 = OhlcvSeries.from_bars([b, Bar(ts_ms=1700000000002, open=102, high=110, low=101, close=108, volume=50)])
    require(s2.n == 2, "from_bars length mismatch")

    print("Core_Types self-test: OK")


if __name__ == "__main__":
    _self_test()
