from __future__ import annotations

"""
Backtest_Engine.py
==================

Purpose
-------
A strict, conservative OHLCV-only backtest engine for Perpetual Futures style strategies.

Core realism rules (non-negotiable)
-----------------------------------
1) No-lookahead execution:
   - Strategy signals are assumed to be computed on bar i close.
   - Position changes are executed at bar (i+1) open.
   - This prevents "decide on close, fill on same close" lookahead.

2) Conservative intrabar stop/take handling (OHLC only):
   - If stop and take are both reachable within the same bar, we assume the WORST outcome.
   - Stop gaps are filled at open if open crosses the stop (worse).
   - Take gaps do NOT give positive slippage: fill at take price (conservative).

3) Costs:
   - Spread + slippage modeled as a bps adjustment of fill price (worse fills).
   - Fees modeled separately on notional at executed fill price.

4) Funding (perps):
   - Funding is applied at timestamps that are multiples of funding_period_ms.
   - Applied BEFORE open execution on that bar, based on position held from previous bar.
   - Notional proxy uses previous bar close (OHLC-only constraint).

Important limitation (for now)
------------------------------
- Liquidation / maintenance margin is not modeled yet.
  We *do* enforce a max margin usage cap per trade, but liquidation logic will come later.

Interface
---------
- backtest_target_position(series, target_pos_by_close, config) -> BacktestResult

Where:
- series is Core_Types.OhlcvSeries (validated upstream)
- target_pos_by_close is a sequence of length N with values in {-1,0,+1}
  representing the desired position at each bar CLOSE.
  Execution uses target_pos_by_close[i-1] at bar i OPEN.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, Side, ValidationError, require


__all__ = [
    "ExitReason",
    "BacktestConfig",
    "EquityPoint",
    "Trade",
    "BacktestResult",
    "apply_costs",
    "backtest_target_position",
]


# -----------------------------
# Enums / data contracts
# -----------------------------
class ExitReason(str, Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    END_OF_DATA = "END_OF_DATA"
    # placeholders for future:
    TIME_STOP = "TIME_STOP"
    LIQUIDATION = "LIQUIDATION"


FundingRateFn = Callable[[int], float]


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    # Account / sizing
    initial_equity: float = 1000.0
    leverage: float = 10.0

    # Costs
    fee_rate_taker: float = 0.0004  # 4 bps
    spread_bps: float = 2.0         # total spread in bps (we apply half each side)
    slippage_bps: float = 1.0       # extra bps per fill

    # Risk & margin constraints
    risk_per_trade_fraction: float = 0.005  # e.g. 0.005 = 0.5% equity risk per trade (before margin cap)
    max_margin_fraction: float = 0.02       # e.g. 0.02 = 2% equity margin cap

    # Default protective exits (can be made strategy-specific later)
    stop_bps: float = 30.0                  # e.g. 30 bps = 0.30% stop distance from entry
    take_bps: Optional[float] = None        # e.g. 60 bps = 0.60% take profit (None disables)

    # Funding
    funding_period_ms: int = 28_800_000     # 8h
    funding_rate_per_period: float = 0.0    # constant, used if funding_rate_fn is None
    funding_rate_fn: Optional[FundingRateFn] = None

    # Storage controls (miner can disable for speed)
    store_trades: bool = True
    store_equity_curve: bool = True

    # End behavior
    close_on_end: bool = True

    def __post_init__(self) -> None:
        # Validate key invariants early (fail fast).
        require(_finite(self.initial_equity) and self.initial_equity > 0.0, "initial_equity must be finite and > 0")
        require(_finite(self.leverage) and self.leverage > 0.0, "leverage must be finite and > 0")

        require(_finite(self.fee_rate_taker) and self.fee_rate_taker >= 0.0, "fee_rate_taker must be >= 0")
        require(_finite(self.spread_bps) and self.spread_bps >= 0.0, "spread_bps must be >= 0")
        require(_finite(self.slippage_bps) and self.slippage_bps >= 0.0, "slippage_bps must be >= 0")
        require((self.spread_bps * 0.5 + self.slippage_bps) < 10_000.0, "total cost bps must be < 10000")

        require(
            _finite(self.risk_per_trade_fraction) and 0.0 <= self.risk_per_trade_fraction <= 1.0,
            "risk_per_trade_fraction must be in [0,1]",
        )
        require(
            _finite(self.max_margin_fraction) and 0.0 <= self.max_margin_fraction <= 1.0,
            "max_margin_fraction must be in [0,1]",
        )

        require(_finite(self.stop_bps) and self.stop_bps > 0.0, "stop_bps must be > 0")
        if self.take_bps is not None:
            require(_finite(self.take_bps) and self.take_bps > 0.0, "take_bps must be None or > 0")

        require(isinstance(self.funding_period_ms, int) and self.funding_period_ms > 0, "funding_period_ms must be int > 0")
        require(_finite(self.funding_rate_per_period), "funding_rate_per_period must be finite")


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts_ms: int
    equity_cash: float     # realized equity (fees/funding/pnl realized)
    equity_mark: float     # conservative mark-to-market equity (estimated close-out value)


@dataclass(frozen=True, slots=True)
class Trade:
    side: Side
    qty: float

    entry_ts: int
    entry_price: float  # executed (includes spread/slippage modeling)
    exit_ts: int
    exit_price: float   # executed (includes spread/slippage modeling)

    entry_fee: float
    exit_fee: float
    funding_pnl: float

    pnl_price: float    # price move pnl using executed prices (slippage/spread already embedded)
    pnl_net: float      # pnl_price + funding_pnl - entry_fee - exit_fee

    reason: ExitReason


@dataclass(frozen=True, slots=True)
class BacktestResult:
    trades: List[Trade]
    equity_curve: List[EquityPoint]

    final_equity_cash: float
    final_equity_mark: float

    max_drawdown: float
    trade_count: int
    winrate: float
    profit_factor: float

    total_fees: float
    total_funding: float


# -----------------------------
# Internal position state
# -----------------------------
@dataclass(slots=True)
class _PositionState:
    side: Side
    qty: float

    entry_ts: int
    entry_price: float

    stop_price: Optional[float]
    take_price: Optional[float]

    entry_fee: float
    funding_accum: float

    notional: float
    margin_used: float


# -----------------------------
# Helpers
# -----------------------------
def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _bps_to_frac(bps: float) -> float:
    return float(bps) / 10_000.0


def _is_funding_time(ts_ms: int, period_ms: int) -> bool:
    # Funding times for 8h (28,800,000ms) align with epoch boundaries.
    return (int(ts_ms) % int(period_ms)) == 0


def _funding_rate(ts_ms: int, cfg: BacktestConfig) -> float:
    if cfg.funding_rate_fn is not None:
        r = float(cfg.funding_rate_fn(int(ts_ms)))
        require(_finite(r), "funding_rate_fn returned non-finite value")
        return r
    return float(cfg.funding_rate_per_period)


# -----------------------------
# Cost model
# -----------------------------
def apply_costs(price: float, *, is_buy: bool, cfg: BacktestConfig) -> float:
    """
    Apply spread/slippage to the executed fill price.

    - Buys get worse (higher) price.
    - Sells get worse (lower) price.
    """
    p = float(price)
    require(_finite(p) and p > 0.0, f"price must be finite and > 0, got {price!r}")

    cost_bps = (cfg.spread_bps * 0.5) + cfg.slippage_bps
    frac = _bps_to_frac(cost_bps)

    if is_buy:
        return p * (1.0 + frac)
    return p * (1.0 - frac)


def _fee(notional: float, cfg: BacktestConfig) -> float:
    n = float(notional)
    require(_finite(n) and n >= 0.0, "notional must be finite and >= 0")
    return n * float(cfg.fee_rate_taker)


# -----------------------------
# Sizing
# -----------------------------
def _compute_qty(equity_cash: float, entry_exec_price: float, cfg: BacktestConfig) -> float:
    """
    Risk-based sizing with a margin usage cap.

    qty_risk  = (equity * risk_fraction) / stop_distance
    stop_dist = entry_price * stop_frac

    qty_cap   = (equity * max_margin_fraction * leverage) / entry_price

    qty = min(qty_risk, qty_cap)
    """
    eq = float(equity_cash)
    require(_finite(eq) and eq > 0.0, "equity_cash must be finite and > 0")

    ep = float(entry_exec_price)
    require(_finite(ep) and ep > 0.0, "entry_exec_price must be finite and > 0")

    stop_frac = _bps_to_frac(cfg.stop_bps)
    require(stop_frac > 0.0, "stop_frac must be > 0")

    risk_usdt = eq * float(cfg.risk_per_trade_fraction)
    # If risk is 0, we allow qty=0 (engine will skip opening).
    if risk_usdt <= 0.0:
        return 0.0

    stop_dist = ep * stop_frac
    require(stop_dist > 0.0, "stop_dist must be > 0")
    qty_risk = risk_usdt / stop_dist

    # Margin cap
    if cfg.max_margin_fraction <= 0.0:
        qty_cap = 0.0
    else:
        margin_cap = eq * float(cfg.max_margin_fraction)
        notional_cap = margin_cap * float(cfg.leverage)
        qty_cap = notional_cap / ep if notional_cap > 0.0 else 0.0

    qty = min(qty_risk, qty_cap) if qty_cap > 0.0 else 0.0
    # Prevent negative/NaN
    if not _finite(qty) or qty < 0.0:
        return 0.0
    return qty


# -----------------------------
# Trade open/close
# -----------------------------
def _open_position(
    equity_cash: float,
    *,
    side: Side,
    entry_ts: int,
    entry_price_raw: float,
    cfg: BacktestConfig,
) -> Tuple[float, Optional[_PositionState], float]:
    """
    Open a position at entry_price_raw (market), executed with costs.

    Returns:
      (new_equity_cash, position_or_none, entry_fee)
    """
    is_buy = (side == Side.LONG)  # long entry buys, short entry sells
    entry_exec = apply_costs(entry_price_raw, is_buy=is_buy, cfg=cfg)

    qty = _compute_qty(equity_cash, entry_exec, cfg)
    if qty <= 0.0:
        # Skip opening if sizing yields zero (e.g. risk=0 or margin cap=0)
        return equity_cash, None, 0.0

    notional = abs(qty) * entry_exec
    entry_fee = _fee(notional, cfg)

    new_equity = float(equity_cash) - float(entry_fee)
    require(_finite(new_equity), "equity became non-finite after entry fee")
    # We allow equity to go slightly negative in edge cases, but this is usually a config problem.

    stop_frac = _bps_to_frac(cfg.stop_bps)
    take_frac = _bps_to_frac(cfg.take_bps) if cfg.take_bps is not None else None

    if side == Side.LONG:
        stop_price = entry_exec * (1.0 - stop_frac)
        take_price = (entry_exec * (1.0 + take_frac)) if take_frac is not None else None
    else:
        stop_price = entry_exec * (1.0 + stop_frac)
        take_price = (entry_exec * (1.0 - take_frac)) if take_frac is not None else None

    margin_used = notional / float(cfg.leverage)

    pos = _PositionState(
        side=side,
        qty=float(qty),
        entry_ts=int(entry_ts),
        entry_price=float(entry_exec),
        stop_price=float(stop_price) if stop_price is not None else None,
        take_price=float(take_price) if take_price is not None else None,
        entry_fee=float(entry_fee),
        funding_accum=0.0,
        notional=float(notional),
        margin_used=float(margin_used),
    )
    return new_equity, pos, float(entry_fee)


def _close_position(
    equity_cash: float,
    *,
    pos: _PositionState,
    exit_ts: int,
    exit_price_raw: float,
    reason: ExitReason,
    cfg: BacktestConfig,
) -> Tuple[float, Trade, float]:
    """
    Close an existing position (market), executed with costs.

    Returns:
      (new_equity_cash, trade, exit_fee)
    """
    closing_is_buy = (pos.side == Side.SHORT)  # closing short buys, closing long sells
    exit_exec = apply_costs(exit_price_raw, is_buy=closing_is_buy, cfg=cfg)

    notional = abs(pos.qty) * exit_exec
    exit_fee = _fee(notional, cfg)

    if pos.side == Side.LONG:
        pnl_price = (exit_exec - pos.entry_price) * pos.qty
    else:
        pnl_price = (pos.entry_price - exit_exec) * pos.qty

    pnl_net = pnl_price + pos.funding_accum - pos.entry_fee - exit_fee

    new_equity = float(equity_cash) + float(pnl_price) - float(exit_fee)

    trade = Trade(
        side=pos.side,
        qty=float(pos.qty),
        entry_ts=int(pos.entry_ts),
        entry_price=float(pos.entry_price),
        exit_ts=int(exit_ts),
        exit_price=float(exit_exec),
        entry_fee=float(pos.entry_fee),
        exit_fee=float(exit_fee),
        funding_pnl=float(pos.funding_accum),
        pnl_price=float(pnl_price),
        pnl_net=float(pnl_net),
        reason=reason,
    )
    return new_equity, trade, float(exit_fee)


def _check_intrabar_exit(pos: _PositionState, *, o: float, h: float, l: float) -> Optional[Tuple[ExitReason, float]]:
    """
    Determine whether stop/take triggers within the current bar.
    Returns (reason, exit_price_raw) if triggers, else None.

    Conservative rules:
    - If both stop and take are possible, choose STOP (worst).
    - Stop gap: if open crossed beyond stop in the bad direction, exit at open (worse).
    - Take gap: never give positive slippage; exit at take price.
    """
    stop = pos.stop_price
    take = pos.take_price

    if stop is None and take is None:
        return None

    o_f, h_f, l_f = float(o), float(h), float(l)

    if pos.side == Side.LONG:
        stop_hit = (stop is not None) and (l_f <= float(stop))
        take_hit = (take is not None) and (h_f >= float(take))

        if stop_hit and take_hit:
            # Worst-case: stop
            st = float(stop)  # type: ignore[arg-type]
            exit_raw = o_f if o_f <= st else st
            return ExitReason.STOP_LOSS, exit_raw

        if stop_hit:
            st = float(stop)  # type: ignore[arg-type]
            exit_raw = o_f if o_f <= st else st
            return ExitReason.STOP_LOSS, exit_raw

        if take_hit:
            tk = float(take)  # type: ignore[arg-type]
            return ExitReason.TAKE_PROFIT, tk

        return None

    # SHORT
    stop_hit = (stop is not None) and (h_f >= float(stop))
    take_hit = (take is not None) and (l_f <= float(take))

    if stop_hit and take_hit:
        st = float(stop)  # type: ignore[arg-type]
        exit_raw = o_f if o_f >= st else st
        return ExitReason.STOP_LOSS, exit_raw

    if stop_hit:
        st = float(stop)  # type: ignore[arg-type]
        exit_raw = o_f if o_f >= st else st
        return ExitReason.STOP_LOSS, exit_raw

    if take_hit:
        tk = float(take)  # type: ignore[arg-type]
        return ExitReason.TAKE_PROFIT, tk

    return None


def _equity_mark_to_market(equity_cash: float, pos: _PositionState, *, mark_price: float, cfg: BacktestConfig) -> float:
    """
    Conservative mark-to-market:
    equity_cash + pnl_if_close_now(mark_price) - estimated exit fee
    Uses the same cost model for a hypothetical close at mark_price.
    """
    closing_is_buy = (pos.side == Side.SHORT)
    exit_exec = apply_costs(mark_price, is_buy=closing_is_buy, cfg=cfg)
    exit_fee = _fee(abs(pos.qty) * exit_exec, cfg)

    if pos.side == Side.LONG:
        pnl_price = (exit_exec - pos.entry_price) * pos.qty
    else:
        pnl_price = (pos.entry_price - exit_exec) * pos.qty

    return float(equity_cash) + float(pnl_price) - float(exit_fee)


# -----------------------------
# Backtest core
# -----------------------------
def backtest_target_position(
    series: OhlcvSeries,
    target_pos_by_close: Sequence[int],
    cfg: BacktestConfig,
) -> BacktestResult:
    """
    Backtest a target position series.

    target_pos_by_close:
      length N, values in {-1,0,+1}
      meaning desired position at bar close i
      execution uses desired = target_pos_by_close[i-1] at bar open i.

    Returns:
      BacktestResult with trades and equity curve depending on cfg storage flags.
    """
    n = len(series.ts_ms)
    require(n > 0, "series must not be empty")
    require(len(target_pos_by_close) == n, f"target_pos_by_close length must match series length ({n})")

    # Local refs for speed
    ts_ms = series.ts_ms
    o = series.open
    h = series.high
    l = series.low
    c = series.close

    equity_cash = float(cfg.initial_equity)
    pos: Optional[_PositionState] = None

    trades: List[Trade] = []
    equity_curve: List[EquityPoint] = []

    total_fees = 0.0
    total_funding = 0.0

    for i in range(n):
        t = int(ts_ms[i])
        o_i = float(o[i])
        h_i = float(h[i])
        l_i = float(l[i])
        c_i = float(c[i])

        # Funding at bar open, based on position from previous bar (avoid ordering ambiguity).
        if i > 0 and _is_funding_time(t, cfg.funding_period_ms) and pos is not None:
            rate = _funding_rate(t, cfg)
            # Use previous close as notional proxy (OHLC-only).
            notional_ref = abs(pos.qty) * float(c[i - 1])
            side_sign = 1.0 if pos.side == Side.LONG else -1.0
            funding_pnl = -float(rate) * float(notional_ref) * float(side_sign)

            equity_cash += funding_pnl
            pos.funding_accum += funding_pnl
            total_funding += funding_pnl

        # Determine desired position for this open (based on previous close).
        desired = 0
        if i > 0:
            try:
                desired = int(target_pos_by_close[i - 1])
            except Exception as e:
                raise ValidationError(f"target_pos_by_close[{i-1}] not int-castable: {target_pos_by_close[i-1]!r}") from e

            require(desired in (-1, 0, 1), f"target_pos_by_close[{i-1}] must be in {{-1,0,1}}, got {desired}")

        current = 0
        if pos is not None:
            current = 1 if pos.side == Side.LONG else -1

        # If desired differs, close/open at this bar open (market), no lookahead.
        if desired != current:
            if pos is not None:
                equity_cash, tr, exit_fee = _close_position(
                    equity_cash,
                    pos=pos,
                    exit_ts=t,
                    exit_price_raw=o_i,
                    reason=ExitReason.SIGNAL,
                    cfg=cfg,
                )
                pos = None

                total_fees += tr.entry_fee + tr.exit_fee
                if cfg.store_trades:
                    trades.append(tr)

            if desired != 0:
                new_side = Side.LONG if desired == 1 else Side.SHORT
                equity_cash, new_pos, entry_fee = _open_position(
                    equity_cash,
                    side=new_side,
                    entry_ts=t,
                    entry_price_raw=o_i,
                    cfg=cfg,
                )
                if new_pos is not None:
                    pos = new_pos
                    total_fees += entry_fee

        # Intrabar protective exits (stop/take)
        if pos is not None:
            trig = _check_intrabar_exit(pos, o=o_i, h=h_i, l=l_i)
            if trig is not None:
                reason, exit_raw = trig
                equity_cash, tr, exit_fee = _close_position(
                    equity_cash,
                    pos=pos,
                    exit_ts=t,
                    exit_price_raw=float(exit_raw),
                    reason=reason,
                    cfg=cfg,
                )
                pos = None

                total_fees += tr.entry_fee + tr.exit_fee
                if cfg.store_trades:
                    trades.append(tr)

        # Equity curve point at bar end (conservative mark-to-market)
        if cfg.store_equity_curve:
            if pos is None:
                eq_mark = equity_cash
            else:
                eq_mark = _equity_mark_to_market(equity_cash, pos, mark_price=c_i, cfg=cfg)

            equity_curve.append(
                EquityPoint(
                    ts_ms=t,
                    equity_cash=float(equity_cash),
                    equity_mark=float(eq_mark),
                )
            )

    # Force close at end-of-data (at last close)
    if cfg.close_on_end and pos is not None:
        t_last = int(ts_ms[-1])
        c_last = float(c[-1])

        equity_cash, tr, exit_fee = _close_position(
            equity_cash,
            pos=pos,
            exit_ts=t_last,
            exit_price_raw=c_last,
            reason=ExitReason.END_OF_DATA,
            cfg=cfg,
        )
        pos = None

        total_fees += tr.entry_fee + tr.exit_fee
        if cfg.store_trades:
            trades.append(tr)

        if cfg.store_equity_curve:
            # Append a final point reflecting flat state after forced close
            equity_curve.append(
                EquityPoint(
                    ts_ms=t_last,
                    equity_cash=float(equity_cash),
                    equity_mark=float(equity_cash),
                )
            )

    # Final equity
    final_equity_cash = float(equity_cash)
    final_equity_mark = float(final_equity_cash)

    # Stats
    max_dd = _max_drawdown([p.equity_mark for p in equity_curve]) if cfg.store_equity_curve else 0.0

    if cfg.store_trades:
        trade_count = len(trades)
        wins = sum(1 for tr in trades if tr.pnl_net > 0.0)
        winrate = (wins / trade_count) if trade_count > 0 else 0.0

        gross_win = sum(tr.pnl_net for tr in trades if tr.pnl_net > 0.0)
        gross_loss = sum(tr.pnl_net for tr in trades if tr.pnl_net < 0.0)  # negative
        if gross_loss < 0.0:
            profit_factor = gross_win / abs(gross_loss) if gross_win > 0.0 else 0.0
        else:
            profit_factor = float("inf") if gross_win > 0.0 else 0.0
    else:
        trade_count = 0
        winrate = 0.0
        profit_factor = 0.0

    return BacktestResult(
        trades=trades if cfg.store_trades else [],
        equity_curve=equity_curve if cfg.store_equity_curve else [],
        final_equity_cash=final_equity_cash,
        final_equity_mark=final_equity_mark,
        max_drawdown=float(max_dd),
        trade_count=int(trade_count),
        winrate=float(winrate),
        profit_factor=float(profit_factor),
        total_fees=float(total_fees),
        total_funding=float(total_funding),
    )


def _max_drawdown(values: Sequence[float]) -> float:
    """
    Max drawdown as a fraction (0.25 = 25%).
    Uses peak-to-trough on the provided equity series.
    """
    if not values:
        return 0.0
    peak = float(values[0])
    require(_finite(peak) and peak > 0.0, "equity series must start with finite > 0")

    max_dd = 0.0
    for v in values:
        x = float(v)
        if not _finite(x):
            raise ValidationError("equity series contains non-finite values")
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak > 0.0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


# -----------------------------
# Minimal self-test
# -----------------------------
def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _self_test() -> None:
    # ---------- Test 1: simple long profit ----------
    series1 = OhlcvSeries(
        ts_ms=[1, 2, 3, 4],
        open=[100, 100, 110, 120],
        high=[100.2, 110.2, 120.2, 120.2],
        low=[99.8, 99.8, 109.8, 119.8],
        close=[100, 110, 120, 120],
        volume=[1, 1, 1, 1],
        symbol="TEST",
        timeframe="1m",
    )
    # Desired long from bar0 close => open at bar1 open; close at bar2 close => executed at bar3 open.
    target1 = [1, 1, 0, 0]

    cfg1 = BacktestConfig(
        initial_equity=1000.0,
        leverage=1.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=0.01,  # $10 risk
        max_margin_fraction=1.0,
        stop_bps=100.0,  # 1% stop => stop dist $1 @ 100
        take_bps=None,
        funding_rate_per_period=0.0,
        store_trades=True,
        store_equity_curve=True,
        close_on_end=True,
    )

    res1 = backtest_target_position(series1, target1, cfg1)
    _expect(res1.trade_count == 1, "Test1: expected 1 trade")
    _expect(res1.trades[0].reason == ExitReason.SIGNAL, "Test1: expected SIGNAL exit")
    # Entry at 100, qty = 10, exit at 120 => pnl 200
    _expect(abs(res1.final_equity_cash - 1200.0) < 1e-9, f"Test1: final equity mismatch {res1.final_equity_cash}")

    # ---------- Test 2: stop loss ----------
    series2 = OhlcvSeries(
        ts_ms=[1, 2],
        open=[100, 100],
        high=[100.5, 100.5],
        low=[99.0, 99.0],
        close=[100, 100],
        volume=[1, 1],
        symbol="TEST",
        timeframe="1m",
    )
    target2 = [1, 1]  # open long at bar1 open

    cfg2 = BacktestConfig(
        initial_equity=1000.0,
        leverage=1.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=0.01,  # $10 risk
        max_margin_fraction=1.0,
        stop_bps=100.0,  # stop at 99
        take_bps=None,
        funding_rate_per_period=0.0,
        store_trades=True,
        store_equity_curve=False,
        close_on_end=True,
    )

    res2 = backtest_target_position(series2, target2, cfg2)
    _expect(res2.trade_count == 1, "Test2: expected 1 trade")
    _expect(res2.trades[0].reason == ExitReason.STOP_LOSS, "Test2: expected STOP_LOSS")
    _expect(abs(res2.final_equity_cash - 990.0) < 1e-9, f"Test2: final equity mismatch {res2.final_equity_cash}")

    # ---------- Test 3: intrabar ambiguity => worst-case STOP ----------
    series3 = OhlcvSeries(
        ts_ms=[1, 2],
        open=[100, 100],
        high=[100.0, 101.0],  # hits take
        low=[100.0, 98.0],    # hits stop
        close=[100, 100],
        volume=[1, 1],
        symbol="TEST",
        timeframe="1m",
    )
    target3 = [1, 1]  # open at bar1 open

    cfg3 = BacktestConfig(
        initial_equity=1000.0,
        leverage=1.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=0.01,
        max_margin_fraction=1.0,
        stop_bps=100.0,   # stop at 99
        take_bps=50.0,    # take at 100.5
        funding_rate_per_period=0.0,
        store_trades=True,
        store_equity_curve=False,
        close_on_end=True,
    )

    res3 = backtest_target_position(series3, target3, cfg3)
    _expect(res3.trade_count == 1, "Test3: expected 1 trade")
    _expect(res3.trades[0].reason == ExitReason.STOP_LOSS, "Test3: expected STOP_LOSS worst-case")

    # ---------- Test 4: funding applied ----------
    series4 = OhlcvSeries(
        ts_ms=[28_798_000, 28_799_000, 28_800_000],  # funding triggers at last bar open
        open=[100, 100, 100],
        high=[100, 100, 100],
        low=[100, 100, 100],
        close=[100, 100, 100],
        volume=[1, 1, 1],
        symbol="TEST",
        timeframe="1m",
    )
    target4 = [1, 1, 1]  # open at bar1 open, hold through funding at bar2 open

    cfg4 = BacktestConfig(
        initial_equity=1000.0,
        leverage=10.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=0.01,
        max_margin_fraction=1.0,
        stop_bps=100.0,
        take_bps=None,
        funding_period_ms=28_800_000,
        funding_rate_per_period=0.01,  # +1% -> longs pay
        store_trades=True,
        store_equity_curve=False,
        close_on_end=True,
    )

    res4 = backtest_target_position(series4, target4, cfg4)
    _expect(abs(res4.total_funding - (-10.0)) < 1e-9, f"Test4: funding mismatch {res4.total_funding}")
    _expect(abs(res4.final_equity_cash - 990.0) < 1e-9, f"Test4: final equity mismatch {res4.final_equity_cash}")

    print("Backtest_Engine self-test: OK")


if __name__ == "__main__":
    _self_test()
