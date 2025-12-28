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

5) Liquidation (perps, isolated approximation):
   - Maintenance margin triggers a forced liquidation when price crosses a computed liquidation level.
   - Liquidation price uses a simple isolated-margin approximation (no tiered MM, no cross effects).
   - We treat liquidation as a hard, worst-case boundary under OHLC intrabar ambiguity.

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, Side, ValidationError, require


__all__ = [
    "ExitReason",
    "BacktestConfig",
    "EquityPoint",
    "Trade",
    "BacktestResult",
    "apply_costs",
    "backtest_target_position",
    "run_backtest",
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
    # Liquidation / maintenance margin (perps, isolated approximation)
    # NOTE: this is intentionally simple and conservative.
    # - maintenance_margin_rate: fraction of notional reserved as maintenance (e.g. 0.005 = 0.5%); set <=0 to disable liquidation modeling
    # - liquidation_fee_rate: additional fee charged on notional upon liquidation (exchange-dependent)
    # - set maintenance_margin_rate=0.0 to disable liquidation modeling entirely
    maintenance_margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.0
    # Costs
    fee_rate_taker: float = 0.0004  # 4 bps
    spread_bps: float = 2.0         # total spread in bps (we apply half each side)
    slippage_bps: float = 1.0       # extra bps per fill
    # Intrabar adverse fills (STOP/LIQ): fraction of overshoot to the bar extreme applied to trigger-level exits.
    # 0.0 = fill at trigger level (legacy); 1.0 = fill at bar extreme (very conservative).
    adverse_fill_slip_mult: float = 0.0
    # Risk & margin constraints
    risk_per_trade_fraction: float = 0.005  # e.g. 0.005 = 0.5% equity risk per trade (before margin cap)
    max_margin_fraction: float = 0.02       # e.g. 0.02 = 2% equity margin cap
    # Exchange constraints (real-world): prevent unrealistic micro orders and enforce lot sizes.
    # - min_notional_usdt: minimum order notional at entry (0 disables)
    # - min_qty: minimum base-asset quantity (0 disables)
    # - qty_step: quantity increment (0 disables); quantity is floored to step to avoid exceeding risk.
    min_notional_usdt: float = 10.0
    min_qty: float = 0.0
    qty_step: float = 0.0
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
        require(
            _finite(self.maintenance_margin_rate) and 0.0 <= self.maintenance_margin_rate < 1.0,
            "maintenance_margin_rate must be finite and in [0,1)",
        )
        require(
            _finite(self.liquidation_fee_rate) and 0.0 <= self.liquidation_fee_rate < 0.25,
            "liquidation_fee_rate must be finite and in [0,0.25)",
        )
        require(
            float(self.maintenance_margin_rate) < (1.0 / float(self.leverage)),
            "maintenance_margin_rate must be < 1/leverage (else immediate liquidation)",
        )
        require(_finite(self.fee_rate_taker) and self.fee_rate_taker >= 0.0, "fee_rate_taker must be >= 0")
        require(_finite(self.spread_bps) and self.spread_bps >= 0.0, "spread_bps must be >= 0")
        require(_finite(self.slippage_bps) and self.slippage_bps >= 0.0, "slippage_bps must be >= 0")
        require(
            _finite(self.adverse_fill_slip_mult) and 0.0 <= self.adverse_fill_slip_mult <= 1.0,
            "adverse_fill_slip_mult must be in [0,1]",
        )
        require((self.spread_bps * 0.5 + self.slippage_bps) < 10_000.0, "total cost bps must be < 10000")

        require(
            _finite(self.risk_per_trade_fraction) and 0.0 <= self.risk_per_trade_fraction <= 1.0,
            "risk_per_trade_fraction must be in [0,1]",
        )
        require(
            _finite(self.max_margin_fraction) and 0.0 <= self.max_margin_fraction <= 1.0,
            "max_margin_fraction must be in [0,1]",
        )
        require(_finite(self.min_notional_usdt) and self.min_notional_usdt >= 0.0, "min_notional_usdt must be >= 0")
        require(_finite(self.min_qty) and self.min_qty >= 0.0, "min_qty must be >= 0")
        require(_finite(self.qty_step) and self.qty_step >= 0.0, "qty_step must be >= 0")

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
    liquidation_fee: float
    funding_pnl: float

    pnl_price: float    # price move pnl using executed prices (slippage/spread already embedded)
    pnl_net: float      # pnl_price + funding_pnl - entry_fee - exit_fee - liquidation_fee

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
    total_liquidation_fees: float
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
    liquidation_price: Optional[float]

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
def _liquidation_price(entry_exec_price: float, side: Side, cfg: BacktestConfig) -> Optional[float]:
    """Approximate isolated liquidation price (very simplified).

    Model (constant maintenance margin rate):
      long  liq ~= entry * (1 - 1/leverage + mmr)
      short liq ~= entry * (1 + 1/leverage - mmr)

    If maintenance_margin_rate <= 0, liquidation modeling is disabled.
    """
    ep = float(entry_exec_price)
    if not _finite(ep) or ep <= 0.0:
        return None

    mmr = float(getattr(cfg, "maintenance_margin_rate", 0.0) or 0.0)
    if not _finite(mmr) or mmr <= 0.0:
        return None

    lev = float(cfg.leverage)
    if not _finite(lev) or lev <= 0.0:
        return None

    inv_lev = 1.0 / lev
    # If mmr >= 1/leverage, liquidation would be at-or-better-than entry (nonsense).
    if (inv_lev - mmr) <= 0.0:
        return None

    if side == Side.LONG:
        liq = ep * (1.0 - inv_lev + mmr)
    else:
        liq = ep * (1.0 + inv_lev - mmr)

    if not _finite(liq) or liq <= 0.0:
        return None
    return float(liq)

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
def _floor_to_step(x: float, step: float) -> float:
    """
    Floor |x| to a positive step size (lot size) in a numerically stable way.
    We always floor (never round up) to avoid overstating exposure/risk.
    """
    s = float(step)
    v = float(x)
    if not _finite(v):
        return 0.0
    if not _finite(s) or s <= 0.0:
        return float(v)

    sign = -1.0 if v < 0.0 else 1.0
    q = abs(v)
    # Add a tiny epsilon to counter floating error that would otherwise floor one step too low.
    steps = math.floor((q / s) + 1e-12)
    out = float(steps) * s
    if not _finite(out) or out < 0.0:
        return 0.0
    return sign * float(out)


def _apply_entry_constraints(qty: float, entry_exec_price: float, cfg: BacktestConfig) -> float:
    """
    Apply exchange-like constraints to an intended *entry* quantity:
      - floor to qty_step (lot size)
      - enforce min_qty
      - enforce min_notional_usdt

    Returns adjusted qty (0.0 => skip trade).
    """
    q = float(qty)
    ep = float(entry_exec_price)
    if not _finite(q) or q <= 0.0:
        return 0.0
    if not _finite(ep) or ep <= 0.0:
        return 0.0

    step = float(getattr(cfg, "qty_step", 0.0) or 0.0)
    if _finite(step) and step > 0.0:
        q = float(_floor_to_step(q, step))
        if not _finite(q) or q <= 0.0:
            return 0.0

    min_q = float(getattr(cfg, "min_qty", 0.0) or 0.0)
    if _finite(min_q) and min_q > 0.0 and q < min_q:
        return 0.0

    min_notional = float(getattr(cfg, "min_notional_usdt", 0.0) or 0.0)
    if _finite(min_notional) and min_notional > 0.0:
        if (q * ep) < min_notional:
            return 0.0

    return float(q)

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
    qty = _apply_entry_constraints(qty, entry_exec, cfg)
    if qty <= 0.0:
        # Skip opening if sizing yields zero / below exchange constraints
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
    liq_price = _liquidation_price(entry_exec, side, cfg)
    margin_used = notional / float(cfg.leverage)

    pos = _PositionState(
        side=side,
        qty=float(qty),
        entry_ts=int(entry_ts),
        entry_price=float(entry_exec),
        stop_price=float(stop_price) if stop_price is not None else None,
        take_price=float(take_price) if take_price is not None else None,
        liquidation_price=float(liq_price) if liq_price is not None else None,
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
 
    liquidation_fee = 0.0
    if reason == ExitReason.LIQUIDATION and float(cfg.liquidation_fee_rate) > 0.0:
        liquidation_fee = float(notional) * float(cfg.liquidation_fee_rate)
        if not _finite(liquidation_fee) or liquidation_fee < 0.0:
            liquidation_fee = 0.0
    if pos.side == Side.LONG:
        pnl_price = (exit_exec - pos.entry_price) * pos.qty
    else:
        pnl_price = (pos.entry_price - exit_exec) * pos.qty

    pnl_net = pnl_price + pos.funding_accum - pos.entry_fee - exit_fee - liquidation_fee

    new_equity = float(equity_cash) + float(pnl_price) - float(exit_fee) - float(liquidation_fee)

    trade = Trade(
        side=pos.side,
        qty=float(pos.qty),
        entry_ts=int(pos.entry_ts),
        entry_price=float(pos.entry_price),
        exit_ts=int(exit_ts),
        exit_price=float(exit_exec),
        entry_fee=float(pos.entry_fee),
        exit_fee=float(exit_fee),
        liquidation_fee=float(liquidation_fee),
        funding_pnl=float(pos.funding_accum),
        pnl_price=float(pnl_price),
        pnl_net=float(pnl_net),
        reason=reason,
    )
    return new_equity, trade, float(exit_fee + liquidation_fee)


def _liquidation_price_from_pos(pos: _PositionState, cfg: BacktestConfig) -> Optional[float]:
    """Compatibility helper: compute liquidation price from an existing position.

    IMPORTANT: do NOT name this `_liquidation_price` (would shadow the primary function).
    """
    return _liquidation_price(float(pos.entry_price), pos.side, cfg)


def _check_intrabar_exit(
    pos: _PositionState, *, o: float, h: float, l: float, cfg: BacktestConfig
) -> Optional[Tuple[ExitReason, float]]:
    """Determine whether stop/take/liquidation triggers within the current bar (OHLC only).

    Conservative rules:
    - No positive slippage on take-profit: exit at TP level.
    - For adverse exits (STOP/LIQ), if OPEN has already gapped beyond the trigger in the bad
      direction, exit at OPEN (worse).
    - If both an adverse exit (STOP/LIQ) and a favorable exit (TP) are possible within the same
      bar, assume the adverse one happens first (worst-case).
    - Liquidation is treated as an adverse trigger; however, if the stop is closer than the
      liquidation boundary, the stop protects (it triggers first).
    """
    # cfg kept for call-site stability (even if unused here)
    stop = pos.stop_price
    take = pos.take_price
    liq = pos.liquidation_price
    if stop is None and take is None and liq is None:
        return None

    o_f = float(o)
    h_f = float(h)
    l_f = float(l)

    if pos.side == Side.LONG:
        # Downward move is adverse; higher (closer) level triggers first.
        adverse_reason: Optional[ExitReason] = None
        adverse_level: Optional[float] = None
        if stop is not None and liq is not None:
            st = float(stop)
            lq = float(liq)
            if st >= lq:
                adverse_reason, adverse_level = ExitReason.STOP_LOSS, st
            else:
                adverse_reason, adverse_level = ExitReason.LIQUIDATION, lq
        elif stop is not None:
            adverse_reason, adverse_level = ExitReason.STOP_LOSS, float(stop)
        elif liq is not None:
            adverse_reason, adverse_level = ExitReason.LIQUIDATION, float(liq)

        take_hit = (take is not None) and (h_f >= float(take))
        adverse_hit = (adverse_level is not None) and (l_f <= float(adverse_level))

        if adverse_hit:
            lvl = float(adverse_level)  # type: ignore[arg-type]
            # If OPEN already gapped beyond the adverse trigger, take the OPEN (worst).
            if o_f <= lvl:
                return adverse_reason or ExitReason.STOP_LOSS, o_f

            # Intrabar adverse slippage model (STOP only): if price overshoots beyond the trigger level,
            # assume the fill drifts some fraction toward the bar low.
            reason = adverse_reason or ExitReason.STOP_LOSS
            exit_raw = lvl
            if reason == ExitReason.STOP_LOSS:
                mult = float(getattr(cfg, "adverse_fill_slip_mult", 0.0) or 0.0)
                if _finite(mult) and mult > 0.0:
                    exit_raw = float(lvl) - float(mult) * (float(lvl) - float(l_f))

                    # If liquidation is modeled and we slip beyond the liquidation boundary,
                    # treat it as liquidation (stop failed to protect under fast moves).
                    if liq is not None:
                        lq = float(liq)
                        if _finite(lq) and float(exit_raw) <= float(lq):
                            reason = ExitReason.LIQUIDATION
                            exit_raw = float(lq)

            return reason, float(exit_raw)

        if take_hit:
            return ExitReason.TAKE_PROFIT, float(take)  # type: ignore[arg-type]
        return None

    # SHORT: upward move is adverse; lower (closer) level triggers first.
    adverse_reason = None
    adverse_level = None
    if stop is not None and liq is not None:
        st = float(stop)
        lq = float(liq)
        if st <= lq:
            adverse_reason, adverse_level = ExitReason.STOP_LOSS, st
        else:
            adverse_reason, adverse_level = ExitReason.LIQUIDATION, lq
    elif stop is not None:
        adverse_reason, adverse_level = ExitReason.STOP_LOSS, float(stop)
    elif liq is not None:
        adverse_reason, adverse_level = ExitReason.LIQUIDATION, float(liq)

    take_hit = (take is not None) and (l_f <= float(take))
    adverse_hit = (adverse_level is not None) and (h_f >= float(adverse_level))

    if adverse_hit:
        lvl = float(adverse_level)  # type: ignore[arg-type]
        # If OPEN already gapped beyond the adverse trigger, take the OPEN (worst).
        if o_f >= lvl:
            return adverse_reason or ExitReason.STOP_LOSS, o_f

        reason = adverse_reason or ExitReason.STOP_LOSS
        exit_raw = lvl
        if reason == ExitReason.STOP_LOSS:
            mult = float(getattr(cfg, "adverse_fill_slip_mult", 0.0) or 0.0)
            if _finite(mult) and mult > 0.0:
                exit_raw = float(lvl) + float(mult) * (float(h_f) - float(lvl))

                if liq is not None:
                    lq = float(liq)
                    if _finite(lq) and float(exit_raw) >= float(lq):
                        reason = ExitReason.LIQUIDATION
                        exit_raw = float(lq)

        return reason, float(exit_raw)

    if take_hit:
        return ExitReason.TAKE_PROFIT, float(take)  # type: ignore[arg-type]
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
    total_liquidation_fees = 0.0
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

                # entry_fee was already counted when the position was opened
                total_fees += float(tr.exit_fee) + float(tr.liquidation_fee)
                total_liquidation_fees += float(tr.liquidation_fee)

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
            trig = _check_intrabar_exit(pos, o=o_i, h=h_i, l=l_i, cfg=cfg)
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

                # entry_fee was already counted when the position was opened
                total_fees += float(tr.exit_fee) + float(tr.liquidation_fee)
                total_liquidation_fees += float(tr.liquidation_fee)

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

        # entry_fee was already counted when the position was opened
        total_fees += float(tr.exit_fee) + float(tr.liquidation_fee)
        total_liquidation_fees += float(tr.liquidation_fee)

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
        total_liquidation_fees=float(total_liquidation_fees),
        total_funding=float(total_funding),
    )

# Miner-compatible signal backtest (entry/exit arrays)
# ------------------------------------------------------

# Small caches to avoid recomputing ATR / structure pivots for the same series across many calls.
_HUB_CACHE: Dict[Tuple[int, int, int, int], Tuple[Any, Any]] = {}
_ATR_CACHE: Dict[Tuple[Tuple[int, int, int, int], int], List[float]] = {}
_PIVOT_CACHE: Dict[Tuple[Tuple[int, int, int, int], str, int, int], List[float]] = {}


def _series_key(series: OhlcvSeries) -> Tuple[int, int, int, int]:
    n = len(series.ts_ms)
    if n <= 0:
        return (id(series), 0, 0, 0)
    return (id(series), n, int(series.ts_ms[0]), int(series.ts_ms[-1]))


def _get_hubs(series: OhlcvSeries) -> Tuple[Any, Any]:
    """
    Lazily construct and cache FeatureStore + MarketStructureStore per series instance.
    Uses a key that includes id + length + first/last ts to reduce risk of id reuse.
    """
    key = _series_key(series)
    hubs = _HUB_CACHE.get(key)
    if hubs is None:
        # Local imports to keep Backtest_Engine usable even if feature modules are not needed.
        from Features_Indicators import FeatureStore
        from Features_MarketStructure import MarketStructureStore

        hubs = (FeatureStore(series), MarketStructureStore(series))
        _HUB_CACHE[key] = hubs
    return hubs


def _get_atr_arr(series: OhlcvSeries, period: int) -> List[float]:
    k = (_series_key(series), int(period))
    arr = _ATR_CACHE.get(k)
    if arr is None:
        ind, _ = _get_hubs(series)
        arr = ind.indicator("atr", period=int(period))
        _ATR_CACHE[k] = arr
    return arr


def _get_pivot_arr(series: OhlcvSeries, level: str, left: int, right: int) -> List[float]:
    k = (_series_key(series), str(level), int(left), int(right))
    arr = _PIVOT_CACHE.get(k)
    if arr is None:
        _, ms = _get_hubs(series)
        arr = ms.feature(str(level), left=int(left), right=int(right))
        _PIVOT_CACHE[k] = arr
    return arr


def _safe_pos_value(arr: Sequence[float], idx: int) -> Optional[float]:
    try:
        v = float(arr[idx])
    except Exception:
        return None
    if not _finite(v) or v <= 0.0:
        return None
    return v


def _compute_qty_for_stop_dist(equity_cash: float, entry_exec_price: float, stop_dist: float, cfg: BacktestConfig) -> float:
    """
    Risk-based sizing, but using a *dynamic* stop distance in price units.
    Mirrors _compute_qty() but replaces cfg.stop_bps with stop_dist.
    """
    eq = float(equity_cash)
    require(eq > 0.0 and _finite(eq), "equity_cash must be finite > 0")
    ep = float(entry_exec_price)
    require(ep > 0.0 and _finite(ep), "entry_exec_price must be finite > 0")
    sd = float(stop_dist)
    if not _finite(sd) or sd <= 0.0:
        return 0.0

    risk_usdt = eq * float(cfg.risk_per_trade_fraction)
    if not _finite(risk_usdt) or risk_usdt <= 0.0:
        return 0.0

    qty_risk = risk_usdt / sd

    # Margin cap
    margin_cap = eq * float(cfg.max_margin_fraction)
    if not _finite(margin_cap) or margin_cap <= 0.0:
        return 0.0

    notional_cap = margin_cap * float(cfg.leverage)
    qty_cap = notional_cap / ep if ep > 0.0 else 0.0

    qty = min(qty_risk, qty_cap)
    if not _finite(qty) or qty <= 0.0:
        return 0.0
    return float(qty)


def run_backtest(
    series: OhlcvSeries,
    entry: Optional[Sequence[float]] = None,
    exit: Optional[Sequence[float]] = None,
    *,
    entry_signal: Optional[Sequence[float]] = None,
    entry_signals: Optional[Sequence[float]] = None,
    exit_signal: Optional[Sequence[float]] = None,
    exit_signals: Optional[Sequence[float]] = None,
    direction: str = "long",
    side: Optional[Side] = None,
    start: int = 0,
    end: Optional[int] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    stop_spec: Any = None,
    stop: Any = None,
    tp_spec: Any = None,
    take_profit: Any = None,
    tp: Any = None,
    time_stop_bars: int = 0,
    cooldown_bars: int = 0,
    cfg: Optional[BacktestConfig] = None,
) -> Dict[str, Any]:
    """
    Miner-compatible backtest wrapper that consumes entry/exit *signals* (arrays) and supports:
      - stop_spec: none/percent/atr/structure (+ trailing atr)
      - tp_spec: none/percent/rr/atr
      - time_stop_bars, cooldown_bars

    Returns a dict with keys expected by Miner_Search._parse_engine_result():
      - equity_curve: List[float] (normalized to 1.0 initial)
      - trades: List[dict] with entry_i, exit_i, entry_price, exit_price, ret, fees
      - fees_paid, funding_paid (fractions of initial equity)

    Execution model matches miner's simple backtester:
      - signals evaluated at bar close i-1, executed at bar open i
      - intrabar stop/take on bar i using OHLC (worst-case if both hit)
      - trailing ATR stop updated on bar close i (for next bar)
      - force close at fold/window end close[end-1] (so folds are independent)
    """
    # Resolve aliases
    if start_idx is not None:
        start = int(start_idx)
    if end_idx is not None:
        end = int(end_idx)

    entry_sig = entry if entry is not None else (entry_signal if entry_signal is not None else entry_signals)
    exit_sig = exit if exit is not None else (exit_signal if exit_signal is not None else exit_signals)

    stop_obj = stop_spec if stop_spec is not None else stop
    tp_obj = tp_spec if tp_spec is not None else (take_profit if take_profit is not None else tp)

    if cfg is None:
        cfg = BacktestConfig()

    n = len(series.ts_ms)
    require(n > 0, "series must have at least 1 bar")
    require(entry_sig is not None, "entry signal array is required")
    require(len(entry_sig) == n, "entry signal array length must match series length")

    if exit_sig is not None and len(exit_sig) != n:
        # Be permissive: treat mismatched exit as "no exit signals"
        exit_sig = None

    start_i = int(start)
    end_i = int(n if end is None else end)
    require(0 <= start_i < end_i <= n, f"invalid window: start={start_i}, end={end_i}, n={n}")

    # Direction / side
    if side is None:
        d = str(direction).lower().strip()
        if d in ("long", "buy", "bull"):
            side = Side.LONG
        elif d in ("short", "sell", "bear"):
            side = Side.SHORT
        else:
            raise ValidationError(f"Unsupported direction: {direction!r}")

    # Stop/TP kinds
    stop_kind = str(getattr(stop_obj, "kind", "none") if stop_obj is not None else "none")
    tp_kind = str(getattr(tp_obj, "kind", "none") if tp_obj is not None else "none")

    # Local refs
    ts_ms = series.ts_ms
    o = series.open
    h = series.high
    l = series.low
    c = series.close

    equity_cash = float(cfg.initial_equity)
    eq_curve: List[float] = [float(cfg.initial_equity)] * n

    pos: Optional[_PositionState] = None
    trades_out: List[Dict[str, Any]] = []

    total_fees = 0.0
    total_liquidation_fees = 0.0
    total_funding = 0.0
    liquidation_count = 0

    # Position bookkeeping
    entry_i_open = -1
    entry_equity_pre_fee = 0.0
    bars_in_pos = 0
    next_entry_open_idx = int(start_i)

    # Helpers ---------------------------------------------------------
    def _set_risk_levels(signal_i: int, entry_px_exec: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Computes stop/take levels at signal_i (bar close), given entry exec price at next bar open.
        Returns (stop_price_raw, take_price_raw). May return (None, None) if required data missing.
        """
        sp: Optional[float] = None
        tp_px: Optional[float] = None

        # ----- Stop -----
        if stop_kind == "none":
            sp = None
        elif stop_kind == "percent":
            stop_pct = float(getattr(stop_obj, "stop_pct"))
            if not _finite(stop_pct) or stop_pct <= 0.0:
                return None, None
            sp = entry_px_exec * (1.0 - stop_pct) if side == Side.LONG else entry_px_exec * (1.0 + stop_pct)
        elif stop_kind == "atr":
            period = int(getattr(stop_obj, "atr_period"))
            mult = float(getattr(stop_obj, "atr_mult"))
            atr_arr = _get_atr_arr(series, period)
            atr_v = _safe_pos_value(atr_arr, signal_i)
            if atr_v is None or not _finite(mult) or mult <= 0.0:
                return None, None
            sp = entry_px_exec - mult * atr_v if side == Side.LONG else entry_px_exec + mult * atr_v
        elif stop_kind == "structure":
            left = int(getattr(stop_obj, "ms_left"))
            right = int(getattr(stop_obj, "ms_right"))
            buffer_bps = float(getattr(stop_obj, "buffer_bps", 0.0))
            level_name = str(getattr(stop_obj, "level", "last_pivot_low" if side == Side.LONG else "last_pivot_high"))
            piv_arr = _get_pivot_arr(series, level_name, left, right)
            lvl = _safe_pos_value(piv_arr, signal_i)
            if lvl is None:
                return None, None
            buf = _bps_to_frac(buffer_bps) if _finite(buffer_bps) else 0.0
            sp = lvl * (1.0 - buf) if side == Side.LONG else lvl * (1.0 + buf)
        else:
            raise ValidationError(f"Unsupported stop kind: {stop_kind!r}")

        # ----- Take profit -----
        if tp_kind == "none":
            tp_px = None
        elif tp_kind == "percent":
            tp_pct = float(getattr(tp_obj, "tp_pct"))
            if not _finite(tp_pct) or tp_pct <= 0.0:
                return None, None
            tp_px = entry_px_exec * (1.0 + tp_pct) if side == Side.LONG else entry_px_exec * (1.0 - tp_pct)
        elif tp_kind == "rr":
            rr = float(getattr(tp_obj, "rr"))
            if sp is None or not _finite(rr) or rr <= 0.0:
                return None, None
            risk = abs(entry_px_exec - sp)
            if not _finite(risk) or risk <= 0.0:
                return None, None
            tp_px = entry_px_exec + rr * risk if side == Side.LONG else entry_px_exec - rr * risk
        elif tp_kind == "atr":
            period = int(getattr(tp_obj, "atr_period"))
            mult = float(getattr(tp_obj, "atr_mult"))
            atr_arr = _get_atr_arr(series, period)
            atr_v = _safe_pos_value(atr_arr, signal_i)
            if atr_v is None or not _finite(mult) or mult <= 0.0:
                return None, None
            tp_px = entry_px_exec + mult * atr_v if side == Side.LONG else entry_px_exec - mult * atr_v
        else:
            raise ValidationError(f"Unsupported take-profit kind: {tp_kind!r}")

        # Sanity
        if sp is not None and (not _finite(sp) or sp <= 0.0):
            sp = None
        if tp_px is not None and (not _finite(tp_px) or tp_px <= 0.0):
            tp_px = None

        # Direction constraints (safety)
        if sp is not None:
            if side == Side.LONG and sp >= entry_px_exec:
                sp = None
            if side == Side.SHORT and sp <= entry_px_exec:
                sp = None
        if tp_px is not None:
            if side == Side.LONG and tp_px <= entry_px_exec:
                tp_px = None
            if side == Side.SHORT and tp_px >= entry_px_exec:
                tp_px = None

        return sp, tp_px

    def _record_trade(tr: Trade, exit_i: int) -> None:
        nonlocal entry_equity_pre_fee
        init = float(cfg.initial_equity)
        if init <= 0.0:
            init = 1.0
        # Account-level return for this trade (net)
        ret = (float(equity_cash) / float(entry_equity_pre_fee)) - 1.0 if entry_equity_pre_fee > 0.0 else 0.0
        trades_out.append(
            {
                "entry_i": int(entry_i_open),
                "exit_i": int(exit_i),
                "entry_price": float(tr.entry_price),
                "exit_price": float(tr.exit_price),
                "ret": float(ret),
                # Net PnL (including fees + funding) as fraction of initial equity.
                # This enables consistent PF/concentration metrics in the miner adapter.
                "pnl": float(tr.pnl_net / init) if init > 0 else 0.0,
                "reason": str(tr.reason.value),
            }
        )

    # Main loop -------------------------------------------------------
    for i in range(start_i, end_i):
        t = int(ts_ms[i])
        o_i = float(o[i])
        h_i = float(h[i])
        l_i = float(l[i])
        c_i = float(c[i])

        sig_i = int(i - 1)

        # Funding (at bar open) on reference notional from previous close
        if pos is not None and i > 0 and _is_funding_time(t, cfg.funding_period_ms):
            rate = _funding_rate(t, cfg)
            notional_ref = abs(float(pos.qty)) * float(c[i - 1])
            side_sign = 1.0 if pos.side == Side.LONG else -1.0
            funding_pnl = -float(rate) * float(notional_ref) * float(side_sign)
            if _finite(funding_pnl) and funding_pnl != 0.0:
                equity_cash += float(funding_pnl)
                pos.funding_accum += float(funding_pnl)
                total_funding += float(funding_pnl)

        # 1) Exit at OPEN (time-stop or explicit exit signal)
        if pos is not None:
            if int(time_stop_bars) > 0 and int(bars_in_pos) >= int(time_stop_bars):
                equity_cash, tr, _ = _close_position(
                    equity_cash,
                    pos=pos,
                    exit_ts=t,
                    exit_price_raw=o_i,
                    reason=ExitReason.TIME_STOP,
                    cfg=cfg,
                )
                pos = None
                total_fees += float(tr.exit_fee) + float(tr.liquidation_fee)
                total_liquidation_fees += float(tr.liquidation_fee)
                if tr.reason == ExitReason.LIQUIDATION:
                    liquidation_count += 1
                _record_trade(tr, exit_i=i)
                next_entry_open_idx = int(i + 1 + max(0, int(cooldown_bars)))
                bars_in_pos = 0
            elif exit_sig is not None and sig_i >= start_i and 0 <= sig_i < n and float(exit_sig[sig_i]) > 0.5:
                equity_cash, tr, _ = _close_position(
                    equity_cash,
                    pos=pos,
                    exit_ts=t,
                    exit_price_raw=o_i,
                    reason=ExitReason.SIGNAL,
                    cfg=cfg,
                )
                pos = None
                total_fees += float(tr.exit_fee) + float(tr.liquidation_fee)
                total_liquidation_fees += float(tr.liquidation_fee)
                if tr.reason == ExitReason.LIQUIDATION:
                    liquidation_count += 1
                _record_trade(tr, exit_i=i)
                next_entry_open_idx = int(i + 1 + max(0, int(cooldown_bars)))
                bars_in_pos = 0

        # 2) Entry at OPEN (uses signal at close i-1)
        if (
            pos is None
            and i >= next_entry_open_idx
            and sig_i >= start_i
            and 0 <= sig_i < n
            and float(entry_sig[sig_i]) > 0.5
        ):
            is_buy_entry = side == Side.LONG  # long buys to open; short sells to open
            entry_exec = apply_costs(o_i, is_buy=is_buy_entry, cfg=cfg)

            sp, tp_px = _set_risk_levels(sig_i, float(entry_exec))

            # Enforce: if a stop/tp was requested but couldn't be computed, skip entry.
            if stop_kind != "none" and sp is None:
                pass
            elif tp_kind != "none" and tp_px is None:
                pass
            else:
                # Dynamic risk sizing: use stop distance if available, else use conservative placeholder.
                if sp is not None:
                    stop_dist = abs(float(entry_exec) - float(sp))
                else:
                    # No explicit stop -> size as if stop is at least 2% away (safer).
                    placeholder_bps = max(float(cfg.stop_bps), 200.0)
                    stop_dist = float(entry_exec) * _bps_to_frac(placeholder_bps)

                qty = _compute_qty_for_stop_dist(equity_cash, float(entry_exec), float(stop_dist), cfg)
                qty = _apply_entry_constraints(qty, float(entry_exec), cfg)
                if qty > 0.0:
                    notional = abs(float(qty)) * float(entry_exec)
                    entry_fee = _fee(notional, cfg)
                    if _finite(entry_fee) and entry_fee >= 0.0 and equity_cash > entry_fee:
                        entry_equity_pre_fee = float(equity_cash)
                        equity_cash -= float(entry_fee)
                        total_fees += float(entry_fee)
                        liq_px = _liquidation_price(float(entry_exec), side, cfg)

                        pos = _PositionState(
                            side=side,
                            qty=float(qty),
                            entry_ts=t,
                            entry_price=float(entry_exec),
                            stop_price=float(sp) if sp is not None else None,
                            take_price=float(tp_px) if tp_px is not None else None,
                            liquidation_price=float(liq_px) if liq_px is not None else None,
                            entry_fee=float(entry_fee),
                            funding_accum=0.0,
                            notional=float(notional),
                            margin_used=float(notional) / float(cfg.leverage),
                        )
                        entry_i_open = int(i)
                        bars_in_pos = 0

        # 3) Intrabar stop/take (same bar)
        if pos is not None:
            trig = _check_intrabar_exit(pos, o=o_i, h=h_i, l=l_i, cfg=cfg)
            if trig is not None:
                reason, exit_raw = trig
                equity_cash, tr, _ = _close_position(
                    equity_cash,
                    pos=pos,
                    exit_ts=t,
                    exit_price_raw=float(exit_raw),
                    reason=reason,
                    cfg=cfg,
                )
                pos = None
                total_fees += float(tr.exit_fee) + float(tr.liquidation_fee)
                total_liquidation_fees += float(tr.liquidation_fee)
                if tr.reason == ExitReason.LIQUIDATION:
                    liquidation_count += 1
                _record_trade(tr, exit_i=i)
                next_entry_open_idx = int(i + 1 + max(0, int(cooldown_bars)))
                bars_in_pos = 0

        # 4) Mark-to-market at CLOSE (equity curve)
        if pos is None:
            eq_curve[i] = float(equity_cash)
        else:
            eq_curve[i] = float(_equity_mark_to_market(equity_cash, pos, mark_price=c_i, cfg=cfg))

            # Trailing ATR stop update at close (for next bar)
            if stop_kind == "atr" and bool(getattr(stop_obj, "trail", False)) and pos.stop_price is not None:
                period = int(getattr(stop_obj, "atr_period"))
                mult = float(getattr(stop_obj, "atr_mult"))
                atr_arr = _get_atr_arr(series, period)
                atr_v = _safe_pos_value(atr_arr, i)
                if atr_v is not None and _finite(mult) and mult > 0.0:
                    if side == Side.LONG:
                        new_sp = float(c_i) - float(mult) * float(atr_v)
                        if _finite(new_sp) and new_sp > float(pos.stop_price):
                            pos.stop_price = float(new_sp)
                    else:
                        new_sp = float(c_i) + float(mult) * float(atr_v)
                        if _finite(new_sp) and new_sp < float(pos.stop_price):
                            pos.stop_price = float(new_sp)

            bars_in_pos += 1

    # Force close at window end (close[end-1]) so folds are independent.
    if pos is not None:
        i_last = int(end_i - 1)
        t_last = int(ts_ms[i_last])
        c_last = float(c[i_last])

        equity_cash, tr, _ = _close_position(
            equity_cash,
            pos=pos,
            exit_ts=t_last,
            exit_price_raw=c_last,
            reason=ExitReason.END_OF_DATA,
            cfg=cfg,
        )
        pos = None
        total_fees += float(tr.exit_fee) + float(tr.liquidation_fee)
        total_liquidation_fees += float(tr.liquidation_fee)
        _record_trade(tr, exit_i=i_last)
        eq_curve[i_last] = float(equity_cash)

    # Flat equity after window
    for i in range(end_i, n):
        eq_curve[i] = float(equity_cash)

    # Normalize equity curve to 1.0 initial (miner convention)
    init = float(cfg.initial_equity)
    if init > 0.0 and _finite(init):
        eq_norm = [float(x) / init for x in eq_curve]
    else:
        eq_norm = list(eq_curve)

    return {
        "equity_curve": eq_norm,
        "trades": trades_out,
        "fees_paid": float(total_fees / init) if init > 0.0 else 0.0,
        "liquidation_paid": float(total_liquidation_fees / init) if init > 0.0 else 0.0,
        "liquidations": int(liquidation_count),
        "funding_paid": float(total_funding / init) if init > 0.0 else 0.0,
    }

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

    # ---------- Test 2b: stop loss with adverse fill slippage ----------
    # stop at 99, low 98, mult=1 => fill at low => loss 20
    series2b = OhlcvSeries(
        ts_ms=[1, 2],
        open=[100, 100],
        high=[100.5, 100.5],
        low=[98.0, 98.0],
        close=[100, 100],
        volume=[1, 1],
        symbol="TEST",
        timeframe="1m",
    )
    target2b = [1, 1]
    cfg2b = BacktestConfig(
        initial_equity=1000.0,
        leverage=1.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        adverse_fill_slip_mult=1.0,
        risk_per_trade_fraction=0.01,  # $10 risk
        max_margin_fraction=1.0,
        stop_bps=100.0,                # 1% stop
        take_bps=None,
        funding_rate_per_period=0.0,
        close_on_end=True,
        store_trades=True,
        store_equity_curve=False,
    )
    res2b = backtest_target_position(series2b, target2b, cfg2b)
    _expect(res2b.trade_count == 1, "Test2b: expected 1 trade")
    _expect(res2b.trades[0].reason == ExitReason.STOP_LOSS, "Test2b: expected STOP_LOSS")
    _expect(abs(res2b.final_equity_cash - 980.0) < 1e-9, f"Test2b: final equity mismatch {res2b.final_equity_cash}")

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
    # ---------- Test 5: liquidation triggers before wide stop ----------
    series5 = OhlcvSeries(
        ts_ms=[1, 2],
        open=[100.0, 100.0],
        high=[100.0, 100.0],
        low=[90.0, 90.0],
        close=[100.0, 100.0],
        volume=[1.0, 1.0],
    )
    target5 = [1, 1]
    cfg5 = BacktestConfig(
        initial_equity=1000.0,
        leverage=10.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=0.01,
        max_margin_fraction=1.0,
        stop_bps=2000.0,  # stop at 80; liq ~90.5 -> liquidation should happen first
        take_bps=None,
        maintenance_margin_rate=0.005,
        store_trades=True,
        store_equity_curve=False,
        close_on_end=True,
    )
    res5 = backtest_target_position(series5, target5, cfg5)
    _expect(len(res5.trades) == 1, "Test5: expected one trade")
    _expect(res5.trades[0].reason == ExitReason.LIQUIDATION, f"Test5: expected LIQUIDATION got {res5.trades[0].reason}")

    # ---------- Test 6: stop protects against liquidation when closer ----------
    cfg6 = BacktestConfig(
        initial_equity=1000.0,
        leverage=10.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=0.01,
        max_margin_fraction=1.0,
        stop_bps=100.0,  # stop at 99; liq ~90.5 -> stop should fire first
        take_bps=None,
        maintenance_margin_rate=0.005,
        store_trades=True,
        store_equity_curve=False,
        close_on_end=True,
    )
    res6 = backtest_target_position(series5, target5, cfg6)
    _expect(len(res6.trades) == 1, "Test6: expected one trade")
    _expect(res6.trades[0].reason == ExitReason.STOP_LOSS, f"Test6: expected STOP_LOSS got {res6.trades[0].reason}")
    # ---------- Test 7: exchange min_notional blocks tiny orders ----------
    series7 = OhlcvSeries(
        ts_ms=[1, 2],
        open=[100.0, 100.0],
        high=[100.0, 100.0],
        low=[100.0, 100.0],
        close=[100.0, 100.0],
        volume=[1.0, 1.0],
        symbol="TEST",
        timeframe="1m",
    )
    target7 = [1, 1]
    cfg7 = BacktestConfig(
        initial_equity=1000.0,
        leverage=10.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=1e-6,
        max_margin_fraction=1.0,
        stop_bps=100.0,
        take_bps=None,
        min_notional_usdt=10.0,
        store_trades=True,
        store_equity_curve=False,
        close_on_end=True,
    )
    res7 = backtest_target_position(series7, target7, cfg7)
    _expect(res7.trade_count == 0, f"Test7: expected 0 trades, got {res7.trade_count}")
    _expect(abs(res7.final_equity_cash - 1000.0) < 1e-9, f"Test7: final equity mismatch {res7.final_equity_cash}")

    # ---------- Test 8: qty_step floors position size (never rounds up) ----------
    target8 = [1, 1, 0, 0]
    cfg8 = BacktestConfig(
        initial_equity=1000.0,
        leverage=2.0,
        fee_rate_taker=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        risk_per_trade_fraction=0.01234,
        max_margin_fraction=1.0,
        stop_bps=100.0,
        take_bps=None,
        qty_step=0.25,
        store_trades=True,
        store_equity_curve=False,
        close_on_end=True,
    )
    res8 = backtest_target_position(series1, target8, cfg8)
    _expect(res8.trade_count == 1, f"Test8: expected 1 trade, got {res8.trade_count}")
    _expect(abs(res8.trades[0].qty - 12.25) < 1e-9, f"Test8: qty mismatch {res8.trades[0].qty}")
    _expect(abs(res8.final_equity_cash - 1245.0) < 1e-9, f"Test8: final equity mismatch {res8.final_equity_cash}")
 

    print("Backtest_Engine self-test: OK")


if __name__ == "__main__":
    _self_test()
