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
- Backtest_Engine (required; single backend)
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

from dataclasses import dataclass, asdict, replace, field
import hashlib
import heapq
import inspect
import json
import itertools
import math
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from Core_Types import OhlcvSeries, ValidationError, require
from Signals_Events import (
    SignalHub,
    FeatureRef,
    Condition,
    AllOf,
    AnyOf,
    Not,
    EventSequence,
    SequenceStep,
)
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
    strategy_spec_to_dsl,
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
def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf (keeps repo scipy-free)."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) (SciPy-free).

    Uses a continued-fraction expansion (Numerical Recipes style) and symmetry
    to remain stable near the tails. Accuracy is sufficient for gating / diagnostics
    in this repo without adding heavy dependencies.
    """
    aa = float(a)
    bb = float(b)
    xx = float(x)

    require(_finite(aa) and _finite(bb), "betainc params must be finite")
    require(aa > 0.0 and bb > 0.0, "betainc params must be > 0")
    if not _finite(xx):
        raise ValidationError("betainc x must be finite")
    if xx <= 0.0:
        return 0.0
    if xx >= 1.0:
        return 1.0

    # Continued fraction for incomplete beta.
    def _betacf(a0: float, b0: float, x0: float) -> float:
        MAXIT = 200
        EPS = 3e-14
        FPMIN = 1e-300

        qab = a0 + b0
        qap = a0 + 1.0
        qam = a0 - 1.0

        c = 1.0
        d = 1.0 - qab * x0 / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d

        for m in range(1, MAXIT + 1):
            m2 = 2 * m

            # even step
            num = m * (b0 - m) * x0
            den = (qam + m2) * (a0 + m2)
            aa2 = num / den if den != 0.0 else 0.0
            d = 1.0 + aa2 * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa2 / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c

            # odd step
            num = -(a0 + m) * (qab + m) * x0
            den = (a0 + m2) * (qap + m2)
            aa2 = num / den if den != 0.0 else 0.0
            d = 1.0 + aa2 * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa2 / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            del_ = d * c
            h *= del_

            if abs(del_ - 1.0) < EPS:
                break

        return float(h)

    # Prefactor
    ln_bt = (
        math.lgamma(aa + bb)
        - math.lgamma(aa)
        - math.lgamma(bb)
        + aa * math.log(xx)
        + bb * math.log1p(-xx)
    )
    bt = math.exp(ln_bt)

    # Use symmetry for better convergence.
    thresh = (aa + 1.0) / (aa + bb + 2.0)
    if xx < thresh:
        out = bt * _betacf(aa, bb, xx) / aa
    else:
        out = 1.0 - (bt * _betacf(bb, aa, 1.0 - xx) / bb)

    if not _finite(out):
        return 1.0 if xx > 0.5 else 0.0
    return max(0.0, min(1.0, float(out)))


def _t_cdf(t: float, df: int) -> float:
    """Student-t CDF with df degrees of freedom (SciPy-free)."""
    v = int(df)
    require(v > 0, "df must be > 0")
    tt = float(t)
    if not _finite(tt):
        return 1.0 if tt > 0.0 else 0.0
    if tt == 0.0:
        return 0.5

    vv = float(v)
    x = vv / (vv + tt * tt)
    ib = _betainc_reg(vv / 2.0, 0.5, x)
    if tt > 0.0:
        return 1.0 - 0.5 * float(ib)
    return 0.5 * float(ib)


def _p_value_two_sided_from_t(t_stat: float, df: Optional[int] = None) -> float:
    """Two-sided p-value from a t-stat.

    - If df is provided (and not too large), uses the Student-t distribution.
    - Otherwise falls back to a standard normal approximation (fast).
    Two-sided p-value from a t-stat using a normal approximation.

    Notes:
    - We intentionally avoid scipy to keep the repo lightweight.
    - For small sample sizes this is only an approximation; treat as a heuristic.
    """
    t = float(t_stat)
    if not _finite(t):
        return 0.0 if t > 0.0 else 1.0

    z = abs(t)
    v = int(df) if df is not None else 0
    if v > 0 and v <= 200:
        cdf = _t_cdf(z, v)
        p = 2.0 * (1.0 - float(cdf))
    else:
        p = 2.0 * (1.0 - _norm_cdf(z))
    # numeric guard
    if not _finite(p):
        return 1.0
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return float(p)

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
def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return 0.5 * (float(ys[mid - 1]) + float(ys[mid]))


def _compute_cscv_pbo_for_results(
    results: Sequence[Any],
    *,
    max_combinations: int = 2000,
    seed: int = 123,
    min_folds: int = 4,
    min_pool: int = 8,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Combinatorial Symmetric Cross-Validation (CSCV) / Probability of Backtest Overfitting (PBO).

    This is a *selection-bias* diagnostic popularized by Bailey et al.:
    - Split the K walk-forward test folds into IS/OOS subsets (combinatorially).
    - For each split, pick the best strategy on IS and observe its rank on OOS.
    - PBO is the fraction of splits where the IS-winner ranks below the OOS median.

    We compute this cheaply from already-computed per-fold returns (no extra backtests).

    Returns:
      (summary, per_strategy_stats_by_hash)

    Notes:
    - Uses aggregate["fold_returns_used"] when available (respects return_mode=absolute|excess).
    - Rank orientation: worst=0.0, best=1.0 for oos_rank_* stats.
    """
    # Gather a deterministic pool (sorted by hash for tie-breaking stability).
    items: List[Tuple[str, List[float]]] = []
    for r in results:
        h = str(getattr(r, "strategy_hash", "") or "")
        if not h:
            continue
        agg = getattr(r, "aggregate", None) or {}
        rets: Any = None
        if isinstance(agg, dict):
            rets = agg.get("fold_returns_used", None)
            if rets is None:
                rets = agg.get("fold_returns", None)
        if isinstance(rets, (list, tuple)) and rets:
            try:
                rr = [float(x) for x in rets]
            except Exception:
                rr = []
        else:
            rr = []
            try:
                fms = getattr(r, "fold_metrics", None) or ()
                for fm in fms:
                    rr.append(float(getattr(fm, "net_return", 0.0) or 0.0))
            except Exception:
                rr = []

        if not rr:
            continue
        if any(not _finite(float(x)) for x in rr):
            continue
        items.append((h, rr))

    if len(items) < int(min_pool):
        return None, {}

    # Enforce consistent K across items
    k = int(len(items[0][1]))
    items = [it for it in items if int(len(it[1])) == int(k)]
    if int(k) < int(min_folds) or len(items) < int(min_pool):
        return None, {}

    items.sort(key=lambda x: x[0])
    ids = [h for h, _ in items]
    mat = [rr for _, rr in items]

    n_strat = int(len(ids))
    if n_strat < int(min_pool):
        return None, {}

    is_k = int(k // 2)
    oos_k = int(k - is_k)
    if is_k <= 0 or oos_k <= 0:
        return None, {}

    # Count combinations without necessarily enumerating all.
    try:
        combos_total = int(math.comb(int(k), int(is_k)))
    except Exception:
        # fallback
        combos_total = 0

    combos: List[Tuple[int, ...]] = []
    max_c = max(1, int(max_combinations))

    if combos_total > 0 and combos_total <= max_c:
        combos = list(itertools.combinations(range(int(k)), int(is_k)))
    else:
        # Sample a bounded number of random fold subsets (unique) for large K.
        rng = random.Random(int(seed))
        seen: set = set()
        tries = 0
        max_tries = max_c * 50
        while len(seen) < max_c and tries < max_tries:
            c = tuple(sorted(rng.sample(range(int(k)), int(is_k))))
            seen.add(c)
            tries += 1
        combos = list(seen)
        combos_total = combos_total if combos_total > 0 else int(len(combos))

    if not combos:
        return None, {}

    lambdas: List[float] = []
    sel_counts: Dict[str, int] = {}
    sel_oos_ranks: Dict[str, List[float]] = {}
    sel_oos_rets: Dict[str, List[float]] = {}

    all_idx = tuple(range(int(k)))

    for is_idx in combos:
        is_set = set(is_idx)
        oos_idx = tuple(i for i in all_idx if i not in is_set)

        # Compute IS winner and all OOS performances.
        oos_perf: List[float] = []
        best_j = 0
        best_is = float("-inf")
        inv_is = 1.0 / float(len(is_idx))
        inv_oos = 1.0 / float(len(oos_idx))

        for j, row in enumerate(mat):
            is_sum = 0.0
            for ii in is_idx:
                is_sum += float(row[ii])
            oos_sum = 0.0
            for ii in oos_idx:
                oos_sum += float(row[ii])
            is_v = float(is_sum) * inv_is
            oos_v = float(oos_sum) * inv_oos
            oos_perf.append(oos_v)
            # strict > => deterministic tie-break to lowest hash (because ids are sorted)
            if float(is_v) > float(best_is):
                best_is = float(is_v)
                best_j = int(j)

        sel_id = ids[int(best_j)]
        sel_oos = float(oos_perf[int(best_j)])

        # OOS mid-rank (worst=1, best=N), then normalize to [0,1] as (rank-1)/(N-1).
        lt = 0
        le = 0
        for v in oos_perf:
            if float(v) < float(sel_oos):
                lt += 1
            if float(v) <= float(sel_oos):
                le += 1
        rank_low = 1 + int(lt)
        rank_high = int(le)
        rank = 0.5 * (float(rank_low) + float(rank_high))

        if n_strat > 1:
            oos_rank01 = (float(rank) - 1.0) / float(n_strat - 1)
        else:
            oos_rank01 = 1.0

        # PBO lambda uses r in (0,1): r = rank/(N+1), with rank 1=worst, N=best.
        r = float(rank) / float(n_strat + 1)
        # numeric guards
        r = max(1e-12, min(1.0 - 1e-12, float(r)))
        lam = float(math.log(float(r) / (1.0 - float(r))))
        lambdas.append(lam)

        sel_counts[sel_id] = int(sel_counts.get(sel_id, 0)) + 1
        sel_oos_ranks.setdefault(sel_id, []).append(float(oos_rank01))
        sel_oos_rets.setdefault(sel_id, []).append(float(sel_oos))

    n_combo = int(len(lambdas))
    pbo = (sum(1 for x in lambdas if float(x) < 0.0) / float(n_combo)) if n_combo > 0 else 1.0

    # Per-strategy stats (only meaningful for those selected at least once).
    per: Dict[str, Dict[str, Any]] = {}
    for sid, cnt in sel_counts.items():
        ranks = sel_oos_ranks.get(sid, []) or []
        rets = sel_oos_rets.get(sid, []) or []
        per[sid] = {
            "selected_count": int(cnt),
            "selected_fraction": float(cnt) / float(n_combo) if n_combo > 0 else 0.0,
            "oos_rank_mean": float(_mean(ranks)) if ranks else None,
            "oos_rank_median": float(_median(ranks)) if ranks else None,
            "oos_return_mean": float(_mean(rets)) if rets else None,
            "oos_return_median": float(_median(rets)) if rets else None,
        }

    # Small helper: show top selection frequencies in summary (audit-friendly)
    top_sel = sorted(sel_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    summary: Dict[str, Any] = {
        "enabled": True,
        "pool_size": int(n_strat),
        "folds": int(k),
        "is_folds": int(is_k),
        "oos_folds": int(oos_k),
        "combinations_total": int(combos_total),
        "combinations_used": int(n_combo),
        "pbo": float(pbo),
        "lambda_mean": float(_mean(lambdas)) if lambdas else 0.0,
        "lambda_median": float(_median(lambdas)) if lambdas else 0.0,
        "top_selected": [(str(s), int(c)) for s, c in top_sel],
    }
    return summary, per

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
    holdout_bars: reserve the LAST N bars as untouched final holdout (never used in folds)
    """
    train_bars: int = 1000
    test_bars: int = 500
    step_bars: int = 500
    purge_bars: int = 0
    embargo_bars: int = 0
    anchored: bool = False
    holdout_bars: int = 0
    def __post_init__(self) -> None:
        require(int(self.train_bars) > 0, "train_bars must be > 0")
        require(int(self.test_bars) > 0, "test_bars must be > 0")
        require(int(self.step_bars) > 0, "step_bars must be > 0")
        require(int(self.purge_bars) >= 0, "purge_bars must be >= 0")
        require(int(self.embargo_bars) >= 0, "embargo_bars must be >= 0")
        require(int(self.holdout_bars) >= 0, "holdout_bars must be >= 0")


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
    backtest_mode: str = "engine"  # engine-only ("auto" is treated as "engine"; "simple" removed)
 

    # Caches
    max_expr_cache: int = 4000
    max_regime_cache: int = 1000

    # Diversification (Loop 12)
    diversify_by_entry: bool = True
    max_variants_per_entry: int = 1
    diversify_by_family: bool = True
    max_per_family: int = 0  # 0 => auto
    # IMPORTANT: what counts as a "variant" for the entry-cap?
    # - "entry": cap applies per direction+entry only (exit/time/risk variants get starved!)
    # - "entry_exit": cap applies per direction+entry+exit
    # - "entry_exit_time": include time_stop/cooldown in the key too
    # - "entry_exit_time_regime": include regime_filter in the key too (recommended default)
    # Why include regime_filter?
    # If you cap variants per entry (max_variants_per_entry=1), and the regime filter is NOT part of the key,
    # you'll almost never evaluate filtered variants because StrategySpace yields the unfiltered one first.
    entry_key_mode: str = "entry_exit_time_regime"
    # Optional overrides for Backtest_Engine.BacktestConfig (engine backend only).
    # Examples:
    #   {"leverage": 5.0, "risk_per_trade_fraction": 0.002}
    #   {"funding_rate_fn": my_fn}  (callable(ts_ms)->rate)
    engine_cfg_overrides: Dict[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        m = str(self.mode).strip().lower()
        require(m in ("iterate", "sample"), f"mode invalid: {self.mode!r}")
        object.__setattr__(self, "mode", m)

        bt = str(self.backtest_mode).strip().lower()
        require(bt in ("auto", "engine"), f"backtest_mode invalid: {self.backtest_mode!r} (allowed: 'engine'/'auto')")
        # Backwards-compatible alias: "auto" => "engine" (no silent fallbacks).
        bt = "engine" if bt == "auto" else bt
        object.__setattr__(self, "backtest_mode", bt)
        ekm = str(getattr(self, "entry_key_mode", "entry_exit")).strip().lower()
        require(ekm in ("entry", "entry_exit", "entry_exit_time", "entry_exit_time_regime"), f"entry_key_mode invalid: {self.entry_key_mode!r}")
        object.__setattr__(self, "entry_key_mode", ekm)
        require(int(self.max_evals) > 0, "max_evals must be > 0")
        require(int(self.top_k) > 0, "top_k must be > 0")
        sp = float(self.sample_prob)
        if m == "sample":
            require(sp > 0.0 and sp <= 1.0, "sample_prob must be in (0,1] when mode='sample'")
        else:
            # iterate-mode doesn't use sample_prob, but keep it bounded for sanity
            require(sp >= 0.0 and sp <= 1.0, "sample_prob must be in [0,1] when mode='iterate'")      
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
    # Perpetual futures safety (engine backend): liquidation is a hard failure.
    # These limits have teeth because the engine backtester (Backtest_Engine) models liquidation.
    max_liquidations_total: int = 0      # reject if sum(liquidations across all folds) exceeds this
    max_liquidations_per_fold: int = 0   # reject if any single fold exceeds this
    # Profitability / realism filters (post-backtest, test folds)
    # Returns are fractions (0.10 = +10%), PF is ratio, turnover is trades per 1000 bars.
    min_mean_return: float = 0.0
    min_worst_fold_return: float = -0.05
    min_profit_factor: float = 1.0
    max_turnover_per_1000: float = 500.0
    # Profit concentration gate (test folds): reject strategies whose gross profits
    # are dominated by a single outlier trade. 1.0 disables.
    max_top_profit_share: float = 1.0
    # Selection-bias / robustness guards (Loop 2)
    # -----------------------------
    # How to interpret "return" for scoring and filters:
    # - "absolute": use strategy net returns (legacy)
    # - "excess"  : use (strategy_return - buy&hold_return_in_direction) per fold
    return_mode: str = "absolute"

    # Penalize low sample sizes in the RETURN component (similar to pf_trade_damp_ref for PF).
    # If trades_total < return_trade_damp_ref, return contribution is scaled by trades_total/ref.
    return_trade_damp_ref: int = 30

    # Fold consistency guard: require at least this fraction of folds to be positive
    # (based on the chosen return_mode). 0 disables.
    min_pos_fold_ratio: float = 0.0

    # Optional t-stat guard (based on fold returns in return_mode). 0 disables.
    min_t_stat: float = 0.0

    # Optional multiple-testing guard: Bonferroni-adjust p-value and require p_adj <= max_p_value_adj.
    # 0 disables.
    max_p_value_adj: float = 0.0

    # Number of "trials" used for the multiple-testing adjustment.
    # 0 => auto (SearchConfig.max_evals).
    multiple_testing_trials: int = 0
    # Optional CSCV stability gate (Loop 15)
    # -----------------------------
    # CSCV is computed from the already available per-fold returns of the *top pool*.
    # These thresholds gate the FINAL ranked pool (no extra backtests):
    # - selected_fraction: how often this strategy is the IS-winner across CSCV splits
    # - oos_rank_median : median normalized OOS rank when selected (0=worst .. 1=best)
    # 0 disables each gate.
    cscv_min_selected_fraction: float = 0.0
    cscv_min_oos_rank_median: float = 0.0
    # -----------------------------
    # Holdout gating (final untouched slice)
    # -----------------------------
    # If WalkForwardConfig.holdout_bars > 0, the miner can compute an "untouched final holdout"
    # and (optionally) require that a candidate ALSO meets minimum criteria there.
    #
    # IMPORTANT: If holdout_required=True, WalkForwardConfig.holdout_bars MUST be > 0.
    # (Fail-fast; do not silently degrade into "no holdout".)
    #
    # Rationale:
    # - Without gating, it's easy to select "robust" strategies that already fail on the most recent segment.
    # - This is a cheap, high-signal sanity check before you'd ever risk money.
    #
    # NOTE: holdout is still a single segment; you should treat it as a last filter, not proof of edge.
    holdout_required: bool = True
    # Minimum absolute number of trades required in the holdout slice.
    # NOTE: Intentionally >1 to avoid passing "1-trade holdouts" which are mostly noise.
    holdout_min_trades: int = 3
    # Additional holdout sample-size guard: require that the holdout contains a minimum share of the
    # *expected* trades given the strategy's out-of-sample trade density.
    #
    # required_trades_by_ratio = ceil(trades_total * (holdout_bars / total_test_bars) * holdout_trade_ratio_min)
    # required_trades = max(holdout_min_trades, required_trades_by_ratio)
    #
    # Example:
    # - total_test_bars=500, holdout_bars=200 => ratio=0.4
    # - trades_total=10, holdout_trade_ratio_min=0.5 => required_by_ratio=ceil(10*0.4*0.5)=2
    # - holdout_min_trades=3 => required_trades=3
    holdout_trade_ratio_min: float = 0.50
    holdout_min_return: float = 0.0
    holdout_max_drawdown_limit: float = 0.50
    holdout_min_profit_factor: float = 1.0
    # Reject strategies whose holdout PnL is dominated by a single trade.
    # 1.0 disables (allows 100% of gross profits from the best trade).
    holdout_max_top_profit_share: float = 1.0
    # Time-bucket profit concentration gate (temporal clustering).
    # Bucket size in ms (default aligns with 8h funding interval).
    holdout_profit_bucket_ms: int = 28_800_000
    # Max share of holdout gross profits contributed by the best time bucket.
    # 1.0 disables.
    holdout_max_top_bucket_profit_share: float = 1.0
    # Volatility-regime profit concentration gate (market-condition dependence).
    # Uses RegimeStore.vol_regime (-1/0/+1) and buckets trades by the regime label at trade exit.
    # 1.0 disables (allows 100% of gross profits from one volatility regime bucket).
    holdout_max_top_vol_profit_share: float = 1.0
    # -----------------------------
    # Holdout segment consistency (temporal stability)
    # -----------------------------
    # Split the holdout into N contiguous segments and require that performance is not
    # carried by a single short sub-window (common failure mode for 1m/5m perps).
    # 0 or 1 disables.
    holdout_segment_count: int = 0
    # Require at least this fraction of segments to have non-negative return (>=0). 0 disables.
    holdout_min_pos_segment_ratio: float = 0.0
    # Require the worst segment return to be >= this value. Use -1.0 to disable (return cannot be < -1).
    holdout_min_segment_return: float = -1.0
    # -----------------------------
    # Holdout stress-gate (real-money critical)
    # -----------------------------
    # If stress_cost_mult > 1, rerun the HOLDOUT slice with stressed costs too and require it still passes.
    holdout_stress_required: bool = True
    holdout_stress_min_return: float = 0.0
    holdout_stress_min_profit_factor: float = 1.0
    # If <=0, defaults to holdout_max_drawdown_limit
    holdout_stress_max_drawdown_limit: float = 0.0
    # Holdout stress retention gate: require stressed holdout return keeps >= X% of base holdout return.
    # This is the single best "fragility killer" before risking money.
    holdout_stress_return_ratio_min: float = 0.50
    # -----------------------------
    # Holdout latency stress (execution delay)
    # -----------------------------
    # Extra bars of *additional* delay applied to entry/exit signals inside the holdout window.
    # Why it matters:
    # - Many 1m strategies are fragile to even tiny execution delays.
    # - If a strategy only works with perfect open fills, it is not research-grade for real money.
    # 0 disables.
    holdout_latency_delay_bars: int = 0
    # Retention gate (only applied when base holdout return > 0):
    # require delayed holdout return >= base_holdout_return * holdout_latency_return_ratio_min.
    # 0 disables this ratio gate (but the delayed holdout must still pass the absolute holdout gates).
    holdout_latency_return_ratio_min: float = 0.30
    # -----------------------------
    # Holdout adverse-fill stress (intrabar STOP ambiguity)
    # -----------------------------
    # Run the holdout again but with a higher adverse_fill_slip_mult to simulate worse intrabar
    # execution when stops are triggered under OHLC ambiguity.
    # 0 disables.
    holdout_adverse_fill_stress_mult: float = 0.0
    # Retention gate (only applied when base holdout return > 0):
    # require stressed holdout return >= base_holdout_return * holdout_adverse_fill_return_ratio_min.
    # 0 disables this ratio gate (but the stressed holdout must still pass the absolute holdout gates).
    holdout_adverse_fill_return_ratio_min: float = 0.30
    # -----------------------------
    # Holdout permutation test (Loop 3)
    # -----------------------------
    # Optional negative-control sanity check: circularly time-shift entry/exit signals inside the
    # HOLDOUT window and re-run the holdout backtest.
    #
    # Interpretation (one-sided):
    # - If base holdout return >= 0: p ~= P(shifted_return >= observed_return)
    # - If base holdout return <  0: p ~= P(shifted_return <= observed_return)
    #
    # - holdout_perm_trials <= 0 disables the test
    # - holdout_perm_max_p <= 0 disables gating (still computed/stored if trials > 0)
    holdout_perm_trials: int = 0
    holdout_perm_max_p: float = 0.0
    # -----------------------------
    # Holdout parameter robustness (Loop 5)
    # -----------------------------
    # Many false positives are "brittle": tiny parameter changes destroy performance.
    # Real edges tend to survive in a small neighborhood.
    #
    # This test perturbs numeric parameters (indicator periods, thresholds, stop/tp params)
    # and re-runs the HOLDOUT slice.
    #
    # - param_robust_trials <= 0 disables the test
    # - param_robust_min_pass_ratio <= 0 disables gating (still computed/stored when trials > 0)
    #
    # Pass condition (one perturbation):
    # - trades >= base_trades * param_robust_trade_ratio_min (>=1)
    # - return retains >= param_robust_return_ratio_min of base holdout return (when base > 0)
    # - still respects holdout drawdown / PF gates
    param_robust_trials: int = 0
    # Relative jitter magnitude (e.g. 0.10 => +/-10% on periods/thresholds).
    param_robust_jitter: float = 0.10
    # Fraction of perturbations that must pass. 0 disables gating.
    param_robust_min_pass_ratio: float = 0.0
    # Retention of base holdout return required for each perturbation (when base>0).
    param_robust_return_ratio_min: float = 0.30
    # Retention of base holdout trade count required for each perturbation (>=1).
    param_robust_trade_ratio_min: float = 0.50
    # Numeric safety (prevents inf/NaN from breaking ranking/filters)
    # Profit factor can become +inf when a sample has no losing trades.
    # We cap it so the miner remains stable and JSON-friendly.
    profit_factor_cap: float = 1000.0
    profit_factor_eps: float = 1e-12
    # When sample sizes are tiny (e.g. 1-2 trades per fold), profit factor is extremely unstable
    # and can dominate the score even though the "edge" is pure noise.
    # We damp the PF term until total trades reach pf_trade_damp_ref.
    pf_trade_damp_ref: int = 30
    # Profit factor transform for scoring:
    # - "log":  pf_term = log(PF) * damp  (old behavior; can dominate when PF is huge)
    # - "tanh": pf_term = tanh(log(PF)) * damp (bounded, robust default)
    pf_transform: str = "tanh"
    # Score weights (test fold aggregates)
    weight_return: float = 1.0
    weight_dd: float = 0.7
    weight_stability: float = 0.5
    weight_pf: float = 0.1
    weight_complexity: float = 0.01
    weight_turnover: float = 0.05
    weight_worst_fold: float = 0.25
    # Execution assumptions (Backtest_Engine)
    fee_bps: float = 4.0       # taker fee per side (bps)
    spread_bps: float = 2.0    # bid-ask spread (bps). Model uses half-spread per fill.
    slippage_bps: float = 1.0  # additional slippage per fill (bps)
    # Intrabar adverse fill slippage for STOP/LIQ triggers (OHLC-only):
    # 0.0 = fill at trigger level (legacy); 1.0 = fill at bar extreme (very conservative).
    adverse_fill_slip_mult: float = 0.0
    # Funding (perpetual futures)
    # Constant funding rate per funding interval.
    # Positive => longs pay shorts. Negative => longs receive.
    # Unit: bps per funding interval (default 8h).
    funding_bps: float = 0.0
    funding_period_ms: int = 28_800_000
    # Optional robustness stress-test: re-run folds with costs multiplied.
    # If stress_cost_mult <= 1, stress test is disabled.
    # Typical research setting: 2.0 (double fee/spread/slippage).
    stress_cost_mult: float = 1.0
    stress_min_mean_return: float = 0.0
    stress_min_worst_fold_return: float = 0.0
    # Stress return retention gates (fractions in [0,1], set 0 to disable):
    # Require stressed performance to retain at least a fraction of base OOS performance.
    # Example: 0.5 means "under x2 costs, keep >=50% of base mean return".
    stress_mean_return_ratio_min: float = 0.50
    stress_worst_fold_return_ratio_min: float = 0.50    
    def __post_init__(self) -> None:
        require(int(self.min_entries_total) >= 0, "min_entries_total must be >= 0")
        require(int(self.max_entries_total) > 0, "max_entries_total must be > 0")
        require(int(self.min_entries_per_fold) >= 0, "min_entries_per_fold must be >= 0")
        require(int(self.min_trades_total) >= 0, "min_trades_total must be >= 0")
        require(int(self.min_trades_per_fold) >= 0, "min_trades_per_fold must be >= 0")
        require(float(self.max_drawdown_limit) > 0.0 and float(self.max_drawdown_limit) <= 1.0, "max_drawdown_limit must be in (0,1]")
        require(int(getattr(self, "max_liquidations_total", 0) or 0) >= 0, "max_liquidations_total must be >= 0")
        require(int(getattr(self, "max_liquidations_per_fold", 0) or 0) >= 0, "max_liquidations_per_fold must be >= 0")
        require(float(self.fee_bps) >= 0.0, "fee_bps must be >= 0")
        require(float(self.spread_bps) >= 0.0, "spread_bps must be >= 0")
        require(float(self.slippage_bps) >= 0.0, "slippage_bps must be >= 0")
        require(
            _finite(float(getattr(self, "adverse_fill_slip_mult", 0.0) or 0.0))
            and 0.0 <= float(getattr(self, "adverse_fill_slip_mult", 0.0) or 0.0) <= 1.0,
            "adverse_fill_slip_mult must be in [0,1]",
        )
        require(_finite(float(self.funding_bps)), "funding_bps must be finite")
        require(int(self.funding_period_ms) > 0, "funding_period_ms must be > 0")
        require(_finite(float(self.min_mean_return)), "min_mean_return must be finite")
        require(_finite(float(self.min_worst_fold_return)), "min_worst_fold_return must be finite")
        require(_finite(float(self.min_profit_factor)) and float(self.min_profit_factor) >= 0.0, "min_profit_factor must be >= 0")
        require(_finite(float(self.max_turnover_per_1000)) and float(self.max_turnover_per_1000) >= 0.0, "max_turnover_per_1000 must be >= 0")
        require(
            _finite(float(getattr(self, "max_top_profit_share", 1.0) or 1.0))
            and 0.0 <= float(getattr(self, "max_top_profit_share", 1.0) or 1.0) <= 1.0,
            "max_top_profit_share must be in [0,1]",
        )
        require(_finite(float(self.stress_cost_mult)) and float(self.stress_cost_mult) >= 0.0, "stress_cost_mult must be >= 0")
        require(_finite(float(self.stress_min_mean_return)), "stress_min_mean_return must be finite")
        require(_finite(float(self.stress_min_worst_fold_return)), "stress_min_worst_fold_return must be finite")
        require(
            _finite(float(self.stress_mean_return_ratio_min)) and 0.0 <= float(self.stress_mean_return_ratio_min) <= 1.0,
            "stress_mean_return_ratio_min must be in [0,1]",
        )
        require(
            _finite(float(self.stress_worst_fold_return_ratio_min)) and 0.0 <= float(self.stress_worst_fold_return_ratio_min) <= 1.0,
            "stress_worst_fold_return_ratio_min must be in [0,1]",
        )

        # holdout gating validation
        require(int(self.holdout_min_trades) >= 0, "holdout_min_trades must be >= 0")
        require(_finite(float(self.holdout_min_return)), "holdout_min_return must be finite")
        require(
            _finite(float(self.holdout_max_drawdown_limit)) and 0.0 < float(self.holdout_max_drawdown_limit) <= 1.0,
            "holdout_max_drawdown_limit must be in (0,1]",
        )
        require(
            _finite(float(self.holdout_min_profit_factor)) and float(self.holdout_min_profit_factor) >= 0.0,
            "holdout_min_profit_factor must be >= 0",
        )
        require(
            _finite(float(self.holdout_trade_ratio_min)) and 0.0 <= float(self.holdout_trade_ratio_min) <= 1.0,
            "holdout_trade_ratio_min must be in [0,1]",
        )
        require(
            _finite(float(getattr(self, "holdout_max_top_profit_share", 1.0) or 1.0))
            and 0.0 <= float(getattr(self, "holdout_max_top_profit_share", 1.0) or 1.0) <= 1.0,
            "holdout_max_top_profit_share must be in [0,1]",
        )
        require(int(getattr(self, "holdout_profit_bucket_ms", 28_800_000) or 28_800_000) > 0, "holdout_profit_bucket_ms must be > 0")
        require(
            _finite(float(getattr(self, "holdout_max_top_bucket_profit_share", 1.0) or 1.0))
            and 0.0 <= float(getattr(self, "holdout_max_top_bucket_profit_share", 1.0) or 1.0) <= 1.0,
            "holdout_max_top_bucket_profit_share must be in [0,1]",
        )
        require(
            _finite(float(getattr(self, "holdout_max_top_vol_profit_share", 1.0) or 1.0))
            and 0.0 <= float(getattr(self, "holdout_max_top_vol_profit_share", 1.0) or 1.0) <= 1.0,
            "holdout_max_top_vol_profit_share must be in [0,1]",
        )
        # Holdout segment consistency (temporal stability)
        require(int(getattr(self, "holdout_segment_count", 0) or 0) >= 0, "holdout_segment_count must be >= 0")
        require(
            _finite(float(getattr(self, "holdout_min_pos_segment_ratio", 0.0) or 0.0))
            and 0.0 <= float(getattr(self, "holdout_min_pos_segment_ratio", 0.0) or 0.0) <= 1.0,
            "holdout_min_pos_segment_ratio must be in [0,1]",
        )
        require(
            _finite(float(getattr(self, "holdout_min_segment_return", -1.0) or -1.0))
            and float(getattr(self, "holdout_min_segment_return", -1.0) or -1.0) >= -1.0,
            "holdout_min_segment_return must be >= -1.0",
        )
        require(_finite(float(self.holdout_stress_min_return)), "holdout_stress_min_return must be finite")
        require(
            _finite(float(self.holdout_stress_min_profit_factor)) and float(self.holdout_stress_min_profit_factor) >= 0.0,
            "holdout_stress_min_profit_factor must be >= 0",
        )
        require(
            _finite(float(self.holdout_stress_max_drawdown_limit)) and float(self.holdout_stress_max_drawdown_limit) >= 0.0,
            "holdout_stress_max_drawdown_limit must be >= 0",
        )
        require(
            _finite(float(self.holdout_stress_return_ratio_min)) and 0.0 <= float(self.holdout_stress_return_ratio_min) <= 1.0,
            "holdout_stress_return_ratio_min must be in [0,1]",
        )
        # holdout latency stress validation (execution delay)
        require(int(getattr(self, "holdout_latency_delay_bars", 0) or 0) >= 0, "holdout_latency_delay_bars must be >= 0")
        require(
            _finite(float(getattr(self, "holdout_latency_return_ratio_min", 0.0) or 0.0))
            and 0.0 <= float(getattr(self, "holdout_latency_return_ratio_min", 0.0) or 0.0) <= 1.0,
            "holdout_latency_return_ratio_min must be in [0,1]",
        )
        # holdout adverse-fill stress validation (intrabar STOP ambiguity)
        require(
            _finite(float(getattr(self, "holdout_adverse_fill_stress_mult", 0.0) or 0.0))
            and 0.0 <= float(getattr(self, "holdout_adverse_fill_stress_mult", 0.0) or 0.0) <= 1.0,
            "holdout_adverse_fill_stress_mult must be in [0,1]",
        )
        require(
            _finite(float(getattr(self, "holdout_adverse_fill_return_ratio_min", 0.0) or 0.0))
            and 0.0 <= float(getattr(self, "holdout_adverse_fill_return_ratio_min", 0.0) or 0.0) <= 1.0,
            "holdout_adverse_fill_return_ratio_min must be in [0,1]",
        )       
        # holdout permutation validation (Loop 3)
        require(int(self.holdout_perm_trials) >= 0, "holdout_perm_trials must be >= 0")
        require(
            _finite(float(self.holdout_perm_max_p)) and 0.0 <= float(self.holdout_perm_max_p) <= 1.0,
            "holdout_perm_max_p must be in [0,1]",
        )
        # holdout parameter robustness validation (Loop 5)
        require(int(self.param_robust_trials) >= 0, "param_robust_trials must be >= 0")
        require(
            _finite(float(self.param_robust_jitter)) and 0.0 <= float(self.param_robust_jitter) <= 1.0,
            "param_robust_jitter must be in [0,1]",
        )
        require(
            _finite(float(self.param_robust_min_pass_ratio)) and 0.0 <= float(self.param_robust_min_pass_ratio) <= 1.0,
            "param_robust_min_pass_ratio must be in [0,1]",
        )
        require(
            _finite(float(self.param_robust_return_ratio_min)) and 0.0 <= float(self.param_robust_return_ratio_min) <= 1.0,
            "param_robust_return_ratio_min must be in [0,1]",
        )
        require(
            _finite(float(self.param_robust_trade_ratio_min)) and 0.0 <= float(self.param_robust_trade_ratio_min) <= 1.0,
            "param_robust_trade_ratio_min must be in [0,1]",
        )
        # numeric safety
        require(_finite(float(self.profit_factor_cap)) and float(self.profit_factor_cap) > 0.0, "profit_factor_cap must be > 0")
        require(_finite(float(self.profit_factor_eps)) and float(self.profit_factor_eps) > 0.0, "profit_factor_eps must be > 0")
        require(int(self.pf_trade_damp_ref) > 0, "pf_trade_damp_ref must be > 0")
        rm = str(getattr(self, "return_mode", "absolute")).strip().lower()
        require(rm in ("absolute", "excess"), "return_mode must be one of: 'absolute', 'excess'")
        object.__setattr__(self, "return_mode", rm)

        require(int(self.return_trade_damp_ref) > 0, "return_trade_damp_ref must be > 0")
        require(
            _finite(float(self.min_pos_fold_ratio)) and 0.0 <= float(self.min_pos_fold_ratio) <= 1.0,
            "min_pos_fold_ratio must be in [0,1]",
        )
        require(_finite(float(self.min_t_stat)), "min_t_stat must be finite")
        require(
            _finite(float(self.max_p_value_adj)) and 0.0 <= float(self.max_p_value_adj) <= 1.0,
            "max_p_value_adj must be in [0,1]",
        )
        require(int(self.multiple_testing_trials) >= 0, "multiple_testing_trials must be >= 0")
        require(
            _finite(float(self.cscv_min_selected_fraction)) and 0.0 <= float(self.cscv_min_selected_fraction) <= 1.0,
            "cscv_min_selected_fraction must be in [0,1]",
        )
        require(
            _finite(float(self.cscv_min_oos_rank_median)) and 0.0 <= float(self.cscv_min_oos_rank_median) <= 1.0,
            "cscv_min_oos_rank_median must be in [0,1]",
        )
        pt = str(getattr(self, "pf_transform", "tanh")).strip().lower()
        require(pt in ("log", "tanh", "none"), "pf_transform must be one of: 'log', 'tanh', 'none'")
        object.__setattr__(self, "pf_transform", pt)
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

    holdout = int(getattr(cfg, "holdout_bars", 0) or 0)
    require(holdout >= 0, "holdout_bars must be >= 0")
    require(holdout < n, f"holdout_bars too large: holdout_bars={holdout} n_bars={n}")
    n_eff = n - holdout
    require(n_eff > 0, "Effective bars after holdout must be > 0")

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

        if test_end > n_eff:
            break
        if train_eff_end <= train_start:
            start += step
            continue

        splits.append(FoldSplit(train=slice(train_start, train_eff_end), test=slice(test_start, test_end)))
        start += step

    require(
        len(splits) > 0,
        "No walk-forward splits possible; increase n_bars or reduce train/test sizes "
        "(note: holdout_bars reduces usable bars).",
    )
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
    avg_holding_bars: float    # mean (exit_i - entry_i) in bars (diagnostic)
    fees_paid: float           # fraction of equity (simple model)
    funding_paid: float        # fraction of equity (simple model; constant funding optional)
    liquidations: int = 0      # (engine) number of liquidation events inside this slice
    liquidation_paid: float = 0.0  # (engine) liquidation fees as fraction of initial equity
    # Profit concentration diagnostics (helps reject "one lucky trade" strategies)
    top_profit_share: float = 0.0  # share of gross profits contributed by the best trade (0..1)
    profit_hhi: float = 0.0        # Herfindahl index over positive PnL shares (0..1)
@dataclass(frozen=True, slots=True)
class CandidateResult:
    strategy_id: str
    strategy_hash: str
    strategy_name: str
    strategy_dsl: str
    direction: str
    tags: Tuple[str, ...]
    complexity: int

    score: float
    fold_metrics: Tuple[FoldMetrics, ...]
    aggregate: Dict[str, Any]
    regime_filter: Dict[str, Any]

    # NEW (Loop 4): Full spec for reproducibility / replay
    spec: Dict[str, Any]
    spec_canonical: Any
    # NEW (Loop 13): In-memory spec object for fast validation/replay (not exported)
    spec_obj: Optional[StrategySpec] = None
    # NEW (Loop 4): Untouched final holdout evaluation (if enabled)
    holdout_metrics: Optional[FoldMetrics] = None
    holdout_aggregate: Optional[Dict[str, Any]] = None

@dataclass(frozen=True, slots=True)
class DebugCandidate:
    """Top candidates for debugging even if they are rejected by strict filters."""
    strategy_id: str
    strategy_hash: str
    direction: str
    tags: Tuple[str, ...]
    complexity: int
    score: float
    status: str                 # "accepted" | "rejected"
    reject_reason: str          # empty when accepted
    aggregate: Dict[str, Any]
    holdout_aggregate: Optional[Dict[str, Any]] = None

@dataclass(frozen=True, slots=True)
class MinerReport:
    evaluated: int
    accepted: int
    rejected: int
    folds: int

    reject_reasons: Dict[str, int]
    candidate_stats: Dict[str, Any]

    top_results: Tuple[CandidateResult, ...]
    debug_top: Tuple[DebugCandidate, ...]
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
# -----------------------------
# Holdout permutation helpers (Loop 3)
# -----------------------------
class _ShiftedSignal(Sequence[float]):
    """Circularly shift a signal inside [start, end) without allocating.

    Outside the window, returns the original base signal. This preserves boundary conditions
    (e.g., the backtester reading signal[start-1] for the first holdout execution bar).
    """

    __slots__ = ("_base", "_start", "_end", "_shift", "_n", "_seg_len")

    def __init__(self, base: Sequence[float], start: int, end: int, shift: int) -> None:
        self._base = base
        self._start = int(start)
        self._end = int(end)
        self._n = int(len(base))
        seg_len = int(end) - int(start)
        self._seg_len = int(seg_len)
        if seg_len <= 0:
            self._shift = 0
        else:
            self._shift = int(shift) % int(seg_len)

    def __len__(self) -> int:
        return int(self._n)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice):
            s, e, step = idx.indices(self._n)
            return [float(self[i]) for i in range(s, e, step)]

        i = int(idx)
        # support negative indexing
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError(i)

        # outside the shift window: keep base signal
        if i < self._start or i >= self._end or self._seg_len <= 0:
            return float(self._base[i])

        # inside the window: circular shift (right shift by `shift`)
        j = i - self._start
        src = (j - self._shift) % self._seg_len
        return float(self._base[self._start + src])
# -----------------------------
# Holdout latency helpers (execution delay)
# -----------------------------
class _DelayedSignal(Sequence[float]):
    """Delay a signal inside [start, end) by `delay` bars (NO wrap), without allocating.

    Outside the window, returns the original base signal. Inside the window:
      out[i] = base[i - delay] if (i - delay) is still inside [start, end)
      else 0.0

    This preserves fold/holdout independence: signals from before `start` never "spill" into the window.
    """

    __slots__ = ("_base", "_start", "_end", "_delay", "_n")

    def __init__(self, base: Sequence[float], start: int, end: int, delay: int) -> None:
        self._base = base
        self._start = int(start)
        self._end = int(end)
        self._n = int(len(base))
        d = int(delay)
        self._delay = int(d if d > 0 else 0)

    def __len__(self) -> int:
        return int(self._n)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, slice):
            s, e, step = idx.indices(self._n)
            return [float(self[i]) for i in range(s, e, step)]

        i = int(idx)
        # support negative indexing
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError(i)

        # outside the delay window: keep base signal
        if i < self._start or i >= self._end or self._delay <= 0:
            return float(self._base[i])

        src = int(i) - int(self._delay)
        if src < self._start or src >= self._end:
            return 0.0
        return float(self._base[src])
# -----------------------------
# Holdout time-concentration helpers (temporal clustering)
# -----------------------------
def _time_bucket_profit_concentration(
    ts_ms: Sequence[int],
    trades: Sequence[Any],
    start: int,
    end: int,
    *,
    bucket_ms: int,
) -> Dict[str, Any]:
    """Compute how concentrated *positive* PnL is across time buckets.

    We bucket trades by exit timestamp and sum *positive* PnL per bucket.
    This is designed as a cheap, high-signal guard against strategies that only
    work during one short volatility burst in the holdout.

    Returns a dict that is safe to JSON-serialize.
    """
    out: Dict[str, Any] = {
        "enabled": False,
        "bucket_ms": int(bucket_ms),
        "bucket_span": 0,
        "positive_buckets": 0,
        "gross_profit": 0.0,
        "top_profit_share": 0.0,
        "profit_hhi": 0.0,
    }
    try:
        bms = int(bucket_ms)
        if bms <= 0:
            return out
        s = int(start)
        e = int(end)
        if e <= s:
            return out
        if not ts_ms:
            return out
        if s < 0 or s >= len(ts_ms):
            return out
        # epoch-ms sanity (avoid synthetic/index-like timestamps)
        EPOCH_MS_MIN = 946_684_800_000
        if int(ts_ms[s]) < int(EPOCH_MS_MIN):
            return out
        e_last = min(len(ts_ms) - 1, max(s, e - 1))
        b0 = int(ts_ms[s]) // bms
        b1 = int(ts_ms[e_last]) // bms
        out["bucket_span"] = int(b1 - b0 + 1) if b1 >= b0 else 0

        # determine whether pnl is meaningful (engine + simple provide pnl, but be robust)
        use_pnl = False
        for t in trades:
            try:
                ei = int(getattr(t, "entry_i", -1))
                if ei < s or ei >= e:
                    continue
                if abs(float(getattr(t, "pnl", 0.0) or 0.0)) > 1e-12:
                    use_pnl = True
                    break
            except Exception:
                continue

        bucket_profit: Dict[int, float] = {}
        for t in trades:
            try:
                ei = int(getattr(t, "entry_i", -1))
                if ei < s or ei >= e:
                    continue
                xi = int(getattr(t, "exit_i", -1))
                if xi < s:
                    xi = s
                if xi >= e:
                    xi = e - 1
                if xi < 0 or xi >= len(ts_ms):
                    continue
                b = int(ts_ms[xi]) // bms
                val = float(getattr(t, "pnl", 0.0) or 0.0) if use_pnl else float(getattr(t, "ret", 0.0) or 0.0)
                if not _finite(val) or val <= 0.0:
                    continue
                bucket_profit[b] = float(bucket_profit.get(b, 0.0)) + float(val)
            except Exception:
                continue

        pos_vals = [float(v) for v in bucket_profit.values() if float(v) > 0.0 and _finite(float(v))]
        gp = float(sum(pos_vals)) if pos_vals else 0.0
        out["gross_profit"] = float(gp)
        out["positive_buckets"] = int(len(pos_vals))

        if gp > 0.0 and pos_vals:
            top_share = max(pos_vals) / gp
            if not _finite(float(top_share)):
                top_share = 0.0
            top_share = max(0.0, min(1.0, float(top_share)))
            out["top_profit_share"] = float(top_share)

            hhi = sum((float(v) / gp) ** 2 for v in pos_vals) if gp > 0.0 else 0.0
            if not _finite(float(hhi)):
                hhi = 0.0
            hhi = max(0.0, min(1.0, float(hhi)))
            out["profit_hhi"] = float(hhi)

        out["enabled"] = True
        return out
    except Exception:
        return out
# -----------------------------
# Holdout regime-concentration helpers (market-condition dependence)
# -----------------------------
def _regime_bucket_profit_concentration(
    reg: RegimeStore,
    trades: Sequence[Any],
    start: int,
    end: int,
    *,
    regime_name: str,
) -> Dict[str, Any]:
    """Compute how concentrated *positive* PnL is across regime buckets.

    We bucket trades by the regime label at their EXIT bar and sum *positive* PnL per bucket.
    This is a cheap guard against strategies that only work in one market condition
    (e.g., only during high-volatility spikes), which is typically where cost assumptions break.

    Returns a dict that is safe to JSON-serialize.
    """
    out: Dict[str, Any] = {
        "enabled": False,
        "regime": str(regime_name),
        "labels": [-1, 0, 1],
        "positive_buckets": 0,
        "gross_profit": 0.0,
        "top_profit_share": 0.0,
        "profit_hhi": 0.0,
        "gross_profit_by_label": {-1: 0.0, 0: 0.0, 1: 0.0},
        "net_by_label": {-1: 0.0, 0: 0.0, 1: 0.0},
        "trades_by_label": {-1: 0, 0: 0, 1: 0},
    }
    try:
        s = int(start)
        e = int(end)
        if e <= s:
            return out
        if reg is None:
            return out
        vec = reg.regime(str(regime_name))
        if not vec:
            return out

        # determine whether pnl is meaningful (engine + simple provide pnl, but be robust)
        use_pnl = False
        for t in trades:
            try:
                ei = int(getattr(t, "entry_i", -1))
                if ei < s or ei >= e:
                    continue
                if abs(float(getattr(t, "pnl", 0.0) or 0.0)) > 1e-12:
                    use_pnl = True
                    break
            except Exception:
                continue

        bucket_profit: Dict[int, float] = {-1: 0.0, 0: 0.0, 1: 0.0}
        bucket_net: Dict[int, float] = {-1: 0.0, 0: 0.0, 1: 0.0}
        bucket_trades: Dict[int, int] = {-1: 0, 0: 0, 1: 0}

        def _lab(x: float) -> int:
            xf = float(x)
            if not _finite(xf):
                return 0
            if xf > 0.5:
                return 1
            if xf < -0.5:
                return -1
            return 0

        for t in trades:
            try:
                ei = int(getattr(t, "entry_i", -1))
                if ei < s or ei >= e:
                    continue
                xi = int(getattr(t, "exit_i", -1))
                if xi < s:
                    xi = s
                if xi >= e:
                    xi = e - 1
                if xi < 0 or xi >= len(vec):
                    continue

                lbl = _lab(float(vec[xi]))
                bucket_trades[lbl] = int(bucket_trades.get(lbl, 0)) + 1

                val = float(getattr(t, "pnl", 0.0) or 0.0) if use_pnl else float(getattr(t, "ret", 0.0) or 0.0)
                if not _finite(val):
                    continue
                bucket_net[lbl] = float(bucket_net.get(lbl, 0.0)) + float(val)
                if val > 0.0:
                    bucket_profit[lbl] = float(bucket_profit.get(lbl, 0.0)) + float(val)
            except Exception:
                continue

        pos_vals = [float(v) for v in bucket_profit.values() if float(v) > 0.0 and _finite(float(v))]
        gp = float(sum(pos_vals)) if pos_vals else 0.0
        out["gross_profit"] = float(gp)
        out["positive_buckets"] = int(len(pos_vals))
        out["gross_profit_by_label"] = {int(k): float(v) for k, v in bucket_profit.items()}
        out["net_by_label"] = {int(k): float(v) for k, v in bucket_net.items()}
        out["trades_by_label"] = {int(k): int(v) for k, v in bucket_trades.items()}

        if gp > 0.0 and pos_vals:
            top_share = max(pos_vals) / gp
            if not _finite(float(top_share)):
                top_share = 0.0
            top_share = max(0.0, min(1.0, float(top_share)))
            out["top_profit_share"] = float(top_share)

            hhi = sum((float(v) / gp) ** 2 for v in pos_vals) if gp > 0.0 else 0.0
            if not _finite(float(hhi)):
                hhi = 0.0
            hhi = max(0.0, min(1.0, float(hhi)))
            out["profit_hhi"] = float(hhi)

        out["enabled"] = True
        return out
    except Exception:
        return out
# -----------------------------
# Holdout segment-consistency helpers (temporal stability)
# -----------------------------
def _holdout_segment_consistency(
    equity_curve: Sequence[float],
    start: int,
    end: int,
    *,
    segments: int,
) -> Dict[str, Any]:
    """Compute temporal consistency stats by splitting [start, end) into segments.

    This is a cheap guard against strategies that only "work" in a short sub-window
    of the holdout but would be unbearable to trade live.

    Returns a dict safe to JSON-serialize.
    """
    out: Dict[str, Any] = {
        "enabled": False,
        "segments": int(segments),
        "segments_eff": 0,
        "bars_per_segment_min": 0,
        "returns": [],
        "pos_segments": 0,
        "pos_ratio": 0.0,
        "min_return": 0.0,
    }
    try:
        nseg = int(segments)
        if nseg <= 1:
            return out
        s = int(start)
        e = int(end)
        if e <= s:
            return out
        if not equity_curve:
            return out

        # Clamp bounds to the curve length (engine adapters may return shorter curves in edge-cases).
        if s < 0:
            s = 0
        if e > len(equity_curve):
            e = int(len(equity_curve))
        if e <= s:
            return out

        total = int(e - s)
        if total <= 0:
            return out

        # Segment boundaries using integer division (covers full span; segments differ by <= 1 bar).
        bounds = [int(s + (total * j) // nseg) for j in range(nseg + 1)]
        bounds[-1] = int(e)

        rets: List[float] = []
        seg_lens: List[int] = []
        pos = 0

        for j in range(nseg):
            a = int(bounds[j])
            b = int(bounds[j + 1])
            ln = int(b - a)
            if ln <= 0:
                continue
            seg_lens.append(int(ln))

            eq0 = float(equity_curve[a])
            eq1 = float(equity_curve[b - 1])

            r = (eq1 / eq0 - 1.0) if _finite(eq0) and eq0 > 0.0 and _finite(eq1) else 0.0
            if not _finite(float(r)):
                r = 0.0
            r = float(r)
            rets.append(r)
            if r >= 0.0:
                pos += 1

        seg_eff = int(len(rets))
        out["segments_eff"] = int(seg_eff)
        out["returns"] = [float(x) for x in rets]
        out["pos_segments"] = int(pos)
        out["pos_ratio"] = float(pos / seg_eff) if seg_eff > 0 else 0.0
        out["min_return"] = float(min(rets) if rets else 0.0)
        out["bars_per_segment_min"] = int(min(seg_lens) if seg_lens else 0)

        out["enabled"] = bool(seg_eff >= 2)
        return out
    except Exception:
        return out

# -----------------------------
# Holdout parameter robustness helpers (Loop 5)
# -----------------------------
def _is_period_like_key(k: str) -> bool:
    """Heuristic: detect indicator params that are usually integer periods/windows."""
    s = str(k).strip().lower()
    if not s:
        return False
    needles = (
        "period",
        "len",
        "length",
        "window",
        "lookback",
        "lb",
        "bars",
        "n",
        "k",
    )
    return any(n in s for n in needles)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _jitter_int(v: int, rng: random.Random, jitter: float, *, min_v: int = 1, max_v: int = 500) -> int:
    base = int(v)
    if base <= 0:
        base = 1
    j = float(jitter)
    if not (_finite(j) and j > 0.0):
        return int(_clamp(float(base), float(min_v), float(max_v)))
    step = max(1, int(round(abs(base) * j)))
    delta = int(step) if rng.random() < 0.5 else -int(step)
    out = int(base + delta)
    out = int(_clamp(float(out), float(min_v), float(max_v)))
    # Ensure change when possible
    if out == base:
        out = int(_clamp(float(base + (1 if delta >= 0 else -1)), float(min_v), float(max_v)))
    return int(out)


def _jitter_float(v: float, rng: random.Random, jitter: float, *, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    base = float(v)
    j = float(jitter)
    if not (_finite(base) and _finite(j) and j > 0.0):
        return float(base)
    u = rng.uniform(-j, j)
    if abs(base) < 1e-12:
        out = float(base + u)
    else:
        out = float(base * (1.0 + u))
    if lo is not None:
        out = max(float(lo), float(out))
    if hi is not None:
        out = min(float(hi), float(out))
    if not _finite(out):
        return float(base)
    return float(out)


def _perturb_feature_ref(fr: FeatureRef, rng: random.Random, jitter: float, *, p: float = 0.6) -> FeatureRef:
    params = dict(fr.params or {})
    changed = False
    for k, v in list(params.items()):
        if rng.random() > float(p):
            continue
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int):
            if _is_period_like_key(str(k)):
                nv = _jitter_int(int(v), rng, jitter, min_v=2, max_v=500)
            else:
                # non-period ints: treat as small steps
                nv = _jitter_int(int(v), rng, max(0.05, float(jitter) * 0.5), min_v=-10_000, max_v=10_000)
            if nv != int(v):
                params[k] = int(nv)
                changed = True
        elif isinstance(v, float):
            nvf = _jitter_float(float(v), rng, jitter)
            if float(nvf) != float(v):
                params[k] = float(nvf)
                changed = True

    # If nothing changed (common when params dict is empty), return original.
    if not changed:
        return fr
    return replace(fr, params=params)


def _perturb_rhs(rhs: Any, rng: random.Random, jitter: float, *, p: float = 0.6) -> Any:
    if rhs is None:
        return None
    if isinstance(rhs, FeatureRef):
        return _perturb_feature_ref(rhs, rng, jitter, p=p)
    if isinstance(rhs, tuple) and len(rhs) == 2:
        a, b = rhs
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if rng.random() <= float(p):
                na = float(_jitter_float(float(a), rng, jitter))
            else:
                na = float(a)
            if rng.random() <= float(p):
                nb = float(_jitter_float(float(b), rng, jitter))
            else:
                nb = float(b)
            if na <= nb:
                return (float(na), float(nb))
            return (float(nb), float(na))
        return rhs
    if isinstance(rhs, int):
        if rng.random() > float(p):
            return rhs
        return int(_jitter_int(int(rhs), rng, max(0.05, float(jitter) * 0.5), min_v=-10_000, max_v=10_000))
    if isinstance(rhs, float):
        if rng.random() > float(p):
            return rhs
        return float(_jitter_float(float(rhs), rng, jitter))
    return rhs


def _perturb_expr(expr: Any, rng: random.Random, jitter: float, *, p: float = 0.6) -> Any:
    if isinstance(expr, Condition):
        lhs2 = _perturb_feature_ref(expr.lhs, rng, jitter, p=p)
        rhs2 = _perturb_rhs(expr.rhs, rng, jitter, p=p)
        if lhs2 is expr.lhs and rhs2 is expr.rhs:
            return expr
        return replace(expr, lhs=lhs2, rhs=rhs2)
    if isinstance(expr, AllOf):
        terms2 = tuple(_perturb_expr(t, rng, jitter, p=p) for t in expr.terms)
        if all(t1 is t2 for t1, t2 in zip(expr.terms, terms2)):
            return expr
        return replace(expr, terms=terms2)
    if isinstance(expr, AnyOf):
        terms2 = tuple(_perturb_expr(t, rng, jitter, p=p) for t in expr.terms)
        if all(t1 is t2 for t1, t2 in zip(expr.terms, terms2)):
            return expr
        return replace(expr, terms=terms2)
    if isinstance(expr, Not):
        t2 = _perturb_expr(expr.term, rng, jitter, p=p)
        if t2 is expr.term:
            return expr
        return replace(expr, term=t2)
    return expr


def _perturb_entry_logic(logic: Any, rng: random.Random, jitter: float, *, p: float = 0.6) -> Any:
    if isinstance(logic, EventSequence):
        steps2: List[SequenceStep] = []
        changed = False
        for st in logic.steps:
            e2 = _perturb_expr(st.expr, rng, jitter, p=p)
            if e2 is not st.expr:
                changed = True
            steps2.append(replace(st, expr=e2))
        if not changed:
            return logic
        return replace(logic, steps=tuple(steps2))
    return _perturb_expr(logic, rng, jitter, p=p)


def _perturb_stop_spec(stop: StopSpec, rng: random.Random, jitter: float, *, p: float = 0.6) -> StopSpec:
    k = str(stop.kind)
    if k == "percent":
        if rng.random() > float(p):
            return stop
        sp = float(_jitter_float(float(stop.stop_pct), rng, jitter, lo=1e-4, hi=0.49))
        return replace(stop, stop_pct=float(sp))
    if k == "atr":
        ap = int(stop.atr_period)
        am = float(stop.atr_mult)
        if rng.random() <= float(p):
            ap = int(_jitter_int(int(ap), rng, jitter, min_v=2, max_v=300))
        if rng.random() <= float(p):
            am = float(_jitter_float(float(am), rng, jitter, lo=0.05, hi=20.0))
        if ap == int(stop.atr_period) and am == float(stop.atr_mult):
            return stop
        return replace(stop, atr_period=int(ap), atr_mult=float(am))
    if k == "structure":
        ml = int(stop.ms_left)
        mr = int(stop.ms_right)
        bb = float(stop.buffer_bps)
        if rng.random() <= float(p):
            ml = int(_jitter_int(int(ml), rng, max(0.10, float(jitter)), min_v=1, max_v=20))
        if rng.random() <= float(p):
            mr = int(_jitter_int(int(mr), rng, max(0.10, float(jitter)), min_v=1, max_v=20))
        if rng.random() <= float(p):
            bb = float(_jitter_float(float(bb), rng, jitter, lo=0.0, hi=500.0))
        if ml == int(stop.ms_left) and mr == int(stop.ms_right) and bb == float(stop.buffer_bps):
            return stop
        return replace(stop, ms_left=int(ml), ms_right=int(mr), buffer_bps=float(bb))
    # none / unknown kinds: leave unchanged
    return stop


def _perturb_tp_spec(tp: TakeProfitSpec, rng: random.Random, jitter: float, *, p: float = 0.6) -> TakeProfitSpec:
    k = str(tp.kind)
    if k == "percent":
        if rng.random() > float(p):
            return tp
        pct = float(_jitter_float(float(tp.tp_pct), rng, jitter, lo=1e-4, hi=0.49))
        return replace(tp, tp_pct=float(pct))
    if k == "rr":
        if rng.random() > float(p):
            return tp
        rr = float(_jitter_float(float(tp.rr), rng, jitter, lo=0.05, hi=50.0))
        return replace(tp, rr=float(rr))
    if k == "atr":
        ap = int(tp.atr_period)
        am = float(tp.atr_mult)
        if rng.random() <= float(p):
            ap = int(_jitter_int(int(ap), rng, jitter, min_v=2, max_v=300))
        if rng.random() <= float(p):
            am = float(_jitter_float(float(am), rng, jitter, lo=0.05, hi=20.0))
        if ap == int(tp.atr_period) and am == float(tp.atr_mult):
            return tp
        return replace(tp, atr_period=int(ap), atr_mult=float(am))
    return tp


def _perturb_strategy_spec(spec: StrategySpec, rng: random.Random, jitter: float, *, p: float = 0.6) -> StrategySpec:
    """Return a perturbed (but still valid) StrategySpec.

    We jitter only numeric parameters that are likely to be overfit knobs:
    - indicator params (periods/thresholds)
    - condition rhs constants
    - stop/take-profit params

    NOTE: We intentionally do NOT perturb direction, operators, shifts, or the regime filter.
    """
    entry2 = _perturb_entry_logic(spec.entry, rng, jitter, p=p)
    exit2 = _perturb_entry_logic(spec.exit, rng, jitter, p=p) if spec.exit is not None else None
    stop2 = _perturb_stop_spec(spec.stop, rng, jitter, p=p)
    tp2 = _perturb_tp_spec(spec.take_profit, rng, jitter, p=p)

    out = replace(spec, entry=entry2, exit=exit2, stop=stop2, take_profit=tp2)
    # If nothing changed (rare), force a deterministic small stop/tp tweak.
    if out.hash() == spec.hash() and float(jitter) > 0.0:
        try:
            # Prefer stop_pct if available; else RR; else ATR mult.
            if str(spec.stop.kind) == "percent":
                sp = float(_jitter_float(float(spec.stop.stop_pct), rng, max(0.05, float(jitter) * 0.5), lo=1e-4, hi=0.49))
                out = replace(out, stop=replace(spec.stop, stop_pct=float(sp)))
            elif str(spec.take_profit.kind) == "rr":
                rr = float(_jitter_float(float(spec.take_profit.rr), rng, max(0.05, float(jitter) * 0.5), lo=0.05, hi=50.0))
                out = replace(out, take_profit=replace(spec.take_profit, rr=float(rr)))
            elif str(spec.stop.kind) == "atr":
                am = float(_jitter_float(float(spec.stop.atr_mult), rng, max(0.05, float(jitter) * 0.5), lo=0.05, hi=20.0))
                out = replace(out, stop=replace(spec.stop, atr_mult=float(am)))
        except Exception:
            # If forcing fails, just return the original (the robustness tester will treat
            # it as a "no-op" perturbation).
            return spec
    return out

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
    ret: float
    fees: float  # fees paid in equity units
    pnl: float = 0.0  # net PnL in equity units (after fees) for PF weighting




@dataclass(slots=True)
class BacktestOutput:
    equity_curve: List[float]
    trades: List[_Trade]
    fees_paid: float
    funding_paid: float
    # Perp realism telemetry (engine backend).
    liquidation_paid: float = 0.0  # fraction of initial equity
    liquidations: int = 0

def _compute_fold_metrics(
    fold_index: int,
    start: int,
    end: int,
    out: BacktestOutput,
    *,
    pf_cap: float = 1000.0,
    pf_eps: float = 1e-12,
) -> FoldMetrics:
    eq = out.equity_curve
    n_eq = len(eq)
    eq_window = eq[start:end] if end <= n_eq else eq[start:]

    # IMPORTANT (no "boundary fee leak"):
    # The first tradable OPEN inside [start, end) can be at index `start`, triggered by a signal at `start-1`.
    # If we measure returns from eq[start] (bar-end), we silently forgive entry costs paid at the window
    # boundary and inflate returns / understate drawdowns. Baseline must therefore be eq[start-1] when available.
    if start > 0 and (start - 1) < n_eq:
        eq0 = float(eq[start - 1])
    else:
        eq0 = float(eq_window[0]) if eq_window else 1.0

    eq_last = float(eq_window[-1]) if eq_window else float(eq0)
    net_return = (eq_last / eq0 - 1.0) if _finite(eq0) and eq0 > 0.0 and _finite(eq_last) else 0.0

    # max drawdown (baseline anchored at eq0, so boundary costs count)
    peak = float(eq0) if _finite(eq0) and float(eq0) > 0.0 else (float(eq_window[0]) if eq_window else 1.0)
    max_dd = 0.0
    for v in eq_window:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # trades (count only entries within window)
    trades = [t for t in out.trades if start <= t.entry_i < end]
    ntr = len(trades)
    if ntr > 0:
        # "bars held" diagnostic. Exit can occur same bar => 0.
        avg_hold = _mean([float(int(t.exit_i) - int(t.entry_i)) for t in trades])
    else:
        avg_hold = 0.0
    # Prefer pnl-weighted PF if pnl is available (non-zero), fallback to ret-based PF
    use_pnl = any(abs(float(getattr(t, "pnl", 0.0))) > 1e-12 for t in trades)

    if ntr > 0:
        if use_pnl:
            wins = sum(1 for t in trades if float(getattr(t, "pnl", 0.0)) > 0.0)
            gross_profit = sum(float(getattr(t, "pnl", 0.0)) for t in trades if float(getattr(t, "pnl", 0.0)) > 0.0)
            gross_loss = -sum(float(getattr(t, "pnl", 0.0)) for t in trades if float(getattr(t, "pnl", 0.0)) < 0.0)
        else:
            wins = sum(1 for t in trades if float(t.ret) > 0.0)
            gross_profit = sum(float(t.ret) for t in trades if float(t.ret) > 0.0)
            gross_loss = -sum(float(t.ret) for t in trades if float(t.ret) < 0.0)

        win_rate = wins / ntr
        # Profit factor is numerically unstable when there are no losing trades (gross_loss == 0) -> +inf.
        # Clamp to keep scoring + JSON exports stable (see ScoringConfig.profit_factor_cap).
        pf_cap_f = float(pf_cap)
        pf_eps_f = float(pf_eps)
        if not _finite(pf_cap_f) or pf_cap_f <= 0.0:
            pf_cap_f = 1000.0
        if not _finite(pf_eps_f) or pf_eps_f <= 0.0:
            pf_eps_f = 1e-12

        pf_raw = (gross_profit / max(gross_loss, pf_eps_f)) if gross_profit > 0.0 else 0.0
        if not _finite(float(pf_raw)) or float(pf_raw) < 0.0:
            pf_raw = 0.0
        profit_factor = min(float(pf_raw), pf_cap_f)
        avg_trade_return = _mean([float(t.ret) for t in trades])
        # Profit concentration diagnostics (positive PnL share)
        top_profit_share = 0.0
        profit_hhi = 0.0
        if float(gross_profit) > float(pf_eps_f):
            if use_pnl:
                pos_vals = [float(getattr(t, "pnl", 0.0)) for t in trades if float(getattr(t, "pnl", 0.0)) > 0.0]
            else:
                pos_vals = [float(t.ret) for t in trades if float(t.ret) > 0.0]
            if pos_vals:
                gp = float(gross_profit)
                top_profit_share = (max(pos_vals) / gp) if gp > 0.0 else 0.0
                if not _finite(float(top_profit_share)):
                    top_profit_share = 0.0
                top_profit_share = max(0.0, min(1.0, float(top_profit_share)))
                profit_hhi = sum((float(p) / gp) ** 2 for p in pos_vals) if gp > 0.0 else 0.0
                if not _finite(float(profit_hhi)):
                    profit_hhi = 0.0
                profit_hhi = max(0.0, min(1.0, float(profit_hhi)))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0
        top_profit_share = 0.0
        profit_hhi = 0.0
    # exposure: count bars between entry and exit indices
    bars_in_pos = 0
    for t in trades:
        ei = max(start, int(t.entry_i))
        xi = min(end, int(t.exit_i) + 1)
        if xi > ei:
            bars_in_pos += (xi - ei)
    exposure = (bars_in_pos / (end - start)) if (end - start) > 0 else 0.0

    turnover = (ntr / (end - start)) * 1000.0 if (end - start) > 0 else 0.0
    # Perp realism telemetry (engine backend)
    liq = int(getattr(out, "liquidations", 0) or 0)
    liq_paid = float(getattr(out, "liquidation_paid", 0.0) or 0.0)
    return FoldMetrics(
        fold_index=int(fold_index),
        start=int(start),
        end=int(end),
        trades=int(ntr),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        net_return=float(net_return),
        max_drawdown=float(max_dd),
        avg_trade_return=float(avg_trade_return),
        exposure=float(exposure),
        turnover=float(turnover),
        avg_holding_bars=float(avg_hold),
        fees_paid=float(out.fees_paid),
        funding_paid=float(out.funding_paid),
        liquidations=int(liq),
        liquidation_paid=float(liq_paid),
        top_profit_share=float(top_profit_share),
        profit_hhi=float(profit_hhi),
    )


# -----------------------------
# Backtest Engine adapter (optional)
# -----------------------------
def _try_backtest_engine(
    series: OhlcvSeries,
    compiled: CompiledStrategy,
    direction: str,
    start: int,
    end: int,
    *,
    cfg: Optional[Any] = None,
) -> Optional[BacktestOutput]:
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

    # Try to pass a proper Side enum if the engine exposes it (avoid passing a string into side=...).
    # Important: if the engine expects Side, passing a raw string can silently flip LONG/SHORT logic.
    side_arg: Any = None
    try:
        d = str(direction).lower().strip()
        if hasattr(BE, "Side"):
            if d in ("long", "buy", "bull"):
                side_arg = BE.Side.LONG
            elif d in ("short", "sell", "bear"):
                side_arg = BE.Side.SHORT
            # If we couldn't map, keep None -> engine can use 'direction' instead.
        else:
            # Engines without a Side enum may accept a string side.
            side_arg = d
    except Exception:
        side_arg = None
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
        "side": side_arg,
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
    # Backtest config (for engine backends that accept one).
    # Only include when non-None so signature-matching doesn't pass None into strict engines.
    if cfg is not None:
        kwargs_pool.update(
            {
                "cfg": cfg,
                "config": cfg,
                "bt_cfg": cfg,
                "backtest_config": cfg,
            }
        )

    def _call_with_signature(fn: Any) -> Any:
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
    liquidation_paid = 0.0
    liquidations = 0
    
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
        liquidation_paid = float(res.get("liquidation_paid", 0.0) or 0.0)
        liquidations = int(res.get("liquidations", 0) or 0)
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
        if hasattr(res, "liquidation_paid"):
            liquidation_paid = float(getattr(res, "liquidation_paid") or 0.0)
        if hasattr(res, "liquidations"):
            liquidations = int(getattr(res, "liquidations") or 0)
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
                    pnl = float(t.get("pnl", t.get("pnl_net", 0.0) or 0.0))
                    parsed_trades.append(_Trade(ei, xi, ep, xp, r, float(t.get("fees", 0.0) or 0.0), pnl=float(pnl)))       
                else:
                    ei = int(getattr(t, "entry_i", getattr(t, "entry_idx", -1)))
                    xi = int(getattr(t, "exit_i", getattr(t, "exit_idx", -1)))
                    ep = float(getattr(t, "entry_price", getattr(t, "entry", _NAN)))
                    xp = float(getattr(t, "exit_price", getattr(t, "exit", _NAN)))
                    r = float(getattr(t, "ret", getattr(t, "return", 0.0)))
                    pnl = float(getattr(t, "pnl", getattr(t, "pnl_net", 0.0) or 0.0))
                    parsed_trades.append(_Trade(ei, xi, ep, xp, r, float(getattr(t, "fees", 0.0) or 0.0), pnl=float(pnl)))      
        except Exception:
            parsed_trades = []

    return BacktestOutput(
        equity_curve=eq,
        trades=parsed_trades,
        fees_paid=float(fees_paid),
        funding_paid=float(funding_paid),
        liquidation_paid=float(liquidation_paid),
        liquidations=int(liquidations),
    )

def _select_backtest_backend(mode: str) -> Tuple[str, Dict[str, Any]]:
    """Pick the Backtest_Engine backend for the whole run (required).

    Why this exists:
    - In strategy mining, silently mixing backtest models (engine for some, simple for others)
      breaks comparability and creates false positives.
    - For "paper/live"-grade outputs we therefore **require** the engine backend and fail fast
      if it cannot be imported or does not expose a miner-compatible entrypoint

    Returns:
        (backend, info)
        backend is: "engine"
        info is a JSON-able dict for reporting/debugging.
    """
    m = str(mode).strip().lower()
    # Backwards-compatible alias: "auto" => "engine"
    if m == "auto":
        m = "engine"
    require(m == "engine", f"backtest_mode invalid: {mode!r} (engine-only)")

    info: Dict[str, Any] = {
        "requested_mode": str(mode).strip().lower(),
        "selected_backend": None,
        "engine_import_ok": False,
        "engine_api": {"functions": [], "classes": [], "has_backtest_target_position": False},
        "note": "",
    }

    # Engine is mandatory.
    try:
        import Backtest_Engine as BE  # type: ignore
        info["engine_import_ok"] = True
    except Exception as e:
        info["note"] = f"Backtest_Engine import failed: {type(e).__name__}"
        raise ValidationError(
            "Backtest_Engine is required (engine-only; no simple fallback). "
            "Fix the import error and re-run."
        ) from e

    # Detect a compatible public entrypoint.
    fn_names = ["run_backtest", "backtest", "simulate", "simulate_signals", "run"]
    cls_names = ["BacktestEngine", "Engine", "Simulator"]

    for name in fn_names:
        if hasattr(BE, name):
            info["engine_api"]["functions"].append(name)
    for name in cls_names:
        if hasattr(BE, name):
            info["engine_api"]["classes"].append(name)

    info["engine_api"]["has_backtest_target_position"] = bool(getattr(BE, "backtest_target_position", None))

    if info["engine_api"]["functions"] or info["engine_api"]["classes"]:
        info["selected_backend"] = "engine"
        return "engine", info

    # Engine exists but isn't wired for miner yet.
    if info["engine_api"]["has_backtest_target_position"]:
        info["note"] = (
            "Backtest_Engine is present but only exposes backtest_target_position(). "
            "Miner_Search expects run_backtest()/backtest()/simulate() (or an Engine class) "
            "for entry/exit-signal based strategies."
        )
    else:
        info["note"] = "Backtest_Engine is present but exposes no compatible entrypoint for Miner_Search."

    raise ValidationError(
        "Backtest_Engine is present but has no miner-compatible entrypoint. "
        "Expose run_backtest(series, entry, exit, ...) (or equivalent) in Backtest_Engine."
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

        n_total = len(series.ts_ms)
        self.splits = make_walkforward_splits(n_total, wf_cfg)
        # Total out-of-sample (test) bars across folds (used for holdout trade sample-size gating).
        self._total_test_bars = sum(int(s.test.stop) - int(s.test.start) for s in self.splits)
        # Untouched final holdout slice (never part of folds), optional.
        self.holdout_slice: Optional[slice] = None
        hb = int(getattr(wf_cfg, "holdout_bars", 0) or 0)
        # Fail-fast: "holdout_required" must not silently degrade into "no holdout".
        # If you claim holdout is required, you MUST reserve a holdout slice.
        if bool(getattr(score_cfg, "holdout_required", False)) and hb <= 0:
            raise ValidationError(
                "ScoringConfig.holdout_required=True, but WalkForwardConfig.holdout_bars==0. "
                "Set holdout_bars>0 (recommended) or set holdout_required=False."
            )

        if hb > 0:
            require(hb < n_total, "holdout_bars must be < total bars")
            self.holdout_slice = slice(int(n_total - hb), int(n_total))
                # Backtest backend selection (engine-only; fail-fast if unavailable).
        self._bt_backend, self._bt_backend_info = _select_backtest_backend(search_cfg.backtest_mode)
        self._bt_backend_counts: Dict[str, int] = {"engine": 0}
        # Separate counters for stress-test re-runs (kept separate so we can audit compute cost).
        self._bt_backend_counts_stress: Dict[str, int] = {"engine": 0}

        # Build a single Backtest_Engine.BacktestConfig upfront.
        # Critical: cost assumptions MUST match ScoringConfig; otherwise CLI flags (fee/spread/slip)
        # silently stop affecting results.
        import Backtest_Engine as BE  # type: ignore

        _cfg = BE.BacktestConfig(
            fee_rate_taker=float(score_cfg.fee_bps) / 10000.0,
            spread_bps=float(score_cfg.spread_bps),
            slippage_bps=float(score_cfg.slippage_bps),
            adverse_fill_slip_mult=float(getattr(score_cfg, "adverse_fill_slip_mult", 0.0) or 0.0),
            funding_period_ms=int(getattr(score_cfg, "funding_period_ms", 28_800_000) or 28_800_000),
            funding_rate_per_period=float(getattr(score_cfg, "funding_bps", 0.0) or 0.0) / 10000.0,
            store_trades=True,
            store_equity_curve=True,
            close_on_end=True,
        )
        overrides = dict(getattr(search_cfg, "engine_cfg_overrides", {}) or {})
        if overrides:
            try:
                _cfg = replace(_cfg, **overrides)
            except Exception as e:
                raise ValidationError(f"Invalid SearchConfig.engine_cfg_overrides: {e}") from e
        self._engine_cfg: Any = _cfg

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
    def _entry_key(self, spec: StrategySpec) -> Any:
        """
        Key used for "variants per entry" cap.
 
        Critical: If this key ignores exit, then max_variants_per_entry=1
        will almost always evaluate ONLY exit=None for each entry template,
        which causes undertrading (positions stay open) and starves exit logic.
 
        entry_key_mode:
          - "entry":                   (direction, entry)
          - "entry_exit":              (direction, entry, exit)
          - "entry_exit_time":         (direction, entry, exit, time_stop, cooldown)
          - "entry_exit_time_regime":  (direction, entry, exit, time_stop, cooldown, regime_filter)
  
        Stop/TP are intentionally ignored here to prevent combinatorial explosion.
        """
        # PERF: Hot loop. Avoid canonical_tuple() here.
        mode = str(getattr(self.search_cfg, "entry_key_mode", "entry_exit")).strip().lower()
        if mode not in ("entry", "entry_exit", "entry_exit_time", "entry_exit_time_regime"):
            mode = "entry_exit"

        def _logic_key(obj: Any) -> Any:
            if obj is None:
                return None
            # EventSequence has `.steps`; ConditionExpr does not.
            if hasattr(obj, "steps"):
                return self.compiler._key_sequence(obj)  # type: ignore[arg-type]
            return self.compiler._key_expr(obj)  # type: ignore[arg-type]
 
        def _regime_key(rf: Any) -> Any:
            # Keep regime key cheap: only structural params, not the evaluated mask.
            if rf is None:
                return None
            all_of = getattr(rf, "all_of", None)
            if not all_of:
                return None
            try:
                return ("RF", tuple(sorted(self.compiler._key_regime_cond(c) for c in all_of)))
            except Exception:
                # Fallback (should be rare): still keep it hashable.
                try:
                    return ("RF_REPR", repr(rf))
                except Exception:
                    return ("RF", "unkeyable")
        parts: List[Any] = [spec.direction, _logic_key(spec.entry)]
        if mode in ("entry_exit", "entry_exit_time", "entry_exit_time_regime"):
            parts.append(_logic_key(spec.exit))
        if mode in ("entry_exit_time", "entry_exit_time_regime"):
            parts.append(int(getattr(spec, "time_stop_bars", 0) or 0))
            parts.append(int(getattr(spec, "cooldown_bars", 0) or 0))
        if mode == "entry_exit_time_regime":
            parts.append(_regime_key(getattr(spec, "regime_filter", None)))
        # Use tuple directly (fast dict key), no sha1 needed.
        return tuple(parts)

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

        entry_counts: Dict[Any, int] = {}
        family_counts: Dict[str, int] = {}

        for spec in self.space.iter_strategies():
            self._cand_stats["space_seen"] += 1

            if mode == "sample":
                if self._rng.random() > p:
                    self._cand_stats["sampled_out"] += 1
                    continue
                self._cand_stats["sampled_in"] += 1

            if div_family:
                fb = self._family_bucket(spec)
                fc = family_counts.get(fb, 0)
                if fc >= max_per_family:
                    self._cand_stats["skipped_family_cap"] += 1
                    continue
            if div_entry:
                ek = self._entry_key(spec)
                c = entry_counts.get(ek, 0)
                if c >= max_per_entry:
                    self._cand_stats["skipped_entry_cap"] += 1
                    continue
            # accept
            if div_entry:
                entry_counts[ek] = c + 1
            if div_family:
                family_counts[fb] = family_counts.get(fb, 0) + 1

            self._cand_stats["yielded"] += 1
            yield spec

    # ---- Entry filters with reason ----
    def _passes_entry_filters(self, entry: Sequence[float]) -> Tuple[bool, str, List[int], int]:
        sc = self.score_cfg

        total_entries = 0
        fold_entries: List[int] = []
        for split in self.splits:
            s = int(split.test.start)  # type: ignore[arg-type]
            e = int(split.test.stop)   # type: ignore[arg-type]
            cnt = 0
            for i in range(s, e):
                if entry[i] > 0.5:
                    cnt += 1
            fold_entries.append(int(cnt))
            if cnt < int(sc.min_entries_per_fold):
                return False, "min_entries_per_fold", fold_entries, int(total_entries + cnt)
            total_entries += cnt

        if total_entries < int(sc.min_entries_total):
            return False, "min_entries_total", fold_entries, int(total_entries)
        if total_entries > int(sc.max_entries_total):
            return False, "max_entries_total", fold_entries, int(total_entries)
        return True, "ok", fold_entries, int(total_entries)
    
    def _run_backtest(self, compiled: CompiledStrategy, direction: str, start: int, end: int) -> BacktestOutput:
        # Engine-only: do NOT silently fall back to a different backtest model.
        engine_out = _try_backtest_engine(self.series, compiled, direction, start, end, cfg=self._engine_cfg)
        if engine_out is None:
            raise ValidationError(
                "Backtest_Engine adapter could not run (returned None). "
                "Refusing to proceed because mining must be comparable and deployable."
            )
        self._bt_backend_counts["engine"] += 1
        return engine_out
    
    def _run_backtest_cost_mult(
        self,
        compiled: CompiledStrategy,
        direction: str,
        start: int,
        end: int,
        *,
        cost_mult: float,
        stress: bool = True,
    ) -> BacktestOutput:
        """Run the same backtest model but with fees/spread/slippage multiplied.

        This is used for robustness checks ("if costs double, does the edge still exist?").
        """
        m = float(cost_mult)
        require(_finite(m) and m > 0.0, "cost_mult must be finite > 0")

        try:
            import Backtest_Engine as BE  # type: ignore
        except Exception as e:
            raise ValidationError("Backtest_Engine import failed during stress test") from e
        base_cfg = self._engine_cfg
        if base_cfg is None:
            base_cfg = BE.BacktestConfig(
                fee_rate_taker=float(self.score_cfg.fee_bps) / 10000.0,
                spread_bps=float(self.score_cfg.spread_bps),
                slippage_bps=float(self.score_cfg.slippage_bps),
                adverse_fill_slip_mult=float(getattr(self.score_cfg, "adverse_fill_slip_mult", 0.0) or 0.0),
                store_trades=True,
                store_equity_curve=True,
                close_on_end=True,
            )
            

        cfg2 = replace(
            base_cfg,
            fee_rate_taker=float(base_cfg.fee_rate_taker) * m,
            spread_bps=float(base_cfg.spread_bps) * m,
            slippage_bps=float(base_cfg.slippage_bps) * m,
        )

        out = _try_backtest_engine(self.series, compiled, direction, start, end, cfg=cfg2)
        if out is None:
            raise ValidationError("Engine stress backtest failed: adapter returned None")

        if stress:
            self._bt_backend_counts_stress["engine"] += 1
        else:
            self._bt_backend_counts["engine"] += 1
        return out
    def _run_backtest_adverse_fill_mult(
        self,
        compiled: CompiledStrategy,
        direction: str,
        start: int,
        end: int,
        *,
        adverse_mult: float,
        stress: bool = True,
    ) -> BacktestOutput:
        """Run the same backtest backend but override adverse_fill_slip_mult.

        This is a robustness check for OHLC-only intrabar ambiguity: stops often fill worse than the
        trigger level during fast moves. Strategies that only survive with perfect stop fills are
        not research-grade for real money.
        """
        m = float(adverse_mult)
        require(_finite(m) and 0.0 <= m <= 1.0, "adverse_mult must be in [0,1]")

        try:
            import Backtest_Engine as BE  # type: ignore
        except Exception as e:
            raise ValidationError("Backtest_Engine import failed during adverse-fill stress test") from e


        base_cfg = self._engine_cfg
        if base_cfg is None:
            base_cfg = BE.BacktestConfig(
                fee_rate_taker=float(self.score_cfg.fee_bps) / 10000.0,
                spread_bps=float(self.score_cfg.spread_bps),
                slippage_bps=float(self.score_cfg.slippage_bps),
                adverse_fill_slip_mult=float(getattr(self.score_cfg, "adverse_fill_slip_mult", 0.0) or 0.0),          
                store_trades=True,
                store_equity_curve=True,
                close_on_end=True,
            )

        cfg2 = replace(
            base_cfg,
            adverse_fill_slip_mult=float(m),
            store_trades=True,
            store_equity_curve=True,
            close_on_end=True,
        )

        out = _try_backtest_engine(self.series, compiled, direction, start, end, cfg=cfg2)
        if out is None:
            raise ValidationError("Engine adverse-fill stress backtest failed: adapter returned None")

        if stress:
            self._bt_backend_counts_stress["engine"] += 1
        else:
            self._bt_backend_counts["engine"] += 1
        return out

    def _holdout_permutation_test(
        self,
        compiled: CompiledStrategy,
        direction: str,
        start: int,
        end: int,
        *,
        spec_hash: str,
        base_return: float,
        trials: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Empirical (one-sided) p-value via circular time-shift of entry/exit signals within the
        HOLDOUT window.

        This is a negative control: if a strategy "wins" the holdout because its signals happened
        to align with one lucky burst, shifting those same signals in time should often produce
        similarly strong returns. If shifting typically destroys performance, that's evidence
        (not proof) that timing is non-random.
        """
        s = int(start)
        e = int(end)
        L = int(e - s)
        t_req = int(trials)

        summary: Dict[str, Any] = {
            "method": "circular_shift",
            "trials_requested": int(t_req),
            "trials_used": 0,
            "valid": 0,
            "errors": 0,
            "p_value": 1.0,
            "base_return": float(base_return),
            "seed": None,
        }

        if t_req <= 0 or L <= 1:
            return 1.0, summary

        # Number of unique non-zero shifts.
        max_unique = max(0, L - 1)
        t_use = min(int(t_req), int(max_unique))
        if t_use <= 0:
            return 1.0, summary

        # Deterministic seed per (strategy, window) so results are stable run-to-run.
        salt = f"{spec_hash}:{s}:{e}:{str(direction).lower().strip()}"
        h = hashlib.sha1(salt.encode("utf-8")).hexdigest()
        mix = int(h[:8], 16)
        seed0 = int(getattr(self.search_cfg, "seed", 0) or 0)
        seed = int(seed0) ^ int(mix)
        summary["seed"] = int(seed)
        rng = random.Random(int(seed))

        if t_use >= max_unique:
            shifts = list(range(1, L))
        else:
            # sample without replacement
            shifts = rng.sample(range(1, L), k=int(t_use))

        perm_rets: List[float] = []
        errors = 0

        pf_cap = float(self.score_cfg.profit_factor_cap)
        pf_eps = float(self.score_cfg.profit_factor_eps)

        for sh in shifts:
            try:
                entry_view = _ShiftedSignal(compiled.entry, s, e, int(sh))
                exit_view = _ShiftedSignal(compiled.exit, s, e, int(sh)) if compiled.exit is not None else None
                comp2 = replace(compiled, entry=entry_view, exit=exit_view)

                bt = self._run_backtest(comp2, direction, s, e)
                fm = _compute_fold_metrics(
                    -3,
                    s,
                    e,
                    bt,
                    pf_cap=pf_cap,
                    pf_eps=pf_eps,
                )
                r = float(fm.net_return)
                if _finite(r):
                    perm_rets.append(float(r))
            except Exception:
                errors += 1

        summary["trials_used"] = int(len(shifts))
        summary["errors"] = int(errors)

        n = int(len(perm_rets))
        summary["valid"] = int(n)
        if n <= 0:
            return 1.0, summary

        base = float(base_return)
        # One-sided test: depends on sign of observed performance.
        if base >= 0.0:
            ge = sum(1 for r in perm_rets if float(r) >= base)
            p = float(ge + 1) / float(n + 1)
        else:
            le = sum(1 for r in perm_rets if float(r) <= base)
            p = float(le + 1) / float(n + 1)

        xs = sorted(float(r) for r in perm_rets)
        mean_r = _mean(xs)
        if n % 2 == 1:
            med_r = xs[n // 2]
        else:
            med_r = 0.5 * (xs[n // 2 - 1] + xs[n // 2])

        p95_idx = int(math.floor(0.95 * float(n - 1)))
        p95_r = xs[p95_idx] if xs else 0.0

        summary.update(
            {
                "p_value": float(p),
                "mean_return": float(mean_r),
                "median_return": float(med_r),
                "p95_return": float(p95_r),
                "best_return": float(max(xs) if xs else 0.0),
                "worst_return": float(min(xs) if xs else 0.0),
            }
        )
        return float(p), summary
    def _holdout_param_robustness_test(
        self,
        spec: StrategySpec,
        *,
        direction: str,
        start: int,
        end: int,
        base_return: float,
        base_trades: int,
        base_min_trades: int,
        min_return: float,
        max_drawdown: float,
        min_profit_factor: float,
        trials: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Parameter-sensitivity robustness test on the holdout window.

        We perturb strategy parameters (thresholds/periods/stop/tp knobs) and require
        that performance doesn't collapse.

        Returns:
          (pass_ratio, summary)
        """
        s = int(start)
        e = int(end)
        L = int(e - s)
        t_req = int(trials)

        sc = self.score_cfg
        jitter = float(getattr(sc, "param_robust_jitter", 0.10) or 0.10)
        min_pass_ratio = float(getattr(sc, "param_robust_min_pass_ratio", 0.0) or 0.0)
        ret_ratio_min = float(getattr(sc, "param_robust_return_ratio_min", 0.30) or 0.30)
        trade_ratio_min = float(getattr(sc, "param_robust_trade_ratio_min", 0.50) or 0.50)
        max_liqs_fold = int(getattr(sc, "max_liquidations_per_fold", 0) or 0)

        summary: Dict[str, Any] = {
            "method": "param_jitter",
            "trials_requested": int(t_req),
            "trials_used": 0,
            "valid": 0,
            "errors": 0,
            "passes": 0,
            "pass_ratio": 0.0,
            "min_pass_ratio": float(min_pass_ratio),
            "jitter": float(jitter),
            "return_ratio_min": float(ret_ratio_min),
            "trade_ratio_min": float(trade_ratio_min),
            "base_return": float(base_return),
            "base_trades": int(base_trades),
            "seed": None,
            "required": {
                "min_return": float(min_return),
                "max_drawdown": float(max_drawdown),
                "min_profit_factor": float(min_profit_factor),
                "base_min_trades": int(base_min_trades),
                "max_liquidations_per_fold": int(max_liqs_fold),
            },
        }

        if t_req <= 0 or L <= 2 or not (_finite(jitter) and float(jitter) > 0.0):
            # disabled or too-small window
            summary["pass_ratio"] = 1.0
            return 1.0, summary

        # Deterministic seed per (strategy, window) so results are stable run-to-run.
        try:
            spec_h = str(spec.hash())
        except Exception:
            spec_h = ""
        salt = f"{spec_h}:{s}:{e}:{str(direction).lower().strip()}:j={float(jitter):.6f}:t={int(t_req)}"
        h = hashlib.sha1(salt.encode("utf-8")).hexdigest()
        mix = int(h[:8], 16)
        seed0 = int(getattr(self.search_cfg, "seed", 0) or 0)
        seed = int(seed0) ^ int(mix)
        summary["seed"] = int(seed)
        rng = random.Random(int(seed))

        # Per-perturbation gates
        base_ret = float(base_return)
        base_tr = int(base_trades)
        ho_min_tr = int(max(1, int(base_min_trades)))

        # Trade retention threshold
        req_by_ratio = int(math.ceil(float(base_tr) * float(trade_ratio_min))) if base_tr > 0 else 1
        req_trades = max(int(ho_min_tr), int(req_by_ratio), 1)

        # Return retention threshold
        req_return = float(min_return)
        if base_ret > 0.0 and float(ret_ratio_min) > 0.0:
            req_return = max(float(req_return), float(base_ret) * float(ret_ratio_min))

        summary["required"].update(
            {
                "required_trades": int(req_trades),
                "required_return": float(req_return),
            }
        )

        pf_cap = float(self.score_cfg.profit_factor_cap)
        pf_eps = float(self.score_cfg.profit_factor_eps)

        rets: List[float] = []
        dds: List[float] = []
        pfs: List[float] = []
        trs: List[int] = []
        liqs: List[int] = []

        passes = 0
        valid = 0
        errors = 0

        for _ in range(int(t_req)):
            try:
                spec2 = _perturb_strategy_spec(spec, rng, float(jitter), p=0.6)
                comp2 = self.compiler.compile(spec2)
                bt = self._run_backtest(comp2, direction, s, e)
                fm = _compute_fold_metrics(-4, s, e, bt, pf_cap=pf_cap, pf_eps=pf_eps)
                r = float(fm.net_return)
                dd = float(fm.max_drawdown)
                pf = float(fm.profit_factor)
                tr = int(fm.trades)
                liq = int(getattr(fm, "liquidations", 0) or 0)
                if not (_finite(r) and _finite(dd) and _finite(pf)):
                    raise ValidationError("non-finite metrics")
                valid += 1
                rets.append(float(r))
                dds.append(float(dd))
                pfs.append(float(pf))
                trs.append(int(tr))
                liqs.append(int(liq))

                ok = True
                if tr < int(req_trades):
                    ok = False
                if r < float(req_return):
                    ok = False
                if dd > float(max_drawdown):
                    ok = False
                if pf < float(min_profit_factor):
                    ok = False
                if int(liq) > int(max_liqs_fold):
                    ok = False
                if ok:
                    passes += 1
            except Exception:
                errors += 1

        summary["trials_used"] = int(t_req)
        summary["valid"] = int(valid)
        summary["errors"] = int(errors)
        summary["passes"] = int(passes)
        pass_ratio = float(passes) / float(max(1, int(t_req)))
        summary["pass_ratio"] = float(pass_ratio)

        if valid > 0:
            xs = sorted(float(r) for r in rets)
            n = len(xs)
            summary.update(
                {
                    "mean_return": float(_mean(xs)),
                    "median_return": float(xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])),
                    "worst_return": float(xs[0]),
                    "best_return": float(xs[-1]),
                    "mean_drawdown": float(_mean(dds)) if dds else 0.0,
                    "mean_profit_factor": float(_mean(pfs)) if pfs else 0.0,
                    "mean_trades": float(_mean([float(x) for x in trs])) if trs else 0.0,
                    "min_trades": int(min(trs)) if trs else 0,
                    "max_trades": int(max(trs)) if trs else 0,
                    "mean_liquidations": float(_mean([float(l) for l in liqs])),
                    "max_liquidations": int(max(liqs) if liqs else 0),
                }
            )
        else:
            summary.update(
                {
                    "mean_return": 0.0,
                    "median_return": 0.0,
                    "worst_return": 0.0,
                    "best_return": 0.0,
                    "mean_drawdown": 0.0,
                    "mean_profit_factor": 0.0,
                    "mean_trades": 0.0,
                    "min_trades": 0,
                    "max_trades": 0,
                    "mean_liquidations": 0.0,
                    "max_liquidations": 0,
                }
            )

        return float(pass_ratio), summary
    def _score(self, spec: StrategySpec, folds: Sequence[FoldMetrics]) -> Tuple[float, Dict[str, Any]]:
        sc = self.score_cfg

        rets = [float(f.net_return) for f in folds]
        dds = [float(f.max_drawdown) for f in folds]
        pfs = [float(f.profit_factor) for f in folds]
        # turnover is defined as "trades per 1000 bars" in FoldMetrics
        turns = [float(getattr(f, "turnover", 0.0) or 0.0) for f in folds]
        holds = [float(getattr(f, "avg_holding_bars", 0.0) or 0.0) for f in folds]
        trades_total = sum(int(getattr(f, "trades", 0) or 0) for f in folds)
        # Perp realism telemetry (engine backend)
        liqs = [int(getattr(f, "liquidations", 0) or 0) for f in folds]
        liq_paid = [float(getattr(f, "liquidation_paid", 0.0) or 0.0) for f in folds]
        liquidations_total = int(sum(liqs))
        liquidation_paid_total = float(sum(liq_paid))
        # Absolute (legacy) aggregates
        mean_ret_abs = _mean(rets)
        std_ret_abs = _std(rets)
        worst_fold_abs = min(rets) if rets else 0.0
        mean_dd = _mean(dds)
        mean_pf = _mean(pfs)
        mean_turn = _mean(turns)
        mean_hold = _mean(holds)

        # Baseline: close-to-close buy&hold return in the same direction (no costs, conservative).
        closes = [float(x) for x in self.series.close]
        n_cl = len(closes)
        baseline_rets: List[float] = []
        for f in folds:
            s = int(getattr(f, "start", 0) or 0)
            e = int(getattr(f, "end", 0) or 0)
            if s < 0 or e <= s or (e - 1) >= n_cl:
                baseline_rets.append(0.0)
                continue
            p0 = float(closes[s])
            p1 = float(closes[e - 1])
            if not (_finite(p0) and _finite(p1) and p0 > 0.0):
                baseline_rets.append(0.0)
                continue
            bh = (p1 / p0) - 1.0
            baseline_rets.append(float(bh if str(spec.direction) == "long" else -bh))

        excess_rets = [float(r) - float(b) for r, b in zip(rets, baseline_rets)]

        rm = str(getattr(sc, "return_mode", "absolute")).strip().lower()
        if rm == "excess":
            used_rets = excess_rets
        else:
            rm = "absolute"
            used_rets = rets

        mean_ret_used = _mean(used_rets)
        std_ret_used = _std(used_rets)
        worst_fold_used = min(used_rets) if used_rets else 0.0

        # Return dampening based on sample size (trades_total)
        rref = float(max(1, int(getattr(sc, "return_trade_damp_ref", 30) or 30)))
        rdamp = (float(trades_total) / rref) if rref > 0.0 else 1.0
        if not _finite(rdamp):
            rdamp = 0.0
        rdamp = max(0.0, min(1.0, float(rdamp)))
        mean_ret_score = float(mean_ret_used) * float(rdamp)
        worst_fold_score = float(worst_fold_used) * float(rdamp)

        # Fold consistency + heuristic significance
        k = int(len(used_rets))
        pos_ratio = (sum(1 for r in used_rets if float(r) > 0.0) / float(k)) if k > 0 else 0.0
        if k > 1 and float(std_ret_used) > 0.0:
            t_stat = float(mean_ret_used) / (float(std_ret_used) / math.sqrt(float(k)))
        elif k > 1 and float(std_ret_used) == 0.0:
            if float(mean_ret_used) > 0.0:
                t_stat = float("inf")
            elif float(mean_ret_used) < 0.0:
                t_stat = float("-inf")
            else:
                t_stat = 0.0
        else:
            t_stat = 0.0
        p_value = float(_p_value_two_sided_from_t(t_stat, df=(k - 1))) if k > 1 else 1.0
 

        # PF term: damp it HARD for tiny sample sizes.
        # With 1-2 trades total, PF can easily hit cap and dominate the score (false positive).
        # Use linear damp instead of log damp:
        #   trades_total=2, ref=30 -> damp ~0.067 (instead of ~0.32 with log)
        # PF term baseline at 1.0: pf=1 -> 0 contribution; pf>1 positive; pf<1 negative
        pf_raw = math.log(max(float(mean_pf), 1e-12)) 
        ref = float(max(1, int(getattr(sc, "pf_trade_damp_ref", 30) or 30)))
        damp = (float(trades_total) / ref) if ref > 0.0 else 1.0
        if not _finite(damp):
            damp = 0.0
        damp = max(0.0, min(1.0, float(damp)))
        pf_mode = str(getattr(sc, "pf_transform", "log")).strip().lower()
        if pf_mode == "none":
            pf_term = 0.0
        elif pf_mode == "tanh":
            # bounded in [-1, 1], prevents PF from dominating the score
            pf_term = float(math.tanh(float(pf_raw))) * float(damp)
        else:
            # "log" (legacy): unbounded (can dominate when PF is huge)
            pf_term = float(pf_raw) * float(damp)
        complexity = float(spec.complexity())

        score = 0.0
        score += float(sc.weight_return) * float(mean_ret_score)
        score -= float(sc.weight_dd) * mean_dd
        score -= float(sc.weight_stability) * float(std_ret_used)
        score += float(sc.weight_pf) * pf_term
        score -= float(sc.weight_complexity) * complexity
        score -= float(sc.weight_turnover) * (mean_turn / 1000.0)
        score += float(sc.weight_worst_fold) * float(worst_fold_score)
        # JSON can't represent inf reliably; keep a big finite sentinel for audit/printing.
        if not _finite(float(t_stat)):
            t_stat_json = 999.0 if float(t_stat) > 0.0 else -999.0
        else:
            t_stat_json = float(t_stat)
        agg = {
            "return_mode": rm,

            # absolute metrics (backward compatible / audit)
            "mean_return": mean_ret_abs,
            "std_return": std_ret_abs,
            "worst_fold_return": worst_fold_abs,

            # baseline + excess (always computed, cheap)
            "mean_baseline_return": float(_mean(baseline_rets)),
            "fold_baseline_returns": baseline_rets,
            "mean_excess_return": float(_mean(excess_rets)),
            "fold_excess_returns": excess_rets,

            # used metrics (depends on return_mode)
            "mean_return_used": float(mean_ret_used),
            "std_return_used": float(std_ret_used),
            "worst_fold_return_used": float(worst_fold_used),
            "fold_returns_used": used_rets,

            # return damp (sample-size shrinkage)
            "return_trade_damp_ref": int(rref),
            "return_trade_damp": float(rdamp),
            "mean_return_score": float(mean_ret_score),
            "worst_fold_return_score": float(worst_fold_score),

            # consistency / significance (heuristic)
            "pos_fold_ratio_used": float(pos_ratio),
            "t_stat_used": float(t_stat_json),
            "p_value_df_used": int(k - 1) if k > 1 else 0,
            "mean_drawdown": mean_dd,
            "mean_profit_factor": mean_pf,
            "pf_transform": str(getattr(sc, "pf_transform", "log")),
            "pf_log": float(pf_raw),
            "trades_total": int(trades_total),
            "liquidations_total": int(liquidations_total),
            "liquidation_paid_total": float(liquidation_paid_total),
            "pf_trade_damp": float(damp),
            "mean_turnover_per_1000": mean_turn,
            "mean_hold_bars": float(mean_hold),
            "complexity": int(spec.complexity()),
            "fold_returns": rets,
            "fold_drawdowns": dds,
            "fold_trades": [int(f.trades) for f in folds],
            "fold_hold_bars": [float(getattr(f, "avg_holding_bars", 0.0) or 0.0) for f in folds],
            "fold_top_profit_share": [float(getattr(f, "top_profit_share", 0.0) or 0.0) for f in folds],
            "fold_profit_hhi": [float(getattr(f, "profit_hhi", 0.0) or 0.0) for f in folds],
        }
        return float(score), agg

    # ---- Main run ----
    def run(self) -> MinerReport:
        cfg = self.search_cfg
        sc = self.score_cfg
        # Perpetual futures safety: liquidation is a hard failure in research-grade runs.
        # These limits have teeth because the engine backend models liquidation.
        max_liqs_total = int(getattr(sc, "max_liquidations_total", 0) or 0)
        max_liqs_fold = int(getattr(sc, "max_liquidations_per_fold", 0) or 0)
        evaluated = 0
        accepted = 0
        rejected = 0

        reject_reasons: Dict[str, int] = {}

        def _rej(reason: str) -> None:
            r = str(reason)
            reject_reasons[r] = int(reject_reasons.get(r, 0)) + 1

        heap: List[Tuple[float, str, CandidateResult]] = []
        # Debug heap: keeps best candidates even if rejected by strict filters (e.g. stress test),
        # so you can see "what almost worked" when Accepted=0.
        debug_heap: List[Tuple[float, str, DebugCandidate]] = []
        debug_k = max(int(cfg.top_k), 20)
        hs = he = None
        if self.holdout_slice is not None:
            hs = int(self.holdout_slice.start)  # type: ignore[arg-type]
            he = int(self.holdout_slice.stop)   # type: ignore[arg-type]
            require(hs is not None and he is not None and 0 <= hs < he <= len(self.series.ts_ms), "invalid holdout slice")
        def _debug_push(
            *,
            score: float,
            spec: StrategySpec,
            agg: Dict[str, Any],
            status: str,
            reject_reason: str,
            compiled: Optional[CompiledStrategy] = None,
            holdout_aggregate: Optional[Dict[str, Any]] = None,
        ) -> None:
            if not _finite(float(score)):
                return
            # Work on a copy so we don't mutate shared agg dicts.
            agg2: Dict[str, Any] = dict(agg) if isinstance(agg, dict) else {}

            ho = holdout_aggregate
            # Optional: compute holdout for debug candidates too, but only when they are debug-competitive.
            if ho is None and compiled is not None and hs is not None and he is not None:
                want = (len(debug_heap) < int(debug_k)) or (float(score) > float(debug_heap[0][0]))
                if want:
                    try:
                        bt_ho = self._run_backtest(compiled, spec.direction, int(hs), int(he))
                        fm_ho = _compute_fold_metrics(-1, int(hs), int(he), bt_ho)
                        ho = {
                            "net_return": float(fm_ho.net_return),
                            "max_drawdown": float(fm_ho.max_drawdown),
                            "profit_factor": float(fm_ho.profit_factor),
                            "trades": int(fm_ho.trades),
                            "win_rate": float(fm_ho.win_rate),
                            "avg_trade_return": float(fm_ho.avg_trade_return),
                            "exposure": float(fm_ho.exposure),
                            "turnover_per_1000": float(fm_ho.turnover),
                            "avg_hold_bars": float(getattr(fm_ho, "avg_holding_bars", 0.0) or 0.0),
                            "liquidations": int(getattr(fm_ho, "liquidations", 0) or 0),
                            "liquidation_paid": float(getattr(fm_ho, "liquidation_paid", 0.0) or 0.0),
                            "start": int(hs),
                            "end": int(he),
                        }
                        # Time-bucket profit concentration (temporal clustering)
                        try:
                            bms = int(getattr(sc, "holdout_profit_bucket_ms", 28_800_000) or 28_800_000)
                            tb = _time_bucket_profit_concentration(
                                self.series.ts_ms,
                                bt_ho.trades,
                                int(hs),
                                int(he),
                                bucket_ms=int(bms),
                            )
                            if isinstance(ho, dict) and isinstance(tb, dict) and bool(tb.get("enabled", False)):
                                ho["time_bucket"] = tb
                        except Exception:
                            pass
                        # Volatility-regime profit concentration (market-condition dependence)
                        try:
                            vb = _regime_bucket_profit_concentration(
                                self.compiler.reg,
                                bt_ho.trades,
                                int(hs),
                                int(he),
                                regime_name="vol_regime",
                            )
                            if isinstance(ho, dict) and isinstance(vb, dict) and bool(vb.get("enabled", False)):
                                ho["vol_regime_bucket"] = vb
                        except Exception:
                            pass
                        # Segment consistency (temporal stability)
                        try:
                            seg_n = int(getattr(sc, "holdout_segment_count", 0) or 0)
                            if int(seg_n) > 1:
                                seg = _holdout_segment_consistency(
                                    bt_ho.equity_curve,
                                    int(hs),
                                    int(he),
                                    segments=int(seg_n),
                                )
                                if isinstance(ho, dict) and isinstance(seg, dict) and bool(seg.get("enabled", False)):
                                    ho["segment_consistency"] = seg
                        except Exception:
                            pass
                    except Exception:
                        ho = None
            # Optional: compute stress metrics for debug candidates too (even if they were rejected
            # before the main stress-test stage, e.g. min_trades_total).
            stress_mult = float(getattr(sc, "stress_cost_mult", 1.0) or 1.0)
            if (
                compiled is not None
                and _finite(stress_mult)
                and stress_mult > 1.0
                and "stress" not in agg2
            ):
                want = (len(debug_heap) < int(debug_k)) or (float(score) > float(debug_heap[0][0]))
                if want:
                    try:
                        stress_folds: List[FoldMetrics] = []
                        stress_total_trades = 0
                        for fi, split in enumerate(self.splits):
                            s2 = int(split.test.start)  # type: ignore[arg-type]
                            e2 = int(split.test.stop)   # type: ignore[arg-type]
                            bt_s = self._run_backtest_cost_mult(
                                compiled, spec.direction, s2, e2, cost_mult=float(stress_mult), stress=True
                            )
                            fm_s = _compute_fold_metrics(fi, s2, e2, bt_s)
                            stress_folds.append(fm_s)
                            stress_total_trades += int(fm_s.trades)

                        s_rets = [float(f.net_return) for f in stress_folds]
                        s_dds = [float(f.max_drawdown) for f in stress_folds]
                        agg2["stress"] = {
                            "cost_mult": float(stress_mult),
                            "mean_return": float(_mean(s_rets)),
                            "worst_fold_return": float(min(s_rets) if s_rets else 0.0),
                            "mean_drawdown": float(_mean(s_dds)),
                            "trades_total": int(stress_total_trades),
                            "fold_returns": s_rets,
                            "fold_drawdowns": s_dds,
                        }
                    except Exception:
                        # Don't let debug telemetry break the run.
                        pass
            dc = DebugCandidate(
                strategy_id=spec.id_str(),
                strategy_hash=spec.hash(),
                direction=spec.direction,
                tags=spec.tags,
                complexity=spec.complexity(),
                score=float(score),
                status=str(status),
                reject_reason=str(reject_reason),
                aggregate=_to_jsonable(dict(agg2)),
                holdout_aggregate=_to_jsonable(ho) if ho is not None else None,
            )

            if len(debug_heap) < int(debug_k):
                heapq.heappush(debug_heap, (float(dc.score), dc.strategy_hash, dc))
            else:
                if float(dc.score) > float(debug_heap[0][0]):
                    heapq.heapreplace(debug_heap, (float(dc.score), dc.strategy_hash, dc))
        # ------------------------------------------------------------
        # Loop : Safe early pruning inside the fold loop
        # ------------------------------------------------------------
        # Motivation:
        # - Many candidates are doomed early (e.g., one catastrophic fold return) but we still
        #   waste CPU running the remaining folds.
        # - Early pruning increases throughput -> you can evaluate more candidates per hour,
        #   which materially increases the probability of discovering profitable strategies.
        #
        # Safety:
        # - The checks below only prune when a candidate cannot possibly pass the final gates
        #   (min_worst_fold_return and min_pos_fold_ratio). This avoids accidentally pruning
        #   candidates that could still pass.
        rm = str(getattr(sc, "return_mode", "absolute")).strip().lower()
        rm = "excess" if rm == "excess" else "absolute"
        min_worst_fold = float(getattr(sc, "min_worst_fold_return", -1.0) or -1.0)
        min_pos_ratio = float(getattr(sc, "min_pos_fold_ratio", 0.0) or 0.0)
        folds_total = int(len(self.splits))
        pos_needed = int(math.ceil(float(min_pos_ratio) * float(folds_total) - 1e-12)) if float(min_pos_ratio) > 0.0 else 0
        closes = [float(x) for x in self.series.close]
        n_cl = int(len(closes))

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
            ok_entry, reason, fold_entries, entries_total = self._passes_entry_filters(compiled.entry)
            if not ok_entry:
                rejected += 1
                _rej(f"entry_filter:{reason}")
                continue

            # ---- evaluate folds
            fold_metrics: List[FoldMetrics] = []
            total_trades = 0
            total_liquidations = 0
            ok = True
            fail_reason = ""
            pos_folds = 0
            for fi, split in enumerate(self.splits):
                start = int(split.test.start)  # type: ignore[arg-type]
                end = int(split.test.stop)     # type: ignore[arg-type]

                try:
                    bt_out = self._run_backtest(compiled, spec.direction, start, end)
                except Exception as e:
                    ok = False
                    fail_reason = f"backtest_fail:{type(e).__name__}"
                    break


                fm = _compute_fold_metrics(
                    fi,
                    start,
                    end,
                    bt_out,
                    pf_cap=float(sc.profit_factor_cap),
                    pf_eps=float(sc.profit_factor_eps),
                )
                # ---- safe early pruning based on "used returns" (absolute vs excess)
                # Compute fold return consistent with _score(return_mode).
                used_fold_ret = float(fm.net_return)
                if rm == "excess":
                    baseline = 0.0
                    if 0 <= int(start) < int(end) <= int(n_cl) and (int(end) - 1) < int(n_cl):
                        p0 = float(closes[int(start)])
                        p1 = float(closes[int(end) - 1])
                        if _finite(p0) and _finite(p1) and p0 > 0.0:
                            bh = (p1 / p0) - 1.0
                            baseline = float(bh if str(spec.direction) == "long" else -bh)
                    used_fold_ret = float(used_fold_ret) - float(baseline)

                # If ANY fold violates min_worst_fold_return, the candidate can never pass.
                if float(used_fold_ret) < float(min_worst_fold):
                    ok = False
                    fail_reason = "min_worst_fold_return"
                    break

                # If min_pos_fold_ratio is enabled, prune when it becomes impossible to reach.
                if float(used_fold_ret) > 0.0:
                    pos_folds += 1
                if int(pos_needed) > 0:
                    rem = int(folds_total - (fi + 1))
                    if int(pos_folds + rem) < int(pos_needed):
                        ok = False
                        fail_reason = "min_pos_fold_ratio"
                        break
                fold_metrics.append(fm)
                total_trades += int(fm.trades)
                liqs_here = int(getattr(fm, "liquidations", 0) or 0)
                total_liquidations += int(liqs_here)
                if int(liqs_here) > int(max_liqs_fold):
                    ok = False
                    fail_reason = "max_liquidations_per_fold"
                    break
                if int(total_liquidations) > int(max_liqs_total):
                    ok = False
                    fail_reason = "max_liquidations_total"
                    break
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
                # Keep a few of these for diagnostics: low-trade candidates often look
                # "amazing" by chance but are not research-grade.
                try:
                    score_tmp, agg_tmp = self._score(spec, fold_metrics)
                except Exception:
                    score_tmp, agg_tmp = float("nan"), {}
                if _finite(float(score_tmp)):
                    # IMPORTANT: attach entry telemetry here too, otherwise debug prints show en_tot=0.
                    try:
                        agg_tmp["entries_total"] = int(entries_total)
                        agg_tmp["fold_entries"] = list(int(x) for x in (fold_entries or []))
                    except Exception:
                        pass
                    _debug_push(
                        score=float(score_tmp),
                        spec=spec,
                        agg=agg_tmp,
                        status="rejected",
                        reject_reason="min_trades_total",
                        compiled=compiled,
                    )
                rejected += 1
                _rej("min_trades_total")
                continue

            # ---- profitability / realism filters (post-backtest)
            score, agg = self._score(spec, fold_metrics)
            # Attach entry telemetry (helps diagnose undertrading vs no-signal)
            agg["entries_total"] = int(entries_total)
            agg["fold_entries"] = list(int(x) for x in (fold_entries or []))
            # Multiple-testing adjusted p-value (Bonferroni; heuristic).
            # Helps you see when a "top" result is likely a best-of-N fluke.
            try:
                p_used = float(agg.get("p_value_used", 1.0) or 1.0)
            except Exception:
                p_used = 1.0
            trials = int(getattr(sc, "multiple_testing_trials", 0) or 0)
            if trials <= 0:
                trials = int(cfg.max_evals)
            trials = max(1, int(trials))
            p_adj = min(1.0, float(p_used) * float(trials))
            agg["multiple_testing_trials"] = int(trials)
            agg["p_value_adj"] = float(p_adj)
            if p_adj > 0.0 and _finite(float(p_adj)):
                try:
                    agg["p_value_adj_log10"] = float(math.log10(float(p_adj)))
                except Exception:
                    pass
            if not _finite(score):
                rejected += 1
                _rej("nonfinite_score")
                continue
            mean_used = float(agg.get("mean_return_used", agg.get("mean_return", 0.0)) or 0.0)
            if float(mean_used) < float(sc.min_mean_return):
                _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="min_mean_return", compiled=compiled)
                rejected += 1
                _rej("min_mean_return")
                continue

            worst_used = float(agg.get("worst_fold_return_used", agg.get("worst_fold_return", 0.0)) or 0.0)
            if float(worst_used) < float(sc.min_worst_fold_return):
                _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="min_worst_fold_return", compiled=compiled)
                rejected += 1
                _rej("min_worst_fold_return")
                continue
            # ---- fold consistency / significance guards (optional, but recommended)
            min_pos = float(getattr(sc, "min_pos_fold_ratio", 0.0) or 0.0)
            if min_pos > 0.0:
                pos = float(agg.get("pos_fold_ratio_used", 0.0) or 0.0)
                if pos < float(min_pos):
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="min_pos_fold_ratio", compiled=compiled)
                    rejected += 1
                    _rej("min_pos_fold_ratio")
                    continue

            min_t = float(getattr(sc, "min_t_stat", 0.0) or 0.0)
            if min_t > 0.0:
                tsv = float(agg.get("t_stat_used", 0.0) or 0.0)
                if tsv < float(min_t):
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="min_t_stat", compiled=compiled)
                    rejected += 1
                    _rej("min_t_stat")
                    continue

            max_p = float(getattr(sc, "max_p_value_adj", 0.0) or 0.0)
            if max_p > 0.0:
                pv = float(agg.get("p_value_adj", 1.0) or 1.0)
                if pv > float(max_p):
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="max_p_value_adj", compiled=compiled)
                    rejected += 1
                    _rej("max_p_value_adj")
                    continue

            if float(agg.get("mean_profit_factor", 0.0)) < float(sc.min_profit_factor):
                _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="min_profit_factor", compiled=compiled)
                rejected += 1
                _rej("min_profit_factor")
                continue
            # ---- Profit concentration gate (OOS folds)
            # Reject strategies whose OOS gross profits are dominated by a single trade in ANY fold.
            # This is a common false-positive pattern in 1m/5m mining (one lucky trend candle / news bar).
            max_top_ps = float(getattr(sc, "max_top_profit_share", 1.0) or 1.0)
            if _finite(max_top_ps) and float(max_top_ps) < 1.0:
                try:
                    tops = [float(getattr(fm, "top_profit_share", 0.0) or 0.0) for fm in fold_metrics]
                    max_top = max(tops) if tops else 0.0
                except Exception:
                    max_top = 0.0
                agg["max_fold_top_profit_share"] = float(max_top)
                agg["max_top_profit_share_limit"] = float(max_top_ps)
                if float(max_top) > float(max_top_ps):
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="oos_top_profit_share", compiled=compiled)
                    rejected += 1
                    _rej("oos_top_profit_share")
                    continue
            if float(agg.get("mean_turnover_per_1000", 0.0)) > float(sc.max_turnover_per_1000):
                _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="max_turnover_per_1000", compiled=compiled)
                rejected += 1
                _rej("max_turnover_per_1000")
                continue
            # ---- Cost stress test (optional, but strongly recommended for real-money research)
            # Re-run the SAME folds with costs multiplied. This often kills fragile 1m strategies.
            stress_mult = float(getattr(sc, "stress_cost_mult", 1.0) or 1.0)
            if _finite(stress_mult) and stress_mult > 1.0:
                stress_folds: List[FoldMetrics] = []
                stress_total_trades = 0
                stress_total_liquidations = 0
                ok_stress = True
                stress_fail_reason = ""

                for fi, split in enumerate(self.splits):
                    s2 = int(split.test.start)  # type: ignore[arg-type]
                    e2 = int(split.test.stop)   # type: ignore[arg-type]
                    try:
                        bt_s = self._run_backtest_cost_mult(
                            compiled, spec.direction, s2, e2, cost_mult=stress_mult, stress=True
                        )
                    except Exception as e:
                        ok_stress = False
                        stress_fail_reason = f"stress_backtest_fail:{type(e).__name__}"
                        break

                    fm_s = _compute_fold_metrics(fi, s2, e2, bt_s)
                    stress_folds.append(fm_s)
                    stress_total_trades += int(fm_s.trades)
                    liqs_s = int(getattr(fm_s, "liquidations", 0) or 0)
                    stress_total_liquidations += int(liqs_s)
                    if int(liqs_s) > int(max_liqs_fold):
                        ok_stress = False
                        stress_fail_reason = "stress_max_liquidations_per_fold"
                        break
                    if int(stress_total_liquidations) > int(max_liqs_total):
                        ok_stress = False
                        stress_fail_reason = "stress_max_liquidations_total"
                        break
                    # keep the same DD guardrail under stress
                    if float(fm_s.max_drawdown) > float(sc.max_drawdown_limit):
                        ok_stress = False
                        stress_fail_reason = "stress_max_drawdown_fail"
                        break

                if not ok_stress:
                    rejected += 1
                    _rej(stress_fail_reason or "stress_fail")
                    continue

                s_rets = [float(f.net_return) for f in stress_folds]
                s_dds = [float(f.max_drawdown) for f in stress_folds]
                s_turn = [float(f.turnover) for f in stress_folds]

                s_mean_ret = _mean(s_rets)
                s_worst = min(s_rets) if s_rets else 0.0
                s_mean_dd = _mean(s_dds)
                s_mean_turn = _mean(s_turn)
                base_mean_ret = float(agg.get("mean_return", 0.0) or 0.0)
                base_worst = float(agg.get("worst_fold_return", 0.0) or 0.0)
                # Store for audit/printing
                agg["stress"] = {
                    "cost_mult": float(stress_mult),
                    "mean_return": float(s_mean_ret),
                    "worst_fold_return": float(s_worst),
                    "mean_drawdown": float(s_mean_dd),
                    "mean_turnover_per_1000": float(s_mean_turn),
                    "trades_total": int(stress_total_trades),
                    "liquidations_total": int(stress_total_liquidations),
                    "fold_returns": s_rets,
                    "fold_drawdowns": s_dds,
                }
                # Ratios (only meaningful when base > 0)
                if base_mean_ret > 0.0:
                    agg["stress"]["mean_return_ratio"] = float(s_mean_ret / base_mean_ret)
                if base_worst > 0.0:
                    agg["stress"]["worst_fold_return_ratio"] = float(s_worst / base_worst)
 
                if float(s_mean_ret) < float(getattr(sc, "stress_min_mean_return", 0.0)):
                    rejected += 1
                    _rej("stress_min_mean_return")
                    continue
                if float(s_worst) < float(getattr(sc, "stress_min_worst_fold_return", -0.10)):
                    rejected += 1
                    _rej("stress_min_worst_fold_return")
                    continue
                # Retention gates (optional but recommended)
                rr_min = float(getattr(sc, "stress_mean_return_ratio_min", 0.0) or 0.0)
                if rr_min > 0.0 and base_mean_ret > 0.0 and float(s_mean_ret) < float(base_mean_ret) * float(rr_min):
                    rejected += 1
                    _rej("stress_mean_return_ratio")
                    continue
                wr_min = float(getattr(sc, "stress_worst_fold_return_ratio_min", 0.0) or 0.0)
                if wr_min > 0.0 and base_worst > 0.0 and float(s_worst) < float(base_worst) * float(wr_min):
                    rejected += 1
                    _rej("stress_worst_fold_return_ratio")
            # ---- Holdout evaluation (untouched final segment)
            holdout_metrics: Optional[FoldMetrics] = None
            holdout_agg: Optional[Dict[str, Any]] = None

            have_holdout = (hs is not None and he is not None)
            holdout_required = bool(getattr(sc, "holdout_required", False))

            # Compute holdout either:
            # - always, when it is required as a gating filter, or
            # - only when heap-competitive (keeps it fast when holdout is informational only).
            heap_competitive = (len(heap) < int(cfg.top_k)) or (float(score) > float(heap[0][0]))
            want_holdout = have_holdout and (holdout_required or heap_competitive)
            if want_holdout:
                try:
                    bt_ho = self._run_backtest(compiled, spec.direction, int(hs), int(he))
                    fm_ho = _compute_fold_metrics(
                        -1,
                        int(hs),
                        int(he),
                        bt_ho,
                        pf_cap=float(sc.profit_factor_cap),
                        pf_eps=float(sc.profit_factor_eps),
                    )
                    holdout_metrics = fm_ho
                    holdout_agg = {
                        "net_return": float(fm_ho.net_return),
                        "max_drawdown": float(fm_ho.max_drawdown),
                        "profit_factor": float(fm_ho.profit_factor),
                        "top_profit_share": float(getattr(fm_ho, "top_profit_share", 0.0) or 0.0),
                        "profit_hhi": float(getattr(fm_ho, "profit_hhi", 0.0) or 0.0),
                        "trades": int(fm_ho.trades),
                        "win_rate": float(fm_ho.win_rate),
                        "avg_trade_return": float(fm_ho.avg_trade_return),
                        "exposure": float(fm_ho.exposure),
                        "turnover_per_1000": float(fm_ho.turnover),
                        "avg_hold_bars": float(getattr(fm_ho, "avg_holding_bars", 0.0) or 0.0),
                        "liquidations": int(getattr(fm_ho, "liquidations", 0) or 0),
                        "liquidation_paid": float(getattr(fm_ho, "liquidation_paid", 0.0) or 0.0),
                        "start": int(hs),
                        "end": int(he),
                    }
                    # Time-bucket profit concentration (temporal clustering)
                    try:
                        bms = int(getattr(sc, "holdout_profit_bucket_ms", 28_800_000) or 28_800_000)
                        tb = _time_bucket_profit_concentration(
                            self.series.ts_ms,
                            bt_ho.trades,
                            int(hs),
                            int(he),
                            bucket_ms=int(bms),
                        )
                        if isinstance(tb, dict) and bool(tb.get("enabled", False)):
                            holdout_agg["time_bucket"] = tb
                    except Exception:
                        pass
                    # Volatility-regime profit concentration (market-condition dependence)
                    try:
                        vb = _regime_bucket_profit_concentration(
                            self.compiler.reg,
                            bt_ho.trades,
                            int(hs),
                            int(he),
                            regime_name="vol_regime",
                        )
                        if isinstance(vb, dict) and bool(vb.get("enabled", False)):
                            holdout_agg["vol_regime_bucket"] = vb
                    except Exception:
                        pass
                    # Segment consistency (temporal stability)
                    try:
                        seg_n = int(getattr(sc, "holdout_segment_count", 0) or 0)
                        if int(seg_n) > 1:
                            seg = _holdout_segment_consistency(
                                bt_ho.equity_curve,
                                int(hs),
                                int(he),
                                segments=int(seg_n),
                            )
                            if isinstance(seg, dict) and bool(seg.get("enabled", False)):
                                holdout_agg["segment_consistency"] = seg
                    except Exception:
                        pass
                except Exception as e:
                    rejected += 1
                    _rej(f"holdout_fail:{type(e).__name__}")
                    continue
            # ---- Holdout gating (critical anti-overfit safety)
            if have_holdout and holdout_required:
                if holdout_agg is None:
                    rejected += 1
                    _rej("holdout_missing")
                    continue

                ho_trades = int(holdout_agg.get("trades", 0) or 0)
                ho_ret = float(holdout_agg.get("net_return", 0.0) or 0.0)
                ho_dd = float(holdout_agg.get("max_drawdown", 0.0) or 0.0)
                ho_pf = float(holdout_agg.get("profit_factor", 0.0) or 0.0)
                ho_liqs = int(holdout_agg.get("liquidations", 0) or 0)
                # Dynamic sample-size requirement on holdout:
                # - absolute floor (holdout_min_trades)
                # - plus a "share of expected" requirement based on holdout bars vs total test bars
                ho_min_tr_abs = int(getattr(sc, "holdout_min_trades", 0) or 0)
                trade_ratio_min = float(getattr(sc, "holdout_trade_ratio_min", 0.0) or 0.0)
                tr_total = int(agg.get("trades_total", 0) or 0)
                holdout_bars = int(he) - int(hs)
                total_test_bars = int(getattr(self, "_total_test_bars", 0) or 0)
                exp_ratio = float(holdout_bars) / float(max(1, total_test_bars))
                req_by_ratio = int(math.ceil(float(tr_total) * exp_ratio * float(trade_ratio_min)))
                ho_min_tr = max(int(ho_min_tr_abs), int(req_by_ratio))
                ho_min_ret = float(getattr(sc, "holdout_min_return", 0.0) or 0.0)
                ho_max_dd = float(getattr(sc, "holdout_max_drawdown_limit", 1.0) or 1.0)
                ho_min_pf = float(getattr(sc, "holdout_min_profit_factor", 0.0) or 0.0)
                # annotate for debugging / JSON inspection
                if isinstance(holdout_agg, dict):
                    holdout_agg["required_min_trades"] = int(ho_min_tr)
                    holdout_agg["required_min_trades_abs"] = int(ho_min_tr_abs)
                    holdout_agg["required_min_trades_by_ratio"] = int(req_by_ratio)
                    holdout_agg["holdout_trade_ratio_min"] = float(trade_ratio_min)
                    holdout_agg["holdout_bars"] = int(holdout_bars)
                    holdout_agg["total_test_bars"] = int(total_test_bars)
                if ho_trades < ho_min_tr:
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_min_trades", compiled=compiled, holdout_aggregate=holdout_agg)
                    rejected += 1
                    _rej("holdout_min_trades")
                    continue
                if int(ho_liqs) > int(max_liqs_fold):
                    _debug_push(
                        score=float(score),
                        spec=spec,
                        agg=agg,
                        status="rejected",
                        reject_reason="holdout_max_liquidations_per_fold",
                        compiled=compiled,
                        holdout_aggregate=holdout_agg,
                    )
                    rejected += 1
                    _rej("holdout_max_liquidations_per_fold")
                    continue
                if ho_ret < ho_min_ret:
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_min_return", compiled=compiled, holdout_aggregate=holdout_agg)
                    rejected += 1
                    _rej("holdout_min_return")
                    continue
                if ho_dd > ho_max_dd:
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_max_drawdown_fail", compiled=compiled, holdout_aggregate=holdout_agg)
                    rejected += 1
                    _rej("holdout_max_drawdown_fail")
                    continue
                if ho_pf < ho_min_pf:
                    _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_min_profit_factor", compiled=compiled, holdout_aggregate=holdout_agg)
                    rejected += 1
                    _rej("holdout_min_profit_factor")
                    continue
                # ---- Holdout profit concentration gate (avoid "one lucky trade" strategies)
                ho_max_top_ps = float(getattr(sc, "holdout_max_top_profit_share", 1.0) or 1.0)
                if isinstance(holdout_agg, dict):
                    holdout_agg["holdout_max_top_profit_share"] = float(ho_max_top_ps)
                if float(ho_max_top_ps) < 1.0:
                    ho_top_ps = float(holdout_agg.get("top_profit_share", 0.0) or 0.0)
                    if ho_top_ps > float(ho_max_top_ps):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_top_profit_share", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_top_profit_share")
                        continue
                # ---- Holdout time-bucket profit concentration gate (temporal dependence)
                max_bucket_share = float(getattr(sc, "holdout_max_top_bucket_profit_share", 1.0) or 1.0)
                if _finite(max_bucket_share) and 0.0 < float(max_bucket_share) < 1.0:
                    tb = holdout_agg.get("time_bucket", None) if isinstance(holdout_agg, dict) else None
                    if isinstance(tb, dict) and bool(tb.get("enabled", False)):
                        span = int(tb.get("bucket_span", 0) or 0)
                        top_s = float(tb.get("top_profit_share", 0.0) or 0.0)
                        if isinstance(holdout_agg, dict):
                            holdout_agg["max_top_bucket_profit_share_allowed"] = float(max_bucket_share)
                        # If the holdout doesn't span at least 2 buckets, this gate is meaningless -> skip.
                        if int(span) >= 2 and _finite(top_s) and float(top_s) > float(max_bucket_share):
                            _debug_push(
                                score=float(score),
                                spec=spec,
                                agg=agg,
                                status="rejected",
                                reject_reason="holdout_top_bucket_profit_share",
                                compiled=compiled,
                                holdout_aggregate=holdout_agg,
                            )
                            rejected += 1
                            _rej("holdout_top_bucket_profit_share")
                            continue
                # ---- Holdout volatility-regime profit concentration gate (market-condition dependence)
                max_vol_share = float(getattr(sc, "holdout_max_top_vol_profit_share", 1.0) or 1.0)
                if isinstance(holdout_agg, dict):
                    holdout_agg["holdout_max_top_vol_profit_share"] = float(max_vol_share)
                if _finite(max_vol_share) and 0.0 < float(max_vol_share) < 1.0:
                    vb = holdout_agg.get("vol_regime_bucket", None) if isinstance(holdout_agg, dict) else None
                    if isinstance(vb, dict) and bool(vb.get("enabled", False)):
                        top_s = float(vb.get("top_profit_share", 0.0) or 0.0)
                        if isinstance(holdout_agg, dict):
                            holdout_agg["max_top_vol_profit_share_allowed"] = float(max_vol_share)
                        if _finite(top_s) and float(top_s) > float(max_vol_share):
                            _debug_push(
                                score=float(score),
                                spec=spec,
                                agg=agg,
                                status="rejected",
                                reject_reason="holdout_top_vol_profit_share",
                                compiled=compiled,
                                holdout_aggregate=holdout_agg,
                            )
                            rejected += 1
                            _rej("holdout_top_vol_profit_share")
                            continue
                # ---- Holdout segment consistency gate (temporal stability)
                seg_n = int(getattr(sc, "holdout_segment_count", 0) or 0)
                seg_pos_min = float(getattr(sc, "holdout_min_pos_segment_ratio", 0.0) or 0.0)
                seg_min_ret = float(getattr(sc, "holdout_min_segment_return", -1.0) or -1.0)
                if int(seg_n) > 1 and (
                    (_finite(seg_pos_min) and float(seg_pos_min) > 0.0)
                    or (_finite(seg_min_ret) and float(seg_min_ret) > -1.0 + 1e-12)
                ):
                    seg = holdout_agg.get("segment_consistency", None) if isinstance(holdout_agg, dict) else None
                    if isinstance(holdout_agg, dict):
                        holdout_agg["holdout_segment_count"] = int(seg_n)
                        holdout_agg["holdout_min_pos_segment_ratio"] = float(seg_pos_min)
                        holdout_agg["holdout_min_segment_return"] = float(seg_min_ret)
                    if isinstance(seg, dict) and bool(seg.get("enabled", False)):
                        seg_eff = int(seg.get("segments_eff", 0) or 0)
                        pos_ratio = float(seg.get("pos_ratio", 0.0) or 0.0)
                        min_r = float(seg.get("min_return", 0.0) or 0.0)
                        bars_min = int(seg.get("bars_per_segment_min", 0) or 0)
                        # If holdout is too short to meaningfully segment, skip this gate.
                        if int(seg_eff) >= 2 and int(bars_min) >= 5:
                            if float(seg_pos_min) > 0.0 and _finite(pos_ratio) and float(pos_ratio) < float(seg_pos_min):
                                _debug_push(
                                    score=float(score),
                                    spec=spec,
                                    agg=agg,
                                    status="rejected",
                                    reject_reason="holdout_min_pos_segment_ratio",
                                    compiled=compiled,
                                    holdout_aggregate=holdout_agg,
                                )
                                rejected += 1
                                _rej("holdout_min_pos_segment_ratio")
                                continue
                            if float(seg_min_ret) > -1.0 + 1e-12 and _finite(min_r) and float(min_r) < float(seg_min_ret):
                                _debug_push(
                                    score=float(score),
                                    spec=spec,
                                    agg=agg,
                                    status="rejected",
                                    reject_reason="holdout_min_segment_return",
                                    compiled=compiled,
                                    holdout_aggregate=holdout_agg,
                                )
                                rejected += 1
                                _rej("holdout_min_segment_return")
                                continue
                    else:
                        if isinstance(holdout_agg, dict):
                            holdout_agg["segment_consistency_skipped"] = True
                # ---- Holdout latency stress test (execution delay; real-money robustness)
                lat_delay = int(getattr(sc, "holdout_latency_delay_bars", 0) or 0)
                lat_rr_min = float(getattr(sc, "holdout_latency_return_ratio_min", 0.0) or 0.0)
                if int(lat_delay) > 0:
                    ho_lat_agg: Dict[str, Any] = {}
                    try:
                        entry_lat = _DelayedSignal(compiled.entry, start=int(hs), end=int(he), delay=int(lat_delay))
                        exit_lat = (
                            _DelayedSignal(compiled.exit, start=int(hs), end=int(he), delay=int(lat_delay))
                            if compiled.exit is not None
                            else None
                        )

                        compiled_lat = CompiledStrategy(
                            entry=entry_lat,  # type: ignore[arg-type]
                            exit=exit_lat,    # type: ignore[arg-type]
                            entry_mask=compiled.entry_mask,
                            stop=compiled.stop,
                            take_profit=compiled.take_profit,
                            time_stop_bars=int(compiled.time_stop_bars),
                            cooldown_bars=int(compiled.cooldown_bars),
                        )

                        bt_ho_lat = self._run_backtest(compiled_lat, spec.direction, int(hs), int(he))
                        fm_ho_lat = _compute_fold_metrics(
                            -3,
                            int(hs),
                            int(he),
                            bt_ho_lat,
                            pf_cap=float(sc.profit_factor_cap),
                            pf_eps=float(sc.profit_factor_eps),
                        )
                        ho_lat_agg = {
                            "delay_bars": int(lat_delay),
                            "net_return": float(fm_ho_lat.net_return),
                            "max_drawdown": float(fm_ho_lat.max_drawdown),
                            "profit_factor": float(fm_ho_lat.profit_factor),
                            "trades": int(fm_ho_lat.trades),
                            "win_rate": float(fm_ho_lat.win_rate),
                            "avg_trade_return": float(fm_ho_lat.avg_trade_return),
                            "exposure": float(fm_ho_lat.exposure),
                            "turnover_per_1000": float(fm_ho_lat.turnover),
                            "avg_hold_bars": float(getattr(fm_ho_lat, "avg_holding_bars", 0.0) or 0.0),
                            "liquidations": int(getattr(fm_ho_lat, "liquidations", 0) or 0),
                            "liquidation_paid": float(getattr(fm_ho_lat, "liquidation_paid", 0.0) or 0.0),
                            "start": int(hs),
                            "end": int(he),
                        }
                        # Ratios vs base holdout (only meaningful when base > 0 / >0 trades)
                        if float(ho_ret) > 0.0:
                            try:
                                ho_lat_agg["return_ratio"] = float(ho_lat_agg["net_return"]) / float(ho_ret)
                            except Exception:
                                pass
                        if int(ho_trades) > 0:
                            try:
                                ho_lat_agg["trade_ratio"] = float(ho_lat_agg["trades"]) / float(ho_trades)
                            except Exception:
                                pass

                        if isinstance(holdout_agg, dict):
                            holdout_agg["latency"] = ho_lat_agg
                    except Exception as e:
                        _debug_push(
                            score=float(score),
                            spec=spec,
                            agg=agg,
                            status="rejected",
                            reject_reason="holdout_latency_fail",
                            compiled=compiled,
                            holdout_aggregate=holdout_agg,
                        )
                        rejected += 1
                        _rej(f"holdout_latency_fail:{type(e).__name__}")
                        continue

                    # Gate: delayed holdout must still pass the same minimum criteria
                    ho_lat_tr = int(ho_lat_agg.get("trades", 0) or 0)
                    ho_lat_ret = float(ho_lat_agg.get("net_return", 0.0) or 0.0)
                    ho_lat_dd = float(ho_lat_agg.get("max_drawdown", 0.0) or 0.0)
                    ho_lat_pf = float(ho_lat_agg.get("profit_factor", 0.0) or 0.0)
                    ho_lat_liqs = int(ho_lat_agg.get("liquidations", 0) or 0)

                    if ho_lat_tr < int(ho_min_tr):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_latency_min_trades", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_latency_min_trades")
                        continue
                    if int(ho_lat_liqs) > int(max_liqs_fold):
                        _debug_push(
                            score=float(score),
                            spec=spec,
                            agg=agg,
                            status="rejected",
                            reject_reason="holdout_latency_max_liquidations_per_fold",
                            compiled=compiled,
                            holdout_aggregate=holdout_agg,
                        )
                        rejected += 1
                        _rej("holdout_latency_max_liquidations_per_fold")
                        continue
                    if ho_lat_ret < float(ho_min_ret):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_latency_min_return", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_latency_min_return")
                        continue
                    if ho_lat_dd > float(ho_max_dd):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_latency_max_drawdown_fail", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_latency_max_drawdown_fail")
                        continue
                    if ho_lat_pf < float(ho_min_pf):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_latency_min_profit_factor", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_latency_min_profit_factor")
                        continue
                    if float(lat_rr_min) > 0.0 and float(ho_ret) > 0.0 and float(ho_lat_ret) < float(ho_ret) * float(lat_rr_min):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_latency_return_ratio", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_latency_return_ratio")
                        continue
                # ---- Holdout adverse-fill stress test (intrabar STOP ambiguity; real-money robustness)
                af_stress = float(getattr(sc, "holdout_adverse_fill_stress_mult", 0.0) or 0.0)
                af_rr_min = float(getattr(sc, "holdout_adverse_fill_return_ratio_min", 0.0) or 0.0)
                base_af = float(getattr(sc, "adverse_fill_slip_mult", 0.0) or 0.0)
                if isinstance(holdout_agg, dict):
                    holdout_agg["holdout_adverse_fill_return_ratio_min"] = float(af_rr_min)
                    holdout_agg["holdout_adverse_fill_stress_mult"] = float(af_stress)
                # Only meaningful when stress > base and stress > 0
                if _finite(af_stress) and float(af_stress) > 0.0 and float(af_stress) > max(0.0, float(base_af)):
                    ho_af_agg: Dict[str, Any] = {}
                    try:
                        bt_ho_af = self._run_backtest_adverse_fill_mult(
                            compiled,
                            spec.direction,
                            int(hs),
                            int(he),
                            adverse_mult=float(af_stress),
                            stress=True,
                        )
                        fm_ho_af = _compute_fold_metrics(
                            -4,
                            int(hs),
                            int(he),
                            bt_ho_af,
                            pf_cap=float(sc.profit_factor_cap),
                            pf_eps=float(sc.profit_factor_eps),
                        )
                        ho_af_agg = {
                            "base_adverse_fill_slip_mult": float(base_af),
                            "stress_adverse_fill_slip_mult": float(af_stress),
                            "net_return": float(fm_ho_af.net_return),
                            "max_drawdown": float(fm_ho_af.max_drawdown),
                            "profit_factor": float(fm_ho_af.profit_factor),
                            "trades": int(fm_ho_af.trades),
                            "win_rate": float(fm_ho_af.win_rate),
                            "avg_trade_return": float(fm_ho_af.avg_trade_return),
                            "exposure": float(fm_ho_af.exposure),
                            "turnover_per_1000": float(fm_ho_af.turnover),
                            "avg_hold_bars": float(getattr(fm_ho_af, "avg_holding_bars", 0.0) or 0.0),
                            "liquidations": int(getattr(fm_ho_af, "liquidations", 0) or 0),
                            "liquidation_paid": float(getattr(fm_ho_af, "liquidation_paid", 0.0) or 0.0),
                            "start": int(hs),
                            "end": int(he),
                        }
                        ho_af_ret = float(ho_af_agg.get("net_return", 0.0) or 0.0)
                        if float(ho_ret) > 0.0:
                            try:
                                ho_af_agg["return_ratio"] = float(ho_af_ret) / float(ho_ret)
                            except Exception:
                                pass
                        if isinstance(holdout_agg, dict):
                            holdout_agg["adverse_fill_stress"] = ho_af_agg
                    except Exception as e:
                        _debug_push(
                            score=float(score),
                            spec=spec,
                            agg=agg,
                            status="rejected",
                            reject_reason="holdout_adverse_fill_fail",
                            compiled=compiled,
                            holdout_aggregate=holdout_agg,
                        )
                        rejected += 1
                        _rej(f"holdout_adverse_fill_fail:{type(e).__name__}")
                        continue

                    # Gate: stressed holdout must still pass the minimum criteria
                    ho_af_tr = int(ho_af_agg.get("trades", 0) or 0)
                    ho_af_ret = float(ho_af_agg.get("net_return", 0.0) or 0.0)
                    ho_af_dd = float(ho_af_agg.get("max_drawdown", 0.0) or 0.0)
                    ho_af_pf = float(ho_af_agg.get("profit_factor", 0.0) or 0.0)
                    ho_af_liqs = int(ho_af_agg.get("liquidations", 0) or 0)

                    if ho_af_tr < int(ho_min_tr):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_adverse_fill_min_trades", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_adverse_fill_min_trades")
                        continue
                    if int(ho_af_liqs) > int(max_liqs_fold):
                        _debug_push(
                            score=float(score),
                            spec=spec,
                            agg=agg,
                            status="rejected",
                            reject_reason="holdout_adverse_fill_max_liquidations_per_fold",
                            compiled=compiled,
                            holdout_aggregate=holdout_agg,
                        )
                        rejected += 1
                        _rej("holdout_adverse_fill_max_liquidations_per_fold")
                        continue
                    if ho_af_ret < float(ho_min_ret):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_adverse_fill_min_return", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_adverse_fill_min_return")
                        continue
                    if ho_af_dd > float(ho_max_dd):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_adverse_fill_max_drawdown_fail", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_adverse_fill_max_drawdown_fail")
                        continue
                    if ho_af_pf < float(ho_min_pf):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_adverse_fill_min_profit_factor", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_adverse_fill_min_profit_factor")
                        continue
                    if float(af_rr_min) > 0.0 and float(ho_ret) > 0.0 and float(ho_af_ret) < float(ho_ret) * float(af_rr_min):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_adverse_fill_return_ratio", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_adverse_fill_return_ratio")
                        continue
                # ---- Holdout permutation test (Loop 3; negative control)
                ho_perm_trials = int(getattr(sc, "holdout_perm_trials", 0) or 0)
                ho_perm_max_p = float(getattr(sc, "holdout_perm_max_p", 0.0) or 0.0)
                # Run only when it matters (heap-competitive) or when gating is enabled.
                want_perm = (ho_perm_trials > 0) and (heap_competitive or ho_perm_max_p > 0.0)
                if want_perm:
                    try:
                        spec_h = str(spec.hash())
                    except Exception:
                        spec_h = ""

                    p_perm, perm_sum = self._holdout_permutation_test(
                        compiled,
                        spec.direction,
                        int(hs),
                        int(he),
                        spec_hash=spec_h,
                        base_return=float(ho_ret),
                        trials=int(ho_perm_trials),
                    )
                    perm_sum["max_p_allowed"] = float(ho_perm_max_p)
                    if isinstance(holdout_agg, dict):
                        holdout_agg["perm_test"] = perm_sum

                    if ho_perm_max_p > 0.0 and float(p_perm) > float(ho_perm_max_p):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_perm_p_value", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_perm_p_value")
                        continue
                # ---- Holdout stress test (same multiplier as fold stress)
                stress_mult = float(getattr(sc, "stress_cost_mult", 1.0) or 1.0)
                ho_stress_required = bool(getattr(sc, "holdout_stress_required", True))
                if _finite(stress_mult) and stress_mult > 1.0 and ho_stress_required:
                    ho_s_min_ret = float(getattr(sc, "holdout_stress_min_return", 0.0) or 0.0)
                    ho_s_min_pf = float(getattr(sc, "holdout_stress_min_profit_factor", 0.0) or 0.0)
                    ho_s_max_dd = float(getattr(sc, "holdout_stress_max_drawdown_limit", 0.0) or 0.0)
                    if ho_s_max_dd <= 0.0:
                        ho_s_max_dd = float(ho_max_dd)

                    try:
                        bt_ho_s = self._run_backtest_cost_mult(
                            compiled,
                            spec.direction,
                            int(hs),
                            int(he),
                            cost_mult=float(stress_mult),
                            stress=True,
                        )
                        fm_ho_s = _compute_fold_metrics(
                            -2,
                            int(hs),
                            int(he),
                            bt_ho_s,
                            pf_cap=float(sc.profit_factor_cap),
                            pf_eps=float(sc.profit_factor_eps),
                        )
                        ho_s_agg = {
                            "cost_mult": float(stress_mult),
                            "net_return": float(fm_ho_s.net_return),
                            "max_drawdown": float(fm_ho_s.max_drawdown),
                            "profit_factor": float(fm_ho_s.profit_factor),
                            "trades": int(fm_ho_s.trades),
                            "win_rate": float(fm_ho_s.win_rate),
                            "avg_trade_return": float(fm_ho_s.avg_trade_return),
                            "exposure": float(fm_ho_s.exposure),
                            "turnover_per_1000": float(fm_ho_s.turnover),
                            "avg_hold_bars": float(getattr(fm_ho_s, "avg_holding_bars", 0.0) or 0.0),
                            "liquidations": int(getattr(fm_ho_s, "liquidations", 0) or 0),
                            "liquidation_paid": float(getattr(fm_ho_s, "liquidation_paid", 0.0) or 0.0),
                            "start": int(hs),
                            "end": int(he),
                        }
                        if isinstance(holdout_agg, dict):
                            holdout_agg["stress"] = ho_s_agg
                    except Exception as e:
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_stress_fail", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej(f"holdout_stress_fail:{type(e).__name__}")
                        continue

                    # Gate: holdout under stress must still pass
                    ho_s_tr = int(ho_s_agg.get("trades", 0) or 0)
                    ho_s_ret = float(ho_s_agg.get("net_return", 0.0) or 0.0)
                    ho_s_dd = float(ho_s_agg.get("max_drawdown", 0.0) or 0.0)
                    ho_s_pf = float(ho_s_agg.get("profit_factor", 0.0) or 0.0)
                    ho_s_liqs = int(ho_s_agg.get("liquidations", 0) or 0)
                    # retention ratio on holdout (only meaningful if base holdout return > 0)
                    ho_rr_min = float(getattr(sc, "holdout_stress_return_ratio_min", 0.0) or 0.0)
                    if float(ho_ret) > 0.0:
                        try:
                            ho_s_agg["return_ratio"] = float(ho_s_ret) / float(ho_ret)
                        except Exception:
                            pass
                    if ho_s_tr < int(ho_min_tr):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_stress_min_trades", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_stress_min_trades")
                        continue
                    if int(ho_s_liqs) > int(max_liqs_fold):
                        _debug_push(
                            score=float(score),
                            spec=spec,
                            agg=agg,
                            status="rejected",
                            reject_reason="holdout_stress_max_liquidations_per_fold",
                            compiled=compiled,
                            holdout_aggregate=holdout_agg,
                        )
                        rejected += 1
                        _rej("holdout_stress_max_liquidations_per_fold")
                        continue
                    if ho_s_ret < float(ho_s_min_ret):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_stress_min_return", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_stress_min_return")
                        continue
                    if ho_s_dd > float(ho_s_max_dd):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_stress_max_drawdown_fail", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_stress_max_drawdown_fail")
                        continue
                    if ho_s_pf < float(ho_s_min_pf):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_stress_min_profit_factor", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_stress_min_profit_factor")
                        continue
                    if ho_rr_min > 0.0 and float(ho_ret) > 0.0 and float(ho_s_ret) < float(ho_ret) * float(ho_rr_min):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_stress_return_ratio", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_stress_return_ratio")
                        continue
                # ---- Holdout parameter robustness test (Loop 5)
                pr_trials = int(getattr(sc, "param_robust_trials", 0) or 0)
                pr_min_pass = float(getattr(sc, "param_robust_min_pass_ratio", 0.0) or 0.0)
                # Run only when it matters (heap-competitive) or when gating is enabled.
                want_pr = (pr_trials > 0) and (heap_competitive or pr_min_pass > 0.0)
                if want_pr:
                    try:
                        pr_ratio, pr_sum = self._holdout_param_robustness_test(
                            spec,
                            direction=spec.direction,
                            start=int(hs),
                            end=int(he),
                            base_return=float(ho_ret),
                            base_trades=int(ho_trades),
                            base_min_trades=int(ho_min_tr),
                            min_return=float(ho_min_ret),
                            max_drawdown=float(ho_max_dd),
                            min_profit_factor=float(ho_min_pf),
                            trials=int(pr_trials),
                        )
                        if isinstance(holdout_agg, dict):
                            holdout_agg["param_robust"] = pr_sum
                    except Exception as e:
                        _debug_push(
                            score=float(score),
                            spec=spec,
                            agg=agg,
                            status="rejected",
                            reject_reason="holdout_param_robust_fail",
                            compiled=compiled,
                            holdout_aggregate=holdout_agg,
                        )
                        rejected += 1
                        _rej(f"holdout_param_robust_fail:{type(e).__name__}")
                        continue

                    if pr_min_pass > 0.0 and float(pr_ratio) < float(pr_min_pass):
                        _debug_push(score=float(score), spec=spec, agg=agg, status="rejected", reject_reason="holdout_param_robust", compiled=compiled, holdout_aggregate=holdout_agg)
                        rejected += 1
                        _rej("holdout_param_robust")
                        continue
            res = CandidateResult(
                strategy_id=spec.id_str(),
                strategy_hash=spec.hash(),
                strategy_name=spec.name,
                strategy_dsl=strategy_spec_to_dsl(spec),
                direction=spec.direction,
                tags=spec.tags,
                complexity=spec.complexity(),
                score=float(score),
                fold_metrics=tuple(fold_metrics),
                aggregate=agg,
                regime_filter=_to_jsonable(spec.regime_filter),

                # NEW (Loop 4): include full spec for exact replay / audit
                spec=_to_jsonable(spec),
                spec_canonical=_to_jsonable(spec.canonical_tuple()),
                spec_obj=spec,
                holdout_metrics=holdout_metrics,
                holdout_aggregate=_to_jsonable(holdout_agg) if holdout_agg is not None else None,
            )



            accepted += 1
            _debug_push(score=float(score), spec=spec, agg=agg, status="accepted", reject_reason="", compiled=compiled, holdout_aggregate=holdout_agg)
 
            if len(heap) < int(cfg.top_k):
                heapq.heappush(heap, (float(res.score), res.strategy_hash, res))
            else:
                if float(res.score) > float(heap[0][0]):
                    heapq.heapreplace(heap, (float(res.score), res.strategy_hash, res))

        top = sorted((x[2] for x in heap), key=lambda r: float(r.score), reverse=True)
        debug_top = sorted((x[2] for x in debug_heap), key=lambda r: float(r.score), reverse=True)

        cand_stats = dict(self._cand_stats)
        cand_stats["evaluated"] = int(evaluated)
        cand_stats["backtest_backend"] = str(self._bt_backend)
        cand_stats["backtest_backend_counts"] = dict(self._bt_backend_counts)
        cand_stats["backtest_backend_info"] = dict(self._bt_backend_info)
        cand_stats["backtest_backend_counts_stress"] = dict(self._bt_backend_counts_stress)
        cand_stats["stress_settings"] = {
            "stress_cost_mult": float(getattr(self.score_cfg, "stress_cost_mult", 1.0) or 1.0),
            "stress_min_mean_return": float(getattr(self.score_cfg, "stress_min_mean_return", 0.0) or 0.0),
            "stress_min_worst_fold_return": float(getattr(self.score_cfg, "stress_min_worst_fold_return", -0.10) or -0.10),
        }
        if self.holdout_slice is not None:
            cand_stats["holdout"] = {
                "start": int(self.holdout_slice.start),  # type: ignore[arg-type]
                "end": int(self.holdout_slice.stop),     # type: ignore[arg-type]
                "bars": int(self.holdout_slice.stop) - int(self.holdout_slice.start),  # type: ignore[arg-type]
            }
        else:
            cand_stats["holdout"] = None
        # Record engine config so runs are reproducible when switching backends.
        if self._bt_backend == "engine":
            try:
                cand_stats["engine_cfg"] = _to_jsonable(asdict(self._engine_cfg)) if self._engine_cfg is not None else None
            except Exception:
                cand_stats["engine_cfg"] = str(self._engine_cfg) if self._engine_cfg is not None else None
        # ---- CSCV / PBO selection-bias diagnostic (Loop 14) ----
        # Uses the already computed per-fold returns of the top pool.
        try:
            cscv_summary, cscv_by_hash = _compute_cscv_pbo_for_results(
                top,
                max_combinations=2000,
                seed=int(getattr(cfg, "seed", 123) or 123),
            )
            # Optional CSCV stability gates (Loop 15)
            cscv_gate_sel = float(getattr(self.score_cfg, "cscv_min_selected_fraction", 0.0) or 0.0)
            cscv_gate_oos = float(getattr(self.score_cfg, "cscv_min_oos_rank_median", 0.0) or 0.0)
            cscv_gate_enabled = (cscv_gate_sel > 0.0 or cscv_gate_oos > 0.0)
            if isinstance(cscv_summary, dict) and cscv_summary:
                cand_stats["cscv"] = cscv_summary
                if cscv_by_hash:
                    annotated: List[CandidateResult] = []
                    for r in top:
                        st = cscv_by_hash.get(str(getattr(r, "strategy_hash", "") or ""))
                        if isinstance(st, dict) and st:
                            agg2 = dict(getattr(r, "aggregate", None) or {})
                            agg2["cscv"] = st
                            r = replace(r, aggregate=_to_jsonable(agg2))
                        annotated.append(r)
                    top = annotated
                # Gate the final pool by CSCV stability stats (no extra backtests)
                if cscv_gate_enabled:
                    if not cscv_by_hash:
                        cand_stats["cscv_gate"] = {
                            "enabled": True,
                            "applied": False,
                            "reason": "cscv_by_hash_empty",
                            "min_selected_fraction": float(cscv_gate_sel),
                            "min_oos_rank_median": float(cscv_gate_oos),
                            "pool_size": int(len(top)),
                        }
                    else:
                        before = int(len(top))
                        kept: List[CandidateResult] = []
                        for r in top:
                            st = {}
                            try:
                                st = cscv_by_hash.get(str(getattr(r, "strategy_hash", "") or "")) or {}
                            except Exception:
                                st = {}
                            ok = True
                            if cscv_gate_sel > 0.0:
                                try:
                                    sf = float(st.get("selected_fraction", 0.0) or 0.0)
                                except Exception:
                                    sf = 0.0
                                if sf < float(cscv_gate_sel):
                                    ok = False
                            if ok and cscv_gate_oos > 0.0:
                                try:
                                    rk = float(st.get("oos_rank_median", 0.0) or 0.0)
                                except Exception:
                                    rk = 0.0
                                if rk < float(cscv_gate_oos):
                                    ok = False
                            if ok:
                                kept.append(r)
                        dropped = before - int(len(kept))
                        top = kept
                        cand_stats["cscv_gate"] = {
                            "enabled": True,
                            "applied": True,
                            "min_selected_fraction": float(cscv_gate_sel),
                            "min_oos_rank_median": float(cscv_gate_oos),
                            "pool_size": int(before),
                            "kept": int(len(top)),
                            "dropped": int(dropped),
                        }
            elif cscv_gate_enabled:
                cand_stats["cscv_gate"] = {
                    "enabled": True,
                    "applied": False,
                    "reason": "cscv_not_computed_or_pool_too_small",
                    "min_selected_fraction": float(cscv_gate_sel),
                    "min_oos_rank_median": float(cscv_gate_oos),
                    "pool_size": int(len(top)),
                }
        except Exception as e:
            cand_stats["cscv"] = {"enabled": False, "error": f"{type(e).__name__}: {e}"}
        return MinerReport(
            evaluated=int(evaluated),
            accepted=int(accepted),
            rejected=int(rejected),
            folds=int(len(self.splits)),
            reject_reasons=reject_reasons,
            candidate_stats=_to_jsonable(cand_stats),
            top_results=tuple(top),
            debug_top=tuple(debug_top),
            search_config=_to_jsonable(asdict(self.search_cfg)),
            wf_config=_to_jsonable(asdict(self.wf_cfg)),
            scoring_config=_to_jsonable(asdict(self.score_cfg)),
        )


# -----------------------------
# Export
# -----------------------------
def save_results_json(path: str, report: MinerReport, meta: Optional[Dict[str, Any]] = None) -> None:
    p = str(path).strip()
    require(p, "path must be non-empty")

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "meta": _to_jsonable(meta) if meta else None,

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
                "strategy_dsl": r.strategy_dsl,
                "direction": r.direction,
                "tags": list(r.tags),
                "complexity": r.complexity,
                "score": r.score,
                "aggregate": r.aggregate,
                "fold_metrics": [_to_jsonable(f) for f in r.fold_metrics],
                "regime_filter": r.regime_filter,

                # NEW (Loop 4): full spec for exact replay
                "spec": r.spec,
                "spec_canonical": r.spec_canonical,
                # NEW (Loop 4): final holdout
                "holdout_metrics": _to_jsonable(r.holdout_metrics) if getattr(r, "holdout_metrics", None) is not None else None,
                "holdout_aggregate": _to_jsonable(r.holdout_aggregate) if getattr(r, "holdout_aggregate", None) is not None else None,

            }
            for r in report.top_results
        ],
        "debug_top": [_to_jsonable(x) for x in getattr(report, "debug_top", ())],
    }

    # keep file clean if no meta is provided
    if payload.get("meta") is None:
        payload.pop("meta", None)

    payload = _to_jsonable(payload)

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
    # --- p-value sanity (Student-t, SciPy-free) ---
    p = _p_value_two_sided_from_t(1.0, df=1)
    require(abs(float(p) - 0.5) < 1e-6, f"p-value sanity failed df=1 t=1: {p}")
    p = _p_value_two_sided_from_t(2.2281388519649385, df=10)  # ~5% two-sided
    require(abs(float(p) - 0.05) < 5e-3, f"p-value sanity failed df=10 t=2.228: {p}")
    p = _p_value_two_sided_from_t(1.959963984540054, df=1_000_000)  # ~5% two-sided (normal limit)
    require(abs(float(p) - 0.05) < 2e-3, f"p-value sanity failed df->inf t=1.96: {p}")
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
        holdout_bars=80,
    )

    search = SearchConfig(
        mode="iterate",
        max_evals=60,
        top_k=10,
        seed=123,
        backtest_mode="engine",
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
        min_mean_return=-1.0,
        min_profit_factor=0.0,
        holdout_required=True,
        holdout_min_trades=0,
        holdout_min_return=-1.0,
        holdout_min_profit_factor=0.0,
        holdout_max_drawdown_limit=0.99,
    )

    miner = StrategyMiner(series, space, wf, search, scoring)
    report = miner.run()
    # Holdout gate should force holdout_aggregate to be computed for accepted candidates.
    if int(getattr(wf, 'holdout_bars', 0) or 0) > 0 and bool(getattr(scoring, 'holdout_required', False)):
        for r in report.top_results:
            require(r.holdout_aggregate is not None, "Expected holdout_aggregate for top_results when holdout_required")

    require(report.evaluated > 0, "Expected evaluated > 0")
    require(isinstance(report.reject_reasons, dict), "reject_reasons must be dict")
    if report.accepted < 1:
        print(
            "WARNING: self-test produced 0 accepted strategies (accepted=0). "
            "This is not necessarily a bug; it can happen if scoring filters are strict or the synthetic series changed."
        )


    # ensure sorted desc
    scores = [r.score for r in report.top_results]
    require(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Top results not sorted by score desc")
    # FoldMetrics baseline sanity: return/DD must be anchored to equity at start-1 (pre-entry),
    # otherwise entry fees paid at the window boundary inflate returns and hide drawdowns.
    bt_out = BacktestOutput(
        equity_curve=[1.0, 1.0, 0.99, 1.0],
        trades=[],
        fees_paid=0.0,
        funding_paid=0.0,
    )
    fm = _compute_fold_metrics(0, 2, 4, bt_out)
    require(abs(float(fm.net_return)) < 1e-9, f"FoldMetrics net_return baseline leak: {fm.net_return}")
    require(abs(float(fm.max_drawdown) - 0.01) < 1e-9, f"FoldMetrics max_drawdown baseline leak: {fm.max_drawdown}")
 
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
