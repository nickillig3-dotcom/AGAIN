from __future__ import annotations

"""
Data_Binance.py
================

Purpose
-------
Robust OHLCV data access for Binance USDⓈ-M Perpetual Futures (public market data).
This module focuses on *realistic, strict, fail-fast* behavior, because strategy mining
is extremely sensitive to silent data issues.

Design principles
-----------------
- OHLCV only.
- Strict validation (uses Core_Types.OhlcvSeries invariants).
- No third-party dependencies (urllib/json only).
- Defensive HTTP retry/backoff for transient failures and rate limits.
- Offline self-test by default; optional live integration test behind env var.

Binance endpoints (USDS-M Futures)
---------------------------------
- Mainnet REST base: https://fapi.binance.com
- Testnet REST base: https://demo-fapi.binance.com
- Klines: GET /fapi/v1/klines
- Funding rates: GET /fapi/v1/fundingRate
Notes
-----
- Public market data endpoints do not require an API key.
- For strategy mining/backtesting, prefer historical closed candles; live candle handling
  will be addressed later (streaming + last candle completeness).
"""

import json
import os
import random
import time
import bisect
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from Core_Types import ContractError, OhlcvSeries, ValidationError, require


__all__ = [
    "BINANCE_USDM_MAINNET",
    "BINANCE_USDM_TESTNET",
    "KLINES_PATH",
    "FUNDING_RATE_PATH",
    "BinanceHttpError",
    "BinanceAPIError",
    "interval_to_ms",
    "normalize_symbol",
    "binance_http_get_json",
    "parse_klines_to_series",
    "download_ohlcv",
    "download_funding_rates",
    "build_funding_rate_fn",
]


BINANCE_USDM_MAINNET: str = "https://fapi.binance.com"
BINANCE_USDM_TESTNET: str = "https://demo-fapi.binance.com"
KLINES_PATH: str = "/fapi/v1/klines"
FUNDING_RATE_PATH: str = "/fapi/v1/fundingRate"
# For your target range 1m..1h; extend later if you want.
_INTERVAL_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
}


# -----------------------------
# Exceptions (HTTP / API layer)
# -----------------------------
class BinanceHttpError(ContractError):
    """Networking / HTTP-level errors when communicating with Binance."""


class BinanceAPIError(BinanceHttpError):
    """
    Binance returned a structured API error (often JSON with `code` and `msg`).

    This is different from purely network/transport errors.
    """

    def __init__(self, message: str, *, http_status: int, api_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.http_status = int(http_status)
        self.api_code = int(api_code) if api_code is not None else None


# -----------------------------
# Helpers
# -----------------------------
def interval_to_ms(interval: str) -> int:
    interval = str(interval).strip()
    require(interval in _INTERVAL_MS, f"Unsupported interval {interval!r}. Supported: {sorted(_INTERVAL_MS.keys())}")
    return _INTERVAL_MS[interval]


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol input to Binance futures format, e.g. "BTCUSDT".
    We tolerate common separators (BTC/USDT, BTC-USDT, BTC_USDT).
    """
    require(symbol is not None, "symbol must not be None")
    s = str(symbol).strip().upper()
    for ch in ("/", "-", "_", " "):
        s = s.replace(ch, "")
    require(len(s) >= 6, f"symbol looks invalid after normalization: {s!r}")
    return s


def _build_url(base_url: str, path: str, params: Dict[str, Any]) -> str:
    base = str(base_url).rstrip("/")
    pth = path if path.startswith("/") else "/" + path

    # Drop None values
    clean: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        clean[k] = v

    qs = urllib.parse.urlencode(clean, doseq=True, safe=":,")
    return f"{base}{pth}" + (f"?{qs}" if qs else "")


def _sleep_backoff(attempt: int, *, base_s: float, cap_s: float) -> None:
    """
    Exponential backoff with jitter.
    attempt=0 -> base_s
    attempt=1 -> base_s*2
    ...
    """
    delay = min(cap_s, base_s * (2.0 ** attempt))
    # jitter in [0, 0.25*delay]
    jitter = random.random() * (0.25 * delay)
    time.sleep(delay + jitter)


def _try_parse_json_bytes(raw: bytes) -> Any:
    txt = raw.decode("utf-8", errors="replace")
    return json.loads(txt)


def _extract_binance_error(payload: Any) -> Tuple[Optional[int], Optional[str]]:
    """
    Binance often returns {"code": -xxxx, "msg": "..."} for errors.
    """
    if isinstance(payload, dict):
        code = payload.get("code")
        msg = payload.get("msg")
        try:
            code_i = int(code) if code is not None else None
        except Exception:
            code_i = None
        msg_s = str(msg) if msg is not None else None
        return code_i, msg_s
    return None, None


# -----------------------------
# HTTP GET (robust, no deps)
# -----------------------------
def binance_http_get_json(
    base_url: str,
    path: str,
    params: Dict[str, Any],
    *,
    timeout_s: float = 10.0,
    max_retries: int = 6,
    backoff_base_s: float = 0.25,
    backoff_cap_s: float = 10.0,
    user_agent: str = "AGAIN-StrategyMiner/0.1 (urllib)",
) -> Any:
    """
    Perform a robust GET request returning parsed JSON.

    Retries:
    - network errors (URLError, timeouts)
    - 408/429/5xx (rate limit / transient server errors)
    - JSON parse failures (rare, but can happen with edge gateway errors)

    Notes:
    - HTTP 418 (auto-ban) is treated as fatal (no retry here; ban duration can be minutes to days).
    """
    require(max_retries >= 0, "max_retries must be >= 0")
    require(timeout_s > 0, "timeout_s must be > 0")

    url = _build_url(base_url, path, params)

    last_error: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "User-Agent": user_agent,
                },
            )

            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = int(getattr(resp, "status", resp.getcode()))
                raw = resp.read()

            # Even if status is 200, some gateways can return HTML; parse strictly.
            try:
                payload = _try_parse_json_bytes(raw)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    _sleep_backoff(attempt, base_s=backoff_base_s, cap_s=backoff_cap_s)
                    continue
                raise BinanceHttpError(f"Failed to parse JSON from Binance. url={url!r}") from e

            # Binance sometimes returns errors as JSON with non-2xx status,
            # but here we already got the payload.
            return payload

        except urllib.error.HTTPError as e:
            status = int(e.code)
            raw = b""
            try:
                raw = e.read()  # type: ignore[attr-defined]
            except Exception:
                raw = b""

            payload: Any = None
            if raw:
                try:
                    payload = _try_parse_json_bytes(raw)
                except Exception:
                    payload = None

            api_code, api_msg = _extract_binance_error(payload)
            msg = api_msg or (raw.decode("utf-8", errors="replace")[:300] if raw else str(e))

            # 418 = banned; do not retry blindly.
            if status == 418:
                raise BinanceAPIError(
                    f"Binance returned HTTP 418 (auto-banned). url={url!r}. msg={msg}",
                    http_status=status,
                    api_code=api_code,
                ) from e

            # Retryable statuses
            if status in (408, 429, 500, 502, 503, 504):
                last_error = e

                if attempt < max_retries:
                    # If rate-limited, respect Retry-After if present.
                    if status == 429:
                        retry_after = None
                        try:
                            ra = e.headers.get("Retry-After")
                            if ra is not None:
                                retry_after = float(ra)
                        except Exception:
                            retry_after = None

                        if retry_after is not None and retry_after > 0:
                            time.sleep(min(backoff_cap_s, retry_after))
                        else:
                            _sleep_backoff(attempt, base_s=backoff_base_s, cap_s=backoff_cap_s)
                    else:
                        _sleep_backoff(attempt, base_s=backoff_base_s, cap_s=backoff_cap_s)
                    continue

            # Non-retryable
            raise BinanceAPIError(
                f"Binance HTTP error. status={status} url={url!r} msg={msg}",
                http_status=status,
                api_code=api_code,
            ) from e

        except (urllib.error.URLError, TimeoutError) as e:
            last_error = e
            if attempt < max_retries:
                _sleep_backoff(attempt, base_s=backoff_base_s, cap_s=backoff_cap_s)
                continue
            raise BinanceHttpError(f"Network error talking to Binance. url={url!r}") from e

        except Exception as e:  # pragma: no cover
            # Unexpected errors should not be masked.
            raise

    # Should never reach here; loop either returns or raises.
    raise BinanceHttpError(f"Request failed after retries. url={url!r}. last_error={last_error!r}")


# -----------------------------
# Parsing: Binance klines -> OhlcvSeries
# -----------------------------
KlineRow = Sequence[Any]


def parse_klines_to_series(
    klines: Sequence[KlineRow],
    *,
    symbol: Optional[str] = None,
    interval: Optional[str] = None,
    allow_empty: bool = False,
) -> OhlcvSeries:
    """
    Parse Binance futures klines into OhlcvSeries.

    Binance kline schema (USDS-M futures) is typically:
    [
      [
        open_time, open, high, low, close, volume,
        close_time, quote_asset_volume, number_of_trades,
        taker_buy_base_volume, taker_buy_quote_volume, ignore
      ],
      ...
    ]

    We only use fields 0..5 (OHLCV).
    """
    require(klines is not None, "klines must not be None")

    if len(klines) == 0:
        if allow_empty:
            s = OhlcvSeries(
                ts_ms=[],
                open=[],
                high=[],
                low=[],
                close=[],
                volume=[],
                symbol=symbol,
                timeframe=interval,
                validate_on_init=False,
            )
            s.validate(allow_empty=True)
            return s
        raise ValidationError("No klines to parse (empty response).")

    ts_ms: List[int] = []
    o: List[float] = []
    h: List[float] = []
    l: List[float] = []
    c: List[float] = []
    v: List[float] = []

    last_ts: Optional[int] = None

    for i, row in enumerate(klines):
        require(isinstance(row, (list, tuple)), f"kline row[{i}] must be list/tuple, got {type(row).__name__}")
        require(len(row) >= 6, f"kline row[{i}] must have at least 6 fields, got {len(row)}")

        try:
            t = int(row[0])
            op = float(row[1])
            hi = float(row[2])
            lo = float(row[3])
            cl = float(row[4])
            vol = float(row[5])
        except Exception as e:
            raise ValidationError(f"Failed to parse kline row[{i}]: {row!r}") from e

        # Ensure strictly increasing timestamps (skip exact duplicates, error on out-of-order)
        if last_ts is not None:
            if t == last_ts:
                continue
            require(t > last_ts, f"Klines not strictly increasing at row[{i}]: {t} <= {last_ts}")
        last_ts = t

        ts_ms.append(t)
        o.append(op)
        h.append(hi)
        l.append(lo)
        c.append(cl)
        v.append(vol)

    series = OhlcvSeries(
        ts_ms=ts_ms,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        symbol=symbol,
        timeframe=interval,
    )
    return series


# -----------------------------
# Download: historical paging
# -----------------------------
def download_ohlcv(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int] = None,
    *,
    base_url: str = BINANCE_USDM_MAINNET,
    limit: int = 1500,
    timeout_s: float = 10.0,
    max_retries: int = 6,
) -> OhlcvSeries:
    """
    Download historical klines (OHLCV) from Binance USDS-M futures.

    Parameters
    ----------
    symbol:
        e.g. "BTCUSDT" (USDT-margined perp symbol). Separators tolerated (BTC/USDT).
    interval:
        One of: 1m, 3m, 5m, 15m, 30m, 1h
    start_ms:
        Inclusive start timestamp in epoch milliseconds.
    end_ms:
        Inclusive end timestamp in epoch milliseconds. If None, downloads forward until Binance returns no more data.

    Behavior
    --------
    - Pages forward using startTime.
    - Deduplicates exact duplicate open_time rows (defensive).
    - Raises ValidationError if nothing is returned.

    Performance / safety
    --------------------
    - Uses `limit` up to 1500 (Binance max for this endpoint).
    - Contains loop safety to avoid infinite loops if the API behaves unexpectedly.
    """
    sym = normalize_symbol(symbol)
    interval = str(interval).strip()
    tf_ms = interval_to_ms(interval)

    require(isinstance(start_ms, int), "start_ms must be int epoch milliseconds")
    require(start_ms > 0, f"start_ms must be > 0, got {start_ms}")

    if end_ms is not None:
        require(isinstance(end_ms, int), "end_ms must be int epoch milliseconds")
        require(end_ms >= start_ms, f"end_ms must be >= start_ms, got end_ms={end_ms}, start_ms={start_ms}")

    require(isinstance(limit, int), "limit must be int")
    require(1 <= limit <= 1500, f"limit must be in [1, 1500], got {limit}")

    ts_ms: List[int] = []
    o: List[float] = []
    h: List[float] = []
    l: List[float] = []
    c: List[float] = []
    v: List[float] = []

    current_start = int(start_ms)
    last_appended_ts: Optional[int] = None

    # Loop safety: if something goes wrong, we break with a clear error.
    max_pages = 1_000_000  # extremely high; mainly to guard against true infinite loops
    pages = 0

    while True:
        pages += 1
        if pages > max_pages:
            raise BinanceHttpError("Exceeded max_pages while downloading klines (safety stop).")

        params: Dict[str, Any] = {
            "symbol": sym,
            "interval": interval,
            "startTime": current_start,
            "limit": limit,
        }
        if end_ms is not None:
            params["endTime"] = int(end_ms)

        payload = binance_http_get_json(
            base_url=base_url,
            path=KLINES_PATH,
            params=params,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

        require(isinstance(payload, list), f"Expected list response from klines endpoint, got {type(payload).__name__}")
        if len(payload) == 0:
            break

        # Append new rows, strict monotonicity
        for row in payload:
            require(isinstance(row, (list, tuple)), f"Kline row must be list/tuple, got {type(row).__name__}")
            require(len(row) >= 6, f"Kline row must have at least 6 fields, got {len(row)}")
            t = int(row[0])

            if last_appended_ts is not None:
                if t == last_appended_ts:
                    # defensive dedup
                    continue
                require(t > last_appended_ts, f"API returned out-of-order klines: {t} <= {last_appended_ts}")

            ts_ms.append(t)
            o.append(float(row[1]))
            h.append(float(row[2]))
            l.append(float(row[3]))
            c.append(float(row[4]))
            v.append(float(row[5]))
            last_appended_ts = t

        last_open_time = int(payload[-1][0])
        next_start = last_open_time + tf_ms

        # Safety: if Binance keeps returning the same last_open_time, avoid infinite loop.
        require(
            next_start > current_start,
            f"Loop safety triggered: next_start={next_start} not > current_start={current_start}. "
            f"last_open_time={last_open_time}, tf_ms={tf_ms}",
        )

        current_start = next_start

        if end_ms is not None and current_start > int(end_ms):
            break

        # If fewer than limit rows, we likely reached the end of available data.
        if len(payload) < limit:
            break

    require(len(ts_ms) > 0, "No klines returned for the requested range.")

    series = OhlcvSeries(
        ts_ms=ts_ms,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
        symbol=sym,
        timeframe=interval,
    )
    return series
# -----------------------------
# Funding rates (perps)
# -----------------------------
FundingRateEntry = Tuple[int, float]  # (fundingTime_ms, fundingRate as fraction, e.g. 0.0001 == 1 bps)


def download_funding_rates(
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    base_url: str = BINANCE_USDM_MAINNET,
    limit: int = 1000,
    timeout_s: int = 10,
    max_retries: int = 6,
) -> List[FundingRateEntry]:
    """
    Download historical funding rates for Binance USDⓈ-M perpetual futures.

    Endpoint: GET /fapi/v1/fundingRate
    Returns list of (fundingTime_ms, fundingRate_fraction).

    Notes:
    - Binance returns funding events at 8h boundaries (00:00/08:00/16:00 UTC).
    - We page forward using startTime=endTime and `limit` (max 1000).
    """
    sym = normalize_symbol(symbol)
    require(isinstance(start_ms, int) and start_ms > 0, f"start_ms must be int epoch ms > 0, got {start_ms}")
    require(isinstance(end_ms, int) and end_ms >= start_ms, f"end_ms must be int >= start_ms, got {end_ms}")

    require(isinstance(limit, int), "limit must be int")
    require(1 <= limit <= 1000, f"limit must be in [1, 1000], got {limit}")

    out: List[FundingRateEntry] = []
    current_start = int(start_ms)
    last_t: Optional[int] = None

    max_pages = 1_000_000
    pages = 0

    while True:
        pages += 1
        require(pages <= max_pages, "Loop safety: too many pages in funding download")

        params = {
            "symbol": sym,
            "startTime": int(current_start),
            "endTime": int(end_ms),
            "limit": int(limit),
        }

        payload = binance_http_get_json(
            base_url=base_url,
            path=FUNDING_RATE_PATH,
            params=params,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

        require(isinstance(payload, list), f"FundingRate payload expected list, got {type(payload).__name__}")
        if not payload:
            break

        appended = 0
        for row in payload:
            require(isinstance(row, dict), f"FundingRate row expected dict, got {type(row).__name__}")
            try:
                t = int(row.get("fundingTime"))
                r = float(row.get("fundingRate"))
            except Exception as e:
                raise ValidationError(f"Invalid fundingRate row: {row!r}") from e

            if last_t is not None and t <= last_t:
                # API sometimes returns duplicates at boundaries; ignore exact duplicates, fail on regressions.
                if t == last_t:
                    continue
                raise ValidationError(f"Funding times not strictly increasing: {t} after {last_t}")

            out.append((t, float(r)))
            last_t = t
            appended += 1

        if appended == 0:
            break

        # advance paging
        require(last_t is not None, "Internal error: last_t None after appends")
        next_start = int(last_t + 1)
        if next_start > int(end_ms):
            break

        # if fewer than limit, we're done
        if len(payload) < int(limit):
            break

        current_start = next_start

    return out


def build_funding_rate_fn(
    funding: Sequence[FundingRateEntry],
    *,
    default_rate: float = 0.0,
    use_last_known: bool = False,
) -> Any:
    """
    Build a callable(ts_ms)->fundingRate.

    For Binance, Backtest_Engine applies funding when ts_ms % funding_period_ms == 0
    (8h boundaries). If your candle timestamps are aligned, exact matches should exist.

    If use_last_known=True, missing timestamps fall back to the last known <= ts_ms.
    """
    mp: Dict[int, float] = {}
    for t, r in funding:
        mp[int(t)] = float(r)
    keys = sorted(mp.keys())

    def _exact(ts_ms: int) -> float:
        return float(mp.get(int(ts_ms), float(default_rate)))

    def _last_known(ts_ms: int) -> float:
        t = int(ts_ms)
        if t in mp:
            return float(mp[t])
        i = bisect.bisect_right(keys, t) - 1
        if i >= 0:
            return float(mp[keys[i]])
        return float(default_rate)

    return _last_known if bool(use_last_known) else _exact


# -----------------------------
# Minimal self-test (offline by default)
# -----------------------------
def _expect_raises(exc_type, fn, *, contains: Optional[str] = None) -> None:
    try:
        fn()
    except exc_type as e:
        if contains is not None:
            require(contains in str(e), f"Expected error message to contain {contains!r}, got {str(e)!r}")
        return
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Expected {exc_type.__name__}, but got {type(e).__name__}: {e}") from e
    raise AssertionError(f"Expected {exc_type.__name__} to be raised, but no exception occurred.")


def _self_test_offline() -> None:
    # Typical Binance kline shape (we only need first 6 fields)
    klines_ok = [
        [1, "10", "11", "9", "10.5", "100", 2, "0", 0, "0", "0", "0"],
        [2, "10.5", "12", "10", "11", "150", 3, "0", 0, "0", "0", "0"],
        [3, "11", "13", "10.5", "12", "200", 4, "0", 0, "0", "0", "0"],
    ]
    s = parse_klines_to_series(klines_ok, symbol="TESTUSDT", interval="1m")
    require(s.n == 3, "parsed series length mismatch")
    require(s.get_bar(0).ts_ms == 1, "first timestamp mismatch")

    # Bad klines: high < low -> should fail validation
    klines_bad = [
        [1, "10", "9", "9.5", "9.7", "100"],
    ]

    def _bad_parse():
        parse_klines_to_series(klines_bad, symbol="TESTUSDT", interval="1m")

    _expect_raises(ValidationError, _bad_parse)

    # interval_to_ms validation
    require(interval_to_ms("1m") == 60_000, "interval_to_ms(1m) incorrect")

    def _bad_interval():
        interval_to_ms("2m")

    _expect_raises(ValidationError, _bad_interval, contains="Unsupported interval")
    # funding rate fn helper
    fr = [(1000, 0.0001), (2000, -0.0002)]
    fn = build_funding_rate_fn(fr, default_rate=0.0, use_last_known=False)
    require(abs(fn(1000) - 0.0001) < 1e-12, "funding exact match failed")
    require(abs(fn(1500) - 0.0) < 1e-12, "funding default missing failed")
    fn2 = build_funding_rate_fn(fr, default_rate=0.0, use_last_known=True)
    require(abs(fn2(1500) - 0.0001) < 1e-12, "funding last_known failed")
    print("Data_Binance offline self-test: OK")


def _self_test_live_optional() -> None:
    """
    Optional live integration test (disabled by default).
    Enable with:
      PowerShell:  $env:BINANCE_LIVE_TEST="1"
      cmd.exe:     set BINANCE_LIVE_TEST=1
    """
    payload = binance_http_get_json(
        base_url=BINANCE_USDM_MAINNET,
        path=KLINES_PATH,
        params={"symbol": "BTCUSDT", "interval": "1m", "limit": 10},
        timeout_s=10.0,
        max_retries=6,
    )
    s = parse_klines_to_series(payload, symbol="BTCUSDT", interval="1m")
    require(s.n > 0, "live test returned empty series")
    # Basic sanity: timestamps increasing
    require(s.ts_ms[0] < s.ts_ms[-1], "live test timestamps not increasing")
    print(f"Data_Binance live test: OK (received {s.n} klines)")


def _self_test() -> None:
    _self_test_offline()
    if os.environ.get("BINANCE_LIVE_TEST", "").strip() == "1":
        _self_test_live_optional()
    print("Data_Binance self-test: OK")


if __name__ == "__main__":
    _self_test()
