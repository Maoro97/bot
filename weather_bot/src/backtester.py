"""
Backtester — simulates the trading strategy on historical data.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .models import BacktestResult, TemperatureBucket
from .probability_engine import ProbabilityEngine
from .edge_detector import EdgeDetector
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class Backtester:
    """
    Simulates the bot's strategy against historical forecast and outcome data
    to measure performance and calibrate risk parameters.

    Expected input CSVs
    -------------------
    historical_forecasts : date, location, ecmwf_temp, gfs_temp, mean_temp, std_temp
    historical_prices    : date, location, bucket_label, price
    historical_outcomes  : date, location, winning_bucket
    """

    def __init__(
        self,
        initial_bankroll: float = 100.0,
        min_edge: float = 0.10,
        min_kelly: float = 0.02,
    ):
        self._bankroll0 = initial_bankroll
        self._engine = ProbabilityEngine(use_t_dist=True)
        self._detector = EdgeDetector(min_edge=min_edge, min_kelly=min_kelly)

    # ─────────────────────────────────────────────────────────────────────────

    def backtest(
        self,
        historical_forecasts: pd.DataFrame,
        historical_prices: pd.DataFrame,
        historical_outcomes: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """
        Run full backtest between *start_date* and *end_date* (inclusive).
        Returns a BacktestResult with detailed statistics.
        """
        bankroll = self._bankroll0
        trades: list[dict] = []
        peak = bankroll

        # Filter date range
        mask_f = (historical_forecasts["date"] >= start_date) & \
                 (historical_forecasts["date"] <= end_date)
        mask_p = (historical_prices["date"] >= start_date) & \
                 (historical_prices["date"] <= end_date)
        mask_o = (historical_outcomes["date"] >= start_date) & \
                 (historical_outcomes["date"] <= end_date)

        forecasts = historical_forecasts[mask_f].copy()
        prices    = historical_prices[mask_p].copy()
        outcomes  = historical_outcomes[mask_o].copy()

        for _, row in forecasts.iterrows():
            date_str  = row["date"]
            location  = row["location"]
            mean_temp = float(row["mean_temp"])
            std_temp  = float(row["std_temp"])

            # Get market prices for this date + location
            day_prices = prices[
                (prices["date"] == date_str) & (prices["location"] == location)
            ]
            if day_prices.empty:
                continue

            # Reconstruct buckets from price rows
            buckets = self._build_buckets(day_prices)
            if not buckets:
                continue

            # Compute model probabilities
            probs = self._engine.forecast_to_distribution(mean_temp, std_temp, buckets)

            # Build market-like price dict
            mkt_prices = dict(zip(day_prices["bucket_label"], day_prices["price"]))

            # Find edges
            for bucket in buckets:
                label = bucket.label
                model_prob = probs.get(label, 0.0)
                mkt_price  = mkt_prices.get(label, 0.0)
                if mkt_price <= 0:
                    continue

                edge = model_prob - mkt_price
                kf   = self._detector.kelly_fraction(model_prob, mkt_price)
                ev   = self._detector.calculate_expected_value(model_prob, mkt_price)

                if abs(edge) < self._detector.min_edge or kf < self._detector.min_kelly:
                    continue

                # Sizing
                size = min(kf * 0.25 * bankroll, 3.0)
                size = max(0.50, size)

                # Outcome lookup
                outcome_row = outcomes[
                    (outcomes["date"] == date_str) & (outcomes["location"] == location)
                ]
                if outcome_row.empty:
                    continue

                winning_bucket = outcome_row.iloc[0]["winning_bucket"]
                won = winning_bucket == label

                if edge > 0 and ev > 0:  # BUY signal
                    pnl = size * (1 - mkt_price) / mkt_price if won else -size
                    bankroll += pnl
                    peak = max(peak, bankroll)

                    trades.append({
                        "date": date_str,
                        "location": location,
                        "bucket": label,
                        "side": "BUY",
                        "model_prob": round(model_prob, 4),
                        "market_price": round(mkt_price, 4),
                        "edge": round(edge, 4),
                        "kelly": round(kf, 4),
                        "ev": round(ev, 4),
                        "size": round(size, 2),
                        "won": won,
                        "pnl": round(pnl, 4),
                        "bankroll": round(bankroll, 2),
                    })

        if not trades:
            logger.warning("No trades generated in backtest period")
            return BacktestResult(
                start_date=start_date, end_date=end_date,
                total_trades=0, win_rate=0, total_pnl=0,
                roi_pct=0, max_drawdown=0, sharpe_ratio=0,
                calibration_score=0, trades=[],
            )

        df = pd.DataFrame(trades)
        wins        = int(df["won"].sum())
        total       = len(df)
        win_rate    = wins / total
        total_pnl   = float(df["pnl"].sum())
        roi_pct     = total_pnl / self._bankroll0 * 100

        # Max drawdown
        running = self._bankroll0 + df["pnl"].cumsum()
        peak_series = running.cummax()
        drawdowns   = (peak_series - running) / peak_series
        max_dd      = float(drawdowns.max())

        # Sharpe (daily P&L)
        daily_pnl = df.groupby("date")["pnl"].sum()
        sharpe = (daily_pnl.mean() / daily_pnl.std() * math.sqrt(252)
                  if daily_pnl.std() > 0 else 0.0)

        # Brier score (calibration)
        brier = ProbabilityEngine.brier_score(
            df["model_prob"].tolist(),
            df["won"].astype(int).tolist(),
        )

        logger.info(
            "Backtest %s->%s: trades=%d win_rate=%.1f%% pnl=$%.2f roi=%.1f%% "
            "maxDD=%.1f%% sharpe=%.2f brier=%.3f",
            start_date, end_date, total, win_rate*100, total_pnl,
            roi_pct, max_dd*100, sharpe, brier,
        )

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_trades=total,
            win_rate=round(win_rate, 4),
            total_pnl=round(total_pnl, 2),
            roi_pct=round(roi_pct, 2),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 3),
            calibration_score=round(brier, 4),
            trades=trades,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_buckets(day_prices: pd.DataFrame) -> list[TemperatureBucket]:
        from .polymarket_client import _parse_bucket
        buckets = []
        for _, row in day_prices.iterrows():
            label    = str(row["bucket_label"])
            token_id = str(row.get("token_id", ""))
            try:
                buckets.append(_parse_bucket(label, token_id))
            except Exception:
                pass
        return buckets

    def save_results(self, result: BacktestResult, output_path: str = "data/backtest_result.json"):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "start_date": result.start_date,
                "end_date": result.end_date,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "total_pnl": result.total_pnl,
                "roi_pct": result.roi_pct,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "calibration_score": result.calibration_score,
                "trades": result.trades,
            }, f, indent=2)
        logger.info("Backtest results saved to %s", output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Open-Meteo historical data fetcher
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    async def fetch_historical_forecasts(
        lat: float, lon: float,
        start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """
        Download historical hourly temperatures from Open-Meteo
        (free historical API, up to 3 months back).
        Returns a DataFrame with columns: date, max_temp, min_temp, mean_temp.
        """
        import httpx
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
            "timezone": "UTC",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        daily = data.get("daily", {})
        df = pd.DataFrame({
            "date":      daily.get("time", []),
            "max_temp":  daily.get("temperature_2m_max", []),
            "min_temp":  daily.get("temperature_2m_min", []),
            "mean_temp": daily.get("temperature_2m_mean", []),
        })
        return df
