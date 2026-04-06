"""
Weather Bot — main entry point.
Orchestrates all components in a continuous async loop.

Usage:
    python -m weather_bot.weather_main            # live mode
    python -m weather_bot.weather_main --paper    # paper-trade (default)
    python -m weather_bot.weather_main --backtest # run a quick backtest and exit
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import yaml

from .src.models import Location, ConsensusForecast, DailyPnL
from .src.weather_fetcher import WeatherFetcher
from .src.probability_engine import ProbabilityEngine
from .src.polymarket_client import PolymarketClient
from .src.edge_detector import EdgeDetector
from .src.risk_manager import RiskManager
from .src.trade_executor import TradeExecutor
from .src.telegram_bot import TelegramReporter

# Reconfigure stdout to UTF-8 on Windows where the default codepage may not
# support Unicode characters (e.g. cp1252, cp1255).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure data directories exist before setting up file logging
Path("data/trades").mkdir(parents=True, exist_ok=True)
Path("data/calibration").mkdir(parents=True, exist_ok=True)
Path("data/pnl").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/weather_bot.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── GFS / ECMWF update hours (UTC) ──────────────────────────────────────────
MODEL_UPDATE_HOURS = {0, 6, 12, 18}

CONFIG_PATH    = Path(__file__).parent / "config" / "settings.yaml"
LOCATIONS_PATH = Path(__file__).parent / "config" / "locations.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def _load_locations() -> list[Location]:
    with open(LOCATIONS_PATH) as f:
        raw = yaml.safe_load(f)
    return [Location(**loc) for loc in raw["locations"]]


class WeatherBot:
    """Main bot orchestrator."""

    LOOP_INTERVAL       = 5 * 60     # 5 minutes
    MODEL_REFRESH_HOURS = {0, 6, 12, 18}
    DAILY_SUMMARY_HOUR  = 23         # UTC hour for daily report

    def __init__(
        self,
        config: dict,
        locations: list[Location],
        paper_trade: bool = True,
    ):
        self._cfg       = config
        self._locations = locations
        self._paper     = paper_trade

        self._fetcher   = WeatherFetcher(cache_ttl=config.get("cache_ttl", 300))
        self._prob_eng  = ProbabilityEngine(
            calibration_dir=config.get("calibration_dir", "data/calibration"),
        )
        self._detector  = EdgeDetector(
            min_edge=config.get("min_edge", 0.10),
            min_kelly=config.get("min_kelly", 0.02),
            min_volume=config.get("min_volume", 500.0),
            max_spread=config.get("max_spread", 0.05),
            min_model_agreement=config.get("min_model_agreement", 0.60),
        )
        self._risk      = RiskManager(
            state_path=config.get("risk_state_path", "data/risk_state.json"),
        )
        self._risk.MAX_BET_SIZE       = config.get("max_bet_size", 3.0)
        self._risk.MAX_DAILY_EXPOSURE = config.get("max_daily_exposure", 50.0)
        self._risk.STOP_LOSS_DAILY    = config.get("stop_loss_daily", 20.0)

        poly_cfg = config.get("polymarket", {})
        self._poly = PolymarketClient(
            private_key=poly_cfg.get("private_key", os.environ.get("POLY_PRIVATE_KEY", "")),
            chain_id=poly_cfg.get("chain_id", 137),
            api_key=poly_cfg.get("api_key", os.environ.get("POLY_API_KEY", "")),
            api_secret=poly_cfg.get("api_secret", os.environ.get("POLY_API_SECRET", "")),
            api_passphrase=poly_cfg.get("api_passphrase", os.environ.get("POLY_API_PASS", "")),
        )

        self._executor = TradeExecutor(
            client=self._poly,
            paper_trade=paper_trade,
            trade_log=config.get("trade_log", "data/trades/trades.jsonl"),
        )

        tg_cfg = config.get("telegram", {})
        self._tg: Optional[TelegramReporter] = None
        if tg_cfg.get("token") and tg_cfg.get("chat_id"):
            self._tg = TelegramReporter(
                token=tg_cfg["token"],
                chat_id=tg_cfg["chat_id"],
                risk_manager=self._risk,
                on_pause=lambda: None,
                on_resume=lambda: None,
            )

        self._running = False
        self._last_model_hour: Optional[int] = None
        self._last_summary_date: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    async def run(self):
        logger.info("WeatherBot starting (paper=%s)", self._paper)
        self._running = True

        if self._tg:
            await self._tg.start()
            await self._tg.send_startup()

        # Start position monitor as background task
        asyncio.create_task(self._executor.monitor_fills(interval=15))

        try:
            while self._running:
                now = datetime.utcnow()
                await self._tick(now)
                await self._check_daily_summary(now)
                await asyncio.sleep(self.LOOP_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Bot loop cancelled")
        finally:
            if self._tg:
                await self._tg.stop()
            logger.info("WeatherBot stopped")

    async def _tick(self, now: datetime):
        """One iteration: fetch markets → forecast → compare → trade."""
        if self._risk.is_paused:
            logger.debug("Bot paused — skipping tick")
            return

        logger.info("── Tick %s ──", now.strftime("%H:%M UTC"))

        # 1. Fetch active weather markets from Polymarket
        try:
            markets = await self._poly.get_weather_markets()
        except Exception as exc:
            logger.error("Failed to fetch markets: %s", exc)
            if self._tg:
                await self._tg.send_error("market fetch", str(exc))
            return

        if not markets:
            logger.info("No active weather markets found")
            return

        logger.info("Processing %d weather markets", len(markets))

        # 2. Get USDC balance for sizing decisions
        usdc_balance = await self._poly.get_usdc_balance()
        open_positions = await self._poly.get_positions()

        # 3. For each market, fetch forecast and look for edges
        for market in markets:
            if not self._running:
                break
            await self._process_market(market, usdc_balance, open_positions)

    async def _process_market(self, market, usdc_balance: float, open_positions: list):
        """Full pipeline for a single market."""
        location = self._find_location(market.location_name)
        if not location:
            logger.debug("No location config for '%s'", market.location_name)
            return

        # Fetch consensus forecast
        try:
            forecast = await self._fetcher.get_consensus_forecast(
                location, target_date=market.target_date
            )
        except Exception as exc:
            logger.warning("Forecast failed for %s: %s", market.location_name, exc)
            return

        # Compute model probabilities
        probs = self._prob_eng.compute_probabilities(forecast, market.buckets)

        # Get order book spread for the most liquid bucket
        spread = 0.03  # default; ideally fetch actual
        try:
            if market.buckets:
                book = await self._poly.get_orderbook(market.buckets[0].token_id)
                spread = book.spread
        except Exception:
            pass

        # Find edges
        signals = self._detector.find_edges(
            market=market,
            model_probs=probs,
            model_agreement=forecast.model_agreement,
            spread=spread,
        )

        # Execute each signal through risk manager → executor
        for signal in signals:
            signal.position_size = self._risk.calculate_position_size(
                signal.kelly_fraction,
                bankroll=max(usdc_balance, 10.0),  # floor for safety
                edge=signal.edge,
            )

            approved, reason = self._risk.check_limits(
                signal, usdc_balance, open_positions
            )
            if not approved:
                logger.info("Trade blocked by risk manager: %s", reason)
                continue

            result = await self._executor.execute_signal(signal)

            if result.success:
                self._risk.record_trade(signal)
                usdc_balance -= signal.position_size

            if self._tg:
                await self._tg.send_trade_alert(result)

    def _find_location(self, name: str) -> Optional[Location]:
        for loc in self._locations:
            if loc.name.lower() == name.lower():
                return loc
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Model update trigger (every 6h)
    # ─────────────────────────────────────────────────────────────────────────

    async def on_model_update(self):
        """
        Called when weather models refresh (00Z / 06Z / 12Z / 18Z).
        Prime opportunity: new forecasts are out but market prices haven't moved yet.
        """
        logger.info("Model update triggered — clearing forecast cache")
        self._fetcher._cache.clear()
        # Force immediate tick
        await self._tick(datetime.utcnow())

    # ─────────────────────────────────────────────────────────────────────────
    # Daily summary
    # ─────────────────────────────────────────────────────────────────────────

    async def _check_daily_summary(self, now: datetime):
        today = date.today().isoformat()
        if (now.hour == self.DAILY_SUMMARY_HOUR and
                self._last_summary_date != today and
                self._tg):
            pnl = DailyPnL(
                date=today,
                trades_count=0,   # would read from trade log in production
                wins=0,
                losses=0,
                gross_pnl=self._risk.daily_pnl,
                fees=0.0,
                net_pnl=self._risk.daily_pnl,
                roi_pct=(self._risk.daily_pnl / max(self._risk.daily_wagered, 1) * 100),
                bankroll_end=await self._poly.get_usdc_balance(),
            )
            await self._tg.send_daily_summary(pnl)
            self._last_summary_date = today

    # ─────────────────────────────────────────────────────────────────────────
    # Graceful shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

async def _async_main(args):
    config    = _load_config()
    locations = _load_locations()

    if args.backtest:
        from .src.backtester import Backtester
        import pandas as pd

        bt = Backtester(initial_bankroll=100.0)
        # Minimal demo using synthetic data (replace with real CSV paths in production)
        logger.info("Running backtest demo with synthetic data...")
        n = 60
        dates = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]
        forecasts = pd.DataFrame({
            "date": dates,
            "location": "London",
            "ecmwf_temp": [15 + i % 5 for i in range(n)],
            "gfs_temp":   [14 + i % 5 for i in range(n)],
            "mean_temp":  [14.5 + i % 5 for i in range(n)],
            "std_temp":   [1.5] * n,
        })
        labels = ["≤13°C", "14°C", "15°C", "16°C", "17°C", "≥18°C"]
        prices = pd.DataFrame({
            "date":         [d for d in dates for _ in labels],
            "location":     "London",
            "bucket_label": labels * n,
            "price":        [0.17] * (n * len(labels)),
        })
        import random; random.seed(42)
        outcomes = pd.DataFrame({
            "date":           dates,
            "location":       "London",
            "winning_bucket": [random.choice(labels) for _ in range(n)],
        })
        result = bt.backtest(forecasts, prices, outcomes, dates[0], dates[-1])
        bt.save_results(result)
        logger.info(
            "Backtest complete: trades=%d win_rate=%.1f%% pnl=$%.2f roi=%.1f%%",
            result.total_trades, result.win_rate*100, result.total_pnl, result.roi_pct,
        )
        return

    bot = WeatherBot(config, locations, paper_trade=not args.live)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bot.stop)

    await bot.run()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Weather Bot")
    parser.add_argument("--live",      action="store_true", help="Live trading (default: paper)")
    parser.add_argument("--backtest",  action="store_true", help="Run backtest and exit")
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
