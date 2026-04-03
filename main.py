#!/usr/bin/env python3
"""
Polymarket Arbitrage Bot
========================
Scans active binary prediction markets on Polymarket and executes
Dutch-book arbitrage (buy YES + NO when combined price < $1 - fee).

Usage:
    python main.py            # Run bot (respects DRY_RUN env var)
    python main.py --scan-only  # Scan once and print opportunities, no orders

Dashboard: http://localhost:5000 (auto-starts with bot)

Requirements:
    Copy .env.example to .env and fill in your credentials.
    pip install -r requirements.txt
"""

import argparse
import asyncio
import os
import signal
import sys

from client.polymarket import PolymarketClient
from config import config
from dashboard.app import (
    add_error,
    add_opportunity,
    add_trade,
    set_meta,
    start_dashboard,
    update_stats,
)
from execution.order_manager import OrderManager
from monitoring.market_scanner import MarketScanner
from notifications.telegram_bot import TelegramNotifier
from strategies.dutch_book import DutchBookStrategy
from utils.logger import logger

_running = True


def _handle_signal(sig, frame):
    global _running
    logger.info("Shutdown signal received, stopping...")
    _running = False


async def run_bot(scan_only: bool = False) -> None:
    global _running

    logger.info("=" * 60)
    logger.info("Polymarket Arbitrage Bot starting")
    logger.info("DRY_RUN={} | MIN_PROFIT={:.1f}% | MAX_SIZE={} USDC",
                config.DRY_RUN, config.MIN_PROFIT_THRESHOLD * 100, config.MAX_ORDER_SIZE_USDC)
    logger.info("=" * 60)

    try:
        config.validate()
    except ValueError as e:
        logger.error("Configuration error: {}", e)
        sys.exit(1)

    # Start web dashboard
    dashboard_port = int(os.getenv("PORT", "5000"))
    start_dashboard(port=dashboard_port)

    # Telegram notifications
    telegram = TelegramNotifier()
    await telegram.notify_startup(config.DRY_RUN)

    client = PolymarketClient()
    strategy = DutchBookStrategy()
    scanner = MarketScanner(client, strategy)
    order_manager = OrderManager(client)

    if not config.DRY_RUN:
        balance = client.get_usdc_balance()
        logger.info("USDC balance: {:.2f}", balance)
        if balance < 1.0:
            logger.error("Balance too low to trade. Deposit USDC on Polygon first.")
            sys.exit(1)

    scan_count = 0

    while _running:
        scan_count += 1
        order_manager.stats.scans = scan_count

        try:
            opportunities = await scanner.scan()
            set_meta(config.DRY_RUN, len(scanner._markets_cache))
            update_stats(order_manager.stats)

            for opp in opportunities:
                if not _running:
                    break
                add_opportunity(opp)
                await telegram.notify_opportunity(opp)

                result = order_manager.execute(opp)
                add_trade(result)
                await telegram.notify_trade(result)

            update_stats(order_manager.stats)

            if scan_count % 10 == 0:
                order_manager.log_stats()

            # Send Telegram stats every 100 scans (~16 min)
            if scan_count % 100 == 0:
                await telegram.notify_stats(order_manager.stats)

        except Exception as exc:
            logger.error("Error during scan #{}: {}", scan_count, exc)
            add_error(str(exc))

        if scan_only:
            logger.info("--scan-only mode: exiting after one scan")
            break

        await asyncio.sleep(config.SCAN_INTERVAL_SECONDS)

    order_manager.log_stats()
    await telegram.notify_stats(order_manager.stats)
    logger.info("Bot stopped after {} scans", scan_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Scan once for opportunities and exit (no orders placed)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    asyncio.run(run_bot(scan_only=args.scan_only))


if __name__ == "__main__":
    main()
