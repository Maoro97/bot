"""
Telegram notification service for the arbitrage bot.
Sends alerts when opportunities are detected or trades are executed.

Setup:
1. Message @BotFather on Telegram → /newbot → get your BOT_TOKEN
2. Message your bot, then visit: https://api.telegram.org/bot<TOKEN>/getUpdates
3. Find your chat_id in the response
4. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from utils.logger import logger

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


class TelegramNotifier:
    def __init__(self) -> None:
        self.enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        self._bot = None
        if self.enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=TELEGRAM_BOT_TOKEN)
                logger.info("Telegram notifications enabled")
            except Exception as exc:
                logger.warning("Telegram setup failed: {}", exc)
                self.enabled = False
        else:
            logger.info("Telegram notifications disabled (no token/chat_id)")

    async def send(self, message: str) -> None:
        if not self.enabled or not self._bot:
            return
        try:
            await self._bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.warning("Telegram send failed: {}", exc)

    async def notify_opportunity(self, opp) -> None:
        msg = (
            f"<b>Arbitrage Found!</b>\n"
            f"Market: <code>{opp.condition_id[:16]}...</code>\n"
            f"Direction: {opp.direction.value}\n"
            f"YES: ${opp.yes_price:.4f} | NO: ${opp.no_price:.4f}\n"
            f"Net Profit: <b>${opp.net_profit:.4f}</b> ({opp.net_profit*100:.2f}%)\n"
            f"Size: ${opp.max_size_usdc:.2f} USDC"
        )
        await self.send(msg)

    async def notify_trade(self, result) -> None:
        opp = result.opportunity
        if result.success:
            msg = (
                f"<b>Trade Executed!</b>\n"
                f"Market: <code>{opp.condition_id[:16]}...</code>\n"
                f"YES order: <code>{result.yes_order_id or 'DRY RUN'}</code>\n"
                f"NO order: <code>{result.no_order_id or 'DRY RUN'}</code>\n"
                f"Est. Profit: <b>${opp.net_profit * opp.max_size_usdc:.4f} USDC</b>"
            )
        else:
            msg = (
                f"<b>Trade Failed</b>\n"
                f"Market: <code>{opp.condition_id[:16]}...</code>\n"
                f"Error: {result.error}"
            )
        await self.send(msg)

    async def notify_stats(self, stats) -> None:
        msg = (
            f"<b>Bot Status</b>\n"
            f"Scans: {stats.scans}\n"
            f"Opportunities: {stats.opportunities_found}\n"
            f"Trades: {stats.trades_succeeded}/{stats.trades_attempted}\n"
            f"Est. Profit: <b>${stats.estimated_profit_usdc:.4f} USDC</b>"
        )
        await self.send(msg)

    async def notify_startup(self, dry_run: bool) -> None:
        mode = "DRY RUN" if dry_run else "LIVE"
        await self.send(f"<b>Bot Started</b> [{mode}]")
