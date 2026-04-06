"""
Telegram Reporter — sends trade alerts, daily summaries, and responds
to user commands for monitoring and controlling the weather bot.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from telegram import Update  # type: ignore
from telegram.ext import (  # type: ignore
    Application, CommandHandler, ContextTypes,
)

from .models import TradeResult, DailyPnL, TradeSignal

if TYPE_CHECKING:
    from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class TelegramReporter:
    """
    Manages outbound alerts and inbound commands over Telegram.
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        risk_manager: "RiskManager",
        on_pause: Optional[Callable] = None,
        on_resume: Optional[Callable] = None,
        on_set_max_bet: Optional[Callable[[float], None]] = None,
    ):
        self._token        = token
        self._chat_id      = chat_id
        self._risk         = risk_manager
        self._on_pause     = on_pause
        self._on_resume    = on_resume
        self._on_set_max_bet = on_set_max_bet

        self._app = Application.builder().token(token).build()
        self._register_handlers()

    # ─────────────────────────────────────────────────────────────────────────
    # Bot startup / shutdown
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self):
        """Initialise and start polling in the background."""
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def stop(self):
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    # ─────────────────────────────────────────────────────────────────────────
    # Outbound messages
    # ─────────────────────────────────────────────────────────────────────────

    async def send(self, text: str):
        """Send a plain text message (with HTML parse mode)."""
        try:
            await self._app.bot.send_message(
                chat_id=self._chat_id,
                text=text,
                parse_mode="HTML",
            )
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)

    async def send_trade_alert(self, result: TradeResult):
        """Alert on every executed trade."""
        s = result.signal
        if result.success:
            icon = "✅" if s.side == "BUY" else "📉"
            mode = "[PAPER]" if True else ""   # paper flag passed via executor
            msg = (
                f"{icon} <b>{s.side} {s.market.location_name} — {s.bucket.label}</b> {mode}\n"
                f"🌡 Market: <b>{s.market_price:.2%}</b>  |  Model: <b>{s.model_prob:.2%}</b>  |  "
                f"Edge: <b>{s.edge*100:+.1f}%</b>\n"
                f"💵 Size: ${s.position_size:.2f}  |  EV: {s.expected_value*100:+.2f}¢ per $\n"
                f"📋 Kelly: {s.kelly_fraction*100:.1f}%  |  Confidence: {s.confidence:.0%}\n"
                f"🕐 {datetime.utcnow().strftime('%H:%M UTC')}"
            )
        else:
            msg = (
                f"⚠️ <b>Trade skipped:</b> {s.market.location_name} {s.bucket.label}\n"
                f"Reason: {result.error}"
            )
        await self.send(msg)

    async def send_daily_summary(self, pnl: DailyPnL):
        """Send end-of-day summary."""
        win_rate = pnl.wins / pnl.trades_count * 100 if pnl.trades_count else 0
        sign = "+" if pnl.net_pnl >= 0 else ""
        msg = (
            f"📊 <b>Daily Summary — {pnl.date}</b>\n\n"
            f"Trades: {pnl.trades_count}  |  Won: {pnl.wins}  |  Lost: {pnl.losses}  "
            f"|  Win Rate: {win_rate:.0f}%\n"
            f"Gross P&L: <b>{sign}${pnl.gross_pnl:.2f}</b>\n"
            f"Fees: -${pnl.fees:.2f}\n"
            f"Net P&L: <b>{sign}${pnl.net_pnl:.2f}</b>  (ROI: {pnl.roi_pct:+.1f}%)\n"
            f"Bankroll: <b>${pnl.bankroll_end:.2f}</b>"
        )
        await self.send(msg)

    async def send_startup(self):
        await self.send(
            "🤖 <b>Weather Bot started</b>\n"
            "Paper-trading mode active. Use /status to check bot state."
        )

    async def send_error(self, context: str, error: str):
        await self.send(f"❌ <b>Error in {context}</b>\n<code>{error}</code>")

    # ─────────────────────────────────────────────────────────────────────────
    # Inbound command handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _register_handlers(self):
        self._app.add_handler(CommandHandler("status",     self._cmd_status))
        self._app.add_handler(CommandHandler("positions",  self._cmd_positions))
        self._app.add_handler(CommandHandler("pnl",        self._cmd_pnl))
        self._app.add_handler(CommandHandler("pause",      self._cmd_pause))
        self._app.add_handler(CommandHandler("resume",     self._cmd_resume))
        self._app.add_handler(CommandHandler("set_max_bet",self._cmd_set_max_bet))
        self._app.add_handler(CommandHandler("markets",    self._cmd_markets))
        self._app.add_handler(CommandHandler("help",       self._cmd_help))

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        s = self._risk.status_dict()
        paused_str = "⏸ PAUSED" if s["paused"] else "▶️ RUNNING"
        msg = (
            f"<b>Bot Status</b>  {paused_str}\n"
            f"Date: {s['date']}\n"
            f"Daily wagered: ${s['daily_wagered']:.2f}\n"
            f"Daily P&L: ${s['daily_pnl']:+.2f}\n"
            f"Open positions: {s['open_positions']}\n"
            f"Max bet: ${s['max_bet']:.2f}\n"
            f"Stop-loss remaining: ${s['stop_loss_remaining']:.2f}"
        )
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_positions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        # This would need a reference to the executor / client; placeholder
        await update.message.reply_text("📋 <i>Fetching positions…</i>", parse_mode="HTML")

    async def _cmd_pnl(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        pnl = self._risk.daily_pnl
        wagered = self._risk.daily_wagered
        roi = (pnl / wagered * 100) if wagered else 0
        await update.message.reply_text(
            f"💰 <b>P&L</b>\nToday: <b>${pnl:+.2f}</b>  (ROI: {roi:+.1f}%)\n"
            f"Wagered today: ${wagered:.2f}",
            parse_mode="HTML",
        )

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._risk.pause()
        if self._on_pause:
            self._on_pause()
        await update.message.reply_text("⏸ Bot paused.")

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        self._risk.resume()
        if self._on_resume:
            self._on_resume()
        await update.message.reply_text("▶️ Bot resumed.")

    async def _cmd_set_max_bet(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        try:
            amount = float(ctx.args[0])  # type: ignore[index]
            self._risk.set_max_bet(amount)
            if self._on_set_max_bet:
                self._on_set_max_bet(amount)
            await update.message.reply_text(f"✅ Max bet set to ${amount:.2f}")
        except (IndexError, ValueError):
            await update.message.reply_text("Usage: /set_max_bet <amount>")

    async def _cmd_markets(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🌡 <i>Fetching active weather markets…</i>", parse_mode="HTML"
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        msg = (
            "<b>Available commands:</b>\n"
            "/status — current bot state\n"
            "/positions — open positions\n"
            "/pnl — today's P&amp;L\n"
            "/pause — pause trading\n"
            "/resume — resume trading\n"
            "/set_max_bet &lt;$&gt; — change max bet size\n"
            "/markets — active weather markets with edges\n"
            "/help — this message"
        )
        await update.message.reply_text(msg, parse_mode="HTML")
