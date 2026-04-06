"""
Risk Manager — enforces position limits, daily exposure caps, and stop-loss.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from .models import TradeSignal, Position

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Controls bet sizing, total exposure, and hard stop-losses.
    All state is persisted to a JSON file so it survives restarts.
    """

    MAX_BET_SIZE: float       = 3.0    # max $ per single bet
    DEFAULT_BET_SIZE: float   = 1.0
    MAX_DAILY_EXPOSURE: float = 50.0   # max total $ wagered per day
    MAX_MARKET_EXPOSURE: float= 10.0   # max $ in a single market
    MAX_CONCURRENT_POSITIONS: int = 30
    STOP_LOSS_DAILY: float    = 20.0   # halt if daily loss > $20
    MIN_BET_SIZE: float       = 0.50   # below this, skip the trade

    def __init__(self, state_path: str = "data/risk_state.json"):
        self._state_path = Path(state_path)
        self._state: dict = self._load_state()

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Could not load risk state: %s", exc)
        return {
            "date": date.today().isoformat(),
            "daily_wagered": 0.0,
            "daily_pnl": 0.0,
            "market_exposure": {},   # condition_id → $ wagered
            "open_positions": 0,
            "paused": False,
        }

    def _save_state(self):
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    def _reset_if_new_day(self):
        today = date.today().isoformat()
        if self._state.get("date") != today:
            logger.info("New day — resetting daily risk counters")
            self._state.update({
                "date": today,
                "daily_wagered": 0.0,
                "daily_pnl": 0.0,
                "market_exposure": {},
            })
            self._save_state()

    # ─────────────────────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        kelly_fraction: float,
        bankroll: float,
        edge: float,
    ) -> float:
        """
        Quarter-Kelly sizing capped at MAX_BET_SIZE.
        f_quarter = kelly * 0.25 * bankroll
        """
        quarter_kelly = kelly_fraction * 0.25 * bankroll
        # Also scale by edge confidence: higher edge → larger fraction
        size = min(quarter_kelly * (1 + edge), self.MAX_BET_SIZE)
        size = max(self.MIN_BET_SIZE, size)
        return round(size, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-trade checks
    # ─────────────────────────────────────────────────────────────────────────

    def check_limits(
        self,
        signal: TradeSignal,
        usdc_balance: float,
        open_positions: Optional[list[Position]] = None,
    ) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        Checks all risk limits before allowing a trade.
        """
        self._reset_if_new_day()

        if self._state.get("paused"):
            return False, "Bot is paused"

        size = signal.position_size

        # 1. USDC balance
        if usdc_balance < size:
            return False, f"Insufficient USDC: have ${usdc_balance:.2f}, need ${size:.2f}"

        # 2. Daily stop-loss
        if self._state["daily_pnl"] <= -self.STOP_LOSS_DAILY:
            return False, f"Daily stop-loss hit (P&L=${self._state['daily_pnl']:.2f})"

        # 3. Daily exposure cap
        if self._state["daily_wagered"] + size > self.MAX_DAILY_EXPOSURE:
            return False, (
                f"Daily exposure cap: wagered=${self._state['daily_wagered']:.2f}, "
                f"limit=${self.MAX_DAILY_EXPOSURE}"
            )

        # 4. Per-market exposure
        cid = signal.market.condition_id
        mkt_exp = self._state["market_exposure"].get(cid, 0.0)
        if mkt_exp + size > self.MAX_MARKET_EXPOSURE:
            return False, (
                f"Market exposure cap: in={mkt_exp:.2f}, limit=${self.MAX_MARKET_EXPOSURE}"
            )

        # 5. Concurrent positions
        n_open = len(open_positions) if open_positions else self._state["open_positions"]
        if n_open >= self.MAX_CONCURRENT_POSITIONS:
            return False, f"Too many open positions: {n_open}/{self.MAX_CONCURRENT_POSITIONS}"

        return True, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # State updates (called after fills)
    # ─────────────────────────────────────────────────────────────────────────

    def record_trade(self, signal: TradeSignal):
        """Update internal counters after a successful fill."""
        self._reset_if_new_day()
        size = signal.position_size
        cid  = signal.market.condition_id

        self._state["daily_wagered"] += size
        self._state["market_exposure"][cid] = (
            self._state["market_exposure"].get(cid, 0.0) + size
        )
        self._state["open_positions"] = self._state.get("open_positions", 0) + 1
        self._save_state()
        logger.info("Risk state updated: daily_wagered=%.2f, positions=%d",
                    self._state["daily_wagered"], self._state["open_positions"])

    def record_resolution(self, pnl: float, condition_id: str):
        """Update P&L and positions when a market resolves."""
        self._reset_if_new_day()
        self._state["daily_pnl"] += pnl
        self._state["open_positions"] = max(0, self._state["open_positions"] - 1)
        self._state["market_exposure"].pop(condition_id, None)
        self._save_state()
        logger.info("Resolution recorded: pnl=%.2f, daily_pnl=%.2f",
                    pnl, self._state["daily_pnl"])

    # ─────────────────────────────────────────────────────────────────────────
    # Manual controls
    # ─────────────────────────────────────────────────────────────────────────

    def pause(self):
        self._state["paused"] = True
        self._save_state()
        logger.warning("Bot PAUSED by user")

    def resume(self):
        self._state["paused"] = False
        self._save_state()
        logger.info("Bot RESUMED by user")

    def set_max_bet(self, amount: float):
        self.MAX_BET_SIZE = amount
        logger.info("MAX_BET_SIZE set to $%.2f", amount)

    # ─────────────────────────────────────────────────────────────────────────
    # Read-only accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_paused(self) -> bool:
        return bool(self._state.get("paused"))

    @property
    def daily_pnl(self) -> float:
        self._reset_if_new_day()
        return self._state["daily_pnl"]

    @property
    def daily_wagered(self) -> float:
        self._reset_if_new_day()
        return self._state["daily_wagered"]

    @property
    def open_positions_count(self) -> int:
        return self._state.get("open_positions", 0)

    def status_dict(self) -> dict:
        self._reset_if_new_day()
        return {
            "date": self._state["date"],
            "daily_wagered": round(self._state["daily_wagered"], 2),
            "daily_pnl": round(self._state["daily_pnl"], 2),
            "open_positions": self._state["open_positions"],
            "paused": self._state["paused"],
            "max_bet": self.MAX_BET_SIZE,
            "stop_loss_remaining": round(
                self.STOP_LOSS_DAILY + self._state["daily_pnl"], 2
            ),
        }
