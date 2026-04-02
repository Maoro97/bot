"""
Executes arbitrage opportunities by placing paired orders on Polymarket.

For BUY_BOTH arb:
  1. Place limit buy order for YES at yes_ask
  2. Place limit buy order for NO at no_ask
  If leg-1 fails, skip leg-2. If leg-2 fails, cancel leg-1.

All order placement is skipped in DRY_RUN mode — only logs are produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from py_clob_client.order_builder.constants import BUY, SELL

from client.polymarket import PolymarketClient
from config import config
from strategies.dutch_book import ArbDirection, ArbOpportunity
from utils.logger import logger


@dataclass
class TradeResult:
    opportunity: ArbOpportunity
    yes_order_id: Optional[str] = None
    no_order_id: Optional[str] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class Stats:
    scans: int = 0
    opportunities_found: int = 0
    trades_attempted: int = 0
    trades_succeeded: int = 0
    estimated_profit_usdc: float = 0.0
    errors: list[str] = field(default_factory=list)


class OrderManager:
    def __init__(self, client: PolymarketClient) -> None:
        self._client = client
        self.stats = Stats()

    def execute(self, opp: ArbOpportunity) -> TradeResult:
        """Execute a two-legged arbitrage trade."""
        self.stats.opportunities_found += 1
        result = TradeResult(opportunity=opp)

        if config.DRY_RUN:
            logger.info(
                "[DRY RUN] Would trade {} | direction={} | YES={:.4f} NO={:.4f} | "
                "net_profit={:.4f} USDC/unit | size={:.2f} USDC",
                opp.condition_id,
                opp.direction,
                opp.yes_price,
                opp.no_price,
                opp.net_profit,
                opp.max_size_usdc,
            )
            result.success = True
            self.stats.trades_succeeded += 1
            self.stats.estimated_profit_usdc += opp.net_profit * opp.max_size_usdc
            return result

        # --- Live trading ---
        balance = self._client.get_usdc_balance()
        required = opp.max_size_usdc * 2  # rough estimate for both legs
        if balance < required:
            msg = f"Insufficient balance {balance:.2f} < {required:.2f} USDC"
            logger.warning(msg)
            result.error = msg
            return result

        side_yes = BUY if opp.direction == ArbDirection.BUY_BOTH else SELL
        side_no = BUY if opp.direction == ArbDirection.BUY_BOTH else SELL
        size = opp.max_size_usdc / (opp.yes_price + opp.no_price)  # number of tokens per leg

        self.stats.trades_attempted += 1

        # Leg 1: YES
        try:
            resp_yes = self._client.place_order(
                token_id=opp.yes_token_id,
                side=side_yes,
                size=size,
                price=opp.yes_price,
            )
            result.yes_order_id = resp_yes.get("orderID") or resp_yes.get("id")
            logger.info("YES leg placed: order_id={}", result.yes_order_id)
        except Exception as exc:
            msg = f"YES leg failed: {exc}"
            logger.error(msg)
            result.error = msg
            self.stats.errors.append(msg)
            return result

        # Leg 2: NO
        try:
            resp_no = self._client.place_order(
                token_id=opp.no_token_id,
                side=side_no,
                size=size,
                price=opp.no_price,
            )
            result.no_order_id = resp_no.get("orderID") or resp_no.get("id")
            logger.info("NO leg placed: order_id={}", result.no_order_id)
        except Exception as exc:
            msg = f"NO leg failed: {exc}"
            logger.error(msg)
            result.error = msg
            self.stats.errors.append(msg)
            # Cancel the YES leg to avoid being left one-sided
            if result.yes_order_id:
                try:
                    self._client.cancel_order(result.yes_order_id)
                    logger.info("YES leg cancelled after NO leg failure")
                except Exception as cancel_exc:
                    logger.error("Could not cancel YES leg: {}", cancel_exc)
            return result

        result.success = True
        self.stats.trades_succeeded += 1
        self.stats.estimated_profit_usdc += opp.net_profit * size
        logger.info(
            "Arb trade complete on {} | estimated_profit={:.4f} USDC",
            opp.condition_id,
            opp.net_profit * size,
        )
        return result

    def log_stats(self) -> None:
        logger.info(
            "Stats | scans={} opps={} trades={}/{} est_profit={:.4f} USDC",
            self.stats.scans,
            self.stats.opportunities_found,
            self.stats.trades_succeeded,
            self.stats.trades_attempted,
            self.stats.estimated_profit_usdc,
        )
