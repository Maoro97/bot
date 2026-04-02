"""
Dutch-book arbitrage strategy for Polymarket binary markets.

A binary market resolves to exactly $1.00 — either YES wins or NO wins.
Therefore:
  - Buying both YES and NO for a total cost < $1.00 guarantees profit.
  - Selling both YES and NO for a total revenue > $1.00 also guarantees profit.

Polymarket charges ~2% fee on winnings, so the real break-even is $0.98 per $1.00 payout.

Buy-both opportunity:  YES_ask + NO_ask < 1.00 - FEE
Sell-both opportunity: YES_bid + NO_bid > 1.00 + FEE  (rare, included for completeness)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from client.polymarket import MarketPrices
from config import config
from utils.logger import logger


class ArbDirection(str, Enum):
    BUY_BOTH = "BUY_BOTH"    # buy YES + buy NO
    SELL_BOTH = "SELL_BOTH"  # sell YES + sell NO


@dataclass
class ArbOpportunity:
    condition_id: str
    yes_token_id: str
    no_token_id: str
    direction: ArbDirection
    yes_price: float   # execution price for YES leg
    no_price: float    # execution price for NO leg
    gross_profit: float  # before fees
    net_profit: float    # after fees
    max_size_usdc: float  # how much to deploy


class DutchBookStrategy:
    """Detects YES+NO mispricing in binary prediction markets."""

    def __init__(self) -> None:
        self.fee = config.POLYMARKET_FEE
        self.min_profit = config.MIN_PROFIT_THRESHOLD
        self.max_size = config.MAX_ORDER_SIZE_USDC

    def evaluate(self, prices: MarketPrices) -> Optional[ArbOpportunity]:
        """Return an ArbOpportunity if one exists, else None."""
        buy_opp = self._check_buy_both(prices)
        if buy_opp:
            return buy_opp
        sell_opp = self._check_sell_both(prices)
        return sell_opp

    # ------------------------------------------------------------------

    def _check_buy_both(self, prices: MarketPrices) -> Optional[ArbOpportunity]:
        """Buy YES + buy NO when combined ask < $1 - fee."""
        total_cost = prices.yes_ask + prices.no_ask
        # After paying total_cost, one side always pays $1.00
        gross = 1.0 - total_cost
        net = gross - self.fee  # subtract Polymarket's fee on winnings

        if net < self.min_profit:
            return None

        # Size: spend at most MAX_ORDER_SIZE_USDC across both legs
        # Each token needs 1 unit for a $1 payout, so size = budget / total_cost
        size = min(self.max_size / total_cost, self.max_size / 2)

        logger.info(
            "BUY_BOTH arb on {} | YES_ask={:.4f} NO_ask={:.4f} | net_profit={:.4f} ({:.2f}%)",
            prices.condition_id,
            prices.yes_ask,
            prices.no_ask,
            net,
            net * 100,
        )
        return ArbOpportunity(
            condition_id=prices.condition_id,
            yes_token_id=prices.yes_token_id,
            no_token_id=prices.no_token_id,
            direction=ArbDirection.BUY_BOTH,
            yes_price=prices.yes_ask,
            no_price=prices.no_ask,
            gross_profit=gross,
            net_profit=net,
            max_size_usdc=size,
        )

    def _check_sell_both(self, prices: MarketPrices) -> Optional[ArbOpportunity]:
        """Sell YES + sell NO when combined bid > $1 + fee (positions required)."""
        total_revenue = prices.yes_bid + prices.no_bid
        gross = total_revenue - 1.0
        net = gross - self.fee

        if net < self.min_profit:
            return None

        size = min(self.max_size / total_revenue, self.max_size / 2)

        logger.info(
            "SELL_BOTH arb on {} | YES_bid={:.4f} NO_bid={:.4f} | net_profit={:.4f} ({:.2f}%)",
            prices.condition_id,
            prices.yes_bid,
            prices.no_bid,
            net,
            net * 100,
        )
        return ArbOpportunity(
            condition_id=prices.condition_id,
            yes_token_id=prices.yes_token_id,
            no_token_id=prices.no_token_id,
            direction=ArbDirection.SELL_BOTH,
            yes_price=prices.yes_bid,
            no_price=prices.no_bid,
            gross_profit=gross,
            net_profit=net,
            max_size_usdc=size,
        )
