"""
Trade Executor — submits limit orders to Polymarket and monitors fills.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    Order, OrderBook, Position, TradeResult, TradeSignal
)
from .polymarket_client import PolymarketClient

logger = logging.getLogger(__name__)

FILL_TIMEOUT    = 60     # seconds to wait for a fill before cancelling
PRICE_IMPROVE   = 0.002  # improve limit price by 0.2% to increase fill probability
TRADE_LOG_PATH  = Path("data/trades/trades.jsonl")


class TradeExecutor:
    """
    Executes TradeSignals on Polymarket and persists results.
    In paper-trade mode all operations are simulated and no real orders are sent.
    """

    def __init__(
        self,
        client: PolymarketClient,
        paper_trade: bool = True,
        trade_log: str = str(TRADE_LOG_PATH),
    ):
        self._client = client
        self.paper_trade = paper_trade
        self._log_path = Path(trade_log)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Main execute path
    # ─────────────────────────────────────────────────────────────────────────

    async def execute_signal(self, signal: TradeSignal) -> TradeResult:
        """
        Full lifecycle:
        1. Validate order book liquidity
        2. Place limit order at improved price
        3. Wait up to FILL_TIMEOUT for fill
        4. Cancel if unfilled
        5. Log result
        """
        token_id = signal.bucket.token_id
        size     = signal.position_size

        # 1. Check order book depth
        try:
            book = await self._client.get_orderbook(token_id)
        except Exception as exc:
            return TradeResult(signal=signal, order=None, success=False,
                               error=f"Failed to fetch order book: {exc}")

        ok, reason = self._check_liquidity(book, signal)
        if not ok:
            logger.info("Liquidity check failed for %s %s: %s",
                        signal.market.location_name, signal.bucket.label, reason)
            return TradeResult(signal=signal, order=None, success=False, error=reason)

        # 2. Determine limit price
        limit_price = self._compute_limit_price(signal, book)

        if self.paper_trade:
            return await self._paper_execute(signal, limit_price, size)

        # 3. Place real order
        try:
            order = await self._client.place_limit_order(
                token_id=token_id,
                side=signal.side,
                price=limit_price,
                size=size,
            )
            logger.info(
                "Order placed: %s %s %s @ %.4f x %.2f (id=%s)",
                signal.side, signal.market.location_name,
                signal.bucket.label, limit_price, size, order.order_id,
            )
        except Exception as exc:
            logger.error("Order placement failed: %s", exc)
            return TradeResult(signal=signal, order=None, success=False,
                               error=str(exc))

        # 4. Wait for fill
        filled_order = await self._wait_for_fill(order)

        # 5. Cancel if not filled
        if filled_order.status not in ("MATCHED", "FILLED"):
            logger.info("Order %s not filled after %ds - cancelling", order.order_id, FILL_TIMEOUT)
            await self._client.cancel_order(order.order_id)
            filled_order.status = "CANCELLED"
            result = TradeResult(signal=signal, order=filled_order, success=False,
                                 error="Order expired unfilled")
        else:
            result = TradeResult(signal=signal, order=filled_order, success=True)

        self._log_trade(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_liquidity(book: OrderBook, signal: TradeSignal) -> tuple[bool, str]:
        """Verify the order book has enough depth for our trade."""
        min_depth = signal.position_size * 2   # want at least 2× our size available
        if book.depth_5pct < min_depth:
            return False, (
                f"Insufficient depth: ${book.depth_5pct:.2f} < ${min_depth:.2f}"
            )
        if book.spread > 0.06:
            return False, f"Spread too wide: {book.spread:.3f}"
        return True, "OK"

    @staticmethod
    def _compute_limit_price(signal: TradeSignal, book: OrderBook) -> float:
        """
        For BUY: place slightly above best ask to improve fill probability.
        Cap at our model probability to avoid over-paying.
        """
        if signal.side == "BUY":
            price = min(book.best_ask + PRICE_IMPROVE, signal.model_prob)
        else:
            price = max(book.best_bid - PRICE_IMPROVE, signal.model_prob)

        return round(min(0.99, max(0.01, price)), 4)

    async def _wait_for_fill(self, order: Order) -> Order:
        """Poll order status until filled or timeout."""
        deadline = asyncio.get_event_loop().time() + FILL_TIMEOUT
        while asyncio.get_event_loop().time() < deadline:
            try:
                updated = await self._client.get_order(order.order_id)
                if updated.status in ("MATCHED", "FILLED"):
                    logger.info("Order %s filled!", order.order_id)
                    return updated
            except Exception as exc:
                logger.debug("Poll error: %s", exc)
            await asyncio.sleep(5)
        return order  # status still OPEN → will be cancelled

    async def _paper_execute(
        self, signal: TradeSignal, price: float, size: float
    ) -> TradeResult:
        """Simulate a fill without touching Polymarket."""
        import uuid
        fake_id = f"PAPER-{uuid.uuid4().hex[:8]}"
        order = Order(
            order_id=fake_id,
            token_id=signal.bucket.token_id,
            side=signal.side,
            price=price,
            size=size,
            status="FILLED",
            filled_at=datetime.utcnow(),
        )
        logger.info(
            "[PAPER] %s %s %s @ %.4f x %.2f",
            signal.side, signal.market.location_name,
            signal.bucket.label, price, size,
        )
        result = TradeResult(signal=signal, order=order, success=True)
        self._log_trade(result)
        return result

    def _log_trade(self, result: TradeResult):
        """Append trade result to JSONL log."""
        record = {
            "ts": datetime.utcnow().isoformat(),
            "success": result.success,
            "error": result.error,
            "location": result.signal.market.location_name,
            "bucket": result.signal.bucket.label,
            "side": result.signal.side,
            "model_prob": result.signal.model_prob,
            "market_price": result.signal.market_price,
            "edge": result.signal.edge,
            "kelly": result.signal.kelly_fraction,
            "ev": result.signal.expected_value,
            "size": result.signal.position_size,
            "order_id": result.order.order_id if result.order else None,
            "fill_price": result.order.price if result.order else None,
            "paper": self.paper_trade,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Could not write trade log: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Position monitoring (WebSocket-style polling fallback)
    # ─────────────────────────────────────────────────────────────────────────

    async def monitor_fills(self, interval: float = 10.0):
        """
        Continuously polls open positions and logs P&L changes.
        Skipped in paper trading mode (no real positions exist).
        """
        if self.paper_trade:
            logger.info("Paper mode - position monitor disabled")
            return
        logger.info("Starting position monitor (interval=%ds)", interval)
        while True:
            try:
                positions = await self._client.get_positions()
                total_upnl = sum(p.unrealized_pnl for p in positions)
                logger.debug("Open positions: %d | Unrealized P&L: $%.2f",
                             len(positions), total_upnl)
            except Exception as exc:
                logger.warning("Position monitor error: %s", exc)
            await asyncio.sleep(interval)
