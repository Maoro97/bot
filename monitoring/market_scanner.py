"""
Market scanner — real-time via WebSocket with REST fallback.

Architecture:
- On startup: loads all active markets via REST, subscribes WebSocket to all token IDs
- On each WebSocket price update: checks arbitrage immediately (sub-second latency)
- Polling fallback: if WebSocket fails, falls back to REST polling every 10s
- Market list refreshes every 5 minutes to pick up new markets
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from client.polymarket import MarketPrices, PolymarketClient
from monitoring.websocket_feed import PriceFeed
from strategies.dutch_book import ArbOpportunity, DutchBookStrategy
from utils.logger import logger


class MarketScanner:
    def __init__(
        self,
        client: PolymarketClient,
        strategy: DutchBookStrategy,
        on_opportunity: Optional[Callable] = None,
    ) -> None:
        self._client = client
        self._strategy = strategy
        self._on_opportunity = on_opportunity  # async callback for real-time mode

        # Market cache: condition_id → market dict
        self._markets: dict[str, dict] = {}
        # Latest prices: token_id → (bid, ask)
        self._prices: dict[str, tuple[Optional[float], Optional[float]]] = {}
        # token_id → condition_id (reverse lookup)
        self._token_to_market: dict[str, str] = {}

        self._last_market_refresh: float = 0
        self._market_refresh_interval: int = 300  # 5 minutes

        self._ws_feed: Optional[PriceFeed] = None
        self._ws_active: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_realtime(self) -> None:
        """Start WebSocket feed. Runs forever — call as asyncio task."""
        await self._refresh_markets()
        self._ws_feed = PriceFeed(on_price_update=self._on_ws_price_update)
        self._ws_feed.subscribe(list(self._prices.keys()))
        self._ws_active = True
        logger.info("Starting real-time WebSocket scanner")

        # Run WebSocket + periodic market refresh concurrently
        await asyncio.gather(
            self._ws_feed.run(),
            self._periodic_market_refresh(),
        )

    async def scan(self) -> list[ArbOpportunity]:
        """REST polling mode — used as fallback or in --scan-only."""
        await self._maybe_refresh_markets()

        if not self._markets:
            logger.warning("No markets available to scan")
            return []

        logger.info("Scanning {} markets for arbitrage...", len(self._markets))

        tasks = [self._evaluate_market(m) for m in self._markets.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        opportunities: list[ArbOpportunity] = []
        for r in results:
            if isinstance(r, ArbOpportunity):
                opportunities.append(r)

        if opportunities:
            logger.info("Found {} arbitrage opportunities this scan", len(opportunities))
        else:
            logger.info("No arbitrage opportunities found this scan")

        return opportunities

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _on_ws_price_update(
        self, token_id: str, bid: Optional[float], ask: Optional[float]
    ) -> None:
        """Called instantly when Polymarket pushes a price change."""
        # Update price cache
        prev_bid, prev_ask = self._prices.get(token_id, (None, None))
        self._prices[token_id] = (
            bid if bid is not None else prev_bid,
            ask if ask is not None else prev_ask,
        )

        # Find which market this token belongs to
        condition_id = self._token_to_market.get(token_id)
        if not condition_id:
            return

        market = self._markets.get(condition_id)
        if not market:
            return

        # Check arbitrage with latest cached prices
        opp = self._evaluate_from_cache(market)
        if opp and self._on_opportunity:
            await self._on_opportunity(opp)

    def _evaluate_from_cache(self, market: dict) -> Optional[ArbOpportunity]:
        """Check arbitrage using cached prices (no API call needed)."""
        tokens = market.get("tokens", [])
        if len(tokens) != 2:
            return None

        yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), tokens[0])
        no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), tokens[1])

        yes_id = yes_token.get("token_id") or yes_token.get("tokenId", "")
        no_id = no_token.get("token_id") or no_token.get("tokenId", "")

        yes_bid, yes_ask = self._prices.get(yes_id, (None, None))
        no_bid, no_ask = self._prices.get(no_id, (None, None))

        if None in (yes_bid, yes_ask, no_bid, no_ask):
            return None

        prices = MarketPrices(
            condition_id=market.get("conditionId", ""),
            yes_token_id=yes_id,
            no_token_id=no_id,
            yes_ask=yes_ask,
            no_ask=no_ask,
            yes_bid=yes_bid,
            no_bid=no_bid,
            liquidity=float(market.get("liquidityNum") or market.get("liquidity") or 0),
        )
        return self._strategy.evaluate(prices)

    # ------------------------------------------------------------------
    # Market refresh
    # ------------------------------------------------------------------

    async def _refresh_markets(self) -> None:
        try:
            markets_list = await self._client.get_active_markets()
            self._markets = {m.get("conditionId", ""): m for m in markets_list}

            # Build price cache and reverse lookup
            for m in markets_list:
                for t in m.get("tokens", []):
                    tid = t.get("token_id") or t.get("tokenId", "")
                    if tid:
                        self._token_to_market[tid] = m.get("conditionId", "")
                        if tid not in self._prices:
                            self._prices[tid] = (None, None)

            self._last_market_refresh = time.time()
            logger.info("Markets refreshed: {} loaded", len(self._markets))

            # Update WebSocket subscriptions if active
            if self._ws_feed:
                await self._ws_feed.update_subscriptions(list(self._prices.keys()))
        except Exception as exc:
            logger.error("Market refresh failed: {}", exc)

    async def _maybe_refresh_markets(self) -> None:
        if time.time() - self._last_market_refresh > self._market_refresh_interval:
            await self._refresh_markets()

    async def _periodic_market_refresh(self) -> None:
        """Background task that refreshes markets every 5 minutes."""
        while True:
            await asyncio.sleep(self._market_refresh_interval)
            await self._refresh_markets()

    async def _evaluate_market(self, market: dict) -> Optional[ArbOpportunity]:
        prices = await self._client.get_market_prices(market)
        if prices is None:
            return None
        return self._strategy.evaluate(prices)
