"""
Scans active Polymarket binary markets and surfaces arbitrage opportunities.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from client.polymarket import PolymarketClient
from strategies.dutch_book import ArbOpportunity, DutchBookStrategy
from utils.logger import logger


class MarketScanner:
    def __init__(self, client: PolymarketClient, strategy: DutchBookStrategy) -> None:
        self._client = client
        self._strategy = strategy
        self._markets_cache: list[dict] = []
        self._cache_rounds: int = 0
        # Refresh full market list every 30 scan cycles (~5 min at 10s intervals)
        self._cache_refresh_every: int = 30

    async def scan(self) -> list[ArbOpportunity]:
        """Fetch prices for all active markets and return detected opportunities."""
        await self._maybe_refresh_markets()

        if not self._markets_cache:
            logger.warning("No markets available to scan")
            return []

        logger.debug("Scanning {} markets for arbitrage...", len(self._markets_cache))

        tasks = [self._evaluate_market(m) for m in self._markets_cache]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        opportunities: list[ArbOpportunity] = []
        for r in results:
            if isinstance(r, ArbOpportunity):
                opportunities.append(r)
            elif isinstance(r, Exception):
                logger.debug("Market eval error: {}", r)

        if opportunities:
            logger.info("Found {} arbitrage opportunities this scan", len(opportunities))
        else:
            logger.debug("No arbitrage opportunities found this scan")

        return opportunities

    async def _evaluate_market(self, market: dict) -> Optional[ArbOpportunity]:
        prices = await self._client.get_market_prices(market)
        if prices is None:
            return None
        return self._strategy.evaluate(prices)

    async def _maybe_refresh_markets(self) -> None:
        if self._cache_rounds % self._cache_refresh_every == 0:
            try:
                self._markets_cache = await self._client.get_active_markets()
                logger.info("Market list refreshed: {} markets", len(self._markets_cache))
            except Exception as exc:
                logger.error("Failed to refresh market list: {}", exc)
        self._cache_rounds += 1
