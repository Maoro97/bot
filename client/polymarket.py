from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import aiohttp
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

from config import config
from utils.logger import logger


@dataclass
class MarketPrices:
    condition_id: str
    yes_token_id: str
    no_token_id: str
    yes_ask: float  # cheapest price to buy YES
    no_ask: float   # cheapest price to buy NO
    yes_bid: float  # highest price to sell YES
    no_bid: float   # highest price to sell NO
    liquidity: float


def _is_valid_private_key(key: str) -> bool:
    """Return True only if the key looks like a real hex private key."""
    if not key:
        return False
    k = key.lower().removeprefix("0x")
    return len(k) == 64 and all(c in "0123456789abcdef" for c in k)


class PolymarketClient:
    def __init__(self) -> None:
        has_key = _is_valid_private_key(config.PRIVATE_KEY)
        has_creds = all([config.API_KEY, config.API_SECRET, config.API_PASSPHRASE])

        creds = ApiCreds(
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            api_passphrase=config.API_PASSPHRASE,
        ) if has_creds else None

        self._client = ClobClient(
            host=config.CLOB_HOST,
            chain_id=config.CHAIN_ID,
            key=config.PRIVATE_KEY if has_key else None,
            creds=creds,
        )
        if not has_key:
            logger.warning("No valid PRIVATE_KEY — running in read-only/scan mode")
        logger.info("Polymarket CLOB client initialized (dry_run={})", config.DRY_RUN)

    # ------------------------------------------------------------------
    # Market discovery (uses CLOB API — guaranteed to have token IDs)
    # ------------------------------------------------------------------

    async def get_active_markets(self) -> list[dict]:
        """Fetch active binary markets from the CLOB API."""
        import asyncio
        loop = asyncio.get_event_loop()
        try:
            resp = await loop.run_in_executor(None, lambda: self._client.get_sampling_simplified_markets())
            raw = resp if isinstance(resp, list) else (resp.get("data") or [])
        except Exception as exc:
            logger.error("CLOB get_sampling_simplified_markets failed: {}", exc)
            # Fallback: Gamma API
            raw = await self._get_markets_gamma()

        markets = []
        for m in raw:
            tokens = m.get("tokens") or []
            if len(tokens) != 2:
                continue
            markets.append(m)

        logger.info("Market discovery: {}/{} binary markets loaded", len(markets), len(raw))
        return markets

    async def _get_markets_gamma(self) -> list[dict]:
        """Fallback: fetch markets from Gamma API."""
        url = f"{config.GAMMA_HOST}/markets"
        params = {"active": "true", "closed": "false", "limit": 500}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            raw = data if isinstance(data, list) else data.get("markets", [])
            logger.info("Gamma API fallback returned {} raw markets", len(raw))
            return raw
        except Exception as exc:
            logger.error("Gamma API fallback also failed: {}", exc)
            return []

    # ------------------------------------------------------------------
    # Price fetching (CLOB API)
    # ------------------------------------------------------------------

    async def get_market_prices(self, market: dict) -> Optional[MarketPrices]:
        """Return best ask/bid prices for both outcomes of a binary market."""
        tokens = market.get("tokens", [])
        if len(tokens) != 2:
            return None

        yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), tokens[0])
        no_token = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), tokens[1])

        yes_id = yes_token.get("token_id") or yes_token.get("tokenId", "")
        no_id = no_token.get("token_id") or no_token.get("tokenId", "")
        if not yes_id or not no_id:
            return None

        try:
            yes_book = self._client.get_order_book(yes_id)
            no_book = self._client.get_order_book(no_id)
        except Exception as exc:
            logger.debug("Failed to fetch order book for {}: {}", market.get("conditionId"), exc)
            return None

        yes_ask = self._best_ask(yes_book)
        no_ask = self._best_ask(no_book)
        yes_bid = self._best_bid(yes_book)
        no_bid = self._best_bid(no_book)

        if yes_ask is None or no_ask is None or yes_bid is None or no_bid is None:
            return None

        return MarketPrices(
            condition_id=market.get("conditionId", ""),
            yes_token_id=yes_id,
            no_token_id=no_id,
            yes_ask=yes_ask,
            no_ask=no_ask,
            yes_bid=yes_bid,
            no_bid=no_bid,
            liquidity=float(market.get("liquidityNum", 0) or 0),
        )

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_order(
        self,
        token_id: str,
        side: str,
        size: float,
        price: float,
    ) -> dict:
        """Place a limit GTC order. Returns the API response dict."""
        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=round(size, 2),
            side=side,
        )
        return self._client.create_and_post_order(order_args)

    def cancel_order(self, order_id: str) -> dict:
        return self._client.cancel(order_id)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_usdc_balance(self) -> float:
        try:
            resp = self._client.get_balance_allowance()
            return float(resp.get("balance", 0))
        except Exception as exc:
            logger.warning("Could not fetch balance: {}", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_ask(order_book) -> Optional[float]:
        """Lowest ask price (cheapest to buy)."""
        asks = getattr(order_book, "asks", []) or []
        if not asks:
            return None
        return float(min(asks, key=lambda o: float(o.price)).price)

    @staticmethod
    def _best_bid(order_book) -> Optional[float]:
        """Highest bid price (best to sell at)."""
        bids = getattr(order_book, "bids", []) or []
        if not bids:
            return None
        return float(max(bids, key=lambda o: float(o.price)).price)
