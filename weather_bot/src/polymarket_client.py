"""
Polymarket CLOB client wrapper.
Uses py-clob-client under the hood; adds weather-market parsing,
retry logic, and order-book depth checks.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime
from typing import Optional

from .models import Market, OrderBook, Order, Position, TemperatureBucket

logger = logging.getLogger(__name__)

# Temperature bucket regex patterns
_RANGE_RE   = re.compile(r"([-\d]+)\s*[°℃C]\s*[-–]\s*([-\d]+)\s*[°℃C]")  # "18°C - 20°C"
_ABOVE_RE   = re.compile(r"(?:≥|>=|above|over)\s*([-\d]+)\s*[°℃C]", re.I)
_BELOW_RE   = re.compile(r"(?:≤|<=|below|under)\s*([-\d]+)\s*[°℃C]", re.I)
_EXACT_RE   = re.compile(r"^([-\d]+)\s*[°℃C]$")

WEATHER_KEYWORDS = {"temperature", "temp", "°c", "celsius", "high", "low", "weather"}


def _parse_bucket(label: str, token_id: str) -> TemperatureBucket:
    """Convert an outcome label like '≤15°C', '18°C', '20°C-22°C' into a TemperatureBucket."""
    low, high = -math.inf, math.inf

    m = _RANGE_RE.search(label)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return TemperatureBucket(label=label, low=low, high=high, token_id=token_id)

    m = _ABOVE_RE.search(label)
    if m:
        low = float(m.group(1))
        return TemperatureBucket(label=label, low=low, high=math.inf, token_id=token_id)

    m = _BELOW_RE.search(label)
    if m:
        high = float(m.group(1))
        return TemperatureBucket(label=label, low=-math.inf, high=high, token_id=token_id)

    m = _EXACT_RE.match(label.strip())
    if m:
        mid = float(m.group(1))
        return TemperatureBucket(label=label, low=mid - 0.5, high=mid + 0.5, token_id=token_id)

    # Fallback: treat as ±inf (catches unexpected formats gracefully)
    logger.warning("Could not parse bucket label '%s' — treating as (-∞, +∞)", label)
    return TemperatureBucket(label=label, low=-math.inf, high=math.inf, token_id=token_id)


def _is_weather_market(question: str, tags: list[str]) -> bool:
    q_lower = question.lower()
    tags_lower = {t.lower() for t in tags}
    return (
        any(kw in q_lower for kw in WEATHER_KEYWORDS)
        or bool(tags_lower & {"weather", "temperature", "climate"})
    )


def _extract_location(question: str, known_locations: list[str]) -> Optional[str]:
    for loc in known_locations:
        if loc.lower() in question.lower():
            return loc
    return None


def _extract_date(question: str) -> Optional[str]:
    """Try to find a date string like 'April 7' or '2024-04-07' in the question."""
    # ISO format
    iso = re.search(r"\d{4}-\d{2}-\d{2}", question)
    if iso:
        return iso.group(0)
    # "Month Day" — approximate: return None and let caller infer from end_date
    return None


class PolymarketClient:
    """
    Thin async wrapper around py-clob-client.
    All blocking SDK calls are run in a thread-pool executor so the async
    event loop is never blocked.
    """

    KNOWN_LOCATIONS = [
        "London", "NYC", "New York", "Hong Kong", "Seoul", "Tokyo",
        "Paris", "Berlin", "Sydney", "Singapore", "Dubai", "Chicago",
    ]

    def __init__(self, private_key: str, chain_id: int = 137,
                 api_key: str = "", api_secret: str = "", api_passphrase: str = ""):
        from py_clob_client.client import ClobClient  # type: ignore
        from py_clob_client.clob_types import ApiCreds  # type: ignore

        host = "https://clob.polymarket.com"
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        ) if api_key else None

        self._client = ClobClient(
            host=host,
            key=private_key,
            chain_id=chain_id,
            creds=creds,
        )
        self._loop = asyncio.get_event_loop()

    async def _run(self, fn, *args, **kwargs):
        """Run a blocking SDK call in a thread executor."""
        return await self._loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ── market discovery ──────────────────────────────────────────────────────

    async def get_weather_markets(self) -> list[Market]:
        """Fetch all active weather/temperature markets from Polymarket."""
        markets: list[Market] = []
        next_cursor = ""

        while True:
            resp = await self._run(
                self._client.get_markets,
                next_cursor=next_cursor,
            )
            if not resp:
                break

            for raw in resp.get("data", []):
                try:
                    market = self._parse_market(raw)
                    if market:
                        markets.append(market)
                except Exception as exc:
                    logger.debug("Skipping market %s: %s", raw.get("condition_id"), exc)

            next_cursor = resp.get("next_cursor", "")
            if not next_cursor or next_cursor == "LTE=":
                break

        logger.info("Found %d active weather markets", len(markets))
        return markets

    def _parse_market(self, raw: dict) -> Optional[Market]:
        """Convert a raw API market dict into our Market dataclass."""
        question: str = raw.get("question", "")
        tags: list[str] = raw.get("tags", []) or []

        if not _is_weather_market(question, tags):
            return None

        condition_id = raw.get("condition_id", "")
        end_date_str = raw.get("end_date_iso", raw.get("end_date", ""))
        try:
            end_date = datetime.fromisoformat(end_date_str.rstrip("Z"))
        except Exception:
            end_date = datetime.utcnow()

        if end_date < datetime.utcnow():
            return None  # already resolved

        tokens: list[dict] = raw.get("tokens", []) or []
        buckets: list[TemperatureBucket] = []
        prices: dict[str, float] = {}

        for token in tokens:
            label    = token.get("outcome", "")
            token_id = token.get("token_id", "")
            price    = float(token.get("price", 0.5))
            bucket   = _parse_bucket(label, token_id)
            buckets.append(bucket)
            prices[label] = price

        if not buckets:
            return None

        location_name = _extract_location(question, self.KNOWN_LOCATIONS) or "Unknown"
        target_date   = _extract_date(question) or end_date.date().isoformat()
        volume        = float(raw.get("volume", 0))

        return Market(
            condition_id=condition_id,
            question=question,
            location_name=location_name,
            target_date=target_date,
            buckets=buckets,
            prices=prices,
            volume=volume,
            end_date=end_date,
        )

    # ── order book ────────────────────────────────────────────────────────────

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch the current order book for a single outcome token."""
        raw = await self._run(self._client.get_order_book, token_id)

        bids = [(float(b["price"]), float(b["size"])) for b in raw.get("bids", [])]
        asks = [(float(a["price"]), float(a["size"])) for a in raw.get("asks", [])]

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 1.0
        spread   = best_ask - best_bid
        mid      = (best_bid + best_ask) / 2

        # Depth within 5% of mid price
        depth = sum(
            p * s
            for p, s in bids + asks
            if abs(p - mid) / max(mid, 1e-9) <= 0.05
        )

        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            depth_5pct=depth,
        )

    # ── trading ───────────────────────────────────────────────────────────────

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Order:
        """
        Submit a limit order.  Minimum size is 5 shares on Polymarket.
        Returns an Order dataclass.
        """
        from py_clob_client.clob_types import OrderArgs, OrderType  # type: ignore

        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=round(size, 2),
            side=side,
        )
        resp = await self._run(
            self._client.create_and_post_order,
            order_args,
        )

        order_id = resp.get("orderID", resp.get("id", "unknown"))
        status   = resp.get("status", "OPEN")

        return Order(
            order_id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=status,
        )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self._run(self._client.cancel, order_id)
            return True
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", order_id, exc)
            return False

    async def get_order(self, order_id: str) -> Order:
        raw = await self._run(self._client.get_order, order_id)
        return Order(
            order_id=order_id,
            token_id=raw.get("asset_id", ""),
            side=raw.get("side", ""),
            price=float(raw.get("price", 0)),
            size=float(raw.get("original_size", 0)),
            status=raw.get("status", "UNKNOWN"),
            filled_at=datetime.fromisoformat(raw["created_at"])
                      if raw.get("status") == "MATCHED" else None,
        )

    # ── positions & balance ───────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        """Return all open positions from the CLOB."""
        resp = await self._run(self._client.get_positions)
        positions: list[Position] = []
        for p in (resp or []):
            try:
                positions.append(Position(
                    token_id=p.get("asset_id", ""),
                    market_condition_id=p.get("condition_id", ""),
                    bucket_label=p.get("outcome", ""),
                    side=p.get("side", "BUY"),
                    size=float(p.get("size", 0)),
                    avg_price=float(p.get("avg_price", 0)),
                    current_price=float(p.get("cur_price", 0)),
                    unrealized_pnl=float(p.get("unrealized_pnl", 0)),
                ))
            except Exception as exc:
                logger.debug("Skipping position: %s", exc)
        return positions

    async def get_usdc_balance(self) -> float:
        """Return USDC balance on Polygon."""
        try:
            bal = await self._run(self._client.get_balance_allowance)
            return float(bal.get("balance", 0))
        except Exception as exc:
            logger.warning("Failed to fetch balance: %s", exc)
            return 0.0
