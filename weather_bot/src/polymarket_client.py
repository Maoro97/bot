"""
Polymarket CLOB client wrapper.

Read-only market data (markets list, order books, prices) is fetched via
plain httpx against Polymarket's public REST endpoints -- no credentials
required.  Order placement / account calls use py-clob-client and only
activate when a private key is supplied (live trading mode).
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime
from typing import Optional

import httpx

from .models import Market, OrderBook, Order, Position, TemperatureBucket

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"

# Temperature bucket regex patterns
_RANGE_RE = re.compile(r"([-\d]+)\s*[°℃C]\s*[-–]\s*([-\d]+)\s*[°℃C]")
_ABOVE_RE = re.compile(r"(?:≥|>=|above|over)\s*([-\d]+)\s*[°℃C]", re.I)
_BELOW_RE = re.compile(r"(?:≤|<=|below|under)\s*([-\d]+)\s*[°℃C]", re.I)
_EXACT_RE = re.compile(r"^([-\d]+)\s*(?:°\s*[Cc]|℃)$")

WEATHER_KEYWORDS = {"temperature", "temp", "°c", "celsius", "high", "low", "weather"}


def _parse_bucket(label: str, token_id: str) -> TemperatureBucket:
    """Convert an outcome label like '<=15°C', '18°C', '20°C-22°C' into a TemperatureBucket."""
    m = _RANGE_RE.search(label)
    if m:
        return TemperatureBucket(label=label, low=float(m.group(1)), high=float(m.group(2)), token_id=token_id)

    m = _ABOVE_RE.search(label)
    if m:
        return TemperatureBucket(label=label, low=float(m.group(1)), high=math.inf, token_id=token_id)

    m = _BELOW_RE.search(label)
    if m:
        return TemperatureBucket(label=label, low=-math.inf, high=float(m.group(1)), token_id=token_id)

    m = _EXACT_RE.match(label.strip())
    if m:
        mid = float(m.group(1))
        return TemperatureBucket(label=label, low=mid - 0.5, high=mid + 0.5, token_id=token_id)

    logger.debug("Could not parse bucket label '%s' -- skipping", label)
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
    iso = re.search(r"\d{4}-\d{2}-\d{2}", question)
    if iso:
        return iso.group(0)
    return None


class PolymarketClient:
    """
    Async Polymarket client.

    Public market data is fetched via httpx (no credentials needed).
    Trading operations require a private_key and use py-clob-client.
    """

    KNOWN_LOCATIONS = [
        "London", "NYC", "New York", "Hong Kong", "Seoul", "Tokyo",
        "Paris", "Berlin", "Sydney", "Singapore", "Dubai", "Chicago",
        "Moscow", "Istanbul", "Tel Aviv", "Bangkok", "Mumbai", "Cairo",
        "Lagos", "Toronto", "Melbourne", "São Paulo", "Buenos Aires",
    ]

    def __init__(
        self,
        private_key: str = "",
        chain_id: int = 137,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
    ):
        self._private_key = private_key
        self._chain_id = chain_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._clob_client = None  # lazy-initialised only when needed for trading

        # Shared async HTTP client for public endpoints
        self._http = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "polymarket-weather-bot/1.0"},
        )

    def _get_clob_client(self):
        """Lazy-load py-clob-client (only required for live trading)."""
        if self._clob_client is None:
            if not self._private_key:
                raise RuntimeError(
                    "Live trading requires POLY_PRIVATE_KEY to be set in settings.yaml "
                    "or as an environment variable."
                )
            from py_clob_client.client import ClobClient  # type: ignore
            from py_clob_client.clob_types import ApiCreds  # type: ignore

            creds = ApiCreds(
                api_key=self._api_key,
                api_secret=self._api_secret,
                api_passphrase=self._api_passphrase,
            ) if self._api_key else None

            self._clob_client = ClobClient(
                host=CLOB_HOST,
                key=self._private_key,
                chain_id=self._chain_id,
                creds=creds,
            )
        return self._clob_client

    async def close(self):
        await self._http.aclose()

    # ── market discovery (public, no credentials) ─────────────────────────────

    async def get_weather_markets(self) -> list[Market]:
        """
        Fetch active weather/temperature markets from Polymarket's public API.

        Polymarket groups temperature buckets under an 'event' (e.g. 'Highest
        temperature in Hong Kong on April 7?').  Each bucket is a separate
        YES/NO market inside that event.  We call /events, then flatten each
        event's sub-markets into one Market object with multiple buckets.
        """
        markets: list[Market] = []
        seen: set[str] = set()
        limit = 100

        # Try different tag filter parameter names against the Gamma API
        tag_filter: dict = {}
        for tp in [{"tag": "weather"}, {"tag": "daily-temperature"},
                   {"tag_slug": "weather"}, {"tag_slug": "daily-temperature"},
                   {"slug": "weather"}]:
            try:
                test = await self._http.get(
                    f"{GAMMA_HOST}/events",
                    params={"active": "true", "closed": "false", "limit": 2, **tp},
                )
                items = test.json() if isinstance(test.json(), list) else test.json().get("data", [])
                if items:
                    tag_filter = tp
                    logger.info("Tag filter %s works (%d results)", tp, len(items))
                    break
                logger.info("Tag filter %s: 0 results", tp)
            except Exception:
                pass

        for page in range(20):
            try:
                resp = await self._http.get(
                    f"{GAMMA_HOST}/events",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": limit,
                        "offset": page * limit,
                        **tag_filter,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("Failed to fetch events (page=%d): %s", page, exc)
                break

            events = data if isinstance(data, list) else data.get("events", data.get("data", []))
            if not events:
                logger.info("Page %d: empty, stopping", page)
                break

            for raw_event in events:
                title = raw_event.get("title", "") or raw_event.get("question", "")
                title_lower = title.lower()

                # Check title keywords (strict — avoid "attempt", "Temple", "Tempo")
                title_match = "temperature" in title_lower or "celsius" in title_lower

                # Also check event tags for weather/temperature labels
                raw_tags = raw_event.get("tags", [])
                tag_labels = set()
                for t in raw_tags:
                    if isinstance(t, dict):
                        tag_labels.add(t.get("label", t.get("slug", "")).lower())
                    else:
                        tag_labels.add(str(t).lower())
                tag_match = bool(tag_labels & {"weather", "temperature", "daily temperature",
                                               "daily-temperature", "climate"})

                if not (title_match or tag_match):
                    continue
                try:
                    market = self._parse_event(raw_event)
                    if market and market.condition_id not in seen:
                        markets.append(market)
                        seen.add(market.condition_id)
                except Exception as exc:
                    logger.debug("Skipping event '%s': %s", title, exc)

            if len(events) < limit:
                break

            # Stop early once we have a good batch
            if len(markets) >= 50:
                break

        logger.info("Found %d active weather markets", len(markets))
        return markets

    def _parse_event(self, event: dict) -> Optional[Market]:
        """
        Convert a Gamma API event dict into our Market dataclass.

        Each event contains a list of sub-markets, one per temperature bucket.
        The sub-market's question is the bucket label (e.g. '28°C') and its
        YES token is what we trade.
        """
        title: str = event.get("title", "") or event.get("question", "")
        if not title:
            return None

        end_date_str = event.get("endDate") or event.get("end_date", "")
        try:
            end_date = datetime.fromisoformat(str(end_date_str).rstrip("Z"))
        except Exception:
            end_date = datetime.utcnow()

        if end_date < datetime.utcnow():
            return None

        sub_markets: list[dict] = event.get("markets", []) or []
        if not sub_markets:
            logger.debug("Event '%s' has no sub-markets", title)
            return None

        buckets: list[TemperatureBucket] = []
        prices: dict[str, float] = {}
        condition_id = event.get("id", event.get("conditionId", ""))

        for sm in sub_markets:
            # The sub-market question IS the bucket label (e.g. "28°C")
            label = sm.get("question", sm.get("title", "")).strip()
            if not label:
                continue

            # YES token id
            clob_ids = sm.get("clobTokenIds", [])
            token_id = clob_ids[0] if clob_ids else sm.get("conditionId", "")

            # YES price (first of outcomePrices)
            outcome_prices = sm.get("outcomePrices", [])
            try:
                price = float(outcome_prices[0]) if outcome_prices else 0.5
            except (ValueError, TypeError):
                price = 0.5

            bucket = _parse_bucket(label, str(token_id))
            # Only keep buckets that parsed to a real range (not -inf/+inf fallback)
            if bucket.low != -math.inf or bucket.high != math.inf:
                buckets.append(bucket)
                prices[label] = price

        if not buckets:
            logger.debug("Event '%s' - no temperature buckets parsed from %d sub-markets",
                         title, len(sub_markets))
            return None

        location_name = _extract_location(title, self.KNOWN_LOCATIONS) or "Unknown"
        target_date = _extract_date(title) or end_date.date().isoformat()
        volume = float(event.get("volume", 0) or 0)

        return Market(
            condition_id=str(condition_id),
            question=title,
            location_name=location_name,
            target_date=target_date,
            buckets=buckets,
            prices=prices,
            volume=volume,
            end_date=end_date,
        )

    def _parse_market(self, raw: dict) -> Optional[Market]:
        question: str = raw.get("question", "") or raw.get("title", "")
        tags: list[str] = raw.get("tags", []) or []
        if isinstance(tags[0], dict) if tags else False:
            tags = [t.get("label", t.get("slug", "")) for t in tags]

        if not _is_weather_market(question, tags):
            return None

        condition_id = raw.get("conditionId") or raw.get("condition_id", "")
        end_date_str = raw.get("endDate") or raw.get("end_date_iso") or raw.get("end_date", "")
        try:
            end_date = datetime.fromisoformat(str(end_date_str).rstrip("Z"))
        except Exception:
            end_date = datetime.utcnow()

        if end_date < datetime.utcnow():
            return None

        # Outcomes / tokens
        tokens: list[dict] = raw.get("tokens", []) or raw.get("outcomes", []) or []
        buckets: list[TemperatureBucket] = []
        prices: dict[str, float] = {}

        for token in tokens:
            label = token.get("outcome") or token.get("title", "")
            token_id = token.get("token_id") or token.get("clobTokenIds", [""])[0] if isinstance(token.get("clobTokenIds"), list) else token.get("clobTokenIds", "")
            price = float(token.get("price", 0.5))
            bucket = _parse_bucket(label, str(token_id))
            buckets.append(bucket)
            prices[label] = price

        if not buckets:
            return None

        location_name = _extract_location(question, self.KNOWN_LOCATIONS) or "Unknown"
        target_date = _extract_date(question) or end_date.date().isoformat()
        volume = float(raw.get("volume", 0) or 0)

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

    # ── order book (public, no credentials) ──────────────────────────────────

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch the current order book for a single outcome token."""
        try:
            resp = await self._http.get(
                f"{CLOB_HOST}/book",
                params={"token_id": token_id},
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:
            logger.warning("Order book fetch failed for %s: %s", token_id, exc)
            raw = {}

        bids = [(float(b["price"]), float(b["size"])) for b in raw.get("bids", [])]
        asks = [(float(a["price"]), float(a["size"])) for a in raw.get("asks", [])]

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 1.0
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2

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

    # ── trading (requires private_key / live mode only) ───────────────────────

    async def place_limit_order(self, token_id: str, side: str, price: float, size: float) -> Order:
        from py_clob_client.clob_types import OrderArgs  # type: ignore

        client = self._get_clob_client()
        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=round(size, 2),
            side=side,
        )
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: client.create_and_post_order(order_args))

        return Order(
            order_id=resp.get("orderID", resp.get("id", "unknown")),
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=resp.get("status", "OPEN"),
        )

    async def cancel_order(self, order_id: str) -> bool:
        try:
            client = self._get_clob_client()
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: client.cancel(order_id))
            return True
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", order_id, exc)
            return False

    async def get_order(self, order_id: str) -> Order:
        client = self._get_clob_client()
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, lambda: client.get_order(order_id))
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

    async def get_positions(self) -> list[Position]:
        client = self._get_clob_client()
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, client.get_positions)
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
        try:
            client = self._get_clob_client()
            loop = asyncio.get_event_loop()
            bal = await loop.run_in_executor(None, client.get_balance_allowance)
            return float(bal.get("balance", 0))
        except Exception as exc:
            logger.warning("Failed to fetch balance: %s", exc)
            return 0.0
