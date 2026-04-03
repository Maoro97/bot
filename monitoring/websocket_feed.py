"""
Real-time WebSocket price feed from Polymarket.

Subscribes to the CLOB WebSocket and receives price updates instantly
instead of polling every 10 seconds. When a price update arrives,
the arbitrage strategy is evaluated immediately.

WebSocket endpoint: wss://ws-subscriptions-clob.polymarket.com/ws/market
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Optional

import aiohttp

from utils.logger import logger

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RECONNECT_DELAY = 5  # seconds before reconnecting after disconnect


class PriceFeed:
    """
    Subscribes to Polymarket WebSocket and calls `on_price_update`
    with (asset_id, best_bid, best_ask) whenever prices change.
    """

    def __init__(self, on_price_update: Callable) -> None:
        self._on_price_update = on_price_update
        self._subscribed_ids: set[str] = set()
        self._running = False
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    def subscribe(self, token_ids: list[str]) -> None:
        """Add token IDs to the subscription list."""
        self._subscribed_ids.update(token_ids)

    async def run(self) -> None:
        """Connect and keep reconnecting on drops."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_listen()
            except Exception as exc:
                logger.warning("WebSocket disconnected: {} — reconnecting in {}s", exc, RECONNECT_DELAY)
            if self._running:
                await asyncio.sleep(RECONNECT_DELAY)

    async def stop(self) -> None:
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _connect_and_listen(self) -> None:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                WS_URL,
                heartbeat=30,
                timeout=aiohttp.ClientWSTimeout(ws_receive=60),
            ) as ws:
                self._ws = ws
                logger.info("WebSocket connected to Polymarket")

                # Subscribe to all token IDs
                await self._send_subscriptions(ws)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_message(msg.data)
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.warning("WebSocket error: {}", ws.exception())
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break

    async def _send_subscriptions(self, ws) -> None:
        if not self._subscribed_ids:
            return
        payload = {
            "assets_ids": list(self._subscribed_ids),
            "type": "market",
        }
        await ws.send_str(json.dumps(payload))
        logger.info("WebSocket subscribed to {} token IDs", len(self._subscribed_ids))

    async def update_subscriptions(self, new_ids: list[str]) -> None:
        """Add new token IDs to an active WebSocket connection."""
        added = set(new_ids) - self._subscribed_ids
        if not added or not self._ws:
            return
        self._subscribed_ids.update(added)
        payload = {"assets_ids": list(added), "type": "market"}
        try:
            await self._ws.send_str(json.dumps(payload))
            logger.debug("WebSocket subscribed to {} new token IDs", len(added))
        except Exception as exc:
            logger.warning("Failed to update WebSocket subscriptions: {}", exc)

    async def _handle_message(self, raw: str) -> None:
        try:
            events = json.loads(raw)
            if not isinstance(events, list):
                events = [events]
            for event in events:
                await self._process_event(event)
        except json.JSONDecodeError:
            pass

    async def _process_event(self, event: dict) -> None:
        """Parse a price change event and call the callback."""
        event_type = event.get("event_type") or event.get("type", "")

        # Price change event
        if event_type in ("price_change", "book"):
            asset_id = event.get("asset_id") or event.get("market", "")
            if not asset_id:
                return

            # Extract best bid/ask from the event
            bids = event.get("bids", [])
            asks = event.get("asks", [])

            best_bid = float(bids[0]["price"]) if bids else None
            best_ask = float(asks[0]["price"]) if asks else None

            if best_bid is not None or best_ask is not None:
                await self._on_price_update(asset_id, best_bid, best_ask)
