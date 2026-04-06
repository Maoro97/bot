"""
Edge Detector — identifies statistically significant pricing discrepancies
between our model probabilities and Polymarket's current prices.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from .models import Market, TradeSignal, TemperatureBucket

logger = logging.getLogger(__name__)


class EdgeDetector:
    """
    Compares model-derived probabilities to Polymarket prices and
    emits TradeSignal objects whenever a statistically significant
    edge is found.
    """

    def __init__(
        self,
        min_edge: float = 0.10,           # minimum |model_prob - market_price|
        min_kelly: float = 0.02,          # minimum Kelly fraction (2%)
        min_volume: float = 500.0,        # minimum market volume in USD
        max_spread: float = 0.05,         # maximum bid-ask spread
        min_model_agreement: float = 0.60,# fraction of models that must agree
        min_ev: float = 0.05,             # minimum expected value per dollar wagered
    ):
        self.min_edge = min_edge
        self.min_kelly = min_kelly
        self.min_volume = min_volume
        self.max_spread = max_spread
        self.min_model_agreement = min_model_agreement
        self.min_ev = min_ev

    # ─────────────────────────────────────────────────────────────────────────

    def calculate_expected_value(self, prob: float, price: float) -> float:
        """
        EV per dollar wagered:
            EV = prob * (1 - price) - (1 - prob) * price
        Positive → edge in our favour.
        """
        return prob * (1.0 - price) - (1.0 - prob) * price

    def kelly_fraction(self, prob: float, price: float) -> float:
        """
        Full Kelly Criterion:
            b  = (1 - price) / price   (net odds, i.e. payout per $1 risked)
            f* = (p*b - q) / b
        Returns 0 if the bet is -EV.
        """
        if price <= 0 or price >= 1:
            return 0.0
        b = (1.0 - price) / price
        q = 1.0 - prob
        f = (prob * b - q) / b
        return max(0.0, f)

    # ─────────────────────────────────────────────────────────────────────────

    def find_edges(
        self,
        market: Market,
        model_probs: dict[str, float],
        model_agreement: float = 1.0,
        spread: float = 0.0,
    ) -> list[TradeSignal]:
        """
        Scan all outcome buckets in *market* and return TradeSignals for
        every bucket where we have a meaningful edge.

        Parameters
        ----------
        market         : the Polymarket market being analysed
        model_probs    : {bucket_label: probability} from ProbabilityEngine
        model_agreement: 0–1 score from ConsensusForecast
        spread         : current bid-ask spread from order book
        """
        if market.volume < self.min_volume:
            logger.debug(
                "Skipping %s - volume $%.0f below threshold",
                market.question[:60], market.volume
            )
            return []

        if spread > self.max_spread:
            logger.debug(
                "Skipping %s - spread %.3f above threshold",
                market.question[:60], spread
            )
            return []

        if model_agreement < self.min_model_agreement:
            logger.debug(
                "Skipping %s - model agreement %.2f below threshold",
                market.question[:60], model_agreement
            )
            return []

        signals: list[TradeSignal] = []

        for bucket in market.buckets:
            label = bucket.label
            model_prob   = model_probs.get(label)
            market_price = market.prices.get(label)

            if model_prob is None or market_price is None:
                continue
            if market_price <= 0.01 or market_price >= 0.99:
                continue  # degenerate prices

            edge = model_prob - market_price
            ev   = self.calculate_expected_value(model_prob, market_price)
            kf   = self.kelly_fraction(model_prob, market_price)

            # Decide direction
            if edge > self.min_edge and ev >= self.min_ev and kf >= self.min_kelly:
                side = "BUY"
            elif edge < -self.min_edge and ev <= -self.min_ev:
                # Sell the "NO" outcome (i.e., buy the complementary)
                # On Polymarket you can short by buying the complementary token,
                # but for simplicity we skip direct shorting here.
                # A negative-edge bucket means the complementary bucket is +edge.
                logger.debug("Negative edge on %s - will catch via complementary bucket", label)
                continue
            else:
                continue

            signal = TradeSignal(
                market=market,
                bucket=bucket,
                side=side,
                model_prob=round(model_prob, 4),
                market_price=round(market_price, 4),
                edge=round(edge, 4),
                kelly_fraction=round(kf, 4),
                expected_value=round(ev, 4),
                position_size=0.0,  # filled by RiskManager
                confidence=round(model_agreement, 3),
            )
            signals.append(signal)
            logger.info(
                "EDGE found: %s | %s | model=%.2f%% market=%.2f%% edge=%.2f%% kelly=%.2f%% EV=%.3f",
                market.location_name,
                label,
                model_prob * 100,
                market_price * 100,
                edge * 100,
                kf * 100,
                ev,
            )

        # Sort by expected value descending — best bets first
        signals.sort(key=lambda s: s.expected_value, reverse=True)
        return signals

    # ─────────────────────────────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────────────────────────────

    def summarise(self, signals: list[TradeSignal]) -> str:
        if not signals:
            return "No edges found."
        lines = [f"Found {len(signals)} edge(s):"]
        for s in signals:
            lines.append(
                f"  {s.market.location_name} | {s.bucket.label} "
                f"| edge={s.edge*100:.1f}% EV={s.expected_value:.3f} kelly={s.kelly_fraction*100:.1f}%"
            )
        return "\n".join(lines)
