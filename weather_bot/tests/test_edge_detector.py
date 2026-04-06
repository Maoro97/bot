"""Tests for EdgeDetector."""
import math
import pytest
from ..src.edge_detector import EdgeDetector
from ..src.models import Market, TemperatureBucket
from datetime import datetime


def _make_market(prices: dict[str, float], volume: float = 1000.0) -> Market:
    buckets = [
        TemperatureBucket("≤15°C", -math.inf, 15.5, "tok1"),
        TemperatureBucket("17°C",   16.5,     17.5,  "tok2"),
        TemperatureBucket("≥20°C",  19.5, math.inf,  "tok3"),
    ]
    return Market(
        condition_id="test-cid",
        question="London daily high on April 7?",
        location_name="London",
        target_date="2024-04-07",
        buckets=buckets,
        prices=prices,
        volume=volume,
        end_date=datetime(2024, 4, 7, 23, 59),
    )


class TestFindEdges:
    def test_detects_buy_edge(self):
        det = EdgeDetector(min_edge=0.08, min_kelly=0.01)
        market = _make_market({"≤15°C": 0.10, "17°C": 0.25, "≥20°C": 0.15})
        model_probs = {"≤15°C": 0.05, "17°C": 0.50, "≥20°C": 0.10}
        signals = det.find_edges(market, model_probs, model_agreement=0.80)
        labels = [s.bucket.label for s in signals]
        assert "17°C" in labels

    def test_no_signal_when_edge_below_threshold(self):
        det = EdgeDetector(min_edge=0.15)
        market = _make_market({"≤15°C": 0.33, "17°C": 0.33, "≥20°C": 0.34})
        model_probs = {"≤15°C": 0.34, "17°C": 0.35, "≥20°C": 0.31}
        signals = det.find_edges(market, model_probs, model_agreement=0.80)
        assert signals == []

    def test_skips_low_volume_market(self):
        det = EdgeDetector(min_edge=0.08, min_volume=500.0)
        market = _make_market({"≤15°C": 0.10, "17°C": 0.20, "≥20°C": 0.15}, volume=100.0)
        signals = det.find_edges(market, {"≤15°C": 0.05, "17°C": 0.55, "≥20°C": 0.10})
        assert signals == []

    def test_skips_wide_spread(self):
        det = EdgeDetector(min_edge=0.08, max_spread=0.05)
        market = _make_market({"≤15°C": 0.10, "17°C": 0.20, "≥20°C": 0.15})
        signals = det.find_edges(
            market,
            {"≤15°C": 0.05, "17°C": 0.55, "≥20°C": 0.10},
            spread=0.10,  # wider than threshold
        )
        assert signals == []

    def test_signals_sorted_by_ev(self):
        det = EdgeDetector(min_edge=0.08, min_kelly=0.01)
        market = _make_market({"≤15°C": 0.05, "17°C": 0.15, "≥20°C": 0.08})
        model_probs = {"≤15°C": 0.30, "17°C": 0.45, "≥20°C": 0.30}
        signals = det.find_edges(market, model_probs, model_agreement=0.90)
        if len(signals) > 1:
            evs = [s.expected_value for s in signals]
            assert evs == sorted(evs, reverse=True)

    def test_skips_low_model_agreement(self):
        det = EdgeDetector(min_edge=0.05, min_model_agreement=0.70)
        market = _make_market({"≤15°C": 0.10, "17°C": 0.20, "≥20°C": 0.15})
        signals = det.find_edges(
            market,
            {"≤15°C": 0.05, "17°C": 0.55, "≥20°C": 0.10},
            model_agreement=0.50,  # below threshold
        )
        assert signals == []


class TestKellyEVMath:
    def test_ev_formula(self):
        det = EdgeDetector()
        ev = det.calculate_expected_value(prob=0.60, price=0.40)
        expected = 0.60 * (1 - 0.40) - 0.40 * 0.40
        assert ev == pytest.approx(expected, rel=1e-6)

    def test_kelly_formula(self):
        det = EdgeDetector()
        prob, price = 0.60, 0.40
        b  = (1 - price) / price
        expected_kf = (prob * b - (1 - prob)) / b
        assert det.kelly_fraction(prob, price) == pytest.approx(expected_kf, rel=1e-6)

    def test_kelly_zero_for_fair_odds(self):
        det = EdgeDetector()
        kf = det.kelly_fraction(prob=0.50, price=0.50)
        assert kf == pytest.approx(0.0, abs=1e-6)
