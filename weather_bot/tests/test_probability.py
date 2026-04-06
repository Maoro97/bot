"""Tests for ProbabilityEngine."""
import math
import pytest
from ..src.probability_engine import ProbabilityEngine
from ..src.models import TemperatureBucket


def _make_buckets():
    return [
        TemperatureBucket("≤15°C",  -math.inf, 15.5,  "tok1"),
        TemperatureBucket("16°C",    15.5,      16.5,  "tok2"),
        TemperatureBucket("17°C",    16.5,      17.5,  "tok3"),
        TemperatureBucket("18°C",    17.5,      18.5,  "tok4"),
        TemperatureBucket("19°C",    18.5,      19.5,  "tok5"),
        TemperatureBucket("≥20°C",   19.5,  math.inf,  "tok6"),
    ]


class TestForecastToDistribution:
    def test_probabilities_sum_to_one(self):
        eng = ProbabilityEngine(use_t_dist=False)
        probs = eng.forecast_to_distribution(17.0, 1.5, _make_buckets())
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_highest_prob_at_mean(self):
        eng = ProbabilityEngine(use_t_dist=False)
        probs = eng.forecast_to_distribution(17.0, 1.0, _make_buckets())
        # Bucket containing 17.0 should have the highest probability
        assert probs["17°C"] == max(probs.values())

    def test_all_probs_positive(self):
        eng = ProbabilityEngine(use_t_dist=False)
        probs = eng.forecast_to_distribution(17.0, 1.5, _make_buckets())
        assert all(p > 0 for p in probs.values())

    def test_skewed_distribution(self):
        eng = ProbabilityEngine(use_t_dist=False)
        # High mean → upper buckets should dominate
        probs = eng.forecast_to_distribution(22.0, 1.0, _make_buckets())
        assert probs["≥20°C"] > probs["≤15°C"]

    def test_t_distribution_fatter_tails(self):
        eng_t    = ProbabilityEngine(use_t_dist=True, df_t=4)
        eng_norm = ProbabilityEngine(use_t_dist=False)
        buckets  = _make_buckets()
        p_t    = eng_t.forecast_to_distribution(17.0, 1.0, buckets)
        p_norm = eng_norm.forecast_to_distribution(17.0, 1.0, buckets)
        # t-dist should put more weight on extreme buckets
        assert p_t["≤15°C"] > p_norm["≤15°C"]

    def test_single_bucket_probability_one(self):
        eng = ProbabilityEngine(use_t_dist=False)
        single = [TemperatureBucket("all", -math.inf, math.inf, "t1")]
        probs = eng.forecast_to_distribution(17.0, 1.0, single)
        assert abs(probs["all"] - 1.0) < 1e-6


class TestKellyAndEV:
    def test_positive_edge_positive_kelly(self):
        eng = ProbabilityEngine()
        from ..src.edge_detector import EdgeDetector
        det = EdgeDetector()
        kf = det.kelly_fraction(prob=0.40, price=0.25)
        assert kf > 0

    def test_no_edge_zero_kelly(self):
        from ..src.edge_detector import EdgeDetector
        det = EdgeDetector()
        kf = det.kelly_fraction(prob=0.25, price=0.25)
        assert kf == pytest.approx(0.0, abs=1e-6)

    def test_ev_positive_when_edge_positive(self):
        from ..src.edge_detector import EdgeDetector
        det = EdgeDetector()
        ev = det.calculate_expected_value(prob=0.40, price=0.25)
        assert ev > 0

    def test_ev_negative_when_no_edge(self):
        from ..src.edge_detector import EdgeDetector
        det = EdgeDetector()
        ev = det.calculate_expected_value(prob=0.20, price=0.30)
        assert ev < 0


class TestBrierScore:
    def test_perfect_prediction(self):
        score = ProbabilityEngine.brier_score([1.0, 0.0], [1, 0])
        assert score == pytest.approx(0.0)

    def test_worst_prediction(self):
        score = ProbabilityEngine.brier_score([0.0, 1.0], [1, 0])
        assert score == pytest.approx(1.0)

    def test_empty_returns_nan(self):
        import math
        score = ProbabilityEngine.brier_score([], [])
        assert math.isnan(score)
