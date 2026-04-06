"""Tests for RiskManager."""
import math
import os
import tempfile
import pytest
from datetime import datetime
from ..src.risk_manager import RiskManager
from ..src.models import TradeSignal, Market, TemperatureBucket


def _make_signal(size: float = 1.0, condition_id: str = "cid-1") -> TradeSignal:
    bucket = TemperatureBucket("17°C", 16.5, 17.5, "tok1")
    market = Market(
        condition_id=condition_id,
        question="London daily high?",
        location_name="London",
        target_date="2024-04-07",
        buckets=[bucket],
        prices={"17°C": 0.20},
        volume=1000.0,
        end_date=datetime(2024, 4, 7),
    )
    return TradeSignal(
        market=market,
        bucket=bucket,
        side="BUY",
        model_prob=0.35,
        market_price=0.20,
        edge=0.15,
        kelly_fraction=0.12,
        expected_value=0.11,
        position_size=size,
        confidence=0.85,
    )


@pytest.fixture
def rm(tmp_path):
    state_file = str(tmp_path / "risk_state.json")
    manager = RiskManager(state_path=state_file)
    return manager


class TestCheckLimits:
    def test_approves_valid_trade(self, rm):
        signal = _make_signal(size=1.0)
        ok, reason = rm.check_limits(signal, usdc_balance=50.0)
        assert ok, reason

    def test_rejects_insufficient_balance(self, rm):
        signal = _make_signal(size=5.0)
        ok, reason = rm.check_limits(signal, usdc_balance=2.0)
        assert not ok
        assert "USDC" in reason

    def test_rejects_when_paused(self, rm):
        rm.pause()
        signal = _make_signal(size=1.0)
        ok, reason = rm.check_limits(signal, usdc_balance=50.0)
        assert not ok
        assert "paused" in reason.lower()

    def test_rejects_when_daily_stop_loss_hit(self, rm):
        # Simulate a large daily loss
        rm._state["daily_pnl"] = -(RiskManager.STOP_LOSS_DAILY + 1)
        signal = _make_signal(size=1.0)
        ok, reason = rm.check_limits(signal, usdc_balance=50.0)
        assert not ok
        assert "stop-loss" in reason.lower()

    def test_rejects_over_daily_exposure(self, rm):
        rm._state["daily_wagered"] = RiskManager.MAX_DAILY_EXPOSURE - 0.5
        signal = _make_signal(size=2.0)
        ok, reason = rm.check_limits(signal, usdc_balance=50.0)
        assert not ok
        assert "Daily exposure" in reason

    def test_rejects_over_market_exposure(self, rm):
        rm._state["market_exposure"]["cid-1"] = RiskManager.MAX_MARKET_EXPOSURE - 0.5
        signal = _make_signal(size=2.0, condition_id="cid-1")
        ok, reason = rm.check_limits(signal, usdc_balance=50.0)
        assert not ok
        assert "Market exposure" in reason

    def test_rejects_too_many_positions(self, rm):
        rm._state["open_positions"] = RiskManager.MAX_CONCURRENT_POSITIONS
        signal = _make_signal(size=1.0)
        ok, reason = rm.check_limits(signal, usdc_balance=50.0)
        assert not ok
        assert "positions" in reason.lower()


class TestPositionSizing:
    def test_quarter_kelly_sizing(self, rm):
        kf = 0.20
        bankroll = 100.0
        size = rm.calculate_position_size(kelly_fraction=kf, bankroll=bankroll, edge=0.15)
        expected_max = kf * 0.25 * bankroll
        assert size <= RiskManager.MAX_BET_SIZE
        assert size >= RiskManager.MIN_BET_SIZE

    def test_size_capped_at_max_bet(self, rm):
        size = rm.calculate_position_size(kelly_fraction=0.99, bankroll=1000.0, edge=0.50)
        assert size <= RiskManager.MAX_BET_SIZE

    def test_size_floor_respected(self, rm):
        size = rm.calculate_position_size(kelly_fraction=0.001, bankroll=1.0, edge=0.01)
        assert size >= RiskManager.MIN_BET_SIZE


class TestStateManagement:
    def test_record_trade_updates_state(self, rm):
        signal = _make_signal(size=2.0)
        rm.record_trade(signal)
        assert rm.daily_wagered == pytest.approx(2.0)
        assert rm.open_positions_count == 1

    def test_record_resolution_updates_pnl(self, rm):
        rm._state["open_positions"] = 1
        rm.record_resolution(pnl=1.50, condition_id="cid-1")
        assert rm.daily_pnl == pytest.approx(1.50)
        assert rm.open_positions_count == 0

    def test_pause_resume(self, rm):
        assert not rm.is_paused
        rm.pause()
        assert rm.is_paused
        rm.resume()
        assert not rm.is_paused
