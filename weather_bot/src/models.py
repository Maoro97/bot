"""
Data models for the Polymarket Weather Bot.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Location:
    name: str               # "London", "NYC", "Hong Kong"
    lat: float
    lon: float
    icao: str               # METAR/TAF station code
    timezone: str           # "Europe/London", "America/New_York"
    resolution_source: str  # "weather_underground", "noaa"
    resolution_station: str # "EGLC", "Central Park"


@dataclass
class ConsensusForecast:
    location: Location
    target_date: str            # "2024-04-07"
    mean_temp: float            # °C
    std_temp: float             # uncertainty in °C
    min_temp: float
    max_temp: float
    model_agreement: float      # 0–1 score
    ecmwf_forecast: Optional[float] = None
    gfs_forecast: Optional[float] = None
    metar_actual: Optional[float] = None
    taf_forecast: Optional[float] = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TemperatureBucket:
    label: str          # "18°C", "≤15°C", "≥22°C"
    low: float          # lower bound (can be -inf)
    high: float         # upper bound (can be +inf)
    token_id: str       # Polymarket outcome token ID


@dataclass
class Market:
    condition_id: str
    question: str
    location_name: str
    target_date: str
    buckets: list[TemperatureBucket]
    prices: dict[str, float]    # {label: price}
    volume: float               # USD
    end_date: datetime
    active: bool = True


@dataclass
class TradeSignal:
    market: Market
    bucket: TemperatureBucket
    side: str               # "BUY" or "SELL"
    model_prob: float
    market_price: float
    edge: float             # model_prob - market_price
    kelly_fraction: float
    expected_value: float
    position_size: float    # USD
    confidence: float       # 0–1, based on model agreement
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Order:
    order_id: str
    token_id: str
    side: str
    price: float
    size: float
    status: str             # "OPEN", "FILLED", "CANCELLED", "EXPIRED"
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    token_id: str
    market_condition_id: str
    bucket_label: str
    side: str
    size: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    opened_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeResult:
    signal: TradeSignal
    order: Optional[Order]
    success: bool
    error: Optional[str] = None
    executed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OrderBook:
    token_id: str
    bids: list[tuple[float, float]]  # [(price, size), ...]
    asks: list[tuple[float, float]]
    best_bid: float
    best_ask: float
    spread: float
    depth_5pct: float   # total USD liquidity within 5% of mid


@dataclass
class DailyPnL:
    date: str
    trades_count: int
    wins: int
    losses: int
    gross_pnl: float
    fees: float
    net_pnl: float
    roi_pct: float
    bankroll_end: float


@dataclass
class BacktestResult:
    start_date: str
    end_date: str
    total_trades: int
    win_rate: float
    total_pnl: float
    roi_pct: float
    max_drawdown: float
    sharpe_ratio: float
    calibration_score: float    # Brier score
    trades: list[dict] = field(default_factory=list)
