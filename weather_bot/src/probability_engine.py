"""
Probability Engine — converts consensus temperature forecasts into
probability distributions over Polymarket temperature buckets.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .models import ConsensusForecast, TemperatureBucket

logger = logging.getLogger(__name__)

# Minimum probability assigned to any bucket (prevents zero-probability edges)
PROB_FLOOR = 1e-4


class ProbabilityEngine:
    """
    Converts a ConsensusForecast into a probability distribution over
    temperature outcome buckets using a (possibly fat-tailed) distribution.
    """

    def __init__(
        self,
        calibration_dir: str = "data/calibration",
        df_t: int = 6,          # degrees of freedom for t-distribution (6 = moderate fat tails)
        use_t_dist: bool = True,
    ):
        self._calibration_dir = Path(calibration_dir)
        self._df_t = df_t
        self._use_t_dist = use_t_dist
        self._bias_cache: dict[str, float] = {}
        self._rmse_cache: dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Core distribution math
    # ─────────────────────────────────────────────────────────────────────────

    def _cdf(self, x: float, mean: float, std: float) -> float:
        """CDF at *x* for the chosen distribution."""
        if self._use_t_dist:
            # Standardise and use t-distribution CDF
            z = (x - mean) / std
            return float(stats.t.cdf(z, df=self._df_t))
        else:
            return float(stats.norm.cdf(x, loc=mean, scale=std))

    def forecast_to_distribution(
        self,
        mean_temp: float,
        std_temp: float,
        buckets: list[TemperatureBucket],
    ) -> dict[str, float]:
        """
        Return a probability for every bucket label using the chosen
        distribution.

        P(bucket) = CDF(high) - CDF(low)

        Open-ended buckets (±∞) are handled automatically since CDF(+∞)=1
        and CDF(−∞)=0.
        """
        probs: dict[str, float] = {}
        for bucket in buckets:
            p = self._cdf(bucket.high, mean_temp, std_temp) - \
                self._cdf(bucket.low, mean_temp, std_temp)
            probs[bucket.label] = max(PROB_FLOOR, p)

        # Renormalise so probabilities sum to 1
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration & uncertainty
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate_uncertainty(
        self,
        model_forecasts: list[float],
        location_name: str,
        target_date: str,
        base_std: float,
    ) -> float:
        """
        Return calibrated std by combining:
        1. Inter-model spread
        2. Historical bias-adjusted RMSE for this location + season
        3. Lead-time inflation

        Falls back to *base_std* if calibration data is unavailable.
        """
        # 1. Inter-model spread contribution
        inter_model = float(np.std(model_forecasts)) if len(model_forecasts) > 1 else 0.0

        # 2. Historical RMSE
        hist_rmse = self._get_historical_rmse(location_name, target_date)

        # 3. Lead-time inflation already baked into base_std from WeatherFetcher
        calibrated = 0.4 * inter_model + 0.6 * (hist_rmse if hist_rmse else base_std)
        return max(0.5, calibrated)

    def apply_bias_correction(
        self,
        raw_mean: float,
        location_name: str,
        target_date: str,
    ) -> float:
        """Subtract historical model bias from the raw mean forecast."""
        bias = self._get_historical_bias(location_name, target_date)
        return raw_mean - bias

    # ─────────────────────────────────────────────────────────────────────────
    # Historical stats helpers (loaded from calibration CSV)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_historical_bias(self, location: str, date_str: str) -> float:
        key = f"{location}_bias"
        if key not in self._bias_cache:
            self._load_calibration(location)
        return self._bias_cache.get(key, 0.0)

    def _get_historical_rmse(self, location: str, date_str: str) -> Optional[float]:
        key = f"{location}_rmse"
        if key not in self._rmse_cache:
            self._load_calibration(location)
        return self._rmse_cache.get(key)

    def _load_calibration(self, location: str):
        """Load bias/RMSE from a per-location CSV if it exists."""
        path = self._calibration_dir / f"{location.lower().replace(' ', '_')}.csv"
        if not path.exists():
            logger.debug("No calibration file for %s", location)
            return

        try:
            df = pd.read_csv(path)
            # Expected columns: forecast_temp, actual_temp
            errors = df["forecast_temp"] - df["actual_temp"]
            self._bias_cache[f"{location}_bias"] = float(errors.mean())
            self._rmse_cache[f"{location}_rmse"] = float(np.sqrt((errors ** 2).mean()))
            logger.info(
                "Loaded calibration for %s: bias=%.2f°C, RMSE=%.2f°C",
                location,
                self._bias_cache[f"{location}_bias"],
                self._rmse_cache[f"{location}_rmse"],
            )
        except Exception as exc:
            logger.warning("Failed to load calibration for %s: %s", location, exc)

    # ─────────────────────────────────────────────────────────────────────────
    # High-level entry point
    # ─────────────────────────────────────────────────────────────────────────

    def compute_probabilities(
        self,
        forecast: ConsensusForecast,
        buckets: list[TemperatureBucket],
    ) -> dict[str, float]:
        """
        Full pipeline:
        1. Bias-correct the mean
        2. Calibrate uncertainty
        3. Compute probability per bucket
        """
        model_temps = [
            t for t in [forecast.ecmwf_forecast, forecast.gfs_forecast, forecast.taf_forecast]
            if t is not None
        ]

        corrected_mean = self.apply_bias_correction(
            forecast.mean_temp,
            forecast.location.name,
            forecast.target_date,
        )

        calibrated_std = self.calibrate_uncertainty(
            model_temps,
            forecast.location.name,
            forecast.target_date,
            forecast.std_temp,
        )

        probs = self.forecast_to_distribution(corrected_mean, calibrated_std, buckets)

        logger.info(
            "Probabilities for %s on %s (mean=%.1f°C ± %.1f°C): %s",
            forecast.location.name,
            forecast.target_date,
            corrected_mean,
            calibrated_std,
            {k: f"{v:.3f}" for k, v in probs.items()},
        )
        return probs

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration record saving
    # ─────────────────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        location_name: str,
        forecast_date: str,
        forecast_temp: float,
        actual_temp: float,
    ):
        """Append a forecast vs. actual record to the calibration CSV."""
        self._calibration_dir.mkdir(parents=True, exist_ok=True)
        path = self._calibration_dir / f"{location_name.lower().replace(' ', '_')}.csv"

        new_row = pd.DataFrame([{
            "date": forecast_date,
            "forecast_temp": forecast_temp,
            "actual_temp": actual_temp,
            "error": forecast_temp - actual_temp,
        }])

        if path.exists():
            existing = pd.read_csv(path)
            df = pd.concat([existing, new_row], ignore_index=True)
        else:
            df = new_row

        df.to_csv(path, index=False)
        logger.info("Recorded calibration for %s: forecast=%.1f actual=%.1f",
                    location_name, forecast_temp, actual_temp)

        # Invalidate cache
        self._bias_cache.pop(f"{location_name}_bias", None)
        self._rmse_cache.pop(f"{location_name}_rmse", None)

    # ─────────────────────────────────────────────────────────────────────────
    # Brier score
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def brier_score(
        predicted_probs: list[float],
        outcomes: list[int],  # 1 if that bucket won, 0 otherwise
    ) -> float:
        """Lower is better. 0.0 = perfect, 0.25 = no skill."""
        n = len(predicted_probs)
        if n == 0:
            return float("nan")
        return sum((p - o) ** 2 for p, o in zip(predicted_probs, outcomes)) / n
