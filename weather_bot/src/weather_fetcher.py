"""
Weather Data Fetcher — pulls forecasts from ECMWF (IFS), GFS, METAR, TAF
and combines them into a consensus forecast.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, date
from statistics import mean, stdev
from typing import Optional

import httpx

from .models import ConsensusForecast, Location

logger = logging.getLogger(__name__)

# ─── endpoints ───────────────────────────────────────────────────────────────
OPEN_METEO_ECMWF  = "https://api.open-meteo.com/v1/ecmwf"
OPEN_METEO_GFS    = "https://api.open-meteo.com/v1/gfs"
AVIATION_METAR    = "https://aviationweather.gov/api/data/metar"
AVIATION_TAF      = "https://aviationweather.gov/api/data/taf"

# ─── model weights for consensus ─────────────────────────────────────────────
MODEL_WEIGHTS = {
    "ecmwf": 0.45,
    "gfs":   0.35,
    "taf":   0.20,
}

# Model-specific uncertainty (°C RMSE baseline)
MODEL_RMSE = {
    "ecmwf": 1.2,
    "gfs":   1.5,
    "taf":   1.0,
}


class WeatherFetcher:
    """
    Fetches temperature forecasts from multiple sources and builds a
    consensus forecast with calibrated uncertainty.
    """

    def __init__(self, timeout: float = 15.0, cache_ttl: int = 120):
        self._timeout = timeout
        self._cache_ttl = cache_ttl          # seconds
        self._cache: dict[str, tuple[datetime, object]] = {}

    # ── internal helpers ─────────────────────────────────────────────────────

    def _cache_key(self, method: str, *args) -> str:
        return f"{method}:{':'.join(str(a) for a in args)}"

    def _from_cache(self, key: str):
        if key in self._cache:
            ts, value = self._cache[key]
            age = (datetime.utcnow() - ts).total_seconds()
            if age < self._cache_ttl:
                return value
        return None

    def _to_cache(self, key: str, value):
        self._cache[key] = (datetime.utcnow(), value)

    async def _get(self, url: str, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    # ── daily high from hourly array ─────────────────────────────────────────

    @staticmethod
    def _daily_high(hourly_times: list[str], hourly_temps: list[float],
                    target_date: str) -> Optional[float]:
        """Return the maximum temperature for *target_date* from hourly arrays."""
        temps_for_day = [
            t for ts, t in zip(hourly_times, hourly_temps)
            if ts.startswith(target_date) and t is not None
        ]
        return max(temps_for_day) if temps_for_day else None

    @staticmethod
    def _daily_mean(hourly_times: list[str], hourly_temps: list[float],
                    target_date: str) -> Optional[float]:
        """Return the mean temperature for *target_date* from hourly arrays."""
        temps_for_day = [
            t for ts, t in zip(hourly_times, hourly_temps)
            if ts.startswith(target_date) and t is not None
        ]
        return mean(temps_for_day) if temps_for_day else None

    # ── public fetch methods ──────────────────────────────────────────────────

    async def fetch_ecmwf(self, lat: float, lon: float, date: str) -> Optional[float]:
        """Return forecast daily-high temperature from ECMWF IFS for *date*."""
        key = self._cache_key("ecmwf", lat, lon, date)
        cached = self._from_cache(key)
        if cached is not None:
            return cached

        try:
            data = await self._get(OPEN_METEO_ECMWF, {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "forecast_days": 7,
                "timezone": "UTC",
            })
            hourly = data.get("hourly", {})
            result = self._daily_high(
                hourly.get("time", []),
                hourly.get("temperature_2m", []),
                date,
            )
            self._to_cache(key, result)
            logger.debug("ECMWF forecast for %s on %s: %.1f°C", (lat, lon), date, result or -999)
            return result
        except Exception as exc:
            logger.warning("ECMWF fetch failed: %s", exc)
            return None

    async def fetch_gfs(self, lat: float, lon: float, date: str) -> Optional[float]:
        """Return forecast daily-high temperature from GFS for *date*."""
        key = self._cache_key("gfs", lat, lon, date)
        cached = self._from_cache(key)
        if cached is not None:
            return cached

        try:
            data = await self._get(OPEN_METEO_GFS, {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "forecast_days": 7,
                "timezone": "UTC",
            })
            hourly = data.get("hourly", {})
            result = self._daily_high(
                hourly.get("time", []),
                hourly.get("temperature_2m", []),
                date,
            )
            self._to_cache(key, result)
            logger.debug("GFS forecast for %s on %s: %.1f°C", (lat, lon), date, result or -999)
            return result
        except Exception as exc:
            logger.warning("GFS fetch failed: %s", exc)
            return None

    async def fetch_metar(self, icao: str) -> Optional[float]:
        """Return current observed temperature from the nearest METAR report."""
        key = self._cache_key("metar", icao)
        cached = self._from_cache(key)
        if cached is not None:
            return cached

        try:
            data = await self._get(AVIATION_METAR, {
                "ids": icao,
                "format": "json",
            })
            if not data:
                return None
            obs = data[0] if isinstance(data, list) else data
            temp = obs.get("temp")
            result = float(temp) if temp is not None else None
            self._to_cache(key, result)
            logger.debug("METAR %s current temp: %s°C", icao, result)
            return result
        except Exception as exc:
            logger.warning("METAR fetch failed for %s: %s", icao, exc)
            return None

    async def fetch_taf(self, icao: str, target_date: str) -> Optional[float]:
        """Return expected high temperature from TAF for *target_date*."""
        key = self._cache_key("taf", icao, target_date)
        cached = self._from_cache(key)
        if cached is not None:
            return cached

        try:
            data = await self._get(AVIATION_TAF, {
                "ids": icao,
                "format": "json",
            })
            if not data:
                return None
            taf = data[0] if isinstance(data, list) else data

            # Extract temperature forecasts from TAF fcsts array
            fcsts = taf.get("fcsts", [])
            temps: list[float] = []
            for fcst in fcsts:
                # TAF may carry temp field (°C) in some formats
                t = fcst.get("temp") or fcst.get("minTemp") or fcst.get("maxTemp")
                if t is not None:
                    try:
                        temps.append(float(t))
                    except (TypeError, ValueError):
                        pass

            result = max(temps) if temps else None
            self._to_cache(key, result)
            logger.debug("TAF %s forecast high: %s°C", icao, result)
            return result
        except Exception as exc:
            logger.warning("TAF fetch failed for %s: %s", icao, exc)
            return None

    # ── consensus ────────────────────────────────────────────────────────────

    async def get_consensus_forecast(self, location: Location,
                                     target_date: Optional[str] = None) -> ConsensusForecast:
        """
        Fetch all sources concurrently and combine into a weighted consensus
        forecast with calibrated uncertainty.
        """
        if target_date is None:
            target_date = date.today().isoformat()

        ecmwf_t, gfs_t, metar_t, taf_t = await asyncio.gather(
            self.fetch_ecmwf(location.lat, location.lon, target_date),
            self.fetch_gfs(location.lat, location.lon, target_date),
            self.fetch_metar(location.icao),
            self.fetch_taf(location.icao, target_date),
            return_exceptions=False,
        )

        # Build weighted mean from available models
        model_values: list[tuple[str, float]] = []
        if ecmwf_t is not None:
            model_values.append(("ecmwf", ecmwf_t))
        if gfs_t is not None:
            model_values.append(("gfs", gfs_t))
        if taf_t is not None:
            model_values.append(("taf", taf_t))

        if not model_values:
            logger.error("No model data available for %s on %s", location.name, target_date)
            raise RuntimeError(f"No forecast data for {location.name} on {target_date}")

        total_weight = sum(MODEL_WEIGHTS[m] for m, _ in model_values)
        weighted_mean = sum(MODEL_WEIGHTS[m] * v for m, v in model_values) / total_weight

        # Uncertainty: inter-model spread + weighted RMSE baseline
        raw_temps = [v for _, v in model_values]
        inter_model_spread = stdev(raw_temps) if len(raw_temps) > 1 else 0.0
        weighted_rmse = sum(MODEL_WEIGHTS[m] * MODEL_RMSE[m] for m, _ in model_values) / total_weight

        # Days ahead: the further the forecast, the higher the uncertainty
        try:
            days_ahead = (date.fromisoformat(target_date) - date.today()).days
        except Exception:
            days_ahead = 1
        time_factor = 1.0 + 0.10 * max(0, days_ahead - 1)  # +10% per extra day

        calibrated_std = (inter_model_spread * 0.5 + weighted_rmse * 0.5) * time_factor

        # Model agreement: 1.0 if all agree within 1°C, degrades with spread
        agreement = max(0.0, 1.0 - inter_model_spread / 3.0)

        all_vals = raw_temps + ([metar_t] if metar_t is not None else [])

        return ConsensusForecast(
            location=location,
            target_date=target_date,
            mean_temp=round(weighted_mean, 2),
            std_temp=round(max(0.5, calibrated_std), 2),   # floor of 0.5°C
            min_temp=round(min(all_vals), 2),
            max_temp=round(max(all_vals), 2),
            model_agreement=round(agreement, 3),
            ecmwf_forecast=ecmwf_t,
            gfs_forecast=gfs_t,
            metar_actual=metar_t,
            taf_forecast=taf_t,
        )
