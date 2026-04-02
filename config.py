import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Wallet / auth
    PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
    API_KEY: str = os.getenv("POLYMARKET_API_KEY", os.getenv("API_KEY", ""))
    API_SECRET: str = os.getenv("POLYMARKET_SECRET", os.getenv("API_SECRET", ""))
    API_PASSPHRASE: str = os.getenv("POLYMARKET_PASSPHRASE", os.getenv("API_PASSPHRASE", ""))
    CHAIN_ID: int = 137  # Polygon mainnet

    # API endpoints
    CLOB_HOST: str = "https://clob.polymarket.com"
    GAMMA_HOST: str = "https://gamma-api.polymarket.com"

    # Trading parameters
    MIN_PROFIT_THRESHOLD: float = float(os.getenv("MIN_PROFIT_THRESHOLD", "0.02"))
    MAX_ORDER_SIZE_USDC: float = float(os.getenv("MAX_ORDER_SIZE_USDC", "50"))
    SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "10"))
    MIN_LIQUIDITY: float = float(os.getenv("MIN_LIQUIDITY", "100"))

    # Polymarket charges ~2% fee on winnings
    POLYMARKET_FEE: float = 0.02

    # Mode
    DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() == "true"

    def validate(self) -> None:
        if not self.PRIVATE_KEY:
            raise ValueError("PRIVATE_KEY is required")
        if not self.DRY_RUN and not all([self.API_KEY, self.API_SECRET, self.API_PASSPHRASE]):
            raise ValueError("API_KEY, API_SECRET, and API_PASSPHRASE are required for live trading")


config = Config()
