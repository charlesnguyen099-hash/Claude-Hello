"""Bot configuration, loaded from environment variables (.env). No secret
ever has a hardcoded default — the bot refuses to start live/testnet
trading without real credentials supplied by the operator.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _list_env(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return default
    return [s.strip().upper() for s in val.split(",") if s.strip()]


DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]


@dataclass(frozen=True)
class Config:
    api_key: str = field(default_factory=lambda: os.getenv("BYBIT_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BYBIT_API_SECRET", ""))
    testnet: bool = field(default_factory=lambda: _bool_env("BYBIT_TESTNET", True))
    dry_run: bool = field(default_factory=lambda: _bool_env("DRY_RUN", True))

    symbols: list[str] = field(default_factory=lambda: _list_env("SYMBOLS", DEFAULT_SYMBOLS))
    poll_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
    )
    category: str = "linear"  # Bybit USDT perpetual futures

    max_concurrent_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_POSITIONS", "4"))
    )
    equity_override_usdt: float | None = field(
        default_factory=lambda: (
            float(os.getenv("EQUITY_OVERRIDE_USDT")) if os.getenv("EQUITY_OVERRIDE_USDT") else None
        )
    )

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate_for_live(self) -> None:
        if self.dry_run:
            return
        if not self.api_key or not self.api_secret:
            raise SystemExit(
                "DRY_RUN=false but BYBIT_API_KEY / BYBIT_API_SECRET are not set. "
                "Refusing to place real orders without credentials. "
                "Set them in your .env file (see .env.example)."
            )


CONFIG = Config()
