"""Entrypoint: python -m bot.main

Runs the scanner in a loop against Config.symbols, polling every
Config.poll_interval_seconds. Defaults to Bybit TESTNET + DRY_RUN=true —
see README.md for how to (deliberately, explicitly) go live.
"""
from __future__ import annotations

import logging
import signal
import sys
import time

from bot.config import CONFIG
from bot.exchange_bybit import BybitExchange
from bot.scanner import Scanner

_running = True


def _handle_sigterm(signum, frame) -> None:
    global _running
    _running = False


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def main() -> None:
    setup_logging(CONFIG.log_level)
    logger = logging.getLogger("bybit_bot.main")

    CONFIG.validate_for_live()

    mode = "TESTNET" if CONFIG.testnet else "MAINNET"
    dry = "DRY_RUN (no real orders)" if CONFIG.dry_run else "LIVE ORDERS"
    logger.warning("Starting bot: %s / %s / symbols=%s", mode, dry, CONFIG.symbols)
    if not CONFIG.testnet and not CONFIG.dry_run:
        logger.warning(
            "This will place REAL orders with REAL funds on Bybit MAINNET. "
            "Max leverage is capped, not literal exchange max — see bot/risk.py."
        )

    exchange = BybitExchange(CONFIG)
    scanner = Scanner(exchange=exchange, config=CONFIG)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    while _running:
        try:
            scanner.run_once()
        except Exception:
            logger.exception("Unhandled error in scan loop; continuing after sleep")
        for _ in range(CONFIG.poll_interval_seconds):
            if not _running:
                break
            time.sleep(1)

    logger.info("Shutdown requested, exiting cleanly.")


if __name__ == "__main__":
    main()
