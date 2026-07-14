"""
Trade Executor — thực thi lệnh và quản lý vị thế
- Đặt lệnh với SL + TP1 (safety net trên sàn)
- Trailing stop kích hoạt tại TRAILING_TRIGGER% đường đến TP1 — bảo vệ lợi nhuận nếu đảo chiều trước TP1
- Break-even SL kích hoạt tại BREAKEVEN_TRIGGER% đường đến TP1
- Tự động đóng lệnh khi signal đảo chiều
"""

import logging
import time
from typing import Callable, Optional

from client import BybitClient
from risk_manager import RiskManager, TradeParams
from strategies.base import Signal
from bot_logger import BotLogger
import config

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, client: BybitClient, risk_mgr: RiskManager, bot_logger: BotLogger):
        self.client    = client
        self.risk_mgr  = risk_mgr
        self.logger    = bot_logger

        self._tp1_hit: dict[str, bool]    = {}
        self._breakeven_set: dict[str, bool] = {}
        # ATR luu lai khi vao lenh — dung de tinh trailing stop distance chinh xac
        self._atr: dict[str, float]       = {}
        # Callback duoc goi khi dong lenh lo — (symbol: str) -> None
        self.on_loss_callback: Optional[Callable[[str], None]] = None

    def execute_signal(
        self,
        symbol: str,
        signal: Signal,
        equity: float,
        open_positions: list[dict],
    ):
        """Xử lý signal mới — vào lệnh nếu đủ điều kiện."""
        if signal.direction == 0:
            return

        existing = [p for p in open_positions if p["symbol"] == symbol]
        if existing:
            pos_side = existing[0]["side"]
            signal_side = "Buy" if signal.direction == 1 else "Sell"
            if pos_side == signal_side:
                return

            logger.info(f"{symbol}: Signal reversal — closing {pos_side} before entering {signal_side}")
            self._close_position(existing[0])
            time.sleep(0.5)

        params = self.risk_mgr.compute_trade(signal, equity, open_positions)
        if not params:
            return

        params.symbol = symbol
        self._enter_trade(symbol, signal, params)

    def _enter_trade(self, symbol: str, signal: Signal, params: TradeParams):
        try:
            self.client.set_leverage(symbol, params.leverage)

            # TP1 dat tren san lam safety net — neu gia cham TP1 san tu dong dong
            # Trailing stop se kich hoat truoc TP1 (tai TRAILING_TRIGGER%) de bao ve lai nhuan neu dao chieu
            order = self.client.place_order(
                symbol=symbol,
                side=params.side,
                qty=params.qty,
                sl=params.sl_price,
                tp=params.tp1_price,
            )

            self._tp1_hit[symbol]      = False
            self._breakeven_set[symbol] = False
            self._atr[symbol]          = signal.atr  # luu ATR de trailing stop chinh xac

            self.logger.log_trade({
                "event":     "open",
                "symbol":    symbol,
                "side":      params.side,
                "qty":       params.qty,
                "leverage":  params.leverage,
                "entry":     signal.entry_price,
                "sl":        params.sl_price,
                "tp1":       params.tp1_price,
                "tp2":       params.tp2_price,
                "strategy":  signal.strategy_name,
                "strength":  signal.strength,
                "reason":    signal.reason,
                "notional":  params.notional_usdt,
                "capital":   params.capital_usdt,
                "fee":       params.fee_usdt,
                "order_id":  order.get("orderId", ""),
            })

            logger.info(
                f"[OPEN] {symbol} {params.side} | qty={params.qty} | "
                f"lev={params.leverage}x | SL={params.sl_price:.4f} | "
                f"TP1={params.tp1_price:.4f} | strategy={signal.strategy_name} | "
                f"{signal.reason}"
            )

        except Exception as e:
            logger.error(f"Failed to enter trade {symbol}: {str(e).encode('ascii', 'replace').decode()}")

    def manage_open_positions(self, open_positions: list[dict]):
        """
        Quan ly vi the dang mo theo 3 muc:
        1. BREAKEVEN_TRIGGER (50%): doi SL ve entry + phi
        2. TRAILING_TRIGGER  (75%): kich hoat trailing stop — bao ve lai nhuan
        3. TP1 (100%): san tu dong dong, hoac trailing stop dong truoc neu dao chieu
        """
        for pos in open_positions:
            symbol     = pos["symbol"]
            entry      = float(pos["avgPrice"])
            mark_price = float(pos.get("markPrice", entry))
            side       = pos["side"]

            tp1_threshold = float(pos.get("takeProfit", 0))

            if tp1_threshold <= 0:
                # Kiem tra emergency close ngay ca khi khong co TP
                if self.risk_mgr.should_close_position(pos, mark_price):
                    logger.warning(f"{symbol}: Emergency close — excessive loss")
                    self._close_position(pos)
                continue

            dist_to_tp1 = abs(tp1_threshold - entry)
            dist_moved  = abs(mark_price - entry)

            # --- Muc 1: Break-even SL tai 50% duong den TP1 ---
            if not self._breakeven_set.get(symbol, False) and dist_to_tp1 > 0:
                if dist_moved >= dist_to_tp1 * config.BREAKEVEN_TRIGGER:
                    try:
                        fee_buffer = entry * config.ROUND_TRIP_FEE
                        be_price   = entry + fee_buffer if side == "Buy" else entry - fee_buffer
                        self.client.update_stop_loss(symbol, round(be_price, 6))
                        self._breakeven_set[symbol] = True
                        logger.info(
                            f"{symbol}: Break-even SL -> {be_price:.4f} "
                            f"(moved {dist_moved:.4f}/{dist_to_tp1:.4f} = "
                            f"{dist_moved/dist_to_tp1*100:.0f}% toward TP1)"
                        )
                    except Exception as e:
                        logger.warning(f"{symbol}: Could not set break-even SL: {e}")

            # --- Muc 2: Trailing stop tai TRAILING_TRIGGER% (75%) duong den TP1 ---
            # Kich hoat TRUOC khi san dong tai TP1 — neu dao chieu thi trailing stop bat duoc loi nhuan
            # Neu gia tiep tuc den TP1 thi san tu dong dong (trailing stop vo hieu)
            if not self._tp1_hit.get(symbol, False) and dist_to_tp1 > 0:
                if dist_moved >= dist_to_tp1 * config.TRAILING_TRIGGER:
                    self._tp1_hit[symbol] = True
                    try:
                        # Trailing distance = TRAILING_STOP_ATR x ATR luc vao lenh
                        atr = self._atr.get(symbol, 0)
                        if atr > 0:
                            trailing = config.TRAILING_STOP_ATR * atr
                        else:
                            trailing = dist_to_tp1 * 0.3  # fallback: 30% TP1 distance
                        self.client.set_trading_stop(symbol, side, round(trailing, 6))
                        logger.info(
                            f"{symbol}: Trailing stop activated at "
                            f"{dist_moved/dist_to_tp1*100:.0f}% of TP1 | "
                            f"trailing={trailing:.4f}"
                        )
                    except Exception as e:
                        logger.warning(f"{symbol}: Could not set trailing stop: {e}")

            # --- Emergency close ---
            if self.risk_mgr.should_close_position(pos, mark_price):
                logger.warning(f"{symbol}: Emergency close — excessive loss")
                self._close_position(pos)

    def _close_position(self, position: dict):
        symbol = position["symbol"]
        side   = position["side"]
        qty    = float(position["size"])
        pnl    = float(position.get("unrealisedPnl", 0))
        try:
            self.client.close_position(symbol, side, qty)
            self.logger.log_trade({
                "event":  "close",
                "symbol": symbol,
                "side":   side,
                "qty":    qty,
                "reason": "signal_reversal_or_emergency",
            })
            logger.info(f"[CLOSE] {symbol} {side} qty={qty} pnl={pnl:.4f}")
            if pnl < 0 and self.on_loss_callback:
                self.on_loss_callback(symbol)
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {str(e).encode('ascii', 'replace').decode()}")
