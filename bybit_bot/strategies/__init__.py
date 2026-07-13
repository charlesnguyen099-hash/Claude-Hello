from .ema_crossover import EMACrossoverStrategy
from .rsi_macd      import RSIMACDStrategy
from .bollinger     import BollingerStrategy
from .supertrend    import SupertrendStrategy
from .vwap_volume   import VWAPVolumeStrategy
from .ichimoku      import IchimokuStrategy
from .breakout      import BreakoutStrategy

ALL_STRATEGIES = [
    EMACrossoverStrategy(),
    RSIMACDStrategy(),
    BollingerStrategy(),
    SupertrendStrategy(),
    VWAPVolumeStrategy(),
    IchimokuStrategy(),
]

BREAKOUT_STRATEGY = BreakoutStrategy(vol_mult=3.5, lookback=20)
