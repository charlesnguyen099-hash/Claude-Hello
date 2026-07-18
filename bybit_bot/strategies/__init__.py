from .ema_crossover    import EMACrossoverStrategy
from .rsi_macd         import RSIMACDStrategy
from .bollinger        import BollingerStrategy
from .supertrend       import SupertrendStrategy
from .vwap_volume      import VWAPVolumeStrategy
from .ichimoku         import IchimokuStrategy
from .breakout         import BreakoutStrategy
from .sustained_trend  import SustainedTrendStrategy

ALL_STRATEGIES = [
    EMACrossoverStrategy(),
    RSIMACDStrategy(),
    BollingerStrategy(),
    SupertrendStrategy(adx_threshold=20),   # 25 qua cao vs config.MIN_ADX=12, 20 la hop ly
    VWAPVolumeStrategy(),
    IchimokuStrategy(),
    SustainedTrendStrategy(),
]

BREAKOUT_STRATEGY = BreakoutStrategy(vol_mult=2.5, lookback=20)   # 3.0 qua cao (hiem gặp), 2.5 hop ly
