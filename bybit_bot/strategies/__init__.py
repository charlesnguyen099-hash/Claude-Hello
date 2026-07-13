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
    SupertrendStrategy(),
    VWAPVolumeStrategy(),
    IchimokuStrategy(),
    SustainedTrendStrategy(),   # strategy thu 7 — bat trend deu + reversal
]

BREAKOUT_STRATEGY = BreakoutStrategy(vol_mult=3.0, lookback=20)
