import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import talib.abstract as ta
import pandas_ta as pta  # pandas_ta is imported but not explicitly used in the provided code.
# If it's for future use or part of an older version, that's okay.
# Otherwise, it can be removed if not needed.
from scipy.signal import argrelextrema

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]
def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):
        window_data = df['ha_close'].iloc[i - window:i + 1]
        if df['ha_close'].iloc[i] == window_data.min() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            minima[i] = -window
        if df['ha_close'].iloc[i] == window_data.max() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            maxima[i] = window

    return minima, maxima


class AlexBattleTankKillerV37(IStrategy):
    """
    Enhanced strategy on the 15-minute timeframe with Market Correlation Filters.

    Key improvements:
      - Fixed Stop Loss and Added Long and Short Exits
      - Dynamic stoploss based on ATR.
      - Dynamic leverage calculation.
      - Murrey Math level calculation (rolling window for performance).
      - Enhanced DCA (Average Price) logic.
      - Translated to English and code structured for clarity.
      - Parameterization of internal constants for optimization.
      - Changed Exit Signals for Opposite.
      - Change SL to -0.15
      - Changed stake amout for renentry
      - FIXED: Prevents opening opposite position when trade is active
      - FIXED: Trailing stop properly disabled
      - NEW: Market correlation filters for better entry timing
    """

    # General strategy parameters
    timeframe = "30m"
    startup_candle_count: int = 200
    stoploss = -0.15
    #trailing_stop = True
    #trailing_stop_positive = 0.008  # Start trailing at 0.8% (very early)
    #trailing_stop_positive_offset = 0.015  # Begin after 1.5% profit
    #trailing_only_offset_is_reached = True
    position_adjustment_enable = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = False
    max_stake_per_trade = 10.0
    max_portfolio_percentage_per_trade = 0.05
    max_entry_position_adjustment = 1
    process_only_new_candles = True
    max_dca_orders = 2
    max_total_stake_per_pair = 10
    max_single_dca_amount = 4

    enable_position_flip = True
    trailing_stop = True
    trailing_only_offset_is_reached = True
    stoploss = -0.10

    # Set these as regular floats first
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.015


    # 🎯 NEW SIGNAL FLIP PARAMETERS
    signal_flip_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)  # NEW
    signal_flip_min_profit = DecimalParameter(-0.05, 0.02, default=-0.02, decimals=3, space="sell", optimize=True, load=True)  # NEW
    signal_flip_max_loss = DecimalParameter(-0.10, -0.03, default=-0.06, decimals=2, space="sell", optimize=True, load=True)  # NEW

    # Signal confirmation parameters
    signal_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True, load=True)
    signal_cooldown_period = IntParameter(3, 15, default=5, space="buy", optimize=True, load=True)
    min_signal_strength = DecimalParameter(0.7, 0.9, default=0.8, decimals=2, space="buy", optimize=True, load=True)

    # Trend stability parameters
    trend_stability_period = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    min_trend_strength = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space="buy", optimize=True, load=True)

    # Signal filtering parameters
    enable_signal_filtering = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    require_volume_confirmation = BooleanParameter(default=True, space="buy", optimize=True, load=True)

    flip_profit_threshold = DecimalParameter(low=0.01, high=0.10, default=0.03, decimals=3, space="buy", optimize=True, load=True)
    # 🚀 FOLLOW THE PRICE PARAMETERS (NEU)
    follow_price_enabled = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    follow_price_activation_profit = DecimalParameter(0.01, 0.05, default=0.02, decimals=3, space="sell", optimize=True, load=True)  # CHANGED: Was 0.01, now 0.02
    follow_price_pullback_pct = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True)  # CHANGED: Range tightened
    follow_price_min_profit = DecimalParameter(0.005, 0.04, default=0.01, decimals=3, space="sell", optimize=True, load=True)

    # 🎯 MML-BASIERTE FOLLOW PRICE (NEU)
    follow_price_use_mml = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    follow_price_mml_buffer = DecimalParameter(0.001, 0.01, default=0.003, decimals=4, space="sell", optimize=True, load=True)

    # 📊 MARKET CONDITION ADJUSTMENTS (NEU)
    follow_price_market_adjustment = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    follow_price_volatile_multiplier = DecimalParameter(0.5, 1.0, default=0.7, decimals=2, space="sell", optimize=True, load=True)
    follow_price_bearish_multiplier = DecimalParameter(0.6, 1.0, default=0.8, decimals=2, space="sell", optimize=True, load=True)

    # 🔧 ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(1.8, 3.0, default=2.2, decimals=1, space="sell", optimize=True, load=True)
    atr_stoploss_minimum = DecimalParameter(-0.04, -0.02, default=-0.03, decimals=2, space="sell", optimize=True, load=True)  # CHANGED: Was -0.05, now tighter range
    atr_stoploss_maximum = DecimalParameter(-0.10, -0.06, default=-0.08, decimals=2, space="sell", optimize=True, load=True)  # CHANGED: Was -0.25, now -0.15
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 3, default=2, space="buy", optimize=True, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True)
    mml_const2 = DecimalParameter(0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True)
    indicator_mml_window = IntParameter(32, 128, default=64, space="buy", optimize=True, load=True)

    # Dynamic Stoploss parameters
    stoploss_atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell", optimize=True,
                                               load=True)
    stoploss_max_reasonable = DecimalParameter(-0.30, -0.10, default=-0.20, decimals=2, space="sell", optimize=True,
                                               load=True)
    # === Hyperopt Parameters ===
    dominance_threshold = IntParameter(1, 10, default=3, space="buy", optimize=True)
    tightness_factor = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)
    long_rsi_threshold = IntParameter(50, 65, default=55, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=40, space="sell", optimize=True)

    # Dynamic Leverage parameters
    # leverage_window_size = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)
    # leverage_base = DecimalParameter(5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=True, load=True)
    # leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    # leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    # leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
    #                                                  load=True)
    # leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
    #                                                  load=True)
    # leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
    #                                                        optimize=True, load=True)
    # leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
    #                                               load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 10, default=5, space="buy", optimize=True, load=True)
    indicator_rolling_window_threshold = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    indicator_rolling_check_window = IntParameter(2, 10, default=4, space="buy", optimize=True, load=True)

    # === Market Correlation Parameters ===
    # Bitcoin correlation parameters
    enable_btc_correlation = BooleanParameter(default=True, space="buy", optimize=False)
    btc_correlation_enabled = BooleanParameter(default=False, space="buy", optimize=False)
    btc_correlation_threshold = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="buy", optimize=True)
    btc_trend_filter_enabled = BooleanParameter(default=False, space="buy", optimize=True)


    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)

    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(24, 72, default=48, space="buy", optimize=True)

    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(default=False, space="buy", optimize=True)  # Optional
    fear_greed_extreme_threshold = IntParameter(20, 30, default=25, space="buy", optimize=True)
    fear_greed_greed_threshold = IntParameter(70, 80, default=75, space="buy", optimize=True)
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    momentum_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True)

    # Market Breadth
    # Add these to your strategy class:
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=True)
    market_breadth_bull_threshold = DecimalParameter(0.35, 0.65, default=0.50, space="buy", optimize=True, load=True)
    market_breadth_bear_threshold = DecimalParameter(0.35, 0.65, default=0.50, space="buy", optimize=True, load=True)
    market_breadth_sensitivity = DecimalParameter(0.05, 0.20, default=0.10, space="buy", optimize=True, load=True)
    breadth_weighting_enabled = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    volume_confirmation_enabled = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # ROI table (minutes to decimal)
    minimal_roi = {
        "0": 0.15,     # 15% - Only exit on big wins
        "60": 0.10,    # 10% after 1 hour
        "180": 0.08,   # 8% after 3 hours
        "360": 0.05,   # 5% after 6 hours
        "720": 0.03,   # 3% after 12 hours
        "1440": 0.01   # 1% after 24 hours
    }

    plot_config = {
        "main_plot": {
            # Price action and key levels
            "actual_stoploss": {"color": "white", "type": "line"},
            "follow_price_level": {"color": "green", "type": "line"},

            # Key MML levels on main chart for better visibility
            "[2/8]P": {"color": "#ff6b6b", "type": "line"},  # 25% - Support/Resistance
            "[4/8]P": {"color": "#4ecdc4", "type": "line"},  # 50% - Key level
            "[6/8]P": {"color": "#45b7d1", "type": "line"},  # 75% - Support/Resistance

            # Entry signals on main chart
            "enter_long": {"color": "lime", "type": "scatter"},
            "enter_short": {"color": "red", "type": "scatter"},
        },
        "subplots": {
            # 📊 EXIT SIGNALS ANALYSIS (NEW - Most Important)
            "exit_signals": {
                "exit_long": {"color": "#ff4757", "type": "scatter"},          # Red exit longs
                "exit_short": {"color": "#2ed573", "type": "scatter"},         # Green exit shorts
                "Signal_Flip_Short": {"color": "#ff9ff3", "type": "scatter"}, # Pink signal flips
                "Signal_Flip_Long": {"color": "#54a0ff", "type": "scatter"},   # Blue signal flips
                "Momentum_Loss": {"color": "#ff6b35", "type": "scatter"},      # Orange momentum loss
                "Support_Break": {"color": "#ff3838", "type": "scatter"},      # Dark red breaks
                "Resistance_Break": {"color": "#2ecc71", "type": "scatter"},   # Green breaks
            },

            # 🎯 TRADE PERFORMANCE INDICATORS (NEW)
            "trade_metrics": {
                "price_change_5": {"color": "#74b9ff", "type": "line"},       # 5-period price change
                "rsi_change": {"color": "#fd79a8", "type": "line"},           # RSI momentum
                "volume_surge": {"color": "#fdcb6e", "type": "line"},         # Volume spikes
            },

            # 📈 RSI WITH EXIT ZONES (ENHANCED)
            "rsi_analysis": {
                "rsi": {"color": "#6c5ce7", "type": "line"},
                "rsi_overbought_70": {"color": "#fd79a8", "type": "line"},    # 70 level
                "rsi_oversold_30": {"color": "#00b894", "type": "line"},      # 30 level
            },

            # 🏔️ EXTREMA ANALYSIS (KEEP - Important for entries)
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },

            # 📍 MIN/MAX VISUALIZATION (KEEP - Entry confirmation)
            "min_max_viz": {
                "maxima": {"color": "#a29db9", "type": "scatter"},            # Changed to scatter
                "minima": {"color": "#aac7fc", "type": "scatter"},            # Changed to scatter
                "maxima_check": {"color": "#e17055", "type": "line"},
                "minima_check": {"color": "#74b9ff", "type": "line"},
            },

            # 🎯 ADDITIONAL MML LEVELS (ENHANCED)
            "murrey_math_levels": {
                "[1/8]P": {"color": "#d63031", "type": "line"},               # 12.5% - Extreme oversold
                "[3/8]P": {"color": "#fd79a8", "type": "line"},               # 37.5% - Support
                "[4/8]P": {"color": "#0984e3", "type": "line"},               # 50% - Key level (duplicate for subplot)
                "[5/8]P": {"color": "#00b894", "type": "line"},               # 62.5% - Resistance
                "[7/8]P": {"color": "#e84393", "type": "line"},               # 87.5% - Extreme overbought
                "[8/8]P": {"color": "#d63031", "type": "line"},               # 100% - Top
                "[0/8]P": {"color": "#d63031", "type": "line"},               # 0% - Bottom
            },

            # 🌐 MARKET CORRELATION (KEEP - Market context)
            "market_correlation": {
                "btc_correlation": {"color": "#0984e3", "type": "line"},
                "market_breadth": {"color": "#00b894", "type": "line"},
                "market_score": {"color": "#6c5ce7", "type": "line"},
                "market_direction": {"color": "#fd79a8", "type": "line"},     # NEW
            },

            # 📊 MARKET REGIME (ENHANCED)
            "market_regime": {
                "market_volatility": {"color": "#e17055", "type": "line"},
                "mcap_trend": {"color": "#fdcb6e", "type": "line"},
                "market_adx": {"color": "#74b9ff", "type": "line"},           # NEW - Trend strength
            },

            # 📉 VOLUME ANALYSIS (NEW - Important for exit confirmation)
            "volume_analysis": {
                "volume": {"color": "#636e72", "type": "bar"},
                "volume_ratio": {"color": "#00cec9", "type": "line"},         # Volume vs average
                "avg_volume": {"color": "#fd79a8", "type": "line"},           # 20-period average
            },

            # 🎯 TREND ANALYSIS (NEW - For exit timing)
            "trend_analysis": {
                "trend_consistency": {"color": "#a29bfe", "type": "line"},    # Multi-timeframe trend
                "trend_strength": {"color": "#fd79a8", "type": "line"},       # Trend momentum
                "strong_uptrend": {"color": "#00b894", "type": "line"},       # Boolean uptrend
                "strong_downtrend": {"color": "#d63031", "type": "line"},     # Boolean downtrend
            },

            # 🔥 SIGNAL QUALITY (NEW - Signal strength analysis)
            "signal_quality": {
                "combined_signal_strength": {"color": "#fd79a8", "type": "line"},  # Overall signal strength
                "mml_signal_strength": {"color": "#74b9ff", "type": "line"},       # MML-based strength
                "rsi_signal_strength": {"color": "#00b894", "type": "line"},       # RSI-based strength
                "volume_strength": {"color": "#fdcb6e", "type": "line"},           # Volume confirmation
            }
        },
    }
    def informative_pairs(self):
        """
        🚀 COMPREHENSIVE: All pairs needed for market analysis
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []

        # Detect mode from first pair
        sample_pair = pairs[0] if pairs else "BTC/USDT:USDT"
        is_futures = ':' in sample_pair

        if is_futures:
            settlement = sample_pair.split(':')[1]
            logger.info(f"📊 FUTURES MODE - Settlement: {settlement}")

            # BTC pairs for correlation and regime
            btc_pairs = [f'BTC/USDT:{settlement}', 'BTC/USDT']

            # Market breadth pairs
            breadth_pairs = [
                f'BTC/USDT:{settlement}', f'ETH/USDT:{settlement}', f'BNB/USDT:{settlement}',
                f'SOL/USDT:{settlement}', f'ADA/USDT:{settlement}', f'AVAX/USDT:{settlement}',
                f'DOT/USDT:{settlement}', f'POL/USDT:{settlement}', f'XRP/USDT:{settlement}'
            ]
        else:
            logger.info(f"📊 SPOT MODE")

            # BTC pairs for correlation and regime
            btc_pairs = ['BTC/USDT', 'BTC/USD']

            # Market breadth pairs
            breadth_pairs = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
                'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'POL/USDT', 'XRP/USDT'
            ]

        # Add BTC pairs (multiple timeframes for regime analysis)
        for btc_pair in btc_pairs:
            for tf in ['15m', '1h', '4h']:
                informative_pairs.append((btc_pair, tf))
                logger.debug(f"📈 Adding BTC: {btc_pair} {tf}")

        # Add breadth pairs (main timeframe only)
        for breadth_pair in breadth_pairs:
            informative_pairs.append((breadth_pair, self.timeframe))
            logger.debug(f"📊 Adding breadth: {breadth_pair} {self.timeframe}")

        logger.info(f"🎯 Total informative pairs: {len(informative_pairs)}")
        return informative_pairs
    def calculate_signal_strength(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate signal strength to avoid weak/uncertain signals
        """

        # RSI signal strength
        dataframe['rsi_signal_strength'] = 0.0
        dataframe.loc[dataframe['rsi'] < 25, 'rsi_signal_strength'] = 1.0  # Very oversold
        dataframe.loc[dataframe['rsi'].between(25, 35), 'rsi_signal_strength'] = 0.8  # Oversold
        dataframe.loc[dataframe['rsi'].between(35, 40), 'rsi_signal_strength'] = 0.6  # Mildly oversold
        dataframe.loc[dataframe['rsi'] > 75, 'rsi_signal_strength'] = 1.0  # Very overbought
        dataframe.loc[dataframe['rsi'].between(65, 75), 'rsi_signal_strength'] = 0.8  # Overbought
        dataframe.loc[dataframe['rsi'].between(60, 65), 'rsi_signal_strength'] = 0.6  # Mildly overbought

        # MML signal strength
        dataframe['mml_signal_strength'] = 0.0

        # Strong MML signals
        dataframe.loc[dataframe['close'] < dataframe['[1/8]P'], 'mml_signal_strength'] = 1.0  # At 12.5%
        dataframe.loc[dataframe['close'] > dataframe['[7/8]P'], 'mml_signal_strength'] = 1.0  # At 87.5%

        # Medium MML signals
        dataframe.loc[dataframe['close'] < dataframe['[2/8]P'], 'mml_signal_strength'] = 0.8  # At 25%
        dataframe.loc[dataframe['close'] > dataframe['[6/8]P'], 'mml_signal_strength'] = 0.8  # At 75%

        # MML bounces
        mml_bounce_strength = 0.0
        dataframe['mml_bounce_strength'] = np.where(
            (dataframe['low'] <= dataframe['[2/8]P']) & (dataframe['close'] > dataframe['[2/8]P']),
            0.9,  # Strong bounce from 25%
            np.where(
                (dataframe['high'] >= dataframe['[6/8]P']) & (dataframe['close'] < dataframe['[6/8]P']),
                0.9,  # Strong rejection at 75%
                mml_bounce_strength
            )
        )

        # Volume confirmation strength
        dataframe['volume_strength'] = np.where(
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5,
            1.0,  # High volume
            np.where(
                dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2,
                0.8,  # Medium volume
                np.where(
                    dataframe['volume'] > dataframe['volume'].rolling(20).mean(),
                    0.6,  # Above average volume
                    0.3   # Low volume
                )
            )
        )

        # Extrema confirmation strength
        dataframe['extrema_strength'] = 0.0
        dataframe.loc[dataframe['minima'] == 1, 'extrema_strength'] = 0.8
        dataframe.loc[dataframe['maxima'] == 1, 'extrema_strength'] = 0.8

        # Combined signal strength
        dataframe['combined_signal_strength'] = (
            dataframe['rsi_signal_strength'] * 0.3 +
            dataframe['mml_signal_strength'] * 0.25 +
            dataframe['mml_bounce_strength'] * 0.2 +
            dataframe['volume_strength'] * 0.15 +
            dataframe['extrema_strength'] * 0.1
        )

        return dataframe

    def check_signal_confirmation(self, dataframe: pd.DataFrame, signal_type: str) -> pd.Series:
        """
        Check if signal persists for required confirmation period
        """
        confirmation_period = self.signal_confirmation_candles.value

        if signal_type == 'long':
            # Check if bullish conditions persist
            bullish_conditions = (
                (dataframe['rsi'] < 45) |  # Oversold or neutral
                (dataframe['close'] > dataframe['[4/8]P']) |  # Above 50% MML
                (dataframe['minima'] == 1) |  # Local bottom
                (dataframe['close'] > dataframe['close'].shift(1))  # Price rising
            )

            # Signal confirmed if conditions met for X candles
            confirmed = bullish_conditions.rolling(confirmation_period).sum() >= (confirmation_period * 0.7)

        elif signal_type == 'short':
            # Check if bearish conditions persist
            bearish_conditions = (
                (dataframe['rsi'] > 55) |  # Overbought or neutral
                (dataframe['close'] < dataframe['[4/8]P']) |  # Below 50% MML
                (dataframe['maxima'] == 1) |  # Local top
                (dataframe['close'] < dataframe['close'].shift(1))  # Price falling
            )

            # Signal confirmed if conditions met for X candles
            confirmed = bearish_conditions.rolling(confirmation_period).sum() >= (confirmation_period * 0.7)
        else:
            confirmed = pd.Series([False] * len(dataframe))

        return confirmed

    def check_signal_cooldown(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Prevent signals too close together
        """
        cooldown_period = self.signal_cooldown_period.value

        # Track recent signals
        dataframe['recent_long_signal'] = dataframe['enter_long'].rolling(cooldown_period).sum()
        dataframe['recent_short_signal'] = dataframe['enter_short'].rolling(cooldown_period).sum()

        # Only allow new signals if no recent signals
        can_signal = (
            (dataframe['recent_long_signal'] == 0) &
            (dataframe['recent_short_signal'] == 0)
        )

        return can_signal

    def detect_market_state(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            """
            Detect current market state to avoid bad timing
            """

            # Price action analysis
            dataframe['price_volatility'] = dataframe['atr'] / dataframe['close']
            dataframe['volatility_percentile'] = dataframe['price_volatility'].rolling(50).rank(pct=True)

            # Volume analysis
            dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume'].rolling(20).mean()
            dataframe['volume_increasing'] = dataframe['volume'] > dataframe['volume'].shift(1)

            # Momentum analysis
            dataframe['momentum_3'] = dataframe['close'].pct_change(3)
            dataframe['momentum_5'] = dataframe['close'].pct_change(5)
            dataframe['momentum_10'] = dataframe['close'].pct_change(10)

            # Trend consistency
            dataframe['consistent_uptrend'] = (
                (dataframe['momentum_3'] > 0) &
                (dataframe['momentum_5'] > 0) &
                (dataframe['momentum_10'] > 0)
            )

            dataframe['consistent_downtrend'] = (
                (dataframe['momentum_3'] < 0) &
                (dataframe['momentum_5'] < 0) &
                (dataframe['momentum_10'] < 0)
            )

            # Market state classification
            dataframe['market_state'] = 'neutral'

            # Strong trending states
            dataframe.loc[
                dataframe['consistent_uptrend'] &
                (dataframe['adx'] > 30) &
                (dataframe['volume_ratio'] > 1.2),
                'market_state'
            ] = 'strong_uptrend'

            dataframe.loc[
                dataframe['consistent_downtrend'] &
                (dataframe['adx'] > 30) &
                (dataframe['volume_ratio'] > 1.2),
                'market_state'
            ] = 'strong_downtrend'

            # Choppy/ranging states
            dataframe.loc[
                (dataframe['adx'] < 20) &
                (dataframe['volatility_percentile'] < 0.5),
                'market_state'
            ] = 'ranging'

            # High volatility states
            dataframe.loc[
                dataframe['volatility_percentile'] > 0.8,
                'market_state'
            ] = 'high_volatility'

            # Reversal detection
            dataframe['potential_reversal'] = (
                # Price making new extremes but momentum diverging
                (
                    (dataframe['high'] >= dataframe['high'].rolling(10).max()) &
                    (dataframe['momentum_3'] < dataframe['momentum_3'].shift(3))
                ) |
                (
                    (dataframe['low'] <= dataframe['low'].rolling(10).min()) &
                    (dataframe['momentum_3'] > dataframe['momentum_3'].shift(3))
                )
            )

            # Breakout detection
            dataframe['potential_breakout'] = (
                (dataframe['close'] > dataframe['high'].rolling(20).max().shift(1)) |
                (dataframe['close'] < dataframe['low'].rolling(20).min().shift(1))
            ) & (dataframe['volume_ratio'] > 1.5)

            return dataframe

    def get_timing_filters(self, dataframe: pd.DataFrame, direction: str) -> pd.Series:
        """
        Get timing-based filters to avoid bad entry timing
        """

        if direction == 'long':
            good_timing = (
                # Avoid entering during strong downtrends
                (dataframe['market_state'] != 'strong_downtrend') &

                # Prefer breakouts or oversold conditions
                (
                    dataframe['potential_breakout'] |
                    (dataframe['rsi'] < 40) |
                    (dataframe['market_state'] == 'strong_uptrend')
                ) &

                # Avoid high volatility unless extremely oversold
                (
                    (dataframe['market_state'] != 'high_volatility') |
                    (dataframe['rsi'] < 25)
                ) &

                # Volume confirmation if required
                (
                    (~self.require_volume_confirmation.value) |
                    (dataframe['volume_ratio'] > 1.1)
                )
            )

        else:  # short
            good_timing = (
                # Avoid entering during strong uptrends
                (dataframe['market_state'] != 'strong_uptrend') &

                # Prefer breakdowns or overbought conditions
                (
                    dataframe['potential_breakout'] |
                    (dataframe['rsi'] > 60) |
                    (dataframe['market_state'] == 'strong_downtrend')
                ) &

                # Avoid high volatility unless extremely overbought
                (
                    (dataframe['market_state'] != 'high_volatility') |
                    (dataframe['rsi'] > 75)
                ) &

                # Volume confirmation if required
                (
                    (~self.require_volume_confirmation.value) |
                    (dataframe['volume_ratio'] > 1.1)
                )
            )

        return good_timing
    # Helper method to check if we have an active position in the opposite direction
    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
            "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
            "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
            "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
            "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        """
        OPTIMIERTE Version - nur alle 5 Candles berechnen und interpolieren
        FIXED: Pandas 2.0+ kompatibel
        """
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        rolling_high = df["high"].rolling(window=window_size, min_periods=window_size).max()
        rolling_low = df["low"].rolling(window=window_size, min_periods=window_size).min()
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value

        # Nur alle 5 Candles berechnen für Performance
        calculation_step = 5

        for i in range(0, len(df), calculation_step):
            if i < window_size - 1:
                continue

            mn_period = rolling_low.iloc[i]
            mx_period = rolling_high.iloc[i]
            current_close = df["close"].iloc[i]

            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][i] = current_close
                continue

            levels = self._calculate_mml_core(mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2)

            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][i] = levels.get(key, current_close)

        # FIXED: Moderne pandas syntax
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index)
            series = series.interpolate(method='linear').bfill().ffill()  # FIXED
            murrey_levels_data[key] = series.tolist()

        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}

    def calculate_market_correlation_simple(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Simplified correlation calculation that avoids index issues
        """
        pair = metadata['pair']
        base_currency = pair.split('/')[0]

        # Skip if this IS BTC
        if base_currency == 'BTC':
            dataframe['btc_correlation'] = 1.0
            dataframe['btc_trend'] = 1
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close'].rolling(20).mean()
            dataframe['btc_sma50'] = dataframe['close'].rolling(50).mean()
            dataframe['returns'] = dataframe['close'].pct_change()
            return dataframe

        # Try to get BTC data
        btc_pairs = ["BTC/USDT:USDT", "BTC/USDT", "BTC/USD"]
        btc_dataframe = None

        for btc_pair in btc_pairs:
            try:
                btc_dataframe, _ = self.dp.get_analyzed_dataframe(btc_pair, self.timeframe)
                if btc_dataframe is not None and not btc_dataframe.empty and len(btc_dataframe) >= 50:
                    logger.info(f"Using {btc_pair} for simplified BTC correlation with {pair}")
                    break
            except:
                continue

        # If no BTC data, use neutral values
        if btc_dataframe is None or btc_dataframe.empty:
            dataframe['btc_correlation'] = 0.5
            dataframe['btc_trend'] = 0
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']
            dataframe['returns'] = dataframe['close'].pct_change()
            logger.warning(f"No BTC data for {pair}, using neutral correlation")
            return dataframe

        # Simple correlation using only latest values
        try:
            # Use shorter period to avoid length issues
            correlation_period = min(20, len(dataframe) // 5, len(btc_dataframe) // 5)
            correlation_period = max(5, correlation_period)

            # Calculate returns
            pair_returns = dataframe['close'].pct_change().dropna()
            btc_returns = btc_dataframe['close'].pct_change().dropna()

            # Use only last N periods for correlation
            if len(pair_returns) >= correlation_period and len(btc_returns) >= correlation_period:
                pair_recent = pair_returns.tail(correlation_period)
                btc_recent = btc_returns.tail(correlation_period)

                # Align lengths
                min_len = min(len(pair_recent), len(btc_recent))
                correlation = pair_recent.tail(min_len).corr(btc_recent.tail(min_len))

                if pd.isna(correlation):
                    correlation = 0.5
            else:
                correlation = 0.5

            # BTC trend (simple)
            btc_close = btc_dataframe['close'].iloc[-1]
            btc_sma20 = btc_dataframe['close'].rolling(20).mean().iloc[-1]
            btc_sma50 = btc_dataframe['close'].rolling(50).mean().iloc[-1]

            if pd.isna(btc_sma20) or pd.isna(btc_sma50):
                btc_trend = 0
            elif btc_close > btc_sma20 > btc_sma50:
                btc_trend = 1
            elif btc_close < btc_sma20 < btc_sma50:
                btc_trend = -1
            else:
                btc_trend = 0

            # Set constant values for entire dataframe
            dataframe['btc_correlation'] = correlation
            dataframe['btc_trend'] = btc_trend
            dataframe['btc_close'] = btc_close
            dataframe['btc_sma20'] = btc_sma20
            dataframe['btc_sma50'] = btc_sma50

            logger.debug(f"{pair} Simple correlation: {correlation:.3f}, BTC trend: {btc_trend}")

        except Exception as e:
            logger.warning(f"Simple correlation failed for {pair}: {e}")
            dataframe['btc_correlation'] = 0.5
            dataframe['btc_trend'] = 0
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']

        # Always calculate returns
        dataframe['returns'] = dataframe['close'].pct_change()

        return dataframe

    def calculate_market_breadth_weighted(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔧 FIXED: Proper logging levels for market breadth calculation
        """
        current_pair = metadata['pair']
        is_futures = ':' in current_pair

        if is_futures:
            settlement = current_pair.split(':')[1]
            primary_pairs = [
                (f"BTC/USDT:{settlement}", 0.40),
                (f"ETH/USDT:{settlement}", 0.25),
                (f"BNB/USDT:{settlement}", 0.10),
                (f"SOL/USDT:{settlement}", 0.08),
                (f"XRP/USDT:{settlement}", 0.07),
                (f"ADA/USDT:{settlement}", 0.05),
                (f"AVAX/USDT:{settlement}", 0.02),
            ]

            fallback_pairs = [
                ("BTC/USDT", 0.40),
                ("ETH/USDT", 0.25),
                ("BNB/USDT", 0.10),
                ("SOL/USDT", 0.08),
                ("XRP/USDT", 0.07),
                ("ADA/USDT", 0.05),
                ("AVAX/USDT", 0.02),
            ]
        else:
            primary_pairs = [
                ("BTC/USDT", 0.40),
                ("ETH/USDT", 0.25),
                ("BNB/USDT", 0.10),
                ("SOL/USDT", 0.08),
                ("XRP/USDT", 0.07),
                ("ADA/USDT", 0.05),
                ("AVAX/USDT", 0.02),
            ]
            fallback_pairs = []

        bullish_weight = 0.0
        bearish_weight = 0.0
        neutral_weight = 0.0
        total_weight = 0.0
        pairs_checked = 0

        logger.debug(f"🔍 DEBUG BREADTH for {metadata['pair']}:")  # Changed from ERROR to DEBUG

        # Try primary pairs with proper logging levels
        for check_pair, weight in primary_pairs:
            try:
                pair_data = self.dp.get_pair_dataframe(check_pair, self.timeframe)

                if pair_data is None or pair_data.empty:
                    logger.debug(f"   ❌ {check_pair}: No data")  # Changed from ERROR to DEBUG
                    continue

                if len(pair_data) < 20:
                    logger.debug(f"   ❌ {check_pair}: Only {len(pair_data)} candles")  # Changed from ERROR to DEBUG
                    continue

                current_close = pair_data['close'].iloc[-1]
                sma20 = pair_data['close'].rolling(20).mean().iloc[-1]

                if pd.isna(current_close) or pd.isna(sma20) or sma20 <= 0:
                    logger.debug(f"   ❌ {check_pair}: Invalid data (close: {current_close}, sma: {sma20})")  # Changed from ERROR to DEBUG
                    continue

                # Enhanced threshold logic
                threshold = 1.002  # 0.2% buffer
                price_vs_sma = current_close / sma20

                logger.debug(f"   📊 {check_pair}: Close=${current_close:.3f}, SMA20=${sma20:.3f}, Ratio={price_vs_sma:.6f}")  # Changed from ERROR to DEBUG

                if current_close > sma20 * threshold:
                    bullish_weight += weight
                    trend_status = "BULLISH"
                elif current_close < sma20 / threshold:
                    bearish_weight += weight
                    trend_status = "BEARISH"
                else:
                    neutral_weight += weight
                    trend_status = "NEUTRAL"

                total_weight += weight
                pairs_checked += 1

                logger.debug(f"   ✅ {check_pair}: {trend_status} (weight: {weight:.2f})")  # Changed from ERROR to DEBUG

            except Exception as e:
                logger.warning(f"   💥 {check_pair} ERROR: {str(e)}")  # Changed from ERROR to WARNING
                continue

        # Enhanced calculation logic with proper logging
        logger.debug(f"📊 BREADTH CALCULATION:")  # Changed from ERROR to DEBUG
        logger.debug(f"   Bullish weight: {bullish_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Bearish weight: {bearish_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Neutral weight: {neutral_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Total weight: {total_weight:.3f}")  # Changed from ERROR to DEBUG
        logger.debug(f"   Pairs checked: {pairs_checked}")  # Changed from ERROR to DEBUG

        if pairs_checked > 0 and total_weight > 0:
            # Include neutral in calculation
            directional_weight = bullish_weight + bearish_weight

            if directional_weight > 0:
                market_breadth = bullish_weight / directional_weight
            else:
                # All neutral = 50%
                market_breadth = 0.5
                logger.debug(f"   🔧 All pairs neutral, setting breadth to 50%")  # Changed from ERROR to DEBUG

            # Market direction
            if bullish_weight > bearish_weight:
                market_direction = 1
            elif bearish_weight > bullish_weight:
                market_direction = -1
            else:
                market_direction = 0

            logger.info(f"📊 {metadata['pair']} Market Breadth: {market_breadth:.1%} ({pairs_checked} pairs)")  # Changed from ERROR to INFO

        else:
            logger.info(f"🔄 {metadata['pair']} No breadth data - using fallback calculation")  # Changed from ERROR to INFO

            # Fallback: Use current pair trend
            try:
                current_close = dataframe['close'].iloc[-1]
                sma20 = dataframe['close'].rolling(20).mean().iloc[-1]

                if not pd.isna(sma20) and sma20 > 0:
                    pair_ratio = current_close / sma20
                    logger.debug(f"   📈 Current pair trend: {pair_ratio:.6f}")  # Changed from ERROR to DEBUG

                    if pair_ratio > 1.01:  # 1% above SMA
                        market_breadth = 0.65
                        market_direction = 1
                        logger.debug(f"   📈 Pair bullish -> 65% breadth")  # Changed from ERROR to DEBUG
                    elif pair_ratio < 0.99:  # 1% below SMA
                        market_breadth = 0.35
                        market_direction = -1
                        logger.debug(f"   📉 Pair bearish -> 35% breadth")  # Changed from ERROR to DEBUG
                    else:
                        market_breadth = 0.5
                        market_direction = 0
                        logger.debug(f"   📊 Pair neutral -> 50% breadth")  # Changed from ERROR to DEBUG
                else:
                    market_breadth = 0.5
                    market_direction = 0
                    logger.debug(f"   🔧 Invalid pair data -> 50% breadth")  # Changed from ERROR to DEBUG

                pairs_checked = 1

            except Exception as e:
                logger.warning(f"   💥 Fallback failed: {e}")  # Changed from ERROR to WARNING
                market_breadth = 0.5
                market_direction = 0
                pairs_checked = 0

        # Prevent 0% breadth with proper logging
        if market_breadth == 0.0:
            logger.info(f"🔧 {metadata['pair']} Adjusting 0% breadth to 35% (extreme bearish)")  # Changed from ERROR to INFO
            market_breadth = 0.35  # Assume bearish but not extreme
            market_direction = -1

        logger.info(f"🎯 {metadata['pair']} Final Breadth: {market_breadth:.1%} (Direction: {market_direction})")  # Changed from ERROR to INFO, improved formatting

        # Set dataframe values
        dataframe['market_breadth'] = market_breadth
        dataframe['market_direction'] = market_direction
        dataframe['bullish_weight'] = bullish_weight
        dataframe['bearish_weight'] = bearish_weight
        dataframe['total_weight'] = total_weight
        dataframe['breadth_pairs_checked'] = pairs_checked

        return dataframe
    # FIX 2: ALIGN MARKET REGIME NAMING (2 minutes)
    def calculate_market_regime_aligned(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔧 FIXED: Proper logging levels for regime calculation
        """
        lookback = self.regime_lookback_period.value
        current_pair = metadata['pair']

        # Detect mode and set BTC pairs
        if ':' in current_pair:
            settlement = current_pair.split(':')[1]
            btc_pairs_to_try = [f'BTC/USDT:{settlement}', 'BTC/USDT']
        else:
            btc_pairs_to_try = ['BTC/USDT', 'BTC/USD']

        btc_data = None
        btc_pair_used = None

        # Try to get BTC data with proper logging
        for btc_pair in btc_pairs_to_try:
            try:
                logger.debug(f"🔍 {current_pair} Trying BTC pair: {btc_pair}")  # Changed from INFO to DEBUG

                btc_data = self.dp.get_pair_dataframe(btc_pair, self.timeframe)

                if btc_data is not None and not btc_data.empty and len(btc_data) >= lookback:
                    btc_pair_used = btc_pair
                    logger.debug(f"✅ {current_pair} Successfully got {len(btc_data)} candles from {btc_pair}")  # Changed from INFO to DEBUG
                    break
                else:
                    available_candles = len(btc_data) if btc_data is not None and not btc_data.empty else 0
                    logger.debug(f"⚠️ {current_pair} {btc_pair} insufficient: {available_candles}/{lookback} candles")  # Changed from WARNING to DEBUG

            except Exception as e:
                logger.debug(f"💥 {current_pair} Error accessing {btc_pair}: {str(e)}")  # Changed from WARNING to DEBUG
                continue

        # Handle case when no BTC data available
        if btc_data is None or btc_data.empty or len(btc_data) < lookback:
            logger.info(f"🔄 {current_pair} No BTC data available, using neutral regime")  # Changed from ERROR to INFO

            # Set fallback values with INFO level
            dataframe['market_regime'] = 'neutral'
            dataframe['market_volatility'] = 0.02
            dataframe['market_adx'] = 25
            dataframe['btc_trend_strength'] = 0.0
            return dataframe

        try:
            # Calculate regime indicators with proper logging
            logger.debug(f"📊 {current_pair} Calculating regime using {btc_pair_used} ({len(btc_data)} candles)")  # Changed from INFO to DEBUG

            # ... rest of regime calculation ...
            # (Keep the existing calculation logic, just change logging levels)

            # Calculate volatility
            btc_returns = btc_data['close'].pct_change()
            market_volatility = btc_returns.rolling(lookback).std()
            current_volatility = market_volatility.iloc[-1] if not market_volatility.empty and not pd.isna(market_volatility.iloc[-1]) else 0.02

            # Calculate ADX with error handling
            try:
                btc_adx = ta.ADX(btc_data, timeperiod=14)
                current_adx = btc_adx.iloc[-1] if len(btc_adx) > 0 and not pd.isna(btc_adx.iloc[-1]) else 25
            except Exception as e:
                logger.debug(f"⚠️ {current_pair} ADX calculation failed: {e}")  # Changed from WARNING to DEBUG
                current_adx = 25

            # Calculate price trend with multiple fallbacks
            current_close = btc_data['close'].iloc[-1]

            # Try different SMA periods
            sma_periods = [50, 30, 20, 10]
            sma_value = None
            sma_period_used = None

            for period in sma_periods:
                try:
                    if len(btc_data) >= period:
                        sma_temp = btc_data['close'].rolling(period).mean().iloc[-1]
                        if not pd.isna(sma_temp) and sma_temp > 0:
                            sma_value = sma_temp
                            sma_period_used = period
                            break
                except:
                    continue

            if sma_value is not None:
                price_trend = (current_close - sma_value) / sma_value
                logger.debug(f"📈 {current_pair} Price trend: {price_trend:.4f} (SMA{sma_period_used})")  # Changed from INFO to DEBUG
            else:
                price_trend = 0.0
                logger.debug(f"⚠️ {current_pair} Could not calculate price trend, using 0")  # Changed from WARNING to DEBUG

            # Regime classification with proper logging
            logger.debug(f"🧮 {current_pair} Regime inputs:")  # Changed from INFO to DEBUG
            logger.debug(f"   Volatility: {current_volatility:.4f}")  # Changed from INFO to DEBUG
            logger.debug(f"   ADX: {current_adx:.2f}")  # Changed from INFO to DEBUG
            logger.debug(f"   Price Trend: {price_trend:.4f}")  # Changed from INFO to DEBUG

            # Determine regime with more conservative thresholds
            if current_volatility > 0.04:
                regime = 'high_volatility'
                reason = f"High volatility ({current_volatility:.3f})"
            elif current_adx > 30 and price_trend > 0.03:
                regime = 'bull_run'
                reason = f"Strong uptrend (ADX:{current_adx:.1f}, trend:{price_trend:.3f})"
            elif current_adx > 30 and price_trend < -0.03:
                regime = 'bear_market'
                reason = f"Strong downtrend (ADX:{current_adx:.1f}, trend:{price_trend:.3f})"
            elif current_adx < 18:
                regime = 'sideways'
                reason = f"Low trend strength (ADX:{current_adx:.1f})"
            elif 18 <= current_adx <= 30:
                regime = 'transitional'
                reason = f"Moderate trend (ADX:{current_adx:.1f})"
            else:
                regime = 'neutral'
                reason = "Default neutral"

            logger.info(f"🏛️ {current_pair} Market Regime: {regime} ({reason})")  # Keep as INFO - this is important

            # Set dataframe values
            dataframe['market_regime'] = regime
            dataframe['market_volatility'] = current_volatility
            dataframe['market_adx'] = current_adx
            dataframe['btc_trend_strength'] = price_trend
            dataframe['regime_calculation_source'] = btc_pair_used

            return dataframe

        except Exception as e:
            logger.warning(f"💥 {current_pair} Regime calculation failed: {str(e)}")  # Changed from ERROR to WARNING

            # Emergency fallback
            dataframe['market_regime'] = 'neutral'
            dataframe['market_volatility'] = 0.02
            dataframe['market_adx'] = 25
            dataframe['btc_trend_strength'] = 0.0

            return dataframe
    def apply_adaptive_breadth_filter(self, dataframe: pd.DataFrame, direction: str = 'long') -> pd.Series:
        """
        🎯 ADAPTIVE: Market breadth filter with dynamic thresholds
        """
        market_regime = dataframe.get('market_regime', 'neutral').iloc[-1] if 'market_regime' in dataframe.columns else 'neutral'
        market_volatility = dataframe.get('market_volatility', 0.02).iloc[-1] if 'market_volatility' in dataframe.columns else 0.02

        if direction == 'long':
            base_threshold = self.market_breadth_bull_threshold.value

            # Adjust based on market conditions
            if market_regime == 'high_volatility':
                # Be more selective in volatile markets
                threshold = base_threshold + self.market_breadth_sensitivity.value
            elif market_regime == 'bull_run':
                # Be less selective in strong bull markets
                threshold = base_threshold - self.market_breadth_sensitivity.value
            elif market_regime == 'sideways':
                # Standard threshold for ranging markets
                threshold = base_threshold
            else:
                threshold = base_threshold

            breadth_ok = dataframe['market_breadth'] > threshold

        else:  # short
            base_threshold = self.market_breadth_bear_threshold.value

            if market_regime == 'high_volatility':
                threshold = base_threshold - self.market_breadth_sensitivity.value
            elif market_regime == 'bear_market':
                threshold = base_threshold + self.market_breadth_sensitivity.value
            elif market_regime == 'sideways':
                threshold = base_threshold
            else:
                threshold = base_threshold

            breadth_ok = dataframe['market_breadth'] < (1 - threshold)

        return breadth_ok
    def calculate_breadth_with_volume(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        📊 ENHANCED: Breadth with volume confirmation
        """
        # Your existing breadth calculation
        dataframe = self.calculate_market_breadth_weighted(dataframe, metadata)

        # Add volume-based confirmation
        current_pair = metadata['pair']
        is_futures = ':' in current_pair

        if is_futures:
            settlement = current_pair.split(':')[1]
            major_pairs = [f"BTC/USDT:{settlement}", f"ETH/USDT:{settlement}"]
        else:
            major_pairs = ["BTC/USDT", "ETH/USDT"]

        volume_strength = 0
        pairs_checked = 0

        for pair in major_pairs:
            try:
                pair_data, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if not pair_data.empty and len(pair_data) >= 20:
                    current_volume = pair_data['volume'].iloc[-1]
                    avg_volume = pair_data['volume'].rolling(20).mean().iloc[-1]

                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        if volume_ratio > 1.2:  # 20% above average
                            volume_strength += 1
                        pairs_checked += 1
            except:
                continue

        # Volume confirmation score
        volume_confirmation = volume_strength / pairs_checked if pairs_checked > 0 else 0.5

        # Combined breadth score
        dataframe['volume_confirmation'] = volume_confirmation
        dataframe['breadth_volume_combined'] = (
            dataframe['market_breadth'] * 0.8 +
            volume_confirmation * 0.2
        )

        return dataframe
    def apply_market_regime_filters(self, df: pd.DataFrame, any_long_signal: pd.Series,
                                   any_short_signal: pd.Series, metadata: dict) -> tuple:
        """
        🧠 INTELLIGENT: Adjust signal sensitivity based on market conditions

        This method modifies your entry signals based on the current market regime
        to improve timing and reduce stop losses.

        Returns: (enhanced_long_signal, enhanced_short_signal)
        """

        try:
            # Get current market regime (use last available value)
            current_regime = df['market_regime'].iloc[-1] if 'market_regime' in df.columns else 'neutral'
            btc_strength = df['btc_strength'].iloc[-1] if 'btc_strength' in df.columns else 0
            fear_greed = df['market_fear_greed'].iloc[-1] if 'market_fear_greed' in df.columns else 50

            # ===========================================
            # REGIME-SPECIFIC SIGNAL ADJUSTMENTS
            # ===========================================

            if current_regime == 'bull_run':
                # 🚀 BULL RUN MODE: Favor longs, restrict shorts
                long_boost = df['rsi'] < 60  # More aggressive long entries
                short_restrict = df['rsi'] > 80  # Only extreme short entries

                enhanced_long_signal = any_long_signal & long_boost
                enhanced_short_signal = any_short_signal & short_restrict

                logger.info(f"{metadata['pair']} 🚀 BULL RUN MODE: Boosting longs, restricting shorts")

            elif current_regime == 'bear_market':
                # 🐻 BEAR MARKET MODE: Favor shorts, restrict longs
                short_boost = df['rsi'] > 40  # More aggressive short entries
                long_restrict = df['rsi'] < 20  # Only extreme long entries

                enhanced_long_signal = any_long_signal & long_restrict
                enhanced_short_signal = any_short_signal & short_boost

                logger.info(f"{metadata['pair']} 🐻 BEAR MARKET MODE: Boosting shorts, restricting longs")

            elif current_regime == 'high_volatility':
                # ⚡ HIGH VOLATILITY MODE: Only trade extremes
                extreme_oversold = df['rsi'] < 25
                extreme_overbought = df['rsi'] > 75

                enhanced_long_signal = any_long_signal & extreme_oversold
                enhanced_short_signal = any_short_signal & extreme_overbought

                logger.info(f"{metadata['pair']} ⚡ HIGH VOLATILITY MODE: Only extreme entries")

            elif current_regime == 'sideways':
                # 📊 SIDEWAYS MODE: Range trading, both directions OK
                range_long = (df['rsi'] < 35) & (df['close'] <= df.get('[3/8]P', df['close']))
                range_short = (df['rsi'] > 65) & (df['close'] >= df.get('[5/8]P', df['close']))

                enhanced_long_signal = any_long_signal & range_long
                enhanced_short_signal = any_short_signal & range_short

                logger.info(f"{metadata['pair']} 📊 SIDEWAYS MODE: Range trading activated")

            elif current_regime == 'transitional':
                # 🔄 TRANSITIONAL MODE: Be more selective
                quality_long = (df['rsi'] < 40) & (df['volume'] > df['volume'].rolling(20).mean() * 1.2)
                quality_short = (df['rsi'] > 60) & (df['volume'] > df['volume'].rolling(20).mean() * 1.2)

                enhanced_long_signal = any_long_signal & quality_long
                enhanced_short_signal = any_short_signal & quality_short

                logger.info(f"{metadata['pair']} 🔄 TRANSITIONAL MODE: Quality-focused entries")

            else:  # neutral
                # 😐 NEUTRAL MODE: Standard filtering
                enhanced_long_signal = any_long_signal
                enhanced_short_signal = any_short_signal

            # ===========================================
            # FEAR/GREED ADJUSTMENTS
            # ===========================================

            if fear_greed > 75:  # Extreme greed
                # Be more careful with longs, favor shorts
                enhanced_long_signal = enhanced_long_signal & (df['rsi'] < 30)
                logger.info(f"{metadata['pair']} 🤑 EXTREME GREED: Restricting longs")

            elif fear_greed < 25:  # Extreme fear
                # Be more careful with shorts, favor longs
                enhanced_short_signal = enhanced_short_signal & (df['rsi'] > 70)
                logger.info(f"{metadata['pair']} 😨 EXTREME FEAR: Restricting shorts")

            return enhanced_long_signal, enhanced_short_signal

        except Exception as e:
            logger.error(f"{metadata['pair']} ❌ Market regime filter error: {e}")
            # Return original signals on error
            return any_long_signal, any_short_signal
    def calculate_total_market_cap_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate total crypto market cap trend
        FIXED: Handles futures format
        """
        # Detect futures mode
        is_futures = ':' in metadata['pair']

        if is_futures:
            settlement = metadata['pair'].split(':')[1]
            # Futures weighted pairs
            major_coins = {
                f"BTC/USDT:{settlement}": 0.4,
                f"ETH/USDT:{settlement}": 0.2,
                f"BNB/USDT:{settlement}": 0.1,
                f"SOL/USDT:{settlement}": 0.05,
                f"ADA/USDT:{settlement}": 0.05,
            }
        else:
            # Spot pairs
            major_coins = {
                "BTC/USDT": 0.4,
                "ETH/USDT": 0.2,
                "BNB/USDT": 0.1,
                "SOL/USDT": 0.05,
                "ADA/USDT": 0.05,
            }

        weighted_trend = 0
        total_weight = 0

        for coin_pair, weight in major_coins.items():
            try:
                coin_data, _ = self.dp.get_analyzed_dataframe(coin_pair, self.timeframe)
                if coin_data.empty or len(coin_data) < self.total_mcap_ma_period.value:
                    # Try spot equivalent if futures not found
                    if is_futures and ':' in coin_pair:
                        spot_pair = coin_pair.split(':')[0]
                        coin_data, _ = self.dp.get_analyzed_dataframe(spot_pair, self.timeframe)
                        if coin_data.empty:
                            continue
                    else:
                        continue

                # Calculate trend using MA
                ma_period = self.total_mcap_ma_period.value
                current_price = coin_data['close'].iloc[-1]
                ma_value = coin_data['close'].rolling(ma_period).mean().iloc[-1]

                # Trend strength
                trend_strength = (current_price - ma_value) / ma_value
                weighted_trend += trend_strength * weight
                total_weight += weight

            except Exception as e:
                logger.debug(f"Could not process {coin_pair} for mcap trend: {e}")
                continue

        if total_weight > 0:
            mcap_trend = weighted_trend / total_weight
        else:
            mcap_trend = 0

        # Classify trend
        if mcap_trend > 0.05:
            mcap_status = 'bullish'
        elif mcap_trend < -0.05:
            mcap_status = 'bearish'
        else:
            mcap_status = 'neutral'

        dataframe['mcap_trend'] = mcap_trend
        dataframe['mcap_status'] = mcap_status

        return dataframe

    def apply_correlation_filters(self, dataframe: pd.DataFrame, direction: str = 'long') -> pd.Series:
        """
        🔧 UPDATED: Apply market correlation filters with enhanced breadth
        """
        conditions = pd.Series(True, index=dataframe.index)

        # Enhanced BTC correlation filter
        if self.enable_btc_correlation.value and 'btc_correlation_ok' in dataframe.columns:
            conditions &= dataframe['btc_correlation_ok']

            # Optional: More restrictive based on BTC trend score
            if 'btc_trend_score' in dataframe.columns:
                if direction == 'long':
                    btc_long_ok = (
                        (dataframe['btc_trend_score'] >= 2) |
                        (dataframe.get('rsi', 50) < 25)
                    )
                    conditions &= btc_long_ok
                else:
                    btc_short_ok = (
                        (dataframe['btc_trend_score'] <= 3) |
                        (dataframe.get('rsi', 50) > 75)
                    )
                    conditions &= btc_short_ok

        # 🔧 UPDATED: Enhanced market breadth filter
        if self.market_breadth_enabled.value and 'market_breadth' in dataframe.columns:
            if hasattr(self, 'apply_adaptive_breadth_filter'):
                # Use adaptive breadth thresholds
                breadth_conditions = self.apply_adaptive_breadth_filter(dataframe, direction)
                conditions &= breadth_conditions
            else:
                # Fallback to original logic
                if direction == 'long':
                    conditions &= (dataframe['market_breadth'] > self.market_breadth_threshold.value)
                else:
                    conditions &= (dataframe['market_breadth'] < (1 - self.market_breadth_threshold.value))

        # Market cap trend filter (keep existing)
        if self.total_mcap_filter_enabled.value and 'mcap_status' in dataframe.columns:
            if direction == 'long':
                conditions &= (dataframe['mcap_status'] != 'bearish')
            else:
                conditions &= (dataframe['mcap_status'] != 'bullish')

        # 🔧 UPDATED: Market regime filter (aligned names)
        if self.regime_filter_enabled.value and 'market_regime' in dataframe.columns:
            # Avoid trading in high volatility regimes
            conditions &= (dataframe['market_regime'] != 'high_volatility')

            # Additional regime-based filtering
            if direction == 'long':
                # Avoid shorts in bull runs, be cautious in bear markets
                bear_market_caution = (
                    (dataframe['market_regime'] != 'bear_market') |
                    (dataframe.get('rsi', 50) < 25)  # Exception for oversold
                )
                conditions &= bear_market_caution
            else:
                # Avoid shorts in bull runs
                bull_run_caution = (
                    (dataframe['market_regime'] != 'bull_run') |
                    (dataframe.get('rsi', 50) > 75)  # Exception for overbought
                )
                conditions &= bull_run_caution

        return conditions

    def calculate_enhanced_trend_filters(self, dataframe: pd.DataFrame) -> pd.DataFrame:
            """
            Enhanced trend detection to avoid choppy market entries
            """

            # Multi-timeframe trend alignment
            dataframe['ema_5'] = ta.EMA(dataframe['close'], timeperiod=5)
            dataframe['ema_13'] = ta.EMA(dataframe['close'], timeperiod=13)
            dataframe['ema_21'] = ta.EMA(dataframe['close'], timeperiod=21)
            dataframe['ema_50'] = ta.EMA(dataframe['close'], timeperiod=50)

            # Trend alignment score
            dataframe['trend_alignment_bull'] = (
                (dataframe['close'] > dataframe['ema_5']) &
                (dataframe['ema_5'] > dataframe['ema_13']) &
                (dataframe['ema_13'] > dataframe['ema_21']) &
                (dataframe['ema_21'] > dataframe['ema_50'])
            ).astype(int)

            dataframe['trend_alignment_bear'] = (
                (dataframe['close'] < dataframe['ema_5']) &
                (dataframe['ema_5'] < dataframe['ema_13']) &
                (dataframe['ema_13'] < dataframe['ema_21']) &
                (dataframe['ema_21'] < dataframe['ema_50'])
            ).astype(int)

            # ADX for trend strength
            dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
            dataframe['strong_trend'] = dataframe['adx'] > 25
            dataframe['very_strong_trend'] = dataframe['adx'] > 40

            # Momentum indicators
            dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = ta.MACD(
                dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )

            # MACD momentum alignment
            dataframe['macd_bull'] = (
                (dataframe['macd'] > dataframe['macd_signal']) &
                (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
            )

            dataframe['macd_bear'] = (
                (dataframe['macd'] < dataframe['macd_signal']) &
                (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
            )

            # Price action patterns
            dataframe['higher_highs'] = (
                (dataframe['high'] > dataframe['high'].shift(1)) &
                (dataframe['high'].shift(1) > dataframe['high'].shift(2))
            )

            dataframe['lower_lows'] = (
                (dataframe['low'] < dataframe['low'].shift(1)) &
                (dataframe['low'].shift(1) < dataframe['low'].shift(2))
            )

            # Higher lows (bullish structure)
            dataframe['higher_lows'] = (
                (dataframe['low'] > dataframe['low'].shift(1)) &
                (dataframe['low'].shift(1) > dataframe['low'].shift(2))
            )

            # Lower highs (bearish structure)
            dataframe['lower_highs'] = (
                (dataframe['high'] < dataframe['high'].shift(1)) &
                (dataframe['high'].shift(1) < dataframe['high'].shift(2))
            )

            # Volatility filter using Bollinger Bands
            dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
                dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
            )

            # BB squeeze detection (low volatility)
            dataframe['bb_squeeze'] = (
                (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle'] < 0.1
            )

            # BB expansion (high volatility breakout)
            dataframe['bb_expansion'] = (
                (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle'] > 0.2
            )

            # Market regime classification
            dataframe['trending_up'] = (
                dataframe['trend_alignment_bull'] &
                dataframe['strong_trend'] &
                dataframe['macd_bull'] &
                (~dataframe['bb_squeeze'])
            )

            dataframe['trending_down'] = (
                dataframe['trend_alignment_bear'] &
                dataframe['strong_trend'] &
                dataframe['macd_bear'] &
                (~dataframe['bb_squeeze'])
            )

            dataframe['choppy_market'] = (
                (dataframe['adx'] < 20) |
                dataframe['bb_squeeze'] |
                (
                    (~dataframe['trend_alignment_bull']) &
                    (~dataframe['trend_alignment_bear'])
                )
            )

            # Market structure breaks
            dataframe['structure_break_bull'] = (
                (dataframe['close'] > dataframe['high'].rolling(20).max().shift(1)) &
                dataframe['trending_up'] &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2)
            )

            dataframe['structure_break_bear'] = (
                (dataframe['close'] < dataframe['low'].rolling(20).min().shift(1)) &
                dataframe['trending_down'] &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2)
            )

            # Trend momentum strength
            dataframe['trend_momentum'] = dataframe['close'].pct_change(10)
            dataframe['trend_momentum_strength'] = abs(dataframe['trend_momentum'])

            # Consistent trend detection (multiple timeframes)
            dataframe['short_term_trend'] = np.where(
                dataframe['close'] > dataframe['ema_5'], 1,
                np.where(dataframe['close'] < dataframe['ema_5'], -1, 0)
            )

            dataframe['medium_term_trend'] = np.where(
                dataframe['close'] > dataframe['ema_21'], 1,
                np.where(dataframe['close'] < dataframe['ema_21'], -1, 0)
            )

            dataframe['long_term_trend'] = np.where(
                dataframe['close'] > dataframe['ema_50'], 1,
                np.where(dataframe['close'] < dataframe['ema_50'], -1, 0)
            )

            # Trend consistency score
            dataframe['trend_consistency'] = (
                dataframe['short_term_trend'] +
                dataframe['medium_term_trend'] +
                dataframe['long_term_trend']
            ) / 3

            # Strong consistent trends
            dataframe['strong_consistent_uptrend'] = dataframe['trend_consistency'] > 0.6
            dataframe['strong_consistent_downtrend'] = dataframe['trend_consistency'] < -0.6

            # Trend change detection
            dataframe['trend_change_up'] = (
                (dataframe['trend_consistency'] > 0) &
                (dataframe['trend_consistency'].shift(1) <= 0)
            )

            dataframe['trend_change_down'] = (
                (dataframe['trend_consistency'] < 0) &
                (dataframe['trend_consistency'].shift(1) >= 0)
            )

            # Support and resistance levels based on EMAs
            dataframe['dynamic_resistance'] = np.maximum.reduce([
                dataframe['ema_5'], dataframe['ema_13'],
                dataframe['ema_21'], dataframe['ema_50']
            ])

            dataframe['dynamic_support'] = np.minimum.reduce([
                dataframe['ema_5'], dataframe['ema_13'],
                dataframe['ema_21'], dataframe['ema_50']
            ])

            # Price position relative to dynamic levels
            dataframe['above_all_emas'] = (
                (dataframe['close'] > dataframe['ema_5']) &
                (dataframe['close'] > dataframe['ema_13']) &
                (dataframe['close'] > dataframe['ema_21']) &
                (dataframe['close'] > dataframe['ema_50'])
            )

            dataframe['below_all_emas'] = (
                (dataframe['close'] < dataframe['ema_5']) &
                (dataframe['close'] < dataframe['ema_13']) &
                (dataframe['close'] < dataframe['ema_21']) &
                (dataframe['close'] < dataframe['ema_50'])
            )

            # Momentum divergence detection
            # Price vs RSI divergence
            price_peaks = dataframe['high'].rolling(10).max()
            price_troughs = dataframe['low'].rolling(10).min()

            dataframe['bullish_divergence'] = (
                (dataframe['low'] == price_troughs) &
                (dataframe['low'] < dataframe['low'].shift(10)) &
                (dataframe['rsi'] > dataframe['rsi'].shift(10))
            )

            dataframe['bearish_divergence'] = (
                (dataframe['high'] == price_peaks) &
                (dataframe['high'] > dataframe['high'].shift(10)) &
                (dataframe['rsi'] < dataframe['rsi'].shift(10))
            )

            return dataframe
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends
        """
        # Linear regression slope
        def calc_slope(series, period=10):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.iloc[-period:].values
            if np.isnan(y).any():
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope

        # Calculate trend strength using multiple timeframes
        df['slope_5'] = df['close'].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        df['slope_10'] = df['close'].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        df['slope_20'] = df['close'].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)

        # Normalize slopes by price
        df['trend_strength_5'] = df['slope_5'] / df['close'] * 100
        df['trend_strength_10'] = df['slope_10'] / df['close'] * 100
        df['trend_strength_20'] = df['slope_20'] / df['close'] * 100

        # Combined trend strength
        df['trend_strength'] = (df['trend_strength_5'] + df['trend_strength_10'] + df['trend_strength_20']) / 3

        # Trend classification
        strong_up_threshold = self.trend_strength_threshold.value
        strong_down_threshold = -self.trend_strength_threshold.value

        df['strong_uptrend'] = df['trend_strength'] > strong_up_threshold
        df['strong_downtrend'] = df['trend_strength'] < strong_down_threshold
        df['ranging'] = (df['trend_strength'].abs() < strong_up_threshold * 0.5)

        return df
    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:

        # ===========================================
        # STAKE LIMITS DEFINIEREN
        # ===========================================

        # Maximaler Stake pro Trade (in USDT) - kannst du anpassen
        MAX_STAKE_PER_TRADE = self.max_stake_per_trade

        # Maximaler Stake basierend auf Portfolio
        try:
            total_portfolio = self.wallets.get_total_stake_amount()
            MAX_STAKE_PERCENTAGE = self.max_portfolio_percentage_per_trade
            max_stake_from_portfolio = total_portfolio * MAX_STAKE_PERCENTAGE
        except:
            # Fallback wenn wallets nicht verfügbar
            max_stake_from_portfolio = MAX_STAKE_PER_TRADE
            total_portfolio = 1000.0  # Dummy value

        # Market condition check für volatility-based stake reduction (DEIN CODE)
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1]
            current_volatility = last_candle.get("volatility", 0.02)

            # Reduziere Stake in hochvolatilen Märkten
            if current_volatility > 0.05:  # 5% ATR/Price ratio
                volatility_reduction = min(0.5, current_volatility * 10)  # Max 50% reduction
                proposed_stake *= (1 - volatility_reduction)
                logger.info(f"{pair} Stake reduced by {volatility_reduction:.1%} due to high volatility ({current_volatility:.2%})")

        # DCA Multiplier Berechnung (DEIN CODE)
        calculated_max_dca_multiplier = 1.0
        if self.position_adjustment_enable:
            num_safety_orders = int(self.max_safety_orders.value)
            volume_scale = self.safety_order_volume_scale.value
            if num_safety_orders > 0 and volume_scale > 0:
                current_order_relative_size = 1.0
                for _ in range(num_safety_orders):
                    current_order_relative_size *= volume_scale
                    calculated_max_dca_multiplier += current_order_relative_size
            else:
                logger.warning(f"{pair}: Could not calculate max_dca_multiplier due to "
                               f"invalid max_safety_orders ({num_safety_orders}) or "
                               f"safety_order_volume_scale ({volume_scale}). Defaulting to 1.0.")
        else:
            logger.debug(f"{pair}: Position adjustment not enabled. max_dca_multiplier is 1.0.")

        if calculated_max_dca_multiplier > 0:
            stake_amount = proposed_stake / calculated_max_dca_multiplier

            # ===========================================
            # NEUE STAKE LIMITS ANWENDEN
            # ===========================================

            # Verschiedene Limits prüfen
            final_stake = min(
                stake_amount,
                MAX_STAKE_PER_TRADE,
                max_stake_from_portfolio,
                max_stake  # Freqtrade's max_stake
            )

            # Bestimme welches Limit gegriffen hat
            limit_reason = "calculated"
            if final_stake == MAX_STAKE_PER_TRADE:
                limit_reason = "max_per_trade"
            elif final_stake == max_stake_from_portfolio:
                limit_reason = "portfolio_percentage"
            elif final_stake == max_stake:
                limit_reason = "freqtrade_max"

            logger.info(f"{pair} Initial stake calculated: {final_stake:.8f} (Proposed: {proposed_stake:.8f}, "
                        f"Calculated Max DCA Multiplier: {calculated_max_dca_multiplier:.2f}, "
                        f"Limited by: {limit_reason}, Portfolio %: {(final_stake/total_portfolio)*100:.1f}%)")

            # Min stake prüfen (DEIN CODE)
            if min_stake is not None and final_stake < min_stake:
                logger.info(f"{pair} Initial stake {final_stake:.8f} was below min_stake {min_stake:.8f}. "
                            f"Adjusting to min_stake. Consider tuning your DCA parameters or proposed stake.")
                final_stake = min_stake

            return final_stake
        else:
            # Fallback (DEIN CODE)
            logger.warning(
                f"{pair} Calculated max_dca_multiplier is {calculated_max_dca_multiplier:.2f}, which is invalid. "
                f"Using proposed_stake: {proposed_stake:.8f}")
            return proposed_stake

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        🎯 IMPROVED: Wait for better entry prices to reduce immediate stop loss risk
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return proposed_rate

        last_candle = dataframe.iloc[-1]

        # Calculate volatility-adjusted entry
        atr = last_candle.get('atr', 0)
        if atr == 0:
            atr = (last_candle['high'] - last_candle['low'])

        # Get MML levels for better entry timing
        mml_25 = last_candle.get('[2/8]P', last_candle['close'])
        mml_50 = last_candle.get('[4/8]P', last_candle['close'])
        mml_75 = last_candle.get('[6/8]P', last_candle['close'])

        if side == "long":
            # For longs: Try to enter closer to support or on pullbacks
            if entry_tag in ["MML_50_Reclaim", "MML_Support_Bounce"]:
                # Enter slightly above support with buffer
                support_buffer = atr * 0.2  # 20% of ATR buffer
                optimal_entry = max(mml_25, mml_50) + support_buffer
                entry_price = min(proposed_rate, optimal_entry)
            elif entry_tag == "MML_Bullish_Breakout":
                # For breakouts: Enter on slight pullback, not at peak
                pullback_entry = proposed_rate - (atr * 0.1)  # 10% ATR pullback
                entry_price = max(pullback_entry, proposed_rate * 0.999)  # Max 0.1% below
            else:
                # Conservative entry for other signals
                entry_price = proposed_rate - (atr * 0.05)  # Small discount

        elif side == "short":
            # For shorts: Try to enter closer to resistance
            if entry_tag in ["MML_50_Breakdown", "MML_Resistance_Reject"]:
                # Enter slightly below resistance with buffer
                resistance_buffer = atr * 0.2
                optimal_entry = min(mml_75, mml_50) - resistance_buffer
                entry_price = max(proposed_rate, optimal_entry)
            elif entry_tag == "MML_Bearish_Breakdown":
                # For breakdowns: Enter on slight bounce, not at bottom
                bounce_entry = proposed_rate + (atr * 0.1)
                entry_price = min(bounce_entry, proposed_rate * 1.001)
            else:
                entry_price = proposed_rate + (atr * 0.05)
        else:
            entry_price = proposed_rate

        # Ensure entry price is reasonable
        max_deviation = proposed_rate * 0.005  # Max 0.5% from proposed
        if side == "long":
            entry_price = max(entry_price, proposed_rate - max_deviation)
        else:
            entry_price = min(entry_price, proposed_rate + max_deviation)

        logger.info(f"{pair} Optimized entry: {entry_price:.6f} (was {proposed_rate:.6f}, tag: {entry_tag})")
        return entry_price

    # ═══════════════════════════════════════════════════════════════
    # 🛑 CRITICAL FIX: TIME-BASED STOP LOSS LOOSENING
    # ═══════════════════════════════════════════════════════════════

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        🔧 REVOLUTIONARY: Time-based stop loss loosening + MML support

        Based on your backtest results:
        - ROI hits at ~1.5 hours with 4% profit
        - Stop loss hits at ~1.25 hours with -6.5% loss
        - Need to give trades more time to develop before stopping out
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty or 'atr' not in dataframe.columns or len(dataframe) < 10:
            logger.warning(f"{pair} Insufficient data for dynamic stop loss. Using conservative default: -0.03")
            return -0.03  # Conservative default

        last_candle = dataframe.iloc[-1]
        last_atr = last_candle.get("atr", 0)

        # Handle missing or invalid ATR
        if pd.isna(last_atr) or last_atr == 0:
            valid_atr = dataframe["atr"].dropna()
            if not valid_atr.empty:
                last_atr = valid_atr.rolling(window=5).mean().iloc[-1]
                logger.info(f"{pair} Using smoothed ATR: {last_atr:.8f}")
            else:
                logger.warning(f"{pair} No valid ATR found. Using fallback stop loss -0.03")
                return -0.03

        if last_atr == 0 or current_rate == 0:
            logger.warning(f"{pair} ATR or rate is 0. Using fallback stop loss -0.03")
            return -0.03

        # ⏰ TIME IN TRADE CALCULATION
        time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600  # Hours

        # Get MML levels for dynamic support/resistance
        if trade.is_short:
            # For shorts: Use MML resistance as dynamic stop
            resistance_level = last_candle.get('[6/8]P', current_rate * 1.02)
            if pd.isna(resistance_level):
                resistance_level = current_rate * 1.02
            mml_stop_distance = abs((resistance_level - current_rate) / current_rate)
        else:
            # For longs: Use MML support as dynamic stop
            support_level = last_candle.get('[2/8]P', current_rate * 0.98)
            if pd.isna(support_level):
                support_level = current_rate * 0.98
            mml_stop_distance = abs((current_rate - support_level) / current_rate)

        # Base ATR stop loss calculation
        base_atr_stop = (last_atr / current_rate) * 2.0  # Start with 2x ATR

        # 🕐 CRITICAL: TIME-BASED STOP LOSS LOOSENING
        # Based on your data: Need more room in first 1.5 hours
        if time_in_trade < 0.25:  # First 15 minutes - tightest (but not too tight)
            time_multiplier = 1.2  # Give some room immediately
        elif time_in_trade < 0.5:  # 15-30 minutes - slight loosening
            time_multiplier = 1.5
        elif time_in_trade < 1.0:  # 30-60 minutes - more room
            time_multiplier = 2.0
        elif time_in_trade < 1.5:  # 1-1.5 hours - CRITICAL PERIOD
            time_multiplier = 2.5  # Much more room where stop losses currently hit
        elif time_in_trade < 2.0:  # 1.5-2 hours - ROI development time
            time_multiplier = 3.0  # Maximum room for ROI to develop
        elif time_in_trade < 4.0:  # 2-4 hours - long-term holds
            time_multiplier = 2.8  # Slight tightening but still wide
        else:  # 4+ hours - prevent runaway losses
            time_multiplier = 2.5

        # 📈 PROFIT-BASED ADJUSTMENTS (override time loosening when profitable)
        if current_profit > 0.08:  # 8%+ profit - lock in gains aggressively
            profit_stop = max(-0.015, -(base_atr_stop * 0.3))  # Very tight
        elif current_profit > 0.06:  # 6%+ profit - good protection
            profit_stop = max(-0.02, -(base_atr_stop * 0.5))
        elif current_profit > 0.04:  # 4%+ profit - moderate protection
            profit_stop = max(-0.025, -(base_atr_stop * 0.7))
        elif current_profit > 0.02:  # 2%+ profit - light protection
            profit_stop = max(-0.03, -(base_atr_stop * 1.0))
        elif current_profit > 0:  # Any profit - don't tighten
            profit_stop = -(base_atr_stop * time_multiplier)
        else:
            # Apply time-based loosening for losing trades
            profit_stop = -(base_atr_stop * time_multiplier)

        # 🎯 MML-BASED STOP LOSS ADJUSTMENT
        # Use MML levels as natural stop areas, but don't make stops too wide
        mml_adjusted_stop = min(
            abs(profit_stop),  # Don't make tighter than profit-based
            max(mml_stop_distance * 1.3, 0.02),  # At least 2%, max 30% past MML
            0.10  # Never wider than 10%
        )

        # 🛡️ SAFETY BOUNDARIES
        # Ensure stop loss isn't too tight or too wide
        if current_profit <= 0:
            # For losing trades: Ensure minimum room based on time
            if time_in_trade < 1.5:  # Critical period - be generous
                min_stop = -0.08  # Minimum 8% room
            else:
                min_stop = -0.06  # Standard minimum

            final_stop = -max(mml_adjusted_stop, abs(min_stop))
        else:
            # For profitable trades: Use calculated stop
            final_stop = -mml_adjusted_stop

        # Final safety check: Never tighter than -1.5% unless big profit
        if final_stop > -0.015 and current_profit < 0.06:
            final_stop = -0.015

        # Never wider than -12%
        final_stop = max(final_stop, -0.12)

        # 📊 DETAILED LOGGING
        logger.info(f"{pair} 🛑 DYNAMIC STOP LOSS:")
        logger.info(f"   ⏰ Time in trade: {time_in_trade:.2f}h")
        logger.info(f"   📈 Current profit: {current_profit:.2%}")
        logger.info(f"   🔢 Time multiplier: {time_multiplier}x")
        logger.info(f"   💰 ATR base: {base_atr_stop:.4f}")
        logger.info(f"   🎯 MML distance: {mml_stop_distance:.4f}")
        logger.info(f"   🛡️ Final stop: {final_stop:.4f}")

        return final_stop

    def minimal_roi_market_adjusted(self, pair: str, trade: Trade, current_time: datetime,
                                   current_rate: float, current_profit: float) -> Dict[int, float]:
        """
        Dynamically adjust ROI based on market conditions
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.minimal_roi

        last_candle = dataframe.iloc[-1]
        market_score = last_candle.get('market_score', 0.5)
        market_regime = last_candle.get('market_regime', 'normal')

        # Copy original ROI
        adjusted_roi = self.minimal_roi.copy()

        # In high volatility or bear market, take profits earlier
        if market_regime == 'high_volatility' or market_score < 0.3:
            # Reduce all ROI targets by 20%
            adjusted_roi = {k: v * 0.8 for k, v in adjusted_roi.items()}
            logger.info(f"{pair} ROI adjusted down due to market conditions")

        # In strong bull market, let winners run
        elif market_score > 0.7 and market_regime == 'strong_trend':
            # Increase ROI targets by 20%
            adjusted_roi = {k: v * 1.2 for k, v in adjusted_roi.items()}
            logger.info(f"{pair} ROI adjusted up due to bullish market")

        return adjusted_roi

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float, current_entry_profit: float,
                              current_exit_profit: float, **kwargs) -> Optional[float]:

        # Get market conditions
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if not dataframe.empty:
            last_candle = dataframe.iloc[-1]
            btc_trend = last_candle.get('btc_trend', 0)
            market_score = last_candle.get('market_score', 0.5)
            market_breadth = last_candle.get('market_breadth', 0.5)
            rsi = last_candle.get('rsi', 50)
        else:
            market_score = 0.5
            market_breadth = 0.5
            rsi = 50

        count_of_entries = trade.nr_of_successful_entries
        count_of_exits = trade.nr_of_successful_exits

        # === ENHANCED PROFIT TAKING BASED ON MARKET CONDITIONS ===

        # More aggressive profit taking in overbought market
        if market_score > 0.75 and current_profit > 0.15 and count_of_exits == 0:
            logger.info(f"{trade.pair} Taking profit early due to market greed: {market_score:.2f}")
            amount_to_sell = (trade.amount * current_rate) * 0.33  # Sell 33%
            return -amount_to_sell

        # Original profit taking logic (keep as is)
        if current_profit > 0.25 and count_of_exits == 0:
            logger.info(f"{trade.pair} Taking partial profit (25%) at {current_profit:.2%}")
            amount_to_sell = (trade.amount * current_rate) * 0.25
            return -amount_to_sell

        # === BEAR MARKET PROFIT TAKING ===
        # Nimm Gewinne schneller mit im Bärenmarkt
        if not trade.is_short and btc_trend < 0:  # Long im Downtrend
            if current_profit > 0.10 and count_of_exits == 0:  # Statt 0.25
                logger.info(f"{trade.pair} Bear market quick profit taking at {current_profit:.2%}")
                amount_to_sell = (trade.amount * current_rate) * 0.5  # 50% verkaufen
                return -amount_to_sell

        if current_profit > 0.40 and count_of_exits == 1:
            logger.info(f"{trade.pair} Taking additional profit (33%) at {current_profit:.2%}")
            amount_to_sell = (trade.amount * current_rate) * (1 / 3)
            return -amount_to_sell

        # === 🔧 ENHANCED DCA LOGIC WITH STRICT CONTROLS ===
        if not self.position_adjustment_enable:
            return None

        # 🛑 USE STRATEGY VARIABLES FOR DCA LIMITS
        max_dca_for_pair = self.max_dca_orders
        max_total_stake = self.max_total_stake_per_pair
        max_single_dca = self.max_single_dca_amount

        # 🛑 CHECK: Already too many DCA orders?
        if count_of_entries > max_dca_for_pair:
            logger.info(f"{trade.pair} 🛑 MAX DCA REACHED: {count_of_entries}/{max_dca_for_pair}")
            return None

        # 🛑 CHECK: Total stake amount already too high?
        if trade.stake_amount >= max_total_stake:
            logger.info(f"{trade.pair} 🛑 MAX STAKE REACHED: {trade.stake_amount:.2f}/{max_total_stake} USDT")
            return None

        # Block DCA in crashing market
        if market_breadth < 0.25 and trade.is_short == False:
            logger.info(f"{trade.pair} Blocking DCA due to bearish market breadth: {market_breadth:.2%}")
            return None

        # More conservative DCA triggers in volatile markets
        if dataframe.iloc[-1].get('market_regime') == 'high_volatility':
            dca_multiplier = 1.5  # Require 50% deeper drawdown
        else:
            dca_multiplier = 1.0

        # Apply multiplier to original DCA logic
        trigger = self.initial_safety_order_trigger.value * dca_multiplier

        if (current_profit > trigger / 2.0 and count_of_entries == 1) or \
           (current_profit > trigger and count_of_entries == 2) or \
           (current_profit > trigger * 1.5 and count_of_entries == 3):
            logger.info(f"{trade.pair} DCA condition not met. Current profit {current_profit:.2%} above threshold")
            return None

        # 🛑 CHECK: Original max safety orders (falls du das noch nutzt)
        if hasattr(self, 'max_safety_orders') and count_of_entries >= self.max_safety_orders.value + 1:
            logger.info(f"{trade.pair} 🛑 Original max_safety_orders reached: {count_of_entries}")
            return None

        try:
            filled_entry_orders = trade.select_filled_orders(trade.entry_side)
            if not filled_entry_orders:
                logger.error(f"{trade.pair} No filled entry orders found for DCA calculation")
                return None

            last_order_cost = filled_entry_orders[-1].cost

            # 🔧 USE STRATEGY VARIABLE FOR DCA SIZING
            base_dca_amount = max_single_dca  # Use strategy variable

            # Progressive DCA sizing (each DCA gets smaller!)
            dca_multipliers = [1.0, 0.8, 0.6]  # 1st: 5 USDT, 2nd: 4 USDT, 3rd: 3 USDT

            if count_of_entries <= len(dca_multipliers):
                current_multiplier = dca_multipliers[count_of_entries - 1]
            else:
                current_multiplier = 0.5  # Fallback für unerwartete Orders

            # Calculate DCA amount
            dca_stake_amount = base_dca_amount * current_multiplier

            # 🛑 HARD CAP: Never exceed remaining budget
            remaining_budget = max_total_stake - trade.stake_amount
            if dca_stake_amount > remaining_budget:
                if remaining_budget > 1:  # Only proceed if at least 1 USDT remaining
                    dca_stake_amount = remaining_budget
                    logger.info(f"{trade.pair} 🔧 DCA capped to remaining budget: {dca_stake_amount:.2f} USDT")
                else:
                    logger.info(f"{trade.pair} 🛑 Insufficient remaining budget: {remaining_budget:.2f} USDT")
                    return None

            # Standard min/max stake checks
            if min_stake is not None and dca_stake_amount < min_stake:
                logger.warning(f"{trade.pair} DCA below min_stake. Adjusting to {min_stake:.2f} USDT")
                dca_stake_amount = min_stake

            if max_stake is not None and (trade.stake_amount + dca_stake_amount) > max_stake:
                available_for_dca = max_stake - trade.stake_amount
                if available_for_dca > (min_stake or 0):
                    dca_stake_amount = available_for_dca
                    logger.warning(f"{trade.pair} DCA reduced due to max_stake: {dca_stake_amount:.2f} USDT")
                else:
                    logger.warning(f"{trade.pair} Cannot DCA due to max_stake limit")
                    return None

            # 🔧 FINAL SAFETY CHECK
            new_total_stake = trade.stake_amount + dca_stake_amount
            if new_total_stake > max_total_stake:
                logger.error(f"{trade.pair} 🚨 SAFETY VIOLATION: Would exceed max total stake!")
                return None

            logger.info(f"{trade.pair} ✅ DCA #{count_of_entries}: +{dca_stake_amount:.2f} USDT "
                       f"(Total: {new_total_stake:.2f}/{max_total_stake} USDT)")

            return dca_stake_amount

        except IndexError:
            logger.error(f"Error calculating DCA stake for {trade.pair}: IndexError accessing last_order")
            return None
        except Exception as e:
            logger.error(f"Error calculating DCA stake for {trade.pair}: {e}")
            return None

    # def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
    #              max_leverage: float, side: str, **kwargs) -> float:
    #     window_size = self.leverage_window_size.value
    #     dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    #     if len(dataframe) < window_size:
    #         logger.warning(
    #             f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
    #         return proposed_leverage
    #     close_prices_series = dataframe["close"].tail(window_size)
    #     high_prices_series = dataframe["high"].tail(window_size)
    #     low_prices_series = dataframe["low"].tail(window_size)
    #     base_leverage = self.leverage_base.value
    #     rsi_array = ta.RSI(close_prices_series, timeperiod=14)
    #     atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
    #     sma_array = ta.SMA(close_prices_series, timeperiod=20)
    #     macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

    #     current_rsi = rsi_array[-1] if rsi_array.size > 0 and not np.isnan(rsi_array[-1]) else 50.0
    #     current_atr = atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
    #     current_sma = sma_array[-1] if sma_array.size > 0 and not np.isnan(sma_array[-1]) else current_rate
    #     current_macd_hist = 0.0

    #     if isinstance(macd_output, pd.DataFrame):
    #         if not macd_output.empty and 'macdhist' in macd_output.columns:
    #             valid_macdhist_series = macd_output['macdhist'].dropna()
    #             if not valid_macdhist_series.empty:
    #                 current_macd_hist = valid_macdhist_series.iloc[-1]

    #     # Apply rules based on indicators
    #     if side == "long":
    #         if current_rsi < self.leverage_rsi_low.value:
    #             base_leverage *= self.leverage_long_increase_factor.value
    #         elif current_rsi > self.leverage_rsi_high.value:
    #             base_leverage *= self.leverage_long_decrease_factor.value

    #         if current_atr > 0 and current_rate > 0:
    #             if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
    #                 base_leverage *= self.leverage_volatility_decrease_factor.value

    #         if current_macd_hist > 0:
    #             base_leverage *= self.leverage_long_increase_factor.value

    #         if current_sma > 0 and current_rate < current_sma:
    #             base_leverage *= self.leverage_long_decrease_factor.value

    #     adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
    #     logger.info(
    #         f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
    #         f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
    #     return adjusted_leverage

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔧 FIXED: Enhanced indicator calculations with proper method calls
        """
        dataframe['actual_stoploss'] = np.nan
        dataframe['follow_price_level'] = np.nan

        # ===========================================
        # YOUR EXISTING INDICATORS (KEEP ALL)
        # ===========================================

        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # Extrema detection (your existing code)
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # Heikin-Ashi and rolling extrema (your existing code)
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, self.h2.value)
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(dataframe, self.h1.value)
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(dataframe, self.h0.value)
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(dataframe, self.cp.value)

        # Murrey Math levels (your existing code)
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(dataframe, window_size=mml_window)
        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # MML oscillator (your existing code)
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")

        if mml_4_8 is not None and mml_plus_3_8 is not None and mml_minus_3_8 is not None:
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - mml_4_8) / osc_denominator)
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # DI Catch and checks (your existing code)
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).max()

        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)

        # Your existing volatility calculations
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = dataframe["volatility_range"].rolling(window=20).mean()
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=20).mean()

        # ===========================================
        # 🚀 ENHANCED MARKET CORRELATION (PROPER ORDER)
        # ===========================================

        # Enhanced BTC correlation FIRST
        if self.enable_btc_correlation.value:
            logger.info(f"🔍 {metadata['pair']} Calculating enhanced BTC correlation...")
            dataframe = self.enhanced_btc_correlation_filter(dataframe, metadata)

        # Legacy BTC correlation (keep for compatibility)
        if self.btc_correlation_enabled.value or self.btc_trend_filter_enabled.value:
            logger.info(f"📈 {metadata['pair']} Calculating legacy BTC correlation...")
            dataframe = self.calculate_market_correlation_simple(dataframe, metadata)

        # 🔧 FIXED: Market breadth calculation BEFORE regime (breadth affects regime scoring)
        if self.market_breadth_enabled.value:
            logger.info(f"📊 {metadata['pair']} Calculating weighted market breadth...")
            dataframe = self.calculate_market_breadth_weighted(dataframe, metadata)

            # Debug breadth calculation
            if len(dataframe) > 0:
                breadth = dataframe['market_breadth'].iloc[-1]
                pairs_checked = dataframe.get('breadth_pairs_checked', [0]).iloc[-1]
                logger.info(f"🎯 {metadata['pair']} Breadth result: {breadth:.2%} ({pairs_checked} pairs)")

        # 🔧 FIXED: Market regime calculation AFTER breadth
        if self.regime_filter_enabled.value:
            logger.info(f"🏛️ {metadata['pair']} Calculating market regime...")

            # 🛠️ DEBUG: Check parameter availability
            if hasattr(self, 'regime_lookback_period'):
                lookback = self.regime_lookback_period.value
                logger.info(f"📊 {metadata['pair']} Using lookback period: {lookback}")
            else:
                logger.error(f"🚨 {metadata['pair']} Missing regime_lookback_period parameter!")
                # Add fallback
                self.regime_lookback_period = IntParameter(24, 72, default=48, space="buy", optimize=True)
                lookback = 48

            # Call the regime calculation
            dataframe = self.calculate_market_regime_aligned(dataframe, metadata)

            # Debug regime calculation
            if len(dataframe) > 0:
                regime = dataframe['market_regime'].iloc[-1]
                volatility = dataframe.get('market_volatility', [0.02]).iloc[-1]
                adx = dataframe.get('market_adx', [25]).iloc[-1]
                logger.info(f"🎯 {metadata['pair']} Regime result: {regime} (vol: {volatility:.3f}, adx: {adx:.1f})")

        # Market cap trend (keep existing)
        if self.total_mcap_filter_enabled.value:
            logger.info(f"💰 {metadata['pair']} Calculating market cap trend...")
            dataframe = self.calculate_total_market_cap_trend(dataframe, metadata)

        # ===========================================
        # NEW: ENHANCED SIGNAL FILTERING INDICATORS
        # ===========================================

        # Enhanced trend detection
        try:
            dataframe = self.calculate_enhanced_trend_filters(dataframe)
            logger.debug(f"✅ {metadata['pair']} Enhanced trend filters calculated")
        except Exception as e:
            logger.warning(f"⚠️ {metadata['pair']} Enhanced trend filters failed: {e}")

        # Market state detection
        try:
            dataframe = self.detect_market_state(dataframe)
            logger.debug(f"✅ {metadata['pair']} Market state detected")
        except Exception as e:
            logger.warning(f"⚠️ {metadata['pair']} Market state detection failed: {e}")

        # Signal strength calculation
        try:
            dataframe = self.calculate_signal_strength(dataframe)
            logger.debug(f"✅ {metadata['pair']} Signal strength calculated")
        except Exception as e:
            logger.warning(f"⚠️ {metadata['pair']} Signal strength calculation failed: {e}")

        # ===========================================
        # 🔧 ENHANCED MARKET SCORE CALCULATION
        # ===========================================

        # Base market score
        dataframe['market_score'] = 0.5

        # BTC correlation component
        if 'btc_correlation' in dataframe.columns:
            dataframe['market_score'] += dataframe['btc_correlation'] * 0.15

        if 'btc_correlation_ok' in dataframe.columns:
            # Enhanced BTC correlation gets higher weight
            dataframe['market_score'] += dataframe['btc_correlation_ok'].astype(float) * 0.20

        # Market breadth component (higher weight due to weighting)
        if 'market_breadth' in dataframe.columns:
            breadth_contribution = (dataframe['market_breadth'] - 0.5) * 0.25
            dataframe['market_score'] += breadth_contribution

        # Market cap trend component
        if 'mcap_trend' in dataframe.columns:
            dataframe['market_score'] += dataframe['mcap_trend'] * 0.10

        # Market regime component (aligned names)
        if 'market_regime' in dataframe.columns:
            regime_scores = {
                'bull_run': 0.85,        # Strong bullish
                'bear_market': 0.15,     # Strong bearish
                'sideways': 0.50,        # Neutral ranging
                'high_volatility': 0.30, # Caution - volatile
                'transitional': 0.50,    # Neutral transition
                'neutral': 0.50          # Default neutral
            }

            dataframe['regime_score'] = dataframe['market_regime'].map(
                lambda x: regime_scores.get(x, 0.5)
            )
            dataframe['market_score'] += (dataframe['regime_score'] - 0.5) * 0.15

        # Enhanced trend alignment (if available)
        if 'trend_alignment_bull' in dataframe.columns:
            dataframe['market_score'] += dataframe['trend_alignment_bull'] * 0.10
            dataframe['market_score'] -= dataframe['trend_alignment_bear'] * 0.10

        # MACD momentum (if available)
        if 'macd_bull' in dataframe.columns:
            dataframe['market_score'] += dataframe['macd_bull'].astype(int) * 0.05
            dataframe['market_score'] -= dataframe['macd_bear'].astype(int) * 0.05

        # Volume confirmation component
        if 'volume_confirmation' in dataframe.columns:
            volume_contribution = (dataframe['volume_confirmation'] - 0.5) * 0.10
            dataframe['market_score'] += volume_contribution

        # Ensure market_score stays within bounds
        dataframe['market_score'] = dataframe['market_score'].clip(0, 1)

        # ===========================================
        # SIGNAL QUALITY INDICATORS
        # ===========================================

        # Signal persistence
        dataframe['signal_persistence'] = 0
        for window in [3, 5, 10]:
            dataframe[f'bullish_persistence_{window}'] = (
                (dataframe['rsi'] < 50).rolling(window).sum() / window
            )
            dataframe[f'bearish_persistence_{window}'] = (
                (dataframe['rsi'] > 50).rolling(window).sum() / window
            )

        # Volume momentum
        dataframe['volume_momentum'] = (
            dataframe['volume'].rolling(3).mean() /
            dataframe['volume'].rolling(20).mean()
        ).fillna(1.0)

        # Price momentum strength
        dataframe['price_momentum_strength'] = abs(
            dataframe['close'].pct_change(5)
        ).rolling(10).mean()

        # ===========================================
        # 🔧 BREADTH-SPECIFIC INDICATORS
        # ===========================================

        # Breadth momentum (rate of change)
        if 'market_breadth' in dataframe.columns:
            dataframe['breadth_momentum'] = dataframe['market_breadth'].diff()
            dataframe['breadth_trend'] = dataframe['market_breadth'].rolling(5).mean()

            # Breadth divergence detection
            price_direction = (dataframe['close'] > dataframe['close'].shift(5)).astype(int)
            breadth_direction = (dataframe['market_breadth'] > 0.5).astype(int)
            dataframe['breadth_price_divergence'] = abs(price_direction - breadth_direction)

        # Market regime strength
        if 'market_adx' in dataframe.columns:
            dataframe['regime_strength'] = dataframe['market_adx'] / 50  # Normalize ADX
            dataframe['strong_regime'] = dataframe['market_adx'] > 30

        # ===========================================
        # 📊 FINAL LOGGING & VALIDATION
        # ===========================================

        try:
            # Log final market conditions
            if len(dataframe) > 0:
                last_row = dataframe.iloc[-1]

                market_breadth = last_row.get('market_breadth', 0.5)
                market_regime = last_row.get('market_regime', 'unknown')
                market_score = last_row.get('market_score', 0.5)
                btc_ok = last_row.get('btc_correlation_ok', True)

                logger.info(f"📊 FINAL MARKET CONDITIONS for {metadata['pair']}:")
                logger.info(f"   🌐 Breadth: {market_breadth:.2%}")
                logger.info(f"   🏛️ Regime: {market_regime}")
                logger.info(f"   📈 Score: {market_score:.2%}")
                logger.info(f"   ₿ BTC OK: {btc_ok}")

                # 🔍 REGIME DEBUGGING
                if market_regime == 'unknown':
                    logger.error(f"🚨 {metadata['pair']} REGIME STILL UNKNOWN!")
                    logger.error(f"   Check regime calculation method")
                    logger.error(f"   Lookback period: {getattr(self, 'regime_lookback_period', 'MISSING')}")

            else:
                logger.error(f"🚨 {metadata['pair']} Empty dataframe after indicators!")

        except Exception as e:
            logger.error(f"💥 {metadata['pair']} Final logging error: {e}")

        return dataframe

    def enhanced_btc_correlation_filter(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🛡️ FIXED: Proper SMA periods for 30min timeframe BTC correlation
        """

        pair = metadata['pair']

        # Skip BTC correlation for BTC pairs
        if 'BTC' in pair:
            logger.info(f"{pair} 🟡 Skipping BTC correlation for BTC pair")
            dataframe['btc_correlation_ok'] = True
            dataframe['btc_trend_score'] = 5  # Max score
            dataframe['btc_status'] = 'BTC_PAIR'
            return dataframe

        try:
            # 🔧 DETERMINE BTC PAIR TO USE
            current_pair = metadata['pair']
            if ':' in current_pair:
                settlement = current_pair.split(':')[1]
                btc_pair = f'BTC/USDT:{settlement}'
                fallback_pair = 'BTC/USDT'
            else:
                btc_pair = 'BTC/USDT'
                fallback_pair = 'BTC/USD'

            # 🚀 TRY TO GET BTC DATA
            btc_dataframe = None
            btc_pair_used = None

            # Try primary BTC pair
            try:
                btc_dataframe = self.dp.get_pair_dataframe(btc_pair, self.timeframe)
                if btc_dataframe is not None and not btc_dataframe.empty and len(btc_dataframe) >= 48:  # Need more candles for 30min
                    btc_pair_used = btc_pair
                    logger.info(f"✅ {pair} Got BTC data from {btc_pair} ({len(btc_dataframe)} candles)")
                else:
                    raise Exception(f"Insufficient data from {btc_pair}")

            except Exception as e:
                logger.warning(f"⚠️ {pair} Primary BTC pair failed: {e}")

                # Try fallback pair
                try:
                    btc_dataframe = self.dp.get_pair_dataframe(fallback_pair, self.timeframe)
                    if btc_dataframe is not None and not btc_dataframe.empty and len(btc_dataframe) >= 48:
                        btc_pair_used = fallback_pair
                        logger.info(f"✅ {pair} Got BTC data from fallback {fallback_pair} ({len(btc_dataframe)} candles)")
                    else:
                        raise Exception(f"Insufficient fallback data from {fallback_pair}")

                except Exception as e2:
                    logger.error(f"💥 {pair} Both BTC pairs failed: {e2}")
                    btc_dataframe = None

            # 🚨 FALLBACK: If no BTC data, use current pair analysis
            if btc_dataframe is None or btc_dataframe.empty:
                logger.warning(f"🔄 {pair} No BTC data, analyzing current pair trend")

                # Use current pair momentum as BTC proxy
                current_momentum = dataframe['close'].pct_change(5).iloc[-1]
                current_rsi = dataframe.get('rsi', pd.Series([50])).iloc[-1]

                if current_momentum > 0.01 or current_rsi < 40:  # Bullish momentum or oversold
                    btc_correlation_ok = True
                    btc_trend_score = 4
                elif current_momentum < -0.02 and current_rsi > 60:  # Bearish momentum and overbought
                    btc_correlation_ok = False
                    btc_trend_score = 1
                else:
                    btc_correlation_ok = True  # Default to allow trades
                    btc_trend_score = 3

                dataframe['btc_correlation_ok'] = btc_correlation_ok
                dataframe['btc_trend_score'] = btc_trend_score
                dataframe['btc_status'] = 'PAIR_PROXY'

                logger.info(f"{pair} 🔄 Using pair proxy: BTC OK = {btc_correlation_ok}")
                return dataframe

            # 📊 CALCULATE BTC INDICATORS WITH PROPER 30MIN TIMEFRAME PERIODS
            try:
                # 🔧 FIXED: Adjusted SMA periods for 30min timeframe
                # Original SMA20 (20 * 30min = 10 hours) -> Use SMA24 (24 * 30min = 12 hours = half day)
                # Original SMA50 (50 * 30min = 25 hours) -> Use SMA48 (48 * 30min = 24 hours = 1 day)

                btc_sma24 = btc_dataframe['close'].rolling(24).mean().iloc[-1]  # 12 hours
                btc_sma48 = btc_dataframe['close'].rolling(48).mean().iloc[-1]  # 24 hours
                btc_close = btc_dataframe['close'].iloc[-1]

                # 🔧 FIXED: Adjusted momentum period for 30min timeframe
                # Original 5 period (5 * 30min = 2.5 hours) -> Use 8 periods (8 * 30min = 4 hours)
                btc_momentum = btc_dataframe['close'].pct_change(8).iloc[-1]  # 4 hour momentum

                # Simple BTC trend analysis with proper periods
                btc_signals = 0

                # Signal 1: Above shorter term SMA (12 hours)
                if not pd.isna(btc_sma24) and btc_close > btc_sma24:
                    btc_signals += 2  # Above SMA24 (12h)
                    logger.debug(f"{pair} BTC above SMA24: ${btc_close:.0f} > ${btc_sma24:.0f}")

                # Signal 2: Positive 4-hour momentum
                if not pd.isna(btc_momentum) and btc_momentum > 0:
                    btc_signals += 1  # Positive 4h momentum
                    logger.debug(f"{pair} BTC positive 4h momentum: {btc_momentum:.3f}")

                # Signal 3: SMA alignment (shorter above longer term)
                if not pd.isna(btc_sma24) and not pd.isna(btc_sma48) and btc_sma24 > btc_sma48:
                    btc_signals += 1  # SMA24 > SMA48 (12h > 24h trend)
                    logger.debug(f"{pair} BTC SMA alignment: 24h>${btc_sma24:.0f} > 48h${btc_sma48:.0f}")

                # Signal 4: Recent volatility check (adjusted for 30min)
                btc_volatility = btc_dataframe['close'].pct_change().rolling(20).std().iloc[-1]  # 10 hour volatility
                if not pd.isna(btc_volatility) and btc_volatility < 0.05:  # Not too volatile
                    btc_signals += 1
                    logger.debug(f"{pair} BTC low volatility: {btc_volatility:.3f}")

                # Signal 5: Price trend over longer period (adjusted)
                if len(btc_dataframe) >= 48:  # 24 hours of data
                    btc_trend_48h = (btc_close - btc_dataframe['close'].iloc[-48]) / btc_dataframe['close'].iloc[-48]
                    if btc_trend_48h > -0.1:  # Not down more than 10% in 24 hours
                        btc_signals += 1
                        logger.debug(f"{pair} BTC 24h trend: {btc_trend_48h:.3f}")

                # 🎯 MORE PERMISSIVE CORRELATION LOGIC
                # Allow trades unless BTC is clearly bearish
                btc_correlation_ok = btc_signals >= 2  # Need at least 2 out of 5 signals

                logger.info(f"{pair} 🔍 BTC ANALYSIS ({btc_pair_used}) - 30min timeframe:")
                logger.info(f"   📊 BTC Price: ${btc_close:.0f}")
                logger.info(f"   📈 SMA24 (12h): ${btc_sma24:.0f}")
                logger.info(f"   📈 SMA48 (24h): ${btc_sma48:.0f}")
                logger.info(f"   🎯 BTC Signals: {btc_signals}/5")
                logger.info(f"   ✅ Correlation OK: {btc_correlation_ok}")

                # Apply to dataframe
                dataframe['btc_correlation_ok'] = btc_correlation_ok
                dataframe['btc_trend_score'] = btc_signals
                dataframe['btc_status'] = f"OK_{btc_pair_used}"
                dataframe['btc_sma24'] = btc_sma24
                dataframe['btc_sma48'] = btc_sma48

                return dataframe

            except Exception as e:
                logger.error(f"{pair} 💥 BTC analysis error: {str(e)}")

                # Final fallback - allow trades
                dataframe['btc_correlation_ok'] = True
                dataframe['btc_trend_score'] = 3
                dataframe['btc_status'] = 'ANALYSIS_ERROR'
                return dataframe

        except Exception as e:
            logger.error(f"{pair} 💥 BTC CORRELATION ERROR: {str(e)}")

            # Emergency fallback - allow trades
            dataframe['btc_correlation_ok'] = True
            dataframe['btc_trend_score'] = 3
            dataframe['btc_status'] = 'ERROR_FALLBACK'
            return dataframe
    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        KORRIGIERT: Saubere Entry-Logic mit sofortigen gegenläufigen Exits + Signal Quality Filter
        """

        # ===========================================
        # INITIALIZE ENTRY COLUMNS FIRST
        # ===========================================
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""
        df["exit_long"] = 0
        df["exit_short"] = 0

        # ===========================================
        # CALCULATE MISSING INDICATORS INLINE (SAFE)
        # ===========================================

        # Calculate green/red candles and consecutive patterns
        df['green_candle'] = (df['close'] > df['open']).astype(int)
        df['red_candle'] = (df['close'] < df['open']).astype(int)
        df['consecutive_green'] = df['green_candle'].rolling(3).sum()
        df['consecutive_red'] = df['red_candle'].rolling(3).sum()

        # Calculate simple trend consistency if missing
        if 'trend_consistency' not in df.columns:
            df['trend_consistency'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)

        # Ensure market_score exists
        if 'market_score' not in df.columns:
            df['market_score'] = 0.5

        # ===========================================
        # MML MARKET STRUCTURE ANALYSIS
        # ===========================================

        # Bullish Market Structure
        bullish_mml = (
            (df["close"] > df["[6/8]P"]) |  # Above 75% = Strong Bull
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))  # Breaking above 50%
        )

        # Bearish Market Structure
        bearish_mml = (
            (df["close"] < df["[2/8]P"]) |  # Below 25% = Strong Bear
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))  # Breaking below 50%
        )

        # Trading Range (Choppy)
        range_bound = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &
            (~bullish_mml) & (~bearish_mml)
        )

        # MML Support/Resistance Bounces
        mml_support_bounce = (
            ((df["low"] <= df["[2/8]P"]) & (df["close"] > df["[2/8]P"])) |  # Bounce from 25%
            ((df["low"] <= df["[4/8]P"]) & (df["close"] > df["[4/8]P"]))   # Bounce from 50%
        )

        mml_resistance_reject = (
            ((df["high"] >= df["[6/8]P"]) & (df["close"] < df["[6/8]P"])) |  # Reject at 75%
            ((df["high"] >= df["[8/8]P"]) & (df["close"] < df["[8/8]P"]))   # Reject at 100%
        )

        # ===========================================
        # LONG ENTRY SIGNALS
        # ===========================================

        long_signal_mml_breakout = (
            (df["close"] > df["[6/8]P"]) &
            (df["close"].shift(1) <= df["[6/8]P"].shift(1)) &  # Fresh breakout
            (df["volume"] > df["volume"].rolling(20).mean() * 1.2) &  # Volume confirmation
            (df["rsi"] > 50) &  # Momentum confirmation
            (df["close"] > df["open"]) &  # Bullish candle
            (df["volume"] > 0)
        )

        long_signal_support_bounce = (
            mml_support_bounce &
            (df["rsi"] < 45) &  # Oversold but not extreme
            (df["close"] > df["close"].shift(1)) &  # Price turning up
            (df["volume"] > df["volume"].rolling(10).mean()) &  # Volume support
            (df["minima"] == 1) &  # Local bottom
            (df["volume"] > 0)
        )

        long_signal_50_reclaim = (
            (df["close"] > df["[4/8]P"]) &
            (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &  # Breaking above 50%
            (df["close"].shift(5) < df["[4/8]P"].shift(5)) &  # Was below 50% recently
            (df["rsi"] > 45) &  # Momentum building
            (df["volume"] > df["volume"].rolling(15).mean() * 1.1) &
            (df["close"] > df["open"]) &  # Bullish candle
            (df["volume"] > 0)
        )

        long_signal_range_long = (
            range_bound &
            (df["low"] <= df["[2/8]P"]) &
            (df["close"] > df["[2/8]P"]) &  # Bounce from 25%
            (df["rsi"] < 40) &  # Oversold
            (df["close"] > df["close"].shift(1)) &  # Turning up
            (df["minima"] == 1) &
            (df["volume"] > 0)
        )

        long_signal_trend_continuation = (
            bullish_mml &
            (df["close"] > df["[5/8]P"]) &  # Above 62.5% - still bullish
            (df["rsi"].between(40, 60)) &  # Healthy pullback
            (df["minima"] == 1) &  # Local bottom
            (df["close"] > df["close"].shift(1)) &  # Turning up
            (df["volume"] > 0)
        )

        long_signal_extreme_oversold = (
            (df["low"] <= df["[1/8]P"]) &
            (df["close"] > df["[1/8]P"]) &  # Bounce from 12.5%
            (df["rsi"] < 30) &  # Oversold
            (df["volume"] > df["volume"].rolling(20).mean() * 1.5) &  # High volume
            (df["close"] > df["close"].shift(1)) &  # Price turning
            (df["minima"] == 1)
        )

        # Combine all Long signals
        any_long_signal = (
            long_signal_mml_breakout |
            long_signal_support_bounce |
            long_signal_50_reclaim |
            long_signal_range_long |
            long_signal_trend_continuation |
            long_signal_extreme_oversold
        )

        # ===========================================
        # SHORT ENTRY SIGNALS (if allowed)
        # ===========================================

        if self.can_short:
            short_signal_bearish_breakdown = (
                (df["close"] < df["[2/8]P"]) &
                (df["close"].shift(1) >= df["[2/8]P"].shift(1)) &  # Fresh breakdown
                (df["volume"] > df["volume"].rolling(20).mean() * 1.2) &  # Volume confirmation
                (df["rsi"] < 50) &  # Momentum confirmation
                (df["close"] < df["open"]) &  # Bearish candle
                (df["volume"] > 0)
            )

            short_signal_resistance_reject = (
                mml_resistance_reject &
                (df["rsi"] > 55) &  # Overbought but not extreme
                (df["close"] < df["close"].shift(1)) &  # Price turning down
                (df["volume"] > df["volume"].rolling(10).mean()) &  # Volume confirmation
                (df["maxima"] == 1) &  # Local top
                (df["volume"] > 0)
            )

            short_signal_50_breakdown = (
                (df["close"] < df["[4/8]P"]) &
                (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &  # Breaking below 50%
                (df["close"].shift(5) > df["[4/8]P"].shift(5)) &  # Was above 50% recently
                (df["rsi"] < 55) &  # Momentum weakening
                (df["volume"] > df["volume"].rolling(15).mean() * 1.1) &
                (df["close"] < df["open"]) &  # Bearish candle
                (df["volume"] > 0)
            )

            short_signal_range_short = (
                range_bound &
                (df["high"] >= df["[6/8]P"]) &
                (df["close"] < df["[6/8]P"]) &  # Reject at 75%
                (df["rsi"] > 60) &  # Overbought
                (df["close"] < df["close"].shift(1)) &  # Turning down
                (df["maxima"] == 1) &
                (df["volume"] > 0)
            )

            short_signal_trend_continuation = (
                bearish_mml &
                (df["close"] < df["[3/8]P"]) &  # Below 37.5% - still bearish
                (df["rsi"].between(40, 60)) &  # Healthy pullback
                (df["maxima"] == 1) &  # Local top
                (df["close"] < df["close"].shift(1)) &  # Turning down
                (df["volume"] > 0)
            )

            short_signal_extreme_overbought = (
                (df["high"] >= df["[7/8]P"]) &
                (df["close"] < df["[7/8]P"]) &  # Reject at 87.5%
                (df["rsi"] > 70) &  # Overbought
                (df["volume"] > df["volume"].rolling(20).mean() * 1.5) &  # High volume
                (df["close"] < df["close"].shift(1)) &  # Price turning
                (df["maxima"] == 1)
            )

            # Combine all Short signals
            any_short_signal = (
                short_signal_bearish_breakdown |
                short_signal_resistance_reject |
                short_signal_50_breakdown |
                short_signal_range_short |
                short_signal_trend_continuation |
                short_signal_extreme_overbought
            )
        else:
            any_short_signal = pd.Series([False] * len(df), index=df.index)

        # ===========================================
        # BASIC MOMENTUM FILTERS
        # ===========================================

        # Block shorts in strong upward momentum
        if self.can_short:
            strong_up_momentum = (
                (df["close"] > df["close"].shift(1)) &
                (df["close"].shift(1) > df["close"].shift(2)) &
                (df["close"].shift(2) > df["close"].shift(3)) &
                (df["rsi"] > 60)
            )
            any_short_signal = any_short_signal & (~strong_up_momentum)

        # Block longs in strong downward momentum
        strong_down_momentum = (
            (df["close"] < df["close"].shift(1)) &
            (df["close"].shift(1) < df["close"].shift(2)) &
            (df["close"].shift(2) < df["close"].shift(3)) &
            (df["rsi"] < 40)
        )
        any_long_signal = any_long_signal & (~strong_down_momentum | (df["rsi"] <= 25))

        # ===========================================
        # 🎯 ENHANCED SIGNAL QUALITY FILTERS
        # ===========================================

        # Calculate ATR ratio for volatility check
        df['atr_ratio'] = df.get('atr', df['high'] - df['low']) / df['close']
        df['atr_ratio'] = df['atr_ratio'].fillna(0.02)  # Default 2% if ATR missing

        # Volume strength calculation
        df['volume_avg_10'] = df['volume'].rolling(10).mean()
        df['volume_avg_20'] = df['volume'].rolling(20).mean()

        # Signal quality filter
        signal_quality_filter = (
            # 1. Volume confirmation - current volume above recent average
            (df['volume'] > df['volume_avg_10']) &

            # 2. Not in extreme volatility (avoid choppy/news-driven moves)
            (df['atr_ratio'] < 0.05) &  # Max 5% ATR ratio

            # 3. Market structure alignment for direction
            (
                # For longs: Price should be above recent support OR extremely oversold
                (
                    (df['close'] > df['close'].rolling(5).min() * 1.005) & any_long_signal
                ) |
                # For shorts: Price should be below recent resistance OR extremely overbought
                (
                    (df['close'] < df['close'].rolling(5).max() * 0.995) & any_short_signal
                ) |
                # Allow extreme RSI conditions regardless of structure
                (df['rsi'] < 25) | (df['rsi'] > 75)
            ) &

            # 4. Avoid signals during volume spikes (news/manipulation)
            (df['volume'] < df['volume_avg_20'] * 3.0) &

            # 5. Price action confirmation - avoid doji/indecision candles
            (abs(df['close'] - df['open']) > (df['high'] - df['low']) * 0.3) &

            # 6. Momentum alignment - price should be moving in signal direction
            (
                # For longs: recent price action should show some upward bias
                (
                    (df['close'] > df['close'].shift(2)) & any_long_signal
                ) |
                # For shorts: recent price action should show some downward bias
                (
                    (df['close'] < df['close'].shift(2)) & any_short_signal
                ) |
                # Exception for extreme reversals
                ((df['rsi'] < 20) & any_long_signal) |
                ((df['rsi'] > 80) & any_short_signal)
            )
        )

        # ===========================================
        # 🚨 ENHANCED TREND PROTECTION FILTERS
        # ===========================================

        # Strong uptrend detection with safe column references
        strong_uptrend = (
            (df['close'] > df['close'].shift(5)) &  # Price up over 5 candles
            (df['close'] > df['close'].shift(10)) & # Price up over 10 candles
            (df['rsi'] > 45) &  # RSI not oversold
            (
                # At least one of these bullish conditions (using safe .get())
                (df['close'] > df.get('[6/8]P', df['close'])) |  # Above 75% MML
                (df.get('market_score', 0.5) > 0.6) |   # Bullish market score
                (df.get('trend_consistency', 0) > 0.3) |  # Trending up
                (df['consecutive_green'] >= 2)  # Multiple green candles
            )
        )

        # Strong downtrend detection with safe column references
        strong_downtrend = (
            (df['close'] < df['close'].shift(5)) &  # Price down over 5 candles
            (df['close'] < df['close'].shift(10)) & # Price down over 10 candles
            (df['rsi'] < 55) &  # RSI not overbought
            (
                # At least one of these bearish conditions (using safe .get())
                (df['close'] < df.get('[2/8]P', df['close'])) |  # Below 25% MML
                (df.get('market_score', 0.5) < 0.4) |   # Bearish market score
                (df.get('trend_consistency', 0) < -0.3) |  # Trending down
                (df['consecutive_red'] >= 2)   # Multiple red candles
            )
        )

        # Extreme market conditions
        extremely_bullish = (
            (df['close'] > df.get('[7/8]P', df['close'] * 1.05)) &  # Above 87.5%
            (df['rsi'] > 60) &
            (df.get('market_score', 0.5) > 0.7) &
            (df['volume'] > df['volume'].rolling(10).mean() * 1.2)
        )

        extremely_bearish = (
            (df['close'] < df.get('[1/8]P', df['close'] * 0.95)) &  # Below 12.5%
            (df['rsi'] < 40) &
            (df.get('market_score', 0.5) < 0.3) &
            (df['volume'] > df['volume'].rolling(10).mean() * 1.2)
        )

        # ===========================================
        # 🛑 APPLY TREND FILTERS TO SIGNALS
        # ===========================================

        # BLOCK SHORTS in strong uptrends (unless extremely overbought)
        short_trend_filter = (
            (~strong_uptrend) |  # Not in strong uptrend
            extremely_bearish |   # OR extremely bearish setup
            (df['rsi'] > 80)     # OR extremely overbought (reversal play)
        )

        # BLOCK LONGS in strong downtrends (unless extremely oversold)
        long_trend_filter = (
            (~strong_downtrend) |  # Not in strong downtrend
            extremely_bullish |    # OR extremely bullish setup
            (df['rsi'] < 20)      # OR extremely oversold (reversal play)
        )

        # ===========================================
        # 🔧 BTC CORRELATION FILTER (if available)
        # ===========================================

        if 'btc_trend' in df.columns:
            # If BTC is strongly bullish, be very careful with shorts
            btc_bullish_filter = (
                (df['btc_trend'] != 1) |  # BTC not bullish
                (df['rsi'] > 75) |        # OR we're extremely overbought
                (df['close'] < df.get('[3/8]P', df['close']))  # OR we're in lower MML region
            )

            # If BTC is strongly bearish, be careful with longs
            btc_bearish_filter = (
                (df['btc_trend'] != -1) |  # BTC not bearish
                (df['rsi'] < 25) |         # OR we're extremely oversold
                (df['close'] > df.get('[5/8]P', df['close']))  # OR we're in upper MML region
            )

            # Apply BTC filters
            short_trend_filter &= btc_bullish_filter
            long_trend_filter &= btc_bearish_filter

        # ===========================================
        # 🎯 APPLY ALL FILTERS AND CREATE FINAL SIGNALS
        # ===========================================

        # Apply all filters to create final signals
        final_long_signal = any_long_signal & long_trend_filter & signal_quality_filter

        if self.can_short:
            final_short_signal = any_short_signal & short_trend_filter & signal_quality_filter
        else:
            final_short_signal = pd.Series([False] * len(df), index=df.index)

        # ===========================================
        # 🔧 HANDLE SIGNAL CONFLICTS
        # ===========================================

        if self.can_short:
            conflicting_signals = final_long_signal & final_short_signal

            # In conflict: Favor longs when oversold, shorts when overbought
            final_short_signal = final_short_signal & (~conflicting_signals | (df["rsi"] >= 50))
            final_long_signal = final_long_signal & (~conflicting_signals | (df["rsi"] < 50))

        # ===========================================
        # SET FINAL ENTRY SIGNALS AND TAGS
        # ===========================================

        # Set Long signals
        df.loc[final_long_signal, "enter_long"] = 1

        # Set Long tags (priority order)
        df.loc[final_long_signal & long_signal_mml_breakout, "enter_tag"] = "MML_Bullish_Breakout"
        df.loc[final_long_signal & long_signal_support_bounce & (df["enter_tag"] == ""), "enter_tag"] = "MML_Support_Bounce"
        df.loc[final_long_signal & long_signal_50_reclaim & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Reclaim"
        df.loc[final_long_signal & long_signal_range_long & (df["enter_tag"] == ""), "enter_tag"] = "MML_Range_Long"
        df.loc[final_long_signal & long_signal_trend_continuation & (df["enter_tag"] == ""), "enter_tag"] = "MML_Trend_Continuation_Long"
        df.loc[final_long_signal & long_signal_extreme_oversold & (df["enter_tag"] == ""), "enter_tag"] = "MML_Extreme_Oversold_Long"

        # Set Short signals (if allowed)
        if self.can_short:
            df.loc[final_short_signal, "enter_short"] = 1

            # Set Short tags (priority order)
            df.loc[final_short_signal & short_signal_bearish_breakdown, "enter_tag"] = "MML_Bearish_Breakdown"
            df.loc[final_short_signal & short_signal_resistance_reject & (df["enter_tag"] == ""), "enter_tag"] = "MML_Resistance_Reject"
            df.loc[final_short_signal & short_signal_50_breakdown & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Breakdown"
            df.loc[final_short_signal & short_signal_range_short & (df["enter_tag"] == ""), "enter_tag"] = "MML_Range_Short"
            df.loc[final_short_signal & short_signal_trend_continuation & (df["enter_tag"] == ""), "enter_tag"] = "MML_Trend_Continuation_Short"
            df.loc[final_short_signal & short_signal_extreme_overbought & (df["enter_tag"] == ""), "enter_tag"] = "MML_Extreme_Overbought_Short"

        # ===========================================
        # 📊 LOGGING FOR DEBUGGING
        # ===========================================

        try:
            # Count filtered signals for analysis
            quality_blocked_longs = any_long_signal & long_trend_filter & (~signal_quality_filter)
            quality_blocked_shorts = any_short_signal & short_trend_filter & (~signal_quality_filter)

            trend_blocked_shorts = any_short_signal & (~short_trend_filter)
            trend_blocked_longs = any_long_signal & (~long_trend_filter)

            # Log filtering statistics
            if trend_blocked_shorts.any():
                logger.info(f"{metadata['pair']} 🛑 TREND BLOCKED {trend_blocked_shorts.sum()} short signals")

            if trend_blocked_longs.any():
                logger.info(f"{metadata['pair']} 🛑 TREND BLOCKED {trend_blocked_longs.sum()} long signals")

            if quality_blocked_longs.any():
                logger.info(f"{metadata['pair']} 🔍 QUALITY BLOCKED {quality_blocked_longs.sum()} long signals")

            if quality_blocked_shorts.any():
                logger.info(f"{metadata['pair']} 🔍 QUALITY BLOCKED {quality_blocked_shorts.sum()} short signals")

            # Log final signal counts
            total_longs = final_long_signal.sum()
            total_shorts = final_short_signal.sum()

            if total_longs > 0 or total_shorts > 0:
                logger.info(f"{metadata['pair']} ✅ FINAL SIGNALS: {total_longs} longs, {total_shorts} shorts")

        except Exception as e:
            logger.warning(f"Logging error in {metadata['pair']}: {e}")

        return df

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🚪 ENHANCED EXIT: Take profits at optimal levels
        """
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''

        # ===========================================
        # PROFIT TARGET EXITS
        # ===========================================

        # Calculate approximate profit levels using price action
        # (This is simplified - in reality, use trade object in custom methods)

        # Strong resistance/support exits
        if '[6/8]P' in dataframe.columns and '[2/8]P' in dataframe.columns:
            mml_75 = dataframe['[6/8]P']
            mml_25 = dataframe['[2/8]P']

            # Exit shorts near strong support
            strong_support_exit = (
                (dataframe['close'] <= mml_25 * 1.005) &  # Near 25% level
                (dataframe['rsi'] < 30) &  # Oversold
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2)
            )

            # Exit longs near strong resistance
            strong_resistance_exit = (
                (dataframe['close'] >= mml_75 * 0.995) &  # Near 75% level
                (dataframe['rsi'] > 70) &  # Overbought
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2)
            )

            dataframe.loc[strong_support_exit, 'exit_short'] = 1
            dataframe.loc[strong_support_exit, 'exit_tag'] = 'mml_target'

            dataframe.loc[strong_resistance_exit, 'exit_long'] = 1
            dataframe.loc[strong_resistance_exit, 'exit_tag'] = 'mml_target'

        # ===========================================
        # MOMENTUM DIVERGENCE EXITS
        # ===========================================

        # Exit when momentum diverges (simple version)
        price_momentum = dataframe['close'].pct_change(5)
        rsi_momentum = dataframe['rsi'].pct_change(5)

        # Bearish divergence (price up, RSI down)
        bearish_divergence = (
            (price_momentum > 0.02) &  # Price rising
            (rsi_momentum < -2) &      # RSI falling
            (dataframe['rsi'] > 65)    # From overbought
        )

        # Bullish divergence (price down, RSI up)
        bullish_divergence = (
            (price_momentum < -0.02) &  # Price falling
            (rsi_momentum > 2) &        # RSI rising
            (dataframe['rsi'] < 35)     # From oversold
        )

        dataframe.loc[bearish_divergence, 'exit_long'] = 1
        dataframe.loc[bearish_divergence, 'exit_tag'] = 'divergence'

        dataframe.loc[bullish_divergence, 'exit_short'] = 1
        dataframe.loc[bullish_divergence, 'exit_tag'] = 'divergence'

        return dataframe

    def should_exit_trade(self, pair: str, trade: 'Trade', current_time: datetime,
                          current_rate: float, current_profit: float, **kwargs) -> bool:
        """
        🔧 IMPROVED: Only flip if profitable or strong signal
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return False

            last_candle = dataframe.iloc[-1]
            is_long_trade = trade.trade_direction == 'long'
            is_short_trade = trade.trade_direction == 'short'

            # 🚨 NEW: Only flip if conditions are met
            can_flip = (
                current_profit > 0.01 or  # At least 1% profit
                current_profit < -0.04 or  # OR significant loss (cut losses)
                last_candle.get('rsi', 50) > 75 or  # OR extremely overbought
                last_candle.get('rsi', 50) < 25     # OR extremely oversold
            )

            if not can_flip:
                return False

            # Check for opposite signals
            if is_long_trade and last_candle.get('enter_short', 0):
                logger.info(f"🔄 {pair}: LONG→SHORT flip (Profit: {current_profit:.2%})")
                return True
            if is_short_trade and last_candle.get('enter_long', 0):
                logger.info(f"🔄 {pair}: SHORT→LONG flip (Profit: {current_profit:.2%})")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ {pair}: Error in should_exit_trade: {e}")
            return False

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        current_profit_ratio = trade.calc_profit_ratio(rate)
        time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600

        logger.warning(f"🚪 {pair} EXIT REQUEST: {exit_reason}")
        logger.warning(f"   💰 Profit: {current_profit_ratio:.4f} ({current_profit_ratio*100:.2f}%)")
        logger.warning(f"   ⏰ Time in trade: {time_in_trade:.1f} hours")

        if exit_reason in ["force_exit", "force_sell", "emergency_exit"]:
            logger.error(f"{pair} 🛑 BLOCKED FORCE EXIT: {exit_reason} (Profit: {current_profit_ratio:.2%})")
            return False

        if exit_reason in ["timeout", "protection", "cooldown"]:
            logger.error(f"{pair} 🛑 BLOCKED PROTECTION EXIT: {exit_reason}")
            return False

        if exit_reason == "roi":
            logger.info(f"{pair} ✅ ROI EXIT: {current_profit_ratio:.2%} after {time_in_trade:.1f}h")
            return True

        if exit_reason in ["stop_loss", "exit_signal", "sell_signal", "custom_exit", "trailing_stop_loss", "trailing_stop"]:
            logger.info(f"{pair} ✅ EXIT: {exit_reason} ({current_profit_ratio:.2%})")
            return True

        logger.info(f"{pair} ✅ FINAL CHECK PASSED: Profit {current_profit_ratio:.2%}")
        return True
    # 🔍 ADD THIS METHOD HERE (anywhere in the class)

    def should_exit_early_warning(self, pair: str, trade: Trade, current_time: datetime,
                                  current_rate: float, current_profit: float) -> bool:
        """
        ⚠️ EARLY WARNING: Exit before stop loss if conditions deteriorate
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return False

            last_candle = dataframe.iloc[-1]
            time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600

            # Don't exit too early
            if time_in_trade < 0.25:  # Give trade at least 15 minutes
                return False

            # Critical deterioration signals
            rsi = last_candle.get('rsi', 50)

            if trade.is_short:
                # Short position warning signs
                warning_signals = (
                    (rsi < 30 and current_profit < -0.02) or  # RSI turning bullish while losing
                    (last_candle.get('close', 0) > last_candle.get('[6/8]P', 0)) or  # Breaking above 75%
                    (current_profit < -0.04 and time_in_trade > 2)  # Deep loss after 2h
                )
            else:
                # Long position warning signs
                warning_signals = (
                    (rsi > 70 and current_profit < -0.02) or  # RSI turning bearish while losing
                    (last_candle.get('close', 0) < last_candle.get('[2/8]P', 0)) or  # Breaking below 25%
                    (current_profit < -0.04 and time_in_trade > 2)  # Deep loss after 2h
                )

            if warning_signals:
                logger.warning(f"⚠️ {pair} EARLY WARNING: Deteriorating conditions detected")
                return True

            return False

        except Exception as e:
            logger.error(f"Early warning error for {pair}: {e}")
            return False

    def validate_backtest_conditions(self, dataframe: pd.DataFrame, metadata: dict) -> None:
        """
        📊 VALIDATION: Check if backtest conditions match live trading
        """

        pair = metadata['pair']

        # Check BTC correlation status
        if 'btc_status' in dataframe.columns:
            btc_status_counts = dataframe['btc_status'].value_counts()
            total_candles = len(dataframe)

            logger.warning(f"{pair} 📊 BACKTEST VALIDATION REPORT:")
            logger.warning(f"   Total candles: {total_candles}")

            for status, count in btc_status_counts.items():
                percentage = (count / total_candles) * 100
                logger.warning(f"   BTC Status '{status}': {count} candles ({percentage:.1f}%)")

            # Alert if significant portion has no BTC data
            no_data_percentage = btc_status_counts.get('NO_DATA', 0) / total_candles * 100
            if no_data_percentage > 10:
                logger.error(f"{pair} 🚨 WARNING: {no_data_percentage:.1f}% of backtest has no BTC data!")
                logger.error(f"{pair} 🚨 BACKTEST RESULTS MAY NOT MATCH LIVE TRADING!")

            # Count signal reduction due to BTC filter
            if 'btc_correlation_ok' in dataframe.columns:
                btc_blocked = (~dataframe['btc_correlation_ok']).sum()
                btc_blocked_pct = (btc_blocked / total_candles) * 100
                logger.info(f"{pair} 🛡️ BTC filter blocked: {btc_blocked} candles ({btc_blocked_pct:.1f}%)")

        else:
            logger.error(f"{pair} 🚨 NO BTC CORRELATION DATA IN BACKTEST!")
            logger.error(f"{pair} 🚨 BACKTEST IS DEFINITELY NOT MATCHING LIVE CONDITIONS!")
    def bot_loop_start(self, **kwargs) -> None:
        """
        🔍 Log any trades that might be candidates for force exit
        """
        try:
            trades = Trade.get_open_trades()
            current_time = datetime.now()

            for trade in trades:
                time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600
                current_profit = trade.calc_profit_ratio(trade.close_rate) if trade.close_rate else 0

                # Warn about very old trades (might become force exits)
                if time_in_trade > 48:  # 2+ days
                    logger.warning(f"⚠️ OLD TRADE {trade.pair}: {time_in_trade:.1f}h old, "
                                 f"Profit: {current_profit:.2%}")

                # Warn about large losses (might become force exits)
                if current_profit < -0.10:  # >10% loss
                    logger.warning(f"⚠️ LARGE LOSS {trade.pair}: {current_profit:.2%}, "
                                 f"Time: {time_in_trade:.1f}h")

        except Exception as e:
            # Don't let debugging break the bot
            logger.debug(f"bot_loop_start debug error: {e}")
