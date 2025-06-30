"""
ZaratustraDCA5 Trading Strategy

A sophisticated momentum-based trading strategy optimized for 5-minute timeframes.
Features dynamic position sizing, market correlation analysis, and advanced risk management.

NOTE: This strategy is under development, and some functions may be disabled or incomplete. Parameters like ROI
targets and advanced stoploss logic are subject to future optimization. Test thoroughly before using it in live trading.

Telegram Profile: https://t.me/bustillo

Choose your coffee style:
- BTC (Classic): bc1qfq46qqhurg8ps73506rtqsr26mfhl9t6vp2ltc
- ETH/ERC-20 & BSC/BEP-20 (Smart): 0x486Ef431878e2a240ea2e7A6EBA42e74632c265c
  (Supports ETH, BNB, USDT, and tokens on: Ethereum, Binance Smart Chain, and EVM-compatible networks.)
- SOL (Speed): 2nrYABUJLjHtUdVTXkcY8ELUK7q3HH4iWXQxQMQDdZa8
- XMR (Privacy): 45kQh8n23AgiY2yEDbMmJdcMGTaHmpn6vFfhECs7EwtPZ7pbyCQAyzDCehtDZSGsWzaDGir1LfA4EGDQP3dtPStsMdrzUG5

Strategy Overview:
-----------------
This strategy combines ADX-based directional movement analysis with market breadth filtering
and BTC correlation to identify high-probability momentum trades. It employs dynamic position
adjustment (DCA) and ATR-based risk management for optimal risk-reward ratios.

Key Features:
- Bidirectional trading (long/short)
- Dynamic position sizing with DCA
- Market correlation filtering
- Murrey Math level confirmation
- ATR-based dynamic stop-loss
- Market breadth analysis

Technical Foundation:
- Primary: ADX family indicators (ADX, PDI, MDI, DX)
- Supporting: RSI, SMA, ATR, Volume analysis
- Timeframe optimized: 5-minute charts
- Market filters: BTC correlation, market breadth, Murrey levels

Risk Management:
- Dynamic stop-loss: 3-10% based on ATR
- Maximum DCA entries: 2 additional positions
- Position adjustment based on trend strength
- Force exit protection for unprofitable trades
"""

import logging
import pandas as pd
import numpy as np
from technical import qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Dict
import talib.abstract as ta
from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter, BooleanParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class ZaratustraDCA5(IStrategy):
    """
    ZaratustraDCA5 - Advanced Momentum Strategy for 5-Minute Timeframes

    This strategy implements a sophisticated trading system that combines:
    1. ADX-based directional movement analysis for momentum detection
    2. Market correlation filtering using BTC trend analysis
    3. Market breadth monitoring across major cryptocurrency pairs
    4. Murrey Math levels for support/resistance confirmation
    5. Dynamic position adjustment with intelligent DCA
    6. ATR-based adaptive risk management

    The strategy is specifically optimized for 5-minute timeframes, providing
    faster signal generation and more aggressive profit-taking compared to
    longer timeframe versions.

    Entry Conditions:
    - Long: DX crosses above PDI with ADX > MDI, trend filters passed
    - Short: DX crosses above MDI with ADX > PDI, trend filters passed

    Risk Management:
    - Dynamic stop-loss based on ATR (1.5x multiplier)
    - Progressive DCA at -3%, -5%, -7% levels
    - Partial profit taking at 8% gain
    - Market condition awareness for position adjustments
    """

    # ==================== STRATEGY CONFIGURATION ====================

    # Core strategy settings
    exit_profit_only = True                    # Only exit when profitable
    ignore_roi_if_entry_signal = True         # Ignore ROI if entry signal active
    can_short = True                          # Enable short positions
    use_exit_signal = True                    # Use exit signals
    use_custom_stoploss = True                # Enable dynamic stop-loss
    stoploss = -0.08                          # Base stop-loss: 8% (tighter for 5m)
    timeframe = '5m'                          # Optimized timeframe
    position_adjustment_enable = True          # Enable DCA functionality
    max_entry_position_adjustment = 2         # Maximum 2 additional entries

    # ==================== RETURN ON INVESTMENT (ROI) ====================
    """
    Aggressive ROI structure optimized for 5-minute trading.
    Targets faster profits due to increased opportunity frequency.
    """
    minimal_roi = {
        "0": 0.04,      # 4% immediate target
        "5": 0.035,     # 3.5% after 5 minutes
        "10": 0.03,     # 3% after 10 minutes
        "20": 0.025,    # 2.5% after 20 minutes
        "30": 0.02,     # 2% after 30 minutes
        "60": 0.015,    # 1.5% after 1 hour
        "120": 0.01,    # 1% after 2 hours
        "240": 0.005,   # 0.5% after 4 hours
        "480": 0        # Break-even after 8 hours
    }

    # ==================== STRATEGY PARAMETERS ====================
    """
    Optimizable parameters for strategy fine-tuning.
    All parameters have been adjusted for 5-minute timeframe characteristics.
    """

    # ADX and directional movement parameters
    adx_high_multiplier = DecimalParameter(
        0.3, 0.7, default=0.5, space='buy', optimize=True,
        load=True, decimals=2
    )  # ATR multiplier for strong trend identification

    adx_low_multiplier = DecimalParameter(
        0.1, 0.5, default=0.3, space='buy', optimize=True,
        load=True, decimals=2
    )  # ATR multiplier for weak trend identification

    adx_minimum = IntParameter(
        15, 25, default=18, space="buy", optimize=True,
        load=True
    )  # Minimum ADX threshold (lowered for 5m sensitivity)

    # Protection parameters
    cooldown_lookback = IntParameter(
        2, 48, default=3, space="protection", optimize=True,
        load=True
    )  # Cooldown period after losses (3 candles = 15 minutes)

    stop_duration = IntParameter(
        12, 120, default=48, space="protection", optimize=True,
        load=True
    )  # Stop-loss guard duration (48 candles = 4 hours)

    use_stop_protection = BooleanParameter(
        default=True, space="protection", optimize=True,
        load=True
    )  # Enable/disable stop-loss guard

    # Market correlation parameters
    btc_correlation_enabled = BooleanParameter(
        default=True, space="buy", optimize=True,
        load=True
    )  # Enable BTC correlation analysis

    btc_trend_filter = BooleanParameter(
        default=True, space="buy", optimize=True,
        load=True
    )  # Apply BTC trend filtering to entries

    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(
        default=True, space="buy", optimize=True,
        load=True
    )  # Enable market breadth analysis

    market_breadth_threshold = DecimalParameter(
        0.3, 0.7, default=0.45, space="buy", optimize=True,
        load=True, decimals=2
    )  # Market breadth threshold for trade filtering

    # Murrey Math parameters
    use_murrey_confirmation = BooleanParameter(
        default=True, space="buy", optimize=True,
        load=True
    )  # Enable Murrey Math level confirmation

    murrey_buffer = DecimalParameter(
        0.001, 0.01, default=0.004, space="buy", optimize=True,
        load=True, decimals=4
    )  # Buffer for Murrey level proximity detection

    @property
    def protections(self):
        """
        Define protection mechanisms to prevent overtrading and manage risk.

        Returns:
            list: Protection configuration including cooldown and stop-loss guard
        """
        prot = [{
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        }]

        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 48,  # 4 hours in 5m timeframe
                "trade_limit": 2,               # Allow 2 trades before triggering
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False          # Apply globally across all pairs
            })

        return prot

    def calculate_btc_correlation(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate correlation with Bitcoin for market sentiment analysis.

        This function analyzes the correlation between the current trading pair
        and Bitcoin to avoid counter-trend trades during strong BTC movements.

        Args:
            dataframe (DataFrame): Current pair's OHLCV data
            metadata (dict): Pair metadata including pair name

        Returns:
            DataFrame: Dataframe with added BTC correlation columns:
                - btc_correlation: Correlation coefficient (-1 to 1)
                - btc_trend: BTC trend direction (1=bullish, -1=bearish, 0=neutral)
        """
        pair = metadata['pair']

        # If trading BTC directly, assume perfect correlation
        if 'BTC' in pair.split('/')[0]:
            dataframe['btc_correlation'] = 1.0
            dataframe['btc_trend'] = 1
            return dataframe

        # Attempt to retrieve BTC data from multiple possible pair formats
        btc_pairs = ["BTC/USDT:USDT", "BTC/USDT"]
        btc_data = None

        for btc_pair in btc_pairs:
            try:
                btc_data, _ = self.dp.get_analyzed_dataframe(btc_pair, self.timeframe)
                if btc_data is not None and len(btc_data) >= 50:
                    break
            except:
                continue

        # Default values if BTC data unavailable
        if btc_data is None or btc_data.empty:
            dataframe['btc_correlation'] = 0.5
            dataframe['btc_trend'] = 0
            return dataframe

        # Calculate BTC trend using shorter SMAs for 5m responsiveness
        btc_sma12 = btc_data['close'].rolling(12).mean()  # 1-hour average
        btc_sma26 = btc_data['close'].rolling(26).mean()  # 2+ hour average

        # Determine BTC trend direction
        if len(btc_data) > 0:
            current_close = btc_data['close'].iloc[-1]
            sma12_current = btc_sma12.iloc[-1] if len(btc_sma12) > 0 else current_close
            sma26_current = btc_sma26.iloc[-1] if len(btc_sma26) > 0 else current_close

            if current_close > sma12_current > sma26_current:
                btc_trend = 1      # Strong bullish trend
            elif current_close < sma12_current < sma26_current:
                btc_trend = -1     # Strong bearish trend
            else:
                btc_trend = 0      # Sideways/unclear trend
        else:
            btc_trend = 0

        # Calculate simple directional correlation over 12 periods (1 hour)
        pair_direction = 1 if dataframe['close'].iloc[-1] > dataframe['close'].iloc[-12] else -1
        btc_direction = 1 if btc_data['close'].iloc[-1] > btc_data['close'].iloc[-12] else -1

        # Correlation: 1 if same direction, -1 if opposite
        correlation = 1.0 if pair_direction == btc_direction else -1.0

        dataframe['btc_correlation'] = correlation
        dataframe['btc_trend'] = btc_trend

        return dataframe

    def calculate_market_breadth(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate market breadth across major cryptocurrency pairs.

        Market breadth analysis helps identify overall market sentiment by
        monitoring what percentage of major cryptocurrencies are in uptrends.
        This filters out trades against prevailing market conditions.

        Args:
            dataframe (DataFrame): Current pair's OHLCV data
            metadata (dict): Pair metadata including pair name

        Returns:
            DataFrame: Dataframe with added market breadth columns:
                - market_breadth: Percentage of major pairs in uptrend (0-1)
                - market_bullish: Count of bullish major pairs
                - market_total: Total major pairs analyzed
        """
        # Major cryptocurrency pairs for breadth analysis
        major_pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]

        # Adjust for futures trading if necessary
        is_futures = ':' in metadata['pair']
        if is_futures:
            settlement = metadata['pair'].split(':')[1]
            major_pairs = [f"{pair.split('/')[0]}/USDT:{settlement}" for pair in major_pairs]

        bullish_count = 0
        total_checked = 0

        for check_pair in major_pairs:
            try:
                # Attempt to get data for the major pair
                pair_data, _ = self.dp.get_analyzed_dataframe(check_pair, self.timeframe)

                # Fallback to spot if futures data unavailable
                if pair_data.empty or len(pair_data) < 20:
                    if is_futures:
                        spot_pair = check_pair.split(':')[0]
                        pair_data, _ = self.dp.get_analyzed_dataframe(spot_pair, self.timeframe)
                        if pair_data.empty:
                            continue
                    else:
                        continue

                # Check if pair is in uptrend using SMA12 for 5m sensitivity
                current_close = pair_data['close'].iloc[-1]
                sma12 = pair_data['close'].rolling(12).mean().iloc[-1]

                if current_close > sma12:
                    bullish_count += 1

                total_checked += 1

            except:
                # Skip pairs with data issues
                continue

        # Calculate market breadth percentage
        market_breadth = bullish_count / total_checked if total_checked > 0 else 0.5

        dataframe['market_breadth'] = market_breadth
        dataframe['market_bullish'] = bullish_count
        dataframe['market_total'] = total_checked

        return dataframe

    def calculate_murrey_levels(self, dataframe: DataFrame) -> DataFrame:
        """
        Calculate simplified Murrey Math levels for support/resistance analysis.

        Murrey Math provides mathematical support and resistance levels based on
        price action over a defined period. These levels help confirm entry points
        and avoid trades in unfavorable price zones.

        Args:
            dataframe (DataFrame): OHLCV data

        Returns:
            DataFrame: Dataframe with added Murrey level columns:
                - murrey_25: 25% level (strong support)
                - murrey_50: 50% level (pivot point)
                - murrey_75: 75% level (strong resistance)
                - above_50: Boolean indicator if price above 50% level
                - near_25: Boolean indicator if price near 25% level
                - near_75: Boolean indicator if price near 75% level
        """
        # Calculation period: 48 candles = 4 hours in 5m timeframe
        period = 48

        # Calculate rolling high and low over the period
        rolling_high = dataframe['high'].rolling(period).max()
        rolling_low = dataframe['low'].rolling(period).min()

        # Calculate range and key Murrey levels
        range_size = rolling_high - rolling_low

        dataframe['murrey_25'] = rolling_low + (range_size * 0.25)   # Strong support
        dataframe['murrey_50'] = rolling_low + (range_size * 0.50)   # Pivot point
        dataframe['murrey_75'] = rolling_low + (range_size * 0.75)   # Strong resistance

        # Price position relative to levels
        dataframe['above_50'] = (dataframe['close'] > dataframe['murrey_50']).astype(int)

        # Proximity detection using configurable buffer
        dataframe['near_25'] = (
            abs(dataframe['close'] - dataframe['murrey_25']) / dataframe['close']
            < self.murrey_buffer.value
        ).astype(int)

        dataframe['near_75'] = (
            abs(dataframe['close'] - dataframe['murrey_75']) / dataframe['close']
            < self.murrey_buffer.value
        ).astype(int)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Calculate all technical indicators required for the strategy.

        This function computes the complete set of technical indicators used
        for signal generation, including core ADX family indicators, supporting
        momentum and trend indicators, and market analysis components.

        Args:
            dataframe (DataFrame): OHLCV data
            metadata (dict): Pair metadata

        Returns:
            DataFrame: Dataframe with all calculated indicators
        """
        # ==================== CORE ADX FAMILY INDICATORS ====================
        # Adjusted to 12-period for increased 5m responsiveness
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=12)        # Average True Range
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=12)        # Average Directional Index
        dataframe['pdi'] = ta.PLUS_DI(dataframe, timeperiod=12)    # Plus Directional Indicator
        dataframe['mdi'] = ta.MINUS_DI(dataframe, timeperiod=12)   # Minus Directional Indicator
        dataframe['dx'] = ta.DX(dataframe, timeperiod=12)          # Directional Movement Index

        # ==================== SUPPORTING INDICATORS ====================
        # RSI for overbought/oversold conditions
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=12)

        # Simple Moving Averages for trend confirmation
        dataframe['sma12'] = ta.SMA(dataframe, timeperiod=12)      # Short-term trend
        dataframe['sma26'] = ta.SMA(dataframe, timeperiod=26)      # Medium-term trend

        # Volume analysis
        dataframe['volume_sma'] = dataframe['volume'].rolling(12).mean()

        # ==================== MARKET ANALYSIS COMPONENTS ====================
        # BTC correlation analysis
        if self.btc_correlation_enabled.value:
            dataframe = self.calculate_btc_correlation(dataframe, metadata)

        # Market breadth analysis
        if self.market_breadth_enabled.value:
            dataframe = self.calculate_market_breadth(dataframe, metadata)

        # Murrey Math levels
        if self.use_murrey_confirmation.value:
            dataframe = self.calculate_murrey_levels(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on ADX directional movement and market filters.

        Entry logic combines primary ADX-based momentum signals with multiple
        market condition filters to ensure high-probability trade setups.

        Long Entry Conditions:
        1. DX crosses above PDI (momentum shift to upside)
        2. ADX > MDI and PDI > MDI (confirmed uptrend)
        3. ADX > minimum threshold (sufficient trend strength)
        4. Volume confirmation and market filter approval
        5. Murrey level confirmation (near support or breaking resistance)

        Short Entry Conditions:
        1. DX crosses above MDI (momentum shift to downside)
        2. ADX > PDI and MDI > PDI (confirmed downtrend)
        3. ADX > minimum threshold (sufficient trend strength)
        4. Volume confirmation and market filter approval
        5. Murrey level confirmation (near resistance or breaking support)

        Args:
            dataframe (DataFrame): Dataframe with calculated indicators
            metadata (dict): Pair metadata

        Returns:
            DataFrame: Dataframe with entry signals in 'enter_long' and 'enter_short' columns
        """
        # ==================== PRIMARY ENTRY CONDITIONS ====================

        # Long entry: DX crosses above PDI with trend confirmation
        long_condition = (
            (qtpylib.crossed_above(dataframe['dx'], dataframe['pdi'])) &
            (dataframe['adx'] > dataframe['mdi']) &
            (dataframe['pdi'] > dataframe['mdi']) &
            (dataframe['adx'] > self.adx_minimum.value) &
            (dataframe['volume'] > 0)
        )

        # Short entry: DX crosses above MDI with trend confirmation
        short_condition = (
            (qtpylib.crossed_above(dataframe['dx'], dataframe['mdi'])) &
            (dataframe['adx'] > dataframe['pdi']) &
            (dataframe['mdi'] > dataframe['pdi']) &
            (dataframe['adx'] > self.adx_minimum.value) &
            (dataframe['volume'] > 0)
        )

        # ==================== BTC CORRELATION FILTER ====================
        if self.btc_trend_filter.value:
            # Avoid shorts during strong BTC uptrends (unless extremely overbought)
            short_condition &= (dataframe['btc_trend'] <= 0) | (dataframe['rsi'] > 70)

            # Avoid longs during strong BTC downtrends (unless extremely oversold)
            long_condition &= (dataframe['btc_trend'] >= 0) | (dataframe['rsi'] < 30)

        # ==================== MARKET BREADTH FILTER ====================
        if self.market_breadth_enabled.value:
            # Long entries only when majority of market is bullish
            long_condition &= (dataframe['market_breadth'] >= self.market_breadth_threshold.value)

            # Short entries only when majority of market is bearish
            short_condition &= (dataframe['market_breadth'] <= (1 - self.market_breadth_threshold.value))

        # ==================== MURREY LEVEL CONFIRMATION ====================
        if self.use_murrey_confirmation.value:
            # Long confirmation: near support, breaking pivot upward, or oversold
            long_murrey = (
                (dataframe['near_25'] == 1) |  # Near 25% support level
                ((dataframe['close'] > dataframe['murrey_50']) &
                 (dataframe['close'].shift(1) <= dataframe['murrey_50'])) |  # Breaking 50% upward
                (dataframe['rsi'] < 30)  # Extremely oversold condition
            )
            long_condition &= long_murrey

            # Short confirmation: near resistance, breaking pivot downward, or overbought
            short_murrey = (
                (dataframe['near_75'] == 1) |  # Near 75% resistance level
                ((dataframe['close'] < dataframe['murrey_50']) &
                 (dataframe['close'].shift(1) >= dataframe['murrey_50'])) |  # Breaking 50% downward
                (dataframe['rsi'] > 70)  # Extremely overbought condition
            )
            short_condition &= short_murrey

        # ==================== ASSIGN ENTRY SIGNALS ====================
        dataframe.loc[long_condition, ['enter_long', 'enter_tag']] = (1, 'zaratustra_long_5m')
        dataframe.loc[short_condition, ['enter_short', 'enter_tag']] = (1, 'zaratustra_short_5m')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on trend strength deterioration.

        Exit conditions focus on detecting when the underlying trend momentum
        that triggered the entry is weakening or reversing.

        Exit Conditions:
        - Long exits: PDI falls below MDI or ADX drops below minimum threshold
        - Short exits: PDI rises above MDI or ADX drops below minimum threshold

        Args:
            dataframe (DataFrame): Dataframe with calculated indicators
            metadata (dict): Pair metadata

        Returns:
            DataFrame: Dataframe with exit signals in 'exit_long' and 'exit_short' columns
        """
        # Long exit: trend weakness or reversal
        exit_long = (
            (qtpylib.crossed_below(dataframe['pdi'], dataframe['mdi'])) |  # Trend reversal
            (dataframe['adx'] < self.adx_minimum.value)                    # Trend weakness
        )

        # Short exit: trend weakness or reversal
        exit_short = (
            (qtpylib.crossed_above(dataframe['pdi'], dataframe['mdi'])) |  # Trend reversal
            (dataframe['adx'] < self.adx_minimum.value)                    # Trend weakness
        )

        dataframe.loc[exit_long, ['exit_long', 'exit_tag']] = (1, 'zaratustra_exit_long_5m')
        dataframe.loc[exit_short, ['exit_short', 'exit_tag']] = (1, 'zaratustra_exit_short_5m')

        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Calculate dynamic stop-loss based on Average True Range (ATR).

        The dynamic stop-loss adapts to market volatility and trade duration:
        - Uses 1.5x ATR as base stop-loss distance
        - Tighter limits for 5m timeframe (3-10% range)
        - Loosens stop-loss for longer-held trades to avoid premature exits

        Args:
            pair (str): Trading pair name
            trade (Trade): Current trade object
            current_time (datetime): Current timestamp
            current_rate (float): Current price
            current_profit (float): Current profit/loss ratio
            **kwargs: Additional parameters

        Returns:
            float: Stop-loss level as negative decimal (e.g., -0.05 for 5% stop)
        """
        # Get current market data
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or 'atr' not in dataframe.columns:
            return -0.08  # Default 8% stop-loss

        # Get latest ATR value
        last_atr = dataframe['atr'].iloc[-1]
        if pd.isna(last_atr) or last_atr <= 0:
            return -0.08

        # Calculate ATR-based stop-loss
        atr_ratio = last_atr / current_rate
        dynamic_sl = -abs(atr_ratio * 1.5)  # 1.5x ATR distance

        # Apply tighter limits for 5m timeframe
        dynamic_sl = max(dynamic_sl, -0.10)  # Maximum 10% loss
        dynamic_sl = min(dynamic_sl, -0.03)  # Minimum 3% loss

        # Loosen stop-loss for trades held longer than 12 hours
        # This prevents forced exits during extended consolidation periods
        time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600
        if time_in_trade > 12:
            dynamic_sl = max(dynamic_sl, -0.12)  # Allow up to 12% loss for old trades

        return dynamic_sl

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Validate trade exits to prevent unprofitable forced exits.

        This function provides additional protection against forced exits that
        would result in significant losses, particularly important for trades
        that might be temporarily underwater but could recover.

        Args:
            pair (str): Trading pair name
            trade (Trade): Current trade object
            order_type (str): Type of exit order
            amount (float): Amount to exit
            rate (float): Exit rate
            time_in_force (str): Order time in force
            exit_reason (str): Reason for exit
            current_time (datetime): Current timestamp
            **kwargs: Additional parameters

        Returns:
            bool: True to allow exit, False to block exit
        """
        # Block trailing stops (not used in this strategy)
        if exit_reason in ["trailing_stop_loss", "trailing_stop"]:
            return False

        # Protect against unprofitable force exits
        if exit_reason in ["force_exit", "force_sell"]:
            current_profit = trade.calc_profit_ratio(rate)
            if current_profit < -0.05:  # Block force exit with >5% loss
                logger.warning(f"{pair} Blocking force exit with loss {current_profit:.2%}")
                return False

        return True

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Implement intelligent position adjustment (DCA) with market awareness.

        This function manages position sizing dynamically based on:
        1. Trade performance and drawdown levels
        2. Market condition analysis
        3. Trend strength indicators
        4. Risk management constraints

        DCA Strategy:
        - First additional entry at -3% from initial entry
        - Second additional entry at -5% from initial entry
        - Third additional entry at -7% from initial entry (maximum)
        - Partial profit taking at +8% gain
        - Position reduction during weak trends
        - Market breadth filtering for DCA decisions

        Args:
            trade (Trade): Current trade object
            current_time (datetime): Current timestamp
            current_rate (float): Current market price
            current_profit (float): Current profit/loss ratio
            min_stake (Optional[float]): Minimum stake amount
            max_stake (float): Maximum stake amount
            current_entry_rate (float): Current entry rate
            current_exit_rate (float): Current exit rate
            current_entry_profit (float): Current entry profit
            current_exit_profit (float): Current exit profit
            **kwargs: Additional parameters

        Returns:
            Optional[float]: Stake adjustment amount (positive=buy more, negative=sell, None=no action)
        """
        # Get current market data
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        # Limit maximum DCA entries
        if trade.nr_of_successful_entries > 2:
            return None

        # ==================== MARKET CONDITION FILTERING ====================
        last_candle = dataframe.iloc[-1]

        # Apply market breadth filter to DCA decisions
        if self.market_breadth_enabled.value and 'market_breadth' in last_candle:
            market_breadth = last_candle['market_breadth']

            # Avoid DCA in long positions during very bearish market conditions
            if not trade.is_short and market_breadth < 0.25:
                logger.info(f"{trade.pair} Avoiding long DCA - bearish market ({market_breadth:.2%})")
                return None

            # Avoid DCA in short positions during very bullish market conditions
            if trade.is_short and market_breadth > 0.75:
                logger.info(f"{trade.pair} Avoiding short DCA - bullish market ({market_breadth:.2%})")
                return None

        # ==================== PROGRESSIVE DCA ENTRY LEVELS ====================
        # Define entry thresholds for each DCA level (optimized for 5m volatility)
        if trade.nr_of_successful_entries > 0:
            entry_rules = {
                1: -0.03,  # First DCA at -3% loss
                2: -0.05,  # Second DCA at -5% loss
                3: -0.07   # Third DCA at -7% loss (maximum)
            }

            threshold = entry_rules.get(trade.nr_of_successful_entries, -0.07)
            if current_profit > threshold:
                return None

        # ==================== PARTIAL PROFIT TAKING ====================
        # Take partial profits on first profitable opportunity
        if current_profit > 0.08 and trade.nr_of_successful_exits == 0:
            logger.info(f"{trade.pair} Taking partial profit at {current_profit:.2%}")
            return -(trade.stake_amount / 2)  # Sell 50% of position

        # ==================== INDICATOR VALIDATION ====================
        # Ensure all required indicators are available
        try:
            atr = last_candle['atr']
            adx = last_candle['adx']
            pdi = last_candle['pdi']
            mdi = last_candle['mdi']
        except KeyError as e:
            logger.warning(f"Error accessing indicators for {trade.pair}: {e}")
            return None

        # Validate indicator values
        if pd.isna(atr) or pd.isna(adx) or pd.isna(pdi) or pd.isna(mdi) or current_rate <= 0:
            return None

        # ==================== DYNAMIC POSITION ADJUSTMENT ====================
        # Calculate volatility-adjusted thresholds
        atr_percent = (atr / current_rate) * 100
        adx_threshold_high = self.adx_minimum.value + (atr_percent * self.adx_high_multiplier.value)
        adx_threshold_low = max(self.adx_minimum.value - (atr_percent * self.adx_low_multiplier.value), 5)

        # ==================== TREND STRENGTH ANALYSIS ====================
        # Increase position size during strong trends
        if adx > adx_threshold_high:
            # Confirm trend direction aligns with trade direction
            trend_aligned = (
                (trade.entry_side == "buy" and pdi > mdi) or
                (trade.entry_side == "sell" and mdi > pdi)
            )

            if trend_aligned:
                # Calculate progressive position sizing
                filled_entries = trade.select_filled_orders(trade.entry_side)
                if filled_entries and filled_entries[0].amount > 0 and filled_entries[0].price > 0:
                    base_stake = filled_entries[0].amount * filled_entries[0].price

                    # Progressive multiplier: 1.0x, 1.4x, 1.8x for subsequent entries
                    multiplier = 1.0 + (0.4 * trade.nr_of_successful_entries)
                    stake_amount = base_stake * multiplier

                    # Apply stake limits
                    if min_stake:
                        stake_amount = max(stake_amount, min_stake)
                    stake_amount = min(stake_amount, max_stake)

                    logger.info(f"Strong trend DCA for {trade.pair}: stake={stake_amount:.4f} "
                              f"(ADX={adx:.1f}, threshold={adx_threshold_high:.1f})")
                    return stake_amount

        # ==================== TREND WEAKNESS MANAGEMENT ====================
        # Reduce position size during weak trends to limit risk
        elif adx < adx_threshold_low:
            reduction = -trade.stake_amount * 0.25  # Reduce position by 25%
            logger.info(f"Weak trend reduction for {trade.pair}: {reduction:.4f} "
                       f"(ADX={adx:.1f}, threshold={adx_threshold_low:.1f})")
            return reduction

        # No position adjustment needed
        return None
