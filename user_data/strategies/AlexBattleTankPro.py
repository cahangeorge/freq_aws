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


class AlexBattleTankProV2(IStrategy):
    """
    Enhanced strategy on the 15-minute timeframe.

    Key improvements:
      - Dynamic stoploss based on ATR.
      - Dynamic leverage calculation.
      - Murrey Math level calculation (rolling window for performance).
      - Enhanced DCA (Average Price) logic.
      - Translated to English and code structured for clarity.
      - Parameterization of internal constants for optimization.
    """

    # General strategy parameters
    timeframe = "1h"
    startup_candle_count: int = 0
    stoploss = -0.15
    trailing_stop = False
    position_adjustment_enable = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = True
    max_stake_per_trade = 5.0
    max_portfolio_percentage_per_trade = 0.05
    max_entry_position_adjustment = 2
    process_only_new_candles = True
    max_dca_orders = 3
    max_total_stake_per_pair = 10
    max_single_dca_amount = 5

    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 6, default=2, space="buy", optimize=True, load=True)
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
    use_stop_protection = BooleanParameter(default=False, space="protection", optimize=True)

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
    long_rsi_threshold = IntParameter(55, 70, default=60, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=40, space="sell", optimize=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)
    leverage_base = DecimalParameter(5.0, 20.0, default=10.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=True, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 10, default=5, space="buy", optimize=True, load=True)
    indicator_rolling_window_threshold = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    indicator_rolling_window_threshold = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    indicator_rolling_check_window = IntParameter(2, 10, default=4, space="buy", optimize=True, load=True)
    #enable_smart_dca = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    dca_rsi_filter = IntParameter(20, 50, default=35, space="buy", optimize=True, load=True)
    dca_profit_threshold = DecimalParameter(-0.05, -0.01, default=-0.025, decimals=3, space="buy", optimize=True, load=True)

    # Profit-taking Parameter
    partial_exit_1_threshold = DecimalParameter(0.15, 0.35, default=0.25, decimals=2, space="sell", optimize=True, load=True)
    partial_exit_1_amount = DecimalParameter(0.20, 0.35, default=0.25, decimals=2, space="sell", optimize=True, load=True)
    partial_exit_2_threshold = DecimalParameter(0.35, 0.55, default=0.45, decimals=2, space="sell", optimize=True, load=True)
    partial_exit_2_amount = DecimalParameter(0.25, 0.40, default=0.33, decimals=2, space="sell", optimize=True, load=True)

    # Market condition filter
    market_condition_window = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.025, decimals=3, space="buy", optimize=True, load=True)
    # ROI table (minutes to decimal)
    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }
    enable_smart_dca = True
    enable_position_flip = True
    #enable_position_flip = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    flip_profit_threshold = DecimalParameter(
        low=0.01, high=0.10, default=0.03, decimals=3, space="buy", optimize=True, load=True
    )

    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            "min_max_viz": {
                "maxima": {"color": "#a29db9", "type": "line"},
                "minima": {"color": "#aac7fc", "type": "line"},
                "maxima_check": {"color": "#a29db9", "type": "line"},
                "minima_check": {"color": "#aac7fc", "type": "line"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},
                "[8/8]P": {"color": "red", "type": "line"},
                "[0/8]P": {"color": "red", "type": "line"},
            }
        },
    }
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

    def calculate_rolling_murrey_math_levels(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        rolling_high = df["high"].rolling(window=window_size, min_periods=window_size).max()
        rolling_low = df["low"].rolling(window=window_size, min_periods=window_size).min()
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        for i in range(len(df)):
            if i < window_size - 1:
                continue
            mn_period = rolling_low.iloc[i]
            mx_period = rolling_high.iloc[i]
            current_close = df["close"].iloc[i]
            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][i] = current_close
                continue
            finalH_period = mx_period
            finalL_period = mn_period
            if finalH_period == finalL_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][i] = current_close
                continue
            levels = AlexBattleTankProV2._calculate_mml_core(mn_period, finalH_period, mx_period, finalL_period, mml_c1, mml_c2)
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][i] = levels.get(key, current_close)
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}
    @property
    def protections(self):
        prot = [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 200,
                "trade_limit": 20,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.15  # 15% max drawdown
            }
        ]

        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 50,  # Reduziert für 15m
                "trade_limit": 3,  # Erhöht
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": True,  # Pro Pair statt global
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
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            logger.warning(f"{pair} Empty DataFrame in custom_entry_price. Returning proposed_rate.")
            return proposed_rate
        last_candle = dataframe.iloc[-1]
        entry_price = (last_candle["close"] + last_candle["open"] + proposed_rate) / 3.0
        if side == "long":
            if proposed_rate < entry_price:
                entry_price = proposed_rate
        elif side == "short":
            if proposed_rate > entry_price:
                entry_price = proposed_rate
        logger.info(
            f"{pair} Calculated Entry Price: {entry_price:.8f} | Last Close: {last_candle['close']:.8f}, "
            f"Last Open: {last_candle['open']:.8f}, Proposed Rate: {proposed_rate:.8f}")
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.000005:
            increment_factor = self.increment_for_unique_price.value if side == "long" else (
                1.0 / self.increment_for_unique_price.value)
            entry_price *= increment_factor
            logger.info(
                f"{pair} Entry price incremented to {entry_price:.8f} (previous: {self.last_entry_price:.8f}) due to proximity.")
        self.last_entry_price = entry_price
        return entry_price

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty or 'atr' not in dataframe.columns or dataframe['atr'].isnull().all():
            logger.warning(
                f"{pair} ATR not available or all NaN for dynamic stoploss. Using default stoploss: {self.stoploss}")
            return self.stoploss
        last_atr = dataframe["atr"].iat[-1]
        if pd.isna(last_atr) or last_atr == 0:
            valid_atr = dataframe["atr"].dropna()
            if not valid_atr.empty:
                last_atr = valid_atr.iat[-1]
            else:
                logger.warning(
                    f"{pair} All ATR values are NaN or no valid ATR found. Using fallback stoploss -0.10.")
                return -0.10
        if last_atr == 0:
            logger.warning(f"{pair} ATR is 0. Using fallback stoploss (-0.10 for 10%).")
            return -0.10
        atr_multiplier = self.stoploss_atr_multiplier.value
        if current_rate == 0:
            logger.warning(
                f"{pair} Current rate is 0. Cannot compute dynamic stoploss. Using default: {self.stoploss}")
            return self.stoploss
        dynamic_sl_ratio = atr_multiplier * last_atr / current_rate
        calculated_stoploss = -abs(dynamic_sl_ratio)
        max_reasonable_sl = self.stoploss_max_reasonable.value
        final_stoploss = max(calculated_stoploss, max_reasonable_sl)
        logger.info(
            f"{pair} Dynamic Stoploss: {final_stoploss:.4f} (ATR: {last_atr:.8f}, Rate: {current_rate:.8f}, SL Ratio: {dynamic_sl_ratio:.4f})")
        return final_stoploss

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float,
                           time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        """
        Erweiterte Exit-Kontrolle mit Position-Flip-Unterstützung
        """
        current_profit_ratio = trade.calc_profit_ratio(rate)

        # Position Flip Check
        if self.enable_position_flip:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            should_flip, new_direction = self.should_flip_position(pair, trade, dataframe)

            if should_flip:
                logger.info(f"{pair} POSITION FLIP DETECTED: {trade.trade_direction} → {new_direction.upper()} "
                           f"(Current profit: {current_profit_ratio:.2%})")
                # Erlaube Exit für Flip nur bei positivem oder break-even Gewinn
                return current_profit_ratio >= -self.flip_profit_threshold.value

        # BLOCKIERE ALLE VERLUSTBRINGENDEN EXIT REASONS (bestehender Code)
        blocked_exits = [
            "trailing_stop_loss",
            "trailing_stop",
            "Risk_Management_Exit",
            "Profit_Target_Reversal",
            "Resistance_Level_Exit",
            "Support_Level_Exit"
        ]

        if exit_reason in blocked_exits and current_profit_ratio > self.stoploss:
            logger.info(f"{pair} BLOCKING exit reason: {exit_reason} (was unprofitable)")
            return False

        # FORCE EXIT - STRIKTER KONTROLLE (bestehender Code)
        if exit_reason == "force_exit":
            if current_profit_ratio > 0.05:
                logger.info(f"{pair} Allowing force_exit with good profit: {current_profit_ratio:.2%}")
                return True
            else:
                logger.info(f"{pair} BLOCKING force_exit with insufficient profit: {current_profit_ratio:.2%}")
                return False

        # ERFOLGREICHE EXITS (bestehender Code)
        successful_exits = [
            "roi", "stoploss", "Exit_Max_Check", "Exit_Max_Confirmed",
            "Exit_Min_Check", "Exit_Min_Confirmed"
        ]

        if exit_reason in successful_exits:
            return True

        # Neue Exits mit strengerem Gewinn-Check (bestehender Code)
        if exit_reason in ["Strong_Trend_Reversal", "Emergency_Exit", "Extreme_Emergency"]:
            return current_profit_ratio > 0.0

        # Standard: Sehr konservativ - nur bei Gewinn (bestehender Code)
        return current_profit_ratio > 0.0 or current_profit_ratio < self.stoploss

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


    def should_flip_position(self, pair: str, current_trade: Trade, dataframe: pd.DataFrame) -> Tuple[bool, str]:
        """
        Prüft ob eine Position geflippt werden soll (Long→Short oder Short→Long)
        Returns: (should_flip, new_direction)
        """
        if not self.enable_position_flip:
            return False, ""

        if dataframe.empty:
            return False, ""

        last_candle = dataframe.iloc[-1]
        current_profit = current_trade.calc_profit_ratio(last_candle["close"])

        # Nur flippen wenn wenigstens minimal profitabel oder break-even
        if current_profit < -self.flip_profit_threshold.value:
            return False, ""

        # Long→Short Flip Bedingungen
        if self.is_long_trade(trade):
            # Starke Short-Signale die Long beenden sollten
            short_flip_conditions = [
                # Confirmed Max Entry Signal
                (last_candle["DI_catch"] == 1 and
                 last_candle["minima_check"] == 1 and
                 last_candle["s_extrema"] > 0 and
                 dataframe["maxima"].shift(1).iloc[-1] == 1 and
                 last_candle["rsi"] > 65),

                # Aggressive Max Entry Signal
                (last_candle["maxima_check"] == 0 and
                 last_candle["rsi"] > 70),

                # Rolling MaxH2 Signal
                (last_candle["maxh2"] > 0 and
                 last_candle["rsi"] > 68)
            ]

            if any(short_flip_conditions):
                return True, "short"

        # Short→Long Flip Bedingungen
        elif current_self.is_short_trade(trade):
            # Starke Long-Signale die Short beenden sollten
            long_flip_conditions = [
                # Confirmed Min Entry Signal
                (last_candle["DI_catch"] == 1 and
                 last_candle["maxima_check"] == 1 and
                 last_candle["s_extrema"] < 0 and
                 dataframe["minima"].shift(1).iloc[-1] == 1 and
                 last_candle["rsi"] < 35),

                # Aggressive Min Entry Signal
                (last_candle["minima_check"] == 0 and
                 last_candle["rsi"] < 30),

                # Rolling MinH2 Signal
                (last_candle["minh2"] < 0 and
                 last_candle["rsi"] < 36)
            ]

            if any(long_flip_conditions):
                return True, "long"

        return False, ""

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
            return proposed_leverage
        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value
        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

        current_rsi = rsi_array[-1] if rsi_array.size > 0 and not np.isnan(rsi_array[-1]) else 50.0
        current_atr = atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
        current_sma = sma_array[-1] if sma_array.size > 0 and not np.isnan(sma_array[-1]) else current_rate
        current_macd_hist = 0.0

        if isinstance(macd_output, pd.DataFrame):
            if not macd_output.empty and 'macdhist' in macd_output.columns:
                valid_macdhist_series = macd_output['macdhist'].dropna()
                if not valid_macdhist_series.empty:
                    current_macd_hist = valid_macdhist_series.iloc[-1]

        # Apply rules based on indicators
        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
        return adjusted_leverage
    # 👇 HIER DIE HELPER-METHODEN HINZUFÜGEN:
        def is_long_trade(self, trade: Trade) -> bool:
            """Determine if trade is long - compatible with all Freqtrade versions"""
            if hasattr(trade, 'is_long'):
                return trade.is_long
            elif hasattr(trade, 'trade_direction'):
                return trade.trade_direction == 'long'
            elif hasattr(trade, 'entry_side'):
                return trade.entry_side == 'buy'
            else:
                return trade.amount > 0

        def is_short_trade(self, trade: Trade) -> bool:
            """Determine if trade is short - compatible with all Freqtrade versions"""
            if hasattr(trade, 'is_short'):
                return trade.is_short
            elif hasattr(trade, 'trade_direction'):
                return trade.trade_direction == 'short'
            elif hasattr(trade, 'entry_side'):
                return trade.entry_side == 'sell'
            else:
                return trade.amount < 0
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Standard indicators
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # Extrema detection
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

        # Heikin-Ashi close
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4

        # Rolling extrema
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, self.h2.value)
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(dataframe, self.h1.value)
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(dataframe, self.h0.value)
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(dataframe, self.cp.value)

        # Murrey Math levels
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels(dataframe, window_size=mml_window)
        for level_name, level_series in murrey_levels.items():
            dataframe[level_name] = level_series

        # MML oscillator
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")

        if mml_4_8 is not None and mml_plus_3_8 is not None and mml_minus_3_8 is not None:
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - mml_4_8) / osc_denominator)
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # DI Catch
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        # Rolling thresholds
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=rolling_window_threshold, min_periods=1).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=rolling_window_threshold, min_periods=1).max()

        # Extrema checks
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0).astype(int)
        dataframe["maxima_check"] = (dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0).astype(int)
        # MARKET CONDITION ANALYSIS am Ende hinzufügen:
        window = self.market_condition_window.value

        # Trend strength
        dataframe["trend_strength"] = abs(dataframe["close"].pct_change(window))
        dataframe["is_trending"] = dataframe["trend_strength"] > self.trend_strength_threshold.value

        # Volatility filter
        dataframe["volatility"] = dataframe["atr"] / dataframe["close"]
        dataframe["volatility_ma"] = dataframe["volatility"].rolling(window=20).mean()
        dataframe["low_volatility"] = dataframe["volatility"] < dataframe["volatility_ma"] * 0.8

        # Market sentiment
        dataframe["bullish_sentiment"] = (
            (dataframe["rsi"] > 45) &
            (dataframe["close"] > dataframe["ema50"]) &
            (dataframe["DI_values"] > 0)
        ).astype(int)

        dataframe["bearish_sentiment"] = (
            (dataframe["rsi"] < 55) &
            (dataframe["close"] < dataframe["ema50"]) &
            (dataframe["DI_values"] < 0)
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate entry signals with corrected logic and enhanced MML integration
        """

        # ===========================================
        # INITIALIZE ENTRY COLUMNS
        # ===========================================
        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""

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
        # ENHANCED LONG ENTRIES WITH MML
        # ===========================================

        # Confirmed long entry - MML Enhanced
        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 1) &
            (df["s_extrema"] < 0) &
            (df["minima"].shift(1) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 35) &
            (
                bullish_mml |  # Strong bullish MML structure
                mml_support_bounce  # Bounce from MML support
            ),
            ["enter_long", "enter_tag"]
        ] = (1, "Confirmed_Min_Entry_MML")

        # Aggressive long entry - MML Filtered
        df.loc[
            (df["minima_check"] == 0) &
            (df["volume"] > 0) &
            (df["rsi"] < 30) &
            (df["close"] > df["[2/8]P"]) &  # Above 25% MML level
            (~bearish_mml),  # Not in bearish structure
            ["enter_long", "enter_tag"]
        ] = (1, "Aggressive_Min_Entry_MML")

        # Transitional long entry - MML Enhanced
        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 0) &
            (df["minima_check"].shift(5) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 32) &
            (df["close"] >= df["[4/8]P"]),  # Above midline
            ["enter_long", "enter_tag"]
        ] = (1, "Transitional_Min_Entry_MML")

        # Rolling long entry - MML Filtered
        df.loc[
            (df["minh2"] < 0) &
            (df["rsi"] < 36) &
            (df["volume"] > 0) &
            (df["close"] > df["[2/8]P"]) &  # Above 25% level
            (bullish_mml | (~bearish_mml)),  # Bullish or neutral MML
            ["enter_long", "enter_tag"]
        ] = (1, "Rolling_MinH2_Entry_MML")

        # MML Breakout Long Entry (NEW)
        df.loc[
            (df["close"] > df["[4/8]P"]) &  # Above midline
            (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &  # Was below
            (df["volume"] > df["volume"].rolling(10).mean() * 1.2) &  # Volume spike
            (df["rsi"] > 45) & (df["rsi"] < 65) &  # Not oversold/overbought
            (df["minima_check"] == 1),
            ["enter_long", "enter_tag"]
        ] = (1, "MML_Breakout_Long")

        # MML Support Bounce Long (NEW)
        df.loc[
            mml_support_bounce &
            (df["volume"] > 0) &
            (df["rsi"] < 40) &
            (df["minima_check"] == 1) &
            (~bearish_mml),  # Not in strong bearish trend
            ["enter_long", "enter_tag"]
        ] = (1, "MML_Support_Bounce_Long")

        # ===========================================
        # ENHANCED SHORT ENTRIES WITH MML
        # ===========================================

        if self.can_short:
            # Confirmed short entry - MML Enhanced
            df.loc[
                (df["DI_catch"] == 1) &
                (df["maxima_check"] == 1) &
                (df["s_extrema"] > 0) &
                (df["maxima"].shift(1) == 1) &
                (df["volume"] > 0) &
                (df["rsi"] > 65) &
                (df["rsi"] < df["rsi"].shift(1)) &  # RSI turning down
                (df["close"] < df["ema50"]) &       # Below trend filter
                (
                    bearish_mml |  # Strong bearish MML structure
                    mml_resistance_reject  # Rejection at MML resistance
                ),
                ["enter_short", "enter_tag"]
            ] = (1, "Confirmed_Max_Entry_MML")

            # Aggressive short entry - MML Filtered
            df.loc[
                (df["maxima_check"] == 0) &
                (df["volume"] > 0) &
                (df["rsi"] > 70) &
                (df["close"] < df["[6/8]P"]) &  # Below 75% MML level
                (~bullish_mml),  # Not in bullish structure
                ["enter_short", "enter_tag"]
            ] = (1, "Aggressive_Max_Entry_MML")

            # Transitional short entry - MML Enhanced
            df.loc[
                (df["DI_catch"] == 1) &
                (df["maxima_check"] == 0) &
                (df["maxima_check"].shift(5) == 1) &
                (df["volume"] > 0) &
                (df["rsi"] > 68) &
                (df["close"] <= df["[4/8]P"]),  # Below midline
                ["enter_short", "enter_tag"]
            ] = (1, "Transitional_Max_Entry_MML")

            # Rolling short entry - MML Filtered
            df.loc[
                (df["maxh2"] > 0) &
                (df["rsi"] > 68) &
                (df["volume"] > 0) &
                (df["close"] < df["[6/8]P"]) &  # Below 75% level
                (bearish_mml | (~bullish_mml)),  # Bearish or neutral MML
                ["enter_short", "enter_tag"]
            ] = (1, "Rolling_MaxH2_Entry_MML")

            # MML Breakdown Short Entry (NEW)
            df.loc[
                (df["close"] < df["[4/8]P"]) &  # Below midline
                (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &  # Was above
                (df["volume"] > df["volume"].rolling(10).mean() * 1.2) &  # Volume spike
                (df["rsi"] > 35) & (df["rsi"] < 55) &  # Not oversold/overbought
                (df["maxima_check"] == 1),
                ["enter_short", "enter_tag"]
            ] = (1, "MML_Breakdown_Short")

            # MML Resistance Rejection Short (NEW)
            df.loc[
                mml_resistance_reject &
                (df["volume"] > 0) &
                (df["rsi"] > 60) &
                (df["maxima_check"] == 1) &
                (~bullish_mml),  # Not in strong bullish trend
                ["enter_short", "enter_tag"]
            ] = (1, "MML_Resistance_Reject_Short")

        # ===========================================
        # MARKET CONDITION FILTERED ENTRIES
        # ===========================================

        # LONG ENTRIES with Market Condition + MML Filter
        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 1) &
            (df["s_extrema"] < 0) &
            (df["minima"].shift(1) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 35) &
            (df["bullish_sentiment"] == 1) &  # Additional filter
            (~df["low_volatility"]) &  # Avoid too quiet markets
            (bullish_mml | (df["close"] > df["[4/8]P"])),  # MML confirmation
            ["enter_long", "enter_tag"]
        ] = (1, "Confirmed_Min_Entry_Filtered_MML")

        # AGGRESSIVE LONG with stricter MML filter
        df.loc[
            (df["minima_check"] == 0) &
            (df["volume"] > 0) &
            (df["rsi"] < 28) &  # Stricter RSI
            (df["is_trending"]) &  # Only in Trending Markets
            (df["bullish_sentiment"] == 1) &
            (df["close"] > df["[4/8]P"]) &  # Above MML midline
            (bullish_mml),  # Strong MML bullish structure
            ["enter_long", "enter_tag"]
        ] = (1, "Aggressive_Min_Entry_Filtered_MML")

        # SHORT ENTRIES with Market Condition + MML Filter
        if self.can_short:
            df.loc[
                (df["DI_catch"] == 1) &
                (df["maxima_check"] == 1) &
                (df["s_extrema"] > 0) &
                (df["maxima"].shift(1) == 1) &
                (df["volume"] > 0) &
                (df["rsi"] > 65) &
                (df["bearish_sentiment"] == 1) &  # Additional filter
                (~df["low_volatility"]) &
                (bearish_mml | (df["close"] < df["[4/8]P"])),  # MML confirmation
                ["enter_short", "enter_tag"]
            ] = (1, "Confirmed_Max_Entry_Filtered_MML")

        # ===========================================
        # RANGE-BOUND TRADING WITH MML
        # ===========================================

        # Range Long (nur bei oversold + MML support)
        df.loc[
            range_bound &
            (df["close"] <= df["[2/8]P"]) &  # Near bottom of range
            (df["rsi"] < 25) &  # Very oversold
            (df["minima_check"] == 1) &
            (df["volume"] > 0) &
            mml_support_bounce,  # Actual bounce confirmation
            ["enter_long", "enter_tag"]
        ] = (1, "MML_Range_Long")

        # Range Short (nur bei overbought + MML resistance)
        if self.can_short:
            df.loc[
                range_bound &
                (df["close"] >= df["[6/8]P"]) &  # Near top of range
                (df["rsi"] > 75) &  # Very overbought
                (df["maxima_check"] == 1) &
                (df["volume"] > 0) &
                mml_resistance_reject,  # Actual rejection confirmation
                ["enter_short", "enter_tag"]
            ] = (1, "MML_Range_Short")

        # ===========================================
        # POSITION FLIP LOGIC WITH MML
        # ===========================================

        if self.enable_position_flip:
            # Long Flip Entries (when Short position exists) - MML Enhanced
            df.loc[
                (df["DI_catch"] == 1) &
                (df["minima_check"] == 1) &
                (df["s_extrema"] < 0) &
                (df["minima"].shift(1) == 1) &
                (df["volume"] > 0) &
                (df["rsi"] < 35) &
                (df["close"] > df["[2/8]P"]),  # Above key MML support
                ["enter_long", "enter_tag"]
            ] = (1, "Flip_To_Long_MML")

            df.loc[
                (df["minima_check"] == 0) &
                (df["volume"] > 0) &
                (df["rsi"] < 30) &
                (df["close"] > df["[2/8]P"]) &  # Above key MML support
                (~bearish_mml),  # Not in strong bearish structure
                ["enter_long", "enter_tag"]
            ] = (1, "Flip_Aggressive_Long_MML")

            # Short Flip Entries (when Long position exists) - MML Enhanced
            if self.can_short:
                df.loc[
                    (df["DI_catch"] == 1) &
                    (df["maxima_check"] == 1) &
                    (df["s_extrema"] > 0) &
                    (df["maxima"].shift(1) == 1) &
                    (df["volume"] > 0) &
                    (df["rsi"] > 65) &
                    (df["close"] < df["[6/8]P"]),  # Below key MML resistance
                    ["enter_short", "enter_tag"]
                ] = (1, "Flip_To_Short_MML")

                df.loc[
                    (df["maxima_check"] == 0) &
                    (df["volume"] > 0) &
                    (df["rsi"] > 70) &
                    (df["close"] < df["[6/8]P"]) &  # Below key MML resistance
                    (~bullish_mml),  # Not in strong bullish structure
                    ["enter_short", "enter_tag"]
                ] = (1, "Flip_Aggressive_Short_MML")

        # When a long signal appears: close any short position AND enter long
        df.loc[df["enter_long"] == 1, "exit_short"] = 1
        # The long entry is already set above, so we keep it

        if self.can_short:
            # When a short signal appears: close any long position AND enter short
            df.loc[df["enter_short"] == 1, "exit_long"] = 1
            # The short entry is already set above, so we keep it

            # Additional safety: If we have conflicting signals, prioritize the stronger one
            # If both long and short signals trigger, use RSI to decide
            conflicting_signals = (df["enter_long"] == 1) & (df["enter_short"] == 1)

            # In conflict, favor longs when oversold, shorts when overbought
            df.loc[
                conflicting_signals & (df["rsi"] < 50),
                ["enter_short", "enter_tag"]
            ] = (0, "")  # Cancel short, keep long

            df.loc[
                conflicting_signals & (df["rsi"] >= 50),
                ["enter_long", "enter_tag"]
            ] = (0, "")  # Cancel long, keep short

        return df


    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate exit signals with enhanced MML targets and levels
        """

        # ===========================================
        # INITIALIZE EXIT COLUMNS
        # ===========================================
        df["exit_long"] = 0
        df["exit_short"] = 0
        df["exit_tag"] = ""

        # ===========================================
        # MML-ENHANCED LONG EXITS
        # ===========================================

        # Long exit at MML resistance levels
        df.loc[
            (
                (df["close"] >= df["[6/8]P"]) |  # At 75% resistance
                (df["close"] >= df["[8/8]P"]) |  # At 100% resistance
                ((df["high"] >= df["[7/8]P"]) & (df["close"] < df["[7/8]P"]))  # Rejection at 87.5%
            ) &
            (df["rsi"] > 70) &
            (df["volume"] > 0),
            ["exit_long", "exit_tag"]
        ] = (1, "MML_Resistance_Exit")

        # Standard long exit with MML confirmation
        df.loc[
            (df["maxima_check"] == 0) &
            (df["volume"] > 0) &
            (df["close"] < df["[4/8]P"]),  # Below MML midline
            ["exit_long", "exit_tag"]
        ] = (1, "Exit_Max_Check_MML")

        # Long exit on MML breakdown
        df.loc[
            (df["close"] < df["[4/8]P"]) &  # Below midline
            (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &  # Was above
            (df["volume"] > df["volume"].rolling(5).mean()) &
            (df["DI_catch"] == 1) &
            (df["s_extrema"] > 0),
            ["exit_long", "exit_tag"]
        ] = (1, "MML_Breakdown_Exit")

        df.loc[
            (df["DI_catch"] == 1) &
            (df["s_extrema"] > 0) &
            (df["maxima"].shift(1) == 1) &
            (df["volume"] > 0) &
            (df["close"] >= df["[6/8]P"]),  # At/above 75% resistance
            ["exit_long", "exit_tag"]
        ] = (1, "Exit_Max_Confirmed_MML")

        # ===========================================
        # MML-ENHANCED SHORT EXITS
        # ===========================================

        if self.can_short:
            # Short exit at MML support levels
            df.loc[
                (
                    (df["close"] <= df["[2/8]P"]) |  # At 25% support
                    (df["close"] <= df["[0/8]P"]) |  # At 0% support
                    ((df["low"] <= df["[1/8]P"]) & (df["close"] > df["[1/8]P"]))  # Bounce from 12.5%
                ) &
                (df["rsi"] < 30) &
                (df["volume"] > 0),
                ["exit_short", "exit_tag"]
            ] = (1, "MML_Support_Exit")

            # Standard short exit with MML confirmation
            df.loc[
                (df["minima_check"] == 0) &
                (df["volume"] > 0) &
                (df["close"] > df["[4/8]P"]),  # Above MML midline
                ["exit_short", "exit_tag"]
            ] = (1, "Exit_Min_Check_MML")

            # Short exit on MML breakout
            df.loc[
                (df["close"] > df["[4/8]P"]) &  # Above midline
                (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &  # Was below
                (df["volume"] > df["volume"].rolling(5).mean()) &
                (df["DI_catch"] == 1) &
                (df["s_extrema"] < 0),
                ["exit_short", "exit_tag"]
            ] = (1, "MML_Breakout_Exit")

            df.loc[
                (df["DI_catch"] == 1) &
                (df["s_extrema"] < 0) &
                (df["minima"].shift(1) == 1) &
                (df["volume"] > 0) &
                (df["close"] <= df["[2/8]P"]),  # At/below 25% support
                ["exit_short", "exit_tag"]
            ] = (1, "Exit_Min_Confirmed_MML")

        # ===========================================
        # MML-ENHANCED FLIP EXIT SIGNALS
        # ===========================================

        if self.enable_position_flip:
            # Exit Long for Flip to Short - MML Enhanced
            df.loc[
                (
                    ((df["DI_catch"] == 1) & (df["s_extrema"] > 0) & (df["maxima"].shift(1) == 1) & (df["rsi"] > 65)) |
                    ((df["maxima_check"] == 0) & (df["rsi"] > 70)) |
                    ((df["maxh2"] > 0) & (df["rsi"] > 68))
                ) &
                (df["close"] >= df["[6/8]P"]),  # At/above resistance
                ["exit_long", "exit_tag"]
            ] = (1, "Flip_Exit_Long_MML")

            # Exit Short for Flip to Long - MML Enhanced
            if self.can_short:
                df.loc[
                    (
                        ((df["DI_catch"] == 1) & (df["s_extrema"] < 0) & (df["minima"].shift(1) == 1) & (df["rsi"] < 35)) |
                        ((df["minima_check"] == 0) & (df["rsi"] < 30)) |
                        ((df["minh2"] < 0) & (df["rsi"] < 36))
                    ) &
                    (df["close"] <= df["[2/8]P"]),  # At/below support
                    ["exit_short", "exit_tag"]
                ] = (1, "Flip_Exit_Short_MML")

        return df
