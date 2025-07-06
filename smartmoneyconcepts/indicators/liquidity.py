import numpy as np
import pandas as pd
from pandas import DataFrame


class Liquidity:
    """Liquidity indicator implementation"""

    @staticmethod
    def calculate(
        ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01
    ) -> DataFrame:
        """
        Liquidity
        Liquidity is when there are multiple highs within a small range of each other,
        or multiple lows within a small range of each other.

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
        range_percent: float - the percentage of the range to determine liquidity

        returns:
        Liquidity = 1 if bullish liquidity, -1 if bearish liquidity
        Level = the level of the liquidity
        End = the index of the last liquidity level
        Swept = the index of the candle that swept the liquidity
        """

        # Make a copy to avoid modifying original data
        shl = swing_highs_lows.copy()
        n = len(ohlc)

        # Calculate the range within which we consider points to be part of same liquidity pool
        # This is a percentage of the total price range
        pip_range = (ohlc["high"].max() - ohlc["low"].min()) * range_percent

        # Convert price data to numpy arrays for faster processing
        ohlc_high = ohlc["high"].values
        ohlc_low = ohlc["low"].values
        # Copy swing point data to allow marking used points
        shl_HL = shl["HighLow"].values.copy()
        shl_Level = shl["Level"].values.copy()

        # Initialize output arrays with NaN
        liquidity = np.full(n, np.nan, dtype=np.float32)  # Type of liquidity
        liquidity_level = np.full(n, np.nan, dtype=np.float32)  # Price level
        liquidity_end = np.full(n, np.nan, dtype=np.float32)  # End of liquidity zone
        liquidity_swept = np.full(
            n, np.nan, dtype=np.float32
        )  # When liquidity is taken

        # Process bullish liquidity (swing highs)
        bull_indices = np.nonzero(np.array(shl_HL == 1, dtype=bool))[
            0
        ]  # Find all swing highs
        for i in bull_indices:
            # Skip if this point was already used in another liquidity zone
            if shl_HL[i] != 1:
                continue

            # Define the range around this swing high
            high_level = shl_Level[i]
            range_low = high_level - pip_range
            range_high = high_level + pip_range
            group_levels = [high_level]  # Track levels in this liquidity zone
            group_end = i  # Track last point in zone

            # Find when this liquidity gets swept (price breaks above range)
            c_start = i + 1
            if c_start < n:
                cond = ohlc_high[c_start:] >= range_high
                swept = c_start + int(np.argmax(cond)) if np.any(cond) else 0
            else:
                swept = 0

            # Look for other swing highs in the same range
            for j in bull_indices:
                if j <= i:  # Skip points we've already processed
                    continue
                if swept and j >= swept:  # Stop if we've passed the sweep point
                    break
                # If this point is within our range, add it to the group
                if shl_HL[j] == 1 and (range_low <= shl_Level[j] <= range_high):
                    group_levels.append(shl_Level[j])
                    group_end = j
                    shl_HL[j] = 0  # Mark as used

            # Only record liquidity if we found multiple points
            if len(group_levels) > 1:
                avg_level = sum(group_levels) / len(group_levels)
                liquidity[i] = 1  # Bullish liquidity
                liquidity_level[i] = avg_level  # Average price level
                liquidity_end[i] = group_end  # Last point in group
                liquidity_swept[i] = swept  # When liquidity was taken

        # Process bearish liquidity (swing lows) - similar logic to bullish
        bear_indices = np.nonzero(np.array(shl_HL == -1, dtype=bool))[
            0
        ]  # Find all swing lows
        for i in bear_indices:
            if shl_HL[i] != -1:
                continue

            # Define range around this swing low
            low_level = shl_Level[i]
            range_low = low_level - pip_range
            range_high = low_level + pip_range
            group_levels = [low_level]
            group_end = i

            # Find when liquidity gets swept (price breaks below range)
            c_start = i + 1
            if c_start < n:
                cond = ohlc_low[c_start:] <= range_low
                swept = c_start + int(np.argmax(cond)) if np.any(cond) else 0
            else:
                swept = 0

            # Look for other swing lows in the same range
            for j in bear_indices:
                if j <= i:
                    continue
                if swept and j >= swept:
                    break
                if shl_HL[j] == -1 and (range_low <= shl_Level[j] <= range_high):
                    group_levels.append(shl_Level[j])
                    group_end = j
                    shl_HL[j] = 0

            # Record bearish liquidity zone if multiple points found
            if len(group_levels) > 1:
                avg_level = sum(group_levels) / len(group_levels)
                liquidity[i] = -1  # Bearish liquidity
                liquidity_level[i] = avg_level  # Average price level
                liquidity_end[i] = group_end  # Last point in group
                liquidity_swept[i] = swept  # When liquidity was taken

        # Convert arrays to pandas Series with descriptive names
        liq_series = pd.Series(liquidity, name="Liquidity")
        level_series = pd.Series(liquidity_level, name="Level")
        end_series = pd.Series(liquidity_end, name="End")
        swept_series = pd.Series(liquidity_swept, name="Swept")

        # Return all components as a DataFrame
        return pd.concat([liq_series, level_series, end_series, swept_series], axis=1)
