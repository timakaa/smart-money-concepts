import numpy as np
import pandas as pd
from pandas import DataFrame


class Retracement:
    """Retracement indicator implementation"""

    @staticmethod
    def calculate(ohlc: DataFrame, swing_highs_lows: DataFrame) -> DataFrame:
        """
        Retracement
        This method returns the percentage of a retracement from the swing high or low

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function

        returns:
        Direction = 1 if bullish retracement, -1 if bearish retracement
        CurrentRetracement% = the current retracement percentage from the swing high or low
        DeepestRetracement% = the deepest retracement percentage from the swing high or low
        """

        # Make a copy to avoid modifying original data
        swing_highs_lows = swing_highs_lows.copy()

        # Initialize arrays to store results
        direction = np.zeros(len(ohlc), dtype=np.int32)  # Trend direction
        current_retracement = np.zeros(
            len(ohlc), dtype=np.float64
        )  # Current retracement %
        deepest_retracement = np.zeros(
            len(ohlc), dtype=np.float64
        )  # Maximum retracement %

        # Track the most recent swing high and low levels
        top = 0  # Most recent swing high
        bottom = 0  # Most recent swing low

        # Process each candle
        for i in range(len(ohlc)):
            if swing_highs_lows["HighLow"][i] == 1:
                # Found new swing high
                direction[i] = 1  # Mark as bullish trend
                top = swing_highs_lows["Level"][i]  # Update top level
            elif swing_highs_lows["HighLow"][i] == -1:
                # Found new swing low
                direction[i] = -1  # Mark as bearish trend
                bottom = swing_highs_lows["Level"][i]  # Update bottom level
            else:
                # No new swing point, maintain previous direction
                direction[i] = direction[i - 1] if i > 0 else 0

            # Calculate retracements based on trend direction
            if direction[i - 1] == 1:  # In bullish trend
                # Calculate retracement from swing high to current low
                # Formula: 100 - ((current_low - bottom) / (top - bottom) * 100)
                current_retracement[i] = round(
                    100 - (((ohlc["low"].iloc[i] - bottom) / (top - bottom)) * 100), 1
                )
                # Update deepest retracement if current is deeper
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == 1
                        else 0
                    ),
                    current_retracement[i],
                )
            if direction[i] == -1:  # In bearish trend
                # Calculate retracement from swing low to current high
                # Formula: 100 - ((current_high - top) / (bottom - top) * 100)
                current_retracement[i] = round(
                    100 - ((ohlc["high"].iloc[i] - top) / (bottom - top)) * 100, 1
                )
                # Update deepest retracement if current is deeper
                deepest_retracement[i] = max(
                    (
                        deepest_retracement[i - 1]
                        if i > 0 and direction[i - 1] == -1
                        else 0
                    ),
                    current_retracement[i],
                )

        # Shift arrays by 1 to align retracements with their trigger candles
        current_retracement = np.roll(current_retracement, 1)
        deepest_retracement = np.roll(deepest_retracement, 1)
        direction = np.roll(direction, 1)

        # Clean up first few entries that might have incorrect calculations
        # due to insufficient historical data
        remove_first_count = 0
        for i in range(len(direction)):
            if i + 1 == len(direction):
                break
            # Count direction changes until we have 3
            if direction[i] != direction[i + 1]:
                remove_first_count += 1
            # Clear data until we have stable calculations
            direction[i] = 0
            current_retracement[i] = 0
            deepest_retracement[i] = 0
            # After 3 direction changes, clear one more candle and stop
            if remove_first_count == 3:
                direction[i + 1] = 0
                current_retracement[i + 1] = 0
                deepest_retracement[i + 1] = 0
                break

        # Convert arrays to pandas Series with descriptive names
        direction = pd.Series(direction, name="Direction")
        current_retracement = pd.Series(current_retracement, name="CurrentRetracement%")
        deepest_retracement = pd.Series(deepest_retracement, name="DeepestRetracement%")

        # Return all components as a DataFrame
        return pd.concat([direction, current_retracement, deepest_retracement], axis=1)
