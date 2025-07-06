import numpy as np
import pandas as pd
from pandas import DataFrame


class SwingHighsLows:
    """Swing Highs and Lows indicator implementation"""

    @staticmethod
    def calculate(ohlc: DataFrame, swing_length: int = 50) -> DataFrame:
        """
        Swing Highs and Lows
        A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
        A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

        parameters:
        swing_length: int - the amount of candles to look back and forward to determine the swing high or low

        returns:
        HighLow = 1 if swing high, -1 if swing low
        Level = the level of the swing high or low
        """

        # Double the swing length since we look both forward and backward
        swing_length *= 2

        # Initial identification of swing points
        # For each point, check if it's the highest/lowest in its window
        swing_highs_lows = np.where(
            # Check if current high is highest in the window
            ohlc["high"]
            == ohlc["high"].shift(-(swing_length // 2)).rolling(swing_length).max(),
            1,  # Mark as swing high
            np.where(
                # Check if current low is lowest in the window
                ohlc["low"]
                == ohlc["low"].shift(-(swing_length // 2)).rolling(swing_length).min(),
                -1,  # Mark as swing low
                np.nan,  # Not a swing point
            ),
        )

        # Clean up overlapping or invalid swing points
        while True:
            # Get positions of all valid swing points
            positions = np.where(~np.isnan(swing_highs_lows))[0]

            # Exit if we have less than 2 swing points
            if len(positions) < 2:
                break

            # Get current and next swing points for comparison
            current = np.array(swing_highs_lows[positions[:-1]], dtype=np.float32)
            next = np.array(swing_highs_lows[positions[1:]], dtype=np.float32)

            # Get the high/low values at these positions
            highs = np.array(ohlc["high"].iloc[positions[:-1]].values, dtype=np.float32)
            lows = np.array(ohlc["low"].iloc[positions[:-1]].values, dtype=np.float32)
            next_highs = np.array(
                ohlc["high"].iloc[positions[1:]].values, dtype=np.float32
            )
            next_lows = np.array(
                ohlc["low"].iloc[positions[1:]].values, dtype=np.float32
            )

            # Track which points to remove
            index_to_remove = np.zeros(len(positions), dtype=bool)

            # Handle consecutive swing highs
            consecutive_highs = (current == 1) & (next == 1)
            # Remove the lower of consecutive highs
            index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
            # Remove the higher of consecutive highs
            index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

            # Handle consecutive swing lows
            consecutive_lows = (current == -1) & (next == -1)
            # Remove the higher of consecutive lows
            index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
            # Remove the lower of consecutive lows
            index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)

            # If no points need removal, we're done
            if not index_to_remove.any():
                break

            # Remove invalid swing points
            swing_highs_lows[positions[index_to_remove]] = np.nan

        # Get final positions of valid swing points
        positions = np.where(~np.isnan(swing_highs_lows))[0]

        # Handle edge cases at start and end of data
        if len(positions) > 0:
            # Start with opposite type of first real swing point
            if swing_highs_lows[positions[0]] == 1:
                swing_highs_lows[0] = -1
            if swing_highs_lows[positions[0]] == -1:
                swing_highs_lows[0] = 1
            # End with opposite type of last real swing point
            if swing_highs_lows[positions[-1]] == -1:
                swing_highs_lows[-1] = 1
            if swing_highs_lows[positions[-1]] == 1:
                swing_highs_lows[-1] = -1

        # Get the price level for each swing point
        level = np.where(
            ~np.isnan(swing_highs_lows),
            np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
            np.nan,
        )

        # Return results as a DataFrame
        return pd.concat(
            [
                pd.Series(swing_highs_lows, name="HighLow"),
                pd.Series(level, name="Level"),
            ],
            axis=1,
        )
