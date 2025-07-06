import numpy as np
import pandas as pd
from pandas import DataFrame


class BreakOfStructure:
    """Break of Structure (BOS) and Change of Character (CHoCH) indicator implementation"""

    @staticmethod
    def calculate(
        ohlc: DataFrame, swing_highs_lows: DataFrame, close_break: bool = True
    ) -> DataFrame:
        """
        BOS - Break of Structure
        CHoCH - Change of Character
        these are both indications of market structure changing

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
        close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.

        returns:
        BOS = 1 if bullish break of structure, -1 if bearish break of structure
        CHOCH = 1 if bullish change of character, -1 if bearish change of character
        Level = the level of the break of structure or change of character
        BrokenIndex = the index of the candle that broke the level
        """

        # Make a copy to avoid modifying the original data
        swing_highs_lows = swing_highs_lows.copy()

        # Arrays to store the sequence of price levels and their high/low classification
        level_order = []  # Stores the actual price levels in sequence
        highs_lows_order = []  # Stores whether each level was a high (1) or low (-1)

        # Initialize arrays to store results
        bos = np.zeros(len(ohlc), dtype=np.int32)  # Break of Structure signals
        choch = np.zeros(len(ohlc), dtype=np.int32)  # Change of Character signals
        level = np.zeros(
            len(ohlc), dtype=np.float32
        )  # Price levels where signals occur
        last_positions = []  # Track positions of valid swing points

        # Iterate through each swing point
        for i in range(len(swing_highs_lows["HighLow"])):
            # Only process valid swing points (not NaN)
            if not np.isnan(swing_highs_lows["HighLow"][i]):
                # Add this swing point's level and type to our sequences
                level_order.append(swing_highs_lows["Level"][i])
                highs_lows_order.append(swing_highs_lows["HighLow"][i])

                # We need at least 4 swing points to identify patterns
                if len(level_order) >= 4:
                    # Check for bullish Break of Structure
                    # Pattern: Low -> High -> Low -> High
                    # Levels must be ascending: level[-4] < level[-2] < level[-3] < level[-1]
                    bos[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-4]
                                < level_order[-2]
                                < level_order[-3]
                                < level_order[-1]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # Check for bearish Break of Structure
                    # Pattern: High -> Low -> High -> Low
                    # Levels must be descending: level[-4] > level[-2] > level[-3] > level[-1]
                    bos[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-4]
                                > level_order[-2]
                                > level_order[-3]
                                > level_order[-1]
                            )
                        )
                        else bos[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3] if bos[last_positions[-2]] != 0 else 0
                    )

                    # Check for bullish Change of Character
                    # Pattern: Low -> High -> Low -> High
                    # Levels must follow: level[-1] > level[-3] > level[-4] > level[-2]
                    choch[last_positions[-2]] = (
                        1
                        if (
                            np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                            and np.all(
                                level_order[-1]
                                > level_order[-3]
                                > level_order[-4]
                                > level_order[-2]
                            )
                        )
                        else 0
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                    # Check for bearish Change of Character
                    # Pattern: High -> Low -> High -> Low
                    # Levels must follow: level[-1] < level[-3] < level[-4] < level[-2]
                    choch[last_positions[-2]] = (
                        -1
                        if (
                            np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                            and np.all(
                                level_order[-1]
                                < level_order[-3]
                                < level_order[-4]
                                < level_order[-2]
                            )
                        )
                        else choch[last_positions[-2]]
                    )
                    level[last_positions[-2]] = (
                        level_order[-3]
                        if choch[last_positions[-2]] != 0
                        else level[last_positions[-2]]
                    )

                # Keep track of this position for future reference
                last_positions.append(i)

        # Initialize array to track when signals get broken/confirmed
        broken = np.zeros(len(ohlc), dtype=np.int32)

        # Find all positions where we have either a BOS or CHOCH signal
        for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)

            # For bullish signals, check if price closes above the level
            if bos[i] == 1 or choch[i] == 1:
                mask = ohlc["close" if close_break else "high"][i + 2 :] > level[i]
            # For bearish signals, check if price closes below the level
            elif bos[i] == -1 or choch[i] == -1:
                mask = ohlc["close" if close_break else "low"][i + 2 :] < level[i]

            # If the level was broken, record when it happened
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                broken[i] = j

                # Clean up any older unbroken signals that this break invalidates
                for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                    if k < i and broken[k] >= j:
                        bos[k] = 0
                        choch[k] = 0
                        level[k] = 0

        # Remove any signals that never got broken/confirmed
        for i in np.where(
            np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
        )[0]:
            bos[i] = 0
            choch[i] = 0
            level[i] = 0

        # Convert zeros to NaN for cleaner output
        bos = np.where(bos != 0, bos, np.nan)
        choch = np.where(choch != 0, choch, np.nan)
        level = np.where(level != 0, level, np.nan)
        broken = np.where(broken != 0, broken, np.nan)

        # Convert arrays to pandas Series with descriptive names
        bos = pd.Series(bos, name="BOS")
        choch = pd.Series(choch, name="CHOCH")
        level = pd.Series(level, name="Level")
        broken = pd.Series(broken, name="BrokenIndex")

        # Return all signals as a single DataFrame
        return pd.concat([bos, choch, level, broken], axis=1)
