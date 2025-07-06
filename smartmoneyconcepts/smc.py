from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime


def inputvalidator(input_="ohlc"):
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):
            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1

            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return dfcheck


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))

        return cls

    return decorate


@apply(inputvalidator(input_="ohlc"))
class smc:
    __version__ = "0.0.26"

    @classmethod
    def fvg(cls, ohlc: DataFrame, join_consecutive=False) -> DataFrame:
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.

        parameters:
        join_consecutive: bool - if there are multiple FVG in a row then they will be merged into one using the highest top and the lowest bottom

        returns:
        FVG = 1 if bullish fair value gap, -1 if bearish fair value gap
        Top = the top of the fair value gap
        Bottom = the bottom of the fair value gap
        MitigatedIndex = the index of the candle that mitigated the fair value gap
        """

        # Detect Fair Value Gaps
        # Bullish FVG: previous high < next low AND current candle is bullish
        # Bearish FVG: previous low > next high AND current candle is bearish
        fvg = np.where(
            (
                (ohlc["high"].shift(1) < ohlc["low"].shift(-1))  # Bullish gap condition
                & (ohlc["close"] > ohlc["open"])  # Confirm bullish candle
            )
            | (
                (ohlc["low"].shift(1) > ohlc["high"].shift(-1))  # Bearish gap condition
                & (ohlc["close"] < ohlc["open"])  # Confirm bearish candle
            ),
            np.where(
                ohlc["close"] > ohlc["open"], 1, -1
            ),  # 1 for bullish, -1 for bearish
            np.nan,  # No FVG detected
        )

        # Calculate the top of each FVG
        # For bullish FVG: next candle's low
        # For bearish FVG: previous candle's low
        top = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["low"].shift(-1),  # Bullish top
                ohlc["low"].shift(1),  # Bearish top
            ),
            np.nan,
        )

        # Calculate the bottom of each FVG
        # For bullish FVG: previous candle's high
        # For bearish FVG: next candle's high
        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                ohlc["close"] > ohlc["open"],
                ohlc["high"].shift(1),  # Bullish bottom
                ohlc["high"].shift(-1),  # Bearish bottom
            ),
            np.nan,
        )

        # Optionally merge consecutive FVGs
        if join_consecutive:
            for i in range(len(fvg) - 1):
                # If two consecutive FVGs are of the same type
                if fvg[i] == fvg[i + 1]:
                    # Take the highest top and lowest bottom
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    # Remove the first FVG since it's merged into the second
                    fvg[i] = top[i] = bottom[i] = np.nan

        # Track when each FVG gets mitigated (price returns to the gap)
        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=np.bool_)
            if fvg[i] == 1:  # Bullish FVG
                # Mitigated when price goes down to touch the top of the gap
                mask = ohlc["low"][i + 2 :] <= top[i]
            elif fvg[i] == -1:  # Bearish FVG
                # Mitigated when price goes up to touch the bottom of the gap
                mask = ohlc["high"][i + 2 :] >= bottom[i]
            # If mitigation found, record the first candle that did it
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        # Clean up mitigation indices for non-FVG candles
        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        # Return all components as a DataFrame
        return pd.concat(
            [
                pd.Series(fvg, name="FVG"),
                pd.Series(top, name="Top"),
                pd.Series(bottom, name="Bottom"),
                pd.Series(mitigated_index, name="MitigatedIndex"),
            ],
            axis=1,
        )

    @classmethod
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> DataFrame:
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

    @classmethod
    def bos_choch(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_break: bool = True
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

    @classmethod
    def ob(
        cls,
        ohlc: DataFrame,
        swing_highs_lows: DataFrame,
        close_mitigation: bool = False,
    ) -> DataFrame:
        """
        OB - Order Blocks
        This method detects order blocks when there is a high amount of market orders exist on a price range.

        parameters:
        swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
        close_mitigation: bool - if True then the order block will be mitigated based on the close of the candle otherwise it will be the high/low.

        returns:
        OB = 1 if bullish order block, -1 if bearish order block
        Top = top of the order block
        Bottom = bottom of the order block
        OBVolume = volume + 2 last volumes amounts
        Percentage = strength of order block (min(highVolume, lowVolume)/max(highVolume, lowVolume))
        """

        # Extract data into numpy arrays for faster processing
        ohlc_len = len(ohlc)
        _open = ohlc["open"].values
        _high = ohlc["high"].values
        _low = ohlc["low"].values
        _close = ohlc["close"].values
        _volume = ohlc["volume"].values
        swing_hl = swing_highs_lows["HighLow"].values

        # Initialize arrays for tracking order blocks and their properties
        crossed = np.full(
            ohlc_len, False, dtype=bool
        )  # Track if swing point is crossed
        ob = np.zeros(
            ohlc_len, dtype=np.int32
        )  # Order block type (1=bullish, -1=bearish)
        top_arr = np.zeros(ohlc_len, dtype=np.float32)  # Top of order block
        bottom_arr = np.zeros(ohlc_len, dtype=np.float32)  # Bottom of order block
        obVolume = np.zeros(ohlc_len, dtype=np.float32)  # Total volume of order block
        lowVolume = np.zeros(ohlc_len, dtype=np.float32)  # Lower volume component
        highVolume = np.zeros(ohlc_len, dtype=np.float32)  # Higher volume component
        percentage = np.zeros(ohlc_len, dtype=np.float32)  # Strength of order block
        mitigated_index = np.zeros(ohlc_len, dtype=np.int32)  # When OB gets mitigated
        breaker = np.full(ohlc_len, False, dtype=bool)  # Track if OB is broken

        # Get indices of swing highs and lows
        swing_high_indices = np.flatnonzero(swing_hl == 1)  # All swing highs
        swing_low_indices = np.flatnonzero(swing_hl == -1)  # All swing lows

        # Process bullish order blocks
        active_bullish = []  # Track active bullish OBs
        for i in range(ohlc_len):
            close_index = i
            # Update existing bullish order blocks
            for idx in active_bullish.copy():
                if breaker[idx]:
                    # If broken and price moves above top, invalidate the OB
                    if _high[close_index] > top_arr[idx]:
                        # Reset this OB's properties
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bullish.remove(idx)
                else:
                    # Check if price breaks below the bottom of OB
                    if (
                        not close_mitigation and _low[close_index] < bottom_arr[idx]
                    ) or (
                        close_mitigation
                        and min(_open[close_index], _close[close_index])
                        < bottom_arr[idx]
                    ):
                        breaker[idx] = True
                        mitigated_index[idx] = close_index - 1

            # Find the last swing high before current candle
            pos = np.searchsorted(swing_high_indices, close_index)
            last_top_index = swing_high_indices[pos - 1] if pos > 0 else None

            # Process potential new bullish order block
            if last_top_index is not None:
                # If price breaks above swing high and hasn't been processed
                if (
                    _close[close_index] > _high[last_top_index]
                    and not crossed[last_top_index]
                ):
                    crossed[last_top_index] = True
                    # Start with previous candle as default OB
                    default_index = close_index - 1
                    obBtm = _high[default_index]
                    obTop = _low[default_index]
                    obIndex = default_index
                    # Look for better OB candidate between swing high and current candle
                    if close_index - last_top_index > 1:
                        start = last_top_index + 1
                        end = close_index  # up to but not including close_index
                        if end > start:
                            segment = _low[start:end]
                            min_val = segment.min()
                            # Take the last occurrence of lowest low
                            candidates = np.nonzero(segment == min_val)[0]
                            if candidates.size:
                                candidate_index = start + candidates[-1]
                                obBtm = _low[candidate_index]
                                obTop = _high[candidate_index]
                                obIndex = candidate_index
                    # Record bullish order block properties
                    ob[obIndex] = 1
                    top_arr[obIndex] = obTop
                    bottom_arr[obIndex] = obBtm
                    # Calculate volume components
                    vol_cur = _volume[close_index]
                    vol_prev1 = _volume[close_index - 1] if close_index >= 1 else 0.0
                    vol_prev2 = _volume[close_index - 2] if close_index >= 2 else 0.0
                    obVolume[obIndex] = vol_cur + vol_prev1 + vol_prev2
                    lowVolume[obIndex] = vol_prev2
                    highVolume[obIndex] = vol_cur + vol_prev1
                    # Calculate strength percentage
                    max_vol = max(highVolume[obIndex], lowVolume[obIndex])
                    percentage[obIndex] = (
                        (min(highVolume[obIndex], lowVolume[obIndex]) / max_vol * 100.0)
                        if max_vol != 0
                        else 100.0
                    )
                    active_bullish.append(obIndex)

        # Process bearish order blocks (similar logic to bullish)
        active_bearish = []
        for i in range(ohlc_len):
            close_index = i
            # Update existing bearish OBs
            for idx in active_bearish.copy():
                if breaker[idx]:
                    if _low[close_index] < bottom_arr[idx]:
                        ob[idx] = 0
                        top_arr[idx] = 0.0
                        bottom_arr[idx] = 0.0
                        obVolume[idx] = 0.0
                        lowVolume[idx] = 0.0
                        highVolume[idx] = 0.0
                        mitigated_index[idx] = 0
                        percentage[idx] = 0.0
                        active_bearish.remove(idx)
                else:
                    if (not close_mitigation and _high[close_index] > top_arr[idx]) or (
                        close_mitigation
                        and max(_open[close_index], _close[close_index]) > top_arr[idx]
                    ):
                        breaker[idx] = True
                        mitigated_index[idx] = close_index

            # Find last swing low before current candle
            pos = np.searchsorted(swing_low_indices, close_index)
            last_btm_index = swing_low_indices[pos - 1] if pos > 0 else None

            # Process potential new bearish order block
            if last_btm_index is not None:
                if (
                    _close[close_index] < _low[last_btm_index]
                    and not crossed[last_btm_index]
                ):
                    crossed[last_btm_index] = True
                    default_index = close_index - 1
                    obTop = _high[default_index]
                    obBtm = _low[default_index]
                    obIndex = default_index
                    if close_index - last_btm_index > 1:
                        start = last_btm_index + 1
                        end = close_index
                        if end > start:
                            segment = _high[start:end]
                            max_val = segment.max()
                            candidates = np.nonzero(segment == max_val)[0]
                            if candidates.size:
                                candidate_index = start + candidates[-1]
                                obTop = _high[candidate_index]
                                obBtm = _low[candidate_index]
                                obIndex = candidate_index
                    # Record bearish order block properties
                    ob[obIndex] = -1
                    top_arr[obIndex] = obTop
                    bottom_arr[obIndex] = obBtm
                    # Calculate volume components
                    vol_cur = _volume[close_index]
                    vol_prev1 = _volume[close_index - 1] if close_index >= 1 else 0.0
                    vol_prev2 = _volume[close_index - 2] if close_index >= 2 else 0.0
                    obVolume[obIndex] = vol_cur + vol_prev1 + vol_prev2
                    lowVolume[obIndex] = vol_cur + vol_prev1
                    highVolume[obIndex] = vol_prev2
                    # Calculate strength percentage
                    max_vol = max(highVolume[obIndex], lowVolume[obIndex])
                    percentage[obIndex] = (
                        (min(highVolume[obIndex], lowVolume[obIndex]) / max_vol * 100.0)
                        if max_vol != 0
                        else 100.0
                    )
                    active_bearish.append(obIndex)

        # Clean up arrays by replacing zeros with NaN where no OB exists
        ob = np.where(ob != 0, ob, np.nan)
        top_arr = np.where(~np.isnan(ob), top_arr, np.nan)
        bottom_arr = np.where(~np.isnan(ob), bottom_arr, np.nan)
        obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
        mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
        percentage = np.where(~np.isnan(ob), percentage, np.nan)

        # Convert arrays to pandas Series with descriptive names
        ob_series = pd.Series(ob, name="OB")
        top_series = pd.Series(top_arr, name="Top")
        bottom_series = pd.Series(bottom_arr, name="Bottom")
        obVolume_series = pd.Series(obVolume, name="OBVolume")
        mitigated_index_series = pd.Series(mitigated_index, name="MitigatedIndex")
        percentage_series = pd.Series(percentage, name="Percentage")

        # Return all components as a DataFrame
        return pd.concat(
            [
                ob_series,
                top_series,
                bottom_series,
                obVolume_series,
                mitigated_index_series,
                percentage_series,
            ],
            axis=1,
        )

    @classmethod
    def liquidity(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01
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

    @classmethod
    def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> DataFrame:
        """
        Previous High Low
        This method returns the previous high and low of the given time frame.

        parameters:
        time_frame: str - the time frame to get the previous high and low 15m, 1H, 4H, 1D, 1W, 1M

        returns:
        PreviousHigh = the previous high
        PreviousLow = the previous low
        BrokenHigh = 1 once price has broken the previous high of the timeframe, 0 otherwise
        BrokenLow = 1 once price has broken the previous low of the timeframe, 0 otherwise
        """

        # Convert index to datetime for resampling
        ohlc.index = pd.to_datetime(ohlc.index)

        # Initialize arrays to store results
        previous_high = np.zeros(len(ohlc), dtype=np.float32)  # Previous period's high
        previous_low = np.zeros(len(ohlc), dtype=np.float32)  # Previous period's low
        broken_high = np.zeros(len(ohlc), dtype=np.int32)  # High break status
        broken_low = np.zeros(len(ohlc), dtype=np.int32)  # Low break status

        # Resample data to desired timeframe
        # For each period, take:
        # - First price as open
        # - Highest price as high
        # - Lowest price as low
        # - Last price as close
        # - Sum of volume
        resampled_ohlc = (
            ohlc.resample(time_frame)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        # Track break status for each period
        currently_broken_high = False
        currently_broken_low = False
        last_broken_time = None

        # Process each candle
        for i in range(len(ohlc)):
            # Find all periods that completed before this candle
            resampled_previous_index = np.where(resampled_ohlc.index < ohlc.index[i])[0]

            # Skip if we don't have enough history
            if len(resampled_previous_index) <= 1:
                previous_high[i] = np.nan
                previous_low[i] = np.nan
                continue

            # Get the second-to-last completed period
            resampled_previous_index = resampled_previous_index[-2]

            # Reset break status when we enter a new period
            if last_broken_time != resampled_previous_index:
                currently_broken_high = False
                currently_broken_low = False
                last_broken_time = resampled_previous_index

            # Get previous period's high and low
            previous_high[i] = resampled_ohlc["high"].iloc[resampled_previous_index]
            previous_low[i] = resampled_ohlc["low"].iloc[resampled_previous_index]

            # Update break status
            # Once broken, stays broken for the rest of the period
            currently_broken_high = (
                ohlc["high"].iloc[i] > previous_high[i] or currently_broken_high
            )
            currently_broken_low = (
                ohlc["low"].iloc[i] < previous_low[i] or currently_broken_low
            )

            # Record current break status
            broken_high[i] = 1 if currently_broken_high else 0
            broken_low[i] = 1 if currently_broken_low else 0

        # Convert arrays to pandas Series with descriptive names
        previous_high = pd.Series(previous_high, name="PreviousHigh")
        previous_low = pd.Series(previous_low, name="PreviousLow")
        broken_high = pd.Series(broken_high, name="BrokenHigh")
        broken_low = pd.Series(broken_low, name="BrokenLow")

        # Return all components as a DataFrame
        return pd.concat([previous_high, previous_low, broken_high, broken_low], axis=1)

    @classmethod
    def sessions(
        cls,
        ohlc: DataFrame,
        session: str,
        start_time: str = "",
        end_time: str = "",
        time_zone: str = "UTC",
    ) -> DataFrame:
        """
        Sessions
        This method returns wwhich candles are within the session specified

        parameters:
        session: str - the session you want to check (Sydney, Tokyo, London, New York, Asian kill zone, London open kill zone, New York kill zone, london close kill zone, Custom)
        start_time: str - the start time of the session in the format "HH:MM" only required for custom session.
        end_time: str - the end time of the session in the format "HH:MM" only required for custom session.
        time_zone: str - the time zone of the candles can be in the format "UTC+0" or "GMT+0"

        returns:
        Active = 1 if the candle is within the session, 0 if not
        High = the highest point of the session
        Low = the lowest point of the session
        """

        # Validate custom session parameters
        if session == "Custom" and (start_time == "" or end_time == ""):
            raise ValueError("Custom session requires a start and end time")

        # Define standard market sessions and their times (in UTC)
        default_sessions = {
            "Sydney": {
                "start": "21:00",
                "end": "06:00",
            },
            "Tokyo": {
                "start": "00:00",
                "end": "09:00",
            },
            "London": {
                "start": "07:00",
                "end": "16:00",
            },
            "New York": {
                "start": "13:00",
                "end": "22:00",
            },
            "Asian kill zone": {
                "start": "00:00",
                "end": "04:00",
            },
            "London open kill zone": {
                "start": "6:00",
                "end": "9:00",
            },
            "New York kill zone": {
                "start": "11:00",
                "end": "14:00",
            },
            "london close kill zone": {
                "start": "14:00",
                "end": "16:00",
            },
            "Custom": {
                "start": start_time,
                "end": end_time,
            },
        }

        # Convert index to datetime and handle timezone
        ohlc.index = pd.to_datetime(ohlc.index)
        if time_zone != "UTC":
            # Convert GMT/UTC+X format to Etc/GMT format for timezone
            time_zone = time_zone.replace("GMT", "Etc/GMT")
            time_zone = time_zone.replace("UTC", "Etc/GMT")
            # Convert from local timezone to UTC
            ohlc.index = ohlc.index.tz_localize(time_zone).tz_convert("UTC")

        # Get session times and convert to datetime objects for comparison
        session_start = datetime.strptime(default_sessions[session]["start"], "%H:%M")
        session_end = datetime.strptime(default_sessions[session]["end"], "%H:%M")

        # Initialize arrays to store results
        active = np.zeros(len(ohlc), dtype=np.int32)  # Session active status
        high = np.zeros(len(ohlc), dtype=np.float32)  # Session high
        low = np.zeros(len(ohlc), dtype=np.float32)  # Session low

        # Process each candle
        for i in range(len(ohlc)):
            # Get current candle's time
            current_time = datetime.strptime(ohlc.index[i].strftime("%H:%M"), "%H:%M")

            # Check if current time is within session
            # Handle both normal sessions and sessions that cross midnight
            is_in_session = (
                # Normal session (e.g., 09:00-17:00)
                (
                    session_start < session_end
                    and session_start <= current_time <= session_end
                )
                or
                # Session crossing midnight (e.g., 21:00-06:00)
                (
                    session_start >= session_end
                    and (session_start <= current_time or current_time <= session_end)
                )
            )

            if is_in_session:
                active[i] = 1  # Mark as active session
                # Update session high (max of current high and previous session high)
                high[i] = max(ohlc["high"].iloc[i], high[i - 1] if i > 0 else 0)
                # Update session low (min of current low and previous session low)
                # Use infinity for initial comparison to ensure first value is captured
                low[i] = min(
                    ohlc["low"].iloc[i],
                    low[i - 1] if i > 0 and low[i - 1] != 0 else float("inf"),
                )

        # Convert arrays to pandas Series with descriptive names
        active = pd.Series(active, name="Active")
        high = pd.Series(high, name="High")
        low = pd.Series(low, name="Low")

        # Return all components as a DataFrame
        return pd.concat([active, high, low], axis=1)

    @classmethod
    def retracements(cls, ohlc: DataFrame, swing_highs_lows: DataFrame) -> DataFrame:
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
