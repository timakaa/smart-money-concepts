import numpy as np
import pandas as pd
from pandas import DataFrame


class OrderBlock:
    """Order Block indicator implementation"""

    @staticmethod
    def calculate(
        ohlc: DataFrame, swing_highs_lows: DataFrame, close_mitigation: bool = False
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
