import numpy as np
import pandas as pd
from pandas import DataFrame


class FairValueGap:
    """Fair Value Gap (FVG) indicator implementation"""

    @staticmethod
    def calculate(ohlc: DataFrame, join_consecutive=False) -> DataFrame:
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
