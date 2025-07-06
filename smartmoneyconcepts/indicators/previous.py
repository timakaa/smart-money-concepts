import numpy as np
import pandas as pd
from pandas import DataFrame


class PreviousHighLow:
    """Previous High Low indicator implementation"""

    @staticmethod
    def calculate(ohlc: DataFrame, time_frame: str = "1D") -> DataFrame:
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
