import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from ..types.enums import SessionType, DEFAULT_SESSIONS


class Sessions:
    """Sessions indicator implementation"""

    @staticmethod
    def calculate(
        ohlc: DataFrame,
        session: str,
        start_time: str = "",
        end_time: str = "",
        time_zone: str = "UTC",
    ) -> DataFrame:
        """
        Sessions
        This method returns which candles are within the session specified

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

        # Get session times from enums
        session_times = DEFAULT_SESSIONS.get(
            SessionType(session),
            {
                "start": start_time,
                "end": end_time,
            },
        )

        # Convert index to datetime and handle timezone
        ohlc.index = pd.to_datetime(ohlc.index)
        if time_zone != "UTC":
            # Convert GMT/UTC+X format to Etc/GMT format for timezone
            time_zone = time_zone.replace("GMT", "Etc/GMT")
            time_zone = time_zone.replace("UTC", "Etc/GMT")
            # Convert from local timezone to UTC
            ohlc.index = ohlc.index.tz_localize(time_zone).tz_convert("UTC")

        # Get session times and convert to datetime objects for comparison
        session_start = datetime.strptime(session_times["start"], "%H:%M")
        session_end = datetime.strptime(session_times["end"], "%H:%M")

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
