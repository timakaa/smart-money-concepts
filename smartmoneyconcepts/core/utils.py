import pandas as pd
from pandas import DataFrame


def validate_ohlc_data(ohlc: DataFrame) -> DataFrame:
    """
    Validates and standardizes OHLC data format.

    Args:
        ohlc (DataFrame): DataFrame containing OHLC data

    Returns:
        DataFrame: Standardized OHLC data with lowercase column names
    """
    # Convert column names to lowercase
    ohlc = ohlc.rename(columns={c: c.lower() for c in ohlc.columns})

    # Check required columns
    required_columns = ["open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in ohlc.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return ohlc


def convert_timezone(df: DataFrame, from_tz: str, to_tz: str = "UTC") -> DataFrame:
    """
    Converts DataFrame index timezone from one timezone to another.

    Args:
        df (DataFrame): DataFrame with datetime index
        from_tz (str): Source timezone (e.g., "UTC", "GMT+0")
        to_tz (str): Target timezone (defaults to "UTC")

    Returns:
        DataFrame: DataFrame with converted timezone
    """
    # Convert GMT/UTC+X format to Etc/GMT format
    from_tz = from_tz.replace("GMT", "Etc/GMT").replace("UTC", "Etc/GMT")
    to_tz = to_tz.replace("GMT", "Etc/GMT").replace("UTC", "Etc/GMT")

    # Convert timezone
    df.index = pd.to_datetime(df.index)
    return df.tz_localize(from_tz).tz_convert(to_tz)
