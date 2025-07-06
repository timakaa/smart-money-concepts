from pandas import DataFrame
from .base import apply, inputvalidator
from ..indicators.fvg import FairValueGap
from ..indicators.swing import SwingHighsLows
from ..indicators.structure import BreakOfStructure
from ..indicators.orderblock import OrderBlock
from ..indicators.liquidity import Liquidity
from ..indicators.previous import PreviousHighLow
from ..indicators.sessions import Sessions
from ..indicators.retracement import Retracement


@apply(inputvalidator(input_="ohlc"))
class SMC:
    __version__ = "0.0.26"

    @classmethod
    def fvg(cls, ohlc: DataFrame, join_consecutive=False) -> DataFrame:
        return FairValueGap.calculate(ohlc, join_consecutive)

    @classmethod
    def swing_highs_lows(cls, ohlc: DataFrame, swing_length: int = 50) -> DataFrame:
        return SwingHighsLows.calculate(ohlc, swing_length)

    @classmethod
    def bos_choch(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_break: bool = True
    ) -> DataFrame:
        return BreakOfStructure.calculate(ohlc, swing_highs_lows, close_break)

    @classmethod
    def ob(
        cls,
        ohlc: DataFrame,
        swing_highs_lows: DataFrame,
        close_mitigation: bool = False,
    ) -> DataFrame:
        return OrderBlock.calculate(ohlc, swing_highs_lows, close_mitigation)

    @classmethod
    def liquidity(
        cls, ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent: float = 0.01
    ) -> DataFrame:
        return Liquidity.calculate(ohlc, swing_highs_lows, range_percent)

    @classmethod
    def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> DataFrame:
        return PreviousHighLow.calculate(ohlc, time_frame)

    @classmethod
    def sessions(
        cls,
        ohlc: DataFrame,
        session: str,
        start_time: str = "",
        end_time: str = "",
        time_zone: str = "UTC",
    ) -> DataFrame:
        return Sessions.calculate(ohlc, session, start_time, end_time, time_zone)

    @classmethod
    def retracements(cls, ohlc: DataFrame, swing_highs_lows: DataFrame) -> DataFrame:
        return Retracement.calculate(ohlc, swing_highs_lows)
