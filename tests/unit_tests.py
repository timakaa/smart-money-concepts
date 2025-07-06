# this file will be used to test the functionality and accuracy of all the indicators in the smartmoneyconcepts package

import os
import sys
import time
import pandas as pd
import unittest

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from smartmoneyconcepts.core import SMC
from smartmoneyconcepts.core.utils import validate_ohlc_data

# define and import test data
test_instrument = "EURUSD"
instrument_data = f"{test_instrument}_15M.csv"
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data", test_instrument)
df = pd.read_csv(os.path.join(TEST_DATA_DIR, instrument_data))
df = df.set_index("Date")
df.index = pd.to_datetime(df.index)
df = validate_ohlc_data(df)


class TestSmartMoneyConcepts(unittest.TestCase):
    # to test each function in the smartmoneyconcepts package
    # each function will be called and the result will be compared to the result data

    def test_fvg(self):
        start_time = time.time()
        fvg_data = SMC.fvg(df)
        fvg_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fvg_result_data.csv")
        )
        print("fvg test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(fvg_data, fvg_result_data, check_dtype=False)

    def test_fvg_consecutive(self):
        start_time = time.time()
        fvg_data = SMC.fvg(df, join_consecutive=True)
        fvg_consecutive_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "fvg_consecutive_result_data.csv")
        )
        print("fvg consecutive test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            fvg_data, fvg_consecutive_result_data, check_dtype=False
        )

    def test_swing_highs_lows(self):
        start_time = time.time()
        swing_highs_lows_data = SMC.swing_highs_lows(df, swing_length=5)
        swing_highs_lows_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "swing_highs_lows_result_data.csv")
        )
        print("swing_highs_lows test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            swing_highs_lows_data, swing_highs_lows_result_data, check_dtype=False
        )

    def test_bos_choch(self):
        start_time = time.time()
        swing_highs_lows_data = SMC.swing_highs_lows(df, swing_length=5)
        bos_choch_data = SMC.bos_choch(df, swing_highs_lows_data)
        bos_choch_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "bos_choch_result_data.csv")
        )
        print("bos_choch test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            bos_choch_data, bos_choch_result_data, check_dtype=False
        )

    def test_ob(self):
        start_time = time.time()
        swing_highs_lows_data = SMC.swing_highs_lows(df, swing_length=5)
        ob_data = SMC.ob(df, swing_highs_lows_data)
        ob_result_data = pd.read_csv(os.path.join(TEST_DATA_DIR, "ob_result_data.csv"))
        print("ob test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(ob_data, ob_result_data, check_dtype=False)

    def test_ob_early_data(self):
        """Ensure early candles do not cause index errors in OB calculation."""
        short_df = pd.DataFrame(
            {
                "open": [1.0, 1.1, 1.2],
                "high": [1.05, 1.15, 1.25],
                "low": [0.95, 1.05, 1.15],
                "close": [1.02, 1.14, 1.24],
                "volume": [5, 6, 7],
            }
        )
        short_df = validate_ohlc_data(short_df)
        swing = SMC.swing_highs_lows(short_df, swing_length=1)
        ob_df = SMC.ob(short_df, swing)
        self.assertEqual(len(ob_df), len(short_df))

    def test_liquidity(self):
        start_time = time.time()
        swing_highs_lows_data = SMC.swing_highs_lows(df, swing_length=5)
        liquidity_data = SMC.liquidity(df, swing_highs_lows_data)
        liquidity_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "liquidity_result_data.csv")
        )
        print("liquidity test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            liquidity_data, liquidity_result_data, check_dtype=False
        )

    def test_previous_high_low(self):
        # test 4h time frame
        start_time = time.time()
        previous_high_low_data = SMC.previous_high_low(df, time_frame="4h")
        previous_high_low_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "previous_high_low_result_data_4h.csv")
        )
        print("previous_high_low test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            previous_high_low_data, previous_high_low_result_data, check_dtype=False
        )

        # test 1D time frame
        start_time = time.time()
        previous_high_low_data = SMC.previous_high_low(df, time_frame="1D")
        previous_high_low_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "previous_high_low_result_data_1D.csv")
        )
        print("previous_high_low test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            previous_high_low_data, previous_high_low_result_data, check_dtype=False
        )

        # test W time frame
        start_time = time.time()
        previous_high_low_data = SMC.previous_high_low(df, time_frame="W")
        previous_high_low_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "previous_high_low_result_data_W.csv")
        )
        print("previous_high_low test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            previous_high_low_data, previous_high_low_result_data, check_dtype=False
        )

    def test_sessions(self):
        start_time = time.time()
        sessions = SMC.sessions(df, session="London")
        sessions_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "sessions_result_data.csv")
        )
        print("sessions test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(sessions, sessions_result_data, check_dtype=False)

    def test_retracements(self):
        start_time = time.time()
        swing_highs_lows_data = SMC.swing_highs_lows(df, swing_length=5)
        retracements = SMC.retracements(df, swing_highs_lows_data)
        retracements_result_data = pd.read_csv(
            os.path.join(TEST_DATA_DIR, "retracements_result_data.csv")
        )
        print("retracements test time: ", time.time() - start_time)
        pd.testing.assert_frame_equal(
            retracements, retracements_result_data, check_dtype=False
        )


if __name__ == "__main__":
    unittest.main()


def generate_results_data():
    fvg_data = SMC.fvg(df)
    fvg_data.to_csv(
        os.path.join("test_data", test_instrument, "fvg_result_data.csv"), index=False
    )

    fvg_data = SMC.fvg(df, join_consecutive=True)
    fvg_data.to_csv(
        os.path.join("test_data", test_instrument, "fvg_consecutive_result_data.csv"),
        index=False,
    )

    swing_highs_lows_data = SMC.swing_highs_lows(df, swing_length=5)
    swing_highs_lows_data.to_csv(
        os.path.join("test_data", test_instrument, "swing_highs_lows_result_data.csv"),
        index=False,
    )

    bos_choch_data = SMC.bos_choch(df, swing_highs_lows_data)
    bos_choch_data.to_csv(
        os.path.join("test_data", test_instrument, "bos_choch_result_data.csv"),
        index=False,
    )

    ob_data = SMC.ob(df, swing_highs_lows_data)
    ob_data.to_csv(
        os.path.join("test_data", test_instrument, "ob_result_data.csv"), index=False
    )

    liquidity_data = SMC.liquidity(df, swing_highs_lows_data)
    liquidity_data.to_csv(
        os.path.join("test_data", test_instrument, "liquidity_result_data.csv"),
        index=False,
    )

    previous_high_low_data = SMC.previous_high_low(df, time_frame="4h")
    previous_high_low_data.to_csv(
        os.path.join(
            "test_data", test_instrument, "previous_high_low_result_data_4h.csv"
        ),
        index=False,
    )

    previous_high_low_data = SMC.previous_high_low(df, time_frame="1D")
    previous_high_low_data.to_csv(
        os.path.join(
            "test_data", test_instrument, "previous_high_low_result_data_1D.csv"
        ),
        index=False,
    )

    previous_high_low_data = SMC.previous_high_low(df, time_frame="W")
    previous_high_low_data.to_csv(
        os.path.join(
            "test_data", test_instrument, "previous_high_low_result_data_W.csv"
        ),
        index=False,
    )

    sessions = SMC.sessions(df, session="London")
    sessions.to_csv(
        os.path.join("test_data", test_instrument, "sessions_result_data.csv"),
        index=False,
    )

    retracements = SMC.retracements(df, swing_highs_lows_data)
    retracements.to_csv(
        os.path.join("test_data", test_instrument, "retracements_result_data.csv"),
        index=False,
    )


generate_results_data()
