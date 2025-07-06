from enum import Enum


class SessionType(Enum):
    SYDNEY = "Sydney"
    TOKYO = "Tokyo"
    LONDON = "London"
    NEW_YORK = "New York"
    ASIAN_KILL_ZONE = "Asian kill zone"
    LONDON_OPEN_KILL_ZONE = "London open kill zone"
    NEW_YORK_KILL_ZONE = "New York kill zone"
    LONDON_CLOSE_KILL_ZONE = "london close kill zone"
    CUSTOM = "Custom"


class TimeFrame(Enum):
    MINUTES_15 = "15m"
    HOUR_1 = "1H"
    HOUR_4 = "4H"
    DAY_1 = "1D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"


DEFAULT_SESSIONS = {
    SessionType.SYDNEY: {
        "start": "21:00",
        "end": "06:00",
    },
    SessionType.TOKYO: {
        "start": "00:00",
        "end": "09:00",
    },
    SessionType.LONDON: {
        "start": "07:00",
        "end": "16:00",
    },
    SessionType.NEW_YORK: {
        "start": "13:00",
        "end": "22:00",
    },
    SessionType.ASIAN_KILL_ZONE: {
        "start": "00:00",
        "end": "04:00",
    },
    SessionType.LONDON_OPEN_KILL_ZONE: {
        "start": "6:00",
        "end": "9:00",
    },
    SessionType.NEW_YORK_KILL_ZONE: {
        "start": "11:00",
        "end": "14:00",
    },
    SessionType.LONDON_CLOSE_KILL_ZONE: {
        "start": "14:00",
        "end": "16:00",
    },
}
