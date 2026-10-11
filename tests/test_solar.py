import numpy as np
import pandas as pd

from uaalerts.features.solar import solar_elevation

KYIV = (np.array([50.45]), np.array([30.52]))


def elev(ts: str) -> float:
    return float(solar_elevation(pd.DatetimeIndex([pd.Timestamp(ts, tz="UTC")]), *KYIV)[0])


def test_kyiv_solar_noon_summer_and_winter() -> None:
    # Solar noon in Kyiv is ~09:58 UTC; max elevation = 90 - lat + declination.
    assert abs(elev("2026-06-21 10:00") - (90 - 50.45 + 23.44)) < 1.0
    assert abs(elev("2026-12-21 10:00") - (90 - 50.45 - 23.44)) < 1.0


def test_night_is_negative() -> None:
    assert elev("2026-06-21 22:00") < 0  # 01:00 Kyiv
    assert elev("2026-12-21 16:00") < 0  # 18:00 Kyiv, after sunset in winter
    assert elev("2026-06-21 16:00") > 0  # 19:00 Kyiv, still light in summer
