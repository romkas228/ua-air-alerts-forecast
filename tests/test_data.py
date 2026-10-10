from pathlib import Path

import pandas as pd
import pytest

from uaalerts.data import sources
from uaalerts.data.panel import build_panel, merge_intervals
from uaalerts.data.regions import BY_NAME_EN, BY_NAME_UK, REGION_IDS
from uaalerts.data.stitch import stitch

FIX = Path(__file__).parent / "fixtures"
UTC = "UTC"
BREAK = pd.Timestamp("2026-09-06 00:00", tz=UTC)


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


@pytest.fixture
def hist() -> pd.DataFrame:
    return sources.parse_vadimkin(FIX / "vadimkin_sample.csv")


@pytest.fixture
def fresh() -> pd.DataFrame:
    return sources.parse_alerts_in_ua(sources.list_snapshots(FIX, sources.ALERTS_IN_UA))


def test_region_mappings_are_complete_and_unique() -> None:
    assert len(set(REGION_IDS)) == len(REGION_IDS) == 26
    assert sorted(BY_NAME_EN.values()) == sorted(REGION_IDS)
    assert sorted(BY_NAME_UK.values()) == sorted(REGION_IDS)


def test_parsers_produce_utc_schema(hist: pd.DataFrame, fresh: pd.DataFrame) -> None:
    assert set(hist["region"]) == {"kyiv_city", "kharkivska"}
    assert hist["alert_level"].isna().all()
    assert set(fresh["region"]) == {"luhanska", "kyiv_city", "kharkivska"}
    assert fresh.loc[fresh["source_id"] == "1", "finished_at"].isna().all()  # ongoing
    assert str(fresh["started_at"].dt.tz) == UTC


def test_unmapped_region_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "oblast,raion,hromada,level,started_at,finished_at,source\n"
        "Atlantis oblast,,,oblast,2026-01-01 00:00:00+00:00,2026-01-01 01:00:00+00:00,x\n"
    )
    with pytest.raises(ValueError, match="unmapped"):
        sources.parse_vadimkin(bad)


def test_newer_snapshot_wins(tmp_path: Path) -> None:
    header = (
        "id,location_uid,location_type,location_title,location_oblast,location_raion,"
        "started_at,finished_at,alert_type,alert_level,threats\n"
    )
    row = "7,31,oblast,м. Київ,м. Київ,,2026-09-10T10:00:00Z,{fin},air_raid,red,\n"
    (tmp_path / "alerts_in_ua_mirror__20260910T103000Z.csv").write_text(header + row.format(fin=""))
    (tmp_path / "alerts_in_ua_mirror__20260910T120000Z.csv").write_text(
        header + row.format(fin="2026-09-10T11:00:00Z")
    )
    df = sources.parse_alerts_in_ua(sources.list_snapshots(tmp_path, sources.ALERTS_IN_UA))
    assert len(df) == 1
    assert df["finished_at"].iloc[0] == ts("2026-09-10 11:00")


def test_merge_intervals_unions_overlaps() -> None:
    s = pd.Series([ts("2026-01-01 00:00"), ts("2026-01-01 00:30"), ts("2026-01-01 02:00")])
    e = pd.Series([ts("2026-01-01 01:00"), ts("2026-01-01 01:30"), ts("2026-01-01 02:10")])
    m = merge_intervals(s, e)
    assert list(m["start"]) == [ts("2026-01-01 00:00"), ts("2026-01-01 02:00")]
    assert list(m["end"]) == [ts("2026-01-01 01:30"), ts("2026-01-01 02:10")]


def test_panel_minutes_and_targets(hist: pd.DataFrame) -> None:
    p = build_panel(hist, ts("2026-09-06 10:00"), ts("2026-09-06 14:00"), BREAK)
    kyiv = p[p["region"] == "kyiv_city"].set_index("hour")
    assert kyiv["alert_minutes"].tolist() == [30, 60, 15, 0]
    assert kyiv["y_any"].tolist() == [1, 1, 1, 0]
    # Overlapping raion + oblast alerts are unioned, not double counted: 11:00-11:40.
    kh = p[p["region"] == "kharkivska"].set_index("hour")
    assert kh.loc[ts("2026-09-06 11:00"), "alert_minutes"] == 40
    # Regions without events are present with zeros.
    assert p[p["region"] == "lvivska"]["y_any"].sum() == 0
    assert p["region"].nunique() == 26


def test_panel_red_is_na_before_regime_break(fresh: pd.DataFrame) -> None:
    p = build_panel(fresh, ts("2026-09-05 22:00"), ts("2026-09-06 14:00"), BREAK)
    assert p.loc[p["hour"] < BREAK, "y_red"].isna().all()
    kh = p[(p["region"] == "kharkivska") & (p["hour"] == ts("2026-09-06 13:00"))]
    assert kh["y_red"].iloc[0] == 1


def test_panel_only_counts_air_raids(fresh: pd.DataFrame) -> None:
    p = build_panel(fresh, ts("2026-09-06 09:00"), ts("2026-09-06 10:00"), BREAK)
    assert p.loc[p["region"] == "kharkivska", "y_any"].sum() == 0  # artillery only


def test_panel_ongoing_alert_is_clipped_at_end(fresh: pd.DataFrame) -> None:
    end = ts("2026-09-06 14:00")
    p = build_panel(fresh, ts("2026-09-06 12:00"), end, BREAK)
    assert p["hour"].max() < end
    lu = p[p["region"] == "luhanska"]
    assert (lu["alert_minutes"] == 60).all()


def test_stitch_takes_each_source_on_its_side(hist: pd.DataFrame, fresh: pd.DataFrame) -> None:
    at = ts("2026-09-06 12:00")
    out = stitch(hist, fresh, at=at)
    assert (out.loc[out["source"] == sources.VADIMKIN, "started_at"] < at).all()
    after = out[out["source"] == sources.ALERTS_IN_UA]
    # fresh rows: either start at/after the stitch point or are still ongoing (Luhansk)
    assert ((after["started_at"] >= at) | after["finished_at"].isna()).all()
    assert "1" in set(after["source_id"])
    assert "4" not in set(after["source_id"])  # finished before stitch point -> dropped


def test_observed_until_uses_data_not_download_time(fresh: pd.DataFrame) -> None:
    assert sources.observed_until(fresh) == ts("2026-09-06 13:30")
