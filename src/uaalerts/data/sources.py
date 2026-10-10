"""Downloading raw snapshots and parsing each source into the unified event schema."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from uaalerts.data.regions import BY_NAME_EN, BY_NAME_UK
from uaalerts.data.schema import validate_events

log = logging.getLogger(__name__)

VADIMKIN = "vadimkin_official"
ALERTS_IN_UA = "alerts_in_ua_mirror"


def download_snapshot(url: str, raw_dir: Path, name: str, now: datetime | None = None) -> Path:
    """Download `url` into `raw_dir/<name>__<UTC timestamp>.csv` and return the path.

    Snapshots are never overwritten: the alerts.in.ua mirror only keeps ~1 month of
    history, so older snapshots are the only record of earlier periods.
    """
    now = now or datetime.now(UTC)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{name}__{now:%Y%m%dT%H%M%SZ}.csv"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    log.info("saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def snapshot_time(path: Path) -> pd.Timestamp:
    """Parse the UTC download time encoded in a snapshot filename."""
    stamp = path.stem.split("__")[-1]
    return pd.Timestamp(datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC))


def observed_until(events: pd.DataFrame) -> pd.Timestamp:
    """Latest moment the source is known to have observed: its newest start/finish stamp.

    A mirror may have been synced long before we downloaded it, so the download time
    would overstate coverage (and turn the unsynced tail into fake "no alert" hours).
    """
    return pd.Timestamp(max(events["started_at"].max(), events["finished_at"].max()))


def list_snapshots(raw_dir: Path, name: str) -> list[Path]:
    return sorted(raw_dir.glob(f"{name}__*.csv"))


def _map_regions(names: pd.Series, mapping: dict[str, str]) -> pd.Series:
    region = names.map(mapping)
    unknown = sorted(names[region.isna()].dropna().unique())
    if unknown:
        raise ValueError(f"unmapped region names: {unknown}")
    return region


def parse_vadimkin(path: Path) -> pd.DataFrame:
    """Vadimkin `official_data_en.csv`: oblast/raion/hromada air-raid alerts, UTC."""
    raw = pd.read_csv(path, dtype="string")
    df = pd.DataFrame(
        {
            "region": _map_regions(raw["oblast"], BY_NAME_EN),
            "raion": raw["raion"],
            "location": raw["hromada"].fillna(raw["raion"]).fillna(raw["oblast"]),
            "geo_level": raw["level"],
            "alert_type": "air_raid",
            "alert_level": pd.Series(pd.NA, index=raw.index, dtype="string"),
            "started_at": pd.to_datetime(raw["started_at"], utc=True, format="ISO8601"),
            "finished_at": pd.to_datetime(raw["finished_at"], utc=True, format="ISO8601"),
            "source": VADIMKIN,
            "source_id": pd.Series(pd.NA, index=raw.index, dtype="string"),
        }
    )
    return validate_events(df)


def parse_alerts_in_ua(paths: list[Path]) -> pd.DataFrame:
    """alerts.in.ua history mirror (JustJimer). Several snapshots are merged by alert id,
    the newest snapshot winning (an alert may be ongoing in an older one)."""
    if not paths:
        raise FileNotFoundError("no alerts.in.ua mirror snapshots")
    frames = []
    for p in sorted(paths):
        frame = pd.read_csv(p, dtype="string")
        frame["_snapshot"] = snapshot_time(p) if "__" in p.stem else pd.Timestamp(0, tz="UTC")
        frames.append(frame)
    raw = (
        pd.concat(frames, ignore_index=True)
        .sort_values("_snapshot")
        .drop_duplicates("id", keep="last")
        .reset_index(drop=True)
    )
    df = pd.DataFrame(
        {
            "region": _map_regions(raw["location_oblast"], BY_NAME_UK),
            "raion": raw["location_raion"],
            "location": raw["location_title"],
            "geo_level": raw["location_type"],
            "alert_type": raw["alert_type"],
            "alert_level": raw["alert_level"],
            "started_at": pd.to_datetime(raw["started_at"], utc=True, format="ISO8601"),
            "finished_at": pd.to_datetime(raw["finished_at"], utc=True, format="ISO8601"),
            "source": ALERTS_IN_UA,
            "source_id": raw["id"],
        }
    )
    return validate_events(df)
