"""Live alert state from siren.pp.ua (keyless mirror of the ukrainealarm API).

Used only for display ("what is happening right now") - never as a model input, because
it is not part of the historical data the model was evaluated on.
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from uaalerts.data.regions import BY_NAME_UK, CENTER_LOCATIONS, REGION_IDS

TIMEOUT = 20


def _get(base: str, path: str) -> Any:
    r = httpx.get(f"{base}/{path}", timeout=TIMEOUT, headers={"Accept-Encoding": "gzip"})
    r.raise_for_status()
    return r.json()


def region_index(base: str) -> dict[str, tuple[str, str, str]]:
    """siren regionId -> (our region id, region type, name)."""
    index: dict[str, tuple[str, str, str]] = {}

    def walk(node: dict[str, Any], region: str | None) -> None:
        name = node["regionName"]
        region = BY_NAME_UK.get(name, region)
        if region is None:  # top-level city entries such as "м. Харків та ..."
            region = _region_of_city(name)
        if region is not None:
            index[str(node["regionId"])] = (region, node["regionType"], name)
        for child in node.get("regionChildIds") or []:
            walk(child, region)

    for state in _get(base, "regions")["states"]:
        walk(state, None)
    return index


def _region_of_city(name: str) -> str | None:
    for region, locs in CENTER_LOCATIONS.items():
        if any(loc.startswith("м. ") and name.startswith(loc) for loc in locs):
            return region
    return None


def _covers_center(region: str, rtype: str, name: str) -> bool:
    if rtype == "State" or region == "kyiv_city":
        return True
    locs = CENTER_LOCATIONS[region]
    if name in locs:
        return True
    # Communities are named "м. Харків та Харківська територіальна громада".
    return any(loc.startswith("м. ") and name.startswith(loc + " ") for loc in locs)


def current_state(base: str) -> pd.DataFrame:
    """One row per region: air alert covering the centre now / red / anywhere in oblast."""
    index = region_index(base)
    rows = {
        r: {"center_alert": False, "center_red": False, "oblast_alert": False} for r in REGION_IDS
    }
    updated = None
    for entry in _get(base, "alerts"):
        hit = index.get(str(entry["regionId"]))
        if hit is None:
            continue
        region, rtype, name = hit
        for alert in entry.get("activeAlerts", []):
            if alert.get("type") != "AIR":
                continue
            levels = {lvl.get("alertLevel") for lvl in alert.get("activeAlertLevels", [])}
            rows[region]["oblast_alert"] = True
            if _covers_center(region, rtype, name):
                rows[region]["center_alert"] = True
                rows[region]["center_red"] |= "Red" in levels
        last = entry.get("lastUpdate")
        updated = max(updated or last, last) if last else updated
    df = pd.DataFrame.from_dict(rows, orient="index").rename_axis("region").reset_index()
    df["fetched_at"] = pd.Timestamp.now(tz="UTC")
    return df
