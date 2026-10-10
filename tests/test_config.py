from typer.testing import CliRunner

from uaalerts.cli import app
from uaalerts.config import get_config


def test_config_is_consistent() -> None:
    cfg = get_config()
    assert cfg.max_lead_hours == 168
    assert all(0 <= h < 24 for h in cfg.origin_hours)
    # Lead buckets must tile 1..max_lead without gaps or overlaps.
    covered = sorted(h for lo, hi in cfg.lead_buckets.values() for h in range(lo, hi + 1))
    assert covered == list(range(1, cfg.max_lead_hours + 1))
    # The embargo must cover the full horizon, otherwise train labels leak into eval.
    assert cfg.embargo_hours >= cfg.max_lead_hours


def test_cli_info() -> None:
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "max lead:   168 h" in result.output
