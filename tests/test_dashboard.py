from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "forecast.parquet").exists(),
    reason="needs a local forecast (uaalerts update --retrain)",
)
@pytest.mark.network
def test_dashboard_runs_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=120).run()
    assert not at.exception
    at.radio[0].set_value("y_red").run()
    at.radio[1].set_value("p_climatology").run()
    assert not at.exception
