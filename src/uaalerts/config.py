"""Project settings: forecasting constants from YAML, secrets/paths from env (.env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Sources(BaseModel):
    vadimkin_official_url: str
    alerts_in_ua_mirror_url: str
    siren_api_url: str


class ForecastConfig(BaseModel):
    timezone: str
    origin_hours: list[int]
    max_lead_hours: int
    lead_buckets: dict[str, tuple[int, int]]
    test_weeks: int
    val_weeks: int
    embargo_hours: int
    regime_break: str
    sources: Sources


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UAALERTS_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    config_path: Path = PROJECT_ROOT / "configs" / "default.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def load_forecast_config(path: Path) -> ForecastConfig:
    with path.open(encoding="utf-8") as f:
        return ForecastConfig.model_validate(yaml.safe_load(f))


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> ForecastConfig:
    return load_forecast_config(get_settings().config_path)
