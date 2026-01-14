"""Configuration management for EDDT."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Simulation settings
    simulation_tick_duration_minutes: int = Field(default=15, alias="SIMULATION_TICK_DURATION_MINUTES")
    working_hours_only: bool = Field(default=True, alias="WORKING_HOURS_ONLY")
    working_hours_start: str = Field(default="08:00", alias="WORKING_HOURS_START")
    working_hours_end: str = Field(default="17:00", alias="WORKING_HOURS_END")

    # LLM Model paths
    tier1_model_path: str = Field(
        default="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        alias="TIER1_MODEL_PATH",
    )
    tier2_model_path: str = Field(
        default="models/qwen2.5-7b-instruct-q4_k_m.gguf",
        alias="TIER2_MODEL_PATH",
    )

    # LLM Configuration
    tier1_n_ctx: int = Field(default=2048, alias="TIER1_N_CTX")
    tier2_n_ctx: int = Field(default=4096, alias="TIER2_N_CTX")
    tier1_n_gpu_layers: int = Field(default=-1, alias="TIER1_N_GPU_LAYERS")
    tier2_n_gpu_layers: int = Field(default=-1, alias="TIER2_N_GPU_LAYERS")

    # LLM Service URLs (for separate containers)
    tier1_url: Optional[str] = Field(default=None, alias="TIER1_URL")
    tier2_url: Optional[str] = Field(default=None, alias="TIER2_URL")

    # Database
    postgres_url: str = Field(
        default="postgresql://user:pass@localhost:5432/eddt",
        alias="POSTGRES_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # Cache settings
    cache_db_path: str = Field(default="decision_cache.db", alias="CACHE_DB_PATH")
    cache_max_entries: int = Field(default=10000, alias="CACHE_MAX_ENTRIES")
    cache_similarity_threshold: float = Field(default=0.95, alias="CACHE_SIMILARITY_THRESHOLD")

    # API settings
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Cloud LLM (Tier 3) - reserved for future
    cloud_api_key: Optional[str] = Field(default=None, alias="CLOUD_API_KEY")
    cloud_model: str = Field(default="claude-3-haiku-20240307", alias="CLOUD_MODEL")
    tier3_budget_per_simulation: float = Field(default=10.0, alias="TIER3_BUDGET_PER_SIMULATION")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=True, alias="LOG_JSON")


# Global settings instance
settings = Settings()
