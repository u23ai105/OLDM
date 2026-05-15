from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Cinematic Restoration"
    
    # Database
    DATABASE_URL: str = "postgresql+psycopg2://postgres:password@localhost:5432/restoration_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Storage
    RAW_STORAGE_PATH: str = "./data/raw"
    PROCESSED_STORAGE_PATH: str = "./data/processed"
    MODEL_WEIGHTS_PATH: str = "./weights"
    
    # GPU
    USE_GPU: bool = True
    VRAM_LIMIT_GB: int = 8
    
    # Prefect
    PREFECT_API_URL: Optional[str] = None
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
