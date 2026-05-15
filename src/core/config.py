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
    
    # GPU — Tuned for 40GB VRAM (allocating 35GB, 5GB headroom for OS/driver)
    USE_GPU: bool = True
    VRAM_LIMIT_GB: int = 35
    CUDA_VISIBLE_DEVICES: str = "0"
    
    # Restoration Settings (DeOldify) — High VRAM config
    DEOLDIFY_RENDER_FACTOR: int = 40        # Max quality (default=21, capped at ~44)
    DEOLDIFY_WATERMARK: bool = False
    
    # Upscaling Settings (Real-ESRGAN) — High VRAM config
    ESRGAN_SCALE: int = 4                   # 4x upscale (can go to 8x with enough VRAM)
    ESRGAN_TILE_SIZE: int = 512             # Larger tiles = faster (default=256, 512 needs ~20GB+)
    ESRGAN_TILE_PAD: int = 32
    ESRGAN_BATCH_SIZE: int = 8              # Process 8 frames in parallel
    ESRGAN_FP16: bool = True                # Half-precision cuts VRAM usage ~50%
    
    # Shorts Generation
    SHORTS_NUM_CLIPS: int = 5               # Generate 5 shorts per movie
    SHORTS_DURATION_SEC: int = 45           # Each short is 45 seconds
    SHORTS_MIN_SCORE: float = 0.6           # Minimum saliency score to qualify
    
    # Prefect
    PREFECT_API_URL: Optional[str] = None
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

settings = Settings()
