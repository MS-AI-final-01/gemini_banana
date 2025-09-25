import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis 캐시 설정
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_DEFAULT_TTL: int = 1800  # 30분
    CACHE_ENABLED: bool = True

    # PostgreSQL 설정
    POSTGRES_URL: str = ""

    # 앱 설정
    APP_NAME: str = "Virtual Fitting API with Redis Cache"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False

    # 기존 Virtual Fitting 설정들
    AZURE_CUSTOM_VISION_ENDPOINT: str = ""
    AZURE_CUSTOM_VISION_KEY: str = ""
    AZURE_STORAGE_CONNECTION_STRING: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
