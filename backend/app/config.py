import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Automatically load .env file if present in BASE_DIR or backend directory
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    backend_env = BASE_DIR / "backend" / ".env"
    if backend_env.exists():
        load_dotenv(dotenv_path=backend_env)

# Ensure data directory exists
data_dir = BASE_DIR / "data"
data_dir.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_PATH = (data_dir / "incidentiq.db").resolve()

def get_cors_origins() -> list[str]:
    cors_env = os.getenv("CORS_ORIGINS")
    origins = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://10.106.114.125:5173",
        "http://10.106.114.125:3000",
        "https://memoryops-frontend.onrender.com",
    ]
    if cors_env:
        for o in cors_env.split(","):
            cleaned = o.strip().rstrip("/")
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
    return origins

class Settings(BaseModel):
    PROJECT_NAME: str = "MemoryOps API"
    VERSION: str = "0.1.0"
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")
    CORS_ORIGINS: list[str] = Field(default_factory=get_cors_origins)
    HINDSIGHT_API_URL: str = os.getenv("HINDSIGHT_API_URL", "https://api.hindsight.vectorize.io")
    HINDSIGHT_API_KEY: str | None = os.getenv("HINDSIGHT_API_KEY", None)
    HINDSIGHT_BANK_ID: str = os.getenv("HINDSIGHT_BANK_ID", "incidentiq")
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY", None)
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

settings = Settings()
