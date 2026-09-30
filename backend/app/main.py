from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import logging

from app.config import settings
from app.database import get_db, check_database_health, engine, Base, SessionLocal
from app.seed import seed_incidents
from app.routers import incidents

logger = logging.getLogger("incidentiq.main")

# Create tables
Base.metadata.create_all(bind=engine)

# Auto-seed database with initial incidents
with SessionLocal() as db_session:
    seed_incidents(db_session)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="MemoryOps API - AI-powered incident response assistant for DevOps/SRE engineers",
)

# Register CORSMiddleware before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception during request {request.method} {request.url}: {exc}")
    origin = request.headers.get("origin")
    headers = {}
    if origin and (origin in settings.CORS_ORIGINS or ".onrender.com" in origin):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Vary"] = "Origin"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers=headers,
    )

app.include_router(incidents.router)
app.include_router(incidents.legacy_router)

class HealthResponse(BaseModel):
    status: str
    database: str
    version: str

@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    db_ok = check_database_health()
    if not db_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "database": "disconnected", "version": settings.VERSION}
        )
    return {
        "status": "healthy",
        "database": "connected",
        "version": settings.VERSION,
    }

@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs"
    }
