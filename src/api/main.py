from fastapi import FastAPI, Depends, HTTPException
from src.core.config import settings
from src.core.logging import logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

@app.get("/")
async def root():
    return {"message": "AI Cinematic Restoration API", "version": "1.0.0"}

@app.post(f"{settings.API_V1_STR}/restoration/start")
async def start_restoration(input_path: str, upscale: bool = True):
    logger.info(f"Received restoration request for {input_path}")
    # Here we would trigger the Prefect flow
    return {"status": "queued", "task_id": "mock-uuid"}

@app.get(f"{settings.API_V1_STR}/restoration/status/{{task_id}}")
async def get_status(task_id: str):
    return {"task_id": task_id, "status": "processing", "progress": 45}
