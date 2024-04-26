from typing import Any
from fastapi import APIRouter

from app.models.pipeline import TransformRequest

transform_router = APIRouter()

@transform_router.post("/transform")
async def predict(pipeline: TransformRequest) -> Any:
    """
    Transforms the input with the provided pipeline
    """
    return "ok"