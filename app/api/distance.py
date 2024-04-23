import textdistance
from fastapi import APIRouter

from app.models.str_distance import Algorithm, DistanceResponse

distance_router = APIRouter()

@distance_router.get("/distance/{algorithm}")
async def distance(method: Algorithm, line1: str, line2: str) -> DistanceResponse:
    """
    Повертає відстань між двома рядками використовуючи заданий алгоритм
    """
    algo = method.value
    fun = getattr(textdistance, algo)
    similarity = fun(line1, line2)
    return DistanceResponse(
        method=algo,
        line1=line1,
        line2=line2,
        similarity=similarity
    )
