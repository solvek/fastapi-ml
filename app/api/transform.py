from typing import Any
from fastapi import APIRouter

from app.models.pipeline import TransformRequest

transform_router = APIRouter()

from sklearn.pipeline import make_pipeline, Pipeline


class ParseInt:
    """
    Тестовий трансформер, який переводить текст в число
    """

    @staticmethod
    def transform(s):
        return int(s)


class IncrementInt:
    """
    Тестовий трансформер, який збільшує число на 1
    """

    @staticmethod
    def transform(n):
        return n+1


@transform_router.post("/transform")
async def predict(req: TransformRequest) -> Any:
    """
    Transforms the input with the provided pipeline
    """

    steps = []
    for step in req.steps:
        klass = globals()[step.transformer]
        t = klass()
        steps.append(t)

    pipeline = make_pipeline(*steps)

    return pipeline.transform(req.input)
