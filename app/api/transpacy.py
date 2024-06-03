from typing import Any
from fastapi import APIRouter

from app.models.clean_model import CleanTextRequest

import spacy
from spacy_cleaner import Cleaner
import importlib
import os


transpacy_router = APIRouter()

cleaner_module = importlib.import_module("spacy_cleaner.processing")


def import_clean_step(name):
    return getattr(cleaner_module, name)


path_model = os.path.join(os.path.dirname(__file__), "models/spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm"
                                                     "-3.7.1")
model = spacy.load(path_model)


@transpacy_router.post("/transpacy")
async def transform(req: CleanTextRequest) -> Any:
    """
    Preprocess given input text using steps
    """

    cleaners = map(import_clean_step, req.steps)
    pipeline = Cleaner(
        model,
        *cleaners
    )
    return pipeline.clean(req.input)
