from typing import Any
from fastapi import APIRouter

from app.models.pipeline import TransformRequest
from sklearn.pipeline import make_pipeline
import nltk

transform_router = APIRouter()

# Downloading NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)


class StandardizeLetters:
    """
    Стандартизація літер, тобто переведення в нижній регістр
    """

    @staticmethod
    def transform(s):
        return s.lower()

def create_transformer(step):
    klass = globals()[step.transformer]
    return klass()


@transform_router.post("/transform")
async def transform(req: TransformRequest) -> Any:
    """
    Transforms the input with the provided pipeline
    """

    steps = map(create_transformer, req.steps)
    pipeline = make_pipeline(*steps)
    return pipeline.transform(req.input)
