from typing import Any
from fastapi import APIRouter

from app.models.pipeline import TransformRequest
from sklearn.pipeline import make_pipeline
import nltk

import re
from collections import Counter
from nltk.corpus import stopwords

transform_router = APIRouter()

# Downloading NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords')


class StandardizeLetters:
    """
    Стандартизація літер, тобто переведення в нижній регістр
    """

    @staticmethod
    def transform(s):
        return s.lower()


class RemovePunctuation:
    """
    Корекція пунктуації
    """

    @staticmethod
    def transform(text):
        return re.sub(r'[^\w\s]', '', text)


class RemoveNumbers:
    @staticmethod
    def transform(text):
        return re.sub(r'\d', '', text)


class RemoveRareWords:
    def __init__(self, threshold=5):
        # print("Threshold is", threshold)
        self.threshold = threshold

    def transform(self, text):
        words = nltk.word_tokenize(text)
        word_freq = Counter(words)

        filtered_words = [word for word in words if word_freq[word] >= self.threshold]

        return ' '.join(filtered_words)


class Tokenize:
    @staticmethod
    def transform(text):
        return nltk.word_tokenize(text)


class FilterStopWords:
    def __init__(self, language="english"):
        # print("Threshold is", threshold)
        self.stop_words = set(stopwords.words(language))

    def transform(self, tokens):
        return [token for token in tokens if token not in self.stop_words]


def create_transformer(step):
    klass = globals()[step.transformer]
    p = step.params
    if p is None:
        return klass()

    return klass(**step.params)


@transform_router.post("/transform")
async def transform(req: TransformRequest) -> Any:
    """
    Transforms the input with the provided pipeline
    """

    steps = map(create_transformer, req.steps)
    pipeline = make_pipeline(*steps)
    return pipeline.transform(req.input)
