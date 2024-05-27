from typing import List

from fastapi import APIRouter
from gensim.models.doc2vec import Doc2Vec
from gensim.utils import simple_preprocess
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import re

nltk.download('stopwords')
stop_words = stopwords.words('english')


def data_cleaning(text):

    # Convert to lower
    text = text.lower()

    remove_stopwords = [word for word in text.split() if word not in stop_words]
    text = ' '.join(remove_stopwords)

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    text = re.sub(r"\d", '', text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text


doc2vec_router = APIRouter()

labels = ['sport', 'tech', 'business', 'entertainment', 'politics']

file_path = os.path.join(os.path.dirname(__file__), "models", "doc2vec_model")
model = Doc2Vec.load(file_path)

vectors = list(map(lambda l: model.dv[l], labels))


def nearest_label(v):
    s = -1
    for i, tag in enumerate(vectors):
        sn = np.dot(tag, v)
        if sn > s:
            s = sn
            idx = i
    return labels[idx]


@doc2vec_router.post("/doc2vec")
async def transform(sentences: List[str]):
    result = []
    for s in sentences:
        cleaned = data_cleaning(s)
        words = simple_preprocess(cleaned, deacc=True)
        v = model.infer_vector(words)
        l = nearest_label(v)
        result.append([s, l])

    return result
