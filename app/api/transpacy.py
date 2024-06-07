from typing import Any
from fastapi import APIRouter

from app.models.spacy_pipe import SpacyPipeRequest

import spacy
from spacy.language import Language
from spacy.tokens import Doc
import os

transpacy_router = APIRouter()


@Language.factory('replace')
class Replace(object):
    # nlp: Language

    def __init__(self, nlp: Language, name: str, old: str, new: str):
        self.nlp = nlp
        self.name = name
        self.old = old
        self.new = new

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        return self.nlp.make_doc(text.replace(self.old, self.new))


@Language.factory('capitalize')
class Capitalize(object):
    # nlp: Language

    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc) -> Doc:
        text = doc.text
        return self.nlp.make_doc(text.upper())


path_model = os.path.join(os.path.dirname(__file__), "models/spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm"
                                                     "-3.7.1")


@transpacy_router.post("/transpacy")
async def transform(req: SpacyPipeRequest) -> Any:
    """
    Preprocess given input text using steps
    """

    d = req.disable
    if d is None:
        d = []

    nlp = spacy.load(path_model, disable=d)
    # print(req)
    for component in req.components:
        p = component.params
        if p is None:
            p = {}

        if component.after is None:
            nlp.add_pipe(component.factory, first=True, config=p)
        else:
            nlp.add_pipe(component.factory, after=component.after, config=p)

    # print(nlp.pipeline)

    doc = nlp(req.input)

    return str(doc)
