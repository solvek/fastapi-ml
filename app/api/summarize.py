from typing import Any

from fastapi import APIRouter
from transformers import pipeline

summarize_router = APIRouter()
# hf_token = os.environ['HF_TOKEN']
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", auth_token=hf_token)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


@summarize_router.post("/summarize")
async def summarize(text: str) -> Any:
    return summarizer(text, max_length=150, min_length=40)
