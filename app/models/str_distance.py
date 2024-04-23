from enum import Enum
from pydantic import BaseModel, Field, StrictStr

Algorithm = Enum('Algorithm', dict(
    path1="hamming",
    path2="mlipns",
    path3="levenshtein",
    path4="damerau_levenshtein",
    path5="jaro_winkler",
    path6="strcmp95",
    path7="needleman_wunsch",
    path8="gotoh",
    path9="smith_waterman",
))

class DistanceResponse(BaseModel):
    method: Algorithm = Field(..., title="method", description="Один з алгоритмів, який підтримує textdistance", example="hamming")
    line1: StrictStr = Field(..., title="line1", description="Перша послідовність символів", example="book")
    line2: StrictStr = Field(..., title="line2", description="Друга послідовність символів", example="cook")
    similarity: float = Field(..., title="similarity", description="Близькість між двома рядками", example=2)