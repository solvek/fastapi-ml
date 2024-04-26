from typing import List, Any, Optional
from pydantic import BaseModel, Field, StrictStr


class Step(BaseModel):
    transformer: StrictStr = Field(..., title="transformer", description="A transformer to be applied",
                                   example="Transformer name")
    params: Optional[dict] = None


class TransformRequest(BaseModel):
    input: Any = Field(..., title="input", description="Input value", example="For example a text")
    steps: List[Step] = Field(..., title="steps", description="List of transformers")
