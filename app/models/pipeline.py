from typing import List, Any, Optional
from pydantic import BaseModel, Field, StrictStr


class Step(BaseModel):
    transformer: StrictStr = Field(..., title="transformer", description="A transformer to be applied",
                                   example="Transformer name")
    params: Optional[dict] = None


class TransformRequest(BaseModel):
    input: Any = Field(..., title="input", description="Input value", example="Inulinases are used for the production of high-fructose syrup 456 and fructooligosaccharides, and are widely utilized in food and pharmaceutical industries. In this study,")
    steps: List[Step] = Field(..., title="steps", description="List of transformers")
