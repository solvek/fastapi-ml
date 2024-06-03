from typing import List

from pydantic import Field, BaseModel


class CleanTextRequest(BaseModel):
    input: str = Field(..., title="input", description="Input value", example="Inulinases are used for the production of high-fructose syrup 456 and fructooligosaccharides, and are widely utilized in food and pharmaceutical industries. In this study,")
    steps: List[str] = Field(..., title="steps", description="List of transformers")