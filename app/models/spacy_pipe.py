from typing import List, Optional, Any

from pydantic import Field, BaseModel, StrictStr


class Component(BaseModel):
    factory: StrictStr = Field(..., title="factory", description="A transformer to be applied",
                               example="Transformer name")
    after: Optional[str] = None
    params: Optional[dict] = None


class SpacyPipeRequest(BaseModel):
    input: StrictStr = Field(..., title="input", description="Input text",
                       example="Inulinases are used for the production of high-fructose syrup 456 and fructooligosaccharides, and are widely utilized in food and pharmaceutical industries. In this study,")
    components: List[Component] = Field(..., title="components", description="List of transformers")
    disable: Optional[List[str]] = None
