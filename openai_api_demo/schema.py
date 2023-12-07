from typing import List
from pydantic import BaseModel

class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
    object: str

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    object: str
    usage: EmbeddingUsage
    