from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    tenant_id: str
    user_id: str
    user_text: str
    system_prompt: str = "You are a helpful assistant."
    policy_version: str = "v1"
    doc_version: str = "v1"
    cache_threshold: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    from_cache: bool
    model_used: str
    cache_similarity: Optional[float] = None
    pii_redacted: bool = False
    safety_blocked: bool = False
    citations: Optional[List[Dict[str, Any]]] = None
