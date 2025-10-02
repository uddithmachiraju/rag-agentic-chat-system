from pydantic import BaseModel
from typing import Any, Dict, Optional

class CoordinatorRequest(BaseModel): 
    action: str
    payload: Optional[Dict[str, Any]] = {}


class CoordinatorResponse(BaseModel):
    success: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None