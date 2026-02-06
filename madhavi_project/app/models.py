from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, constr

AssessmentType = Literal[
    "Urgent Referral",
    "Urgent Investigation",
    "No Urgent Action",
    "Insufficient Evidence",
]

ConfidenceType = Literal["high", "medium", "low"]


class Citation(BaseModel):
    source: str = Field(..., description="Source identifier")
    page: int = Field(..., ge=1)
    chunk_id: str
    excerpt: str


class AgentOutput(BaseModel):
    assessment: AssessmentType
    reasoning: constr(strip_whitespace=True)
    citations: List[Citation]
    confidence: ConfidenceType


class ErrorInfo(BaseModel):
    code: constr(strip_whitespace=True)
    message: constr(strip_whitespace=True)


class AssessRequest(BaseModel):
    patient_id: constr(strip_whitespace=True, min_length=1, max_length=64)


class ChatMessage(BaseModel):
    role: constr(strip_whitespace=True, min_length=1, max_length=16)
    content: constr(strip_whitespace=True, min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=1, max_length=4000)
    history: Optional[List[ChatMessage]] = None


class AssessResponse(BaseModel):
    correlation_id: str
    disclaimer: str
    result: AgentOutput
    status: constr(strip_whitespace=True, min_length=1) = "ok"
    errors: Optional[List[ErrorInfo]] = None


class ChatResponse(BaseModel):
    correlation_id: str
    disclaimer: str
    result: AgentOutput
    status: constr(strip_whitespace=True, min_length=1) = "ok"
    errors: Optional[List[ErrorInfo]] = None
