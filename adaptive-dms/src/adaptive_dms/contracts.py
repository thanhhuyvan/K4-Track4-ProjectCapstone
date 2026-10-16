"""Stable data contracts shared by edge inference and cloud personalization."""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class EventSource(StrEnum):
    GLOBAL = "global"
    OWNER_ADAPTER = "owner_adapter"
    FALLBACK = "fallback"


class DriverEvent(BaseModel):
    """A temporally-smoothed prediction produced on the vehicle."""

    event_type: str = Field(
        pattern="^(attentive|visual_distraction|manual_distraction|drowsy|unknown)$"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    source: EventSource
    timestamp_ms: int = Field(ge=0)
    model_version: str


class OwnerAdapter(BaseModel):
    """Small personal artifact; never changes the frozen global encoder."""

    owner_id: UUID
    base_model_version: str
    adapter_version: str
    attention_threshold: float = Field(ge=0.0, le=1.0)
    min_event_duration_ms: int = Field(ge=0, default=500)
