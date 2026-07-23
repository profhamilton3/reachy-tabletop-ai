"""Pydantic schema for VLM planner output. No raw joint values allowed."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, field_validator


class TaskType(str, Enum):
    point = "point"
    pick_place = "pick_place"
    handover = "handover"
    sort = "sort"


class Destination(str, Enum):
    left_tray = "left_tray"
    right_tray = "right_tray"
    handover_zone = "handover_zone"
    point_only = "point_only"


class Plan(BaseModel):
    task_type: TaskType
    target_id: str
    target_description: str
    destination: Destination
    confidence: float
    requires_confirmation: bool
    safety_notes: list[str] = []
    brief_reason: str

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    @field_validator("target_id")
    @classmethod
    def target_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("target_id must not be empty")
        return v
