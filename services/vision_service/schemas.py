from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class StrictBaseModel(BaseModel):
    """Reject unknown fields so schema drift is caught during development."""

    class Config:
        extra = "forbid"


class Evidence(StrictBaseModel):
    """File-level evidence produced by one vision pipeline run."""

    image_path: str
    annotated_image_path: str | None = None
    groundingdino_json_path: str | None = None
    pipeline_json_path: str | None = None
    detection_evidence_paths: list[str] = Field(default_factory=list)


class VisionDetection(StrictBaseModel):
    """Normalized object observation after detection, segmentation, and checks."""

    detection_id: int | None = None
    part: str
    label: str
    status: Literal["ok", "missing", "wrong_part", "low_confidence", "ambiguous", "not_checked"]
    confidence: float | None
    segmentation_confidence: float | None = None
    bbox: list[float] | None = None
    roi: str | None = None
    roi_status: Literal["inside", "outside", "not_configured", "not_checked"] = "not_checked"
    mask_path: str | None = None
    crop_path: str | None = None
    overlay_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class VisionDefect(StrictBaseModel):
    """Conservative defect-like observation that requires downstream handling."""

    defect_type: str
    part: str | None = None
    severity: Literal["low", "medium", "high"]
    confidence: float | None
    evidence_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class VisionResult(StrictBaseModel):
    """Human-readable and audit-friendly output from the vision service."""

    schema_version: str = "1.0"
    work_order_id: str
    station_id: str
    step_id: str
    camera_id: str
    view_id: str
    image_id: str
    model_name: str
    model_version: str
    lighting_profile: str | None = None
    detections: list[VisionDetection] = Field(default_factory=list)
    defects: list[VisionDefect] = Field(default_factory=list)
    step_status: Literal["passed", "failed", "needs_confirmation", "not_checked"]
    step_status_confidence: float | None
    evidence: Evidence
    timestamp: str
    raw: dict[str, Any] = Field(default_factory=dict)


class EventModel(StrictBaseModel):
    """Model identity stored in the common event envelope."""

    name: str
    version: str


class EventRule(StrictBaseModel):
    """Rule identity stored in the common event envelope."""

    rule_id: str
    rule_version: str


class EventMedia(StrictBaseModel):
    """Media references attached to an event-engine-facing output."""

    audio_path: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    annotated_image_paths: list[str] = Field(default_factory=list)


class OperatorConfirmation(StrictBaseModel):
    """Confirmation state requested by the vision observation."""

    required: bool
    confirmed: bool | None = None
    confirmed_by: str | None = None
    confirmed_at: str | None = None


class VisionEvent(StrictBaseModel):
    """Event-engine-facing wrapper around VisionResult."""

    schema_version: str = "1.0"
    event_id: str
    parent_event_id: str | None = None
    correlation_id: str
    causation_id: str | None = None
    idempotency_key: str
    timestamp: str
    work_order_id: str
    station_id: str
    operator_id: str | None = None
    source: Literal["VISION"] = "VISION"
    event_type: str = "VISION_STEP_CHECKED"
    step_id: str
    payload: dict[str, Any]
    confidence: float | None
    confidence_unavailable_reason: str | None = None
    model: EventModel
    rule: EventRule
    media: EventMedia
    operator_confirmation: OperatorConfirmation
