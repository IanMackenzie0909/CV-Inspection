from __future__ import annotations

import hashlib
from typing import Any

from .schemas import EventMedia, EventModel, EventRule, OperatorConfirmation, VisionEvent, VisionResult


def deterministic_suffix(*values: str) -> str:
    """Create a short repeatable suffix for event IDs in prototype runs."""
    digest = hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()
    return digest[:12].upper()


def build_vision_event(
    *,
    vision_result: VisionResult,
    rule_version: str,
    operator_id: str | None,
) -> VisionEvent:
    """Wrap a VisionResult in the event envelope expected by the workflow layer."""
    suffix = deterministic_suffix(
        vision_result.work_order_id,
        vision_result.station_id,
        vision_result.step_id,
        vision_result.image_id,
        vision_result.timestamp,
    )
    annotated_paths = []
    if vision_result.evidence.annotated_image_path:
        annotated_paths.append(vision_result.evidence.annotated_image_path)

    payload: dict[str, Any] = vision_result.dict()
    # Failed or uncertain visual observations request confirmation. The event
    # engine decides how that confirmation affects workflow state.
    operator_confirmation_required = vision_result.step_status in {"failed", "needs_confirmation"}
    return VisionEvent(
        event_id=f"EVT-{suffix}",
        parent_event_id=None,
        correlation_id=f"CORR-{vision_result.work_order_id}",
        causation_id=None,
        idempotency_key=(
            f"{vision_result.work_order_id}:{vision_result.step_id}:"
            f"VISION_STEP_CHECKED:{vision_result.image_id}:v1"
        ),
        timestamp=vision_result.timestamp,
        work_order_id=vision_result.work_order_id,
        station_id=vision_result.station_id,
        operator_id=operator_id,
        step_id=vision_result.step_id,
        payload=payload,
        confidence=vision_result.step_status_confidence,
        confidence_unavailable_reason=None if vision_result.step_status_confidence is not None else "NO_DETECTIONS_OR_RULES",
        model=EventModel(name=vision_result.model_name, version=vision_result.model_version),
        rule=EventRule(rule_id="RULE-VISION-STEP-CHECK-v1", rule_version=rule_version),
        media=EventMedia(
            audio_path=None,
            image_paths=[vision_result.evidence.image_path],
            annotated_image_paths=annotated_paths,
        ),
        operator_confirmation=OperatorConfirmation(required=operator_confirmation_required),
    )
