from __future__ import annotations

from typing import Any

from .config import PartsLexicon, expected_parts_for_step
from .schemas import VisionDetection


def _thresholds(safety_rules: dict[str, Any]) -> dict[str, float]:
    """Read confidence thresholds with safe defaults for early MVP configs."""
    confidence_thresholds = safety_rules.get("confidence_thresholds") or {}
    return {
        "vision_high": float(confidence_thresholds.get("vision_high", 0.9)),
        "vision_low": float(confidence_thresholds.get("vision_low", 0.7)),
        "vision_detection_min": float(confidence_thresholds.get("vision_detection_min", 0.3)),
        "segmentation_low": float(confidence_thresholds.get("segmentation_low", 0.7)),
    }


def _bbox_center_inside_roi(
    bbox_xyxy: list[float] | None,
    roi_name: str | None,
    camera_config: dict[str, Any],
    camera_id: str,
    view_id: str,
    image_size: dict[str, Any],
) -> str:
    """Check whether the detection center falls inside the configured ROI."""
    if bbox_xyxy is None or not roi_name:
        return "not_checked"

    camera = None
    for candidate in camera_config.get("cameras", []) or []:
        if candidate.get("camera_id") == camera_id and candidate.get("view_id") == view_id:
            camera = candidate
            break
    if not camera:
        return "not_configured"

    roi = (camera.get("rois") or {}).get(roi_name)
    if not roi:
        return "not_configured"

    width = float(image_size.get("width", 0) or 0)
    height = float(image_size.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return "not_checked"

    # ROIs are stored as normalized coordinates so the same config survives
    # different camera resolutions.
    rx1, ry1, rx2, ry2 = [float(value) for value in roi.get("bbox_xyxy_norm", [0.0, 0.0, 1.0, 1.0])]
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    inside = (rx1 * width) <= center_x <= (rx2 * width) and (ry1 * height) <= center_y <= (ry2 * height)
    return "inside" if inside else "outside"


def normalize_detection_candidates(
    dino_result: dict[str, Any],
    sam_results: list[dict[str, Any]],
    lexicon: PartsLexicon,
) -> list[dict[str, Any]]:
    """Merge raw detection and segmentation outputs into candidate objects."""
    sam_by_detection_id = {int(item["detection_id"]): item for item in sam_results}
    candidates: list[dict[str, Any]] = []
    for raw_detection in dino_result.get("detections", []) or []:
        detection_id = int(raw_detection["id"])
        sam_entry = sam_by_detection_id.get(detection_id, {})
        sam_output = sam_entry.get("output") or {}
        mask = sam_output.get("mask") or {}
        label = str(raw_detection.get("label", "unknown"))
        candidates.append(
            {
                "detection_id": detection_id,
                "part": lexicon.part_key_for_label(label),
                "label": label,
                "confidence": raw_detection.get("confidence"),
                "segmentation_confidence": mask.get("score"),
                "bbox": raw_detection.get("bbox_xyxy"),
                "mask_bbox": mask.get("bbox_xyxy"),
                "mask_area_ratio": mask.get("area_ratio"),
                "crop_path": raw_detection.get("crop_image_path"),
                "mask_path": sam_output.get("mask_image_path"),
                "overlay_path": sam_output.get("result_image_path"),
                "raw_detection": raw_detection,
                "raw_segmentation": sam_output,
            }
        )
    return candidates


def check_step(
    *,
    dino_result: dict[str, Any],
    sam_results: list[dict[str, Any]],
    workflow_config: dict[str, Any],
    safety_rules: dict[str, Any],
    camera_config: dict[str, Any],
    parts_lexicon: PartsLexicon,
    step_id: str,
    text_prompt: str,
    camera_id: str,
    view_id: str,
) -> tuple[list[VisionDetection], str, float | None, list[str]]:
    """Compare model candidates against the expected parts for one SOP step.

    This function does not decide whether production can proceed. It only
    converts visual evidence into a workflow-oriented observation for the event
    engine.
    """
    thresholds = _thresholds(safety_rules)
    image_size = dino_result.get("image_size") or {}
    expected_parts = expected_parts_for_step(workflow_config, step_id, text_prompt, parts_lexicon)
    candidates = normalize_detection_candidates(dino_result, sam_results, parts_lexicon)

    detections: list[VisionDetection] = []
    notes: list[str] = []
    matched_candidate_ids: set[int] = set()
    blocking_failure = False
    needs_confirmation = False
    confidence_values: list[float] = []

    for expected in expected_parts:
        # For each expected SOP part, keep the highest-confidence visual
        # candidate and classify the result against thresholds and ROI rules.
        part = str(expected["part"])
        required = bool(expected.get("required", True))
        roi = expected.get("roi")
        min_confidence = float(expected.get("min_confidence", thresholds["vision_detection_min"]))
        matching = [candidate for candidate in candidates if candidate["part"] == part]
        matching.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
        best = matching[0] if matching else None

        if best is None or float(best.get("confidence") or 0.0) < min_confidence:
            # A missing critical part is reported as a failed step observation,
            # but the event engine remains responsible for the final action.
            status = "missing" if required else "not_checked"
            confidence = float(best.get("confidence") or 0.0) if best is not None else 0.0
            if required and parts_lexicon.is_critical(part):
                blocking_failure = True
            elif required:
                needs_confirmation = True
            detections.append(
                VisionDetection(
                    detection_id=best.get("detection_id") if best else None,
                    part=part,
                    label=best.get("label", part) if best else part,
                    status=status,
                    confidence=confidence,
                    segmentation_confidence=best.get("segmentation_confidence") if best else None,
                    bbox=best.get("bbox") if best else None,
                    roi=roi,
                    roi_status="not_checked",
                    mask_path=best.get("mask_path") if best else None,
                    crop_path=best.get("crop_path") if best else None,
                    overlay_path=best.get("overlay_path") if best else None,
                    notes=[f"Required part not detected above min_confidence={min_confidence:.2f}"],
                )
            )
            confidence_values.append(confidence)
            continue

        detection_id = int(best["detection_id"])
        matched_candidate_ids.add(detection_id)
        confidence = float(best.get("confidence") or 0.0)
        segmentation_confidence = best.get("segmentation_confidence")
        roi_status = _bbox_center_inside_roi(best.get("bbox"), roi, camera_config, camera_id, view_id, image_size)
        detection_notes: list[str] = []

        if confidence < thresholds["vision_low"]:
            # Low confidence uses a fallback path: request confirmation instead
            # of converting uncertainty into a high-risk automated block.
            status = "low_confidence"
            needs_confirmation = True
            detection_notes.append(f"Detection confidence is below vision_low={thresholds['vision_low']:.2f}")
        elif roi_status == "outside":
            status = "ambiguous"
            needs_confirmation = True
            detection_notes.append(f"Detection center is outside expected ROI '{roi}'")
        else:
            status = "ok"

        if segmentation_confidence is not None and float(segmentation_confidence) < thresholds["segmentation_low"]:
            needs_confirmation = True
            detection_notes.append(
                f"Segmentation confidence is below segmentation_low={thresholds['segmentation_low']:.2f}"
            )

        detections.append(
            VisionDetection(
                detection_id=detection_id,
                part=part,
                label=str(best["label"]),
                status=status,
                confidence=confidence,
                segmentation_confidence=segmentation_confidence,
                bbox=best.get("bbox"),
                roi=roi,
                roi_status=roi_status,
                mask_path=best.get("mask_path"),
                crop_path=best.get("crop_path"),
                overlay_path=best.get("overlay_path"),
                notes=detection_notes,
            )
        )
        confidence_values.append(confidence)

    expected_part_keys = {str(item["part"]) for item in expected_parts}
    for candidate in candidates:
        # A confident detection that is not expected in this step is treated as
        # possible wrong-part evidence.
        detection_id = int(candidate["detection_id"])
        confidence = float(candidate.get("confidence") or 0.0)
        if detection_id in matched_candidate_ids or candidate["part"] in expected_part_keys:
            continue
        if confidence < thresholds["vision_detection_min"]:
            continue
        needs_confirmation = True
        detections.append(
            VisionDetection(
                detection_id=detection_id,
                part=str(candidate["part"]),
                label=str(candidate["label"]),
                status="wrong_part",
                confidence=confidence,
                segmentation_confidence=candidate.get("segmentation_confidence"),
                bbox=candidate.get("bbox"),
                roi=None,
                roi_status="not_checked",
                mask_path=candidate.get("mask_path"),
                crop_path=candidate.get("crop_path"),
                overlay_path=candidate.get("overlay_path"),
                notes=["Detected part is not expected for this step"],
            )
        )
        confidence_values.append(confidence)

    if not expected_parts:
        # This branch keeps the CV output useful before a complete SOP exists.
        # The model evidence is exported, but no step pass/fail claim is made.
        notes.append("No workflow step config found and no expected parts could be inferred from text prompt.")
        for candidate in candidates:
            confidence = float(candidate.get("confidence") or 0.0)
            detections.append(
                VisionDetection(
                    detection_id=int(candidate["detection_id"]),
                    part=str(candidate["part"]),
                    label=str(candidate["label"]),
                    status="not_checked",
                    confidence=confidence,
                    segmentation_confidence=candidate.get("segmentation_confidence"),
                    bbox=candidate.get("bbox"),
                    roi=None,
                    roi_status="not_checked",
                    mask_path=candidate.get("mask_path"),
                    crop_path=candidate.get("crop_path"),
                    overlay_path=candidate.get("overlay_path"),
                    notes=["No expected part rule was available for this detection"],
                )
            )
            confidence_values.append(confidence)

    if blocking_failure:
        step_status = "failed"
    elif needs_confirmation:
        step_status = "needs_confirmation"
    elif detections:
        step_status = "passed"
    else:
        step_status = "not_checked"

    step_confidence = min(confidence_values) if confidence_values else None
    return detections, step_status, step_confidence, notes
