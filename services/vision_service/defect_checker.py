from __future__ import annotations

from typing import Any

from .schemas import VisionDefect


def check_defects(sam_results: list[dict[str, Any]]) -> list[VisionDefect]:
    """Flag suspicious segmentation evidence without claiming true defects.

    A dedicated defect model should replace or extend this file later. For now,
    these checks only identify cases where the mask evidence looks unreliable
    enough to require confirmation.
    """
    defects: list[VisionDefect] = []
    for sam_entry in sam_results:
        label = str(sam_entry.get("label", "unknown"))
        output = sam_entry.get("output") or {}
        mask = output.get("mask") or {}
        area_ratio = mask.get("area_ratio")
        score = mask.get("score")
        overlay_path = output.get("result_image_path")

        # A tiny mask often means the segmentation missed the target object.
        if area_ratio is not None and float(area_ratio) < 0.005:
            defects.append(
                VisionDefect(
                    defect_type="mask_area_too_small",
                    part=label,
                    severity="medium",
                    confidence=score,
                    evidence_path=overlay_path,
                    notes=["Segmentation mask is very small; request recapture or human confirmation."],
                )
            )
        # A near-full-crop mask often means the crop or prompt is too broad.
        if area_ratio is not None and float(area_ratio) > 0.98:
            defects.append(
                VisionDefect(
                    defect_type="mask_area_too_large",
                    part=label,
                    severity="medium",
                    confidence=score,
                    evidence_path=overlay_path,
                    notes=["Segmentation mask covers almost the whole crop; candidate may be unreliable."],
                )
            )
    return defects
