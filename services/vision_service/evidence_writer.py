from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting for audit readability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject arrays or scalar payloads."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def copy_original_image(image_path: Path, evidence_dir: Path, image_id: str) -> Path:
    """Copy the input image into the evidence folder for traceability."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix or ".jpg"
    destination = evidence_dir / f"{image_id}_original{suffix}"
    if image_path.resolve() != destination.resolve():
        shutil.copy2(image_path, destination)
    return destination


def collect_detection_evidence_paths(dino_result: dict[str, Any], sam_results: list[dict[str, Any]]) -> list[str]:
    """Collect crop, mask, and overlay paths referenced by raw model outputs."""
    paths: list[str] = []
    for detection in dino_result.get("detections", []) or []:
        for key in ["crop_image_path", "sam_foreground_prior_mask_path"]:
            value = detection.get(key)
            if value:
                paths.append(str(value))
    for sam_entry in sam_results:
        output = sam_entry.get("output") or {}
        for key in ["result_image_path", "mask_image_path", "masked_object_path"]:
            value = output.get(key)
            if value:
                paths.append(str(value))
    return paths
