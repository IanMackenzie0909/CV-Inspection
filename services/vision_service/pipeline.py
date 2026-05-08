from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PartsLexicon, load_yaml, model_info
from .defect_checker import check_defects
from .detector import (
    build_groundingdino_command,
    build_mobilesam_command,
    resolve_python,
    run_command,
)
from .event_builder import build_vision_event
from .evidence_writer import collect_detection_evidence_paths, copy_original_image, read_json, save_json
from .schemas import Evidence, VisionResult
from .step_checker import check_step


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VisionPipelineConfig:
    """Runtime configuration for one vision-layer execution."""

    # Model inputs and output location.
    image: str
    text_prompt: str
    output_dir: str

    # GroundingDINO and MobileSAM runtime controls.
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    max_detections: int = 0
    dino_device: str = "cuda"
    sam_device: str = "cuda"
    dino_python: str | None = None
    sam_python: str | None = None
    dino_config: str = str(ROOT_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    dino_checkpoint: str = str(ROOT_DIR / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth")
    sam_checkpoint: str = str(ROOT_DIR / "MobileSAM-fast-finetuning" / "weights" / "mobile_sam.pt")

    # Workflow metadata is copied into both vision_result.json and the event
    # payload so downstream services can audit the recognition result.
    work_order_id: str = "WO-UNSPECIFIED"
    station_id: str = "ST-A01"
    step_id: str = "S00"
    camera_id: str = "CAM-A01"
    view_id: str = "top"
    image_id: str | None = None
    operator_id: str | None = None
    lighting_profile: str | None = None

    # Externalized configuration keeps SOP-like rules out of the Python code.
    workflow_config: str = str(ROOT_DIR / "configs" / "workflow_steps.yaml")
    parts_lexicon: str = str(ROOT_DIR / "configs" / "parts_lexicon.yaml")
    safety_rules: str = str(ROOT_DIR / "configs" / "safety_rules.yaml")
    camera_config: str = str(ROOT_DIR / "configs" / "camera_config.yaml")
    model_registry: str = str(ROOT_DIR / "configs" / "model_registry.yaml")


def _resolve_path(value: str) -> Path:
    """Resolve user-provided paths relative to the current working directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _timestamp() -> str:
    """Return an ISO timestamp with the local timezone attached."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _default_image_id(image_path: Path) -> str:
    """Create a stable image identifier when the caller does not provide one."""
    stem = "".join(character if character.isalnum() else "-" for character in image_path.stem).strip("-")
    return f"IMG-{stem or 'input'}"


def _camera_lighting_profile(camera_config: dict[str, Any], camera_id: str, view_id: str) -> str | None:
    """Look up the configured lighting profile for the active camera view."""
    for camera in camera_config.get("cameras", []) or []:
        if camera.get("camera_id") == camera_id and camera.get("view_id") == view_id:
            value = camera.get("lighting_profile")
            return str(value) if value else None
    return None


def run_vision_pipeline(config: VisionPipelineConfig) -> dict[str, Any]:
    """Run detection, segmentation, step checking, and event export."""
    image_path = _resolve_path(config.image)
    output_dir = _resolve_path(config.output_dir)

    # Keep raw model outputs and system-level outputs side by side. This makes
    # debugging easier without forcing downstream consumers to parse raw outputs.
    dino_output_dir = output_dir / "groundingdino"
    sam_output_root = output_dir / "mobilesam"
    evidence_dir = output_dir / "evidence"
    merged_json_path = output_dir / "pipeline_result.json"
    vision_result_path = output_dir / "vision_result.json"
    vision_event_path = output_dir / "vision_event.json"

    dino_repo_dir = ROOT_DIR / "GroundingDINO"
    sam_repo_dir = ROOT_DIR / "MobileSAM-fast-finetuning"
    dino_script = dino_repo_dir / "demo" / "prompt_detection_and_crop.py"
    sam_script = sam_repo_dir / "demo_point_prompt.py"

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if config.max_detections < 0:
        raise ValueError("--max-detections must be 0 or greater")
    if not dino_script.exists():
        raise FileNotFoundError(f"GroundingDINO script not found: {dino_script}")
    if not sam_script.exists():
        raise FileNotFoundError(f"MobileSAM script not found: {sam_script}")

    # Load all operational rules before running the models so configuration
    # errors fail fast rather than after expensive inference.
    workflow_config = load_yaml(_resolve_path(config.workflow_config))
    parts_lexicon_payload = load_yaml(_resolve_path(config.parts_lexicon))
    safety_rules = load_yaml(_resolve_path(config.safety_rules))
    camera_config = load_yaml(_resolve_path(config.camera_config))
    model_registry = load_yaml(_resolve_path(config.model_registry))
    lexicon = PartsLexicon(parts_lexicon_payload)

    # GroundingDINO and MobileSAM may live in separate virtual environments.
    # The MobileSAM resolver also tries GroundingDINO/env because this project
    # currently has torch and the MobileSAM package importable there.
    dino_python = resolve_python(config.dino_python, [dino_repo_dir / "env"])
    sam_python = resolve_python(config.sam_python, [ROOT_DIR / "env", sam_repo_dir / "env", dino_repo_dir / "env"])
    dino_config = _resolve_path(config.dino_config)
    dino_checkpoint = _resolve_path(config.dino_checkpoint)
    sam_checkpoint = _resolve_path(config.sam_checkpoint)

    dino_output_dir.mkdir(parents=True, exist_ok=True)
    sam_output_root.mkdir(parents=True, exist_ok=True)

    dino_command = build_groundingdino_command(
        python_path=dino_python,
        script_path=dino_script,
        image_path=image_path,
        text_prompt=config.text_prompt,
        config_path=dino_config,
        checkpoint_path=dino_checkpoint,
        output_dir=dino_output_dir,
        box_threshold=config.box_threshold,
        text_threshold=config.text_threshold,
        device=config.dino_device,
        max_detections=config.max_detections,
    )
    run_command(dino_command, cwd=dino_repo_dir)

    # GroundingDINO produces candidate object boxes. These are treated as model
    # evidence, not as production decisions.
    dino_result = read_json(dino_output_dir / "detections.json")
    merged_results: list[dict[str, Any]] = []
    mobilesam_outputs: list[dict[str, Any]] = []

    for detection in dino_result.get("detections", []) or []:
        # Each candidate box is segmented independently so the evidence can be
        # traced from detection -> crop -> mask -> final step status.
        sam_output_dir = sam_output_root / f"detection_{int(detection['id']):03d}"
        sam_command = build_mobilesam_command(
            python_path=sam_python,
            script_path=sam_script,
            detection=detection,
            checkpoint_path=sam_checkpoint,
            output_dir=sam_output_dir,
            device=config.sam_device,
        )
        run_command(sam_command, cwd=sam_repo_dir)
        sam_result = read_json(sam_output_dir / "mask_result.json")
        foreground_point = detection.get("sam_foreground_point_in_crop_xy")
        background_point = detection.get("sam_background_point_in_crop_xy")
        mobilesam_entry = {
            "detection_id": detection["id"],
            "label": detection["label"],
            "prompt_type": (
                "points_and_box"
                if foreground_point is not None or background_point is not None
                else "box"
            ),
            "output": sam_result,
        }
        mobilesam_outputs.append(mobilesam_entry)
        merged_results.append({"detection": detection, "mobilesam": sam_result})

    # Preserve the original merged model output for compatibility with the
    # earlier prototype and for low-level debugging.
    raw_payload = {
        "pipeline": "groundingdino_to_mobilesam_auto_point_and_box_prompt",
        "input_image_path": str(image_path),
        "text_prompt": config.text_prompt,
        "output_dir": str(output_dir),
        "python_executables": {
            "groundingdino": str(dino_python),
            "mobilesam": str(sam_python),
        },
        "groundingdino": dino_result,
        "mobilesam": mobilesam_outputs,
        "results": merged_results,
    }
    save_json(merged_json_path, raw_payload)

    image_id = config.image_id or _default_image_id(image_path)
    evidence_image_path = copy_original_image(image_path, evidence_dir, image_id)
    detection_evidence_paths = collect_detection_evidence_paths(dino_result, mobilesam_outputs)

    # The step checker maps model observations to workflow-oriented statuses
    # such as ok, missing, low_confidence, or wrong_part.
    detections, step_status, step_confidence, step_notes = check_step(
        dino_result=dino_result,
        sam_results=mobilesam_outputs,
        workflow_config=workflow_config,
        safety_rules=safety_rules,
        camera_config=camera_config,
        parts_lexicon=lexicon,
        step_id=config.step_id,
        text_prompt=config.text_prompt,
        camera_id=config.camera_id,
        view_id=config.view_id,
    )
    defects = check_defects(mobilesam_outputs)

    # The current defect checker is conservative. It never claims a production
    # defect from appearance alone; it requests confirmation when evidence looks
    # unreliable or suspicious.
    if defects and step_status == "passed":
        step_status = "needs_confirmation"

    service_model = model_info(model_registry, "vision_service", "V-RAWA vision_service")
    timestamp = _timestamp()
    annotated_path = dino_result.get("annotated_image_path")
    lighting_profile = config.lighting_profile or _camera_lighting_profile(
        camera_config,
        config.camera_id,
        config.view_id,
    )

    # vision_result.json is the human/debug/audit view of the CV layer.
    vision_result = VisionResult(
        work_order_id=config.work_order_id,
        station_id=config.station_id,
        step_id=config.step_id,
        camera_id=config.camera_id,
        view_id=config.view_id,
        image_id=image_id,
        model_name=service_model["name"],
        model_version=service_model["version"],
        lighting_profile=lighting_profile,
        detections=detections,
        defects=defects,
        step_status=step_status,
        step_status_confidence=step_confidence,
        evidence=Evidence(
            image_path=str(evidence_image_path),
            annotated_image_path=str(annotated_path) if annotated_path else None,
            groundingdino_json_path=str(dino_output_dir / "detections.json"),
            pipeline_json_path=str(merged_json_path),
            detection_evidence_paths=detection_evidence_paths,
        ),
        timestamp=timestamp,
        raw={
            "step_checker_notes": step_notes,
            "groundingdino": {
                "model": model_info(model_registry, "groundingdino", "GroundingDINO"),
                "box_threshold": config.box_threshold,
                "text_threshold": config.text_threshold,
            },
            "mobilesam": model_info(model_registry, "mobilesam", "MobileSAM"),
        },
    )
    save_json(vision_result_path, vision_result.dict())

    # vision_event.json is the event-engine-facing version. It carries the same
    # evidence but wraps it in the common event envelope.
    rule_version = str(safety_rules.get("rule_version", "unknown"))
    vision_event = build_vision_event(
        vision_result=vision_result,
        rule_version=rule_version,
        operator_id=config.operator_id,
    )
    save_json(vision_event_path, vision_event.dict())

    print(f"Saved merged pipeline JSON: {merged_json_path}")
    print(f"Saved vision result JSON:   {vision_result_path}")
    print(f"Saved vision event JSON:    {vision_event_path}")
    print(f"Processed detections:       {len(merged_results)}")
    print(f"Step status:                {vision_result.step_status}")

    return {
        "pipeline_result_path": str(merged_json_path),
        "vision_result_path": str(vision_result_path),
        "vision_event_path": str(vision_event_path),
        "vision_result": vision_result.dict(),
        "vision_event": vision_event.dict(),
    }
