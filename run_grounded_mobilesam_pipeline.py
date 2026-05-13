from __future__ import annotations

import argparse

from services.vision_service.pipeline import ROOT_DIR, VisionPipelineConfig, run_vision_pipeline


def parse_args() -> argparse.Namespace:
    """Build the CLI contract for the V-RAWA vision-layer entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the V-RAWA vision layer: GroundingDINO candidate detection, "
            "MobileSAM segmentation, step checking, and evidence export."
        )
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--text-prompt", required=True, help="Text prompt for GroundingDINO detection")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "pipeline_outputs"),
        help="Directory where GroundingDINO, MobileSAM, merged, and vision-layer outputs will be saved",
    )
    parser.add_argument("--box-threshold", type=float, default=0.3, help="GroundingDINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument(
        "--max-detections",
        type=int,
        default=0,
        help="Maximum number of GroundingDINO detections to process. Use 0 for all.",
    )
    parser.add_argument("--dino-device", default="cuda", help="Device passed to the GroundingDINO step")
    parser.add_argument("--sam-device", default="cuda", help="Device passed to the MobileSAM step")
    parser.add_argument("--dino-python", default=None, help="Python executable for the GroundingDINO environment")
    parser.add_argument("--sam-python", default=None, help="Python executable for the MobileSAM environment")
    parser.add_argument(
        "--dino-config",
        default=str(ROOT_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
        help="GroundingDINO config file",
    )
    parser.add_argument(
        "--dino-checkpoint",
        default=str(ROOT_DIR / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"),
        help="GroundingDINO checkpoint file",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=str(ROOT_DIR / "MobileSAM-fast-finetuning" / "weights" / "mobile_sam.pt"),
        help="MobileSAM checkpoint file",
    )

    # These fields are not needed by the model itself, but they are required for
    # audit logs and for the event engine to relate the CV result to a workflow.
    parser.add_argument("--work-order-id", default="WO-UNSPECIFIED", help="Work order ID for audit output")
    parser.add_argument("--station-id", default="ST-A01", help="Station ID for audit output")
    parser.add_argument("--step-id", default="S00", help="Workflow step ID used by the step checker")
    parser.add_argument("--camera-id", default="CAM-A01", help="Camera ID for multi-view-ready output")
    parser.add_argument("--view-id", default="top", help="View ID for multi-view-ready output")
    parser.add_argument("--image-id", default=None, help="Optional image ID. Defaults to one derived from the filename.")
    parser.add_argument("--operator-id", default=None, help="Optional operator ID for event output")
    parser.add_argument("--lighting-profile", default=None, help="Optional lighting profile override")

    # Configuration files keep SOP, camera, model, and safety rules outside the
    # code so they can evolve without changing the pipeline implementation.
    parser.add_argument(
        "--workflow-config",
        default=str(ROOT_DIR / "configs" / "workflow_steps.yaml"),
        help="Workflow step configuration YAML",
    )
    parser.add_argument(
        "--parts-lexicon",
        default=str(ROOT_DIR / "configs" / "parts_lexicon.yaml"),
        help="Parts lexicon YAML",
    )
    parser.add_argument(
        "--safety-rules",
        default=str(ROOT_DIR / "configs" / "safety_rules.yaml"),
        help="Safety and confidence rules YAML",
    )
    parser.add_argument(
        "--camera-config",
        default=str(ROOT_DIR / "configs" / "camera_config.yaml"),
        help="Camera and ROI configuration YAML",
    )
    parser.add_argument(
        "--model-registry",
        default=str(ROOT_DIR / "configs" / "model_registry.yaml"),
        help="Model registry YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The dataclass is the boundary between CLI parsing and the reusable service
    # pipeline. Other callers can construct this object without using argparse.
    config = VisionPipelineConfig(
        image=args.image,
        text_prompt=args.text_prompt,
        output_dir=args.output_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        max_detections=args.max_detections,
        dino_device=args.dino_device,
        sam_device=args.sam_device,
        dino_python=args.dino_python,
        sam_python=args.sam_python,
        dino_config=args.dino_config,
        dino_checkpoint=args.dino_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        work_order_id=args.work_order_id,
        station_id=args.station_id,
        step_id=args.step_id,
        camera_id=args.camera_id,
        view_id=args.view_id,
        image_id=args.image_id,
        operator_id=args.operator_id,
        lighting_profile=args.lighting_profile,
        workflow_config=args.workflow_config,
        parts_lexicon=args.parts_lexicon,
        safety_rules=args.safety_rules,
        camera_config=args.camera_config,
        model_registry=args.model_registry,
    )
    run_vision_pipeline(config)


if __name__ == "__main__":
    main()
