from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert

from groundingdino.util.inference import annotate, load_image, load_model, predict


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run GroundingDINO detection, save cropped detections, and export JSON metadata."
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--text-prompt", required=True, help="Text prompt for GroundingDINO")
    parser.add_argument(
        "--config",
        default=str(repo_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"),
        help="Path to the GroundingDINO config file",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(repo_dir / "weights" / "groundingdino_swint_ogc.pth"),
        help="Path to the GroundingDINO checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the annotated image, crops, and JSON file will be saved",
    )
    parser.add_argument("--box-threshold", type=float, default=0.3, help="Detection box threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text matching threshold")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda, cuda:0, or cpu")
    parser.add_argument("--max-detections", type=int, default=0, help="Maximum number of detections to keep. Use 0 for all.")
    return parser.parse_args()


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip().lower()).strip("_")
    return cleaned or "object"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def find_distance_peak(binary_mask: np.ndarray) -> tuple[list[int] | None, float]:
    if int(np.count_nonzero(binary_mask)) == 0:
        return None, 0.0

    distance_map = cv2.distanceTransform((binary_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _, max_distance, _, max_location = cv2.minMaxLoc(distance_map)
    if max_distance <= 0.0:
        return None, 0.0
    return [int(max_location[0]), int(max_location[1])], float(max_distance)


def infer_prompt_points(crop_rgb: np.ndarray) -> tuple[list[int] | None, list[int] | None, dict, np.ndarray | None]:
    crop_height, crop_width = crop_rgb.shape[:2]
    border = max(3, min(crop_height, crop_width) // 40)

    border_pixels = np.concatenate(
        [
            crop_rgb[:border, :, :].reshape(-1, 3),
            crop_rgb[-border:, :, :].reshape(-1, 3),
            crop_rgb[:, :border, :].reshape(-1, 3),
            crop_rgb[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    color_distance = np.linalg.norm(crop_rgb.astype(np.float32) - background_color.astype(np.float32), axis=2)

    if float(color_distance.max()) <= 0.0:
        return None, None, {
            "status": "failed",
            "method": "border_color_distance_transform",
            "reason": "zero_color_distance",
        }, None

    normalized_distance = cv2.normalize(color_distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    otsu_threshold, foreground_mask = cv2.threshold(
        normalized_distance,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((3, 3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
    if component_count > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_index = int(np.argmax(component_areas)) + 1
        foreground_mask = np.where(labels == largest_component_index, 255, 0).astype(np.uint8)

    if int(np.count_nonzero(foreground_mask)) == 0:
        return None, None, {
            "status": "failed",
            "method": "border_color_distance_transform",
            "reason": "empty_foreground_mask",
            "otsu_threshold": float(otsu_threshold),
        }, None

    foreground_point, foreground_peak = find_distance_peak(foreground_mask)
    if foreground_point is None:
        moments = cv2.moments(foreground_mask)
        if moments["m00"] == 0:
            return None, None, {
                "status": "failed",
                "method": "border_color_distance_transform",
                "reason": "no_distance_peak",
                "otsu_threshold": float(otsu_threshold),
            }, foreground_mask
        foreground_point = [
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        ]
        foreground_point_method = "foreground_centroid_fallback"
    else:
        foreground_point_method = "distance_transform_peak"

    inverse_mask = np.where(foreground_mask > 0, 0, 255).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse_mask, connectivity=8)
    hole_mask = np.zeros_like(inverse_mask)
    largest_hole_area = 0
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        w = int(stats[component_index, cv2.CC_STAT_WIDTH])
        h = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + w) >= crop_width or (y + h) >= crop_height
        if touches_border:
            continue
        if area > largest_hole_area:
            largest_hole_area = area
            hole_mask = np.where(labels == component_index, 255, 0).astype(np.uint8)

    background_point = None
    background_point_method = None
    background_peak = 0.0
    if largest_hole_area > 0:
        background_point, background_peak = find_distance_peak(hole_mask)
        if background_point is None:
            hole_moments = cv2.moments(hole_mask)
            if hole_moments["m00"] > 0:
                background_point = [
                    int(round(hole_moments["m10"] / hole_moments["m00"])),
                    int(round(hole_moments["m01"] / hole_moments["m00"])),
                ]
                background_point_method = "hole_centroid_fallback"
        else:
            background_point_method = "hole_distance_transform_peak"

    return foreground_point, background_point, {
        "status": "ok",
        "method": "border_color_distance_transform",
        "foreground_point_method": foreground_point_method,
        "background_point_method": background_point_method,
        "border_width_px": border,
        "estimated_background_rgb": [round(float(value), 3) for value in background_color.tolist()],
        "otsu_threshold": float(otsu_threshold),
        "foreground_area_ratio": float(np.count_nonzero(foreground_mask) / float(crop_width * crop_height)),
        "foreground_distance_peak": float(foreground_peak),
        "hole_area_ratio": float(largest_hole_area / float(crop_width * crop_height)),
        "background_distance_peak": float(background_peak),
    }, foreground_mask


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    crops_dir = output_dir / "crops"
    json_path = output_dir / "detections.json"
    annotated_path = output_dir / "groundingdino_annotated.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if args.max_detections < 0:
        raise ValueError("--max-detections must be 0 or greater")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or run on a CUDA machine.")

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    image_rgb, image_tensor = load_image(str(image_path))
    image_pil = Image.fromarray(image_rgb)
    image_width, image_height = image_pil.size

    model = load_model(
        model_config_path=str(config_path),
        model_checkpoint_path=str(checkpoint_path),
        device=args.device,
    )

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )

    if len(boxes) > 0:
        order = torch.argsort(logits, descending=True)
        if args.max_detections:
            order = order[: args.max_detections]
        boxes = boxes[order]
        logits = logits[order]
        phrases = [phrases[index] for index in order.tolist()]

    if len(boxes) > 0:
        annotated_bgr = annotate(
            image_source=image_rgb,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )
    else:
        annotated_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(annotated_path), annotated_bgr)

    detections: list[dict] = []
    if len(boxes) > 0:
        boxes_xyxy = box_convert(
            boxes=boxes * torch.tensor([image_width, image_height, image_width, image_height]),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).tolist()

        for index, (bbox_xyxy, score, phrase) in enumerate(zip(boxes_xyxy, logits.tolist(), phrases), start=1):
            x1, y1, x2, y2 = bbox_xyxy
            x1 = max(0.0, min(float(x1), float(image_width)))
            y1 = max(0.0, min(float(y1), float(image_height)))
            x2 = max(0.0, min(float(x2), float(image_width)))
            y2 = max(0.0, min(float(y2), float(image_height)))

            crop_x1 = max(0, min(int(x1), image_width - 1))
            crop_y1 = max(0, min(int(y1), image_height - 1))
            crop_x2 = max(crop_x1 + 1, min(int(torch.ceil(torch.tensor(x2)).item()), image_width))
            crop_y2 = max(crop_y1 + 1, min(int(torch.ceil(torch.tensor(y2)).item()), image_height))

            crop = image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_width, crop_height = crop.size
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            point_x_in_crop = max(0.0, min(center_x - crop_x1, max(crop_width - 1, 0)))
            point_y_in_crop = max(0.0, min(center_y - crop_y1, max(crop_height - 1, 0)))
            crop_rgb = np.asarray(crop)
            sam_foreground_point_in_crop_xy, sam_background_point_in_crop_xy, sam_point_metadata, sam_foreground_prior_mask = infer_prompt_points(crop_rgb)

            crop_filename = f"detection_{index:03d}_{sanitize_label(phrase)}.png"
            crop_path = crops_dir / crop_filename
            crop.save(crop_path)
            foreground_prior_mask_filename = f"detection_{index:03d}_{sanitize_label(phrase)}_foreground_prior_mask.png"
            foreground_prior_mask_path = crops_dir / foreground_prior_mask_filename
            if sam_foreground_prior_mask is not None:
                cv2.imwrite(str(foreground_prior_mask_path), sam_foreground_prior_mask)

            detections.append(
                {
                    "id": index,
                    "label": phrase,
                    "confidence": float(score),
                    "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                    "bbox_xywh": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                    "crop_box_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
                    "center_xy": [round(center_x, 3), round(center_y, 3)],
                    "center_in_crop_xy": [round(point_x_in_crop, 3), round(point_y_in_crop, 3)],
                    "sam_foreground_point_in_crop_xy": sam_foreground_point_in_crop_xy,
                    "sam_background_point_in_crop_xy": sam_background_point_in_crop_xy,
                    "sam_foreground_point_metadata": sam_point_metadata,
                    "sam_box_prompt_in_crop_xyxy": [0, 0, crop_width, crop_height],
                    "sam_foreground_prior_mask_path": str(foreground_prior_mask_path) if sam_foreground_prior_mask is not None else None,
                    "crop_size": {"width": crop_width, "height": crop_height},
                    "crop_image_path": str(crop_path),
                }
            )

    payload = {
        "input_image_path": str(image_path),
        "text_prompt": args.text_prompt,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "device": args.device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "image_size": {"width": image_width, "height": image_height},
        "annotated_image_path": str(annotated_path),
        "detection_count": len(detections),
        "detections": detections,
    }
    save_json(json_path, payload)

    print(f"Saved annotated image: {annotated_path}")
    print(f"Saved detection JSON:  {json_path}")
    print(f"Detections kept:       {len(detections)}")


if __name__ == "__main__":
    main()
