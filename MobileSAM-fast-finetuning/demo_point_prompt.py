from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mobile_sam import SamPredictor, sam_model_registry


MODEL_TYPE = "vit_t"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run MobileSAM prompt-based segmentation on a single image and export images plus JSON metadata."
    )
    parser.add_argument("--input", required=True, help="Path to the input image")
    parser.add_argument("--point-x", type=float, help="Optional prompt point x coordinate in image pixels")
    parser.add_argument("--point-y", type=float, help="Optional prompt point y coordinate in image pixels")
    parser.add_argument("--negative-point-x", type=float, help="Optional negative prompt point x coordinate in image pixels")
    parser.add_argument("--negative-point-y", type=float, help="Optional negative prompt point y coordinate in image pixels")
    parser.add_argument("--hole-point-x", type=float, help="Optional center hole prompt x coordinate in image pixels")
    parser.add_argument("--hole-point-y", type=float, help="Optional center hole prompt y coordinate in image pixels")
    parser.add_argument("--box-x1", type=float, help="Optional prompt box x1 coordinate in image pixels")
    parser.add_argument("--box-y1", type=float, help="Optional prompt box y1 coordinate in image pixels")
    parser.add_argument("--box-x2", type=float, help="Optional prompt box x2 coordinate in image pixels")
    parser.add_argument("--box-y2", type=float, help="Optional prompt box y2 coordinate in image pixels")
    parser.add_argument(
        "--checkpoint",
        default=str(script_dir / "weights" / "mobile_sam.pt"),
        help="Path to the MobileSAM checkpoint",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where results will be saved")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda, cuda:0, or cpu")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay strength")
    parser.add_argument("--label", default="target", help="Optional label used in the result image text")
    parser.add_argument("--prior-mask", help="Optional prior foreground mask used to choose the best SAM candidate")
    parser.add_argument("--disable-mask-cleanup", action="store_true", help="Disable conservative final mask cleanup")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_predictor(checkpoint_path: Path, device: torch.device) -> SamPredictor:
    model = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint_path))
    model = model.to(device=device)
    model.eval()
    return SamPredictor(model)


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    prior_mask: np.ndarray | None,
) -> tuple[int, dict]:
    if prior_mask is None:
        best_index = int(np.argmax(scores))
        return best_index, {
            "selection_method": "highest_sam_score",
            "selected_index": best_index,
            "candidate_scores": [
                {
                    "index": int(index),
                    "sam_score": float(score),
                    "prior_iou": None,
                }
                for index, score in enumerate(scores)
            ],
        }

    candidate_scores = []
    ranked_candidates = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        prior_iou = compute_mask_iou(mask.astype(bool), prior_mask.astype(bool))
        candidate_scores.append(
            {
                "index": int(index),
                "sam_score": float(score),
                "prior_iou": float(prior_iou),
            }
        )
        ranked_candidates.append((prior_iou, float(score), int(index)))

    ranked_candidates.sort(reverse=True)
    best_index = ranked_candidates[0][2]
    return best_index, {
        "selection_method": "prior_iou_then_sam_score",
        "selected_index": best_index,
        "candidate_scores": candidate_scores,
    }


def refine_mask_with_prior(best_mask: np.ndarray, prior_mask: np.ndarray | None, selection_debug: dict) -> tuple[np.ndarray, dict]:
    if prior_mask is None:
        return best_mask, {
            "applied": False,
            "reason": "no_prior_mask",
        }

    selected_index = int(selection_debug.get("selected_index", 0))
    for candidate in selection_debug["candidate_scores"]:
        if candidate["index"] == selected_index:
            selected_prior_iou = candidate["prior_iou"]
            break
    else:
        selected_prior_iou = None

    if selected_prior_iou is None or selected_prior_iou >= 0.5:
        return best_mask, {
            "applied": False,
            "reason": "selected_mask_matches_prior",
            "selected_prior_iou": selected_prior_iou,
        }

    refined_mask = np.logical_and(best_mask.astype(bool), prior_mask.astype(bool))
    if int(refined_mask.sum()) == 0:
        return best_mask, {
            "applied": False,
            "reason": "prior_intersection_empty",
            "selected_prior_iou": selected_prior_iou,
        }

    refined_mask_uint8 = (refined_mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    refined_mask_uint8 = cv2.morphologyEx(refined_mask_uint8, cv2.MORPH_OPEN, kernel)
    refined_mask_uint8 = cv2.morphologyEx(refined_mask_uint8, cv2.MORPH_CLOSE, kernel)

    return refined_mask_uint8 > 0, {
        "applied": True,
        "reason": "intersected_with_prior_mask",
        "selected_prior_iou": selected_prior_iou,
    }


def select_hole_mask(hole_masks: np.ndarray, hole_scores: np.ndarray) -> tuple[int, dict]:
    candidate_scores = []
    ranked_candidates = []

    for index, (mask, score) in enumerate(zip(hole_masks, hole_scores)):
        mask_bool = mask.astype(bool)
        area = int(mask_bool.sum())
        touches_border = bool(
            mask_bool[0, :].any()
            or mask_bool[-1, :].any()
            or mask_bool[:, 0].any()
            or mask_bool[:, -1].any()
        )
        candidate_scores.append(
            {
                "index": int(index),
                "sam_score": float(score),
                "area": area,
                "touches_border": touches_border,
            }
        )
        ranked_candidates.append((not touches_border, -area, float(score), int(index)))

    ranked_candidates.sort(reverse=True)
    best_index = ranked_candidates[0][3]
    return best_index, {
        "selection_method": "prefer_internal_small_region",
        "selected_index": best_index,
        "candidate_scores": candidate_scores,
    }


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if component_count <= 1:
        return mask.astype(bool)

    largest_component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_component = labels == largest_component_index
    return largest_component.astype(bool)


def cleanup_final_mask(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    cleaned_mask = keep_largest_component(mask.astype(bool))
    cleaned_mask_uint8 = (cleaned_mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(cleaned_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned_mask.astype(bool), {
            "applied": False,
            "reason": "no_external_contour",
        }

    largest_contour = max(contours, key=cv2.contourArea)
    outer_filled = np.zeros_like(cleaned_mask_uint8)
    cv2.drawContours(outer_filled, [largest_contour], -1, 255, thickness=-1)

    inverse_mask = cv2.bitwise_not(cleaned_mask_uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse_mask, connectivity=8)
    height, width = cleaned_mask.shape[:2]

    largest_hole_area = 0
    largest_hole_mask = None
    for component_index in range(1, component_count):
        x = int(stats[component_index, cv2.CC_STAT_LEFT])
        y = int(stats[component_index, cv2.CC_STAT_TOP])
        w = int(stats[component_index, cv2.CC_STAT_WIDTH])
        h = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if touches_border:
            continue
        if area > largest_hole_area:
            largest_hole_area = area
            largest_hole_mask = labels == component_index

    outer_area = int(np.count_nonzero(outer_filled))
    hole_ratio = float(largest_hole_area / float(outer_area)) if outer_area > 0 else 0.0

    if largest_hole_mask is not None and hole_ratio >= 0.02:
        final_mask = np.logical_and(outer_filled > 0, np.logical_not(largest_hole_mask))
        cleanup_mode = "outer_contour_minus_largest_hole"
    else:
        final_mask = cleaned_mask.astype(bool)
        cleanup_mode = "largest_component_only"

    return final_mask.astype(bool), {
        "applied": True,
        "mode": cleanup_mode,
        "outer_area_pixels": outer_area,
        "largest_hole_area_pixels": int(largest_hole_area),
        "largest_hole_ratio": hole_ratio,
    }


def subtract_hole_from_outer_mask(
    outer_mask: np.ndarray,
    hole_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    outer_area = int(outer_mask.sum())
    hole_area = int(hole_mask.sum())
    if outer_area == 0 or hole_area == 0:
        return outer_mask.astype(bool), {
            "applied": False,
            "reason": "empty_outer_or_hole_mask",
        }

    touches_border = bool(
        hole_mask[0, :].any()
        or hole_mask[-1, :].any()
        or hole_mask[:, 0].any()
        or hole_mask[:, -1].any()
    )
    hole_ratio_vs_outer = float(hole_area / float(outer_area))
    contained_ratio = float(np.logical_and(hole_mask, outer_mask).sum() / float(hole_area))

    if touches_border or hole_ratio_vs_outer >= 0.8 or contained_ratio < 0.95:
        return outer_mask.astype(bool), {
            "applied": False,
            "reason": "hole_mask_failed_validation",
            "touches_border": touches_border,
            "hole_ratio_vs_outer": hole_ratio_vs_outer,
            "contained_ratio": contained_ratio,
        }

    ring_mask = np.logical_and(outer_mask, np.logical_not(hole_mask))
    ring_mask = keep_largest_component(ring_mask)

    return ring_mask.astype(bool), {
        "applied": True,
        "reason": "subtracted_center_hole_from_outer_mask",
        "touches_border": touches_border,
        "hole_ratio_vs_outer": hole_ratio_vs_outer,
        "contained_ratio": contained_ratio,
    }


def render_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    positive_point_xy: tuple[int, int] | None,
    negative_point_xy: tuple[int, int] | None,
    prompt_box_xyxy: list[int] | None,
    score: float,
    label: str,
    alpha: float,
) -> tuple[np.ndarray, list[int]]:
    overlay = image_bgr.copy()
    mask_color = np.array([0, 255, 255], dtype=np.uint8)
    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
    blended = cv2.addWeighted(
        overlay,
        1.0 - alpha,
        np.full_like(overlay, mask_color),
        alpha,
        0.0,
    )
    overlay = np.where(mask_rgb, blended, overlay)

    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(mask_uint8)
    bbox_xyxy = [int(x), int(y), int(x + w), int(y + h)] if w > 0 and h > 0 else [0, 0, 0, 0]
    if w > 0 and h > 0:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 165, 255), 2)

    if prompt_box_xyxy is not None:
        px1, py1, px2, py2 = prompt_box_xyxy
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (255, 0, 255), 2)

    if positive_point_xy is not None:
        cv2.circle(overlay, positive_point_xy, 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, positive_point_xy, 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    if negative_point_xy is not None:
        cv2.circle(overlay, negative_point_xy, 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, negative_point_xy, 10, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    text = f"{label} score={score:.3f}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    text_x = bbox_xyxy[0] if w > 0 else 8
    text_y = max((bbox_xyxy[1] - 8) if h > 0 else 24, 24)
    cv2.rectangle(
        overlay,
        (text_x, text_y - text_height - baseline - 4),
        (text_x + text_width + 8, text_y + 4),
        (0, 165, 255),
        -1,
    )
    cv2.putText(
        overlay,
        text,
        (text_x + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )

    return overlay, bbox_xyxy


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    json_path = output_dir / "mask_result.json"
    overlay_path = output_dir / "result_overlay.png"
    mask_path = output_dir / "object_mask.png"
    masked_object_path = output_dir / "masked_object.png"

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or run on a CUDA machine.")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {input_path}")

    prior_mask = None
    prior_mask_path = None
    if args.prior_mask:
        prior_mask_path = Path(args.prior_mask).resolve()
        if not prior_mask_path.exists():
            raise FileNotFoundError(f"Prior mask not found: {prior_mask_path}")
        prior_mask = cv2.imread(str(prior_mask_path), cv2.IMREAD_GRAYSCALE)
        if prior_mask is None:
            raise RuntimeError(f"Failed to read prior mask: {prior_mask_path}")
        prior_mask = prior_mask > 0

    image_height, image_width = image_bgr.shape[:2]

    use_point_prompt = args.point_x is not None or args.point_y is not None
    use_negative_point_prompt = args.negative_point_x is not None or args.negative_point_y is not None
    use_hole_point_prompt = args.hole_point_x is not None or args.hole_point_y is not None
    use_box_prompt = any(value is not None for value in [args.box_x1, args.box_y1, args.box_x2, args.box_y2])

    if use_point_prompt and not (args.point_x is not None and args.point_y is not None):
        raise ValueError("--point-x and --point-y must be provided together.")
    if use_negative_point_prompt and not (args.negative_point_x is not None and args.negative_point_y is not None):
        raise ValueError("--negative-point-x and --negative-point-y must be provided together.")
    if use_hole_point_prompt and not (args.hole_point_x is not None and args.hole_point_y is not None):
        raise ValueError("--hole-point-x and --hole-point-y must be provided together.")
    if use_box_prompt and not all(value is not None for value in [args.box_x1, args.box_y1, args.box_x2, args.box_y2]):
        raise ValueError("--box-x1, --box-y1, --box-x2, and --box-y2 must be provided together.")
    if not use_point_prompt and not use_negative_point_prompt and not use_hole_point_prompt and not use_box_prompt:
        raise ValueError("At least one prompt must be provided. Use either point coordinates or a box.")

    positive_point_xy: tuple[int, int] | None = None
    negative_point_xy: tuple[int, int] | None = None
    hole_point_xy: tuple[int, int] | None = None
    prompt_box_xyxy: list[int] | None = None
    prompt_points: list[list[int]] = []
    prompt_labels: list[int] = []
    box = None

    if use_point_prompt:
        point_x = int(np.clip(round(args.point_x), 0, image_width - 1))
        point_y = int(np.clip(round(args.point_y), 0, image_height - 1))
        positive_point_xy = (point_x, point_y)
        prompt_points.append([point_x, point_y])
        prompt_labels.append(1)

    if use_negative_point_prompt:
        negative_point_x = int(np.clip(round(args.negative_point_x), 0, image_width - 1))
        negative_point_y = int(np.clip(round(args.negative_point_y), 0, image_height - 1))
        negative_point_xy = (negative_point_x, negative_point_y)
        prompt_points.append([negative_point_x, negative_point_y])
        prompt_labels.append(0)

    if use_hole_point_prompt:
        hole_point_x = int(np.clip(round(args.hole_point_x), 0, image_width - 1))
        hole_point_y = int(np.clip(round(args.hole_point_y), 0, image_height - 1))
        hole_point_xy = (hole_point_x, hole_point_y)

    if use_box_prompt:
        box_x1 = int(np.clip(round(args.box_x1), 0, image_width - 1))
        box_y1 = int(np.clip(round(args.box_y1), 0, image_height - 1))
        box_x2 = int(np.clip(round(args.box_x2), box_x1 + 1, image_width))
        box_y2 = int(np.clip(round(args.box_y2), box_y1 + 1, image_height))
        prompt_box_xyxy = [box_x1, box_y1, box_x2, box_y2]
        box = np.array(prompt_box_xyxy, dtype=np.float32)

    predictor = build_predictor(checkpoint_path, device)
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    point_coords = np.array(prompt_points, dtype=np.float32) if prompt_points else None
    point_labels = np.array(prompt_labels, dtype=np.int32) if prompt_labels else None
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    best_index, selection_debug = select_best_mask(masks, scores, prior_mask)
    best_mask = masks[best_index].astype(bool)
    best_score = float(scores[best_index])

    hole_mask = None
    hole_debug = {
        "applied": False,
        "reason": "no_hole_prompt",
    }
    if hole_point_xy is not None:
        hole_masks, hole_scores, _ = predictor.predict(
            point_coords=np.array([[hole_point_xy[0], hole_point_xy[1]]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        hole_index, hole_selection_debug = select_hole_mask(hole_masks, hole_scores)
        hole_mask = hole_masks[hole_index].astype(bool)
        best_mask, hole_refinement = subtract_hole_from_outer_mask(best_mask, hole_mask)
        hole_debug = {
            "applied": bool(hole_refinement["applied"]),
            "prompt_point_xy": list(hole_point_xy),
            "selection": hole_selection_debug,
            "refinement": hole_refinement,
        }

    if not hole_debug["applied"]:
        refined_mask, refinement_debug = refine_mask_with_prior(best_mask, prior_mask, selection_debug)
        best_mask = refined_mask.astype(bool)
    else:
        refinement_debug = {
            "applied": False,
            "reason": "skipped_prior_refinement_after_hole_subtraction",
        }

    if args.disable_mask_cleanup:
        cleanup_debug = {
            "applied": False,
            "reason": "disabled_by_flag",
        }
    else:
        best_mask, cleanup_debug = cleanup_final_mask(best_mask)

    overlay_bgr, bbox_xyxy = render_overlay(
        image_bgr=image_bgr,
        mask=best_mask,
        positive_point_xy=positive_point_xy,
        negative_point_xy=negative_point_xy,
        prompt_box_xyxy=prompt_box_xyxy,
        score=best_score,
        label=args.label,
        alpha=args.alpha,
    )

    mask_uint8 = best_mask.astype(np.uint8) * 255
    masked_object = np.zeros_like(image_bgr)
    masked_object[best_mask] = image_bgr[best_mask]

    cv2.imwrite(str(overlay_path), overlay_bgr)
    cv2.imwrite(str(mask_path), mask_uint8)
    cv2.imwrite(str(masked_object_path), masked_object)

    x1, y1, x2, y2 = bbox_xyxy
    area_pixels = int(best_mask.sum())
    payload = {
        "input_image_path": str(input_path),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "image_size": {"width": image_width, "height": image_height},
        "prompt_type": (
            "points_and_box"
            if use_box_prompt and (use_point_prompt or use_negative_point_prompt)
            else "points"
            if use_point_prompt or use_negative_point_prompt
            else "box"
        ),
        "prompt_point_xy": list(positive_point_xy) if positive_point_xy is not None else None,
        "prompt_negative_point_xy": list(negative_point_xy) if negative_point_xy is not None else None,
        "prompt_hole_point_xy": list(hole_point_xy) if hole_point_xy is not None else None,
        "prompt_box_xyxy": prompt_box_xyxy,
        "prior_mask_path": str(prior_mask_path) if prior_mask_path is not None else None,
        "result_image_path": str(overlay_path),
        "mask_image_path": str(mask_path),
        "masked_object_path": str(masked_object_path),
        "mask": {
            "score": best_score,
            "area_pixels": area_pixels,
            "area_ratio": float(area_pixels / float(image_width * image_height)),
            "bbox_xyxy": bbox_xyxy,
            "bbox_xywh": [x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
            "prompt_type": (
                "points_and_box"
                if use_box_prompt and (use_point_prompt or use_negative_point_prompt)
                else "points"
                if use_point_prompt or use_negative_point_prompt
                else "box"
            ),
            "prompt_point_xy": list(positive_point_xy) if positive_point_xy is not None else None,
            "prompt_negative_point_xy": list(negative_point_xy) if negative_point_xy is not None else None,
            "prompt_hole_point_xy": list(hole_point_xy) if hole_point_xy is not None else None,
            "prompt_box_xyxy": prompt_box_xyxy,
            "contour_count": int(len(cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])),
            "selection": selection_debug,
            "hole_subtraction": hole_debug,
            "refinement": refinement_debug,
            "cleanup": cleanup_debug,
        },
    }
    save_json(json_path, payload)

    print(f"Saved result image: {overlay_path}")
    print(f"Saved mask image:   {mask_path}")
    print(f"Saved mask JSON:    {json_path}")


if __name__ == "__main__":
    main()
