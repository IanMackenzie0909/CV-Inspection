from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def candidate_python_paths(base_dir: Path) -> list[Path]:
    """Return common virtual-environment Python executable locations."""
    return [
        base_dir / "bin" / "python",
        base_dir / "Scripts" / "python.exe",
    ]


def resolve_python(explicit_path: str | None, fallback_dirs: list[Path]) -> Path:
    """Find the Python executable used to run an external model script."""
    if explicit_path:
        python_path = Path(explicit_path).expanduser()
        if not python_path.is_absolute():
            python_path = (Path.cwd() / python_path).absolute()
        if python_path.exists():
            return python_path
        raise FileNotFoundError(f"Python executable not found: {python_path}")

    for directory in fallback_dirs:
        for candidate in candidate_python_paths(directory):
            if candidate.exists():
                return candidate.absolute()

    # Fall back to the current interpreter only after checking project-specific
    # virtual environments.
    return Path(sys.executable).absolute()


def run_command(command: list[str], cwd: Path) -> None:
    """Run a model subprocess and surface failures to the caller."""
    print(f"Running in {cwd}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def build_groundingdino_command(
    python_path: Path,
    script_path: Path,
    image_path: Path,
    text_prompt: str,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    box_threshold: float,
    text_threshold: float,
    device: str,
    max_detections: int,
) -> list[str]:
    """Build the GroundingDINO subprocess command in one auditable place."""
    return [
        str(python_path),
        str(script_path),
        "--image",
        str(image_path),
        "--text-prompt",
        text_prompt,
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--box-threshold",
        str(box_threshold),
        "--text-threshold",
        str(text_threshold),
        "--device",
        device,
        "--max-detections",
        str(max_detections),
    ]


def build_mobilesam_command(
    python_path: Path,
    script_path: Path,
    detection: dict,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
) -> list[str]:
    """Build the MobileSAM command from one GroundingDINO detection record."""
    crop_width = int(detection["crop_size"]["width"])
    crop_height = int(detection["crop_size"]["height"])
    foreground_point = detection.get("sam_foreground_point_in_crop_xy")
    background_point = detection.get("sam_background_point_in_crop_xy")
    hole_point = detection.get("center_in_crop_xy")

    command = [
        str(python_path),
        str(script_path),
        "--input",
        detection["crop_image_path"],
        "--box-x1",
        "0",
        "--box-y1",
        "0",
        "--box-x2",
        str(crop_width),
        "--box-y2",
        str(crop_height),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--label",
        detection["label"],
    ]
    if detection.get("sam_foreground_prior_mask_path"):
        # The prior mask helps MobileSAM select the candidate mask that best
        # matches the foreground estimate from the crop.
        command.extend(["--prior-mask", detection["sam_foreground_prior_mask_path"]])
    if foreground_point is not None:
        # Positive and negative points make the segmentation less dependent on
        # the raw bounding box alone.
        command.extend(["--point-x", str(foreground_point[0]), "--point-y", str(foreground_point[1])])
    if background_point is not None:
        command.extend(
            [
                "--negative-point-x",
                str(background_point[0]),
                "--negative-point-y",
                str(background_point[1]),
            ]
        )
    if hole_point is not None:
        command.extend(["--hole-point-x", str(hole_point[0]), "--hole-point-y", str(hole_point[1])])
    return command
