from __future__ import annotations

import unittest

from services.vision_service.config import PartsLexicon
from services.vision_service.step_checker import check_step


class VisionServiceStepCheckerTest(unittest.TestCase):
    """Focused tests for translating CV evidence into step-check observations."""

    def setUp(self) -> None:
        # The tests use tiny in-memory configs so they do not depend on the
        # sample YAML files or on expensive model inference.
        self.lexicon = PartsLexicon(
            {
                "bearing": {"aliases_en": ["bearing"], "criticality": "high"},
                "flexspline": {"aliases_en": ["flexspline"], "criticality": "high"},
            }
        )
        self.safety_rules = {
            "confidence_thresholds": {
                "vision_high": 0.9,
                "vision_low": 0.7,
                "vision_detection_min": 0.3,
                "segmentation_low": 0.7,
            }
        }
        self.camera_config = {
            "cameras": [
                {
                    "camera_id": "CAM-A01",
                    "view_id": "top",
                    "rois": {
                        "full_image": {"bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]},
                    },
                }
            ]
        }

    def test_low_confidence_expected_part_needs_confirmation(self) -> None:
        """A detected required part below vision_low should request confirmation."""
        dino_result = {
            "image_size": {"width": 100, "height": 100},
            "detections": [
                {
                    "id": 1,
                    "label": "bearing",
                    "confidence": 0.42,
                    "bbox_xyxy": [10, 10, 40, 40],
                    "crop_image_path": "crop.png",
                }
            ],
        }
        sam_results = [
            {
                "detection_id": 1,
                "label": "bearing",
                "output": {
                    "result_image_path": "overlay.png",
                    "mask_image_path": "mask.png",
                    "mask": {"score": 0.95, "area_ratio": 0.2, "bbox_xyxy": [0, 0, 20, 20]},
                },
            }
        ]
        workflow_config = {
            "steps": {
                "S01": {
                    "expected_parts": [
                        {"part": "bearing", "required": True, "roi": "full_image", "min_confidence": 0.3}
                    ]
                }
            }
        }

        detections, step_status, step_confidence, notes = check_step(
            dino_result=dino_result,
            sam_results=sam_results,
            workflow_config=workflow_config,
            safety_rules=self.safety_rules,
            camera_config=self.camera_config,
            parts_lexicon=self.lexicon,
            step_id="S01",
            text_prompt="bearing",
            camera_id="CAM-A01",
            view_id="top",
        )

        self.assertEqual(step_status, "needs_confirmation")
        self.assertEqual(step_confidence, 0.42)
        self.assertEqual(notes, [])
        self.assertEqual(detections[0].status, "low_confidence")
        self.assertEqual(detections[0].roi_status, "inside")

    def test_missing_critical_part_fails_step(self) -> None:
        """A missing high-criticality required part should fail the step check."""
        workflow_config = {
            "steps": {
                "S02": {
                    "expected_parts": [
                        {"part": "flexspline", "required": True, "roi": "full_image", "min_confidence": 0.3}
                    ]
                }
            }
        }
        detections, step_status, step_confidence, _ = check_step(
            dino_result={"image_size": {"width": 100, "height": 100}, "detections": []},
            sam_results=[],
            workflow_config=workflow_config,
            safety_rules=self.safety_rules,
            camera_config=self.camera_config,
            parts_lexicon=self.lexicon,
            step_id="S02",
            text_prompt="flexspline",
            camera_id="CAM-A01",
            view_id="top",
        )

        self.assertEqual(step_status, "failed")
        self.assertEqual(step_confidence, 0.0)
        self.assertEqual(detections[0].status, "missing")


if __name__ == "__main__":
    unittest.main()
