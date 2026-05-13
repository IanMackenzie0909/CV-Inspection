# GroundingDINO + MobileSAM Pipeline

## 正式入口

```bash
python3 run_grounded_mobilesam_pipeline.py \
  --image harmonic_drive/bearing.jpg \
  --text-prompt "bearing" \
  --output-dir Global-output \
  --work-order-id WO-20260427-001 \
  --station-id ST-A01 \
  --step-id S01 \
  --camera-id CAM-A01 \
  --view-id top
```

## 指定兩個虛擬環境的 Python

```bash
python3 run_grounded_mobilesam_pipeline.py \
  --image harmonic_drive/bearing.jpg \
  --text-prompt "bearing" \
  --output-dir Global-output \
  --dino-python GroundingDINO/env/bin/python \
  --sam-python env/bin/python
```

## 強制使用 CPU

```bash
python3 run_grounded_mobilesam_pipeline.py \
  --image harmonic_drive/bearing.jpg \
  --text-prompt "bearing" \
  --output-dir Global-output \
  --dino-device cpu \
  --sam-device cpu
```

## 預設行為

- `GroundingDINO` 先做文字提示偵測
- 依偵測框裁切出目標區域
- 自動估 `foreground point`、`background point`、`hole point`
- `MobileSAM` 使用 `box + points` 做分割
- 最後做保守型 cleanup，輸出整體輪廓遮罩

## 主要輸出

- `Global-output/groundingdino/groundingdino_annotated.png`
- `Global-output/groundingdino/detections.json`
- `Global-output/mobilesam/detection_001/result_overlay.png`
- `Global-output/mobilesam/detection_001/object_mask.png`
- `Global-output/mobilesam/detection_001/masked_object.png`
- `Global-output/mobilesam/detection_001/mask_result.json`
- `Global-output/pipeline_result.json`
- `Global-output/vision_result.json`
- `Global-output/vision_event.json`
- `Global-output/evidence/IMG-*_original.*`

## V-RAWA 影像辨識層輸出

`vision_result.json` 是給 HMI、除錯與稽核使用的影像層結果，包含：

- `work_order_id`、`station_id`、`step_id`
- `camera_id`、`view_id`、`image_id`
- `detections[].part/status/confidence/bbox/roi`
- `defects[]`
- `step_status`
- `step_status_confidence`
- `evidence`

`vision_event.json` 是給事件引擎使用的 `VISION_STEP_CHECKED` 事件，保留：

- `event_id`
- `correlation_id`
- `idempotency_key`
- `confidence`
- `model`
- `rule`
- `media`
- `operator_confirmation`

低信心結果會輸出 `step_status = needs_confirmation`，不會直接當成高風險放行或自動上報。

## 設定檔

- `configs/parts_lexicon.yaml`：零件名稱、alias、criticality
- `configs/workflow_steps.yaml`：每個 step 的 expected parts 與 ROI
- `configs/safety_rules.yaml`：confidence threshold、風險與覆寫規則
- `configs/camera_config.yaml`：camera/view/ROI 設定
- `configs/model_registry.yaml`：模型名稱與版本紀錄

## JSON 重點欄位

### `detections.json`

- `sam_foreground_point_in_crop_xy`
- `sam_background_point_in_crop_xy`
- `center_in_crop_xy`
- `sam_box_prompt_in_crop_xyxy`

### `mask_result.json`

- `prompt_point_xy`
- `prompt_negative_point_xy`
- `prompt_hole_point_xy`
- `prompt_box_xyxy`
- `mask.selection`
- `mask.hole_subtraction`
- `mask.cleanup`

## 如果之後要關掉 cleanup

- 目前總控腳本還沒有直接暴露這個開關
- `MobileSAM` 子腳本可單獨使用：

```bash
env/bin/python MobileSAM-fast-finetuning/demo_point_prompt.py --help
```

- 其中可加：

```bash
--disable-mask-cleanup
```
