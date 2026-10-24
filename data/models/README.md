# data/models

## best.onnx — ONNX copy of the tabletop detector (#75)

The OpenCV-DNN engine in `src/reachy_ai/perception/pair_detector.py` loads this file when no TFLite runtime is
installed (Intel Macs). It must behave like the deployed `tests/best.tflite`
(SHA-256 `60dad10f…ff7e`; full hashes in `best.onnx.provenance.json`, which the loader and
`tests/test_detector_onnx.py` verify).

- **Route used: `tf2onnx` conversion of `tests/best.tflite`** (the `--from-tflite` fallback), opset 12, input
  RGB float32 NCHW `[1, 3, 320, 320]`, output `[1, 7, 2100]` with normalized boxes.
  Agreement with the TFLite engine on the 9 fixture images with detections: identical label sets and counts,
  worst IoU 1.000000, worst confidence difference 0.000003.
- **Route tried first: Ultralytics `best.pt` -> ONNX** (`yolov8n_board-4`, from
  `data/annotations/my_yolo_dataset-20260925T151100Z-1-001.zip`; ultralytics 8.4.163, torch 2.5.1, onnx 1.17.0,
  `opset=12`, `imgsz=320`). It did **not** reproduce the deployed model: worst IoU 0.0 (missing matches) and
  confidence differences up to 0.68, with different label sets on 1 of 9 images (`E_full_swapped`).
  So the shipped `best.tflite` is not numerically the same as that checkpoint (different or later weights).
- `imgsz` is 320 (not 640): the deployed model's input is 320x320.
- Regenerate with `python scripts/export_detector_onnx.py --from-tflite` (needs `python3.10`; installs
  tf2onnx and TensorFlow into a throwaway venv outside the repo; nothing becomes a dependency of the viewer).
