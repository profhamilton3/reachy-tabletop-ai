"""ONNX (OpenCV DNN) detector engine: agreement with TFLite, no-TFLite smoke, provenance, import guard (#75)."""

import ast
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import cv2
import pytest

from reachy_ai.perception import pair_detector
from reachy_ai.perception.pair_detector import DEFAULT_MODEL, ONNX_ENGINE, ONNX_MODEL, PairDetector, engine_name

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ONNX_MODEL.with_name(ONNX_MODEL.name + ".provenance.json")
ZIP_PATH = ROOT / "data" / "annotations" / "my_yolo_dataset-20260925T151100Z-1-001.zip"
FIXTURE_IMAGES = sorted([*(ROOT / "tests" / "fixtures" / "t3").glob("*/right.jpg"),
                         *(ROOT / "tests" / "fixtures").glob("sim_head_sweep*/**/right.jpg")])
IOU_MIN, CONFIDENCE_TOL = 0.9, 0.05


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def run_engine(monkeypatch, engine, pixels):
    monkeypatch.setenv(pair_detector.ENGINE_ENV, engine)
    return PairDetector().analyze(pixels)


def compare(tflite, onnx):
    """Return (label sets equal, worst IoU, worst confidence difference, all boxes matched)."""
    same_labels = {d["label"] for d in tflite["detections"]} == {d["label"] for d in onnx["detections"]}
    worst_iou, worst_conf, matched = 1.0, 0.0, len(tflite["detections"]) == len(onnx["detections"])
    free = list(onnx["detections"])
    for want in tflite["detections"]:
        options = [(iou(want["box_xywh"], got["box_xywh"]), got) for got in free if got["label"] == want["label"]]
        if not options:
            matched = False
            continue
        best, got = max(options, key=lambda item: item[0])
        free.remove(got)
        worst_iou = min(worst_iou, best)
        worst_conf = max(worst_conf, abs(want["confidence"] - got["confidence"]))
    return same_labels, worst_iou, worst_conf, matched


def test_agreement_with_tflite_engine(monkeypatch, capsys):
    if pair_detector.runtime_name() is None:
        pytest.skip("No TFLite runtime (ai_edge_litert or tflite_runtime) is installed; agreement needs both engines")
    assert FIXTURE_IMAGES, "fixture images missing"
    worst_iou, worst_conf, compared = 1.0, 0.0, 0
    for path in FIXTURE_IMAGES:
        pixels = cv2.imread(str(path))
        tflite = run_engine(monkeypatch, "tflite", pixels)
        if not tflite["detections"]:
            continue
        onnx = run_engine(monkeypatch, "onnx", pixels)
        same_labels, image_iou, image_conf, matched = compare(tflite, onnx)
        assert same_labels, f"{path.parent.name}: label sets differ"
        assert matched, f"{path.parent.name}: boxes are not one-to-one"
        assert image_iou >= IOU_MIN, f"{path.parent.name}: IoU {image_iou:.3f}"
        assert image_conf <= CONFIDENCE_TOL, f"{path.parent.name}: confidence differs by {image_conf:.3f}"
        worst_iou, worst_conf, compared = min(worst_iou, image_iou), max(worst_conf, image_conf), compared + 1
    assert compared >= 3
    with capsys.disabled():
        print(f"\n[onnx agreement] images={compared} worst_iou={worst_iou:.6f} worst_conf_diff={worst_conf:.6f}")


def test_onnx_engine_runs_with_tflite_hidden(monkeypatch):
    real_find_spec = importlib.util.find_spec
    hidden = {"ai_edge_litert", "tflite_runtime"}
    monkeypatch.setattr(pair_detector.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name in hidden else real_find_spec(name, *a, **k))
    for name in hidden:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delenv(pair_detector.ENGINE_ENV, raising=False)
    assert pair_detector.runtime_name() is None and engine_name() == ONNX_ENGINE
    detector = PairDetector()
    assert detector.status()["available"] and detector.status()["engine"] == ONNX_ENGINE
    result = detector.analyze(cv2.imread(str(FIXTURE_IMAGES[0])))
    assert result["engine"] == ONNX_ENGINE and result["model"]["engine"] == ONNX_ENGINE
    assert result["model"]["sha256"] == sha256(ONNX_MODEL)
    assert result["model"]["labels"] == ["cube", "cylinder", "empty"]
    for item in result["detections"]:
        assert set(item) == {"class_id", "label", "confidence", "box_xywh"}
        assert item["label"] in result["model"]["labels"] and len(item["box_xywh"]) == 4


def test_engine_order_and_override(monkeypatch):
    monkeypatch.delenv(pair_detector.ENGINE_ENV, raising=False)
    monkeypatch.setattr(pair_detector, "runtime_name", lambda: "tflite_runtime")
    assert engine_name() == "tflite_runtime"
    monkeypatch.setenv(pair_detector.ENGINE_ENV, "onnx")
    assert engine_name() == ONNX_ENGINE
    monkeypatch.setenv(pair_detector.ENGINE_ENV, "nonsense")
    assert not PairDetector().status()["available"]


def test_provenance_matches_files():
    record = json.loads(PROVENANCE.read_text())
    assert record["onnx_sha256"] == sha256(ONNX_MODEL) and record["onnx_bytes"] == ONNX_MODEL.stat().st_size
    assert record["reproduces"]["sha256"] == sha256(DEFAULT_MODEL)
    assert record["source_zip_sha256"] == sha256(ZIP_PATH)
    assert record["output_box_units"] in ("normalized", "input_pixels")
    with zipfile.ZipFile(DEFAULT_MODEL) as archive:
        names = json.loads(archive.read("metadata.json"))["names"]
    assert record["classes"] == [names[str(i)] for i in range(len(names))] == pair_detector.LABELS
    assert {"route", "versions", "export_settings"} <= set(record)
    if record["route"] == "ultralytics-pt":
        assert len(record["source_best_pt_sha256"]) == 64


def test_no_export_toolchain_imported_in_src():
    banned = {"ultralytics", "torch", "tf2onnx"}
    found = []
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            found += [f"{path.name}: {m}" for m in modules if m.split(".")[0] in banned]
    assert found == []
