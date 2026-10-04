"""On-demand CPU inference using the existing exported tabletop YOLO model.

The repository's best.tflite model has RGB float NCHW input and normalized
xywh + class-score output. This follows testbestInt8_Tflite.ipynb, converting
SDK BGR pixels to RGB exactly once. No training or camera access occurs here.
"""

import hashlib
import importlib.util
import json
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "tests" / "best.tflite"


class DetectorUnavailable(RuntimeError):
    pass


class DetectorBusy(RuntimeError):
    pass


def runtime_name():
    for name in ("ai_edge_litert", "tflite_runtime"):
        if importlib.util.find_spec(name) is not None:
            return name
    return None


def decode_predictions(output, labels, width, height, confidence=0.30, nms_iou=0.45):
    """Map normalized model boxes to source pixels with class-aware NMS."""
    expected = 4 + len(labels)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != expected:
        raise DetectorUnavailable("Unsupported detector output; expected [1, 4 + classes, candidates]")
    boxes, scores, classes = [], [], []
    for row in output[0].T:
        if not np.isfinite(row).all():
            continue
        class_id = int(np.argmax(row[4:]))
        score = float(row[4 + class_id])
        if score < confidence or row[2] <= 0 or row[3] <= 0:
            continue
        center_x, center_y, box_w, box_h = row[:4]
        x1 = int(np.clip(round((center_x - box_w / 2) * width), 0, width))
        y1 = int(np.clip(round((center_y - box_h / 2) * height), 0, height))
        x2 = int(np.clip(round((center_x + box_w / 2) * width), 0, width))
        y2 = int(np.clip(round((center_y + box_h / 2) * height), 0, height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(score)
        classes.append(class_id)
    detections = []
    for class_id in sorted(set(classes)):
        candidates = [i for i, value in enumerate(classes) if value == class_id]
        keep = cv2.dnn.NMSBoxes([boxes[i] for i in candidates], [scores[i] for i in candidates],
                               confidence, nms_iou)
        for index in np.asarray(keep).reshape(-1):
            i = candidates[int(index)]
            detections.append({"class_id": class_id, "label": labels[class_id],
                               "confidence": round(scores[i], 6), "box_xywh": boxes[i]})
    return sorted(detections, key=lambda d: d["confidence"], reverse=True)


def draw_detections(pixels, detections):
    """Annotate a copy; source frames remain immutable."""
    image = pixels.copy()
    palette = ((128, 203, 196), (249, 202, 144), (125, 212, 129))
    for item in detections:
        x, y, w, h = item["box_xywh"]
        color = palette[item["class_id"] % len(palette)]
        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), color, 2)
        label = f"{item['label']} {item['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = min(image.shape[0] - 5, max(th + 5, y))
        tx = min(x, max(0, image.shape[1] - tw - 8))
        cv2.rectangle(image, (tx, ty - th - 5), (tx + tw + 6, ty + 3), (20, 25, 35), -1)
        cv2.putText(image, label, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return image


class PairDetector:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.path = Path(model_path).expanduser().resolve()
        self.lock = threading.Lock()
        self.interpreter = None
        self.model = None
        self.last_error = None

    def status(self):
        available = self.path.is_file() and runtime_name() is not None
        reason = self.last_error
        if not self.path.is_file():
            reason = "Detector model missing. Set --detector-model to the existing best.tflite file."
        elif runtime_name() is None:
            reason = "Install the analysis extra on the workstation: pip install -e '.[analysis]'"
        return {"available": available and reason is None, "model_name": self.path.name,
                "reason": reason, "eye": "right", "execution": "on demand, workstation CPU"}

    def _load(self):
        if self.interpreter is not None:
            return
        state = self.status()
        if not state["available"]:
            raise DetectorUnavailable(state["reason"])
        try:
            if runtime_name() == "ai_edge_litert":
                from ai_edge_litert.interpreter import Interpreter
            else:
                from tflite_runtime.interpreter import Interpreter
            with zipfile.ZipFile(self.path) as archive:
                metadata = json.loads(archive.read("metadata.json"))
            names = metadata["names"]
            self.labels = [names[str(i)] for i in range(len(names))] if isinstance(names, dict) else list(names)
            if metadata.get("task") != "detect" or self.labels != ["cube", "cylinder", "empty"]:
                raise ValueError("Expected the existing cube/cylinder/empty detector export")
            interpreter = Interpreter(model_path=str(self.path), num_threads=2)
            interpreter.allocate_tensors()
            inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Expected one input and one output tensor")
            self.input, self.output = inputs[0], outputs[0]
            if list(self.input["shape"]) != [1, 3, 320, 320] or self.input["dtype"] != np.float32:
                raise ValueError("Expected RGB float32 input [1, 3, 320, 320]")
            if list(self.output["shape"]) != [1, 7, 2100] or self.output["dtype"] != np.float32:
                raise ValueError("Expected float32 output [1, 7, 2100]")
            self.model = {"name": self.path.name, "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                          "labels": self.labels, "export_version": metadata.get("version"),
                          "export_date": metadata.get("date"), "runtime": runtime_name()}
            self.interpreter = interpreter
        except DetectorUnavailable:
            raise
        except Exception as exc:
            self.last_error = f"Unable to load the supported detector ({type(exc).__name__}). Check model/runtime."
            raise DetectorUnavailable(self.last_error) from exc

    def analyze(self, pixels):
        if not self.lock.acquire(blocking=False):
            raise DetectorBusy("An analysis is already running. Please try again shortly.")
        try:
            self._load()
            started = time.monotonic()
            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (320, 320)).astype(np.float32) / 255.0
            tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
            self.interpreter.set_tensor(self.input["index"], tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output["index"])
            height, width = pixels.shape[:2]
            detections = decode_predictions(output, self.labels, width, height)
            return {"model": dict(self.model), "detections": detections,
                    "parameters": {"confidence_threshold": 0.30, "nms_iou_threshold": 0.45,
                                   "input": "RGB float32 NCHW 320x320, resized without padding",
                                   "box_format": "source-pixel xywh"},
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 1)}
        except (DetectorUnavailable, DetectorBusy):
            raise
        except Exception as exc:
            raise DetectorUnavailable(f"Detector inference failed ({type(exc).__name__})") from exc
        finally:
            self.lock.release()
