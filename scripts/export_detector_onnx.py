#!/usr/bin/env python3
"""One-time export of the tabletop detector to ONNX (#75).

Run once on a machine that can install PyTorch (Apple Silicon is fine):

    python scripts/export_detector_onnx.py            # best.pt -> ONNX with Ultralytics
    python scripts/export_detector_onnx.py --from-tflite   # fallback: tests/best.tflite -> ONNX with tf2onnx

This script uses only the standard library. Ultralytics, PyTorch, onnx and tf2onnx are installed into a
THROWAWAY virtual environment outside the repo and never become dependencies of the viewer.

The deployed detector (tests/best.tflite) takes RGB float32 NCHW 320x320 and returns [1, 7, 2100]. The ONNX
export therefore uses imgsz=320, not 640, so both engines see the same input and return the same output shape.

Outputs: data/models/best.onnx and data/models/best.onnx.provenance.json.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "annotations" / "my_yolo_dataset-20260925T151100Z-1-001.zip"
WEIGHTS_MEMBER = "my_yolo_dataset/runs/yolov8n_board-4/weights/best.pt"
TFLITE_PATH = ROOT / "tests" / "best.tflite"
OUT_DIR = ROOT / "data" / "models"
OUT_ONNX = OUT_DIR / "best.onnx"
OUT_PROVENANCE = OUT_DIR / "best.onnx.provenance.json"

# Versions the export was run with. Ultralytics matches the 8.4.163 recorded in best.tflite's metadata.
PINNED = {
    "pt": ["ultralytics==8.4.163", "torch==2.5.1", "onnx==1.17.0", "onnxslim==0.1.48", "onnxruntime==1.20.1"],
    "tflite": ["tf2onnx==1.16.1", "tensorflow==2.15.1", "numpy<2", "onnx==1.16.2"],
}
SETTINGS = {"format": "onnx", "opset": 12, "imgsz": 320, "dynamic": False, "simplify": True, "half": False}
EXPECTED_CLASSES = ["cube", "cylinder", "empty"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inner_export(weights, out_dir):
    """Runs inside the throwaway venv: YOLO(best.pt).export(...). Prints a JSON result line."""
    import onnx
    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names
    classes = [names[i] for i in range(len(names))]
    produced = model.export(**SETTINGS)
    target = Path(out_dir) / "best.onnx"
    Path(produced).replace(target)
    print("RESULT " + json.dumps({"onnx": str(target), "classes": classes,
                                  "versions": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                                               "onnx": onnx.__version__}}))


def make_venv(scratch, packages):
    venv = scratch / "venv"
    subprocess.run(["python3.10", "-m", "venv", str(venv)], check=True)
    python = venv / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", *packages], check=True)
    return python


def export_from_pt(scratch):
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extract(WEIGHTS_MEMBER, scratch / "zip")
    weights = scratch / "zip" / WEIGHTS_MEMBER
    python = make_venv(scratch, PINNED["pt"])
    work = scratch / "out"
    work.mkdir()
    done = subprocess.run([str(python), str(Path(__file__).resolve()), "--inner", str(weights), str(work)],
                          check=True, capture_output=True, text=True)
    result = json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT "))[7:])
    return Path(result["onnx"]), {
        "route": "ultralytics-pt",
        "source_best_pt_member": WEIGHTS_MEMBER,
        "source_best_pt_sha256": sha256(weights),
        "versions": result["versions"],
        "export_settings": SETTINGS,
        "classes": result["classes"],
        "output_box_units": "input_pixels",
    }


def export_from_tflite(scratch):
    python = make_venv(scratch, PINNED["tflite"])
    target = scratch / "best.onnx"
    subprocess.run([str(python), "-m", "tf2onnx.convert", "--tflite", str(TFLITE_PATH), "--output", str(target),
                    "--opset", str(SETTINGS["opset"])], check=True)
    versions = json.loads(subprocess.run(
        [str(python), "-c", "import json,onnx,tf2onnx,tensorflow as t;"
         "print(json.dumps({'tf2onnx':tf2onnx.__version__,'tensorflow':t.__version__,'onnx':onnx.__version__}))"],
        check=True, capture_output=True, text=True).stdout)
    return target, {"route": "tf2onnx-tflite-fallback", "versions": versions,
                    "export_settings": {"opset": SETTINGS["opset"], "converter": "tf2onnx --tflite"},
                    "classes": EXPECTED_CLASSES,
                    "output_box_units": "normalized",
                    "note": "Fallback: used because the best.pt export did not agree with best.tflite."}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--inner":
        inner_export(sys.argv[2], sys.argv[3])
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-tflite", action="store_true",
                        help="fallback: convert tests/best.tflite with tf2onnx (only if .pt fails agreement)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="detector-onnx-") as name:
        scratch = Path(name)
        onnx_path, record = export_from_tflite(scratch) if args.from_tflite else export_from_pt(scratch)
        if record["classes"] != EXPECTED_CLASSES:
            sys.exit(f"Unexpected classes {record['classes']}; expected {EXPECTED_CLASSES}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_ONNX.write_bytes(onnx_path.read_bytes())
    record.update({
        "onnx_file": OUT_ONNX.name,
        "onnx_sha256": sha256(OUT_ONNX),
        "onnx_bytes": OUT_ONNX.stat().st_size,
        "source_zip": ZIP_PATH.name,
        "source_zip_sha256": sha256(ZIP_PATH),
        "reproduces": {"file": "tests/best.tflite", "sha256": sha256(TFLITE_PATH)},
        "input": "RGB float32 NCHW [1, 3, 320, 320], values 0..1, resized without padding",
        "output": "float32 [1, 7, 2100]: cx, cy, w, h (see output_box_units) + 3 class scores",
    })
    OUT_PROVENANCE.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUT_ONNX} ({record['onnx_bytes']} bytes) and {OUT_PROVENANCE.name}")


if __name__ == "__main__":
    main()
