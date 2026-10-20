"""In-memory zip of one retained stereo pair; nothing is written to disk or sent anywhere."""

import hashlib
import io
import json
import zipfile
from importlib.metadata import PackageNotFoundError, version

MAX_BUNDLE_BYTES = 16 * 1024 * 1024
ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # fixed, so the same pair always yields the same bytes


class BundleTooLarge(ValueError):
    pass


def viewer_version():
    try:
        return version("reachy-ai")
    except PackageNotFoundError:
        return None


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def build_bundle(files, versions, calibration_id):
    """files: {name: bytes}, in archive order. Returns zip bytes; manifest.json lists every other file."""
    manifest = {"schema_version": "1", "calibration_id": calibration_id, "versions": versions,
                "files": [{"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in files.items()]}
    total = sum(len(d) for d in files.values())
    if total > MAX_BUNDLE_BYTES:
        raise BundleTooLarge(f"Pair is {total / 2**20:.1f} MiB; the limit is {MAX_BUNDLE_BYTES // 2**20} MiB")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in {**files, "manifest.json": json_bytes(manifest)}.items():
            info = zipfile.ZipInfo(name, ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED if name.endswith((".jpg", ".png")) else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    if out.tell() > MAX_BUNDLE_BYTES:
        raise BundleTooLarge("Bundle exceeds the size limit")
    return out.getvalue()
