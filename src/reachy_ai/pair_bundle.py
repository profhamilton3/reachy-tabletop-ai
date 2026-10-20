"""In-memory zip of retained stereo pairs (one pair, or a whole data-capture run); nothing is written
to disk or sent anywhere. A capture-run bundle is the same format: the pairs sit under steps/ and
capture.json carries the procedure, step labels and zoom per step; the manifest lists every file."""

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


def build_bundle(files, versions, calibration_id, procedure=None):
    """files: {name: bytes}, in archive order. Returns zip bytes; manifest.json lists every other file.

    `procedure` ({id, name, version}) is added to the manifest for a data-capture run."""
    manifest = {"schema_version": "1", "calibration_id": calibration_id, "versions": versions,
                **({"procedure": procedure} if procedure else {}),
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


class BadBundle(ValueError):
    pass


def read_bundle(source):
    """Open a returned bundle (path or bytes) and check every file against the manifest's SHA-256.

    Returns (manifest, {name: bytes}) without manifest.json. Raises BadBundle on a missing,
    extra or altered file."""
    handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        with zipfile.ZipFile(handle) as archive:
            contents = {name: archive.read(name) for name in archive.namelist()}
    except zipfile.BadZipFile as exc:
        raise BadBundle("not a zip file") from exc
    if "manifest.json" not in contents:
        raise BadBundle("manifest.json is missing")
    manifest = json.loads(contents.pop("manifest.json"))
    listed = {entry["name"]: entry for entry in manifest.get("files", [])}
    if set(listed) != set(contents):
        raise BadBundle("files differ from the manifest: "
                        + ", ".join(sorted(set(listed) ^ set(contents))))
    for name, data in contents.items():
        if hashlib.sha256(data).hexdigest() != listed[name]["sha256"]:
            raise BadBundle(f"{name} does not match its SHA-256 in the manifest")
    return manifest, contents
