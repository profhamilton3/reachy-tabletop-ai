"""Physical stereo profile: the versioned default loads, impossible geometry is refused (#33)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from reachy_ai.perception.stereo_depth import (
    DEFAULT_CALIBRATION,
    StereoGeometry,
    check_physical_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "calibration" / "historical" / "stereo_calibration_2026_08_21_REJECTED.npz"


def profile():
    return json.loads(DEFAULT_CALIBRATION.read_text())


def parts(doc):
    return ([np.array(doc[e]["K"]) for e in ("left", "right")], [np.array(doc[e]["dist"]) for e in ("left", "right")],
            np.array(doc["R"]), np.array(doc["T"]))


def test_default_is_the_versioned_profile_and_it_loads():
    assert DEFAULT_CALIBRATION.name == "reachy12_stereo_2026_08_27.json"
    g = StereoGeometry("robot")
    assert g.error is None
    assert g.identity["profile_id"] == "reachy12-stereo-2026-08-27-v1"
    assert g.identity["baseline_m"] == pytest.approx(0.078, abs=0.003)
    assert g.size == (480, 640)
    assert "never recorded" in g.identity["zoom_caveat"]
    doc = profile()
    for key in ("pictures", "method", "date_derived", "intrinsics", "pnp_rms_px_max"):
        assert doc["provenance"][key]
    assert doc["provenance"]["pnp_rms_px_max"] <= 1.3
    # Measured pose from #33: right eye ~ (0.0746, 0.0093, 0.0222) m in the left frame.
    assert doc["right_centre_in_left_m"] == pytest.approx([0.0746, 0.0093, 0.0222], abs=0.002)
    g.prepare(np.zeros((640, 480, 3), np.uint8).shape)  # rectifies as left-to-right horizontal stereo
    assert g.maps is not None


def test_old_npz_is_refused_with_the_geometry_reason():
    g = StereoGeometry("robot", calibration=HISTORICAL)
    assert g.error and g.error.startswith("Saved stereo profile refused.")
    for reason in ("baseline", "along the optical axis", "focal lengths differ", "distortion"):
        assert reason in g.error
    assert g.status()["available"] is False
    assert not (ROOT / "scripts" / "stereo_calibration.npz").exists()


def test_regenerated_profile_matches_the_committed_one():
    sys.path.insert(0, str(ROOT / "scripts"))
    import derive_stereo_calibration as derive
    rows, rotation, translation, centre = derive.derive()
    committed = profile()
    assert len(rows) == len(committed["provenance"]["pairs_used"]) >= 4
    # The derivation varies by about 0.5 mm across OpenCV builds (CI differs from the build that
    # derived the committed profile); the pair-to-pair spread is 3.6-3.9 mm.
    np.testing.assert_allclose(centre, committed["right_centre_in_left_m"], atol=1e-3)
    np.testing.assert_allclose(translation, committed["T"], atol=1e-3)


@pytest.mark.parametrize("change, reason", [
    (lambda K, D, R, T: (K, D, R, T * 3), "baseline"),
    (lambda K, D, R, T: (K, D, R, T * 0.3), "baseline"),
    (lambda K, D, R, T: (K, D, R, np.array([-0.08, 0, 0.2])), "optical axis"),
    (lambda K, D, R, T: ([K[0], K[1] * [[2.2, 1, 1], [1, 2.2, 1], [1, 1, 1]]], D, R, T), "focal lengths differ"),
    (lambda K, D, R, T: (K, [D[0], np.array([4.0, -7.0, 0, 0, 7.4])], R, T), "distortion"),
    (lambda K, D, R, T: (K, [np.array([-0.3, 0.1, 0.2, 0, 0]), D[1]], R, T), "distortion"),
])
def test_each_impossible_feature_is_named(change, reason):
    reasons = check_physical_geometry(*change(*parts(profile())))
    assert any(reason in r for r in reasons), reasons


def test_good_profile_has_no_reasons_and_sim_is_not_checked():
    assert check_physical_geometry(*parts(profile())) == []
    assert StereoGeometry("sim").error is None


def test_a_tampered_profile_file_is_refused(tmp_path):
    doc = profile()
    doc["T"] = [-0.5, 0, 0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    g = StereoGeometry("robot", calibration=path)
    assert g.error and "baseline" in g.error
