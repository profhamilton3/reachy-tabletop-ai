"""Focus check (#77): sharpness on the board, the focus sweep, lens commands behind the confirm and dry-run rules,
the exact lens restore, the focus-folded calibration id, the zoom FOV check v5 and the camera_focus inventory probe.

Offline: a fake lens with a synthetic sharpness model (peaked, flat, noisy) stands in for the SDK and the camera."""

import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from reachy_ai.camera_zoom import LensCommander, calibration_id, focus_text, zoom_status
from reachy_ai.pair_bundle import read_bundle
from reachy_ai.perception import focus as focus_mod
from reachy_ai.perception import procedures as proc
from reachy_ai.perception import scene_contract
from reachy_ai.stereo import CommandBusy, StereoService, StereoSettings, _lens_result
from tests.focus_fakes import FakeLens, flat, peaked
from tests.test_data_capture import RecordingSetter, jpeg, meta, viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_capture_bundle as analyse  # noqa: E402
import reachy_audio_helper as helper  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "inventory"
INTER = {"left": {"zoom_level": "INTER", "zoom": 270, "focus": 331},
         "right": {"zoom_level": "INTER", "zoom": 270, "focus": 321}}


# --- sharpness on the board region ------------------------------------------------------------------------

def checker(blur=0, size=(240, 320), cell=16):
    ys, xs = np.indices(size)
    image = (((ys // cell + xs // cell) % 2) * 255).astype(np.uint8)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_sharpness_falls_with_blur_and_the_laplacian_is_the_monotone_measure():
    sharp, soft, softer = (focus_mod.sharpness(checker(b)) for b in (0, 2, 5))
    assert sharp["laplacian_var"] > soft["laplacian_var"] > softer["laplacian_var"]
    # Canny density is reported with it, but it is not monotone at mild blur (a soft edge can fire more of
    # Canny's pixels than a hairline one), so the sweep decides on the Laplacian: it only dies at strong blur.
    assert focus_mod.sharpness(checker(12))["canny_density"] < 0.2 * sharp["canny_density"]


def test_the_region_limits_the_measure_to_the_board():
    image = checker(0)
    image[:, 160:] = 128  # a flat half: no texture there
    textured = np.zeros(image.shape[:2], np.uint8)
    textured[:, :160] = 1
    flat_half = 1 - textured
    assert focus_mod.sharpness(image, textured)["laplacian_var"] > 10 * focus_mod.sharpness(image, flat_half)[
        "laplacian_var"]
    with pytest.raises(ValueError, match="empty"):
        focus_mod.sharpness(image, np.zeros(image.shape[:2], np.uint8))


def test_region_is_the_t3_quads_then_the_grid_then_a_centre_crop():
    gray = focus_mod.to_gray(checker(0))
    quads = {"r1c1": np.array([[20, 200], [100, 200], [100, 120], [20, 120]], float)}
    mask, source = focus_mod.board_region(gray, quads, find_grid=lambda g: None, margin_px=0)
    assert source == "t3_reference" and mask[160, 60] == 1 and mask[10, 10] == 0
    grid = [np.array([[200, 100], [280, 100], [280, 40], [200, 40]], float)]
    mask, source = focus_mod.board_region(gray, None, find_grid=lambda g: grid, margin_px=0)
    assert source == "grid" and mask[70, 240] == 1 and mask[200, 20] == 0
    mask, source = focus_mod.board_region(gray, None, find_grid=lambda g: None)
    assert source == "centre_crop" and mask[120, 160] == 1 and mask[5, 5] == 0
    assert mask.sum() == pytest.approx(0.25 * gray.size, rel=0.02)


def test_three_frames_are_averaged():
    mask = focus_mod.centre_crop_mask((240, 320))
    one = focus_mod.measure_frames([checker(0)], mask)
    three = focus_mod.measure_frames([checker(0), checker(2), checker(5)], mask)
    assert one["frames"] == 1 and three["frames"] == focus_mod.FRAMES_PER_MEASURE == 3
    assert three["laplacian_var"] < one["laplacian_var"]
    with pytest.raises(ValueError):
        focus_mod.average([])


# --- Pollen's range and the sweep -------------------------------------------------------------------------

def test_pollen_range_follows_the_zoom_and_stays_inside_the_motor():
    assert focus_mod.pollen_range(60) == (366, 467)
    assert focus_mod.pollen_range(270) == (301, 428)
    assert focus_mod.pollen_range(457) == (0, 250)  # the IN preset's focus 70 / 42 lies inside
    assert focus_mod.pollen_range(None) == (0, 500)
    for zoom, preset in ((60, 455), (270, 331), (457, 70)):
        low, high = focus_mod.pollen_range(zoom)
        assert low <= preset <= high


def sweep(lens, eye="left", **kw):
    return focus_mod.run_sweep(eye, lens.state[eye]["focus"], lens.state[eye]["zoom"],
                               set_focus=lambda v: lens.set_focus(eye, v), measure=lambda: lens.measure(eye), **kw)


def test_the_sweep_scans_coarse_then_fine_and_applies_the_peak():
    lens = FakeLens(peaked(left=348, right=322))
    result = sweep(lens)
    assert result["outcome"] == "applied" and abs(result["chosen"] - 348) <= 1
    assert lens.state["left"]["focus"] == result["chosen"]  # applied, and verified by read-back
    coarse = [p["focus"] for p in result["curve"] if p["phase"] == "coarse"]
    fine = [p["focus"] for p in result["curve"] if p["phase"] == "fine"]
    assert coarse == list(range(301, 429, 10)) and result["range"] == [301, 428]
    assert fine and all(abs(v - 351) <= 10 and (v - 351) % 2 == 0 for v in fine)  # +-10 about the best coarse value
    assert result["verify_ok"] is True and result["verified"]["laplacian_var"] > 900
    assert result["metric"] == "laplacian_var" and result["peak_at_range_edge"] is False
    # the last command is the chosen value; every command is a logged set_focus
    assert lens.calls[-2][0] == "set_focus" and lens.calls[-2][2] == result["chosen"]
    assert result["commands"] == len(result["curve"]) + 1


def test_a_noisy_peaked_curve_still_finds_the_peak():
    lens = FakeLens(peaked(left=360), noise=15.0)
    result = sweep(lens)
    assert result["outcome"] == "applied" and abs(result["chosen"] - 360) <= 4


def test_a_flat_curve_reports_no_focus_effect_and_keeps_the_current_value():
    for model, noise in ((flat(), 0.0), (flat(), 3.0)):  # the simulator; a broken motor with sensor noise
        lens = FakeLens(model, noise=noise)
        result = sweep(lens)
        assert result["outcome"] == "no_focus_effect" and result["chosen"] == 331
        assert "no focus effect" in result["note"]
        assert lens.state["left"]["focus"] == 331  # the starting value is back
        assert not any(p["phase"] == "fine" for p in result["curve"])  # no fine scan on a dead curve
        assert result["restored"]["ok"] is True


def test_a_peak_at_the_edge_of_the_range_is_flagged():
    result = sweep(FakeLens(peaked(left=301)))  # the range starts at 301 for zoom 270
    assert result["outcome"] == "applied" and result["peak_at_range_edge"] is True
    assert "edge" in result["note"]


def test_a_dry_run_sweep_sends_nothing_and_says_so():
    lens = FakeLens(peaked(), dry_run=True)
    result = sweep(lens, dry_run=True)
    assert result["outcome"] == "dry_run" and result["curve"] == [] and result["chosen"] == 331
    assert [c[0] for c in lens.calls] == ["measure"]  # one reading, no command
    assert result["planned_coarse"][0] == 301 and "dry-run" in result["note"]


def test_a_focus_command_that_does_not_take_restores_the_start_and_stops():
    lens = FakeLens(peaked(), stuck=True)
    result = sweep(lens)
    assert result["outcome"] == "command_failed" and result["chosen"] == 331
    assert len([c for c in lens.calls if c[0] == "set_focus"]) == 2  # the failed first point, the restore


def test_a_measure_failure_puts_the_starting_focus_back():
    lens = FakeLens(peaked())

    def broken():
        raise TimeoutError("no frames")
    result = focus_mod.run_sweep("left", 331, 270, set_focus=lambda v: lens.set_focus("left", v), measure=broken)
    assert result["outcome"] == "unmeasurable" and lens.state["left"]["focus"] == 331


# --- the lens commander: dry-run, live, bounded ----------------------------------------------------------------

class FakeCamera:
    def __init__(self, zoom=270, focus=331, level="INTER"):
        self.zoom, self.focus, self.level, self.commands = zoom, focus, level, []

    @property
    def zoom_level(self):
        return types.SimpleNamespace(name=self.level)

    def zoom_homing(self):
        self.commands.append("homing")
        self.__dict__.update(zoom=0, focus=0)  # the lens runs to its origin; not a zoom or focus command

    def start_autofocus(self):
        self.commands.append("start_autofocus")

    def stop_autofocus(self):
        self.commands.append("stop_autofocus")

    def __setattr__(self, name, value):
        if name in ("zoom", "focus") and hasattr(self, "commands"):
            self.commands.append((name, value))
        super().__setattr__(name, value)


def commander(**kw):
    client = types.SimpleNamespace(left_camera=FakeCamera(), right_camera=FakeCamera(focus=321))
    return client, LensCommander(client, sleep=lambda s: None, **kw)


def test_reads_always_work_and_commands_are_a_dry_run_by_default():
    client, lens = commander()
    assert lens.read()["right"] == {"zoom_level": "INTER", "zoom": 270, "focus": 321}
    record = lens.set_focus("left", 340)
    assert record["dry_run"] and record["sent"] is False and record["ok"] is False
    for call in (lens.homing("left"), lens.autofocus("right"), lens.set_zoom("left", 100)):
        assert call["sent"] is False
    assert client.left_camera.commands == [] and client.right_camera.commands == []


def test_a_live_command_is_sent_and_verified_by_read_back():
    client, lens = commander(dry_run=False)
    record = lens.set_focus("left", 340)
    assert record["sent"] and record["ok"] and record["before"]["focus"] == 331 and record["after"]["focus"] == 340
    assert client.left_camera.commands == [("focus", 340)]
    assert lens.set_zoom("right", 100)["after"]["zoom"] == 100
    assert lens.homing("left")["sent"] and client.left_camera.commands[-1] == "homing"
    lens.autofocus("right")
    assert client.right_camera.commands[-2:] == ["start_autofocus", "stop_autofocus"]  # always stopped


def test_autofocus_is_stopped_even_if_the_wait_is_interrupted():
    client, lens = commander(dry_run=False)

    def interrupted(seconds):
        raise KeyboardInterrupt

    lens._sleep = interrupted
    with pytest.raises(KeyboardInterrupt):
        lens.autofocus("left")
    assert client.left_camera.commands == ["start_autofocus", "stop_autofocus"]


@pytest.mark.parametrize("value", [-1, 501, 3.5, "far", None, True])
def test_focus_values_outside_the_motor_range_are_refused_before_any_command(value):
    client, lens = commander(dry_run=False)
    with pytest.raises(ValueError, match="focus must be an integer in 0-500"):
        lens.set_focus("left", value)
    assert client.left_camera.commands == []
    with pytest.raises(ValueError):
        lens.set_zoom("left", 601)
    with pytest.raises(ValueError):
        lens.set_focus("middle", 100)


def test_an_sdk_failure_is_reported_by_type_only():
    client, lens = commander(dry_run=False)

    class Broken(FakeCamera):
        def __setattr__(self, name, value):
            if name == "focus" and hasattr(self, "commands"):
                raise ConnectionError("10.0.0.5 refused")
            super().__setattr__(name, value)

    client.left_camera = Broken()
    record = lens.set_focus("left", 340)
    assert record["ok"] is False and record["error"] == "ConnectionError" and "10.0.0.5" not in json.dumps(record)


def test_the_sdk_process_answers_a_lens_request_and_never_raises():
    _, lens = commander(dry_run=False)
    read = _lens_result(lens, {"id": "1", "op": "read"})
    assert read["kind"] == "lens_result" and [r["eye"] for r in read["records"]] == ["left", "right"]
    done = _lens_result(lens, {"id": "2", "op": "set_focus", "eye": "left", "value": 345})
    assert done["records"][0]["ok"] and done["lens"]["left"]["focus"] == 345 and not done["dry_run"]
    bad = _lens_result(lens, {"id": "3", "op": "set_focus", "eye": "left", "value": 9999})
    assert bad["error"] == "ValueError"
    _, dry = commander()
    assert _lens_result(dry, {"id": "4", "op": "homing", "eye": "left"})["lens"] is None


def test_the_service_folds_a_live_read_back_into_the_diagnostics_and_the_calibration():
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    for eye, focus in (("left", 331), ("right", 321)):
        service._accept({"kind": "diagnostics", "section": eye,
                         "values": {"zoom_level": "INTER", "zoom": 270, "focus": focus}})
    lens = {"left": {"zoom_level": "INTER", "zoom": 270, "focus": 345, "focus_error": "x"},
            "right": {"zoom_level": "INTER", "zoom": 270, "focus": 321}}
    service._accept({"kind": "lens_result", "id": "a", "op": "set_focus", "dry_run": False, "records": [],
                     "lens": lens, "finished_ns": 5})
    assert service.diagnostics["left"]["focus"] == 345 and "focus_error" not in service.diagnostics["left"]
    assert service.command_done_ns == 5
    service._accept({"kind": "lens_result", "id": "b", "op": "set_focus", "dry_run": True, "records": [],
                     "lens": {"left": {"focus": 1}}, "finished_ns": 6})
    assert service.diagnostics["left"]["focus"] == 345  # a dry run changes nothing


def test_fresh_frames_wait_for_frames_after_the_last_command():
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    base = time.monotonic_ns()
    for n in range(1, 4):
        service._accept({"kind": "frame", "eye": "left", "sequence": n, "observed_ns": base + n * 10,
                         "pixels": np.zeros((8, 8, 3), np.uint8)})
    with pytest.raises(TimeoutError, match="1 of 3"):
        service.fresh_frames("left", 3, after_ns=base + 25, timeout_s=0.1)  # only sequence 3 is newer
    assert len(service.fresh_frames("left", 1, after_ns=base + 25, timeout_s=0.1)) == 1


# --- the Focus check procedure: confirms, dry-run, homing, restore --------------------------------------------

def focus_check():
    return proc.CaptureRun(proc.get_procedure("focus_check"))


def test_the_procedure_orders_its_steps_and_every_command_step_has_its_own_confirm_field():
    p = proc.get_procedure("focus_check")
    assert p.kind == "focus" and p.name == "Focus check"
    assert [s.focus for s in p.steps if s.focus] == ["read", "measure", "homing", "sweep", "autofocus", "finish",
                                                      "measure"]
    assert [s.id for s in p.steps if s.head] == ["head_board_view", "end_neutral"]
    assert [s.id for s in p.steps if s.capture and s.focus] == ["focus_before", "focus_after"]
    assert all(s.zoom is None for s in p.steps)  # no zoom *level* command; homing restores it through the setter


def test_read_and_measure_send_nothing_and_need_no_confirm():
    run, lens = focus_check(), FakeLens(dry_run=True)
    read = run.apply_focus("focus_read", lens, confirmed=False)
    assert [r["focus"] for r in read] == [331, 321] and run.lens_start() == INTER
    run.apply_focus("focus_before", lens, confirmed=False)
    assert {c[0] for c in lens.calls} <= {"read", "measure"}
    assert lens.regions_reset == 1  # the region is fixed per step, not per frame


@pytest.mark.parametrize("step", ["focus_homing", "focus_sweep", "focus_autofocus", "focus_finish"])
def test_a_lens_command_without_the_confirm_is_refused_and_sends_nothing(step):
    run, lens = focus_check(), FakeLens()
    run.apply_focus("focus_read", lens, False)
    lens.calls.clear()
    with pytest.raises(proc.FocusNotConfirmed):
        run.apply_focus(step, lens, confirmed=False, choice="restore")
    assert lens.calls == []
    assert run.focus_results[step][-1]["outcome"] == "refused"  # recorded, but it is not an attempt
    assert run.capture_blocker("focus_before") is not None


def test_in_dry_run_every_command_step_logs_and_changes_nothing():
    run, lens = focus_check(), FakeLens(peaked(), dry_run=True)
    run.apply_focus("focus_read", lens, False)
    homing = run.apply_focus("focus_homing", lens, True)
    sweep_records = run.apply_focus("focus_sweep", lens, True)
    run.apply_focus("focus_autofocus", lens, True)
    finish = run.apply_focus("focus_finish", lens, True, choice="restore")
    assert all(r["dry_run"] for r in homing if "dry_run" in r)
    assert [r["outcome"] for r in sweep_records] == ["dry_run", "dry_run"]
    assert lens.state == INTER and all(r["dry_run"] and not r["ok"] for r in finish)


def test_homing_needs_the_starting_state_then_restores_the_starting_zoom_level():
    run, lens = focus_check(), FakeLens(peaked())
    with pytest.raises(proc.FocusRefused, match="Read lens"):
        run.apply_focus("focus_homing", lens, True)
    assert not any(c[0] in ("homing", "set_level") for c in lens.calls)
    run.apply_focus("focus_read", lens, False)
    records = run.apply_focus("focus_homing", lens, True)
    homed = [c for c in lens.calls if c[0] == "homing"]
    assert homed == [("homing", "left"), ("homing", "right")]
    assert lens.calls[lens.calls.index(("homing", "right")) + 1] == ("set_level", "INTER")  # then the level
    restored = [r for r in records if r["action"] == "restore_level"]
    assert [r["target"] for r in restored] == ["INTER", "INTER"] and all(r["ok"] for r in restored)
    assert lens.state["left"]["zoom_level"] == "INTER"


def test_homing_refuses_eyes_that_started_at_different_levels_and_flags_a_level_it_cannot_set():
    split = FakeLens(state={"left": {**INTER["left"], "zoom_level": "IN"}, "right": INTER["right"]})
    run = focus_check()
    run.apply_focus("focus_read", split, False)
    with pytest.raises(proc.FocusRefused, match="different or unknown"):
        run.apply_focus("focus_homing", split, True)
    refusing = FakeLens(fail_level=True)
    run = focus_check()
    run.apply_focus("focus_read", refusing, False)
    records = run.apply_focus("focus_homing", refusing, True)
    assert all(r["manual"] and "by hand" in r["note"] for r in records if r["action"] == "restore_level")


def test_the_starting_state_is_the_first_complete_read_before_any_command():
    run, lens = focus_check(), FakeLens(peaked())
    incomplete = FakeLens(state={"left": {"zoom_level": "INTER", "zoom": None, "focus": 331}, "right": INTER["right"]})
    run.apply_focus("focus_read", incomplete, False)
    assert run.lens_start() is None
    run.apply_focus("focus_read", lens, False)
    assert run.lens_start() == INTER
    run.apply_focus("focus_sweep", lens, True)
    later = run.apply_focus("focus_read", lens, False)  # after a command: an extra read-back, never the start
    assert run.lens_start() == INTER and all(r.get("extra") for r in later)


def test_a_sweep_then_the_exact_restore_of_the_starting_integers():
    run, lens = focus_check(), FakeLens(peaked(left=380, right=390), state={
        "left": {"zoom_level": "INTER", "zoom": 270, "focus": 331},
        "right": {"zoom_level": "INTER", "zoom": 270, "focus": 321}})
    run.apply_focus("focus_read", lens, False)
    sweeps = run.apply_focus("focus_sweep", lens, True)
    assert [s["outcome"] for s in sweeps] == ["applied", "applied"]
    assert lens.state["left"]["focus"] != 331 and lens.state["right"]["focus"] != 321
    lens.calls.clear()
    finish = run.apply_focus("focus_finish", lens, True, choice="restore")
    assert lens.state == INTER and all(r["ok"] and r["mode"] == "raw" for r in finish)
    # zoom already equal: only focus is commanded; the exact starting integers are what is set
    assert [c for c in lens.calls if c[0].startswith("set_")] == [("set_focus", "left", 331),
                                                                  ("set_focus", "right", 321)]


def test_the_restore_commands_zoom_before_focus_when_both_differ():
    run, lens = focus_check(), FakeLens()
    run.apply_focus("focus_read", lens, False)
    for eye in lens.state:
        lens.state[eye].update(zoom=457, focus=70, zoom_level="IN")  # somebody changed the zoom by hand
    lens.calls.clear()
    run.apply_focus("focus_finish", lens, True, choice="restore")
    assert [c[:2] for c in lens.calls if c[0].startswith("set_")] == [
        ("set_zoom", "left"), ("set_focus", "left"), ("set_zoom", "right"), ("set_focus", "right")]
    assert lens.state == INTER


def test_a_restore_that_does_not_read_back_is_a_manual_step_and_an_unknown_start_sends_nothing():
    run, lens = focus_check(), FakeLens(peaked(left=380))
    run.apply_focus("focus_read", lens, False)
    run.apply_focus("focus_sweep", lens, True)
    lens.stuck = True
    records = run.apply_focus("focus_finish", lens, True, choice="restore")
    assert any(r["manual"] and "by hand" in r["note"] for r in records)
    unknown, fresh = focus_check(), FakeLens()
    records = unknown.apply_focus("focus_finish", fresh, True, choice="restore")
    assert all(r["manual"] for r in records) and not any(c[0].startswith("set_") for c in fresh.calls)


def test_finish_keep_and_sweep_choices_and_a_bad_choice():
    run, lens = focus_check(), FakeLens(peaked(left=380, right=390))
    run.apply_focus("focus_read", lens, False)
    with pytest.raises(ValueError, match="choice must be one of"):
        run.apply_focus("focus_finish", lens, True, choice="whatever")
    assert run.apply_focus("focus_finish", lens, True, choice="sweep")[0]["ok"] is False  # no sweep yet
    run.apply_focus("focus_sweep", lens, True)
    for eye in lens.state:
        lens.state[eye]["focus"] = 100
    again = run.apply_focus("focus_finish", lens, True, choice="sweep")
    assert all(r["ok"] for r in again) and abs(lens.state["left"]["focus"] - 380) <= 1
    kept = run.apply_focus("focus_finish", lens, True, choice="keep")
    assert all(r["choice"] == "keep" for r in kept)


def test_pollen_autofocus_is_measured_before_and_after_per_eye():
    run, lens = focus_check(), FakeLens(peaked(left=330, right=330))
    run.apply_focus("focus_read", lens, False)
    records = run.apply_focus("focus_autofocus", lens, True)
    assert [r["outcome"] for r in records] == ["ran", "ran"]
    assert records[0]["lens_after"]["focus"] == 330 and records[0]["sharpness_after"]["laplacian_var"] > 900
    assert [c for c in lens.calls if c[0] == "autofocus"] == [("autofocus", "left"), ("autofocus", "right")]


# --- the bundle and the analyser -------------------------------------------------------------------------------

def photo(blur):
    return proc.encode_jpeg(checker(blur))


def focus_bundle(model, blur_after=0, choice="restore", dry_run=False):
    run, lens = focus_check(), FakeLens(model, dry_run=dry_run)
    run.apply_focus("focus_read", lens, False)
    run.apply_focus("focus_before", lens, False)
    run.add_capture("focus_before", meta("before", zoom="INTER"), photo(4), photo(4))
    run.apply_focus("focus_sweep", lens, True)
    run.apply_focus("focus_autofocus", lens, True)
    run.apply_focus("focus_finish", lens, True, choice=choice)
    run.apply_focus("focus_after", lens, False)
    run.add_capture("focus_after", meta("after", zoom="INTER"), photo(blur_after), photo(blur_after))
    return run.build_zip({"viewer": "9.9", "sdk": "0.7.0"})


def test_the_bundle_holds_the_curves_the_values_and_the_pairs_before_and_after():
    manifest, files = read_bundle(focus_bundle(peaked(left=380, right=390)))
    assert manifest["procedure"]["id"] == "focus_check"
    capture = json.loads(files["capture.json"])
    assert capture["kind"] == "focus" and capture["start_lens"] == INTER
    record = json.loads(files["focus/run_record.json"])
    assert [s["eye"] for s in record["sweeps"]] == ["left", "right"] and record["start_text"] == "L331/R321"
    assert all(len(s["curve"]) > 15 and s["start_focus"] in (331, 321) for s in record["sweeps"])
    assert len(record["sharpness"]) == 4 and {r["step"] for r in record["sharpness"]} == {"focus_before",
                                                                                         "focus_after"}
    assert record["finish"][0]["choice"] == "restore" and len(record["autofocus"]) == 2
    csv_lines = files["focus/curves.csv"].decode().splitlines()
    assert csv_lines[0] == "step_id,eye,phase,focus,laplacian_var,canny_density" and len(csv_lines) > 30
    assert {"steps/03_focus_before/cap1/left.jpg", "steps/08_focus_after/cap1/right.jpg"} <= set(files)


def test_the_analyser_reports_the_peak_the_restore_and_the_pair_sharpness():
    _, files, capture = analyse.load(focus_bundle(peaked(left=380, right=390), blur_after=0))
    result = analyse.analyze_focus(capture, files)
    left = result["eyes"]["left"]
    assert result["kind"] == "focus" and result["start_lens"] == INTER
    assert left["sweep"]["outcome"] == "applied" and abs(left["sweep"]["peak_focus"] - 380) <= 1
    assert left["sweep"]["chosen"] == left["sweep"]["peak_focus"] and left["sweep"]["flat"] is False
    assert left["autofocus"]["outcome"] == "ran" and left["finish"]["choice"] == "restore" and left["finish"]["ok"]
    # blur 4 before, sharp after: the bundle's own pairs show the gain
    assert left["pair_sharpness_ratio"] > 5 and not result["findings"]
    analyse.print_focus(result)  # prints only: no exception on a full result


def test_a_flat_curve_is_a_finding_no_focus_effect_in_the_analyser():
    _, files, capture = analyse.load(focus_bundle(flat()))
    result = analyse.analyze_focus(capture, files)
    assert result["eyes"]["left"]["sweep"]["flat"] is True
    assert any(f.startswith("left: no focus effect") and "simulator" in f for f in result["findings"])
    assert result["eyes"]["left"]["sweep"]["chosen"] == 331


def test_a_dry_run_bundle_says_nothing_was_sent():
    _, files, capture = analyse.load(focus_bundle(peaked(), dry_run=True, choice="keep"))
    result = analyse.analyze_focus(capture, files)
    assert result["eyes"]["left"]["sweep"]["dry_run"] is True
    assert any("sweep was a dry run" in f for f in result["findings"])


# --- the calibration id and the depth warning -----------------------------------------------------------------

def test_focus_is_folded_into_the_calibration_id_after_the_zoom():
    assert calibration_id("abc", "INTER", {"left": 331, "right": 321}) == "abc@zoom=INTER,focus=L331/R321"
    assert calibration_id("abc", "INTER", {"left": 331, "right": 321}) != calibration_id(
        "abc", "INTER", {"left": 345, "right": 321})  # a refocused camera is not the calibrated one
    assert calibration_id("abc", "OUT", {"left": None, "right": 445}) == "abc@zoom=OUT,focus=L?/R445"
    assert calibration_id("abc", "OUT", {"left": None, "right": None}) == "abc@zoom=OUT,focus=unknown"
    assert calibration_id("abc", "OUT") == "abc@zoom=OUT"  # no focus given: the old id
    assert calibration_id(None, "OUT", {"left": 1, "right": 1}) is None
    assert focus_text({"left": 331, "right": 321}) == "L331/R321"


def robot_scene(calibration, zoom="OUT"):
    data = json.loads((ROOT / "tests/fixtures/perceived_scene/valid/robot_empty_scene.json").read_text())
    data["rig"]["calibration_id"], data["rig"]["zoom_level"] = calibration, zoom
    return data


def test_the_scene_contract_accepts_a_focus_folded_robot_id_for_the_same_zoom_only():
    for cal in ("abc123@zoom=OUT", "abc123@zoom=OUT,focus=L455/R445", "abc123@zoom=OUT,focus=unknown"):
        assert scene_contract.PerceivedScene.from_dict(robot_scene(cal)).rig.calibration_id == cal
    for cal in ("abc123@zoom=IN,focus=L455/R445", "abc123@zoom=OUT,focus=", "abc123@zoom=OUT,focus=a,b"):
        with pytest.raises(scene_contract.SceneContractError, match="does not carry the zoom_level"):
            scene_contract.PerceivedScene.from_dict(robot_scene(cal))


def test_the_depth_status_warns_unverified_at_this_lens_state():
    readbacks = {"left": {"zoom_level": "OUT", "zoom": 60, "focus": 455},
                 "right": {"zoom_level": "OUT", "zoom": 60, "focus": 445}}
    status = zoom_status("OUT", readbacks)  # the profile records no lens state (null today)
    assert status["focus"] == {"left": 455, "right": 445} and status["focus_matches_profile"] is None
    assert "unverified at this lens state (focus L455/R445)" in status["warning"]
    recorded = {"left": {"focus": 455}, "right": {"focus": 445}}
    ok = zoom_status("OUT", readbacks, recorded)
    assert ok["focus_matches_profile"] is True and "lens state" not in (ok["warning"] or "")
    moved = zoom_status("OUT", {**readbacks, "left": {**readbacks["left"], "focus": 300}}, recorded)
    assert moved["focus_matches_profile"] is False
    assert "unverified at this lens state: focus L300/R445 differs from the profile's (L455/R445)" in moved["warning"]
    unread = zoom_status("OUT", {"left": {"zoom_level": "OUT", "focus": None}, "right": readbacks["right"]})
    assert "focus could not be read" in unread["warning"]
    plain = zoom_status("OUT", {"left": "OUT", "right": "OUT"})  # levels only: no focus section at all
    assert plain["focus"] is None and "lens state" not in (plain["warning"] or "")


def test_the_august_profile_records_no_lens_state_so_the_robot_is_always_unverified():
    from reachy_ai.perception.stereo_depth import PairDepth  # noqa: PLC0415
    profile = PairDepth("robot").status()["profile"]
    assert "lens_state" in profile and profile["lens_state"] is None  # null today; the key is what a profile adds


def test_the_viewer_folds_focus_into_the_robot_id_and_leaves_the_sim_plain():
    service, http = viewer(source="robot", level="OUT")
    for eye, focus in (("left", 455), ("right", 445)):
        service._accept({"kind": "diagnostics", "section": eye,
                         "values": {"zoom_level": "OUT", "zoom": 60, "focus": focus}})
    with http as client:
        pair = client.post("/api/stereo/pairs").json()
        assert pair["camera_context"]["calibration_id"].endswith("@zoom=OUT,focus=L455/R445")
        zoom = client.get("/api/stereo/status").json()["zoom"]
        assert "unverified at this lens state" in zoom["warning"]
    sim_service, sim_http = viewer(source="sim", level="OUT")
    for eye in ("left", "right"):
        sim_service._accept({"kind": "diagnostics", "section": eye,
                             "values": {"zoom_level": "OUT", "zoom": 60, "focus": 455}})
    with sim_http as client:
        context = client.post("/api/stereo/pairs").json()["camera_context"]
        assert "@" not in context["calibration_id"]
        assert client.get("/api/stereo/status").json()["zoom"]["warning"] is None


# --- the viewer's routes -----------------------------------------------------------------------------------------

def test_the_focus_route_refuses_without_the_confirm_and_logs_a_dry_run_with_it():
    service, http = viewer()
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "focus_check"})
        base = "/api/stereo/capture-run/steps/"
        assert client.post(base + "focus_read/focus", json={}).status_code == 200  # a read: no confirm
        assert client.post(base + "focus_before/focus", json={}).status_code == 200  # a measure: no confirm
        sent = len(service.lens_requests)
        for step in ("focus_homing", "focus_sweep", "focus_autofocus"):
            assert client.post(base + f"{step}/focus", json={}).status_code == 409
        assert len(service.lens_requests) == sent  # nothing reached the SDK process
        done = client.post(base + "focus_sweep/focus", json={"confirm": True}).json()
        assert done["dry_run"] is True and [r["outcome"] for r in done["records"]] == ["dry_run", "dry_run"]
        assert all(r.get("points") == 0 for r in done["records"])  # a sweep's curve stays out of the page state
        assert client.post(base + "focus_finish/focus", json={"confirm": True, "choice": "x"}).status_code == 422
        assert client.post(base + "nope/focus", json={"confirm": True}).status_code == 404
        assert client.post(base + "end_neutral/focus", json={"confirm": True}).status_code == 422
        assert client.post(base + "focus_homing/focus", json={"confirm": True}).status_code in (200, 409)


def test_a_busy_lens_queue_is_a_409_and_a_dead_connection_a_503():
    service, http = viewer()

    def busy(*a, **k):
        raise CommandBusy("A zoom or lens command is already pending")

    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "focus_check"})
        service.request_lens = busy
        assert client.post("/api/stereo/capture-run/steps/focus_read/focus", json={}).status_code == 409
        service.request_lens = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Camera service is not connected"))
        assert client.post("/api/stereo/capture-run/steps/focus_read/focus", json={}).status_code == 503


def test_lens_requests_need_a_live_connection_and_a_known_operation():
    service = StereoService(StereoSettings(source="robot"))
    with pytest.raises(RuntimeError, match="not connected"):
        service.request_lens("read")
    with pytest.raises(ValueError, match="unknown lens operation"):
        service.request_lens("reset_everything")


# --- zoom FOV check v5 -------------------------------------------------------------------------------------------

def fov_run(lens, restore=True):
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    run.apply_zoom("zoom_start", RecordingSetter(start="INTER"), False, lens=lens)
    for step in ("zoom_out", "zoom_inter", "zoom_in"):
        run.apply_zoom(step, RecordingSetter(), True, lens=lens)
        assert "not been refocused" in run.capture_blocker(step)
        records = run.apply_focus(step, lens, True)
        assert records[0]["action"] == "sweep" and run.capture_blocker(step) is None
        run.add_capture(step, meta(step, zoom=step.split("_")[1].upper()), jpeg(), jpeg())
    if restore:
        run.apply_zoom("zoom_restore", RecordingSetter(start="INTER"), True, lens=lens)
    return run


def test_zoom_fov_v5_records_the_raw_start_refocuses_each_level_and_restores_exactly():
    assert proc.get_procedure("zoom_fov").version == "5"
    lens = FakeLens(peaked(left=332, right=322))
    run = fov_run(lens)
    assert run.lens_start() == INTER
    for eye in lens.state:  # the zoom steps moved the fake lens by sweeping its focus; the restore put it back
        assert lens.state[eye] == INTER[eye]
    restore = run.zoom_commands["zoom_restore"]
    assert [r["mode"] for r in restore] == ["raw", "raw"] and all(r["ok"] for r in restore)
    assert [r["target"] for r in restore] == [{"zoom": 270, "focus": 331}, {"zoom": 270, "focus": 321}]
    assert not any(c[0] == "set_level" for c in lens.calls)  # the level setter is not used: the integers are
    assert [len(run.focus_results[s]) for s in ("zoom_out", "zoom_inter", "zoom_in")] == [2, 2, 2]


def test_zoom_fov_v5_restore_needs_the_confirm_and_a_dry_run_changes_nothing():
    lens = FakeLens(peaked())
    run = fov_run(lens, restore=False)
    lens.calls.clear()
    with pytest.raises(proc.ZoomNotConfirmed):
        run.apply_zoom("zoom_restore", RecordingSetter(), False, lens=lens)
    assert lens.calls == []
    lens.dry_run = True
    records = run.apply_zoom("zoom_restore", RecordingSetter(), True, lens=lens)
    assert all(r["dry_run"] and not r["ok"] and "manual" not in r for r in records)


def test_zoom_fov_without_the_raw_start_falls_back_to_the_level_restore():
    run = proc.CaptureRun(proc.get_procedure("zoom_fov"))
    unreadable = FakeLens(state={"left": {"zoom_level": "ZERO", "zoom": None, "focus": None},
                                 "right": {"zoom_level": "ZERO", "zoom": None, "focus": None}})
    run.apply_zoom("zoom_start", RecordingSetter(start="ZERO"), False, lens=unreadable)
    assert run.lens_start() is None
    refused = run.apply_zoom("zoom_restore", RecordingSetter(dry_run=False, accepts=False), True, lens=unreadable)
    assert all(r["manual"] and r["note"] == proc.ZERO_RESTORE_NOTE and r["action"] == "restore" for r in refused)


def test_the_zoom_analyser_reports_the_refocus_and_the_exact_lens_restore():
    lens = FakeLens(peaked(left=332, right=322))
    _, files, capture = analyse.load(fov_run(lens).build_zip({}))
    assert capture["start_lens"] == INTER
    result = analyse.analyze_zoom(capture, files)
    assert result["restore_status"] == "lens restored exactly: left zoom 270, focus 331; right zoom 270, focus 321"
    assert len(result["refocus"]) == 6 and all(r["outcome"] == "applied" for r in result["refocus"])
    flat_lens = FakeLens(flat())
    _, files, capture = analyse.load(fov_run(flat_lens).build_zip({}))
    result = analyse.analyze_zoom(capture, files)
    assert any("no focus effect" in f for f in result["findings"])
    failed = FakeLens(peaked(left=350, right=360))
    run = fov_run(failed, restore=False)
    failed.stuck = True
    run.apply_zoom("zoom_restore", RecordingSetter(), True, lens=failed)
    _, files, capture = analyse.load(run.build_zip({}))
    # the sweeps left the fake lens at its peaks, not the starting focus, and the stuck lens ignores the restore
    status, findings = analyse.restore_findings(capture)
    assert status is None and any("Restore FAILED (left)" in f for f in findings)


# --- the camera_focus inventory probe -----------------------------------------------------------------------------

def probe_run(grep="camera_service_grep_commented.txt", nodes="ros2_node_list.txt"):
    calls = []

    def run(argv, timeout):
        calls.append((argv, timeout))
        out = grep if argv[0] == "grep" else nodes
        return {"command": " ".join(argv), "exit": 0, "timed_out": False, "truncated": False,
                "stdout": (FIXTURES / out).read_text(), "stderr": ""}

    run.calls = calls
    return run


def camera_prober(run, ros2=True, **kw):
    return helper.Prober(run=run, which=(lambda n: "/opt/ros/bin/" + n) if ros2 else (lambda n: None),
                         names=({"reachy-nuc"}, {"siva"}), release_path=str(FIXTURES / "none"), **kw)


def test_the_camera_focus_probe_runs_a_fixed_grep_and_ros2_node_list_only():
    run = probe_run()
    reply = camera_prober(run, camera_service_file="/opt/reachy/camera_zoom_service.py").probe("camera_focus")
    assert reply["outcome"] == "ok" and reply["redacted"] is True
    assert [argv for argv, _ in run.calls] == [
        ["grep", "-n", "-E", "homing|autofocus", "/opt/reachy/camera_zoom_service.py"], ["ros2", "node", "list"]]
    assert helper.CAMERA_FOCUS_GREP == "homing|autofocus"
    only_nodes = probe_run()
    camera_prober(only_nodes).probe("camera_focus")  # no file configured: no grep
    assert [argv for argv, _ in only_nodes.calls] == [["ros2", "node", "list"]]
    only_grep = probe_run()
    camera_prober(only_grep, ros2=False, camera_service_file="/opt/reachy/camera_zoom_service.py").probe(
        "camera_focus")
    assert [argv[0] for argv, _ in only_grep.calls] == ["grep"]
    nothing = camera_prober(probe_run(), ros2=False).probe("camera_focus")
    assert nothing["outcome"] == "not_run" and "no --camera-service-file" in nothing["reason"]


@pytest.mark.parametrize("path", ["camera.py", "/opt/../etc/x.py", "/opt/a b/c.py", "/opt/c.sh", "-x/y.py",
                                  "/opt/reachy/$(id).py", "/opt/c.py;ls"])
def test_the_service_file_path_is_checked_at_startup(path):
    with pytest.raises(ValueError):
        helper.checked_service_file(path)
    assert helper.checked_service_file("/opt/reachy/camera_zoom_service.py")
    assert helper.checked_service_file(None) is None


def test_grep_finding_nothing_is_an_answer_not_a_failure():
    def run(argv, timeout):
        silent = argv[0] == "grep"
        return {"command": " ".join(argv), "exit": 1 if silent else 0, "timed_out": False, "truncated": False,
                "stdout": "" if silent else (FIXTURES / "ros2_node_list.txt").read_text(), "stderr": "",
                **({"error": "grep failed (exit 1): no message"} if silent else {})}

    reply = camera_prober(run, camera_service_file="/opt/reachy/camera_zoom_service.py").probe("camera_focus")
    assert reply["outcome"] == "ok"


def analysed(grep, nodes, **kw):
    prober = camera_prober(probe_run(grep, nodes), camera_service_file="/opt/reachy/camera_zoom_service.py", **kw)
    reply = prober.probe("camera_focus")
    return analyse.camera_focus_result(reply, {"outcome": reply["outcome"], "reason": None})


def test_the_analyser_reports_commented_homing_and_autofocus_and_a_missing_focus_node():
    result = analysed("camera_service_grep_commented.txt", "ros2_node_list_no_focus.txt")
    assert result["grep"]["homing"]["commented"] and not result["grep"]["homing"]["active"]
    assert result["grep"]["homing"]["definition"] and result["grep"]["autofocus"]["mention"]
    text = " | ".join(result["findings"])
    assert "no active homing call, so it does not home the lens at boot" in text
    assert "no active autofocus call, so it does not autofocus at boot" in text
    assert "no focus node in `ros2 node list`" in text
    assert result["nodes"] == ["/camera_zoom_service", "/reachy_sdk_server"]
    assert "homing lines 0/1/1 and autofocus lines 0/1/1 (active/definition/commented); focus node NOT running" \
        in result["summary"]


def test_the_analyser_is_quiet_when_the_service_homes_and_autofocuses_and_the_node_runs():
    result = analysed("camera_service_grep_active.txt", "ros2_node_list.txt")
    assert result["findings"] == [] and len(result["grep"]["homing"]["active"]) == 1
    assert "focus node running (3 nodes)" in result["summary"]


def test_a_probe_that_did_not_run_is_unknown_never_clean():
    result = analyse.camera_focus_result({"outcome": "not_run", "reason": "helper unreachable"}, None)
    assert result["findings"] == [] and result["summary"] == "not run (helper unreachable)"
    assert analyse.camera_focus_result({}, None)["summary"] == "not run"


def test_the_viewer_redacts_node_names_that_look_like_hosts():
    run = proc.CaptureRun(proc.get_procedure("robot_inventory"))
    run.probes = {"camera_focus": {"commands": [{"stdout": "/reachy_focus\n/reachy\n"}]}}
    run.redact_hosts = ("reachy",)
    after, _ = run.redacted_nuc()
    assert after[0]["camera_focus"]["commands"][0]["stdout"] == "/reachy_focus\n/reachy\n"
