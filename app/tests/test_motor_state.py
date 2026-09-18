"""Tests for motor position tracking in app/capture/camera_control.py.

The motor state file persists steps taken in the current capture cycle so an
interrupted run can automatically resume to the origin (0 degrees) on the
next capture. These tests cover the read/write helpers and the
return-to-origin logic in isolation from real hardware.
"""
import pytest

from app.settings import config as cfg
from app.capture import camera_control as cc


@pytest.fixture
def temp_state_file(tmp_path, monkeypatch):
    """Redirect MOTOR_STATE_FILE to tmp_path so tests don't touch real state."""
    state_path = tmp_path / ".motor_state"
    monkeypatch.setattr(cfg, "MOTOR_STATE_FILE", str(state_path))
    return state_path


# ---------------------------------------------------------------------------
# _read_motor_steps
# ---------------------------------------------------------------------------

class TestReadMotorSteps:
    def test_missing_file_returns_zero(self, temp_state_file):
        assert cc._read_motor_steps() == 0

    def test_empty_file_returns_zero(self, temp_state_file):
        temp_state_file.write_text("")
        assert cc._read_motor_steps() == 0

    def test_whitespace_only_returns_zero(self, temp_state_file):
        temp_state_file.write_text("   \n")
        assert cc._read_motor_steps() == 0

    def test_invalid_content_returns_zero(self, temp_state_file):
        temp_state_file.write_text("not-a-number")
        assert cc._read_motor_steps() == 0

    def test_valid_value_read(self, temp_state_file):
        temp_state_file.write_text("3")
        assert cc._read_motor_steps() == 3

    def test_negative_value_clamped_to_zero(self, temp_state_file):
        temp_state_file.write_text("-5")
        assert cc._read_motor_steps() == 0

    def test_value_above_cycle_clamped(self, temp_state_file):
        temp_state_file.write_text(str(cfg.MOTOR_STEPS_PER_CYCLE + 10))
        assert cc._read_motor_steps() == cfg.MOTOR_STEPS_PER_CYCLE


# ---------------------------------------------------------------------------
# _write_motor_steps
# ---------------------------------------------------------------------------

class TestWriteMotorSteps:
    def test_round_trip(self, temp_state_file):
        cc._write_motor_steps(4)
        assert cc._read_motor_steps() == 4

    def test_overwrites_previous_value(self, temp_state_file):
        cc._write_motor_steps(3)
        cc._write_motor_steps(7)
        assert cc._read_motor_steps() == 7

    def test_creates_missing_parent_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "sub" / "dir" / ".motor_state"
        monkeypatch.setattr(cfg, "MOTOR_STATE_FILE", str(nested))
        cc._write_motor_steps(2)
        assert nested.read_text() == "2"


# ---------------------------------------------------------------------------
# _return_motor_to_origin
# ---------------------------------------------------------------------------

class TestReturnMotorToOrigin:
    def test_no_rotation_when_state_zero(self, temp_state_file, monkeypatch):
        calls = []
        monkeypatch.setattr(cc, "_rotate_motor",
                            lambda ip, port, deg: calls.append((ip, port, deg)))
        assert cc._return_motor_to_origin("1.2.3.4", 18812) is True
        assert calls == []

    def test_no_rotation_when_file_missing(self, temp_state_file, monkeypatch):
        # temp_state_file fixture points to a path but doesn't create the file
        calls = []
        monkeypatch.setattr(cc, "_rotate_motor",
                            lambda ip, port, deg: calls.append((ip, port, deg)))
        cc._return_motor_to_origin("1.2.3.4", 18812)
        assert calls == []

    def test_rotates_remaining_steps_after_interruption(self, temp_state_file, monkeypatch):
        temp_state_file.write_text("3")
        calls = []
        monkeypatch.setattr(cc, "_rotate_motor",
                            lambda ip, port, deg: calls.append((ip, port, deg)))
        assert cc._return_motor_to_origin("1.2.3.4", 18812) is True
        remaining = cfg.MOTOR_STEPS_PER_CYCLE - 3
        expected_deg = remaining * cfg.MOTOR_DEGREES_PER_STEP
        assert calls == [("1.2.3.4", 18812, expected_deg)]

    def test_resets_state_to_zero_after_successful_return(self, temp_state_file, monkeypatch):
        temp_state_file.write_text("5")
        monkeypatch.setattr(cc, "_rotate_motor", lambda *a, **kw: None)
        cc._return_motor_to_origin("1.2.3.4", 18812)
        assert cc._read_motor_steps() == 0

    def test_returns_false_on_rotate_failure(self, temp_state_file, monkeypatch):
        temp_state_file.write_text("4")

        def boom(*a, **kw):
            raise RuntimeError("motor unreachable")
        monkeypatch.setattr(cc, "_rotate_motor", boom)
        assert cc._return_motor_to_origin("1.2.3.4", 18812) is False

    def test_preserves_state_on_rotate_failure(self, temp_state_file, monkeypatch):
        """If rotation fails, keep the recorded step count so the next attempt
        can still recover."""
        temp_state_file.write_text("4")

        def boom(*a, **kw):
            raise RuntimeError("motor unreachable")
        monkeypatch.setattr(cc, "_rotate_motor", boom)
        cc._return_motor_to_origin("1.2.3.4", 18812)
        assert cc._read_motor_steps() == 4

    @pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 6, 7])
    def test_all_intermediate_step_counts(self, temp_state_file, monkeypatch, steps):
        temp_state_file.write_text(str(steps))
        calls = []
        monkeypatch.setattr(cc, "_rotate_motor",
                            lambda ip, port, deg: calls.append(deg))
        cc._return_motor_to_origin("1.2.3.4", 18812)
        assert calls == [(cfg.MOTOR_STEPS_PER_CYCLE - steps) * cfg.MOTOR_DEGREES_PER_STEP]
