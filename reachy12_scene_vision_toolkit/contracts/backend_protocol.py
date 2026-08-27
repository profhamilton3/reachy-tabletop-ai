"""Proposed backend boundary for Reachy 1.2 simulation.

This is a design contract, not a drop-in patch for the reviewed repository.
Keep transport, gRPC, ROS, rendering, and web concerns outside backend classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Sequence


class BackendState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    PAUSED = "paused"
    DEGRADED = "degraded"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    dynamics: bool = False
    contacts: bool = False
    rgb_cameras: tuple[str, ...] = ()
    depth_cameras: tuple[str, ...] = ()
    segmentation: bool = False
    scene_reload: bool = False


@dataclass(frozen=True, slots=True)
class JointCommand:
    name: str
    goal_position_rad: float | None = None
    speed_limit_rad_s: float | None = None
    torque_limit_percent: float | None = None
    compliant: bool | None = None


@dataclass(frozen=True, slots=True)
class JointSample:
    position_rad: float
    velocity_rad_s: float = 0.0
    effort: float = 0.0
    compliant: bool = False


@dataclass(frozen=True, slots=True)
class Pose:
    position_xyz_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ObjectSample:
    object_id: str
    pose: Pose
    linear_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class CameraFrame:
    camera: str
    sequence: int
    sim_step: int
    sim_time_s: float
    wall_time_ns: int
    width: int
    height: int
    encoding: str
    encoded_bytes: bytes
    scene_revision: str


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    sequence: int
    sim_step: int
    sim_time_s: float
    wall_time_ns: int
    state: BackendState
    scene_revision: str
    joints: Mapping[str, JointSample]
    objects: tuple[ObjectSample, ...] = ()
    camera_frames: Mapping[str, CameraFrame] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SceneLoadResult:
    accepted: bool
    scene_revision: str
    warnings: tuple[str, ...] = ()


class SimulatorBackend(Protocol):
    """Authoritative simulation backend.

    Concurrency contract:
    - exactly one owner advances the backend;
    - snapshots are immutable and coherent;
    - readers never call ``step``;
    - command submission is bounded and non-blocking or deadline-bounded;
    - camera rendering does not hold the state mutation lock.
    """

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def pause(self, paused: bool) -> None:
        ...

    def submit_joint_commands(self, commands: Sequence[JointCommand]) -> int:
        """Submit commands and return a monotonically increasing command sequence."""
        ...

    def load_scene(self, scene_document: Mapping[str, object]) -> SceneLoadResult:
        ...

    def reset(self, *, seed: int | None = None) -> None:
        ...

    def step(self, dt_s: float) -> SimulationSnapshot:
        """Advance state. Called only by the authoritative simulation loop."""
        ...

    def latest_snapshot(self) -> SimulationSnapshot:
        """Return the latest immutable snapshot without advancing time."""
        ...
