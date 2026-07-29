#!/usr/bin/env python3
"""Live habitat-sim agent that streams RGBD frames to a running mapper.

The online-ROS analogue for simulation: where the robot stack runs
``frame_pub`` nodes feeding ``streaming_mapper`` over ROS topics, this script
plays the robot. It wanders a habitat scene (random navmesh goals connected
by geodesic shortest paths, yaw facing the direction of travel), renders RGB
+ metric depth at each step, and pushes the frames over TCP to a
``scene_graph.offline.run --source stream`` process, which maps them as they
arrive.

Runs OUTSIDE the docker container, in a habitat-sim (~0.2.5) environment.

Usage (terminal A, container)::

    python -m scene_graph.offline.run --source stream --stream-port 5555 \
        --save-path /data/out/habitat_live.pt --covisibility --viser

Usage (terminal B, host, habitat env)::

    python scripts/habitat_online_sim.py --scene /path/to/scene.glb \
        --port 5555 --fps 5 --num-goals 12

The scene needs a navmesh (`.navmesh` next to the .glb is picked up
automatically; otherwise one is recomputed with default agent settings).
Habitat test scenes work out of the box::

    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes \
        --data-path /path/to/habitat_data
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import socket
import struct
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("habitat_online_sim")

_LEN = struct.Struct("<Q")

# OpenGL cam-to-world (habitat) -> OpenCV cam-to-world: negate Y and Z columns.
_OPENGL_TO_OPENCV = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True, help="Path to the scene .glb/.ply.")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Mapper stream host.")
    p.add_argument("--port", type=int, default=5555, help="Mapper stream port.")
    p.add_argument("--fps", type=float, default=5.0, help="Frames per second to render + publish.")
    p.add_argument("--speed", type=float, default=0.5, help="Agent speed in m/s along the path.")
    p.add_argument("--max-turn-deg", type=float, default=30.0, help="Max yaw change per frame (deg).")
    p.add_argument("--num-goals", type=int, default=10, help="Random navigation goals to visit.")
    p.add_argument("--min-goal-dist", type=float, default=2.0, help="Min geodesic distance for a goal (m).")
    p.add_argument("--eye-height", type=float, default=1.5, help="Camera height above navmesh (m).")
    p.add_argument("--image-size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    p.add_argument("--hfov-deg", type=float, default=90.0, help="Horizontal field of view.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--camera-name", type=str, default="habitat")
    p.add_argument("--connect-retry-s", type=float, default=60.0, help="Keep retrying the mapper for this long.")
    return p.parse_args()


def make_simulator(scene: Path, w: int, h: int, hfov_deg: float, eye_height: float, seed: int):
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene)
    backend.random_seed = seed

    def _sensor(uuid: str, sensor_type):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [h, w]
        spec.position = [0.0, eye_height, 0.0]
        spec.hfov = hfov_deg
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _sensor("color", habitat_sim.SensorType.COLOR),
        _sensor("depth", habitat_sim.SensorType.DEPTH),
    ]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    sim.seed(seed)
    if not sim.pathfinder.is_loaded:
        LOGGER.info("No navmesh found; recomputing with default settings")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return sim


def intrinsics_dict(w: int, h: int, hfov_deg: float) -> dict:
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    # Habitat uses square pixels; vertical FOV follows from the aspect ratio.
    return {"fx": fx, "fy": fx, "cx": w / 2.0, "cy": h / 2.0, "width": w, "height": h}


def yaw_towards(direction: np.ndarray) -> float:
    """Yaw about +Y so the agent's forward (-Z) points along ``direction`` (xz)."""
    return math.atan2(-float(direction[0]), -float(direction[2]))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def plan_wander(sim, rng: np.random.Generator, num_goals: int, min_goal_dist: float, step_m: float):
    """Random-goal tour: geodesic shortest paths, polyline-interpolated at step_m."""
    import habitat_sim

    pos = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
    waypoints: list[np.ndarray] = []
    for _ in range(num_goals):
        for _attempt in range(50):
            goal = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            sp = habitat_sim.ShortestPath()
            sp.requested_start = pos
            sp.requested_end = goal
            if not sim.pathfinder.find_path(sp):
                continue
            if sp.geodesic_distance < min_goal_dist or not math.isfinite(sp.geodesic_distance):
                continue
            polyline = [np.asarray(p, dtype=np.float32) for p in sp.points]
            for a, b in zip(polyline[:-1], polyline[1:]):
                seg = b - a
                length = float(np.linalg.norm(seg))
                if length < 1e-6:
                    continue
                n_steps = max(1, int(length / step_m))
                for i in range(n_steps):
                    waypoints.append(a + seg * (i / n_steps))
            waypoints.append(polyline[-1])
            pos = polyline[-1]
            break
    return waypoints


def sensor_pose_opencv(agent) -> np.ndarray:
    """4x4 OpenCV-convention cam-to-world from the color sensor state."""
    import quaternion  # numpy-quaternion, ships with habitat-sim

    s = agent.get_state().sensor_states["color"]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(quaternion.as_rotation_matrix(s.rotation), dtype=np.float32) @ _OPENGL_TO_OPENCV
    T[:3, 3] = np.asarray(s.position, dtype=np.float32)
    return T


def connect(host: str, port: int, retry_s: float) -> socket.socket:
    deadline = time.monotonic() + retry_s
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)
            sock.settimeout(30.0)
            return sock
        except OSError as exc:
            if time.monotonic() > deadline:
                raise SystemExit(f"Could not reach mapper at {host}:{port}: {exc}")
            LOGGER.info("Mapper not up yet (%s); retrying ...", exc)
            time.sleep(2.0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    args = parse_args()
    import habitat_sim  # noqa: F401  (fail fast before opening sockets)
    import quaternion

    w, h = int(args.image_size[0]), int(args.image_size[1])
    step_m = float(args.speed) / float(args.fps)
    rng = np.random.default_rng(args.seed)

    sim = make_simulator(args.scene, w, h, args.hfov_deg, args.eye_height, args.seed)
    agent = sim.get_agent(0)
    waypoints = plan_wander(sim, rng, args.num_goals, args.min_goal_dist, step_m)
    if len(waypoints) < 2:
        raise SystemExit("Could not plan a wander on this navmesh (too small / disconnected?)")
    LOGGER.info("Planned %d poses (~%.0f s at %.1f fps)", len(waypoints), len(waypoints) / args.fps, args.fps)

    intr = intrinsics_dict(w, h, args.hfov_deg)
    sock = connect(args.host, args.port, args.connect_retry_s)
    LOGGER.info("Connected to mapper at %s:%d", args.host, args.port)

    max_turn = math.radians(args.max_turn_deg)
    yaw = yaw_towards(waypoints[1] - waypoints[0])
    period = 1.0 / float(args.fps)
    t0 = time.monotonic()
    sent = 0
    try:
        for i, pos in enumerate(waypoints):
            # Face the direction of travel, turn-rate limited.
            if i + 1 < len(waypoints):
                d = waypoints[i + 1] - pos
                if float(np.linalg.norm(d[[0, 2]])) > 1e-4:
                    target = yaw_towards(d)
                    yaw += float(np.clip(wrap_angle(target - yaw), -max_turn, max_turn))

            state = agent.get_state()
            state.position = pos
            state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
            state.sensor_states = {}  # let habitat re-derive sensors from the agent
            agent.set_state(state, infer_sensor_states=True)

            obs = sim.get_sensor_observations()
            rgb = np.ascontiguousarray(obs["color"][..., :3], dtype=np.uint8)
            depth = np.ascontiguousarray(obs["depth"], dtype=np.float32)

            stamp_ns = int((time.monotonic() - t0) * 1e9)
            frame = {
                "camera": args.camera_name,
                "rgb": rgb,
                "depth_f32": depth,
                "T_world_cam": sensor_pose_opencv(agent),
                "rgb_instrinsics": intr,
                "depth_instrinsics": intr,
                "stamp_ns": np.int64(stamp_ns),
                "frame_id": f"{args.camera_name}_{i:06d}",
                "received_time": time.time(),
            }
            payload = pickle.dumps(frame, protocol=4)
            sock.sendall(_LEN.pack(len(payload)) + payload)
            sent += 1
            if sent % 50 == 0:
                LOGGER.info("published %d/%d frames", sent, len(waypoints))

            # Real-time pacing.
            next_t = t0 + sent * period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        sock.sendall(_LEN.pack(0))
        LOGGER.info("Done: published %d frames", sent)
    finally:
        sock.close()
        sim.close()


if __name__ == "__main__":
    main()
