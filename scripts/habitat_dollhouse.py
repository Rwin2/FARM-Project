#!/usr/bin/env python3
"""Render dollhouse-style overview images of a habitat scene.

Places the camera above the scene bounds (one straight top-down view plus
four oblique orbit angles) and renders the textured mesh, producing the
"dollhouse" overviews used to pick language-query targets for a mapped
scene. Runs on the host in the habitat-sim env (not in docker).

Usage::

    python scripts/habitat_dollhouse.py --scene /path/to/scene.glb --out-dir /path/to/out
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True, help="Path to the scene .glb/.ply.")
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for the PNGs.")
    p.add_argument("--size", type=int, default=1280, help="Square image size (px).")
    p.add_argument("--hfov-deg", type=float, default=70.0)
    p.add_argument("--oblique-pitch-deg", type=float, default=-50.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import habitat_sim
    import quaternion
    from PIL import Image

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(args.scene)

    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "color"
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.resolution = [args.size, args.size]
    spec.position = [0.0, 0.0, 0.0]
    spec.hfov = args.hfov_deg

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    agent = sim.get_agent(0)

    lower, upper = sim.pathfinder.get_bounds()
    center = (np.asarray(lower) + np.asarray(upper)) / 2.0
    extent = float(np.linalg.norm(np.asarray(upper) - np.asarray(lower)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    views = [("topdown", 0.0, -math.pi / 2)] + [
        (f"oblique_{i}", math.radians(a), math.radians(args.oblique_pitch_deg))
        for i, a in enumerate((45, 135, 225, 315))
    ]
    for name, yaw, pitch in views:
        dist = extent * (0.7 if "oblique" in name else 0.9)
        q = quaternion.from_rotation_vector([0.0, yaw, 0.0]) * quaternion.from_rotation_vector([pitch, 0.0, 0.0])
        fwd = quaternion.as_rotation_matrix(q) @ np.array([0.0, 0.0, -1.0])
        state = agent.get_state()
        state.position = center - fwd * dist
        state.rotation = q
        state.sensor_states = {}
        agent.set_state(state, infer_sensor_states=True)
        rgb = sim.get_sensor_observations()["color"][..., :3]
        out = args.out_dir / f"{name}.png"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(out)
        print(f"saved {out}")
    sim.close()


if __name__ == "__main__":
    main()
