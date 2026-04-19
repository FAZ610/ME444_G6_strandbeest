"""
Trace plotting utility for Strandbeest MuJoCo models (single-leg or multi-leg).

Features:
- Headless simulation + CSV export.
- Matplotlib plots of all discovered leg traces (pin/knee/foot + crank circles).
- Live MuJoCo viewer mode with trajectory trails for all legs.

Usage:
  python3 plot_trace.py --xml strandbeest_all_legs.xml --speed 1.5 --seconds 10
  python3 plot_trace.py --xml strandbeest_all_legs.xml --live --seconds 20
"""

from __future__ import annotations

import argparse
from collections import deque
import colorsys
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Dict, List

import mujoco
import mujoco.viewer
import numpy as np

from test import find_valid_config


@dataclass
class LegIds:
    pivot_body: int
    pin_body: int
    knee_body: int
    foot_geom: int


@dataclass
class LegTrace:
    pivot: List[np.ndarray]
    pin: List[np.ndarray]
    knee: List[np.ndarray]
    foot: List[np.ndarray]


@dataclass
class TraceBundle:
    t: List[float]
    by_leg: Dict[str, LegTrace]


def _actuator_name(model: mujoco.MjModel, aid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    return "" if name is None else str(name)


def _actuator_mask_for_only_leg(model: mujoco.MjModel, only_leg: str | None) -> np.ndarray:
    mask = np.ones(model.nu, dtype=bool)
    if model.nu == 0 or not only_leg:
        return mask

    key = only_leg.strip()
    if not key:
        return mask

    selected = np.zeros(model.nu, dtype=bool)
    for aid in range(model.nu):
        aname = _actuator_name(model, aid)
        if not aname:
            continue
        suffix = aname[len("crank_motor_"):] if aname.startswith("crank_motor_") else aname
        if key == aname or key == suffix:
            selected[aid] = True

    if not np.any(selected):
        available = [
            _actuator_name(model, aid) for aid in range(model.nu)
            if _actuator_name(model, aid)
        ]
        raise ValueError(
            f"No actuator matched only-leg='{only_leg}'. Available actuators: {available}"
        )
    return selected


def _set_drive_ctrl(data: mujoco.MjData, mask: np.ndarray, speed: float) -> None:
    if data.ctrl.size == 0:
        return
    data.ctrl[:] = 0.0
    data.ctrl[mask] = speed


def _filter_legs(legs: Dict[str, LegIds], only_leg: str | None) -> Dict[str, LegIds]:
    if not only_leg:
        return legs
    key = only_leg.strip()
    if key in legs:
        return {key: legs[key]}

    # Allow passing actuator-style selector like crank_motor_p01_l.
    if key.startswith("crank_motor_"):
        suffix = key[len("crank_motor_"):]
        if suffix in legs:
            return {suffix: legs[suffix]}

    raise ValueError(f"No leg matched only-leg='{only_leg}'. Available legs: {list(legs.keys())}")


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"Body '{name}' not found in model")
    return bid


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"Geom '{name}' not found in model")
    return gid


def _obj_name(model: mujoco.MjModel, objtype: mujoco.mjtObj, objid: int) -> str | None:
    name = mujoco.mj_id2name(model, objtype, objid)
    return None if name is None else str(name)


def _candidate_suffixes(model: mujoco.MjModel) -> List[str]:
    suffixes = set()
    patt = re.compile(r"^bar_j(?P<sfx>$|_.*)")
    for bid in range(model.nbody):
        name = _obj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name:
            continue
        m = patt.match(name)
        if m:
            suffixes.add(m.group("sfx"))
    if not suffixes:
        suffixes.add("")

    def sfx_key(s: str) -> tuple[int, str]:
        if s == "":
            return (0, s)
        m = re.search(r"(\d+)$", s)
        if m:
            return (1, f"{int(m.group(1)):04d}")
        return (1, s)

    return sorted(suffixes, key=sfx_key)


def _first_existing_body(model: mujoco.MjModel, names: List[str]) -> int:
    for n in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid >= 0:
            return int(bid)
    raise ValueError(f"None of bodies found: {names}")


def _first_existing_geom(model: mujoco.MjModel, names: List[str]) -> int:
    for n in names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            return int(gid)
    raise ValueError(f"None of geoms found: {names}")


def _suffix_to_leg_name(sfx: str, idx: int) -> str:
    if sfx == "":
        return "leg1"
    return sfx.lstrip("_") if sfx.lstrip("_") else f"leg{idx+1}"


def discover_leg_ids(model: mujoco.MjModel) -> Dict[str, LegIds]:
    legs: Dict[str, LegIds] = {}
    suffixes = _candidate_suffixes(model)

    for i, sfx in enumerate(suffixes):
        leg_name = _suffix_to_leg_name(sfx, i)

        pivot = _first_existing_body(model, [f"crank_axle{sfx}", "crank_axle"])
        pin = _first_existing_body(model, [f"bar_j{sfx}", "bar_j"])
        knee = _first_existing_body(model, [f"bar_k{sfx}", "bar_k"])
        foot = _first_existing_geom(model, [f"foot{sfx}", f"foot_geom{sfx}", "foot", "foot_geom"])

        legs[leg_name] = LegIds(
            pivot_body=pivot,
            pin_body=pin,
            knee_body=knee,
            foot_geom=foot,
        )

    if not legs:
        raise RuntimeError("No legs discovered from model naming (expected bodies like bar_j or bar_j_*)")

    return legs


def _init_simulation(xml_path: str, settle_steps: int) -> tuple[mujoco.MjModel, mujoco.MjData, Dict[str, LegIds]]:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def _max_eq_err() -> float:
        return float(np.max(np.abs(data.efc_pos))) if data.nefc > 0 else 0.0

    # Use projection solver first: for this model it consistently lands on
    # the intended leg branch better than direct keyframe warm-up.
    err = find_valid_config(model, data, crank_angle=0.0, verbose=False)
    if err > 1e-2:
        print(
            f"Warning: initial closed-chain projection residual is high ({err:.2e}); "
            "attempting warm-up settle before failing."
        )

    data.ctrl[:] = 0.0
    for _ in range(max(0, settle_steps)):
        mujoco.mj_step(model, data)

    final_err = _max_eq_err()
    if final_err > 1e-2:
        # Fallback: try author keyframe if solver path did not converge.
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            for _ in range(max(0, settle_steps)):
                mujoco.mj_step(model, data)

            key_err = _max_eq_err()
            if key_err <= 1e-2:
                legs = discover_leg_ids(model)
                return model, data, legs

            raise RuntimeError(
                "Could not find a valid closed-chain configuration after warm-up "
                f"(solver_init={err:.2e}, solver_settle={final_err:.2e}, key_settle={key_err:.2e})."
            )

        raise RuntimeError(
            "Could not find a valid closed-chain configuration after warm-up "
            f"(solver_init={err:.2e}, solver_settle={final_err:.2e})."
        )

    legs = discover_leg_ids(model)
    return model, data, legs


def _sample_leg_points(data: mujoco.MjData, ids: LegIds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        data.xpos[ids.pivot_body].copy(),
        data.xpos[ids.pin_body].copy(),
        data.xpos[ids.knee_body].copy(),
        data.geom_xpos[ids.foot_geom].copy(),
    )


def _make_trace_bundle(legs: Dict[str, LegIds]) -> TraceBundle:
    return TraceBundle(
        t=[],
        by_leg={
            leg: LegTrace(pivot=[], pin=[], knee=[], foot=[])
            for leg in legs.keys()
        },
    )


def _append_sample(bundle: TraceBundle, t: float, leg_points: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    bundle.t.append(float(t))
    for leg, pts in leg_points.items():
        pivot, pin, knee, foot = pts
        rec = bundle.by_leg[leg]
        rec.pivot.append(pivot)
        rec.pin.append(pin)
        rec.knee.append(knee)
        rec.foot.append(foot)


def simulate_traces(
    xml_path: str,
    crank_speed: float,
    seconds: float,
    settle_steps: int,
    sample_every: int,
    only_leg: str | None = None,
) -> tuple[mujoco.MjModel, TraceBundle]:
    model, data, legs = _init_simulation(xml_path, settle_steps)
    legs = _filter_legs(legs, only_leg)
    drive_mask = _actuator_mask_for_only_leg(model, only_leg)

    n_steps = int(max(1, round(seconds / model.opt.timestep)))
    sample_every = max(1, int(sample_every))

    traces = _make_trace_bundle(legs)

    for i in range(n_steps):
        if model.nu > 0:
            _set_drive_ctrl(data, drive_mask, crank_speed)

        mujoco.mj_step(model, data)

        if i % sample_every == 0:
            leg_points = {leg: _sample_leg_points(data, ids) for leg, ids in legs.items()}
            _append_sample(traces, data.time, leg_points)

    return model, traces


def _add_line_geom(
    scn: mujoco.MjvScene,
    p0: np.ndarray,
    p1: np.ndarray,
    rgba: np.ndarray,
    width_px: float,
) -> bool:
    if scn.ngeom >= scn.maxgeom:
        return False

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        float(width_px),
        p0.astype(np.float64),
        p1.astype(np.float64),
    )
    scn.ngeom += 1
    return True


def _leg_base_colors(leg_names: List[str]) -> Dict[str, np.ndarray]:
    colors: Dict[str, np.ndarray] = {}
    n = max(1, len(leg_names))
    for i, leg in enumerate(leg_names):
        rgb = colorsys.hsv_to_rgb(float(i) / float(n), 0.85, 0.95)
        colors[leg] = np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)
    return colors


def _draw_live_trails(
    viewer: mujoco.viewer.Handle,
    trails: Dict[tuple[str, str], deque[np.ndarray]],
    leg_colors: Dict[str, np.ndarray],
    circle_specs: Dict[str, tuple[np.ndarray, float]],
    width_px: float,
) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0

    trail_keys = list(trails.keys())
    if not trail_keys:
        return

    circle_segments = 36
    circle_budget = len(circle_specs) * circle_segments
    available_for_trails = max(0, scn.maxgeom - circle_budget)
    max_seg_per_trail = max(1, (available_for_trails // len(trail_keys)) - 1)

    channel_gain = {
        "pin": 1.00,
        "knee": 0.75,
        "foot": 0.55,
    }

    for leg, ch in trail_keys:
        pts = list(trails[(leg, ch)])
        if len(pts) < 2:
            continue
        pts = pts[-(max_seg_per_trail + 1):]

        base = leg_colors[leg].copy()
        base[:3] *= channel_gain[ch]
        base[3] = 1.0

        for i in range(1, len(pts)):
            if not _add_line_geom(scn, pts[i - 1], pts[i], base, width_px):
                return

    # Draw ideal crank circles in the XZ plane around each crank pivot.
    for leg, spec in circle_specs.items():
        center, radius = spec
        if radius <= 0.0:
            continue

        col = leg_colors[leg].copy()
        col[3] = 0.85

        angles = np.linspace(0.0, 2.0 * np.pi, circle_segments + 1)
        x = center[0] + radius * np.cos(angles)
        z = center[2] + radius * np.sin(angles)
        y = np.full_like(x, center[1])

        for i in range(1, len(angles)):
            p0 = np.array([x[i - 1], y[i - 1], z[i - 1]], dtype=np.float64)
            p1 = np.array([x[i], y[i], z[i]], dtype=np.float64)
            if not _add_line_geom(scn, p0, p1, col, 1.6):
                return


def run_live_traces(
    xml_path: str,
    crank_speed: float,
    seconds: float,
    settle_steps: int,
    sample_every: int,
    trail_length: int,
    viewer_hz: float,
    only_leg: str | None = None,
) -> tuple[mujoco.MjModel, TraceBundle]:
    model, data, legs = _init_simulation(xml_path, settle_steps)
    legs = _filter_legs(legs, only_leg)
    drive_mask = _actuator_mask_for_only_leg(model, only_leg)
    traces = _make_trace_bundle(legs)

    sample_every = max(1, int(sample_every))
    trail_length = max(2, int(trail_length))
    viewer_hz = max(1.0, float(viewer_hz))

    trails: Dict[tuple[str, str], deque[np.ndarray]] = {}
    for leg in legs.keys():
        trails[(leg, "pin")] = deque(maxlen=trail_length)
        trails[(leg, "knee")] = deque(maxlen=trail_length)
        trails[(leg, "foot")] = deque(maxlen=trail_length)

    circle_specs: Dict[str, tuple[np.ndarray, float]] = {}

    leg_colors = _leg_base_colors(list(legs.keys()))

    sim_steps_per_frame = max(1, int(round((1.0 / viewer_hz) / model.opt.timestep)))
    step_idx = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -15.0
        viewer.cam.distance = 3.4
        viewer.cam.lookat[:] = [0.0, 0.0, 0.25]

        while viewer.is_running() and data.time < seconds:
            frame_start = time.perf_counter()

            for _ in range(sim_steps_per_frame):
                if model.nu > 0:
                    _set_drive_ctrl(data, drive_mask, crank_speed)

                mujoco.mj_step(model, data)
                step_idx += 1

                if step_idx % sample_every == 0:
                    leg_points = {leg: _sample_leg_points(data, ids) for leg, ids in legs.items()}
                    _append_sample(traces, data.time, leg_points)

                    for leg, pts in leg_points.items():
                        pivot, pin, knee, foot = pts
                        trails[(leg, "pin")].append(pin)
                        trails[(leg, "knee")].append(knee)
                        trails[(leg, "foot")].append(foot)
                        radius_xz = float(np.linalg.norm((pin - pivot)[[0, 2]]))
                        circle_specs[leg] = (pivot.copy(), radius_xz)

            with viewer.lock():
                _draw_live_trails(
                    viewer,
                    trails=trails,
                    leg_colors=leg_colors,
                    circle_specs=circle_specs,
                    width_px=2.2,
                )

            viewer.sync()

            elapsed = time.perf_counter() - frame_start
            frame_budget = sim_steps_per_frame * model.opt.timestep
            if elapsed < frame_budget:
                time.sleep(frame_budget - elapsed)

    return model, traces


def save_csv(path: Path, traces: TraceBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    legs = list(traces.by_leg.keys())
    header = ["t"]
    for leg in legs:
        for ch in ("pivot", "pin", "knee", "foot"):
            header += [f"{leg}_{ch}_x", f"{leg}_{ch}_y", f"{leg}_{ch}_z"]

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i in range(len(traces.t)):
            row: List[float] = [traces.t[i]]
            for leg in legs:
                rec = traces.by_leg[leg]
                row += rec.pivot[i].tolist()
                row += rec.pin[i].tolist()
                row += rec.knee[i].tolist()
                row += rec.foot[i].tolist()
            w.writerow(row)


def save_plot(path: Path, traces: TraceBundle, title: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if len(traces.t) < 2 or not traces.by_leg:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)

    legs = list(traces.by_leg.keys())
    n = len(legs)
    cmap = plt.get_cmap("tab10")
    colors = {leg: cmap(i % 10) for i, leg in enumerate(legs)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=140)
    ax_pin, ax_knee, ax_foot = axes

    for leg in legs:
        rec = traces.by_leg[leg]
        pivot = np.array(rec.pivot)
        pin = np.array(rec.pin)
        knee = np.array(rec.knee)
        foot = np.array(rec.foot)

        col = colors[leg]
        ax_pin.plot(pin[:, 0], pin[:, 2], linewidth=1.6, color=col, label=f"{leg} pin")
        ax_knee.plot(knee[:, 0], knee[:, 2], linewidth=1.6, color=col, label=f"{leg} knee")
        ax_foot.plot(foot[:, 0], foot[:, 2], linewidth=1.9, color=col, label=f"{leg} foot")

        center = pivot.mean(axis=0)
        radii = np.linalg.norm(pin - pivot, axis=1)
        r = float(np.mean(radii))
        th = np.linspace(0.0, 2.0 * np.pi, 240)
        cx = center[0] + r * np.cos(th)
        cz = center[2] + r * np.sin(th)
        ax_pin.plot(cx, cz, linestyle="--", linewidth=1.1, color=col, alpha=0.75)

    ax_pin.set_title("Crank Pin Traces + Ideal Circles")
    ax_knee.set_title("Knee Traces")
    ax_foot.set_title("Foot Traces")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.grid(True, alpha=0.25)

    ax_foot.legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Strandbeest trace curves from MuJoCo simulation")
    p.add_argument("--xml", default="strandbeest_12legs.xml", help="Path to MuJoCo XML")
    p.add_argument("--speed", type=float, default=1.5, help="Crank speed [rad/s]")
    p.add_argument("--seconds", type=float, default=10.0, help="Simulation horizon [s]")
    p.add_argument("--settle-steps", type=int, default=1000, help="Zero-control warm-up steps")
    p.add_argument("--sample-every", type=int, default=2, help="Record every N simulation steps")
    p.add_argument("--live", action="store_true", help="Run interactive viewer with live trajectory trails")
    p.add_argument(
        "--only-leg", default="",
        help=(
            "Drive/trace only one leg (e.g. p01_l or crank_motor_p01_l). "
            "Default uses all legs."
        ),
    )
    p.add_argument("--trail-length", type=int, default=1200, help="Max points kept in each live trail")
    p.add_argument("--viewer-hz", type=float, default=60.0, help="Target viewer update rate in live mode")
    p.add_argument("--out", default="trace_plot.png", help="Output PNG path")
    p.add_argument("--csv", default="trace_data.csv", help="Output CSV path")
    args = p.parse_args()

    if args.live:
        model, traces = run_live_traces(
            xml_path=args.xml,
            crank_speed=args.speed,
            seconds=args.seconds,
            settle_steps=args.settle_steps,
            sample_every=args.sample_every,
            trail_length=args.trail_length,
            viewer_hz=args.viewer_hz,
            only_leg=(args.only_leg.strip() or None),
        )
    else:
        model, traces = simulate_traces(
            xml_path=args.xml,
            crank_speed=args.speed,
            seconds=args.seconds,
            settle_steps=args.settle_steps,
            sample_every=args.sample_every,
            only_leg=(args.only_leg.strip() or None),
        )

    csv_path = Path(args.csv)
    save_csv(csv_path, traces)

    png_path = Path(args.out)
    ok_plot = save_plot(
        png_path,
        traces,
        title=(
            f"Strandbeest Traces ({len(traces.by_leg)} legs) | speed={args.speed:+.2f} rad/s, "
            f"dt={model.opt.timestep:.4f} s"
        ),
    )

    print(f"Saved trace CSV: {csv_path}")
    if ok_plot:
        print(f"Saved trace plot: {png_path}")
    else:
        print("PNG not generated (matplotlib unavailable or too few samples collected).")


if __name__ == "__main__":
    main()
