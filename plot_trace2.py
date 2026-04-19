"""
trace_plot.py
=============
Run the stair-climber simulation and plot foot trajectories in the
world frame, overlaid on the stair profile. Shows:

  • strandbeest_traces.png    — x–z paths of every foot as the walker moves
  • strandbeest_body_x.png    — chassis x vs time (is it making progress?)
  • strandbeest_foot_z.png    — foot z vs time for every foot

The script drives the walker exactly like sim_stairclimber.py but runs
headless for a fixed duration and logs the data.

Usage
-----
  python3 trace_plot.py
  python3 trace_plot.py --xml strandbeest_stairclimber.xml --duration 15 --speed 1.5
  python3 trace_plot.py --speed 2.5 --duration 20
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import mujoco


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

FOOT_NAMES = [
    "foot_p05_l", "foot_p05_r",     # rear pair (green/cyan)
    "foot_p06_l", "foot_p06_r",     # middle pair (blue variant)
    "foot_p07_l", "foot_p07_r",     # front pair (red — modified)
]

FOOT_COLORS = {
    "foot_p05_l": "#2ecc71",   # green
    "foot_p05_r": "#16a085",   # darker green
    "foot_p06_l": "#3498db",   # blue
    "foot_p06_r": "#2874a6",   # darker blue
    "foot_p07_l": "#e74c3c",   # red
    "foot_p07_r": "#c0392b",   # darker red
}


def _is_healthy(data: mujoco.MjData) -> bool:
    return (np.all(np.isfinite(data.qacc)) and
            np.all(np.isfinite(data.qpos)) and
            np.all(np.isfinite(data.qvel)) and
            float(np.max(np.abs(data.qacc))) < 1e7 and
            float(np.max(np.abs(data.qvel))) < 1e4)


def _eq_error(data: mujoco.MjData) -> float:
    if data.nefc == 0:
        return 0.0
    return float(np.max(np.abs(data.efc_pos)))


# ──────────────────────────────────────────────────────────────────────────────
# Startup projection (same as sim_stairclimber.py)
# ──────────────────────────────────────────────────────────────────────────────

def find_valid_config(model: mujoco.MjModel, data: mujoco.MjData,
                      n_trials: int = 80, n_project: int = 80,
                      tol: float = 1e-3) -> float:
    rng = np.random.default_rng(42)
    best_err = np.inf
    best_qpos = data.qpos.copy()

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos.copy()

    slide_idx = 0
    master_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   "joint_crossbar_crank_p05_l")
    master_qidx = int(model.jnt_qposadr[master_jid])

    for trial in range(n_trials):
        data.qpos[:] = base_qpos
        if trial > 0:
            for qidx in range(model.nq):
                if qidx == slide_idx or qidx == master_qidx:
                    continue
                data.qpos[qidx] += rng.uniform(-0.2, 0.2)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0

        prev = np.inf
        for _ in range(n_project):
            mujoco.mj_forward(model, data)
            mujoco.mj_projectConstraint(model, data)
            data.qpos[slide_idx] = 0.0
            err = _eq_error(data)
            if err < tol or abs(prev - err) < 1e-9:
                break
            prev = err

        err = _eq_error(data)
        if err < best_err:
            best_err = err
            best_qpos = data.qpos.copy()
        if best_err < tol:
            break

    mujoco.mj_resetData(model, data)
    data.qpos[:] = best_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return best_err


def settle(model: mujoco.MjModel, data: mujoco.MjData, n_steps: int = 2000):
    data.ctrl[:] = 0.0
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        if not _is_healthy(data):
            break
    mujoco.mj_forward(model, data)


# ──────────────────────────────────────────────────────────────────────────────
# Stair-profile extraction (for plot overlay)
# ──────────────────────────────────────────────────────────────────────────────

def extract_stair_profile(model: mujoco.MjModel) -> list[tuple[float, float, float, float]]:
    """
    Return a list of (x_min, x_max, z_min, z_max) boxes for all geoms
    that belong to the staircase body (including the ramp).
    Euler-tilted geoms are approximated by their axis-aligned bounding box.
    """
    stair_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "staircase")
    if stair_bid < 0:
        return []

    boxes = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != stair_bid:
            continue
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        half = model.geom_size[gid].copy()
        # Use world transform after an mj_forward call
        data_tmp = mujoco.MjData(model)
        mujoco.mj_forward(model, data_tmp)
        pos = data_tmp.geom_xpos[gid].copy()
        mat = data_tmp.geom_xmat[gid].reshape(3, 3).copy()

        # AABB corners of the tilted box
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                    corners.append(pos + mat @ local)
        corners = np.array(corners)
        boxes.append((float(corners[:, 0].min()), float(corners[:, 0].max()),
                      float(corners[:, 2].min()), float(corners[:, 2].max())))
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
# Simulation loop (headless, logs data)
# ──────────────────────────────────────────────────────────────────────────────

def run_and_log(model: mujoco.MjModel, data: mujoco.MjData,
                duration: float, speed: float,
                log_hz: int = 100) -> dict:
    """
    Run the sim for `duration` seconds driving the master crank at `speed`
    rad/s. Sample foot/body positions at `log_hz` Hz.
    Returns a dict of numpy arrays:
        t, body_x, crank_theta, foot_xz[foot_name] = (N, 2)
    """
    dt = model.opt.timestep
    steps_total    = int(round(duration / dt))
    steps_per_log  = max(1, int(round(1.0 / (log_hz * dt))))
    n_logs         = steps_total // steps_per_log + 1

    # Pre-resolve ids
    foot_gids = {}
    for n in FOOT_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if gid >= 0:
            foot_gids[n] = gid
    slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_slide_x")
    slide_qidx = int(model.jnt_qposadr[slide_jid]) if slide_jid >= 0 else 0
    master_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   "joint_crossbar_crank_p05_l")
    master_qidx = int(model.jnt_qposadr[master_jid])

    t_arr       = np.zeros(n_logs)
    body_x_arr  = np.zeros(n_logs)
    crank_arr   = np.zeros(n_logs)
    foot_xz     = {n: np.zeros((n_logs, 2)) for n in foot_gids}

    idx = 0
    last_good = data.qpos.copy()
    for step in range(steps_total):
        data.ctrl[0] = speed
        mujoco.mj_step(model, data)

        if not _is_healthy(data) or _eq_error(data) > 0.4:
            print(f"    t={data.time:.2f}s: instability — restoring last good pose")
            data.qpos[:] = last_good
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            if step % 20 == 0:
                last_good = data.qpos.copy()

        if step % steps_per_log == 0 and idx < n_logs:
            t_arr[idx]      = data.time
            body_x_arr[idx] = float(data.qpos[slide_qidx])
            crank_arr[idx]  = float(data.qpos[master_qidx])
            for n, gid in foot_gids.items():
                foot_xz[n][idx, 0] = float(data.geom_xpos[gid, 0])
                foot_xz[n][idx, 1] = float(data.geom_xpos[gid, 2])
            idx += 1

    return {
        "t":        t_arr[:idx],
        "body_x":   body_x_arr[:idx],
        "crank":    crank_arr[:idx],
        "foot_xz":  {n: v[:idx] for n, v in foot_xz.items()},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_traces(log: dict, stair_boxes: list, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    # Stair cross-section
    for (xmn, xmx, zmn, zmx) in stair_boxes:
        ax.fill_between([xmn, xmx], zmn, zmx,
                        color="#b5a187", alpha=0.6, linewidth=0.5,
                        edgecolor="#6d5a3f", zorder=1)

    # Ground line
    all_x = np.concatenate([v[:, 0] for v in log["foot_xz"].values()])
    x_lo, x_hi = float(all_x.min()) - 0.3, float(all_x.max()) + 0.3
    if stair_boxes:
        x_lo = min(x_lo, min(b[0] for b in stair_boxes) - 0.3)
        x_hi = max(x_hi, max(b[1] for b in stair_boxes) + 0.3)
    ax.axhline(0, color="#6d6d6d", linewidth=0.8, zorder=0)

    # Foot traces
    for name, xz in log["foot_xz"].items():
        ax.plot(xz[:, 0], xz[:, 1],
                color=FOOT_COLORS.get(name, "black"),
                linewidth=1.3, alpha=0.85,
                label=name.replace("foot_", ""),
                zorder=3)

    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world z [m]")
    ax.set_title("Foot trajectories (world frame) over staircase")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.02, 0.35)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_body_progress(log: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(log["t"], log["body_x"], linewidth=1.8, color="#2c3e50")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("chassis x [m]")
    ax.set_title("Chassis forward progress")
    ax.grid(True, alpha=0.3)

    # Secondary axis: crank angle modulo 2π
    ax2 = ax.twinx()
    ax2.plot(log["t"], np.degrees(log["crank"] % (2 * np.pi)),
             linewidth=0.7, color="#e67e22", alpha=0.5)
    ax2.set_ylabel("master crank angle [deg]", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_foot_heights(log: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, xz in log["foot_xz"].items():
        ax.plot(log["t"], xz[:, 1],
                color=FOOT_COLORS.get(name, "black"),
                linewidth=1.2, alpha=0.85,
                label=name.replace("foot_", ""))
    ax.set_xlabel("time [s]")
    ax.set_ylabel("foot z [m]")
    ax.set_title("Foot height over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary stats
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(log: dict, stair_boxes: list) -> None:
    print("\nSummary:")
    t = log["t"]
    if len(t) < 2:
        print("  not enough data")
        return
    dx = log["body_x"][-1] - log["body_x"][0]
    dt = t[-1] - t[0]
    avg_v = dx / dt if dt > 0 else 0.0
    print(f"  duration          : {dt:.2f} s")
    print(f"  chassis displacement: {dx:+.3f} m  ({avg_v:+.3f} m/s avg)")

    max_step_z = max((b[3] for b in stair_boxes), default=0.0)
    reached = {}
    for n, xz in log["foot_xz"].items():
        reached[n] = float(xz[:, 1].max())
    print(f"  max stair height  : {max_step_z:.3f} m")
    print("  peak foot heights :")
    for n, z in reached.items():
        marker = "✓ cleared" if z > max_step_z - 0.002 else ""
        print(f"    {n:<14s} {z:.3f} m  {marker}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--xml", default="strandbeest_climber.xml",
                    help="path to MuJoCo XML (default: strandbeest_climber.xml)")
    ap.add_argument("--duration", type=float, default=12.0,
                    help="simulation duration in seconds (default: 12)")
    ap.add_argument("--speed", type=float, default=1.5,
                    help="crank speed in rad/s (default: 1.5)")
    ap.add_argument("--log-hz", type=int, default=100,
                    help="logging rate in Hz (default: 100)")
    ap.add_argument("--outdir", default=".",
                    help="output directory (default: .)")
    args = ap.parse_args()

    print(f"Loading {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    print(f"  nq={model.nq}  nu={model.nu}  neq={model.neq}")

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print("\nFinding valid initial configuration...")
    err = find_valid_config(model, data)
    print(f"  max_eq_err after projection = {err:.2e}")

    print("Settling 2000 steps...")
    settle(model, data, n_steps=2000)
    print(f"  max_eq_err after settle    = {_eq_error(data):.2e}")

    print(f"\nRunning sim for {args.duration:.1f} s at crank speed {args.speed:+.2f} rad/s")
    print(f"  logging at {args.log_hz} Hz")
    log = run_and_log(model, data,
                      duration=args.duration,
                      speed=args.speed,
                      log_hz=args.log_hz)
    print(f"  collected {len(log['t'])} log samples")

    stair_boxes = extract_stair_profile(model)
    print(f"  found {len(stair_boxes)} stair geoms for overlay")

    print_summary(log, stair_boxes)

    os.makedirs(args.outdir, exist_ok=True)
    print("\nWriting plots...")
    plot_traces(log, stair_boxes,
                os.path.join(args.outdir, "strandbeest_traces.png"))
    plot_body_progress(log,
                       os.path.join(args.outdir, "strandbeest_body_x.png"))
    plot_foot_heights(log,
                      os.path.join(args.outdir, "strandbeest_foot_z.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()