"""
Strandbeest Stair-Ascending Simulation
=======================================
Drives the 3-pair (6-leg) stair-ascending Strandbeest in MuJoCo.

Pair 05 (y=0.04, rear)   — regular Jansen legs   (green)
Pair 06 (y=0.16, middle) — regular Jansen legs   (blue)
Pair 07 (y=0.28, front)  — MODIFIED "high-step"  (red)   ← stair-climber

A single motor drives the master crank on p05_l. All other cranks are
rigidly coupled to the master via joint-equality constraints, so the
entire 6-leg assembly behaves as if driven by one physical shaft.

Usage
-----
  python3 sim_stairclimber.py
  python3 sim_stairclimber.py --speed 2.0
  python3 sim_stairclimber.py --xml strandbeest_stairclimber.xml --speed 1.2
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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
# Startup: constraint projection
# ──────────────────────────────────────────────────────────────────────────────

def find_valid_config(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_trials: int = 80,
    n_project: int = 80,
    tol: float = 1e-3,
    verbose: bool = True,
) -> float:
    """
    Multi-start projection to resolve closed-chain equality constraints.
    The modified front-leg triangle means the keyframe is slightly off the
    closed-chain manifold; projection pulls the joint angles onto it.
    """
    rng = np.random.default_rng(42)
    best_err = np.inf
    best_qpos = data.qpos.copy()

    # Start from keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos.copy()

    slide_idx = 0  # root_slide_x is the first qpos
    crank_master_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                         "joint_crossbar_crank_p05_l")
    master_qidx = int(model.jnt_qposadr[crank_master_jid])

    for trial in range(n_trials):
        data.qpos[:] = base_qpos
        if trial > 0:
            # Perturb non-root, non-master joints; leave slide and master alone
            for qidx in range(model.nq):
                if qidx == slide_idx or qidx == master_qidx:
                    continue
                data.qpos[qidx] += rng.uniform(-0.2, 0.2)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0

        prev_err = np.inf
        for _ in range(n_project):
            mujoco.mj_forward(model, data)
            mujoco.mj_projectConstraint(model, data)
            # keep slide root fixed at 0
            data.qpos[slide_idx] = 0.0
            err = _eq_error(data)
            if err < tol or abs(prev_err - err) < 1e-9:
                break
            prev_err = err

        err = _eq_error(data)
        if err < best_err:
            best_err = err
            best_qpos = data.qpos.copy()
            if verbose:
                print(f"    trial {trial:3d}: max_eq_err = {err:.2e}  ← new best")
        if best_err < tol:
            break

    mujoco.mj_resetData(model, data)
    data.qpos[:] = best_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return best_err


def settle(model: mujoco.MjModel, data: mujoco.MjData,
           n_steps: int = 2000, verbose: bool = True) -> None:
    data.ctrl[:] = 0.0
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        if not _is_healthy(data):
            if verbose:
                print(f"    NaN during settle at step {i} — restoring last good")
            mujoco.mj_resetData(model, data)
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            mujoco.mj_forward(model, data)
            return
        if i % 50 == 0:
            saved_qpos = data.qpos.copy()
            saved_qvel = data.qvel.copy()
    mujoco.mj_forward(model, data)
    if verbose:
        print(f"    after settle ({n_steps} steps): max_eq_err = {_eq_error(data):.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(xml_path: str, crank_speed: float) -> None:
    xml_file = Path(xml_path)
    if not xml_file.exists():
        available = ", ".join(sorted(p.name for p in Path(".").glob("*.xml")))
        raise FileNotFoundError(
            f"XML file not found: {xml_path}. Available XML files: {available}"
        )

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    print(f"Model : {xml_path}")
    print(f"  timestep    : {dt*1e3:.1f} ms")
    print(f"  nq / nv     : {model.nq} / {model.nv}")
    print(f"  actuators   : {model.nu}")
    print(f"  eq constrs  : {model.neq}")
    print(f"  crank speed : {crank_speed:+.2f} rad/s")
    print()

    # Start from keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    # ── Step 1: find valid configuration ─────────────────────────────────────
    print("Finding valid initial configuration...")
    err = find_valid_config(model, data)
    if err > 1e-2:
        print(f"\n  Warning: projection residual is high (max_eq_err={err:.2e}). "
              "Proceeding to settle.\n")
    else:
        print(f"\n  Configuration found: max_eq_err = {err:.2e}\n")

    # ── Step 2: settle ───────────────────────────────────────────────────────
    print("Settling (2000 steps at zero ctrl)...")
    settle(model, data, n_steps=2000, verbose=True)
    print()

    good_qpos = data.qpos.copy()
    good_qvel = data.qvel.copy()

    # ── Step 3: interactive viewer ───────────────────────────────────────────
    print("Viewer open — close window or Ctrl-C to quit.\n")
    steps_per_frame = max(1, int(round(1.0 / (60.0 * dt))))

    consecutive_bad = 0
    last_good_qpos  = good_qpos.copy()
    last_good_qvel  = good_qvel.copy()

    root_slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_slide_x")
    slide_qidx     = int(model.jnt_qposadr[root_slide_jid]) if root_slide_jid >= 0 else -1
    crossbar_bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crossbar")

    # Track feet to log "first foot on a step" event
    foot_gids = []
    for name in ("foot_p05_l", "foot_p05_r", "foot_p06_l", "foot_p06_r",
                 "foot_p07_l", "foot_p07_r"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            foot_gids.append((name, gid))
    stair_base_x = 2.0  # matches the XML

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera — side-on, looking at walker moving toward stairs
        viewer.cam.azimuth   = 90.0
        viewer.cam.elevation = -12.0
        viewer.cam.distance  = 4.5
        if crossbar_bid >= 0:
            viewer.cam.lookat[:] = data.xpos[crossbar_bid]
        else:
            viewer.cam.lookat[:] = [0.0, 0.14, 0.4]

        frame = 0
        stairs_first_touched = False
        while viewer.is_running():
            t0 = time.perf_counter()

            # Drive master crank
            if model.nu > 0:
                data.ctrl[0] = crank_speed

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # Camera follows walker in x
            if slide_qidx >= 0:
                viewer.cam.lookat[0] = float(data.qpos[slide_qidx])

            viewer.sync()
            frame += 1

            # ── Health check ──
            healthy  = _is_healthy(data)
            eq_err   = _eq_error(data)
            unstable = (not healthy) or (eq_err > 0.4)

            if unstable:
                consecutive_bad += 1
                if consecutive_bad >= 3:
                    print(f"  t={data.time:.3f}s: instability "
                          f"(eq_err={eq_err:.3f}, healthy={healthy}) — hard reset")
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = last_good_qpos
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    consecutive_bad = 0
            else:
                consecutive_bad = 0
                if frame % 5 == 0:
                    last_good_qpos = data.qpos.copy()
                    last_good_qvel = data.qvel.copy()

            # Detect first stair contact
            if not stairs_first_touched and slide_qidx >= 0:
                body_x = float(data.qpos[slide_qidx])
                for name, gid in foot_gids:
                    foot_world_x = float(data.geom_xpos[gid, 0])
                    foot_z       = float(data.geom_xpos[gid, 2])
                    if foot_world_x > stair_base_x - 0.2 and foot_z > 0.02:
                        print(f"  t={data.time:.3f}s: {name} reached stairs "
                              f"(foot_x={foot_world_x:.2f}, foot_z={foot_z:.3f})")
                        stairs_first_touched = True
                        break

            # Periodic log
            if frame % 200 == 0:
                if model.nu > 0:
                    aid = 0
                    jid = int(model.actuator_trnid[aid, 0])
                    dof = int(model.jnt_dofadr[jid])
                    crank_vel = float(data.qvel[dof])
                else:
                    crank_vel = 0.0
                body_x = float(data.qpos[slide_qidx]) if slide_qidx >= 0 else 0.0
                print(f"  t={data.time:7.3f}s  "
                      f"x={body_x:+.3f} m  "
                      f"crank_rate={crank_vel:+.2f} rad/s  "
                      f"eq_err={eq_err:.4f}")

            # Real-time pacing
            elapsed = time.perf_counter() - t0
            budget  = dt * steps_per_frame
            if elapsed < budget:
                time.sleep(budget - elapsed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Strandbeest Stair-Ascending MuJoCo simulation")
    ap.add_argument("--xml", default="strandbeest_climber.xml",
                    help="path to MuJoCo XML "
                         "(default: strandbeest_climber.xml)")
    ap.add_argument("--speed", type=float, default=1.5,
                    help="crank speed in rad/s (default: 1.5)")
    args = ap.parse_args()
    run_simulation(xml_path=args.xml, crank_speed=args.speed)