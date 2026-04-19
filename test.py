"""
Strandbeest Single-Leg MuJoCo Simulation
==========================================
Numerically solves a valid initial configuration by iterating
mj_forward + mj_projectConstraint, then runs the interactive viewer.

Usage
-----
  python3 sim.py                                  # defaults
  python3 sim.py --xml strandbeest_single_leg.xml --speed 1.5
  python3 sim.py --speed -2.0                     # reverse
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_healthy(data: mujoco.MjData) -> bool:
    """True if state is finite and accelerations/velocities are not exploding."""
    return (np.all(np.isfinite(data.qacc)) and
            np.all(np.isfinite(data.qpos)) and
            np.all(np.isfinite(data.qvel)) and
            float(np.max(np.abs(data.qacc))) < 1e6 and
            float(np.max(np.abs(data.qvel))) < 1e4)


def _eq_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Max absolute equality-constraint position residual (0 if none active)."""
    if data.nefc == 0:
        return 0.0
    return float(np.max(np.abs(data.efc_pos)))


def _actuator_name(model: mujoco.MjModel, aid: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    return "" if name is None else str(name)


def _actuator_mask_for_only_leg(model: mujoco.MjModel, only_leg: str | None) -> np.ndarray:
    """Boolean mask of actuators to drive for an optional leg selector."""
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


def _first_crank_qpos_index(model: mujoco.MjModel) -> int:
    """Return qpos index of the first actuated crank joint, fallback to 0."""
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        if jid >= 0:
            return int(model.jnt_qposadr[jid])
    return 0


def _root_qpos_dims(model: mujoco.MjModel) -> int:
    """Heuristic root qpos span at the beginning of qpos (0 for fixed root)."""
    if model.njnt == 0:
        return 0
    j0 = int(model.jnt_type[0])
    if j0 == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if j0 == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    # slide/hinge roots use one qpos slot
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Configuration solver
# ──────────────────────────────────────────────────────────────────────────────

def find_valid_config(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    crank_angle: float = 0.0,
    n_trials: int = 80,
    n_project: int = 60,
    tol: float = 1e-3,
    verbose: bool = True,
) -> float:
    """
    Multi-start constraint projection to find qpos where all equality
    constraints are satisfied.

    For each trial:
      1. Set qpos[0] = crank_angle, randomise remaining DOFs slightly.
      2. Iterate mj_forward → mj_projectConstraint to pull joints onto
         the closed-chain manifold.
      3. Keep the configuration with the lowest constraint residual.

    Returns the best achieved max |efc_pos|.
    """
    rng = np.random.default_rng(42)
    best_err = np.inf
    best_qpos = data.qpos.copy()
    crank_qidx = _first_crank_qpos_index(model)
    root_dims = _root_qpos_dims(model)

    # Prefer an author-provided keyframe if present: closed chains are often
    # infeasible near zero pose but feasible near the designed assembly pose.
    base_qpos = None
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        base_qpos = data.qpos.copy()
    else:
        mujoco.mj_resetData(model, data)
        base_qpos = data.qpos.copy()

    for trial in range(n_trials):
        # Reset to a candidate configuration
        data.qpos[:] = base_qpos
        base_root = base_qpos[:root_dims].copy() if root_dims > 0 else None
        data.qpos[crank_qidx] = crank_angle
        if trial == 0:
            # First trial: exact base pose (keyframe when available).
            pass
        else:
            # Small perturbations around base pose help projection converge.
            data.qpos[crank_qidx] = crank_angle + rng.uniform(-0.35, 0.35)
            for qidx in range(model.nq):
                if qidx < root_dims or qidx == crank_qidx:
                    continue
                data.qpos[qidx] += rng.uniform(-0.2, 0.2)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0

        # Iterative projection
        prev_err = np.inf
        for _ in range(n_project):
            mujoco.mj_forward(model, data)
            mujoco.mj_projectConstraint(model, data)
            # For floating-base models, keep the assembly pose root fixed while
            # solving closed-chain internals; this prevents the whole model from
            # drifting out of view during projection.
            if root_dims > 0 and base_root is not None:
                data.qpos[:root_dims] = base_root
            err = _eq_error(model, data)
            # Early exit if converged or stalled
            if err < tol or abs(prev_err - err) < 1e-9:
                break
            prev_err = err

        err = _eq_error(model, data)
        if err < best_err:
            best_err  = err
            best_qpos = data.qpos.copy()
            if verbose:
                print(f"    trial {trial:3d}: max_eq_err = {err:.2e}  ← new best")
        if best_err < tol:
            break

    # Restore best configuration found
    mujoco.mj_resetData(model, data)
    data.qpos[:] = best_qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return best_err


# ──────────────────────────────────────────────────────────────────────────────
# Settling pass
# ──────────────────────────────────────────────────────────────────────────────

def settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int = 1000,
    verbose: bool = True,
) -> None:
    """
    Run n_steps at zero control to damp transient velocities.
    If NaN is detected mid-settle, the last good state is restored.
    """
    data.ctrl[:] = 0.0
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    for i in range(n_steps):
        mujoco.mj_step(model, data)
        if not _is_healthy(data):
            if verbose:
                print(f"    NaN during settle at step {i} — restoring last good state")
            mujoco.mj_resetData(model, data)
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            mujoco.mj_forward(model, data)
            return
        # Save a checkpoint every 50 steps
        if i % 50 == 0:
            saved_qpos = data.qpos.copy()
            saved_qvel = data.qvel.copy()

    mujoco.mj_forward(model, data)
    if verbose:
        err = _eq_error(model, data)
        print(f"    after settle ({n_steps} steps): max_eq_err = {err:.2e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(xml_path: str, crank_speed: float, only_leg: str | None = None) -> None:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    root_dims = _root_qpos_dims(model)

    print(f"Model : {xml_path}")
    print(f"  timestep    : {dt*1e3:.1f} ms")
    print(f"  nq/nv       : {model.nq} / {model.nv}")
    print(f"  actuators   : {model.nu}")
    print(f"  eq.constrs  : {model.neq}")
    print(f"  crank speed : {crank_speed:+.2f} rad/s")

    drive_mask = _actuator_mask_for_only_leg(model, only_leg)
    if model.nu > 0:
        active = [
            _actuator_name(model, aid)
            for aid in range(model.nu)
            if drive_mask[aid]
        ]
        print(f"  drive mode  : {'all actuators' if len(active) == model.nu else f'only {active}'}")
    print()

    # Start from keyframe if provided by the model.
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

    # ── Step 1: find a valid initial configuration ───────────────────────────
    print("Finding valid initial configuration...")
    err = find_valid_config(model, data, crank_angle=0.0, verbose=True)

    if err > 1e-2:
        print(
            f"\n  Warning: initial projection residual is high (max_eq_err={err:.2e}). "
            "Proceeding to settle before final validation.\n"
        )
    else:
        print(f"\n  Configuration found: max_eq_err = {err:.2e}\n")

    # ── Step 2: settle ───────────────────────────────────────────────────────
    print("Settling (1000 steps at zero ctrl)...")
    settle(model, data, n_steps=1000, verbose=True)
    post_settle_err = _eq_error(model, data)
    final_start_err = post_settle_err
    if post_settle_err > 1e-2:
        if model.nkey > 0:
            print("  Projection path still invalid; retrying from keyframe and settling...")
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            settle(model, data, n_steps=1000, verbose=True)
            key_err = _eq_error(model, data)
            final_start_err = key_err
        else:
            final_start_err = post_settle_err

    if final_start_err > 1e-2:
        print(
            "  Warning: startup residual remains high "
            f"(initial={err:.2e}, final={final_start_err:.2e}). "
            "Continuing with runtime safeguards enabled."
        )
    print()

    # Snapshot of the settled pose for hard resets
    good_qpos = data.qpos.copy()
    good_qvel = data.qvel.copy()

    # ── Step 3: interactive viewer ───────────────────────────────────────────
    print("Viewer open — close window or Ctrl-C to quit.\n")

    # Run roughly real-time at 60 Hz with several sim steps per render frame.
    steps_per_frame = max(1, int(round(1.0 / (60.0 * dt))))

    consecutive_bad = 0
    last_good_qpos  = good_qpos.copy()
    last_good_qvel  = good_qvel.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        root_slide_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_slide_x")
        viewer.cam.azimuth = 140.0
        viewer.cam.elevation = -15.0
        viewer.cam.distance = 2.8
        crossbar_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crossbar")
        if crossbar_bid >= 0:
            viewer.cam.lookat[:] = data.xpos[crossbar_bid]
        else:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.20]

        frame = 0
        while viewer.is_running():
            t0 = time.perf_counter()

            # Drive crank at requested speed
            if model.nu > 0:
                _set_drive_ctrl(data, drive_mask, crank_speed)

            # Step forward
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            # Follow slide-root walking motion without switching to full body tracking.
            if root_slide_jid >= 0:
                qidx = int(model.jnt_qposadr[root_slide_jid])
                viewer.cam.lookat[0] = float(data.qpos[qidx])

            viewer.sync()
            frame += 1

            # ── Health check ──────────────────────────────────────────────
            healthy  = _is_healthy(data)
            eq_err   = _eq_error(model, data)
            unstable = (not healthy) or (eq_err > 0.3)
            out_of_bounds = False
            if root_dims >= 3:
                out_of_bounds = float(np.linalg.norm(data.qpos[:3])) > 10.0
            if out_of_bounds:
                unstable = True

            if unstable:
                consecutive_bad += 1
                if consecutive_bad >= 3:
                    print(f"  t={data.time:.3f}s: instability detected "
                          f"(eq_err={eq_err:.3f}, healthy={healthy}) — hard reset")
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = last_good_qpos
                    data.qvel[:] = last_good_qvel * 0.0   # zero vel on reset
                    mujoco.mj_forward(model, data)
                    consecutive_bad = 0
            else:
                consecutive_bad = 0
                # Save a rolling checkpoint every 5 frames
                if frame % 5 == 0:
                    last_good_qpos = data.qpos.copy()
                    last_good_qvel = data.qvel.copy()

            # ── Periodic console log ──────────────────────────────────────
            if frame % 200 == 0:
                if model.nu > 0:
                    active_ids = np.flatnonzero(drive_mask)
                    joint_ids = model.actuator_trnid[active_ids, 0]
                    dof_ids = model.jnt_dofadr[joint_ids]
                    crank_vels = data.qvel[dof_ids]
                    crank_vel = float(np.mean(crank_vels))
                    crank_span = float(np.max(crank_vels) - np.min(crank_vels))
                else:
                    crank_vel = 0.0
                    crank_span = 0.0
                print(f"  t={data.time:7.3f}s  "
                      f"crank_mean={crank_vel:+.2f} rad/s  "
                      f"crank_span={crank_span:.2f}  "
                      f"eq_err={eq_err:.4f}  "
                      f"nefc={data.nefc}")

            # ── Real-time pacing ──────────────────────────────────────────
            elapsed = time.perf_counter() - t0
            budget  = dt * steps_per_frame
            if elapsed < budget:
                time.sleep(budget - elapsed)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strandbeest single-leg MuJoCo simulation"
    )
    parser.add_argument(
        "--xml", default="model.xml",
        help="Path to MuJoCo XML  (default: model.xml)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.5,
        help="Crank speed in rad/s  (default: 1.5, range 0.3–5.0)",
    )
    parser.add_argument(
        "--only-leg", default="",
        help=(
            "Drive only one actuator/leg (e.g. p01_l or crank_motor_p01_l). "
            "Default drives all actuators."
        ),
    )
    args = parser.parse_args()
    run_simulation(
        xml_path=args.xml,
        crank_speed=args.speed,
        only_leg=(args.only_leg.strip() or None),
    )