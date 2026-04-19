"""
extract_gait.py
===============
Extract the kinematic gait of a Strandbeest MuJoCo model.

For each leg, sweeps the crank through a full 2π revolution, solves the
closed-chain equality constraints at every sample, and records the foot's
position relative to its crank-axle. Produces:

  • strandbeest_gait.png      — foot paths (Jansen curves) + foot height vs crank
  • strandbeest_footfall.png  — footfall diagram (which feet touch ground when)
  • strandbeest_gait.csv      — raw numeric data per leg per sample

Usage
-----
  python3 extract_gait.py --xml file
  python3 extract_gait.py --xml file --samples 720
"""

from __future__ import annotations
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import mujoco

file = "strandbeest_4legs_2pairs.xml"
FOOT_PREFIX  = "foot_"            # foot geoms are named foot_p04_l, foot_p05_r, ...
CRANK_PREFIX = "joint_crossbar_crank_"


# ──────────────────────────────────────────────────────────────────────────────
# Introspection
# ──────────────────────────────────────────────────────────────────────────────

def build_leg_map(model: mujoco.MjModel) -> dict:
    """Map leg suffix (e.g. 'p04_l') → {foot geom id, crank qpos idx, axle body id}."""
    legs = {}
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not gname or not gname.startswith(FOOT_PREFIX + "p"):
            continue
        suf   = gname[len(FOOT_PREFIX):]
        jname = CRANK_PREFIX + suf
        jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        legs[suf] = {
            "foot_gid":   gid,
            "crank_jid":  jid,
            "crank_qidx": int(model.jnt_qposadr[jid]),
            "axle_bid":   int(model.jnt_bodyid[jid]),
        }
    return legs


# ──────────────────────────────────────────────────────────────────────────────
# Constraint projection (kinematic)
# ──────────────────────────────────────────────────────────────────────────────

def project(model: mujoco.MjModel, data: mujoco.MjData,
            n_iter: int = 120, tol: float = 1e-8) -> float:
    """
    Iterate mj_forward → mj_projectConstraint to pull qpos onto the closed-
    chain equality manifold. Returns the final max |efc_pos|.
    """
    prev = np.inf
    for _ in range(n_iter):
        mujoco.mj_forward(model, data)
        mujoco.mj_projectConstraint(model, data)
        if data.nefc == 0:
            break
        err = float(np.max(np.abs(data.efc_pos)))
        if err < tol or abs(prev - err) < 1e-14:
            break
        prev = err
    mujoco.mj_forward(model, data)
    return float(np.max(np.abs(data.efc_pos))) if data.nefc else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Per-leg sweep
# ──────────────────────────────────────────────────────────────────────────────

def sweep_leg(model: mujoco.MjModel, data: mujoco.MjData,
              leg: dict, n_samples: int = 360) -> np.ndarray:
    """
    Rotate a single leg's crank through one revolution and record
    foot position relative to the axle attachment, sample by sample.

    Returns array of shape (n_samples, 5):
        columns = [theta_delta, x_rel, y_rel, z_rel, eq_err]
    theta_delta is the offset from the keyframe phase (0 .. 2π).
    """
    qidx     = leg["crank_qidx"]
    foot_gid = leg["foot_gid"]
    axle_bid = leg["axle_bid"]

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    theta0     = float(data.qpos[qidx])
    out        = np.zeros((n_samples, 5))
    thetas     = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    last_qpos  = data.qpos.copy()

    for i, t in enumerate(thetas):
        # Warm-start from previous solution → projection converges instantly
        data.qpos[:]        = last_qpos
        data.qpos[qidx]     = theta0 + t
        data.qvel[:]        = 0.0
        err = project(model, data)

        foot_w   = data.geom_xpos[foot_gid].copy()
        axle_w   = data.xpos[axle_bid].copy()
        rel      = foot_w - axle_w
        out[i]   = (t, rel[0], rel[1], rel[2], err)
        last_qpos = data.qpos.copy()

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_jansen_curve(legs_data: dict, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: xz foot trajectories (the classic Jansen curve)
    for suf, arr in legs_data.items():
        ax1.plot(arr[:, 1], arr[:, 3], linewidth=1.8, label=suf, alpha=0.85)
    ax1.set_xlabel("x relative to crank axle [m]")
    ax1.set_ylabel("z relative to crank axle [m]")
    ax1.set_title("Foot trajectory (Jansen curve)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Right: foot height vs crank angle
    for suf, arr in legs_data.items():
        ax2.plot(np.degrees(arr[:, 0]), arr[:, 3], linewidth=1.8, label=suf, alpha=0.85)
    ax2.set_xlabel("crank angle Δθ [deg]")
    ax2.set_ylabel("foot z relative to axle [m]")
    ax2.set_title("Foot height over one crank revolution")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_footfall(legs_data: dict, keyframe_phases: dict,
                  out_path: str, ground_tol: float = 0.005) -> None:
    """
    Footfall diagram: for each leg, paint the crank-angle windows where
    the foot is within `ground_tol` of its lowest point (i.e. in contact).
    All legs are aligned to a shared master crank angle using their
    keyframe phase offsets, so the diagram shows the true gait pattern.
    """
    legs   = list(legs_data.keys())
    master = "p04_l" if "p04_l" in legs else legs[0]
    master_phase0 = keyframe_phases[master]

    fig, ax = plt.subplots(figsize=(12, 0.7 * len(legs) + 1.2))

    for row, suf in enumerate(legs):
        arr  = legs_data[suf]
        z    = arr[:, 3]
        zmin = np.min(z)
        contact = z < (zmin + ground_tol)

        # Map each leg's local crank angle into the master timebase
        phase_offset  = keyframe_phases[suf] - master_phase0
        theta_global  = (arr[:, 0] + phase_offset) % (2 * np.pi)

        order        = np.argsort(theta_global)
        theta_sorted = theta_global[order]
        contact_srt  = contact[order]

        for i in range(len(theta_sorted) - 1):
            if contact_srt[i]:
                ax.barh(row,
                        width=theta_sorted[i + 1] - theta_sorted[i],
                        left=theta_sorted[i],
                        height=0.72,
                        color="tab:green", edgecolor="none")

        # Duty cycle print
        duty = 100.0 * np.count_nonzero(contact) / contact.size
        ax.text(2 * np.pi + 0.1, row,
                f"{duty:.0f}% duty",
                va="center", fontsize=9, color="gray")

    ax.set_yticks(range(len(legs)))
    ax.set_yticklabels(legs)
    ax.set_ylim(-0.6, len(legs) - 0.4)
    ax.set_xlim(0, 2 * np.pi + 0.5)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
    ax.set_xticklabels([f"{int(np.degrees(x))}°"
                        for x in np.linspace(0, 2 * np.pi, 9)])
    ax.set_xlabel("master crank angle")
    ax.set_title(f"Footfall diagram — ground-contact band "
                 f"(tol = {ground_tol*1000:.0f} mm above z_min)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def save_csv(legs_data: dict, out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["leg", "theta_rad", "x_m", "y_m", "z_m", "eq_err"])
        for suf, arr in legs_data.items():
            for row in arr:
                w.writerow([suf,
                            f"{row[0]:.6f}",
                            f"{row[1]:.6f}",
                            f"{row[2]:.6f}",
                            f"{row[3]:.6f}",
                            f"{row[4]:.2e}"])
    print(f"  wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Gait summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def summarise(legs_data: dict, ground_tol: float = 0.005) -> None:
    print("\nGait summary (per leg, relative to crank axle)")
    print(f"  {'leg':<8} {'stride [m]':>12} {'step height [m]':>18} "
          f"{'z_min [m]':>12} {'duty [%]':>10}")
    for suf, arr in legs_data.items():
        x = arr[:, 1]; z = arr[:, 3]
        stride  = float(x.max() - x.min())
        step_h  = float(z.max() - z.min())
        zmin    = float(z.min())
        duty    = 100.0 * np.count_nonzero(z < zmin + ground_tol) / z.size
        print(f"  {suf:<8} {stride:>12.4f} {step_h:>18.4f} "
              f"{zmin:>12.4f} {duty:>10.1f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--xml", default=file,
                    help="path to MuJoCo XML (default: {file})")
    ap.add_argument("--samples", type=int, default=360,
                    help="samples per crank revolution (default: 360)")
    ap.add_argument("--outdir", default=".",
                    help="where to save PNGs and CSV (default: .)")
    ap.add_argument("--ground-tol", type=float, default=0.005,
                    help="foot-height band counted as contact, metres "
                         "(default: 0.005 → 5 mm)")
    args = ap.parse_args()

    print(f"Loading {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)
    legs  = build_leg_map(model)

    if not legs:
        raise SystemExit("No legs found — expected geoms named foot_p*.")

    print(f"  found legs: {list(legs.keys())}")
    print(f"  nq={model.nq}  neq={model.neq}  samples/rev={args.samples}")

    # Record keyframe crank phases so the footfall diagram can align legs
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    keyframe_phases = {suf: float(data.qpos[leg["crank_qidx"]])
                       for suf, leg in legs.items()}

    print("\nSweeping cranks...")
    legs_data = {}
    for suf, leg in legs.items():
        legs_data[suf] = sweep_leg(model, data, leg, n_samples=args.samples)
        max_err = float(np.max(legs_data[suf][:, 4]))
        print(f"  {suf}: done  (max constraint residual: {max_err:.2e})")

    summarise(legs_data, ground_tol=args.ground_tol)

    os.makedirs(args.outdir, exist_ok=True)
    print("\nWriting outputs...")
    plot_jansen_curve(
        legs_data,
        os.path.join(args.outdir, "strandbeest_gait.png"))
    plot_footfall(
        legs_data, keyframe_phases,
        os.path.join(args.outdir, "strandbeest_footfall.png"),
        ground_tol=args.ground_tol)
    save_csv(
        legs_data,
        os.path.join(args.outdir, "strandbeest_gait.csv"))
    print("\nDone.")


if __name__ == "__main__":
    main()