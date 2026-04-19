"""
Walk runner for the 6-legs / 3-pairs Strandbeest model.

Usage:
  python3 walk_6legs_3pairs.py
  python3 walk_6legs_3pairs.py --speed 1.8
  python3 walk_6legs_3pairs.py --only-leg p04_l
"""

from __future__ import annotations

import argparse

from test import run_simulation

file = "strandbeest_12legs.xml"
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run walking simulation for " + file
    )
    parser.add_argument(
        "--xml",
        default=file,
        help="Path to MuJoCo XML (default: {file} with free root joint)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=2.5,
        help="Crank speed in rad/s (default: 1.5)",
    )
    parser.add_argument(
        "--only-leg",
        default="",
        help=(
            "Drive only one actuator/leg (e.g. p04_l or crank_motor_p04_l). "
            "Default drives all actuators."
        ),
    )
    args = parser.parse_args()

    run_simulation(
        xml_path=args.xml,
        crank_speed=args.speed,
        only_leg=(args.only_leg.strip() or None),
    )


if __name__ == "__main__":
    main()
