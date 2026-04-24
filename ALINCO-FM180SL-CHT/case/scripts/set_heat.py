"""
set_heat.py  –  Set LED heat source in W for 3-D half-model.

Usage:
    python3 scripts/set_heat.py --watts 2.5

Updates the fvOptions heat value (half-model = watts/2).
Must be run AFTER splitMeshRegions and the aluminum LED topoSet.
"""

import argparse
import os
import re

CASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FVOPTS    = os.path.join(CASE, "system", "aluminum", "fvOptions")


def main():
    parser = argparse.ArgumentParser(description="Set LED heat source in watts.")
    parser.add_argument("--watts", type=float, required=True, help="Total heat input [W]")
    args = parser.parse_args()

    Q_total = args.watts
    Q_half  = Q_total / 2.0

    with open(FVOPTS, "r") as f:
        content = f.read()

    # Replace the value inside:  h   (VALUE 0);
    pattern = r"(h\s+\()[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(\s+0\s*\);)"
    if not re.search(pattern, content):
        raise RuntimeError(f"Could not find 'h (... 0);' in {FVOPTS}")

    new_content = re.sub(pattern, rf"\g<1>{Q_half:.4f}\2", content)

    with open(FVOPTS, "w") as f:
        f.write(new_content)

    print(f"  Total heat  = {Q_total} W")
    print(f"  Half-model  = {Q_half} W")
    print(f"  Updated {FVOPTS}")


if __name__ == "__main__":
    main()
