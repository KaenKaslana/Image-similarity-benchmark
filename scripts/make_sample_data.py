"""Generate a synthetic paired dataset into data/reference and data/candidate.

Default (``--kind objects``): six different objects (robot, mug, table, bottle,
house, lamp), each as orthographic front / side / top views named
``<object>_<view>.png``. Every candidate carries a different kind of deviation
(identical, missing part, proportion change, thinner neck, colour change +
added part, misalignment).

Usage (from the project root)::

    python scripts/make_sample_data.py                     # six objects x three views, RGBA, 512 px
    python scripts/make_sample_data.py --mode RGB          # solid white background instead of alpha
    python scripts/make_sample_data.py --kind three-view   # one box+cylinder object, front/side/top
    python scripts/make_sample_data.py --kind shapes       # abstract shapes (identical/shifted/recoloured/reshaped)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.objects import write_object_set  # noqa: E402
from src.synthetic import write_sample_set, write_three_view_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=PROJECT_ROOT / "data" / "reference")
    parser.add_argument("--candidate", type=Path, default=PROJECT_ROOT / "data" / "candidate")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mode", choices=["RGBA", "RGB", "L"], default="RGBA")
    parser.add_argument(
        "--kind",
        choices=["objects", "three-view", "shapes"],
        default="objects",
        help="objects: six different objects x three views (default); "
        "three-view: one box+cylinder object; shapes: abstract shapes",
    )
    args = parser.parse_args()
    writers = {"objects": write_object_set, "three-view": write_three_view_set, "shapes": write_sample_set}
    names = writers[args.kind](args.reference, args.candidate, size=args.size, mode=args.mode)
    print(f"Wrote {len(names)} pairs to {args.reference} and {args.candidate}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
