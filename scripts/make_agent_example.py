"""Build the example tasks of the agent benchmark (mug, table, chair).

Writes, for each object
  benchmarks/example/answers/<name>.glb       reference model used for scoring (hidden from the agent)
  workspaces/example/refs/<name>/<view>.png   reference images the agent may look at

    python scripts/make_agent_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.render import RenderOptions, load_mesh, render_view  # noqa: E402


def mug() -> trimesh.Trimesh:
    # trimesh primitives are built along +Z; rotate them to +Y up.
    to_y = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    wall = trimesh.creation.annulus(r_min=0.42, r_max=0.5, height=1.0, sections=64)
    bottom = trimesh.creation.cylinder(radius=0.42, height=0.08, sections=64)
    bottom.apply_translation([0, 0, -0.46])
    body = trimesh.util.concatenate([wall, bottom])
    body.apply_transform(to_y)
    handle = trimesh.creation.torus(major_radius=0.27, minor_radius=0.055, major_sections=48, minor_sections=16)
    handle.apply_translation([0.62, 0.0, 0.0])  # torus lies in the XY plane: already a vertical ring
    return trimesh.util.concatenate([body, handle])


def _box(extents, centre) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(centre)
    return b


def table() -> trimesh.Trimesh:
    """Rectangular dining table: thin top, four square legs (+Y up, long side along X)."""
    parts = [_box([1.2, 0.06, 0.7], [0, 0.72, 0])]
    parts += [_box([0.06, 0.69, 0.06], [x, 0.345, z]) for x in (-0.53, 0.53) for z in (-0.28, 0.28)]
    return trimesh.util.concatenate(parts)


def chair() -> trimesh.Trimesh:
    """Simple chair facing +Z: square seat, four legs, backrest at the back (-Z)."""
    parts = [_box([0.46, 0.05, 0.46], [0, 0.45, 0])]
    parts += [_box([0.04, 0.45, 0.04], [x, 0.225, z]) for x in (-0.2, 0.2) for z in (-0.2, 0.2)]
    parts += [_box([0.04, 0.5, 0.04], [x, 0.72, -0.2]) for x in (-0.2, 0.2)]  # back posts
    parts += [_box([0.44, 0.18, 0.03], [0, 0.86, -0.2])]  # backrest panel
    return trimesh.util.concatenate(parts)


OBJECTS = {"mug": mug, "table": table, "chair": chair}


def main() -> None:
    opts = RenderOptions(size=512, views=("front", "side", "top", "iso"))
    for name, build in OBJECTS.items():
        answer = ROOT / "benchmarks" / "example" / "answers" / f"{name}.glb"
        refs = ROOT / "workspaces" / "example" / "refs" / name
        answer.parent.mkdir(parents=True, exist_ok=True)
        refs.mkdir(parents=True, exist_ok=True)
        build().export(answer)
        mesh = load_mesh(answer)
        for view in opts.views:
            img = Image.fromarray(render_view(mesh, view, opts), mode="RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.alpha_composite(img)
            bg.convert("RGB").save(refs / f"{view}.png", optimize=True)
        print(f"{answer.relative_to(ROOT)} ({len(mesh.faces)} faces), refs/{name}/{{{','.join(opts.views)}}}.png")


if __name__ == "__main__":
    main()
