"""
set_heat.py  –  Set heat-flux boundary condition in W.

Usage:
    python3 scripts/set_heat.py --watts 10

Must be run AFTER splitMeshRegions so that the aluminum polyMesh exists.
Reads the 'base' patch area from the mesh, then writes the correct
fixedGradient value into 0.orig/aluminum/T.
"""

import argparse
import os
import re
import struct

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY = os.path.join(CASE, "constant", "aluminum", "polyMesh")
T_FILE = os.path.join(CASE, "0.orig", "aluminum", "T")
KAPPA = 205.0  # W/(m·K) – aluminum thermal conductivity


# ── OpenFOAM file helpers ─────────────────────────────────────────────────────

def _read_text(path):
    with open(path, "r") as f:
        text = f.read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _read_list(path):
    """Return a flat list of floats from an OpenFOAM field file (ASCII only)."""
    text = _read_text(path)
    m = re.search(r"\b(\d+)\s*\(", text)
    if not m:
        return []
    n = int(m.group(1))
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    body = text[start : i - 1]
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", body)
    return [float(x) for x in nums]


# ── boundary file parser ──────────────────────────────────────────────────────

def read_boundary(poly_dir):
    """Return dict {patch_name: {nFaces, startFace}} from boundary file."""
    path = os.path.join(poly_dir, "boundary")
    text = _read_text(path)
    # find top-level patch count
    m = re.search(r"\b(\d+)\s*\n\s*\{", text)
    count = int(m.group(1)) if m else 0

    patches = {}
    for pm in re.finditer(r"(\w[\w_]*)\s*\{([^}]*)\}", text):
        name = pm.group(1)
        body = pm.group(2)
        nf = re.search(r"nFaces\s+(\d+)", body)
        sf = re.search(r"startFace\s+(\d+)", body)
        if nf and sf:
            patches[name] = {"nFaces": int(nf.group(1)), "startFace": int(sf.group(1))}
    return patches


# ── face-area calculator ──────────────────────────────────────────────────────

def patch_area(poly_dir, patch_name):
    """Compute total area of a named patch from faces + points files."""
    patches = read_boundary(poly_dir)
    if patch_name not in patches:
        available = list(patches.keys())
        raise KeyError(f"Patch '{patch_name}' not found. Available: {available}")

    n_faces = patches[patch_name]["nFaces"]
    start_face = patches[patch_name]["startFace"]

    # Read points
    pts_raw = _read_list(os.path.join(poly_dir, "points"))
    points = []
    for i in range(0, len(pts_raw), 3):
        points.append((pts_raw[i], pts_raw[i + 1], pts_raw[i + 2]))

    # Read faces (each face is a variable-length list of point indices)
    faces_text = _read_text(os.path.join(poly_dir, "faces"))
    # Each face looks like:  4(0 1 2 3)
    face_list = []
    for fm in re.finditer(r"\d+\(([^)]+)\)", faces_text):
        idxs = list(map(int, fm.group(1).split()))
        face_list.append(idxs)

    if not face_list:
        raise RuntimeError("Could not parse faces file.")

    # Sum area of base patch faces
    total_area = 0.0
    for fi in range(start_face, start_face + n_faces):
        verts = [points[k] for k in face_list[fi]]
        # Compute area via cross-product of triangles (fan from verts[0])
        v0 = verts[0]
        for j in range(1, len(verts) - 1):
            v1, v2 = verts[j], verts[j + 1]
            ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            cx = ay * bz - az * by
            cy = az * bx - ax * bz
            cz = ax * by - ay * bx
            total_area += 0.5 * (cx**2 + cy**2 + cz**2) ** 0.5

    return total_area


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Set heat-flux BC in watts.")
    parser.add_argument("--watts", type=float, required=True, help="Heat input [W]")
    args = parser.parse_args()

    Q = args.watts
    A = patch_area(POLY, "base")
    q = Q / A
    gradient = q / KAPPA   # positive: heat into solid at base patch

    print(f"  Area        = {A:.6f} m²")
    print(f"  Heat flux   = {q:.2f} W/m²")
    print(f"  Gradient    = {gradient:.4f} K/m")

    # Patch 0.orig/aluminum/T
    with open(T_FILE, "r") as f:
        content = f.read()

    new_content = re.sub(
        r"(gradient\s+uniform\s+)[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(\s*;)",
        rf"\g<1>{gradient:.4f}\2",
        content,
    )

    if new_content == content:
        raise RuntimeError("Could not find 'gradient uniform ...' line to replace.")

    with open(T_FILE, "w") as f:
        f.write(new_content)

    print(f"  Updated {T_FILE}")
    print(f"  → {Q} W applied to base patch.")


if __name__ == "__main__":
    main()
