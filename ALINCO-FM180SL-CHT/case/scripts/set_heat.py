"""
set_heat.py  –  Set heat-flux boundary condition in W.

Usage:
    python3 scripts/set_heat.py --watts 2.5

Must be run AFTER splitMeshRegions so that the aluminum polyMesh exists.
Reads the 'base' patch area from the mesh, then writes the correct
fixedGradient value into 0.orig/aluminum/T.
"""

import argparse
import os
import re

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY = os.path.join(CASE, "constant", "aluminum", "polyMesh")
T_FILE = os.path.join(CASE, "0.orig", "aluminum", "T")
KAPPA = 205.0  # W/(m·K) – aluminum thermal conductivity


def _read_text(path):
    with open(path, "r") as f:
        text = f.read()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _read_list(path):
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


def read_boundary(poly_dir):
    path = os.path.join(poly_dir, "boundary")
    text = _read_text(path)
    patches = {}
    for pm in re.finditer(r"(\w[\w_]*)\s*\{([^}]*)\}", text):
        name = pm.group(1)
        body = pm.group(2)
        nf = re.search(r"nFaces\s+(\d+)", body)
        sf = re.search(r"startFace\s+(\d+)", body)
        if nf and sf:
            patches[name] = {"nFaces": int(nf.group(1)), "startFace": int(sf.group(1))}
    return patches


def patch_area(poly_dir, patch_name):
    patches = read_boundary(poly_dir)
    if patch_name not in patches:
        available = list(patches.keys())
        raise KeyError(f"Patch '{patch_name}' not found. Available: {available}")

    n_faces = patches[patch_name]["nFaces"]
    start_face = patches[patch_name]["startFace"]

    pts_raw = _read_list(os.path.join(poly_dir, "points"))
    points = []
    for i in range(0, len(pts_raw), 3):
        points.append((pts_raw[i], pts_raw[i + 1], pts_raw[i + 2]))

    faces_text = _read_text(os.path.join(poly_dir, "faces"))
    face_list = []
    for fm in re.finditer(r"\d+\(([^)]+)\)", faces_text):
        idxs = list(map(int, fm.group(1).split()))
        face_list.append(idxs)

    if not face_list:
        raise RuntimeError("Could not parse faces file.")

    total_area = 0.0
    for fi in range(start_face, start_face + n_faces):
        verts = [points[k] for k in face_list[fi]]
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


def main():
    parser = argparse.ArgumentParser(description="Set heat-flux BC in watts.")
    parser.add_argument("--watts", type=float, required=True, help="Heat input [W]")
    args = parser.parse_args()

    Q = args.watts
    A = patch_area(POLY, "base")
    q = Q / A
    gradient = q / KAPPA

    print(f"  Patch area  = {A:.6f} m²")
    print(f"  Heat flux   = {q:.2f} W/m²")
    print(f"  Gradient    = {gradient:.4f} K/m")

    with open(T_FILE, "r") as f:
        content = f.read()

    pattern = r"(gradient\s+uniform\s+)[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(\s*;)"

    if not re.search(pattern, content):
        raise RuntimeError("Could not find 'gradient uniform ...' line to replace.")

    new_content = re.sub(pattern, rf"\g<1>{gradient:.4f}\2", content)

    with open(T_FILE, "w") as f:
        f.write(new_content)

    print(f"  Updated {T_FILE}")
    print(f"  → {Q} W applied to base patch.")


if __name__ == "__main__":
    main()
