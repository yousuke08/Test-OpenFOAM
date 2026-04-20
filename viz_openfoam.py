#!/usr/bin/env python3
"""
Visualize OpenFOAM laplacianFoam results for E-shaped aluminum.
Reads native OpenFOAM ASCII files: polyMesh/points, polyMesh/cells, T field.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

CASE = "/home/user/Test-OpenFOAM/E-AluminumHeat"
TIME = "200"

# ── helpers ──────────────────────────────────────────────────────────────────

def read_scalar_list(path):
    """Read an OpenFOAM ASCII file containing a scalar list."""
    with open(path) as f:
        text = f.read()
    # Find the block after 'internalField' or just the first numeric block
    # Pattern: integer on its own line followed by ( ... )
    m = re.search(r'\b(\d+)\s*\(\s*([\d\s.e+\-]+?)\s*\)', text, re.DOTALL)
    if m:
        n = int(m.group(1))
        vals = list(map(float, m.group(2).split()))
        return np.array(vals[:n])
    raise ValueError(f"Cannot parse scalar list in {path}")

def read_vector_list(path):
    """Read an OpenFOAM ASCII file containing a vector list ( (x y z) ... )."""
    with open(path) as f:
        text = f.read()
    # Extract all (x y z) tuples
    tuples = re.findall(r'\(\s*([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s*\)', text)
    return np.array([[float(a), float(b), float(c)] for a, b, c in tuples])

def read_cell_owner(path):
    """Read owner file to map faces to cells."""
    with open(path) as f:
        text = f.read()
    m = re.search(r'\b(\d+)\s*\(\s*([\d\s]+?)\s*\)', text, re.DOTALL)
    if m:
        n = int(m.group(1))
        vals = list(map(int, m.group(2).split()))
        return np.array(vals[:n])
    raise ValueError(f"Cannot parse owner in {path}")

# ── load data ────────────────────────────────────────────────────────────────
print("Reading mesh …", end=" ", flush=True)
points = read_vector_list(f"{CASE}/constant/polyMesh/points")
print(f"{len(points)} points", end=", ", flush=True)

owner   = read_cell_owner(f"{CASE}/constant/polyMesh/owner")
neighbour = read_cell_owner(f"{CASE}/constant/polyMesh/neighbour")

# Read faces (list of lists)
with open(f"{CASE}/constant/polyMesh/faces") as f:
    txt = f.read()

# Each face is like:  4(0 1 2 3)  or  3(...)
face_re = re.compile(r'\d+\s*\(([^)]+)\)')
raw_faces = []
# Find the big block: look for number-of-faces line then the list
m_block = re.search(r'\b(\d+)\s*\n\s*\(', txt)
if m_block:
    block_start = txt.index('(', m_block.start())
    block_inner = txt[block_start+1:]
    depth = 1
    end = 0
    for i, c in enumerate(block_inner):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0:
            end = i
            break
    block_txt = block_inner[:end]
    for match in face_re.finditer(block_txt):
        raw_faces.append(list(map(int, match.group(1).split())))

# Compute cell centres from face owners
n_cells = owner.max() + 1
cell_pts_sum  = np.zeros((n_cells, 3))
cell_pts_cnt  = np.zeros(n_cells, dtype=int)

for fid, face_nodes in enumerate(raw_faces):
    cx = points[face_nodes, 0].mean()
    cy = points[face_nodes, 1].mean()
    cz = points[face_nodes, 2].mean()
    ow = owner[fid]
    cell_pts_sum[ow] += [cx, cy, cz]
    cell_pts_cnt[ow] += 1

for fid, face_nodes in enumerate(raw_faces):
    if fid < len(neighbour):
        nb = neighbour[fid]
        if nb >= 0:
            cx = points[face_nodes, 0].mean()
            cy = points[face_nodes, 1].mean()
            cz = points[face_nodes, 2].mean()
            cell_pts_sum[nb] += [cx, cy, cz]
            cell_pts_cnt[nb] += 1

cell_centres = cell_pts_sum / cell_pts_cnt[:, None]
print(f"{n_cells} cells")

print("Reading T field …", end=" ", flush=True)
T_vals = read_scalar_list(f"{CASE}/{TIME}/T")
T_C = T_vals - 273.15   # convert K → °C
print(f"range {T_C.min():.1f} – {T_C.max():.1f} °C")

# ── use z ≈ 0.01 (mid-plane) slice for 2-D plot ──────────────────────────────
# Since the domain has symmetry in z (front/back symmetryPlane), all cells
# at different z are valid, just pick the mid-plane
cx = cell_centres[:, 0] * 1000   # m → mm
cy = cell_centres[:, 1] * 1000
cz = cell_centres[:, 2] * 1000

# ── figure ───────────────────────────────────────────────────────────────────
BG = "#0d1117"
fig, axes = plt.subplots(1, 2, figsize=(17, 8), facecolor=BG)

vmin, vmax = 20.0, 100.0
fill_levels = np.linspace(vmin, vmax, 120)
line_levels = np.arange(25, 100, 5)

def _style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.set_xlabel("x  (mm)", color="white", fontsize=11)
    ax.set_ylabel("y  (mm)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=12, pad=8)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 108)
    ax.set_ylim(-3, 108)

# ─── panel 1 : filled contour ───────────────────────────────────────────────
ax1 = axes[0]
tri_obj = tri.Triangulation(cx, cy)

# mask triangles outside E-shape (large triangles spanning gaps)
# any triangle with all vertices inside the E is valid
def in_e(x, y):
    return (
        (x <= 20) |
        (y <= 20) |
        ((y >= 40) & (y <= 60)) |
        (y >= 80)
    )

in_mask = in_e(cx[tri_obj.triangles[:, 0]],
               cy[tri_obj.triangles[:, 0]]) & \
          in_e(cx[tri_obj.triangles[:, 1]],
               cy[tri_obj.triangles[:, 1]]) & \
          in_e(cx[tri_obj.triangles[:, 2]],
               cy[tri_obj.triangles[:, 2]])
tri_obj.set_mask(~in_mask)

cf = ax1.tricontourf(tri_obj, T_C, levels=fill_levels,
                     cmap="plasma", vmin=vmin, vmax=vmax, extend="both")
cl = ax1.tricontour(tri_obj, T_C, levels=line_levels,
                    colors="white", linewidths=0.6, alpha=0.65)
ax1.clabel(cl, fmt="%d°C", fontsize=7, colors="white", inline=True)

cb = fig.colorbar(cf, ax=ax1, fraction=0.038, pad=0.03)
cb.set_label("Temperature  (°C)", color="white", fontsize=11)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

_style(ax1, f"OpenFOAM laplacianFoam  —  T at t = {TIME} s  (steady state)")

ax1.annotate("Heat source\n100 °C", xy=(1, 50),
             xytext=(-30, 50), color="#ff6e6e", fontsize=9, ha="center", va="center",
             arrowprops=dict(arrowstyle="->", color="#ff6e6e", lw=1.5))
ax1.text(107, 50, "Cooled\n20 °C", color="#74b9ff", fontsize=9, ha="left", va="center")

for y_gap, lbl in [(30, "Air gap"), (70, "Air gap")]:
    ax1.text(60, y_gap, lbl, color="#aaaaaa", fontsize=8, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#555", lw=0.8))

# ─── panel 2 : scatter by z-layer ───────────────────────────────────────────
ax2 = axes[1]
sc = ax2.scatter(cx, cy, c=T_C, cmap="plasma", vmin=vmin, vmax=vmax,
                 s=10, linewidths=0, alpha=0.9)
cb2 = fig.colorbar(sc, ax=ax2, fraction=0.038, pad=0.03)
cb2.set_label("Temperature  (°C)", color="white", fontsize=11)
cb2.ax.yaxis.set_tick_params(color="white")
plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
_style(ax2, "Cell-centre temperatures (all z-layers)")

fig.suptitle(
    "E-shaped Aluminum Heatsink  —  OpenFOAM laplacianFoam (v1912)\n"
    "Material: Al (DT = 8.42×10⁻⁵ m²/s, κ = 205 W/m·K)  |  1088 cells\n"
    "Base: 100 °C (x=0)   →   Cooled surfaces: 20 °C",
    color="white", fontsize=12, fontweight="bold", y=1.03,
)
plt.tight_layout()
out = "/home/user/Test-OpenFOAM/heat_contour_OF.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
