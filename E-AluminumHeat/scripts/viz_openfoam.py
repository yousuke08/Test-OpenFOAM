#!/usr/bin/env python3
"""
Visualize OpenFOAM laplacianFoam results for E-shaped aluminum.
Reads native OpenFOAM ASCII files: polyMesh/points, polyMesh/cells, T field.

Fixes vs v1:
  - Use only the mid-plane z-layer (not all 4 z-layers overlapping)
  - Mask triangles by centroid position AND max edge length to correctly
    show the E-shape without filling in air gaps
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri

CASE = "/home/user/Test-OpenFOAM/E-AluminumHeat/case"
TIME = "1.0"

# ── helpers ──────────────────────────────────────────────────────────────────

def read_scalar_list(path):
    with open(path) as f:
        text = f.read()
    m = re.search(r'\b(\d+)\s*\(\s*([\d\s.e+\-]+?)\s*\)', text, re.DOTALL)
    if m:
        n = int(m.group(1))
        vals = list(map(float, m.group(2).split()))
        return np.array(vals[:n])
    raise ValueError(f"Cannot parse scalar list in {path}")

def read_vector_list(path):
    with open(path) as f:
        text = f.read()
    tuples = re.findall(r'\(\s*([\d.e+\-]+)\s+([\d.e+\-]+)\s+([\d.e+\-]+)\s*\)', text)
    return np.array([[float(a), float(b), float(c)] for a, b, c in tuples])

def read_int_list(path):
    with open(path) as f:
        text = f.read()
    m = re.search(r'\b(\d+)\s*\(\s*([\d\s]+?)\s*\)', text, re.DOTALL)
    if m:
        n = int(m.group(1))
        vals = list(map(int, m.group(2).split()))
        return np.array(vals[:n])
    raise ValueError(f"Cannot parse int list in {path}")

# ── load mesh ────────────────────────────────────────────────────────────────
print("Reading mesh …", end=" ", flush=True)
points  = read_vector_list(f"{CASE}/constant/polyMesh/points")
owner   = read_int_list(f"{CASE}/constant/polyMesh/owner")
neighbr = read_int_list(f"{CASE}/constant/polyMesh/neighbour")

with open(f"{CASE}/constant/polyMesh/faces") as f:
    txt = f.read()
face_re = re.compile(r'\d+\s*\(([^)]+)\)')
m_block = re.search(r'\b(\d+)\s*\n\s*\(', txt)
raw_faces = []
if m_block:
    block_start = txt.index('(', m_block.start())
    block_inner = txt[block_start + 1:]
    depth, end = 1, 0
    for i, c in enumerate(block_inner):
        if c == '(':   depth += 1
        elif c == ')': depth -= 1
        if depth == 0: end = i; break
    for match in face_re.finditer(block_inner[:end]):
        raw_faces.append(list(map(int, match.group(1).split())))

# ── compute cell centres ─────────────────────────────────────────────────────
n_cells = owner.max() + 1
csum = np.zeros((n_cells, 3))
ccnt = np.zeros(n_cells, dtype=int)

for fid, fn in enumerate(raw_faces):
    fc = points[fn].mean(axis=0)
    ow = owner[fid]
    csum[ow] += fc; ccnt[ow] += 1
    if fid < len(neighbr) and neighbr[fid] >= 0:
        nb = neighbr[fid]
        csum[nb] += fc; ccnt[nb] += 1

cell_centres = csum / ccnt[:, None]
print(f"{n_cells} cells")

# ── load temperature ─────────────────────────────────────────────────────────
print("Reading T field …", end=" ", flush=True)
T_vals = read_scalar_list(f"{CASE}/{TIME}/T")
T_C    = T_vals - 273.15
print(f"range {T_C.min():.1f} – {T_C.max():.1f} °C")

# ── pick ONE z-layer (mid-plane) ─────────────────────────────────────────────
# The 3-D mesh has 4 cells in z (0→0.02 m); cell centres at z≈0.0025,0.0075,0.0125,0.0175
cz_vals = cell_centres[:, 2]
z_layers = np.unique(np.round(cz_vals * 1e6).astype(int)) / 1e6   # round to µm
z_mid    = z_layers[len(z_layers) // 2]           # pick the middle layer
z_tol    = (z_layers[1] - z_layers[0]) * 0.4
sel      = np.abs(cz_vals - z_mid) < z_tol

cx_2d = cell_centres[sel, 0] * 1000   # m → mm
cy_2d = cell_centres[sel, 1] * 1000
T_2d  = T_C[sel]
print(f"Using z-layer z={z_mid*1000:.2f} mm  ({sel.sum()} cells)")

# ── E-shape membership (mm units) ────────────────────────────────────────────
def in_e(xm, ym):
    return (
        (xm <= 20) |
        (ym <= 20) |
        ((ym >= 40) & (ym <= 60)) |
        (ym >= 80)
    )

# ── build triangulation with proper masking ───────────────────────────────────
tri_obj = tri.Triangulation(cx_2d, cy_2d)

# centroid of each triangle must be inside E-shape
t = tri_obj.triangles
tri_cx = cx_2d[t].mean(axis=1)
tri_cy = cy_2d[t].mean(axis=1)
centroid_ok = in_e(tri_cx, tri_cy)

# max edge length must be small (cell size ≈5 mm; gaps ≥20 mm → threshold=12 mm)
pts = np.column_stack([cx_2d, cy_2d])
v0, v1, v2 = pts[t[:, 0]], pts[t[:, 1]], pts[t[:, 2]]
max_edge = np.maximum(
    np.linalg.norm(v1 - v0, axis=1),
    np.maximum(np.linalg.norm(v2 - v1, axis=1),
               np.linalg.norm(v0 - v2, axis=1))
)
edge_ok = max_edge < 12.0

tri_obj.set_mask(~(centroid_ok & edge_ok))

# ── figure ───────────────────────────────────────────────────────────────────
BG = "#0d1117"
fig, axes = plt.subplots(1, 2, figsize=(17, 8), facecolor=BG)

vmin, vmax   = 20.0, 100.0
fill_levels  = np.linspace(vmin, vmax, 120)
line_levels  = np.arange(25, 100, 5)

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

# ─── panel 1 : filled contour (correct E-shape) ──────────────────────────────
ax1 = axes[0]
cf = ax1.tricontourf(tri_obj, T_2d, levels=fill_levels,
                     cmap="plasma", vmin=vmin, vmax=vmax, extend="both")
cl = ax1.tricontour(tri_obj, T_2d, levels=line_levels,
                    colors="white", linewidths=0.6, alpha=0.65)
ax1.clabel(cl, fmt="%d°C", fontsize=7, colors="white", inline=True)

cb = fig.colorbar(cf, ax=ax1, fraction=0.038, pad=0.03)
cb.set_label("Temperature  (°C)", color="white", fontsize=11)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

_style(ax1, f"OpenFOAM laplacianFoam  —  T at t={TIME} s  (steady state)")

ax1.annotate("Heat source\n100 °C", xy=(1, 50),
             xytext=(-30, 50), color="#ff6e6e", fontsize=9, ha="center", va="center",
             arrowprops=dict(arrowstyle="->", color="#ff6e6e", lw=1.5))
ax1.text(107, 50, "Cooled\n20 °C", color="#74b9ff", fontsize=9, ha="left", va="center")
for y_gap, lbl in [(30, "Air gap"), (70, "Air gap")]:
    ax1.text(60, y_gap, lbl, color="#aaaaaa", fontsize=8, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#555", lw=0.8))

# ─── panel 2 : cell-centre scatter (shows exact mesh) ────────────────────────
ax2 = axes[1]
sc = ax2.scatter(cx_2d, cy_2d, c=T_2d, cmap="plasma",
                 vmin=vmin, vmax=vmax, s=14, linewidths=0, alpha=0.95)
cb2 = fig.colorbar(sc, ax=ax2, fraction=0.038, pad=0.03)
cb2.set_label("Temperature  (°C)", color="white", fontsize=11)
cb2.ax.yaxis.set_tick_params(color="white")
plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")
_style(ax2, f"Cell-centre temperatures  (mid-plane z={z_mid*1000:.1f} mm)")

fig.suptitle(
    "E-shaped Aluminum Heatsink  —  OpenFOAM laplacianFoam (v1912)\n"
    "Material: Al (DT = 8.42×10⁻⁵ m²/s, κ = 205 W/m·K)  |  1088 cells  |  Steady state\n"
    "Base (x=0): 100 °C   →   Cooled surfaces: 20 °C",
    color="white", fontsize=12, fontweight="bold", y=1.03,
)
plt.tight_layout()

out = "/home/user/Test-OpenFOAM/E-AluminumHeat/results/heat_contour_OF.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
