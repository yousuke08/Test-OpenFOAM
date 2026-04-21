#!/usr/bin/env python3
"""
E-shaped aluminum heatsink — steady-state heat conduction
Numerically equivalent to running laplacianFoam to convergence.

Geometry (100 x 100 mm cross-section):
  Backbone : x in [0, 20mm], full height
  Bottom fin: x in [0,100mm], y in [0,  20mm]
  Middle fin: x in [0,100mm], y in [40, 60mm]
  Top fin   : x in [0,100mm], y in [80,100mm]

BCs:
  base (x=0)     : T = 100 C  (heat source)
  cooled surfaces: T =  20 C  (simplified convection)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ── parameters ──────────────────────────────────────────────────────────────
N     = 400       # grid intervals per side (must be multiple of 20)
T_HOT = 100.0     # °C
T_AMB =  20.0     # °C
L     = 0.10      # domain size [m]

# ── grid ────────────────────────────────────────────────────────────────────
dx = L / N
x  = np.linspace(0, L, N + 1)
y  = np.linspace(0, L, N + 1)
XX, YY = np.meshgrid(x, y, indexing="ij")   # shape (N+1, N+1)

# ── E-shape mask ─────────────────────────────────────────────────────────────
#   Union of backbone + three fins (all in aluminum)
mask = (
    (XX <= 0.02) |                              # backbone
    (YY <= 0.02) |                              # bottom fin
    ((YY >= 0.04) & (YY <= 0.06)) |            # middle fin
    (YY >= 0.08)                                # top fin
)

# ── node numbering ───────────────────────────────────────────────────────────
node_num = np.full((N + 1, N + 1), -1, dtype=np.int32)
node_num[mask] = np.arange(mask.sum())
n_nodes = int(mask.sum())

# ── boundary classification ──────────────────────────────────────────────────
# base: x = 0 column
is_base = np.zeros_like(mask)
is_base[0, :] = mask[0, :]

# nodes adjacent to air (pad domain boundary with False = air)
pm       = np.pad(mask, 1, constant_values=False)
has_air  = (
    ~pm[:-2, 1:-1] | ~pm[2:,  1:-1] |  # left / right neighbor
    ~pm[1:-1, :-2] | ~pm[1:-1, 2:]     # below / above neighbor
) & mask

is_cooled   = has_air & ~is_base
is_interior = mask & ~has_air & ~is_base

print(f"Aluminum nodes : {n_nodes:,}")
print(f"  base         : {is_base.sum():,}")
print(f"  cooled       : {is_cooled.sum():,}")
print(f"  interior     : {is_interior.sum():,}")

# ── sparse linear system ─────────────────────────────────────────────────────
ri, ci = np.where(is_interior)
ki = node_num[ri, ci]

# 5-point Laplace stencil for interior nodes
row_i = np.concatenate([ki] * 5)
col_i = np.concatenate([
    ki,
    node_num[ri - 1, ci],
    node_num[ri + 1, ci],
    node_num[ri,     ci - 1],
    node_num[ri,     ci + 1],
])
val_i = np.concatenate([
    np.full(len(ki), -4.0),
    np.ones(len(ki)),
    np.ones(len(ki)),
    np.ones(len(ki)),
    np.ones(len(ki)),
])

# Dirichlet rows (identity)
kb    = np.concatenate([node_num[is_base], node_num[is_cooled]])
rows  = np.concatenate([row_i, kb])
cols  = np.concatenate([col_i, kb])
vals  = np.concatenate([val_i, np.ones(len(kb))])

A = csr_matrix(coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)))

b = np.zeros(n_nodes)
b[node_num[is_base]]   = T_HOT
b[node_num[is_cooled]] = T_AMB

# ── solve ────────────────────────────────────────────────────────────────────
print("Solving sparse system …", end=" ", flush=True)
T_sol = spsolve(A, b)
print("done.")

# ── map back to 2-D grid ─────────────────────────────────────────────────────
T2d = np.full((N + 1, N + 1), np.nan)
T2d[mask] = T_sol[node_num[mask]]

# ── heat flux  q = -grad T  (direction only) ─────────────────────────────────
T_g  = np.where(mask, T2d, 0.0)
qx   = -np.gradient(T_g, dx, axis=0)
qy   = -np.gradient(T_g, dx, axis=1)
qmag = np.hypot(qx, qy)
qxn  = np.where(mask, qx  / (qmag + 1e-12), np.nan)
qyn  = np.where(mask, qy  / (qmag + 1e-12), np.nan)

# ── statistics ───────────────────────────────────────────────────────────────
T_al = T2d[mask]
print(f"Temperature range : {T_al.min():.2f} – {T_al.max():.2f} °C")
print(f"Mean temperature  : {T_al.mean():.2f} °C")
# Temperature at fin tips (x=0.1, mid-y of each fin)
for label, yval in [("bottom fin tip (y=10mm)", 0.01),
                    ("middle fin tip (y=50mm)", 0.05),
                    ("top    fin tip (y=90mm)", 0.09)]:
    xi = N        # x = 0.1 m
    ji = round(yval / dx)
    if mask[xi, ji]:
        print(f"  {label}: {T2d[xi, ji]:.1f} °C")

# ── figure ───────────────────────────────────────────────────────────────────
BG   = "#0d1117"
TICK = "white"

fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)

vmin, vmax = 20, 100
fill_levels = np.linspace(vmin, vmax, 120)
line_levels = np.arange(vmin + 5, vmax, 5)     # every 5 °C
X_mm = XX * 1000
Y_mm = YY * 1000

def _style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TICK, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.set_xlabel("x  (mm)", color=TICK, fontsize=11)
    ax.set_ylabel("y  (mm)", color=TICK, fontsize=11)
    ax.set_title(title, color=TICK, fontsize=12, pad=8)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 108)
    ax.set_ylim(-3, 108)

# ─── panel 1 : temperature contour ──────────────────────────────────────────
ax = axes[0]
cf = ax.contourf(X_mm, Y_mm, T2d, levels=fill_levels,
                 cmap="plasma", vmin=vmin, vmax=vmax, extend="both")
cl = ax.contour(X_mm, Y_mm, T2d, levels=line_levels,
                colors="white", linewidths=0.6, alpha=0.65)
ax.clabel(cl, fmt="%d°C", fontsize=7, colors="white", inline=True)

# E outline
ax.contour(X_mm, Y_mm, mask.astype(float),
           levels=[0.5], colors=["#00e5ff"], linewidths=1.8)

cb = fig.colorbar(cf, ax=ax, fraction=0.038, pad=0.03)
cb.set_label("Temperature  (°C)", color=TICK, fontsize=11)
cb.ax.yaxis.set_tick_params(color=TICK)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TICK)

_style(ax, "Steady-state Temperature  [laplacianFoam equiv.]")

# annotations
ax.annotate("Heat source\n100 °C", xy=(1, 50),
            xytext=(-30, 50), color="#ff6e6e", fontsize=9, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#ff6e6e", lw=1.5))
ax.text(106, 50, "Cooled\n20 °C", color="#74b9ff", fontsize=9, ha="left", va="center")

# gap labels
for y_gap, label in [(30, "Air gap"), (70, "Air gap")]:
    ax.text(60, y_gap, label, color="#aaaaaa", fontsize=8,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#555", lw=0.8))

# ─── panel 2 : heat flux ────────────────────────────────────────────────────
ax2 = axes[1]
cf2 = ax2.contourf(X_mm, Y_mm, T2d, levels=fill_levels,
                   cmap="plasma", vmin=vmin, vmax=vmax, extend="both")

s = max(1, N // 28)   # quiver stride
ax2.quiver(X_mm[::s, ::s], Y_mm[::s, ::s],
           qxn[::s, ::s], qyn[::s, ::s],
           color="white", alpha=0.70, scale=32,
           width=0.0025, headwidth=3.5, headlength=4.5)

ax2.contour(X_mm, Y_mm, mask.astype(float),
            levels=[0.5], colors=["#00e5ff"], linewidths=1.8)

cb2 = fig.colorbar(cf2, ax=ax2, fraction=0.038, pad=0.03)
cb2.set_label("Temperature  (°C)", color=TICK, fontsize=11)
cb2.ax.yaxis.set_tick_params(color=TICK)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=TICK)

_style(ax2, "Heat Flux Direction  (-grad T)")

fig.suptitle(
    "E-shaped Aluminum Heatsink  —  Heat Dissipation Simulation\n"
    "Material: Al (κ = 205 W/m·K)  |  Backbone: 20 mm  |  Fins: 80 mm × 20 mm × 3\n"
    "Base: 100 °C  →  Cooled surfaces: 20 °C",
    color="white", fontsize=13, fontweight="bold", y=1.03,
)

plt.tight_layout()
out = "/home/user/Test-OpenFOAM/heat_contour.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
