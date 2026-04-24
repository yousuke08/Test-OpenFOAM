"""
3D visualization for ALINCO FM180SL CHT (LED point-source half-model).
Shows X-Y slices at three Z positions: LED center, midpoint, open end.
Reads OpenFOAM ASCII files directly (no ParaView needed).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os, re, sys

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def latest_time():
    dirs = [d for d in os.listdir(CASE)
            if re.match(r'^\d+\.?\d*$', d) and float(d) > 0
            and os.path.isdir(os.path.join(CASE, d))]
    return max(dirs, key=lambda x: float(x)) if dirs else None

TIME = latest_time()
if TIME is None:
    raise RuntimeError("No time directories found. Run the simulation first.")
print(f"Reading t = {TIME} s")

OUT_DIR = os.path.join(CASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _remove_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text

def _find_outer_list(text):
    m = re.search(r'\b(\d+)\s*\(', text)
    if not m:
        raise ValueError("No list found")
    n = int(m.group(1))
    start = m.end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == '(':   depth += 1
        elif text[i] == ')': depth -= 1
        i += 1
    return n, text[start:i-1]

def read_points(region):
    path = os.path.join(CASE, "constant", region, "polyMesh", "points")
    with open(path) as f:
        raw = _remove_comments(f.read())
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    n, body = _find_outer_list(raw)
    nums = list(map(float, re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', body)))
    return np.array(nums[:3*n]).reshape(n, 3)

def read_int_list(path):
    with open(path) as f:
        raw = _remove_comments(f.read())
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    n, body = _find_outer_list(raw)
    return np.array(list(map(int, re.findall(r'\d+', body)))[:n])

def read_faces(region):
    path = os.path.join(CASE, "constant", region, "polyMesh", "faces")
    with open(path) as f:
        raw = _remove_comments(f.read())
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    n, body = _find_outer_list(raw)
    faces = []
    for fm in re.finditer(r'(\d+)\s*\(([^)]+)\)', body):
        verts = list(map(int, fm.group(2).split()))
        faces.append(verts[:int(fm.group(1))])
    return faces

def cell_centres_3d(region):
    pts    = read_points(region)
    owners = read_int_list(os.path.join(CASE, "constant", region, "polyMesh", "owner"))
    faces  = read_faces(region)
    n_cells = int(owners.max()) + 1
    cx = np.zeros(n_cells); cy = np.zeros(n_cells); cz = np.zeros(n_cells)
    cnt = np.zeros(n_cells, dtype=int)
    for fi, face in enumerate(faces):
        if fi >= len(owners): break
        fc = pts[face].mean(axis=0)
        ow = owners[fi]
        cx[ow] += fc[0]; cy[ow] += fc[1]; cz[ow] += fc[2]
        cnt[ow] += 1
    m = cnt > 0
    cx[m] /= cnt[m]; cy[m] /= cnt[m]; cz[m] /= cnt[m]
    return cx, cy, cz

def read_scalar_field(region, time, field):
    path = os.path.join(CASE, time, region, field)
    with open(path) as f:
        raw = f.read()
    m = re.search(r'internalField\s+uniform\s+([-+\d.eE]+)', raw)
    if m:
        return np.array([float(m.group(1))])
    clean = _remove_comments(raw)
    clean = re.sub(r'FoamFile\s*\{[^}]*\}', '', clean)
    idx = clean.find('internalField')
    n, body = _find_outer_list(clean[idx:])
    return np.array(list(map(float, body.split())))

def read_vector_field(region, time, field):
    path = os.path.join(CASE, time, region, field)
    with open(path) as f:
        raw = f.read()
    m = re.search(r'internalField\s+uniform\s+\(([^)]+)\)', raw)
    if m:
        return np.array([[float(x) for x in m.group(1).split()]])
    clean = _remove_comments(raw)
    clean = re.sub(r'FoamFile\s*\{[^}]*\}', '', clean)
    idx = clean.find('internalField')
    n, body = _find_outer_list(clean[idx:])
    nums = list(map(float, re.sub(r'[()]', ' ', body).split()))
    return np.array(nums[:3*n]).reshape(n, 3)


# ── Load data ─────────────────────────────────────────────────────────────────

print("Computing cell centres (3D) ...")
cx_al, cy_al, cz_al = cell_centres_3d("aluminum")
cx_air, cy_air, cz_air = cell_centres_3d("air")

print("Reading fields ...")
T_al  = read_scalar_field("aluminum", TIME, "T") - 273.15
T_air = read_scalar_field("air",      TIME, "T") - 273.15
U_air = read_vector_field("air",      TIME, "U")

print(f"  Al  T: {T_al.min():.2f} – {T_al.max():.2f} °C  (n={len(T_al)})")
print(f"  Air T: {T_air.min():.2f} – {T_air.max():.2f} °C  (n={len(T_air)})")
if U_air.shape[0] > 1:
    print(f"  Air |U|max = {np.linalg.norm(U_air,axis=1).max():.4f} m/s")
print(f"  Z range: aluminum [{cz_al.min():.4f}, {cz_al.max():.4f}] m")

# Z-slices to plot (m): LED center, midpoint, open-end
# Half-model: z=0 (open) to z=0.150 (LED/symmetry)
Z_SLICES   = [0.145, 0.075, 0.005]
Z_LABELS   = ["z=145mm (LED center)", "z=75mm (midpoint)", "z=5mm (open end)"]
DZ_HALF    = 0.006   # half-width of slice band

# Global colour limits for consistent colour scale across slices
T_min_al  = T_al.min()
T_max_al  = T_al.max()
T_min_air = max(T_air.min(), 19.0)   # clip below-ambient artefacts
T_max_air = T_al.max()               # same ceiling as aluminum

# ALINCO FM180SL aluminum outline
al_outline = [
    [(-0.006,0),( 0.006,0),( 0.006,0.001),(-0.006,0.001)],
    [(-0.006,0.001),(-0.005,0.001),(-0.005,0.008),(-0.006,0.008)],
    [(-0.0005,0.001),(0.0005,0.001),(0.0005,0.008),(-0.0005,0.008)],
    [( 0.005,0.001),( 0.006,0.001),( 0.006,0.008),( 0.005,0.008)],
]


def plot_slice(ax_al, ax_air, z_center, label):
    """Plot one z-slice onto pre-created axes."""
    dz = DZ_HALF

    # ── Aluminum slice ─────────────────────────────────
    mask_al = np.abs(cz_al - z_center) < dz
    if mask_al.sum() >= 3:
        x_al = cx_al[mask_al]; y_al = cy_al[mask_al]; T_a = T_al[mask_al]
        try:
            tr = tri.Triangulation(x_al, y_al)
            cs = ax_al.tricontourf(tr, T_a, levels=20,
                                   vmin=T_min_al, vmax=T_max_al, cmap="hot")
            ax_al.tricontour(tr, T_a, levels=6, colors='k', linewidths=0.3, alpha=0.5)
        except Exception:
            cs = ax_al.scatter(x_al, y_al, c=T_a, cmap='hot',
                               vmin=T_min_al, vmax=T_max_al, s=60)
    ax_al.set_xlim(-0.008, 0.008); ax_al.set_ylim(-0.001, 0.010)
    ax_al.set_aspect("equal")
    ax_al.set_title(f"Al temp — {label}", fontsize=9)
    ax_al.set_xlabel("x [m]"); ax_al.set_ylabel("y [m]")
    for pts in al_outline:
        ax_al.add_patch(plt.Polygon(pts, fill=False, edgecolor='cyan',
                                    linewidth=0.8, linestyle='--', alpha=0.7))

    # ── Air slice ──────────────────────────────────────
    mask_air = np.abs(cz_air - z_center) < dz
    if mask_air.sum() >= 3:
        x_a = cx_air[mask_air]; y_a = cy_air[mask_air]
        T_s = np.clip(T_air[mask_air], T_min_air, T_max_air)
        try:
            tr2 = tri.Triangulation(x_a, y_a)
            cs2 = ax_air.tricontourf(tr2, T_s, levels=20,
                                     vmin=T_min_air, vmax=T_max_air, cmap="coolwarm")
            ax_air.tricontour(tr2, T_s, levels=5, colors='k', linewidths=0.3, alpha=0.4)
        except Exception:
            cs2 = ax_air.scatter(x_a, y_a, c=T_s, cmap='coolwarm',
                                 vmin=T_min_air, vmax=T_max_air, s=20)
        # velocity arrows
        if U_air.shape[0] == len(cx_air) and mask_air.sum() > 0:
            U_sl = U_air[mask_air]
            spd  = np.linalg.norm(U_sl, axis=1)
            eps  = 1e-10
            Unx = U_sl[:, 0] / (spd + eps)
            Uny = U_sl[:, 1] / (spd + eps)
            step = max(1, mask_air.sum() // 120)
            ax_air.quiver(x_a[::step], y_a[::step],
                          Unx[::step], Uny[::step],
                          color='white', alpha=0.70, scale=35,
                          width=0.002, headwidth=3.5, headlength=4.5)
            ax_air.text(-0.105, 0.110, f"|U|max={spd.max():.3f}m/s",
                        fontsize=7, color='black', va='top')
    for pts in al_outline:
        ax_air.add_patch(plt.Polygon(pts, fill=True, facecolor='gray',
                                     edgecolor='white', linewidth=0.6, alpha=0.5))
    ax_air.set_xlim(-0.112, 0.112); ax_air.set_ylim(-0.055, 0.115)
    ax_air.set_aspect("equal")
    ax_air.set_title(f"Air temp + vel — {label}", fontsize=9)
    ax_air.set_xlabel("x [m]"); ax_air.set_ylabel("y [m]")


# ── Figure: 3 rows × 2 cols ──────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(14, 20))
fig.suptitle(
    f"ALINCO FM180SL — 3-D LED Point Source CHT\n"
    f"Half-model (z-symm at z=150mm) | Q=2.5W | ambient=20°C | t={float(TIME):.0f}s",
    fontsize=12, fontweight='bold')

for row, (z_c, lbl) in enumerate(zip(Z_SLICES, Z_LABELS)):
    plot_slice(axes[row, 0], axes[row, 1], z_c, lbl)

# Shared colorbars
import matplotlib.cm as cm
from matplotlib.colors import Normalize
sm_al  = plt.cm.ScalarMappable(cmap='hot',      norm=Normalize(T_min_al,  T_max_al))
sm_air = plt.cm.ScalarMappable(cmap='coolwarm', norm=Normalize(T_min_air, T_max_air))
sm_al.set_array([]); sm_air.set_array([])
fig.colorbar(sm_al,  ax=axes[:, 0], label="T [°C]", shrink=0.6, pad=0.02)
fig.colorbar(sm_air, ax=axes[:, 1], label="T [°C]", shrink=0.6, pad=0.02)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT_DIR, f"cht_3d_slices_t{TIME}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")


# ── Figure 2: Z-profile of max temperature ───────────────────────────────────

# For each z-layer in aluminum, find max T
z_layers = np.unique(np.round(cz_al, 4))
T_max_per_z = []
for z in z_layers:
    mask = np.abs(cz_al - z) < 0.006
    if mask.sum() > 0:
        T_max_per_z.append(T_al[mask].max())
    else:
        T_max_per_z.append(np.nan)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(np.array(z_layers)*1000, T_max_per_z, 'r-o', markersize=5, linewidth=2)
ax2.axvline(145, color='orange', linestyle='--', alpha=0.8, label='LED zone (z=145mm)')
ax2.axvline(140, color='orange', linestyle=':', alpha=0.6, label='LED zone edge')
ax2.set_xlabel("z [mm]"); ax2.set_ylabel("Max aluminum temperature [°C]")
ax2.set_title(
    f"ALINCO FM180SL — Max Al Temperature vs Z  (t={float(TIME):.0f}s)\n"
    f"LED at z=140–150mm (symmetry at 150mm) | Q=2.5W")
ax2.legend(); ax2.grid(True, alpha=0.3)
out2 = os.path.join(OUT_DIR, f"cht_3d_zprofile_t{TIME}.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved -> {out2}")
