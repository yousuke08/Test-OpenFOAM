"""
Create animated GIF from 3D CHT results (ALINCO FM180SL LED point-source).
Shows LED-center slice (x-y) + Z-profile of max aluminum temperature.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import os, re, glob, sys
import imageio.v2 as imageio
import tempfile

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(CASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── I/O helpers (same as viz_cht_3d.py) ─────────────────────────────────────

def _remove_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text

def _find_outer_list(text):
    m = re.search(r'\b(\d+)\s*\(', text)
    if not m: raise ValueError("No list found")
    n, start, depth, i = int(m.group(1)), m.end(), 1, m.end()
    while i < len(text) and depth > 0:
        if text[i] == '(':   depth += 1
        elif text[i] == ')': depth -= 1
        i += 1
    return n, text[start:i-1]

def read_points(region):
    with open(os.path.join(CASE, "constant", region, "polyMesh", "points")) as f:
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
    with open(os.path.join(CASE, "constant", region, "polyMesh", "faces")) as f:
        raw = _remove_comments(f.read())
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    n, body = _find_outer_list(raw)
    faces = []
    for fm in re.finditer(r'(\d+)\s*\(([^)]+)\)', body):
        faces.append(list(map(int, fm.group(2).split()))[:int(fm.group(1))])
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
    with open(path) as f: raw = f.read()
    m = re.search(r'internalField\s+uniform\s+([-+\d.eE]+)', raw)
    if m: return np.array([float(m.group(1))])
    clean = _remove_comments(raw)
    clean = re.sub(r'FoamFile\s*\{[^}]*\}', '', clean)
    n, body = _find_outer_list(clean[clean.find('internalField'):])
    return np.array(list(map(float, body.split())))

def read_vector_field(region, time, field):
    path = os.path.join(CASE, time, region, field)
    with open(path) as f: raw = f.read()
    m = re.search(r'internalField\s+uniform\s+\(([^)]+)\)', raw)
    if m: return np.array([[float(x) for x in m.group(1).split()]])
    clean = _remove_comments(raw)
    clean = re.sub(r'FoamFile\s*\{[^}]*\}', '', clean)
    n, body = _find_outer_list(clean[clean.find('internalField'):])
    nums = list(map(float, re.sub(r'[()]', ' ', body).split()))
    return np.array(nums[:3*n]).reshape(n, 3)


# ── Find time directories ─────────────────────────────────────────────────────

times = sorted(
    [d for d in os.listdir(CASE)
     if re.match(r'^\d+\.?\d*$', d) and float(d) > 0
     and os.path.isdir(os.path.join(CASE, d, "aluminum"))],
    key=float)

if not times:
    raise RuntimeError("No time directories with aluminum/ found.")
print(f"Time steps found: {times}")

# ── Pre-compute mesh geometry (constant) ─────────────────────────────────────

print("Computing cell centres ...")
cx_al, cy_al, cz_al   = cell_centres_3d("aluminum")
cx_air, cy_air, cz_air = cell_centres_3d("air")

# Z-layers in aluminum
z_layers = np.unique(np.round(cz_al, 4))
DZ_HALF  = 0.006    # ±6mm band
Z_LED    = 0.145    # LED-center slice

al_outline = [
    [(-0.006,0),( 0.006,0),( 0.006,0.001),(-0.006,0.001)],
    [(-0.006,0.001),(-0.005,0.001),(-0.005,0.008),(-0.006,0.008)],
    [(-0.0005,0.001),(0.0005,0.001),(0.0005,0.008),(-0.0005,0.008)],
    [( 0.005,0.001),( 0.006,0.001),( 0.006,0.008),( 0.005,0.008)],
]

# ── Determine global colour range from all time steps ─────────────────────────

print("Scanning temperature range across all time steps ...")
T_global_max = 20.0
for t in times:
    T_al  = read_scalar_field("aluminum", t, "T") - 273.15
    T_global_max = max(T_global_max, T_al.max())
T_global_min = 20.0  # ambient

print(f"Global T range: {T_global_min:.1f} – {T_global_max:.1f} °C")

# ── Generate one frame per time step ─────────────────────────────────────────

tmp_dir = tempfile.mkdtemp()
frame_paths = []

# Mask for LED-center slice
mask_al_led  = np.abs(cz_al  - Z_LED) < DZ_HALF
mask_air_led = np.abs(cz_air - Z_LED) < DZ_HALF

for t in times:
    print(f"  Frame t={t} s ...")
    T_al  = read_scalar_field("aluminum", t, "T") - 273.15
    T_air = read_scalar_field("air",      t, "T") - 273.15
    U_air = read_vector_field("air",      t, "U")

    # Z-profile
    T_max_per_z = [T_al[np.abs(cz_al-z)<DZ_HALF].max()
                   if (np.abs(cz_al-z)<DZ_HALF).any() else np.nan
                   for z in z_layers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"ALINCO FM180SL — 3-D LED Point Source CHT\n"
        f"Q=2.5W total | ambient=20°C | t={float(t):.0f}s | "
        f"Al max T={T_al.max():.1f}°C",
        fontsize=11, fontweight='bold')

    # ── Panel 1: Al temperature at LED center ─────────────────────────────
    ax = axes[0]
    x_a = cx_al[mask_al_led]; y_a = cy_al[mask_al_led]; T_s = T_al[mask_al_led]
    if len(x_a) >= 3:
        try:
            tr = tri.Triangulation(x_a, y_a)
            ax.tricontourf(tr, T_s, levels=20,
                           vmin=T_global_min, vmax=T_global_max, cmap='hot')
            ax.tricontour(tr, T_s, levels=6, colors='k', linewidths=0.3, alpha=0.5)
        except Exception:
            ax.scatter(x_a, y_a, c=T_s, cmap='hot',
                       vmin=T_global_min, vmax=T_global_max, s=80)
    for pts in al_outline:
        ax.add_patch(plt.Polygon(pts, fill=False, edgecolor='cyan',
                                 linewidth=0.8, linestyle='--', alpha=0.7))
    ax.set_xlim(-0.008, 0.008); ax.set_ylim(-0.001, 0.010)
    ax.set_aspect("equal")
    ax.set_title("Al Temp @ z=145mm (LED center)", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    sm = plt.cm.ScalarMappable(cmap='hot',
                               norm=Normalize(T_global_min, T_global_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="T [°C]", shrink=0.7)

    # ── Panel 2: Air temp + velocity at LED center ────────────────────────
    ax = axes[1]
    x_r = cx_air[mask_air_led]; y_r = cy_air[mask_air_led]
    T_r = np.clip(T_air[mask_air_led], T_global_min, T_global_max)
    if len(x_r) >= 3:
        try:
            tr2 = tri.Triangulation(x_r, y_r)
            ax.tricontourf(tr2, T_r, levels=20,
                           vmin=T_global_min, vmax=T_global_max, cmap='coolwarm')
        except Exception:
            ax.scatter(x_r, y_r, c=T_r, cmap='coolwarm',
                       vmin=T_global_min, vmax=T_global_max, s=20)
    if U_air.shape[0] == len(cx_air) and mask_air_led.sum() > 0:
        U_sl = U_air[mask_air_led]
        spd  = np.linalg.norm(U_sl, axis=1)
        eps  = 1e-10
        step = max(1, mask_air_led.sum() // 120)
        ax.quiver(x_r[::step], y_r[::step],
                  (U_sl[:,0]/(spd+eps))[::step], (U_sl[:,1]/(spd+eps))[::step],
                  color='white', alpha=0.70, scale=35,
                  width=0.002, headwidth=3.5, headlength=4.5)
        ax.text(-0.105, 0.110, f"|U|max={spd.max():.3f}m/s",
                fontsize=7, color='k', va='top')
    for pts in al_outline:
        ax.add_patch(plt.Polygon(pts, fill=True, facecolor='gray',
                                 edgecolor='white', linewidth=0.6, alpha=0.5))
    ax.set_xlim(-0.112, 0.112); ax.set_ylim(-0.055, 0.115)
    ax.set_aspect("equal")
    ax.set_title("Air Temp+Vel @ z=145mm (LED center)", fontsize=10)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    sm2 = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=Normalize(T_global_min, T_global_max))
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax, label="T [°C]", shrink=0.7)

    # ── Panel 3: Z-profile of max Al temperature ──────────────────────────
    ax = axes[2]
    ax.plot(np.array(z_layers)*1000, T_max_per_z, 'r-o', markersize=5, linewidth=2)
    ax.axvline(145, color='orange', linestyle='--', alpha=0.8, label='LED center')
    ax.axvline(140, color='orange', linestyle=':', alpha=0.6, label='LED zone edge')
    ax.set_xlim(0, 155); ax.set_ylim(T_global_min - 0.5, T_global_max + 0.5)
    ax.set_xlabel("z [mm]"); ax.set_ylabel("Max Al T [°C]")
    ax.set_title("Z-profile of Max Al Temperature", fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    frame_path = os.path.join(tmp_dir, f"frame_{t:>06}.png")
    plt.savefig(frame_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    frame_paths.append(frame_path)

# ── Assemble GIF ─────────────────────────────────────────────────────────────

out_gif = os.path.join(OUT_DIR, "cht_3d_animation.gif")
frames = [imageio.imread(p) for p in frame_paths]
imageio.mimsave(out_gif, frames, duration=0.5, loop=0)
print(f"\nGIF saved -> {out_gif}  ({len(frames)} frames)")

# Clean up temp frames
import shutil; shutil.rmtree(tmp_dir)
