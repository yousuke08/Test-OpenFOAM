"""
Generate animated GIF from chtMultiRegionFoam results.
Each frame = one write-time snapshot (t=20,40,...,200 s).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from PIL import Image
import os, re, io

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(CASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# collect time steps (skip t=0 which has no fields); handles both "20" and "1800.5"
times = sorted(
    [d for d in os.listdir(CASE)
     if re.match(r'^\d+\.?\d*$', d) and float(d) > 0
     and os.path.isdir(os.path.join(CASE, d, "air"))
     and os.path.isfile(os.path.join(CASE, d, "air", "T"))],
    key=lambda x: float(x)
)
print(f"Frames: {len(times)} ({times[0]} – {times[-1]})")


# ── parser (reused from viz_cht.py) ─────────────────────────────────────────

def _remove_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return re.sub(r'//[^\n]*', '', text)

def _find_outer_list(text):
    m = re.search(r'\b(\d+)\s*\(', text)
    n = int(m.group(1))
    start = m.end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == '(':   depth += 1
        elif text[i] == ')': depth -= 1
        i += 1
    return n, text[start:i-1]

def _clean(path):
    with open(path) as f:
        raw = _remove_comments(f.read())
    return re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)

def read_points(region):
    n, body = _find_outer_list(_clean(
        os.path.join(CASE, "constant", region, "polyMesh", "points")))
    nums = list(map(float, re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', body)))
    return np.array(nums[:3*n]).reshape(n, 3)

def read_int_list(path):
    n, body = _find_outer_list(_clean(path))
    return np.array(list(map(int, re.findall(r'\d+', body)))[:n])

def read_faces(region):
    n, body = _find_outer_list(_clean(
        os.path.join(CASE, "constant", region, "polyMesh", "faces")))
    return [[int(v) for v in fm.group(2).split()][:int(fm.group(1))]
            for fm in re.finditer(r'(\d+)\s*\(([^)]+)\)', body)]

def cell_centres(region):
    pts    = read_points(region)
    owners = read_int_list(os.path.join(CASE, "constant", region, "polyMesh", "owner"))
    faces  = read_faces(region)
    nc = int(owners.max()) + 1
    cx, cy, cnt = np.zeros(nc), np.zeros(nc), np.zeros(nc, int)
    for fi, face in enumerate(faces):
        if fi >= len(owners): break
        fc = pts[face].mean(axis=0)
        ow = owners[fi]
        cx[ow] += fc[0]; cy[ow] += fc[1]; cnt[ow] += 1
    m = cnt > 0
    cx[m] /= cnt[m]; cy[m] /= cnt[m]
    return cx, cy

def read_scalar(region, t, field):
    path = os.path.join(CASE, str(t), region, field)
    with open(path) as f: raw = f.read()
    m = re.search(r'internalField\s+uniform\s+([-+\d.eE]+)', raw)
    if m: return np.array([float(m.group(1))])
    clean = re.sub(r'FoamFile\s*\{[^}]*\}', '', _remove_comments(raw))
    idx = clean.find('internalField')
    n, body = _find_outer_list(clean[idx:])
    return np.array(list(map(float, body.split())))

def read_vector(region, t, field):
    path = os.path.join(CASE, str(t), region, field)
    with open(path) as f: raw = f.read()
    m = re.search(r'internalField\s+uniform\s+\(([^)]+)\)', raw)
    if m: return np.array([[float(x) for x in m.group(1).split()]])
    clean = re.sub(r'FoamFile\s*\{[^}]*\}', '', _remove_comments(raw))
    idx = clean.find('internalField')
    n, body = _find_outer_list(clean[idx:])
    nums = list(map(float, re.sub(r'[()]', ' ', body).split()))
    return np.array(nums[:3*n]).reshape(n, 3)


# ── pre-compute mesh centres (constant across time) ──────────────────────────

print("Computing mesh cell centres …")
cx_al,  cy_al  = cell_centres("aluminum")
cx_air, cy_air = cell_centres("air")
triang_al  = tri.Triangulation(cx_al,  cy_al)
triang_air = tri.Triangulation(cx_air, cy_air)

# colour range fixed across all frames for consistent animation
T_al_all  = np.concatenate([read_scalar("aluminum", t, "T") - 273.15 for t in times])
T_air_all = np.concatenate([read_scalar("air",      t, "T") - 273.15 for t in times])
al_vmin,  al_vmax  = T_al_all.min(),  T_al_all.max()
air_vmin, air_vmax = T_air_all.min(), T_air_all.max()

al_outline = [
    [(0,0),(0.10,0),(0.10,0.02),(0,0.02)],
    [(0,0.02),(0.02,0.02),(0.02,0.10),(0,0.10)],
    [(0.04,0.02),(0.06,0.02),(0.06,0.10),(0.04,0.10)],
    [(0.08,0.02),(0.10,0.02),(0.10,0.10),(0.08,0.10)],
]


# ── generate frames ───────────────────────────────────────────────────────────

def make_frame(t):
    T_al  = read_scalar("aluminum", t, "T") - 273.15
    T_air = read_scalar("air",      t, "T") - 273.15
    U_air = read_vector("air",      t, "U")

    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        f"E-shape Al Heatsink — Conjugate Heat Transfer + Natural Convection\n"
        f"Fins pointing UP  |  hotspot 2.5 W (ø3 mm)  |  ambient = 20 °C  |  t = {float(t):.1f} s",
        fontsize=11, fontweight='bold', color='white')

    levels_al  = np.linspace(al_vmin,  al_vmax,  21)
    levels_air = np.linspace(air_vmin, air_vmax, 21)

    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

    # --- left: aluminum ---
    ax = axes[0]
    ax.set_title("Aluminum Temperature [°C]", color='white', fontsize=10)
    cs = ax.tricontourf(triang_al, T_al, levels=levels_al,
                        cmap="hot", vmin=al_vmin, vmax=al_vmax, extend='both')
    ax.tricontour(triang_al, T_al, levels=8, colors='k', linewidths=0.4, alpha=0.4)
    cb = plt.colorbar(cs, ax=ax)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    cb.set_label("T [°C]", color='white')
    for pts in al_outline:
        ax.add_patch(plt.Polygon(pts, fill=False, edgecolor='cyan',
                                 linewidth=0.8, linestyle='--', alpha=0.7))
    ax.text(0.05, 0.01, "hotspot\n2.5 W", ha='center', va='center',
            fontsize=7, color='cyan')
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_xlim(-0.002, 0.102); ax.set_ylim(-0.002, 0.102)
    ax.set_aspect("equal")

    # --- right: air temperature + velocity ---
    ax = axes[1]
    ax.set_title("Air Temperature [°C]  +  Velocity Vectors", color='white', fontsize=10)
    cs2 = ax.tricontourf(triang_air, T_air, levels=levels_air,
                         cmap="coolwarm", vmin=air_vmin, vmax=air_vmax, extend='both')
    ax.tricontour(triang_air, T_air, levels=6, colors='k', linewidths=0.3, alpha=0.3)
    cb2 = plt.colorbar(cs2, ax=ax)
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
    cb2.set_label("T [°C]", color='white')
    if U_air.shape[0] > 1:
        step = max(1, len(cx_air) // 120)
        spd  = np.linalg.norm(U_air, axis=1)
        ax.quiver(cx_air[::step], cy_air[::step],
                  U_air[::step, 0], U_air[::step, 1],
                  color='white', alpha=0.75, scale_units='xy', scale=4)
        ax.text(-0.495, 0.490, f"|U|max = {spd.max():.3f} m/s",
                fontsize=7, color='white', va='top')
    for pts in al_outline:
        ax.add_patch(plt.Polygon(pts, fill=True, facecolor='#3a3a5c',
                                 edgecolor='white', linewidth=0.7, alpha=0.5))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_xlim(-0.502, 0.602); ax.set_ylim(-0.002, 0.502)
    ax.set_aspect("equal")

    plt.tight_layout()

    # render to PIL image via in-memory PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── build and save GIF ───────────────────────────────────────────────────────

print("Rendering frames …")
frames = []
for t in times:
    print(f"  t = {t} s", end='\r')
    frames.append(make_frame(t))
print(f"\n{len(frames)} frames rendered.")

# hold last frame longer
hold = 4
gif_frames = frames + [frames[-1]] * hold

out_gif = os.path.join(OUT_DIR, "cht_animation.gif")
gif_frames[0].save(
    out_gif,
    save_all=True,
    append_images=gif_frames[1:],
    duration=400,        # ms per frame
    loop=0,              # loop forever
    optimize=False,
)
print(f"Saved → {out_gif}  ({os.path.getsize(out_gif)//1024} KB)")
