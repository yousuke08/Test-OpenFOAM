"""
Visualize chtMultiRegionFoam results for ALINCO FM180SL.
Reads OpenFOAM ASCII files directly (no ParaView needed).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os, re

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def latest_time():
    dirs = [d for d in os.listdir(CASE)
            if re.match(r'^\d+\.?\d*$', d) and float(d) > 0
            and os.path.isdir(os.path.join(CASE, d))]
    return max(dirs, key=lambda x: float(x)) if dirs else None

TIME = latest_time()
if TIME is None:
    raise RuntimeError("No time directories found.")
print(f"Reading t = {TIME} s")

OUT_DIR = os.path.join(CASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)


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
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
        i += 1
    body = text[start:i-1]
    return n, body

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
    nums = list(map(int, re.findall(r'\d+', body)))
    return np.array(nums[:n])

def read_faces(region):
    path = os.path.join(CASE, "constant", region, "polyMesh", "faces")
    with open(path) as f:
        raw = _remove_comments(f.read())
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    n, body = _find_outer_list(raw)
    faces = []
    for fm in re.finditer(r'(\d+)\s*\(([^)]+)\)', body):
        n_v = int(fm.group(1))
        verts = list(map(int, fm.group(2).split()))
        faces.append(verts[:n_v])
    return faces

def cell_centres(region):
    pts    = read_points(region)
    owners = read_int_list(os.path.join(CASE, "constant", region, "polyMesh", "owner"))
    faces  = read_faces(region)
    n_cells = int(owners.max()) + 1
    cx  = np.zeros(n_cells)
    cy  = np.zeros(n_cells)
    cnt = np.zeros(n_cells, dtype=int)
    for fi, face in enumerate(faces):
        if fi >= len(owners):
            break
        fc  = pts[face].mean(axis=0)
        ow  = owners[fi]
        cx[ow] += fc[0]
        cy[ow] += fc[1]
        cnt[ow] += 1
    m = cnt > 0
    cx[m] /= cnt[m]
    cy[m] /= cnt[m]
    return cx, cy

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
    if idx < 0:
        raise ValueError(f"No internalField in {path}")
    sub = clean[idx:]
    n, body = _find_outer_list(sub)
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
    sub = clean[idx:]
    n, body = _find_outer_list(sub)
    nums = list(map(float, re.sub(r'[()]', ' ', body).split()))
    return np.array(nums[:3*n]).reshape(n, 3)


print("Computing aluminum cell centres ...")
cx_al, cy_al = cell_centres("aluminum")
print("Reading aluminum/T ...")
T_al = read_scalar_field("aluminum", TIME, "T") - 273.15

print("Computing air cell centres ...")
cx_air, cy_air = cell_centres("air")
print("Reading air/T and air/U ...")
T_air = read_scalar_field("air", TIME, "T") - 273.15
U_air = read_vector_field("air", TIME, "U")

print(f"\n  Al  T: {T_al.min():.1f} - {T_al.max():.1f} deg C  (n={len(T_al)})")
print(f"  Air T: {T_air.min():.1f} - {T_air.max():.1f} deg C  (n={len(T_air)})")
if U_air.shape[0] > 1:
    spd = np.linalg.norm(U_air, axis=1)
    print(f"  Air |U| max = {spd.max():.4f} m/s")


# ALINCO FM180SL aluminum outline (meters, origin at backbone center bottom)
al_outline = [
    [(-0.006,0),( 0.006,0),( 0.006,0.001),(-0.006,0.001)],   # backbone
    [(-0.006,0.001),(-0.005,0.001),(-0.005,0.008),(-0.006,0.008)],   # left fin
    [(-0.0005,0.001),(0.0005,0.001),(0.0005,0.008),(-0.0005,0.008)], # mid fin
    [( 0.005,0.001),( 0.006,0.001),( 0.006,0.008),( 0.005,0.008)],  # right fin
]

fig, axes = plt.subplots(1, 2, figsize=(14, 9))
fig.suptitle(
    f"ALINCO FM180SL E-channel — Conjugate Heat Transfer + Natural Convection\n"
    f"Fins UP  |  Q=2.5W (full backbone)  |  ambient=20 deg C  |  t={float(TIME):.0f} s",
    fontsize=12, fontweight='bold')

# ── left panel: aluminum temperature ──
ax = axes[0]
ax.set_title("Aluminum Temperature [deg C]", fontsize=11)
if len(T_al) >= 3 and len(cx_al) == len(T_al):
    try:
        triang_al = tri.Triangulation(cx_al, cy_al)
        cs = ax.tricontourf(triang_al, T_al, levels=20, cmap="hot")
        ax.tricontour(triang_al, T_al, levels=8, colors='k', linewidths=0.4, alpha=0.5)
        plt.colorbar(cs, ax=ax, label="T [deg C]")
    except Exception:
        sc = ax.scatter(cx_al, cy_al, c=T_al, cmap='hot', s=50)
        plt.colorbar(sc, ax=ax, label="T [deg C]")
else:
    sc = ax.scatter(cx_al, cy_al, c=T_al, cmap='hot', s=50)
    plt.colorbar(sc, ax=ax, label="T [deg C]")
for pts in al_outline:
    ax.add_patch(plt.Polygon(pts, fill=False, edgecolor='cyan',
                             linewidth=1.0, linestyle='--', alpha=0.8))
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_xlim(-0.008, 0.008); ax.set_ylim(-0.001, 0.010)
ax.set_aspect("equal")

# ── right panel: air temperature + velocity ──
ax = axes[1]
ax.set_title("Air Temperature [deg C]  +  Velocity Vectors", fontsize=11)
T_air_clipped = np.clip(T_air, -50, T_al.max())  # clip unphysical below-ambient
if len(T_air) >= 3 and len(cx_air) == len(T_air):
    try:
        triang_air = tri.Triangulation(cx_air, cy_air)
        cs2 = ax.tricontourf(triang_air, T_air_clipped, levels=20, cmap="coolwarm")
        ax.tricontour(triang_air, T_air_clipped, levels=6, colors='k', linewidths=0.4, alpha=0.4)
        plt.colorbar(cs2, ax=ax, label="T [deg C]")
    except Exception:
        sc2 = ax.scatter(cx_air, cy_air, c=T_air_clipped, cmap='coolwarm', s=30)
        plt.colorbar(sc2, ax=ax, label="T [deg C]")
else:
    sc2 = ax.scatter(cx_air, cy_air, c=T_air_clipped, cmap='coolwarm', s=30)
    plt.colorbar(sc2, ax=ax, label="T [deg C]")
if U_air.shape[0] > 1 and U_air.shape[0] == len(cx_air):
    step = max(1, len(cx_air) // 150)
    spd  = np.linalg.norm(U_air, axis=1)
    eps  = 1e-10
    Unx  = U_air[:, 0] / (spd + eps)
    Uny  = U_air[:, 1] / (spd + eps)
    ax.quiver(cx_air[::step], cy_air[::step],
              Unx[::step], Uny[::step],
              color='white', alpha=0.70, scale=35,
              width=0.002, headwidth=3.5, headlength=4.5)
    ax.text(-0.100, 0.108, f"|U|max = {spd.max():.3f} m/s",
            fontsize=7, color='black', va='top')
for pts in al_outline:
    ax.add_patch(plt.Polygon(pts, fill=True, facecolor='gray',
                             edgecolor='white', linewidth=0.8, alpha=0.5))
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_xlim(-0.112, 0.112); ax.set_ylim(-0.055, 0.115)
ax.set_aspect("equal")

plt.tight_layout()
out = os.path.join(OUT_DIR, f"cht_result_t{TIME}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out}")
