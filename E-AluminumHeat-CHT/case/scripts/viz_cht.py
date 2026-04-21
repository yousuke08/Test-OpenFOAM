"""
Visualize chtMultiRegionFoam results.
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
    times = [int(d) for d in os.listdir(CASE)
             if d.isdigit() and int(d) > 0
             and os.path.isdir(os.path.join(CASE, d))]
    return str(max(times)) if times else None

TIME = latest_time()
if TIME is None:
    raise RuntimeError("No time directories found.")
print(f"Reading t = {TIME} s")

OUT_DIR = os.path.join(CASE, "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ── parser helpers ───────────────────────────────────────────────────────────

def _remove_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text

def _find_outer_list(text):
    """Find N and the body inside the outermost  N ( ... )  list."""
    # find first bare integer not inside a string
    m = re.search(r'\b(\d+)\s*\(', text)
    if not m:
        raise ValueError("No list found")
    n = int(m.group(1))
    start = m.end()   # position just after '('
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
    # Remove FoamFile{...} block
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    n, body = _find_outer_list(raw)
    nums = list(map(float, re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', body)))
    return np.array(nums[:3*n]).reshape(n, 3)

def read_int_list(path):
    with open(path) as f:
        raw = _remove_comments(f.read())
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    # Note/neiStart etc come before the actual list
    n, body = _find_outer_list(raw)
    nums = list(map(int, re.findall(r'\d+', body)))
    return np.array(nums[:n])

def read_faces(region):
    path = os.path.join(CASE, "constant", region, "polyMesh", "faces")
    with open(path) as f:
        raw = _remove_comments(f.read())
    raw = re.sub(r'FoamFile\s*\{[^}]*\}', '', raw)
    n, body = _find_outer_list(raw)
    # Each face: N(v0 v1 ... vN-1)
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
    # Find internalField nonuniform entry then list
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


# ── read data ────────────────────────────────────────────────────────────────

print("Computing aluminum cell centres …")
cx_al, cy_al = cell_centres("aluminum")
print("Reading aluminum/T …")
T_al = read_scalar_field("aluminum", TIME, "T") - 273.15

print("Computing air cell centres …")
cx_air, cy_air = cell_centres("air")
print("Reading air/T and air/U …")
T_air = read_scalar_field("air", TIME, "T") - 273.15
U_air = read_vector_field("air", TIME, "U")

print(f"\n  Al  T: {T_al.min():.1f} – {T_al.max():.1f} °C  (n={len(T_al)})")
print(f"  Air T: {T_air.min():.1f} – {T_air.max():.1f} °C  (n={len(T_air)})")
if U_air.shape[0] > 1:
    spd = np.linalg.norm(U_air, axis=1)
    print(f"  Air |U| max = {spd.max():.4f} m/s")


# ── plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 9))
fig.suptitle(
    f"E-shape Al Heatsink — Conjugate Heat Transfer + Natural Convection\n"
    f"Fins pointing UP  |  base = 100 °C  |  ambient = 20 °C  |  t = {TIME} s",
    fontsize=12, fontweight='bold')

al_outline = [
    [(0,0),(0.10,0),(0.10,0.02),(0,0.02)],            # backbone
    [(0,0.02),(0.02,0.02),(0.02,0.10),(0,0.10)],       # left fin
    [(0.04,0.02),(0.06,0.02),(0.06,0.10),(0.04,0.10)], # middle fin
    [(0.08,0.02),(0.10,0.02),(0.10,0.10),(0.08,0.10)], # right fin
]

# ── left panel: aluminum temperature ──
ax = axes[0]
ax.set_title("Aluminum Temperature  [°C]", fontsize=11)
if len(T_al) >= 3 and len(cx_al) == len(T_al):
    triang_al = tri.Triangulation(cx_al, cy_al)
    cs = ax.tricontourf(triang_al, T_al, levels=20, cmap="hot")
    ax.tricontour(triang_al, T_al, levels=8, colors='k', linewidths=0.4, alpha=0.5)
    plt.colorbar(cs, ax=ax, label="T [°C]")
else:
    sc = ax.scatter(cx_al, cy_al, c=T_al, cmap='hot', s=30)
    plt.colorbar(sc, ax=ax, label="T [°C]")
for pts in al_outline:
    ax.add_patch(plt.Polygon(pts, fill=False, edgecolor='cyan',
                             linewidth=1.0, linestyle='--', alpha=0.8))
ax.text(0.05, 0.01, "100°C\n(base)", ha='center', va='center',
        fontsize=7, color='cyan')
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_xlim(-0.002, 0.102); ax.set_ylim(-0.002, 0.102)
ax.set_aspect("equal")

# ── right panel: air temperature + velocity ──
ax = axes[1]
ax.set_title("Air Temperature [°C]  +  Velocity Vectors", fontsize=11)
if len(T_air) >= 3 and len(cx_air) == len(T_air):
    triang_air = tri.Triangulation(cx_air, cy_air)
    cs2 = ax.tricontourf(triang_air, T_air, levels=20, cmap="coolwarm")
    ax.tricontour(triang_air, T_air, levels=6, colors='k', linewidths=0.4, alpha=0.4)
    plt.colorbar(cs2, ax=ax, label="T [°C]")
else:
    sc2 = ax.scatter(cx_air, cy_air, c=T_air, cmap='coolwarm', s=30)
    plt.colorbar(sc2, ax=ax, label="T [°C]")
if U_air.shape[0] > 1 and U_air.shape[0] == len(cx_air):
    step = max(1, len(cx_air) // 150)
    ax.quiver(cx_air[::step], cy_air[::step],
              U_air[::step, 0], U_air[::step, 1],
              color='white', alpha=0.85, scale_units='xy', scale=4)
for pts in al_outline:
    ax.add_patch(plt.Polygon(pts, fill=True, facecolor='gray',
                             edgecolor='white', linewidth=0.8, alpha=0.35))
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_xlim(-0.002, 0.102); ax.set_ylim(-0.002, 0.202)
ax.set_aspect("equal")

plt.tight_layout()
out = os.path.join(OUT_DIR, f"cht_result_t{TIME}.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")
