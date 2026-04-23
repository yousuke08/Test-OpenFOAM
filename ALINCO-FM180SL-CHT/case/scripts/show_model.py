"""
ALINCO FM180SL E-channel cross-section preview — fins pointing UP.
Rotated so backbone is at bottom (y=0) and 3 fins extend upward.

Original product dimensions:
  total span (backbone width) = 12mm  → x: [0, 12mm]
  fin height (incl backbone)  = 8mm   → y: [0, 8mm]
  wall thickness               = 1mm
  gap between fins             = 4.5mm

Heat source: bottom face of backbone (y=0)
Air space: 50mm on all sides
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ── dimensions (mm) ──────────────────────────────────────────────────────────
t      = 1.0    # wall thickness
BW     = 12.0   # backbone total width  (original H=12mm)
FH     = 8.0    # total height incl backbone  (original W=8mm)
gap    = 4.5    # gap between fins  = (BW - 3*t) / 2 = 4.5 ✓
L      = 20.0   # air margin shown (real domain = 50mm)

# derived x-coordinates of fins
x_left_fin   = 0.0              # left fin:   x=[0,   t]
x_mid_fin    = t + gap          # middle fin: x=[5.5, 6.5]
x_right_fin  = t + gap + t + gap  # right fin:  x=[11, 12]

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#0d0d1a')

al_color = '#a0a8b8'
al_edge  = '#d0d8e8'

# Air background
ax.add_patch(patches.Rectangle((-L, -L), BW + 2*L, FH + 2*L,
    facecolor='#1a3a6e', edgecolor='gray', linewidth=0.5, alpha=0.4, zorder=1))

# Air gaps between fins (inside channel)
for x0, x1 in [(t, x_mid_fin), (x_mid_fin + t, x_right_fin)]:
    ax.add_patch(patches.Rectangle((x0, t), x1 - x0, FH - t,
        facecolor='#1a3a6e', edgecolor=None, alpha=0.9, zorder=2))

# ── aluminum ─────────────────────────────────────────────────────────────────
def al_rect(x, y, w, h):
    ax.add_patch(patches.Rectangle((x, y), w, h,
        facecolor=al_color, edgecolor=al_edge, linewidth=0.8, zorder=3))

al_rect(0, 0,        BW, t)          # backbone (full width, bottom)
al_rect(0, 0,        t,  FH)         # left fin
al_rect(x_mid_fin, 0, t, FH)         # middle fin
al_rect(x_right_fin, 0, t, FH)       # right fin

# ── heat source marker (bottom face of backbone, y=0) ────────────────────────
ax.plot([0, BW], [0, 0], color='#ff4040', linewidth=5, zorder=6,
        solid_capstyle='round', label='Heat source (y=0, backbone bottom)')
ax.plot(0, 0, 'r*', markersize=14, zorder=7)
ax.text(0.5, -1.8, 'Origin (0, 0)', color='#ff8080', fontsize=8)

# ── dimension annotations ─────────────────────────────────────────────────────
def arrow(x0, y0, x1, y1, color, label, lx, ly, ha='center', va='bottom'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
    ax.text(lx, ly, label, color=color, fontsize=8, ha=ha, va=va)

arrow(0,   -3, BW,  -3,  'cyan',    f'Backbone = {BW:.0f} mm',  BW/2,  -4.5)
arrow(-3,   0, -3,  FH,  'cyan',    f'H = {FH:.0f} mm',         -4.5,  FH/2, ha='right', va='center')
arrow(t,  FH+2, x_mid_fin, FH+2, 'orange', f'{gap:.1f} mm',   (t+x_mid_fin)/2, FH+3)
arrow(x_mid_fin+t, FH+2, x_right_fin, FH+2, 'orange', f'{gap:.1f} mm', (x_mid_fin+t+x_right_fin)/2, FH+3)
arrow(0,  FH+5, t,  FH+5, '#80ff80', f't={t:.0f}mm', t/2, FH+6)

# ── domain boundary markers ───────────────────────────────────────────────────
for v in [-L, FH+L]:
    ax.axhline(y=v, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
for v in [-L, BW+L]:
    ax.axvline(x=v, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.text(-L+0.3, FH+L-1, 'Open air boundary (50mm all sides)',
        color='white', fontsize=7, alpha=0.7)

# ── axis ─────────────────────────────────────────────────────────────────────
ax.set_xlim(-L - 5, BW + L + 3)
ax.set_ylim(-L - 7, FH + L + 4)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]', color='white')
ax.set_ylabel('y [mm]', color='white')
ax.tick_params(colors='white')
for s in ax.spines.values():
    s.set_edgecolor('gray')
ax.grid(True, alpha=0.15, color='gray')
ax.legend(loc='upper right', fontsize=9,
          facecolor='#2a2a4e', edgecolor='gray', labelcolor='white')
ax.set_title(
    'ALINCO FM180SL  Al E-channel  — fins pointing UP\n'
    'Backbone=12mm wide  fin height=8mm  t=1mm  gap=4.5mm',
    color='white', fontsize=11, fontweight='bold')

plt.tight_layout()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "model_preview.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out}")
