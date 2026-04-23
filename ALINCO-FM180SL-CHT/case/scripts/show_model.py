"""
ALINCO FM180SL E-channel cross-section preview.
Dimensions: H=12mm, W=8mm, t=1mm, gap=4.5mm
Heat source: backbone left face at x=0
Air space: 50mm on all sides
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ── dimensions (mm) ──────────────────────────────────────────────────────────
t    = 1.0    # wall thickness
H    = 12.0   # total height
W    = 8.0    # total width
gap  = 4.5    # gap between flanges  = (H - 3*t) / 2 = 4.5 ✓
L    = 20.0   # air margin on each side (shown); real domain = 50mm

# derived y-coordinates
y_bot_top   = t               # 1.0  bottom flange top
y_mid_bot   = t + gap         # 5.5  middle flange bottom
y_mid_top   = t + gap + t     # 6.5  middle flange top
y_top_bot   = t + gap + t + gap  # 11.0 top flange bottom

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#0d0d1a')

# Air background
ax.add_patch(patches.Rectangle((-L, -L), W + 2*L, H + 2*L,
    facecolor='#1a3a6e', edgecolor='gray', linewidth=0.5, alpha=0.4, zorder=1))

# Air inside channel (inner gaps)
for (y0, y1) in [(y_bot_top, y_mid_bot), (y_mid_top, y_top_bot)]:
    ax.add_patch(patches.Rectangle((t, y0), W - t, y1 - y0,
        facecolor='#1a3a6e', edgecolor=None, alpha=0.9, zorder=2))

# ── aluminum ─────────────────────────────────────────────────────────────────
al_color = '#a0a8b8'
al_edge  = '#d0d8e8'

def al_rect(x, y, w, h):
    ax.add_patch(patches.Rectangle((x, y), w, h,
        facecolor=al_color, edgecolor=al_edge, linewidth=0.8, zorder=3))

al_rect(0, 0,           t, H)      # backbone (full height)
al_rect(0, 0,           W, t)      # bottom flange
al_rect(0, y_mid_bot,   W, t)      # middle flange
al_rect(0, y_top_bot,   W, t)      # top flange

# ── heat source marker (backbone left face, x=0) ─────────────────────────────
ax.plot([0, 0], [0, H], color='#ff4040', linewidth=5, zorder=6,
        solid_capstyle='round', label='Heat source face (x=0)')
ax.plot(0, 0, 'r*', markersize=14, zorder=7)
ax.text(0.3, -1.8, 'Origin (0, 0)', color='#ff8080', fontsize=8)

# ── dimension annotations ─────────────────────────────────────────────────────
def arrow(x0, y0, x1, y1, color, label, lx, ly, ha='center', va='bottom'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.2))
    ax.text(lx, ly, label, color=color, fontsize=8, ha=ha, va=va)

arrow(0, -3, W, -3,     'cyan',   f'W = {W:.0f} mm',  W/2, -4)
arrow(-3, 0, -3, H,     'cyan',   f'H = {H:.0f} mm',  -4.5, H/2, ha='right', va='center')
arrow(W+1, t, W+1, y_mid_bot, 'orange', f'{gap:.1f} mm', W+2.2, (t+y_mid_bot)/2, ha='left', va='center')
arrow(W+1, y_mid_top, W+1, y_top_bot, 'orange', f'{gap:.1f} mm', W+2.2, (y_mid_top+y_top_bot)/2, ha='left', va='center')
arrow(0, H+2, t, H+2,   '#80ff80', f't={t:.0f} mm',  t/2, H+3)

# ── air domain boundary (actual 50mm, shown truncated) ───────────────────────
ax.axhline(y=-L, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.axhline(y=H+L, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.axvline(x=-L, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.axvline(x=W+L, color='white', linewidth=0.5, linestyle='--', alpha=0.4)
ax.text(-L+0.5, H+L-1, 'Open air boundary (50mm margin, all sides)',
        color='white', fontsize=7, alpha=0.7)

# ── axis settings ─────────────────────────────────────────────────────────────
ax.set_xlim(-L - 6, W + L + 5)
ax.set_ylim(-L - 6, H + L + 2)
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
    'ALINCO FM180SL  Al E-channel cross-section\n'
    'H=12mm  W=8mm  t=1mm  gap=4.5mm  (extruded 2000mm)',
    color='white', fontsize=11, fontweight='bold')

plt.tight_layout()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "model_preview.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved → {out}")
