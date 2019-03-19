"""
Generate the logo of this project
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import COORDS_STAR, BONDS_INTRA


LINE_WIDTH = 5
MARKER_SIZE = 12
FONT_SIZE = 60
color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))

shrink_inner, shrink_outer = 0.30, 0.15
coords_star = np.array(COORDS_STAR, copy=True)
coords_star[1:7, :] *= (1 - shrink_inner)
coords_star[7:, :] *= (1 - shrink_outer)

fig, ax = plt.subplots()
fig.set_size_inches(4, 2)
fig.subplots_adjust(
    top=1.0, bottom=0.0, left=0.0, right=0.4, hspace=0.2, wspace=0.2
)
# Plot the intra-cluster bonds
for bonds, color in zip(BONDS_INTRA, COLORS[3:6]):
    for (v0, index0), (v1, index1) in bonds:
        bond = coords_star[[index0, index1]]
        ax.plot(bond[:, 0], bond[:, 1], color=color, lw=LINE_WIDTH)

ax.plot(
    coords_star[0, 0], coords_star[0, 1],
    color=COLORS[0], marker="o", ls="", ms=MARKER_SIZE,
)
ax.plot(
    coords_star[1:7, 0], coords_star[1:7, 1],
    color=COLORS[1], marker="o", ls="", ms=MARKER_SIZE,
)
ax.plot(
    coords_star[7:, 0], coords_star[7:, 1],
    color=COLORS[2], marker="o", ls="", ms=MARKER_SIZE,
)
ax.text(
    1.3, 0.5, r"$TaS_2$", fontsize=FONT_SIZE,
    ha="left", va="center", transform=ax.transAxes
)

ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig("icons/TaS2Logo.svg")
plt.close("all")