"""
Show the First Brillouin Zone of the David-Star and the original triangular
lattice
"""


import matplotlib.pyplot as plt

from TaS2DataBase import Gamma, Ms_CELL, Ms_STAR, Ks_CELL, Ks_STAR


linewidth = 4
marker_size = 16
font_size = "xx-large"

color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))

fig, ax = plt.subplots()

# Draw the 1st BZ boundary
boundary = Ks_CELL[[0, 1, 2, 3, 4, 5, 0]]
ax.plot(boundary[:, 0], boundary[:, 1], color=colors[0], lw=linewidth)
boundary = Ks_STAR[[0, 1, 2, 3, 4, 5, 0]]
ax.plot(boundary[:, 0], boundary[:, 1], color=colors[1], lw=linewidth)

# Draw and annotate the Gamma point
ax.plot(
    Gamma[0], Gamma[1], marker='o', color=colors[2], markersize=marker_size,
)
ax.annotate(
    r"$\Gamma$", (Gamma[0], Gamma[1]),
    ha="left", va="bottom",  fontsize=font_size,
)

# Draw all M-points, annotate two inequivalent M-points
ax.plot(
    Ms_CELL[:, 0], Ms_CELL[:, 1],
    marker="o", ls="", color=colors[3], markersize=marker_size,
)
ax.annotate(
    r"$M_1$", (Ms_CELL[0, 0], Ms_CELL[0, 1]),
    ha="left", va="bottom", fontsize=font_size,
)
ax.annotate(
    r"$M_1^{'}$", (Ms_CELL[1, 0], Ms_CELL[1, 1]),
    ha="left", va="bottom", fontsize=font_size,
)

ax.plot(
    Ms_STAR[:, 0], Ms_STAR[:, 1],
    marker="o", ls="", color=colors[4], markersize=marker_size,
)
ax.annotate(
    r"$M_2$", (Ms_STAR[0, 0], Ms_STAR[0, 1]),
    ha="left", va="bottom", fontsize=font_size,
)
ax.annotate(
    r"$M_2^{'}$", (Ms_STAR[1, 0], Ms_STAR[1, 1]),
    ha="left", va="bottom", fontsize=font_size,
)

# Draw all K-points, annotate two inequivalent K-points
ax.plot(
    Ks_CELL[:, 0], Ks_CELL[:, 1],
    marker='o', ls="", color=colors[5], markersize=marker_size,
)
ax.annotate(
    r"$K_1$", (Ks_CELL[3, 0], Ks_CELL[3, 1]),
    ha="left", va="bottom", fontsize=font_size,
)
ax.annotate(
    r"$K_1^{'}$", (Ks_CELL[4, 0], Ks_CELL[4, 1]),
    ha="left", va="bottom", fontsize=font_size,
)

ax.plot(
    Ks_STAR[:, 0], Ks_STAR[:, 1],
    marker='o', ls="", color=colors[6], markersize=marker_size,
)
ax.annotate(
    r"$K_2$", (Ks_STAR[3, 0], Ks_STAR[3, 1]),
    ha="left", va="center", fontsize=font_size,
)
ax.annotate(
    r"$K_2^{'}$", (Ks_STAR[4, 0], Ks_STAR[4, 1]),
    ha="left", va="center", fontsize=font_size,
)

title = "1st BZ of the David-Star\nand the original triangular lattice"
ax.set_title(title, fontsize=font_size)
ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig("FirstBrillouinZone.jpg", dpi=1000)
plt.close("all")
