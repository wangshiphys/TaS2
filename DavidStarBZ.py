"""
Show the First Brillouin Zone correspond to the David-Star superlattice and
the original triangular lattice
"""


import matplotlib.pyplot as plt

from TaS2DataBase import Gamma, Ms_CELL, Ms_STAR, Ks_CELL, Ks_STAR


linewidth = 6
marker_size = 18
font_size = "xx-large"

color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))

fig, ax = plt.subplots()

# Draw the 1st BZ boundary
boundary = Ks_CELL[[0, 1, 2, 3, 4, 5, 0]]
ax.plot(boundary[:, 0], boundary[:, 1], color=colors[0], lw=linewidth)
boundary = Ks_STAR[[0, 1, 2, 3, 4, 5, 0]]
ax.plot(boundary[:, 0], boundary[:, 1], color=colors[1], lw=2*linewidth/3)

# Draw the Gamma point
ax.plot(
    Gamma[0], Gamma[1], marker='o', color=colors[2], markersize=marker_size,
)

# Draw all M-points
ax.plot(
    Ms_CELL[:, 0], Ms_CELL[:, 1],
    marker="o", ls="", color=colors[3], markersize=marker_size,
)
ax.plot(
    Ms_STAR[:, 0], Ms_STAR[:, 1],
    marker="o", ls="", color=colors[4], markersize=2*marker_size/3,
)

# Draw all K-points
ax.plot(
    Ks_CELL[:, 0], Ks_CELL[:, 1],
    marker='o', ls="", color=colors[5], markersize=marker_size,
)
ax.plot(
    Ks_STAR[:, 0], Ks_STAR[:, 1],
    marker='o', ls="", color=colors[6], markersize=2*marker_size/3,
)

title = "1st BZ correspond to the David-Star\n" \
        "superlattice and the original triangular lattice"
ax.set_title(title, fontsize=font_size)
ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig("demo/FirstBrillouinZone.jpg", dpi=1000)
plt.close("all")
