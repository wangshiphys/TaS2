"""
Show the First Brillouin Zone correspond to the Star-of-David super-lattice
and the original triangular lattice
"""


import matplotlib.pyplot as plt

from TaS2DataBase import Gamma, Ms_CELL, Ms_STAR, Ks_CELL, Ks_STAR


color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))

# The ratio of the line-width and marker-size between the two Brillouin Zone
ratio = 0.6

fig, ax = plt.subplots()

# Draw the 1st BZ boundary
boundary = Ks_CELL[[0, 1, 2, 3, 4, 5, 0]]
ax.plot(boundary[:, 0], boundary[:, 1], color=COLORS[0])
boundary = Ks_STAR[[0, 1, 2, 3, 4, 5, 0]]
ax.plot(boundary[:, 0], boundary[:, 1], color=COLORS[1])

# Draw the Gamma point
ax.plot(Gamma[0], Gamma[1], marker='o', color=COLORS[2])

# Draw all M-points
ax.plot(
    Ms_CELL[:, 0], Ms_CELL[:, 1],
    marker="o", ls="", color=COLORS[3],
)
ax.plot(
    Ms_STAR[:, 0], Ms_STAR[:, 1],
    marker="o", ls="", color=COLORS[4],
)

# Draw all K-points
ax.plot(
    Ks_CELL[:, 0], Ks_CELL[:, 1],
    marker='o', ls="", color=COLORS[5],
)
ax.plot(
    Ks_STAR[:, 0], Ks_STAR[:, 1],
    marker='o', ls="", color=COLORS[6],
)

ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig("demo/FirstBrillouinZone.jpg")
plt.close("all")
