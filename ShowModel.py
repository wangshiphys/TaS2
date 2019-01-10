"""
An illustration of our David-Star tight-binding model
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import  As_STAR, COORDS_STAR


POINT_TYPE_A = (0, )
POINT_TYPE_B = (1, 2, 3, 4, 5, 6)
POINT_TYPE_C = (7, 8, 9, 10, 11, 12)


color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))

alpha = 0.5
line_width = 3
marker_size = 26
line_style = "dashed"
font_size = "xx-large"
hopping = r"$C_{i\sigma}^{\dag}C_{j\sigma}$"
legends = ["$t_{0}$".format(i) + hopping for i in range(5)]

shrink_inner = 0.3
shrink_outer = 0.2
coords_star = np.array(COORDS_STAR, copy=True)
coords_star[1:7, :] *= (1 - shrink_inner)
coords_star[7:, :] *= (1 - shrink_outer)


fig, ax = plt.subplots()
# Bonds between the central point and the points of the inner loop in the
# David-Star
for p0 in POINT_TYPE_A:
    for p1 in POINT_TYPE_B:
        bond = coords_star[[p0, p1]]
        bond_type0, = ax.plot(
            bond[:, 0], bond[:, 1],
            color=colors[0], lw=line_width, ls=line_style, alpha=alpha,
        )

# Bonds between the inner loop in the David-Star
bond = coords_star[[1, 2, 3, 4, 5, 6, 1]]
bond_type1, = ax.plot(
    bond[:, 0], bond[:, 1],
    color=colors[1], lw=line_width, ls=line_style, alpha=alpha,
)

# Bonds between the inner loop and the outer loop in the David-Star
bond = coords_star[[1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12, 1]]
bond_type2, = ax.plot(
    bond[:, 0], bond[:, 1],
    color=colors[2], lw=line_width, ls=line_style, alpha=alpha,
)

# Bonds between points from one David-Star's outer loop and another
# David-Star's outer loop
bond = np.array([coords_star[9], coords_star[12] + As_STAR[0]])
bond_type3, = ax.plot(
    bond[:, 0], bond[:, 1],
    color=colors[3], lw=line_width, ls=line_style, alpha=alpha,
)

# Bonds between points from one David-Star's inner loop and another
# David-Star's outer loop
bond = np.array([coords_star[3], coords_star[12] + As_STAR[0]])
bond_type4, = ax.plot(
    bond[:, 0], bond[:, 1],
    color=colors[4], lw=line_width, ls=line_style, alpha=alpha,
)
ax.legend(
    [bond_type0, bond_type1, bond_type2, bond_type3, bond_type4],
    legends, loc="upper left", fontsize=font_size,
)

# Show the three type points
for points in [coords_star, coords_star + As_STAR[0]]:
    ax.plot(
        points[0, 0], points[0, 1],
        color=colors[3], marker="o", ls="", ms=marker_size,
    )
    ax.plot(
        points[1:7, 0], points[1:7, 1],
        color=colors[1], marker="o", ls="", ms=marker_size,
    )
    ax.plot(
        points[7:, 0], points[7:, 1],
        color=colors[2], marker="o", ls="", ms=marker_size,
    )

# Annotate the points with their indices
for i in range(13):
    ax.annotate(
        "{0}".format(i), (coords_star[i, 0], coords_star[i, 1]),
        ha="center", va="center", fontsize=font_size,
    )

ax.set_title("David-Star Tight-Binding Model", fontsize=font_size)
ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig("David-Star Tight-Binding Model.jpg", dpi=1000)
plt.close("all")
