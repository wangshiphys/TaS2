"""
A demonstration of our David-Star tight-binding model

The David-Star tight-binding model was proposed to simulate the properties of
the commensurate charge-density-wave(CCDW) phase of 1T-TaS2. At low
temperatures, 1T-TaS2 undergoes a commensurate reconstruction into a
David-Star unit cell containing 13 Ta atoms. There are three inequivalent
types of Ta atoms in the David-Star designated as "a", "b" and "c"(In this
script, the three types of Ta atoms are represented by three different
colors: "red", "orange" and "green"). The David-Stars are centered on atoms
of type "a" and characterized by a shrinkage of the ab, bc and ac
inter-atomic distances by 6.4, 3.2 and 4.28%, respectively.

After the reconstruction, the original nearest-neighbor bonds are of
different length, so does the hopping coefficients. There are two kinds of
bond length within the David-Stars and three kinds between the
nearest-neighbor David-Stars. We also treat the center Ta atoms specially,
so hoppings on the "ab" bonds are taken to be different from that on the
"bb" bonds. The hopping coefficients for the six types of bonds are t0, t1,
t2, t3, t4, t5 respectively. See also the `TaS2DataBase` module for the
definition of the different types of bond.
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import  As_STAR, COORDS_STAR, ALL_BONDS


line_width = 2
marker_size = 25
font_size = "xx-large"
color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))[[0, 1, 2, 3, -2, -1]]

shrink_inner, shrink_outer = 0.2, 0.1
# shrink_inner, shrink_outer = 0.00, 0.00
# shrink_inner, shrink_outer = 0.064, 0.042783

coords_star = np.array(COORDS_STAR, copy=True)
coords_star[1:7, :] *= (1 - shrink_inner)
coords_star[7:, :] *= (1 - shrink_outer)

legend_handles = []
hopping = r"$c_{i\sigma}^{\dag}c_{j\sigma}$"
legend_labels = ["$t_{0}$".format(i) + hopping for i in range(6)]
line_styles = ["solid"] * 3 + ["dotted"] * 3
log_template = "Bond: (p0=(({0:>2d}, {1:>2d}), {2:>2d}), " \
               "p1=(({3:>2d}, {4:>2d}), {5:>2d})), Length = {6:.8f}"

fig, ax = plt.subplots()
for bonds, line_style, color in zip(ALL_BONDS, line_styles, colors):
    for (v0, index0), (v1, index1) in bonds:
        p0 = np.dot(v0, As_STAR) + coords_star[index0]
        p1 = np.dot(v1, As_STAR) + coords_star[index1]
        length = np.linalg.norm(p0 - p1)
        line, = ax.plot(
            (p0[0], p1[0]), (p0[1], p1[1]),
            color=color, lw=line_width, ls=line_style,
        )
        print(log_template.format(*v0, index0, *v1, index1, length))
    print("=" * 80, flush=True)
    legend_handles.append(line)
ax.legend(
    legend_handles, legend_labels, loc="upper right", fontsize="x-large",
)

# Annotate the points with their indices
for translation in [(0, 0), (1, 0), (0, 1), (-1, 1)]:
    points = coords_star + np.dot(translation, As_STAR)
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
    for i in range(13):
        ax.annotate(
            "{0}".format(i), (points[i, 0], points[i, 1]),
            ha="center", va="center", fontsize=font_size,
        )

ax.set_title(
    "Demonstration of the David-Star Tight-Binding Model", fontsize=font_size
)
ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig("demo/David-Star Tight-Binding Model.jpg", dpi=1000)
plt.close("all")
