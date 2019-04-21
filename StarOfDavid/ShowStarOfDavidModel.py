"""
A demonstration of our Star-of-David tight-binding model

See also the document of the `StarOfDavid` package for the definition of the
Star-Of-David tight-binding model
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from StarOfDavid import BONDS_INTER, BONDS_INTRA
from TaS2DataBase import As_STAR, COORDS_STAR

color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))


# For demonstration
shrink_inner, shrink_outer = 0.2, 0.1

# No shrink
# shrink_inner, shrink_outer = 0.00, 0.00

# Actual shrinkage after the reconstruction
# shrink_inner, shrink_outer = 0.064, 0.042783

coords_star = np.array(COORDS_STAR, copy=True)
coords_star[1:7, :] *= (1 - shrink_inner)
coords_star[7:, :] *= (1 - shrink_outer)

# hopping_term = r"$c_{i\sigma}^{\dag}c_{j\sigma}$"
# mu_term = r"$c_{i\sigma}^{\dag}c_{i\sigma}$"
# hopping_labels = ["$t_{0}$".format(i) + hopping_term for i in range(6)]
# mu_labels = [
#     "A, $\mu_0$" + mu_term,
#     "B, $\mu_1$" + mu_term,
#     "C, $\mu_2$" + mu_term,
# ]
hopping_labels = ["$t_{0}$".format(i) for i in range(6)]
mu_labels = ["A, $\mu_0$", "B, $\mu_1$", "C, $\mu_2$"]
log_template = "Bond: (p0=(({0:>2d}, {1:>2d}), {2:>2d}), " \
               "p1=(({3:>2d}, {4:>2d}), {5:>2d})), Length = {6:.8f}"

legend_handles = []
fig, ax = plt.subplots()
# Plot the intra-cluster bonds
for bonds, color in zip(BONDS_INTRA, COLORS[3:6]):
    for (v0, index0), (v1, index1) in bonds:
        start = coords_star[index0]
        end = coords_star[index1]
        length = np.linalg.norm(end - start)
        bond = np.array([start, end]) + np.dot([0, 0], As_STAR)
        line, = ax.plot(bond[:, 0], bond[:, 1], color=color)
        print(log_template.format(*v0, index0, *v1, index1, length))

        # The equivalent bond in the adjacent cluster
        bond = np.array([start, end]) + np.dot([1, 0], As_STAR)
        ax.plot(bond[:, 0], bond[:, 1], color=color)
    print("=" * 80, flush=True)
    # Save the representative bond of this type
    legend_handles.append(line)

# Plot the inter-cluster bonds
for bonds, color in zip(BONDS_INTER, COLORS[6:9]):
    for (v0, index0), (v1, index1) in bonds:
        if v1 == (1, 0):
            p0 = np.dot(v0, As_STAR) + coords_star[index0]
            p1 = np.dot(v1, As_STAR) + coords_star[index1]
            length = np.linalg.norm(p0 - p1)
            line, = ax.plot(
                (p0[0], p1[0]), (p0[1], p1[1]),
                color=color, ls="dotted",
            )
            print(log_template.format(*v0, index0, *v1, index1, length))
    print("=" * 80, flush=True)
    # Save the representative bond of this type
    legend_handles.append(line)

# Plot the different types of Ta atoms
for translation in [(0, 0), (1, 0)]:
    points = coords_star + np.dot(translation, As_STAR)
    p0, = ax.plot(
        points[0, 0], points[0, 1],
        color=COLORS[0], marker="o", ls="",
    )
    p1, = ax.plot(
        points[1:7, 0], points[1:7, 1],
        color=COLORS[1], marker="o", ls="",
    )
    p2, = ax.plot(
        points[7:, 0], points[7:, 1],
        color=COLORS[2], marker="o", ls="",
    )

legend_handles  += [p0, p1, p2]
legend_labels = hopping_labels + mu_labels
ax.legend(
    legend_handles, legend_labels, loc="best",
    ncol=3, markerscale=0.5, framealpha=0.5, shadow=True,
    borderpad=0.1, borderaxespad=0.1,
    labelspacing=0.0, handlelength=1.2,
    handletextpad=0.2, columnspacing=0.2,
)

# ax.text(
#     1, 1, "(a)", ha="right", va="top",
#     transform=ax.transAxes, fontsize=FONT_SIZE
# )

fig_path = Path("demo/")
fig_path.mkdir(parents=True, exist_ok=True)
fig_name = "Star-of-David Tight-Binding Model.jpg"
ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig(fig_path / fig_name)
plt.close("all")
