"""
A demonstration of our Star-of-David tight-binding model

The Star-of-David tight-binding model was proposed to simulate the band
structure of the commensurate charge-density-wave(CCDW) phase of 1T-TaS2. At
low temperatures, 1T-TaS2 undergoes a commensurate reconstruction into a
Star-of-David unit cell containing 13 Ta atoms. There are three inequivalent
types of Ta atoms in the Star-of-David unit cell designated as "A", "B",
and "C". In this script, the three types of Ta atoms are represented by three
different COLORS: "blue", "orange" and "green". The Star-of-David unit cell
are centered on atoms of type "A" and characterized by a shrinkage of the
"AB", "BC" and "AC" inter-atomic distances by 6.4, 3.2 and 4.28%,
respectively. The super-lattice has 6-fold rotation symmetry.

After the reconstruction, the original nearest-neighbor bonds are of
different length, so does the hopping amplitude. There are two kinds of bond
length within the unit cells and three kinds between the adjacent unit cells.
We also treat the center Ta atoms specially, so hopping on the "AB" bonds are
taken to be different from that on the "BB" bonds. The hopping amplitudes for
six types of bonds are t0, t1, t2, t3, t4, t5 respectively. See also the
`TaS2DataBase` module for the definition of the different types of bond.
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import As_STAR, COORDS_STAR, BONDS_INTRA, BONDS_INTER
from TaS2DataBase import LINE_WIDTH, MARKER_SIZE, FONT_SIZE, COLORS, FIG_DPI


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
        line, = ax.plot(bond[:, 0], bond[:, 1], color=color, lw=LINE_WIDTH)
        print(log_template.format(*v0, index0, *v1, index1, length))

        # The equivalent bond in the adjacent cluster
        bond = np.array([start, end]) + np.dot([1, 0], As_STAR)
        ax.plot(bond[:, 0], bond[:, 1], color=color, lw=LINE_WIDTH)
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
                color=color, lw=LINE_WIDTH, ls="dotted",
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
        color=COLORS[0], marker="o", ls="", ms=MARKER_SIZE,
    )
    p1, = ax.plot(
        points[1:7, 0], points[1:7, 1],
        color=COLORS[1], marker="o", ls="", ms=MARKER_SIZE,
    )
    p2, = ax.plot(
        points[7:, 0], points[7:, 1],
        color=COLORS[2], marker="o", ls="", ms=MARKER_SIZE,
    )

legend_handles  += [p0, p1, p2]
legend_labels = hopping_labels + mu_labels
ax.legend(
    legend_handles, legend_labels, loc="best", fontsize=FONT_SIZE,
    ncol=3, markerscale=0.5, framealpha=0.5, shadow=True,
    borderpad=0.1, borderaxespad=0.1,
    labelspacing=0.0, handlelength=1.2,
    handletextpad=0.2, columnspacing=0.2,
)

# ax.set_title("Star-of-David Tight-Binding Model", fontsize=FONT_SIZE)
# ax.text(
#     1, 1, "(a)", ha="right", va="top",
#     transform=ax.transAxes, fontsize=FONT_SIZE
# )

ax.set_aspect("equal")
ax.set_axis_off()
plt.show()
fig.savefig("demo/Star-of-David Tight-Binding Model.jpg", dpi=FIG_DPI)
plt.close("all")
