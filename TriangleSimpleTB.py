"""
This script demonstrate the basic aspects of the simple nearest-neighbor
tight-binding model defined on the triangular lattice, including the First
Brillouin Zone, the energy band along the Gamma-M-K-Gamma path, the density
of states and the Fermi surface for specified filling

The simple nearest-neighbor tight-binding model on the triangular lattice is
defined as follow:
    Only nearest-neighbor hopping is considered;
    Every lattice site has two single particle states: spin-up and spin-down;
    No spin-flipping hopping is involved;
The resulting dispersion relation is:
    E(k) = -2t * (cos(k*r1) + cos(k*r2) + cos(k*r3))
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import Bs_CELL, Rs_CELL, Gamma, Ms_CELL, Ks_CELL, KPath


# Averaged number of electron per lattice site
e_num = 1.0
# The number of k-points in the MK segment
min_numk = 5000
# The number of k-points along b1 and b2
numkx = numky = 10000
# Total number of electron in the system, it must be an integer
total_electron_num = numkx * numky * e_num
assert total_electron_num == int(total_electron_num)
total_electron_num = int(total_electron_num)
# For spin degenerated system, if total electron number is even, then `kth`
# k-points are occupied; if total electron number is odd, the `kth+1`
# k-points are occupied and the last k-point is half filled
kth = total_electron_num // 2


linewidth = 6
fontsize = "xx-large"
color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))


# Generate the k-points along the Gamma-M-K-Gamma path
GMKPath, xticks = KPath(
    [Gamma, Ms_CELL[0], Ks_CELL[0]], min_num=min_numk
)
# Calculate the energies along the Gamma-M-K-Gamma path
GMKPath_Es = -2 * np.sum(np.cos(np.matmul(GMKPath, Rs_CELL.T)), axis=-1)
################################################################################


# Calculate the E(k)s for a k-mesh in the equivalent first Brillouin Zone
ratio_kx = np.linspace(0, 1, numkx, endpoint=False)
ratio_ky = np.linspace(0, 1, numky, endpoint=False)
ratio_mesh = np.stack(
    np.meshgrid(ratio_kx, ratio_ky, indexing="ij"), axis=-1
)
BZMesh_Es = -2 * np.sum(
    np.cos(np.matmul(ratio_mesh, np.matmul(Bs_CELL, Rs_CELL.T))), axis=-1
)
del ratio_kx, ratio_ky, ratio_mesh

# Calculate the chemical potential and the Density of States
BZMesh_Es = np.partition(BZMesh_Es, kth=[kth-1, kth], axis=None)
if total_electron_num % 2:
    mu = BZMesh_Es[kth]
else:
    mu = (BZMesh_Es[kth-1] + BZMesh_Es[kth]) / 2
print("The chemical potential mu = {0}".format(mu))

step = 1e-2
omega_min = np.floor(np.min(BZMesh_Es)) - 0.2
omega_max = np.ceil(np.max(BZMesh_Es)) + 0.2
hist, bins = np.histogram(BZMesh_Es, np.arange(omega_min, omega_max, step))
DoS = hist / (numkx * numky * step)
omegas = bins[1:] - step / 2
del BZMesh_Es, step, hist, bins
################################################################################


# Plot the Fermi surface
length_GM = np.linalg.norm(Ms_CELL[0])
length_GK = np.linalg.norm(Ks_CELL[0])
vectors = np.array([[length_GK, 0], [0, length_GM]])
ratio_kx = np.linspace(
    -1.1, 1.1, int(min_numk * length_GK / length_GM)
)
ratio_ky = np.linspace(-1.1, 1.1, min_numk)
ratio_mesh = np.stack(
    np.meshgrid(ratio_kx, ratio_ky, indexing="ij"), axis=-1
)
kpoints = np.matmul(ratio_mesh, vectors)
BZMesh_Es = -2 * np.sum(np.cos(np.matmul(kpoints, Rs_CELL.T)), axis=-1)
del length_GM, length_GK, vectors, ratio_kx, ratio_ky, ratio_mesh


fig, axes = plt.subplots(2, 2)

# Plot the boundary of the First Brillouin Zone
boundary = Ks_CELL[[0, 1, 2, 3, 4, 5, 0]]
axes[0, 0].plot(
    boundary[:, 0], boundary[:, 1], color=colors[0], lw=linewidth,
    marker="o", ms=15, mec=colors[1], mfc=colors[1]
)
# Plot the Gamma-M-K-Gamma path
axes[0, 0].plot(
    GMKPath[:, 0], GMKPath[:, 1],
    color=colors[3], lw=linewidth/2, ls="dashed"
)
axes[0, 0].set_title(
    "First BZ\n" + r"$\Gamma-M-K-\Gamma$", fontsize=fontsize
)
axes[0, 0].set_aspect("equal")
axes[0, 0].set_axis_off()

axes[0, 1].plot(GMKPath_Es, lw=linewidth)
axes[0, 1].axhline(y=mu, lw=linewidth/3, ls="dashed", color="gray")
axes[0, 1].text(
    0, mu, r"$E_F={0:.3f}$".format(mu),
    color=colors[4], fontsize="x-large", ha="left", va="bottom",
)
axes[0, 1].set_xlim(0, GMKPath_Es.shape[0] - 1)
axes[0, 1].set_xticks(xticks)
axes[0, 1].set_xticklabels(
    [r"$\Gamma$", "$M$", "$K$", r"$\Gamma$"], fontsize=fontsize
)
axes[0, 1].set_ylabel(r"$E/t$", fontsize=fontsize)
axes[0, 1].set_title(r"EB along $\Gamma-M-K-\Gamma$", fontsize=fontsize)
axes[0, 1].grid(axis="x")

axes[1, 0].plot(boundary[:, 0], boundary[:, 1], color=colors[0], lw=linewidth)
axes[1, 0].contour(
    kpoints[:, :, 0], kpoints[:, :, 1], BZMesh_Es,
    levels=[mu], colors="r", linewidths=linewidth/3,
)
axes[1, 0].set_title("Fermi Surface", fontsize=fontsize)
axes[1, 0].set_aspect("equal")
axes[1, 0].set_axis_off()

axes[1, 1].plot(omegas, DoS, lw=linewidth)
axes[1, 1].axvline(x=mu, lw=linewidth/3, ls="dashed", color="gray")
axes[1, 1].text(
    mu, 0, r"$E_F={0:.3f}$".format(mu),
    color=colors[4], fontsize="x-large", ha="left", va="bottom",
)
axes[1, 1].set_xlim(omega_min, omega_max)
axes[1, 1].set_xlabel(r"$\omega/t$", fontsize=fontsize)
axes[1, 1].set_ylabel("DoS", fontsize=fontsize)
axes[1, 1].set_title("Density of States", fontsize=fontsize)

plt.show()
fig.savefig("demo/TriangleSimpleTB.jpg", dpi=1000)
plt.close("all")
