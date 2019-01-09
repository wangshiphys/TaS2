"""
Nearest-neighbor tight-binding model defined on the triangular lattice
"""


import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import brentq

from TaS2DataBase import Bs_CELL, Rs_CELL, Gamma, Ms_CELL, Ks_CELL, KPath


# Core function for calculating the chemical potential
def Chemical_Potential_Core(mu, Es, filling):
    return 2 * np.count_nonzero(Es < mu) / Es.shape[0] - filling


min_numk = 5000
numkx = numky = 10000


linewidth = 6
fontsize = "xx-large"
color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))


# Generate the k-points along the Gamma-M-K-Gamma path
GMKPath, indices = KPath(
    [Gamma, Ms_CELL[0], Ks_CELL[0]], min_num=min_numk
)
# Calculate the energies along the Gamma-M-K-Gamma path
GMKPath_Es = -2 * np.sum(np.cos(np.dot(GMKPath, Rs_CELL.T)), axis=-1)
################################################################################


# Calculate the E(k)s of a k-mesh in the first Brillouin Zone
ratio_kx = np.linspace(0, 1, numkx, endpoint=False)
ratio_ky = np.linspace(0, 1, numky, endpoint=False)
ratio_mesh = np.stack(
    np.meshgrid(ratio_kx, ratio_ky, indexing="ij"), axis=-1
).reshape((-1, 2))
BZMesh_Es = -2 * np.sum(
    np.cos(np.linalg.multi_dot([ratio_mesh, Bs_CELL, Rs_CELL.T])), axis=-1
)
del ratio_kx, ratio_ky, ratio_mesh

# Calculate the chemical potential and the Density of States
omega_min = np.floor(np.min(BZMesh_Es)) - 0.5
omega_max = np.ceil(np.max(BZMesh_Es)) + 0.5
mu = brentq(
    Chemical_Potential_Core, omega_min, omega_max, args=(BZMesh_Es, 1)
)
step = 1e-2
hist, bins = np.histogram(BZMesh_Es, np.arange(omega_min, omega_max, step))
DoS = hist / (BZMesh_Es.shape[0] * step)
omegas = bins - step / 2
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
Ks = np.append(Ks_CELL, Ks_CELL[[0]], axis=0)
axes[0, 0].plot(
    Ks[:, 0], Ks[:, 1], color=colors[0], lw=linewidth,
    marker="o", ms=15, mec=colors[1], mfc=colors[1]
)
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
axes[0, 1].set_xticks(indices)
axes[0, 1].set_xticklabels(
    [r"$\Gamma$", "$M$", "$K$", r"$\Gamma$"], fontsize=fontsize
)
axes[0, 1].set_ylabel(r"$E/t$", fontsize=fontsize)
axes[0, 1].set_title(r"EB along $\Gamma-M-K-\Gamma$", fontsize=fontsize)
axes[0, 1].grid(axis="x")

axes[1, 0].plot(omegas[1:], DoS, lw=linewidth)
axes[1, 0].axvline(x=mu, lw=linewidth/3, ls="dashed", color="gray")
axes[1, 0].text(
    mu, 0, r"$E_F={0:.3f}$".format(mu),
    color=colors[4], fontsize="x-large", ha="left", va="bottom",
)
axes[1, 0].set_xlim(omega_min, omega_max)
axes[1, 0].set_xlabel(r"$\omega/t$", fontsize=fontsize)
axes[1, 0].set_ylabel("DoS", fontsize=fontsize)
axes[1, 0].set_title("Density of States", fontsize=fontsize)

axes[1, 1].plot(Ks[:, 0], Ks[:, 1], color=colors[0], lw=linewidth)
axes[1, 1].contour(
    kpoints[:, :, 0], kpoints[:, :, 1], BZMesh_Es,
    levels=[mu], colors="r", linewidths=linewidth/3,
)
axes[1, 1].set_title("Fermi Surface", fontsize=fontsize)
axes[1, 1].set_aspect("equal")
axes[1, 1].set_axis_off()

plt.show()
fig.savefig("TriangleSimpleTB.jpg", dpi=1000)
plt.close("all")
