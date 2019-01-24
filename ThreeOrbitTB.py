"""
Solving and demonstrating the three orbit tight-binding model Hamiltonian
proposed by N. V. Smith et al.

See also:
N. V. Smith, S. D. Kevan, and F. J. DiSalvo, J. Phys. C 18, 3175 (1985).
K. Rossnagel, and N. V. Smith, Phys. Rev. B 73, 073106 (2006).
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import Gamma, Ms_CELL, Ks_CELL, KPath


# Model Parameters
dd_sigma = -0.746
dd_pi = 0.24068
dd_delta = 0.10912
d_z2 = 6.39195
d_xy = 6.33073
mu = 5.10622


def Unreconstructed():
    """
    The unreconstructed energy band
    """

    GMKPath, xticks = KPath([Gamma, Ms_CELL[0], Ks_CELL[0]], min_num=200)
    xtick_labels = [r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"]

    xs = GMKPath[:, 0] / 2
    ys = GMKPath[:, 1] * np.sqrt(0.75)
    coeff0 = np.cos(xs) * np.cos(ys)
    coeff1 = np.sin(xs) * np.sin(ys)
    coeff2 = np.cos(2 * xs)

    HM = np.zeros((GMKPath.shape[0], 3, 3), dtype=np.complex128)
    HM[:, 0, 0] = d_z2 + 0.5 * (dd_sigma + 3 * dd_delta) * (2 * coeff0 + coeff2)

    tmp = d_xy + 0.25 * (3 * dd_sigma + 12 * dd_pi + dd_delta) * coeff0
    tmp += 0.5 * (3 * dd_sigma + dd_delta) * coeff2
    HM[:, 1, 1] = tmp

    tmp = d_xy + 0.25 * (9 * dd_sigma + 4 * dd_pi + 3 * dd_delta) * coeff0
    tmp += 2 * dd_pi * coeff2
    HM[:, 2, 2] = tmp

    tmp = np.sqrt(0.75) * (dd_sigma - dd_delta) * (coeff0 - coeff2)
    HM[:, 0, 1] = HM[:, 1, 0] = tmp

    HM[:, 0, 2] = HM[:, 2, 0] = 1.5 * (dd_sigma - dd_delta) * coeff1

    tmp = 0.25 * np.sqrt(3) * (3 * dd_sigma - 4 * dd_pi + dd_delta) * coeff1
    HM[:, 1, 2] = HM[:, 2, 1] = tmp
    del tmp

    Es, Vecs = np.linalg.eigh(HM)
    Es -= mu
    amplitudes = (Vecs * Vecs.conj()).real

    line_width = 6
    font_size = "xx-large"

    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 1])

    ax0.plot(Es, lw=line_width)
    ax0.set_xlim(0, Es.shape[0])
    ax0.set_ylabel(r"$(E-E_F)/eV$")
    ax0.axhline(y=0, ls="dashed", color="gray", lw=line_width/2)
    ax0.annotate(
        r"$E_F$", (0, 0), ha="left", va="bottom",
        fontsize=font_size, color="gray"
    )
    ax0.set_title(
        "Energy band of the unreconstructed system", fontsize=font_size
    )
    ax0.set_xticks(xticks)
    ax0.set_xticklabels(xtick_labels, fontsize=font_size)
    ax0.grid(axis="x", ls="dashed")

    for index, ax in enumerate([ax1, ax2, ax3]):
        CS = ax.contourf(amplitudes[:, :, index].T, levels=100, cmap="rainbow")
        ax.set_title("Bond Index = {0}".format(index + 1))
        ax.set_yticks(range(3))
        ax.set_xticks(xticks)
        if index == 2:
            ax.set_xticklabels(xtick_labels, fontsize=font_size)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.grid(axis="both", ls="dashed")
        fig.colorbar(CS, ax=ax, format="%.2f")
    plt.show()


if __name__ == "__main__":
    Unreconstructed()