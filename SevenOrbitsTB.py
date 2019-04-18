"""
Solving and demonstrating the Seven-Orbits Tight-Binding model Hamiltonian
proposed by Shuang Qiao et al.

See also:
    Shuang Qiao et al., Phys. Rev. X 7, 041054 (2017)
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import *


color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))

ONSITE_TERMS = [
    [(0, 0)],
    [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
    [(1, 4), (2, 5), (3, 6)],
    [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)],
    [(1, 3), (2, 4), (3, 5), (4, 6), (5, 1), (6, 2)],
]
INTER_TERMS_A = [(3, 6), (1, 4), (3, 1), (4, 6)]
INTER_TERMS_B = [(2, 5), (6, 3), (2, 6), (3, 5)]
INTER_TERMS_C = [(1, 4), (5, 2), (1, 5), (2, 4)]
INTER_TERMS = {
    0: INTER_TERMS_B,
    1: INTER_TERMS_C,
    2: INTER_TERMS_A,
}


class SevenOrbitTBSolver(BaseTBSolver):
    """
    Construct and solve the Seven-Orbits Tight-Binding model Hamiltonian
    proposed by Shuang Qiao et al.
    """

    default_model_params = {
        "mu_c": 0.0,
        "t_sc": 1.0,
        "t_ss1": 1.0,
        "t_ss2": 1.0,
        "t_ss3": 1.0,
        "t_ss4": 1.0,
        "t_ss5": 1.0,
        "t_ss6": 1.0,
    }

    def __init__(self, e_num=13, min_num=100, numkx=500, numky=None):
        """
        See also:
            BaseTBSolver
        """

        super().__init__(
            orbit_num=7, e_num=e_num, min_num=min_num, numkx=numkx, numky=numky
        )

    # Construct the Seven-Orbits Tight-Binding model Hamiltonian in the k-space
    # Diagonalize the model Hamiltonian
    def _EnergyCore(self, exponents, *, return_vectors=True, **model_params):
        new_model_params = dict(SevenOrbitTBSolver.default_model_params)
        new_model_params.update(model_params)
        onsite_term_coeffs = [
            new_model_params["mu_c"] / 2, new_model_params["t_sc"],
            new_model_params["t_ss1"], new_model_params["t_ss3"],
            new_model_params["t_ss5"],
        ]
        inter_term_coeffs = [
            new_model_params["t_ss2"], new_model_params["t_ss4"],
            new_model_params["t_ss6"], new_model_params["t_ss6"],
        ]

        shape = (exponents.shape[0], self._orbit_num, self._orbit_num)
        HM = np.zeros(shape, dtype=np.complex128)

        for coeff, terms in zip(onsite_term_coeffs, ONSITE_TERMS):
            for row, col in terms:
                HM[:, row, col] += coeff
        for index, terms in INTER_TERMS.items():
            for coeff, (row, col) in zip(inter_term_coeffs, terms):
                HM[:, row, col] += coeff * exponents[:, index]
        HM += np.transpose(HM.conj(), axes=(0, 2, 1))

        if return_vectors:
            return np.linalg.eigh(HM)
        else:
            return np.linalg.eigvalsh(HM)

    def VisualizeDOS(self, gamma=0.01, **model_params):
        """
        Plot the orbital projected densities of states for the two types of
        orbits

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            Default: 0.01
        model_params : other key-word arguments
            Specifying the Seven-Orbits Tight-Binding model parameters
            The recognizable key-word arguments are:
                mu_c, t_sc, t_ss1, t_ss2, t_ss3, t_ss4, t_ss5, t_ss6
        """

        omegas, projected_dos = self.DOS(gamma=gamma, **model_params)
        global_dos = np.sum(projected_dos, axis=-1)

        fig, axes = plt.subplots(1, 3, sharex=True)
        line0, = axes[0].plot(omegas, global_dos, color=COLORS[0])
        line1, = axes[0].plot(omegas, projected_dos[:, 0], color=COLORS[1])
        line2, = axes[0].plot(
            omegas, np.sum(projected_dos[:, 1:], axis=-1), color=COLORS[2]
        )
        axes[1].plot(omegas, projected_dos[:, 0], color=COLORS[1])
        axes[2].plot(omegas, projected_dos[:, 1], color=COLORS[2])
        axes[0].set_xlim(omegas[0], omegas[-1])
        axes[0].legend(
            [line0, line1, line2], ["DOS", "DOS-C", "DOS-S"], loc="best"
        )
        plt.show()
        plt.close("all")

    def __call__(self, gamma=0.01, **model_params):
        """
        The main entrance of instance of this class

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            Default: 0.01
        model_params : other key-word arguments
            Specifying the Seven-Orbits Tight-Binding model parameters
            The recognizable key-word arguments are:
                mu_c, t_sc, t_ss1, t_ss2, t_ss3, t_ss4, t_ss5, t_ss6
        """

        GE, mu, avg_electron_nums = self.AverageElectronNumber(**model_params)
        omegas, projected_dos = self.DOS(gamma=gamma, **model_params)
        global_dos = np.sum(projected_dos, axis=-1)

        msg0 = "The averaged number of electron per unit-cell: {0}"
        msg1 = "The averaged number of electron on the {0}th orbit: {1}"
        print("The chemical potential: {0}".format(mu))
        print("The ground state energy per Star-of-David: {0}".format(GE))
        print(msg0.format(np.sum(avg_electron_nums)))
        for index, avg_electron_num in enumerate(avg_electron_nums):
            print(msg1.format(index, avg_electron_num))

        fig, axes = plt.subplots(1, 4, sharey=True)
        axes[0].plot(self._GMKGPathEs)
        axes[0].set_xlim(0, len(self._GMKGPathEs) - 1)
        axes[0].set_ylim(omegas[0], omegas[-1])
        axes[0].set_xticks(self._GMKGIndices)
        axes[0].set_xticklabels(self._GMKGLabels)
        axes[0].set_ylabel(r"$E$", rotation="horizontal")
        axes[0].grid(axis="x", ls="dashed")

        axes[1].plot(global_dos, omegas)
        axes[2].plot(projected_dos[:, 0], omegas)
        axes[3].plot(projected_dos[:, 1], omegas)

        for ax, label in zip(axes, ["EB", "DOS", "DOS-C", "DOS-S"]):
            ax.set_xlabel(label)
            ax.axhline(mu, ls="dashed", color="gray")
        axes[2].text(
            0.5, 0.55, "$E_F={0:.3f}$".format(mu),
            ha="center", va="center", transform=axes[2].transAxes
        )
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    model_params = {
        "mu_c": 0.212,
        "t_sc": 0.162,
        "t_ss1": 0.150,
        "t_ss2": 0.091,
        "t_ss3": 0.072,
        "t_ss4": 0.050,
        "t_ss5": 0.042,
        "t_ss6": 0.042,
    }

    Solver = SevenOrbitTBSolver(e_num=13, numkx=200, numky=200)
    Solver.VisualizeDOS(**model_params)
    Solver()
    Solver(**model_params)
