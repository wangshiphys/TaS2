"""
Construct and solve the Star-of-David Tight-Binding model Hamiltonian

For the definition of the Star-of-David model, see the docstring of the
`ShowStarOfDavidModel.py` script.
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import *


LINE_WIDTH = 3
color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))


class StarOfDavidTBSolver(BaseTBSolver):
    """
    Construct and solve the Star-of-David Tight-Binding model Hamiltonian
    """

    default_model_params = {
        "t0": 1.0,
        "t1": 1.0,
        "t2": 1.0,
        "t3": 1.0,
        "t4": 1.0,
        "t5": 1.0,
        "mu0": 0.0,
        "mu1": 0.0,
        "mu2": 0.0,
    }

    def __init__(self, e_num=13, min_num=100, numkx=500, numky=None):
        """
        See also:
            BaseTBSolver
        """

        super().__init__(
            orbit_num=13, e_num=e_num, min_num=min_num, numkx=numkx, numky=numky
        )

    # Construct the Star-of-David Tight-Binding model Hamiltonian in the k-space
    # Diagonalize the model Hamiltonian
    def _EnergyCore(self, exponents, *, return_vectors=True, **model_params):
        # In the following explanation, the coordinates of the SDs are
        # specified with respect to the translation vectors in the real space.
        # The displace from SD (1, 0) to SD (0, 0) is R0 = Rs_Star[0];
        # The displace from SD (0, 1) to SD (0, 0) is R1 = Rs_Star[1];
        # The displace from SD (-1, 1) to SD (0, 0) is R2 = Rs_Star[2];
        # The corresponding phase factors after Fourier transformation is
        # k * R0, k * R1, k * R2, respectively.
        # The three columns in `exponents` correspond to
        # exp(1j*k*R0), exp(1j*k*R1), exp(1j*k*R2), respectively.
        # Every row in `exponents` corresponds to a k-point.
        defaults = StarOfDavidTBSolver.default_model_params
        t0 = model_params.get("t0", defaults["t0"])
        t1 = model_params.get("t1", defaults["t1"])
        t2 = model_params.get("t2", defaults["t2"])
        t3 = model_params.get("t3", defaults["t3"])
        t4 = model_params.get("t4", defaults["t4"])
        t5 = model_params.get("t5", defaults["t5"])
        mu0 = model_params.get("mu0", defaults["mu0"])
        mu1 = model_params.get("mu1", defaults["mu1"])
        mu2 = model_params.get("mu2", defaults["mu2"])

        table = {(1, 0): 0, (0, 1): 1, (-1, 1): 2}
        all_points = [POINT_TYPE_A, POINT_TYPE_B, POINT_TYPE_C]
        shape = (exponents.shape[0], self._orbit_num, self._orbit_num)
        H = np.zeros(shape, dtype=np.complex128)

        # On-site chemical potential terms
        for points, mu in zip(all_points, [mu0, mu1, mu2]):
            for p in points:
                # The diagonal term will be doubled after adding the
                # Hermitian conjugate
                H[:, p, p] += mu / 2

        # Intra-Cluster hopping terms
        for bonds, hopping in zip(BONDS_INTRA, [t0, t1, t2]):
            for (star_index0, p0), (star_index1, p1) in bonds:
                H[:, p0, p1] -= hopping

        # Inter-Cluster hopping terms
        for bonds, hopping in zip(BONDS_INTER, [t3, t4, t5]):
            for (star_index0, p0), (star_index1, p1) in bonds:
                H[:, p0, p1] -= hopping * exponents[:, table[star_index1]]
        H += np.transpose(H.conj(), axes=(0, 2, 1))

        if return_vectors:
            return np.linalg.eigh(H)
        else:
            return np.linalg.eigvalsh(H)

    def Verify(self, mu=0.0):
        """
        Verify that this tight-binding model is equivalent to the
        simple nearest-neighbor tight-binding model defined on the triangular
        lattice when all the hopping parameters and the on-site chemical
        potentials are set to be identical

        See also the `TriangleSimpleTB.py` file for the definition of the
        simple nearest-neighbor tight-binding model

        Parameters
        ----------
        mu : float, optional
            On-site chemical potential
            Default: 0.0
        """

        GMKGLabels = (r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$")
        GMKGPath, GMKGIndices = KPath(
            [Gamma, Ms_CELL[0], Ks_CELL[0]], loop=True
        )
        Es_Cell = mu - 2 * np.sum(
            np.cos(np.matmul(GMKGPath, Rs_CELL.T)), axis=-1
        )
        Es_Star = self._EnergyCore(
            np.exp(1j * np.matmul(GMKGPath, Rs_STAR.T)),
            t0=1.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0, t5=1.0,
            mu0=mu, mu1=mu, mu2=mu, return_vectors=False
        )

        fig, ax = plt.subplots()
        ax.plot(Es_Star, lw=LINE_WIDTH, alpha=0.8)
        ax.plot(Es_Cell, lw=LINE_WIDTH/3, ls="dashed", color="black")

        ax.set_xlim(0, len(Es_Cell) - 1)
        ax.set_xticks(GMKGIndices)
        ax.set_xticklabels(GMKGLabels)
        ax.set_ylabel(r"$E$", rotation="horizontal")
        ax.grid(axis="both", linestyle="dashed")
        plt.show()
        plt.close("all")

    def VisualizeDOS(self, gamma=0.01, **model_params):
        """
        Plot the atomic projected densities of states for the three
        inequivalent Ta atoms as well as the global densities

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            Default: 0.01
        model_params : other key-word arguments
            Specifying the Star-of-David tight-binding model parameters
            The recognizable key-word arguments are:
                t0, t1, t2, t3, t4, t5, m0, mu1, mu2
        """

        omegas, projected_dos = self.DOS(gamma=gamma, **model_params)
        global_dos = np.sum(projected_dos, axis=-1)

        fig, axes = plt.subplots(2, 2, sharex=True)
        line0, = axes[0, 0].plot(
            omegas, global_dos, lw=LINE_WIDTH, color=COLORS[0]
        )
        line1, = axes[0, 0].plot(
            omegas, projected_dos[:, 0], lw=LINE_WIDTH, color=COLORS[1]
        )
        line2, = axes[0, 0].plot(
            omegas, np.sum(projected_dos[:, 1:7], axis=-1),
            lw=LINE_WIDTH, color=COLORS[2]
        )
        line3, = axes[0, 0].plot(
            omegas, np.sum(projected_dos[:, 7:], axis=-1),
            lw=LINE_WIDTH, color=COLORS[3]
        )
        axes[0, 1].plot(
            omegas, projected_dos[:, 0], lw=LINE_WIDTH, color=COLORS[1]
        )
        axes[1, 0].plot(
            omegas, projected_dos[:, 1], lw=LINE_WIDTH, color=COLORS[2]
        )
        axes[1, 1].plot(
            omegas, projected_dos[:, 7], lw=LINE_WIDTH, color=COLORS[3]
        )

        axes[0, 0].set_xlim(omegas[0], omegas[-1])
        axes[0, 0].legend(
            [line0, line1, line2, line3], ["DOS", "LDOS-A", "LDOS-B", "LDOS-C"],
            loc="best"
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
            Specifying the Star-of-David tight-binding model parameters
            The recognizable key-word arguments are:
                t0, t1, t2, t3, t4, t5, m0, mu1, mu2
        """

        GE, mu, avg_electron_nums = self.AverageElectronNumber(**model_params)
        omegas, projected_dos = self.DOS(gamma=gamma, **model_params)
        global_dos = np.sum(projected_dos, axis=-1)

        msg0 = "The averaged number of electron per Star-of-David: {0}"
        msg1 = "The averaged number of electron on the {0:2d}th Ta atom: {1}"
        print("The chemical potential: {0}".format(mu))
        print("The ground state energy per Star-of-David: {0}".format(GE))
        print(msg0.format(np.sum(avg_electron_nums)))
        for index, avg_electron_num in enumerate(avg_electron_nums):
            print(msg1.format(index, avg_electron_num))

        fig, axes = plt.subplots(1, 5, sharey=True)
        axes[0].plot(self._GMKGPathEs, lw=LINE_WIDTH)
        axes[0].set_xlim(0, len(self._GMKGPathEs) - 1)
        axes[0].set_ylim(omegas[0], omegas[-1])
        axes[0].set_xticks(self._GMKGIndices)
        axes[0].set_xticklabels(self._GMKGLabels)
        axes[0].set_ylabel(r"$E$", rotation="horizontal")
        axes[0].grid(axis="x", ls="dashed", lw=LINE_WIDTH/4)

        axes[1].plot(global_dos, omegas, lw=LINE_WIDTH)
        axes[2].plot(projected_dos[:, 0], omegas, lw=LINE_WIDTH)
        axes[3].plot(projected_dos[:, 1], omegas, lw=LINE_WIDTH)
        axes[4].plot(projected_dos[:, 7], omegas, lw=LINE_WIDTH)

        for ax, label in zip(axes, ["EB", "DOS", "LDOS-A", "LDOS-B", "LDOS-C"]):
            ax.axhline(mu, ls="dashed", lw=LINE_WIDTH/2, color="gray")
            ax.set_xlabel(label)
        # axes[2].text(
        #     0.5, 0.55, "$E_F={0:.3f}$".format(mu),
        #     ha="center", va="center",
        #     transform=axes[2].transAxes,
        # )

        plt.show()
        plt.close("all")


if __name__ == "__main__":
    model_params = {
        "t0": 0.2,
        "t1": 0.2,
        "t2": 0.8,
        "t3": 1.0,
        "t4": 1.0,
        "t5": 1.0,
        "mu0": 0.0,
        "mu1": -0.6,
        "mu2": -0.1,
    }
    Solver = StarOfDavidTBSolver(e_num=6, numkx=1)
    Solver.Verify()
    Solver.VisualizeDOS(**model_params)
    Solver()
    Solver(**model_params)
    Solver(**model_params)
