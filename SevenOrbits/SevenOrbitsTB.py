"""
Solving and demonstrating the Seven-Orbits Tight-Binding model Hamiltonian
"""


import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import AoC, CREATION, IndexTable, KPath, lattice_generator

from SevenOrbits.SevenOrbitsModel import *
from TaS2DataBase import Bs_CELL, Gamma, Ms_CELL, Ks_CELL, Lorentzian

color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))


class SevenOrbitsTBASolver:
    """
    Solving and demonstrating the Seven-Orbits Tight-Binding model Hamiltonian
    """

    def __init__(self, e_num=13, min_num=100, numk=500):
        """
        Customize the newly created instance

        Prepare the k-points along the Gamma-M-K-Gamma path and the k-mesh in
        the equivalent First Brillouin Zone.

        Parameters
        ----------
        e_num : int or float, optional
            The averaged number of electrons per unit-cell
            The total number of electrons in the system is:
                numk * numk * e_num, which must be an integer
            default: 13
        min_num : int, optional
            The number of k-point on the shortest k-path segment(MK segment)
            The number of k-point on other k-path segments are scaled
            according to their length
            default: 100
        numkx : int, optional
            Specify the number of k-points along the 1st and 2nd
            translation vectors in the equivalent first Brillouin Zone
            default: numk = 500
        """

        assert isinstance(numk, int) and numk > 0
        assert isinstance(e_num, (int, float)) and (0 < e_num < STATE_NUM)

        total_electron_num = numk * numk * e_num
        if total_electron_num != int(total_electron_num):
            raise ValueError("The total number of electron must be integer")

        GMKGLabels = (r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$")
        GMKGPath, GMKGIndices = KPath(
            [Gamma, Ms_CELL[0], Ks_CELL[0]], min_num=min_num, loop=True
        )

        ratio = np.linspace(0, 1, numk, endpoint=False)
        ratio_mesh = np.stack(
            np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
        ).reshape((-1, 2))
        BZMesh = np.matmul(ratio_mesh, Bs_CELL)

        self._numk = numk
        self._e_num_total = int(total_electron_num)
        self._BZMesh = BZMesh
        self._GMKGPath = GMKGPath
        self._GMKGLabels = GMKGLabels
        self._GMKGIndices = GMKGIndices

    @property
    def total_electron_num(self):
        """
        The `total_electron_num` attribute
        """

        return self._e_num_total

    @property
    def numk(self):
        """
        The `numk` attribute
        """

        return self._numk

    # Construct and diagonalize the Seven-Orbits Tight-Binding model
    # Hamiltonian in k-space
    def _EnergyCore(self, kpoints, *, return_vectors=True, **model_params):
        cell = lattice_generator("triangle", num0=1, num1=1)
        row_aoc_indices_table = IndexTable(
            AoC(CREATION, site=site, spin=spin, orbit=orbit)
            for site in cell.points for orbit in ORBITS for spin in SPINS
        )

        hoppings_intra, hoppings_inter = HoppingTerms(cell, **model_params)
        HM = np.zeros((len(kpoints), STATE_NUM, STATE_NUM), dtype=np.complex128)
        for term in hoppings_intra:
            c, a = term.components
            row = row_aoc_indices_table(c)
            col = row_aoc_indices_table(a.dagger())
            HM[:, row, col] += term.coeff
        for term in hoppings_inter:
            c, a = term.components
            p0, dR0 = cell.decompose(c.site)
            p1, dR1 = cell.decompose(a.site)
            row = row_aoc_indices_table(c.derive(site=p0))
            col = row_aoc_indices_table(a.derive(site=p1).dagger())
            HM[:, row, col] += term.coeff * np.exp(
                1j * np.dot(kpoints, dR1 - dR0)
            )
        HM += np.transpose(HM.conj(), axes=(0, 2, 1))

        if return_vectors:
            return np.linalg.eigh(HM)
        else:
            return np.linalg.eigvalsh(HM)

    def _TypicalSolver(self, **model_params):
        params_id = ",".join(
            "{0}={1:.4f}".format(key, model_params[key])
            for key in sorted(model_params)
        )
        if not hasattr(self, "_params_id") or self._params_id != params_id:
            GMKGPathEs = self._EnergyCore(
                self._GMKGPath, return_vectors=False, **model_params
            )
            BZMeshEs, BZMeshVectors = self._EnergyCore(
                self._BZMesh, **model_params
            )
            self._params_id = params_id
            self._GMKGPathEs = GMKGPathEs
            self._BZMeshEs = BZMeshEs.reshape((-1, ))
            self._BZMeshProbs = np.transpose(
                (BZMeshVectors * BZMeshVectors.conj()).real, axes=(0, 2, 1)
            ).reshape((-1, BZMeshVectors.shape[1]))

    def DOS(self, gamma=0.01, **model_params):
        """
        Calculate the orbital and spin projected densities of states

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            Default: 0.01
        model_params : other key-word arguments
            Specifying the tight-binding model parameters

        Returns
        -------
        omegas : array of floats with shape (N, )
            The energy levels where the densities of states are calculated
        projected_dos : array of floats with shape (N, ORBIT_NUM * SPIN_NUM)
            The corresponding projected densities of states.
            Every column correspond to an single-particle state in the unit-cell
        """

        self._TypicalSolver(**model_params)
        E_min = np.min(self._BZMeshEs)
        E_max = np.max(self._BZMeshEs)
        extra = 0.1 * (E_max - E_min)
        omegas = np.arange(E_min - extra, E_max + extra, 0.01)
        projected_dos = np.array(
            [
                np.dot(
                    Lorentzian(xs=omega, x0=self._BZMeshEs, gamma=gamma),
                    self._BZMeshProbs
                ) for omega in omegas
            ]
        ) / (self._numk * self._numk)
        return omegas, projected_dos

    def AverageElectronNumber(self, **model_params):
        """
        Calculate the averaged number of electrons on each orbit and spin

        Parameters
        ----------
        model_params : other key-word arguments
            Specifying the tight-binding model parameters

        Returns
        -------
        GE : float
            The ground state energy per unit-cell of the system
        mu : float
            The chemical potential
        avg_electron_nums : array
            The averaged number of electrons on each orbit and spin
        """

        self._TypicalSolver(**model_params)
        kth = self._e_num_total
        BZMeshEs = self._BZMeshEs
        BZMeshProbs = self._BZMeshProbs
        partition_indices = np.argpartition(BZMeshEs, kth=[kth-1, kth])
        GE = np.sum(BZMeshEs[partition_indices[0:kth]])
        avg_electron_nums = np.sum(
            BZMeshProbs[partition_indices[0:kth]], axis=0
        )
        index0 = partition_indices[kth-1]
        index1 = partition_indices[kth]
        mu = (BZMeshEs[index0] + BZMeshEs[index1]) / 2

        GE /= (self._numk * self._numk)
        avg_electron_nums /= (self._numk * self._numk)
        return GE, mu, avg_electron_nums

    def VisualizeDOS(self, gamma=0.01, **model_params):
        """
        Plot the orbital projected densities of states for the two types of
        orbits

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            default: 0.01
        model_params : other key-word arguments
            Specifying the Seven-Orbits Tight-Binding model parameters
            The recognizable key-word arguments are:
                Muc, tsc, tss1, tss2, tss3, tss4, tss5, tss6
        """

        omegas, projected_dos = self.DOS(gamma=gamma, **model_params)
        global_dos = np.sum(projected_dos, axis=-1)

        fig, axes = plt.subplots(1, 3, sharex=True, facecolor="gray")
        line0, = axes[0].plot(omegas, global_dos, color=COLORS[0])
        line1, = axes[0].plot(
            omegas, np.sum(projected_dos[:, 0:2], axis=-1), color=COLORS[1]
        )
        line2, = axes[0].plot(
            omegas, np.sum(projected_dos[:, 2:], axis=-1), color=COLORS[2]
        )
        axes[1].plot(omegas, projected_dos[:, 0], color=COLORS[1])
        axes[2].plot(omegas, projected_dos[:, 2], color=COLORS[2])
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
            default: 0.01
        model_params : other key-word arguments
            Specifying the Seven-Orbits Tight-Binding model parameters
            The recognizable key-word arguments are:
                Muc, tsc, tss1, tss2, tss3, tss4, tss5, tss6
        """

        GE, mu, avg_electron_nums = self.AverageElectronNumber(**model_params)
        omegas, projected_dos = self.DOS(gamma=gamma, **model_params)
        global_dos = np.sum(projected_dos, axis=-1)

        msg0 = "The averaged number of electron per unit-cell: {0}"
        msg1 = "The averaged number of electron " \
               "on the {0}th orbit with spin-{1}: {2}"
        print("The chemical potential: {0}".format(mu))
        print("The ground state energy per unit-cell: {0}".format(GE))
        print(msg0.format(np.sum(avg_electron_nums)))
        for orbit in ORBITS:
            for which, spin in enumerate(["down", "up"]):
                index = orbit * SPIN_NUM + which
                print(msg1.format(orbit, spin, avg_electron_nums[index]))

        fig, axes = plt.subplots(1, 4, sharey=True, facecolor="gray")
        axes[0].plot(self._GMKGPathEs)
        axes[0].set_xlim(0, len(self._GMKGPathEs) - 1)
        axes[0].set_ylim(omegas[0], omegas[-1])
        axes[0].set_xticks(self._GMKGIndices)
        axes[0].set_xticklabels(self._GMKGLabels)
        axes[0].set_ylabel(r"$E$", rotation="horizontal")
        axes[0].grid(axis="x", ls="dashed")

        axes[1].plot(global_dos, omegas)
        axes[2].plot(projected_dos[:, 0], omegas)
        axes[3].plot(projected_dos[:, 2], omegas)

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
    Solver = SevenOrbitsTBASolver(e_num=13, numk=200)
    Solver.VisualizeDOS()
    Solver()