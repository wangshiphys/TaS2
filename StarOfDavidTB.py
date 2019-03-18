"""
Construct and solve the Star-of-David Tight-Binding model Hamiltonian

For the definition of the Star-of-David model, see the docstring of the
`ShowStarOfDavidModel.py` script.
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import *


class StarOfDavidTB:
    """
    Construct and solve the Star-of-David Tight-Binding model Hamiltonian
    """

    def __init__(self, e_num=13, min_num=100, numkx=500, numky=None):
        """
        Customize the newly created instance

        Prepare the k-points along the Gamma-M-K-Gamma path and the k-mesh in
        the equivalent First Brillouin Zone; Calculate the corresponding
        exponents exp(1j*k*r1), exp(1j*k*r2) and exp(1j*k*r3) for reuse.

        Parameters
        ----------
        e_num : int or float, optional
            The averaged number of electrons per unit-cell
            The total number of electrons in the system is:
                numkx * numky * e_num, which must be an integer
            Default: 13
        min_num : int, optional
            The number of k-point on the shortest k-path segment(MK segment)
            The number of k-point on other k-path segments are scaled
            according to their length
            Default: 100
        numkx, numky : int, optional
            Specify the number of k-points along the 1st and 2nd
            translation vectors in the equivalent first Brillouin Zone
            Default: numkx = 500, numky = numkx
        """

        assert 0 <= e_num <= 26, "The averaged number of electrons per " \
                                 "unit-cell must be non-negative and no " \
                                 "larger than 26"
        assert isinstance(numkx, int) and numkx > 0
        assert (numky is None) or (isinstance(numky, int) and numky > 0)

        if numky is None:
            numky = numkx
        total_electron_num = numkx * numky * e_num
        if total_electron_num != int(total_electron_num):
            raise ValueError("The total number of electron must be integer")

        GMKGLabels = (r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$")
        GMKGPath, GMKGIndices = KPath(
            [Gamma, Ms_STAR[0], Ks_STAR[0]], min_num=min_num, loop=True
        )
        GMKGPathExponents = np.exp(1j * np.matmul(GMKGPath, Rs_STAR.T))

        ratio_kx = np.linspace(0, 1, numkx, endpoint=False)
        ratio_ky = np.linspace(0, 1, numky, endpoint=False)
        ratio_mesh = np.stack(
            np.meshgrid(ratio_kx, ratio_ky, indexing="ij"), axis=-1
        ).reshape((-1, 2))
        BZMeshExponents = np.exp(
            1j * np.linalg.multi_dot([ratio_mesh, Bs_STAR, Rs_STAR.T])
        )

        self._numkx = numkx
        self._numky = numky
        self._total_electron_num = int(total_electron_num)
        # Cached for reuse
        self._GMKGIndices = GMKGIndices
        self._GMKGLabels = GMKGLabels
        self._GMKGPathExponents = GMKGPathExponents
        self._BZMeshExponents = BZMeshExponents

    @property
    def total_electron_num(self):
        """
        The `total_electron_num` attribute
        """

        return self._total_electron_num

    @property
    def numkx(self):
        """
        The `numkx` attribute
        """

        return self._numkx

    @property
    def numky(self):
        """
        The `numky` attribute
        """

        return self._numky

    # Construct the Star-of-David Tight-Binding model Hamiltonian in the k-space
    # Diagonalize the model Hamiltonian
    @staticmethod
    def _EnergyCore(
            exponents, *, t0=1.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0,t5=1.0,
            mu0=0.0, mu1=0.0, mu2=0.0, return_vectors=True
    ):
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

        table = {(1, 0): 0, (0, 1): 1, (-1, 1): 2}
        all_points = [POINT_TYPE_A, POINT_TYPE_B, POINT_TYPE_C]
        H = np.zeros((exponents.shape[0], 13, 13), dtype=np.complex128)

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

    @staticmethod
    def Verify(min_num=100, mu=0.0):
        """
        Verify that this tight-binding model is equivalent to the
        simple nearest-neighbor tight-binding model defined on the triangular
        lattice when all the hopping parameters and the on-site chemical
        potentials are set to be identical

        See also the `TriangleSimpleTB.py` file for the definition of the
        simple nearest-neighbor tight-binding model

        Parameters
        ----------
        min_num : int, optional
            The number of k-point on the shortest k-path segment(MK segment)
            The number of k-point on other k-path segments are scaled
            according to their length
            Default: 100
        mu : float, optional
            On-site chemical potential
            Default: 0.0
        """

        GMKGLabels = (r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$")
        GMKGPath, GMKGIndices = KPath(
            [Gamma, Ms_CELL[0], Ks_CELL[0]], min_num=min_num, loop=True
        )
        Es_Cell = mu - 2 * np.sum(
            np.cos(np.matmul(GMKGPath, Rs_CELL.T)), axis=-1
        )
        Es_Star = StarOfDavidTB._EnergyCore(
            np.exp(1j * np.matmul(GMKGPath, Rs_STAR.T)),
            t0=1.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0, t5=1.0,
            mu0=mu, mu1=mu, mu2=mu, return_vectors=False
        )

        smallest = np.ceil(np.min(Es_Star))
        largest = np.floor(np.max(Es_Star))
        yticks = np.arange(smallest, largest + 1)
        ylabels = ["{0:.0f}".format(y) for y in yticks]

        fig, ax = plt.subplots()
        ax.plot(Es_Star, lw=LINE_WIDTH, alpha=0.8)
        ax.plot(Es_Cell, lw=LINE_WIDTH/3, ls="dashed", color="black")

        ax.set_xlim(0, len(Es_Cell) - 1)
        ax.set_xticks(GMKGIndices)
        ax.set_yticks(yticks)
        ax.set_xticklabels(GMKGLabels, fontsize=LABEL_SIZE)
        ax.set_yticklabels(ylabels, fontsize=LABEL_SIZE)
        ax.set_ylabel(
            r"$E$", fontsize=LABEL_SIZE, rotation="horizontal", labelpad=15
        )
        ax.grid(axis="both", linestyle="dashed")
        ax.set_title(
            "EB of the SD tight-binding model", fontsize=FONT_SIZE, pad=15
        )
        plt.show()
        plt.close("all")

    @staticmethod
    def _params2identifier(**kwargs):
        mu_keys = ["mu0", "mu1", "mu2"]
        t_keys = ["t0", "t1", "t2", "t3", "t4", "t5"]
        return tuple(
            [
                kwargs.get(key, default)
                for keys, default in zip([t_keys, mu_keys], [1.0, 0.0])
                for key in keys
            ]
        )

    def _TypicalSolver(self, **kwargs):
        params_id = self._params2identifier(**kwargs)
        if not hasattr(self, "_params_id") or self._params_id != params_id:
            GMKGPathEs = self._EnergyCore(
                self._GMKGPathExponents, return_vectors=False, **kwargs
            )
            BZMeshEs, BZMeshVectors = self._EnergyCore(
                self._BZMeshExponents, **kwargs
            )
            self._params_id = params_id
            self._GMKGPathEs = GMKGPathEs
            self._BZMeshEs = BZMeshEs.reshape((-1, ))
            self._BZMeshProbs = np.transpose(
                (BZMeshVectors * BZMeshVectors.conj()).real, axes=(0, 2, 1)
            ).reshape((-1, 13))

    def DOS(self, gamma=0.01, **kwargs):
        """
        Calculate the atomic projected densities of states

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            Default: 0.01
        kwargs : other key-word arguments
            Specifying the Star-of-David tight-binding model parameters
            The recognizable key-word arguments are:
                t0, t1, t2, t3, t4, t5, m0, mu1, mu2

        Returns
        -------
        omegas : array of floats with shape (N, )
            The energy levels where the densities of states are calculated
        local_dos : array of floats with shape (N, 13)
            The corresponding atomic projected densities of states.
            Every column correspond to a Ta atoms in the Star-of-David unit-cell
        """

        self._TypicalSolver(**kwargs)
        E_min = np.min(self._BZMeshEs)
        E_max = np.max(self._BZMeshEs)
        extend = (E_max - E_min) * 0.1
        omega_min = E_min - extend
        omega_max = E_max + extend
        omegas = np.arange(omega_min, omega_max, 0.01)

        local_dos = np.array(
            [
                np.dot(
                    Lorentzian(xs=omega, x0=self._BZMeshEs, gamma=gamma),
                    self._BZMeshProbs
                ) for omega in omegas
            ]
        ) / (self._numkx * self._numky)
        return omegas, local_dos

    def VisualizeDOS(self, gamma=0.01, **kwargs):
        """
        Plot the atomic projected densities of states for the three
        inequivalent Ta atoms as well as the global densities

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            Default: 0.01
        kwargs : other key-word arguments
            Specifying the Star-of-David tight-binding model parameters
            The recognizable key-word arguments are:
                t0, t1, t2, t3, t4, t5, m0, mu1, mu2
        """

        omegas, local_dos = self.DOS(gamma=gamma, **kwargs)
        global_dos = np.sum(local_dos, axis=-1)
        fig, axes = plt.subplots(2, 2, sharex=True)
        line0, = axes[0, 0].plot(
            omegas, global_dos, lw=LINE_WIDTH, color=COLORS[0]
        )
        line1, = axes[0, 0].plot(
            omegas, local_dos[:, 0], lw=LINE_WIDTH, color=COLORS[1]
        )
        line2, = axes[0, 0].plot(
            omegas, np.sum(local_dos[:, 1:7], axis=-1),
            lw=LINE_WIDTH, color=COLORS[2]
        )
        line3, = axes[0, 0].plot(
            omegas, np.sum(local_dos[:, 7:], axis=-1),
            lw=LINE_WIDTH, color=COLORS[3]
        )
        axes[0, 1].plot(omegas, local_dos[:, 0], lw=LINE_WIDTH, color=COLORS[1])
        axes[1, 0].plot(omegas, local_dos[:, 1], lw=LINE_WIDTH, color=COLORS[2])
        axes[1, 1].plot(omegas, local_dos[:, 7], lw=LINE_WIDTH, color=COLORS[3])

        axes[0, 0].set_xlim(omegas[0], omegas[-1])
        axes[0, 0].legend(
            [line0, line1, line2, line3], ["DOS", "LDOS-A", "LDOS-B", "LDOS-C"],
            loc="best", fontsize=LABEL_SIZE
        )
        for ax in axes.flat:
            for which, spine in ax.spines.items():
                spine.set_linewidth(SPINE_WIDTH)
            ax.tick_params(
                axis="both", which="both",
                length=TICK_LENGTH, width=TICK_WIDTH, labelsize=LABEL_SIZE
            )
        plt.show()
        plt.close("all")

    def AverageElectronNumber(self, **kwargs):
        """
        Calculate the averaged number of electron for the three inequivalent
        Ta atoms as well as the global chemical potential

        Parameters
        ----------
        kwargs : other key-word arguments
            Specifying the Star-of-David tight-binding model parameters
            The recognizable key-word arguments are:
                t0, t1, t2, t3, t4, t5, m0, mu1, mu2
        """

        self._TypicalSolver(**kwargs)
        kth = self._total_electron_num // 2
        partition_indices = np.argpartition(self._BZMeshEs, kth=[kth-1, kth])
        avg_electron_nums = 2 * np.sum(
            self._BZMeshProbs[partition_indices[0:kth]], axis=0
        )

        # The total number of electron is odd
        if self._total_electron_num % 2:
            mu = self._BZMeshEs[partition_indices[kth]]
            avg_electron_nums += self._BZMeshProbs[partition_indices[kth]]
        # The total number of electron is even
        else:
            index0, index1 = partition_indices[kth-1:kth+1]
            mu = (self._BZMeshEs[index0] + self._BZMeshEs[index1]) / 2
        avg_electron_nums /= (self._numkx * self._numky)
        return mu, avg_electron_nums

    def __call__(self, gamma=0.01, **kwargs):
        """
        The main entrance of instance of this class

        Parameters
        ----------
        gamma : float, optional
            Specifying the width of the Lorentzian function
            Default: 0.01
        kwargs : other key-word arguments
            Specifying the Star-of-David tight-binding model parameters
            The recognizable key-word arguments are:
                t0, t1, t2, t3, t4, t5, m0, mu1, mu2
        """

        mu, avg_electron_nums = self.AverageElectronNumber(**kwargs)
        omegas, local_dos = self.DOS(gamma=gamma, **kwargs)
        global_dos = np.sum(local_dos, axis=-1)

        msg0 = "The averaged number of electron per Star-of-David: {0}"
        msg1 = "The averaged number of electron on the {0:2d}th Ta atom: {1}"
        print("The chemical potential: {0}".format(mu))
        print(msg0.format(np.sum(avg_electron_nums)))
        for index, avg_electron_num in enumerate(avg_electron_nums):
            print(msg1.format(index, avg_electron_num))

        yticks = np.arange(np.ceil(omegas[0]),  np.floor(omegas[-1]))
        ylabels = ["{0:.0f}".format(y) for y in yticks]

        fig, axes = plt.subplots(1, 5, sharey=True)
        axes[0].plot(self._GMKGPathEs, lw=LINE_WIDTH)
        axes[0].set_xlim(0, len(self._GMKGPathEs) - 1)
        axes[0].set_ylim(omegas[0], omegas[-1])
        axes[0].set_xticks(self._GMKGIndices)
        axes[0].set_yticks(yticks)
        axes[0].set_xticklabels(self._GMKGLabels)
        axes[0].set_yticklabels(ylabels)
        axes[0].set_ylabel(
            r"$E$", fontsize=LABEL_SIZE, rotation="horizontal", labelpad=15
        )
        axes[0].grid(axis="x", ls="dashed", lw=LINE_WIDTH/4)

        axes[1].plot(global_dos, omegas, lw=LINE_WIDTH)
        axes[2].plot(local_dos[:, 0], omegas, lw=LINE_WIDTH)
        axes[3].plot(local_dos[:, 1], omegas, lw=LINE_WIDTH)
        axes[4].plot(local_dos[:, 7], omegas, lw=LINE_WIDTH)

        for ax in axes:
            ax.axhline(mu, ls="dashed", lw=LINE_WIDTH/2, color="gray")
            for which, spine in ax.spines.items():
                spine.set_linewidth(SPINE_WIDTH)
            ax.tick_params(
                axis="both", which="both",
                length=TICK_LENGTH, width=TICK_WIDTH, labelsize=LABEL_SIZE
            )
        for ax, xlabel in zip(axes[1:], ["DOS", "LDOS-A", "LDOS-B", "LDOS-C"]):
            ax.tick_params(labelbottom=True)
            ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
        axes[2].text(
            0.5, 0.55, "$E_F={0:.3f}$".format(mu),
            ha="center", va="center",
            transform=axes[2].transAxes, fontsize=FONT_SIZE
        )

        plt.show()
        # fig.savefig("demo/EB and DOS.jpg", dpi=200)
        plt.close("all")


if __name__ == "__main__":
    gamma = 0.02
    kwargs = {
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
    Solver = StarOfDavidTB(numkx=500)
    Solver.Verify()
    Solver.VisualizeDOS(**kwargs)
    Solver()
    Solver(**kwargs)
