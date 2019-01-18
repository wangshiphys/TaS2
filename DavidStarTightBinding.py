"""
Constructing and solving the David-Star Tight-Binding model Hamiltonian

For the definition of the David-Star model, see the docstring of the
`ShowModel.py` script.
"""


import matplotlib.pyplot as plt
import numpy as np

from TaS2DataBase import *


color_map = plt.get_cmap("tab10")
COLORS = color_map(range(color_map.N))


class DavidStarTB:
    """
    Constructing and solving the David-Star Tight-Binding model Hamiltonian
    """
    def __init__(self, e_num=13, min_num=100, numkx=500, numky=None):
        """
        Customize the newly created instance

        Prepare the k-points along the Gamma-M-K-Gamma path and the k-mesh in
        the equivalent First Brillouin Zone; Calculate the corresponding phase
        factors exp(1j*k*r1), exp(1j*k*r2) and exp(1j*k*r3) for reuse.

        Parameters
        ----------
        e_num : int or float, optional
            The averaged number of electrons per David-Star
            The total number of electrons in the system is:
                numkx * numky * e_num, which must be an integer
            Default: 13
        min_num : int, optional
            The number of k-point on the shortest k-path segment(MK segment)
            The number of k-point on other k-path segments are scaled
            according to their length
            Default: 100
        numkx, numky : int, optional
            Specify the number of k-points along the the 1st and 2nd
            translation vectors in the equivalent first Brillouin Zone
            Default: numkx = 500, numky = numkx
        """

        msg = "The averaged number of electrons per David-Star must be " \
              "non-negative and no larger than 26"
        assert 0 <= e_num <= 26, msg
        assert isinstance(numkx, int) and numkx > 0
        assert (numky is None) or (isinstance(numky, int) and numky > 0)

        if numky is None:
            numky = numkx
        total_electron_num = numkx * numky * e_num
        msg = "The total number of electrons in the system must be integer"
        assert total_electron_num == int(total_electron_num), msg

        xlabels = (r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$")
        GMKPath, xticks = KPath(
            [Gamma, Ms_STAR[0], Ks_STAR[0]], min_num=min_num
        )
        GMKPathPhaseFactors = np.exp(1j * np.matmul(GMKPath, Rs_STAR.T))

        ratio_kx = np.linspace(0, 1, numkx, endpoint=False)
        ratio_ky = np.linspace(0, 1, numky, endpoint=False)
        ratio_mesh = np.stack(
            np.meshgrid(ratio_kx, ratio_ky, indexing="ij"), axis=-1
        ).reshape((-1, 2))
        BZMeshPhaseFactors = np.exp(
            1j * np.linalg.multi_dot([ratio_mesh, Bs_STAR, Rs_STAR.T])
        )

        self._numkx = numkx
        self._numky = numky
        self._total_electron_num = int(total_electron_num)
        # Cached for reuse
        self._xticks = xticks
        self._xlabels = xlabels
        self._GMKPathPhaseFactors = GMKPathPhaseFactors
        self._BZMeshPhaseFactors = BZMeshPhaseFactors

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

    # Construct the David-Star Tight-Binding model Hamiltonian in the k-space
    # Diagonalize the model Hamiltonian
    @staticmethod
    def _EnergyCore(phase_factors, t0=1.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0,
                    t5=1.0, mu0=0.0, mu1=0.0, mu2=0.0):
        # Map the translations along b1, b2 of the David-Star to the index of
        # the David-Star
        table = {(1, 0): 0, (0, 1): 1, (-1, 1): 2}
        all_points = [POINT_TYPE_A, POINT_TYPE_B, POINT_TYPE_C]
        H = np.zeros((phase_factors.shape[0], 13, 13), dtype=np.complex128)

        # On-site chemical potential terms
        for points, mu in zip(all_points, [mu0, mu1, mu2]):
            for p in points:
                H[:, p, p] = mu / 2

        # Intra-Cluster hopping terms
        for bonds, hopping in zip(BONDS_INTRA, [t0, t1, t2]):
            for (star_index0, p0), (star_index1, p1) in bonds:
                H[:, p0, p1] = -hopping

        # Inter-Cluster hopping terms
        for bonds, hopping in zip(BONDS_INTER, [t3, t4, t5]):
            for (star_index0, p0), (star_index1, p1) in bonds:
                H[:, p0, p1] = -hopping * phase_factors[:, table[star_index1]]
        H += np.transpose(H.conj(), axes=(0, 2, 1))
        return np.linalg.eigh(H)

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

        xlabels = [r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"]
        GMKPath, xticks = KPath(
            [Gamma, Ms_CELL[0], Ks_CELL[0]], min_num=min_num, loop=True
        )
        Es_Cell = mu - 2 * np.sum(
            np.cos(np.matmul(GMKPath, Rs_CELL.T)), axis=-1
        )
        Es_Star, trash = DavidStarTB._EnergyCore(
            np.exp(1j * np.matmul(GMKPath, Rs_STAR.T)),
            t0=1.0, t1=1.0, t2=1.0, t3=1.0, t4=1.0, t5=1.0,
            mu0=mu, mu1=mu, mu2=mu,
        )
        E_min = np.floor(np.min(Es_Star))
        E_max = np.ceil(np.max(Es_Star))
        yticks = np.arange(E_min, E_max + 1)
        ylabels = ["{0}".format(y) for y in yticks]

        alpha = 0.9
        line_width = 6
        font_size = "xx-large"
        fig, ax = plt.subplots()

        # The columns represent separate data sets
        ax.plot(Es_Star, lw=line_width, alpha=alpha)
        ax.plot(Es_Cell, lw=line_width/3, ls="dashed", color="black")

        ax.set_xlim(0, len(Es_Cell) - 1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xlabels, fontsize=font_size)
        ax.set_yticklabels(ylabels, fontsize=font_size)
        ax.set_ylabel(r"$E$", fontsize=font_size)
        ax.grid(axis="both", linestyle="dashed")
        ax.set_title(
            "EB of the David-Star tight-binding model", fontsize=font_size
        )
        plt.show()
        plt.close("all")

    def _BZMesh(self, **kwargs):
        Es, Vecs = self._EnergyCore(self._BZMeshPhaseFactors, **kwargs)
        Es = Es.reshape((-1,))
        Vecs = np.transpose(Vecs, axes=(0, 2, 1)).reshape((-1, 13))
        kth = self._total_electron_num // 2
        partition_indices = np.argpartition(Es, kth=[kth-1, kth])
        tmp = Vecs[partition_indices[0:kth]]
        avg_electron_nums = 2 * np.sum((tmp * tmp.conj()).real, axis=0)

        # The total number of electrons is odd
        if self._total_electron_num % 2:
            mu = Es[partition_indices[kth]]
            tmp = Vecs[partition_indices[kth]]
            avg_electron_nums += (tmp * tmp.conj()).real
        # The total number of electrons is even
        else:
            mu = (Es[partition_indices[kth-1]] + Es[partition_indices[kth]]) / 2
        avg_electron_nums /= (self._numkx * self._numky)

        step = 1e-2
        omega_min = np.floor(np.min(Es)) - 0.2
        omega_max = np.ceil(np.max(Es)) + 0.2
        hist, bins = np.histogram(Es, np.arange(omega_min, omega_max, step))
        DoS = hist / (self._numkx * self._numky * step)
        omegas = bins[1:] - step / 2
        return mu, avg_electron_nums, DoS, omegas

    def _GMKPath(self, **kwargs):
        Es, Vecs = self._EnergyCore(self._GMKPathPhaseFactors, **kwargs)
        amplitudes = (Vecs * Vecs.conj()).real
        return Es, amplitudes

    def __call__(self, **kwargs):
        mu, avg_electron_nums, DoS, omegas = self._BZMesh(**kwargs)
        GMKPath_Es, amplitudes = self._GMKPath(**kwargs)

        msg0 = "The averaged number of electron per David-Star: {0}"
        msg1 = "The averaged number of electron on the {0:2d}th Ta atom: {1}"
        print("The chemical potential: {0}".format(mu))
        print(msg0.format(np.sum(avg_electron_nums)))
        for index, avg_electron_num in enumerate(avg_electron_nums):
            print(msg1.format(index, avg_electron_num))

        figures = []
        for index in [0, 7]:
            fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
            for i in range(2):
                for j in range(3):
                    ax = axes[i, j]
                    CS = ax.contourf(
                        amplitudes[:, :, index].T, levels=100, cmap="rainbow"
                    )
                    ax.set_title("Bond Index = {0}".format(index + 1))
                    ax.set_yticks(range(13))
                    ax.set_xticks(self._xticks)
                    ax.set_xticklabels(self._xlabels)
                    ax.grid(axis="both", ls="dashed")
                    fig.colorbar(CS, ax=ax, format="%.2f")
                    index += 1
            figures.append(fig)

        fig, ax = plt.subplots()
        CS = ax.contourf(amplitudes[:, :, 6].T, levels=100, cmap="rainbow")
        ax.set_title("Band Index = 7")
        ax.set_yticks(range(13))
        ax.set_xticks(self._xticks)
        ax.set_xticklabels(self._xlabels)
        ax.grid(axis="both", ls="dashed")
        fig.colorbar(CS, ax=ax, format="%.2f")
        figures.append(fig)

        line_width = 6
        E_min = np.floor(np.min(GMKPath_Es))
        E_max = np.ceil(np.max(GMKPath_Es))
        yticks = np.arange(E_min, E_max + 1)
        ylabels = ["{0:.0f}".format(y) for y in yticks]

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.plot(GMKPath_Es, linewidth=line_width)
        ax0.axhline(mu, ls="dashed", color="gray")
        ax0.set_xlim(0, len(GMKPath_Es) - 1)
        ax0.set_xticks(self._xticks)
        ax0.set_yticks(yticks)
        ax0.set_xticklabels(self._xlabels)
        ax0.set_yticklabels(ylabels)
        ax0.set_ylabel(r"$E$")
        ax0.grid(axis="x", ls="dashed")
        ax0.set_title(r"EB along $\Gamma-M-K-\Gamma$ Path")

        ax1.plot(omegas, DoS, linewidth=line_width / 2, color=COLORS[1])
        ax1.axvline(x=mu, ls="dashed", linewidth=1, color=COLORS[0])
        ax1.text(
            mu, 0, r"$E_F={0:.3f}$".format(mu),
            color=COLORS[0], ha="center", va="top"
        )
        ax1.set_xlim(omegas[0], omegas[-1])
        ax1.set_xlabel(r"$\omega$")
        ax1.set_ylabel("DoS")
        ax1.set_title("Density of States")
        figures.append(fig)

        figures[0].subplots_adjust(
            top=0.922, bottom=0.122, left=0.091, right=0.971,
            hspace=0.2, wspace=0.309
        )
        figures[1].subplots_adjust(
            top=0.926, bottom=0.081, left=0.066, right=0.935,
            hspace=0.244, wspace=0.299
        )
        figures[2].subplots_adjust(
            top=0.926, bottom=0.081, left=0.066, right=0.935,
            hspace=0.244, wspace=0.339
        )
        figures[3].subplots_adjust(
            top=0.926, bottom=0.081, left=0.066, right=0.977,
            hspace=0.2, wspace=0.2
        )

        plt.show()
        plt.close("all")


if __name__ == "__main__":
    Solver = DavidStarTB(numkx=500)
    Solver(
        t0=0.2, t1=0.2, t2=0.8, t3=1.0, t4=1.0, t5=1.0,
        mu0=0.4, mu1=-0.2, mu2=0.1
    )
