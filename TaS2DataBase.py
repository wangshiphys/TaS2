"""
Some commonly used data of this project
"""


import numpy as np
from HamiltonianPy import KPath

__all__ = [
    "COORDS_CELL", "COORDS_STAR",

    "As_CELL", "Bs_CELL", "Rs_CELL",

    "As_STAR", "Bs_STAR", "Rs_STAR",

    "Gamma", "Ms_CELL", "Ks_CELL", "Ms_STAR", "Ks_STAR",

    "Lorentzian", "BaseTBSolver",
]


# Coordinates of the points in the unit cell
COORDS_CELL = np.array([[0.0, 0.0]], dtype=np.float64)

COORDS_STAR = np.array(
    [
        # The central point of the David Star(DS), the index of this point is 0
        [0.0, 0.0],
        # The inner loop six points of the DS, the indices of these points are
        # from 1 to 6 correspond to their order in this array
        [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2], [1.0, 0.0],
        [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2], [-1.0, 0.0],
        # The outer loop six points of the DS, the indices of these points
        # are from 7 to 12 correspond to their order in this array
        [0.0, -np.sqrt(3)], [1.5, -np.sqrt(3)/2], [1.5, np.sqrt(3)/2],
        [0.0, np.sqrt(3)], [-1.5, np.sqrt(3)/2], [-1.5, -np.sqrt(3)/2]
    ],
    dtype=np.float64
)
################################################################################


# Translation vectors of the real and reciprocal space(k-space)
# Translation vectors of the original triangular lattice
As_CELL = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=np.float64)
# The corresponding translation vectors in k-space
Bs_CELL = 2 * np.pi * np.linalg.inv(As_CELL.T)
# Three inequivalent nearest neighbors
Rs_CELL = np.array(
    [As_CELL[0], As_CELL[1], As_CELL[1] - As_CELL[0]], dtype=np.float64
)

# Translation vectors of the David Star
# As_STAR = np.array(
#     [[3.5, np.sqrt(3)/2], [1.0, 2 * np.sqrt(3)]], dtype=np.float64
# )
As_STAR = np.array(
    [3 * As_CELL[0] + As_CELL[1], 4 * As_CELL[1] - As_CELL[0]], dtype=np.float64
)
# The corresponding translation vectors in k-space
Bs_STAR = 2 * np.pi * np.linalg.inv(As_STAR.T)
# Three inequivalent nearest neighbors
Rs_STAR = np.array(
    [As_STAR[0], As_STAR[1], As_STAR[1] - As_STAR[0]], dtype=np.float64
)
################################################################################


# High symmetry points in k-space
Gamma = np.array([0.0, 0.0], dtype=np.float64)

# Linear combination coefficients of the real space translation vectors
_coeffs = np.array(
    [[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]], dtype=np.int64
)

# The middle points of the edges of the First Brillouin Zone(M-points)
Ms_CELL = np.dot(_coeffs, Bs_CELL) / 2
Ms_STAR = np.dot(_coeffs, Bs_STAR) / 2

# The corner of the First Brillouin Zone(K-points)
_coeffs = _coeffs[[*range(1, _coeffs.shape[0]), 0]] + _coeffs
Ks_CELL = np.dot(_coeffs, Bs_CELL) / 3
Ks_STAR = np.dot(_coeffs, Bs_STAR) / 3

del _coeffs
################################################################################


# Simulation of the Delta function
def Lorentzian(xs, x0=0.0, gamma=0.01):
    """
    The Lorentzian function

    Parameters
    ----------
    xs : float or array of floats
        The independent variable of the Lorentzian function
    x0 : float or array of floats, optional
        The center of the Lorentzian function
        Default: 0.0
    gamma : float, optional
        Specifying the width of the Lorentzian function
        Default: 0.01

    Returns
    -------
    res : float or array of floats
        1. `xs` and `x0` are both scalar, then the corresponding function
        value is returned;
        2. `xs` and/or `x0` are array of floats, the two parameters are
        broadcasted to calculated the expression `xs -x0`, and the
        corresponding function values are returned.

    See also
    --------
    numpy.broadcast
    http://mathworld.wolfram.com/LorentzianFunction.html
    """

    gamma /= 2
    return gamma / np.pi / ((xs - x0) ** 2 + gamma ** 2)


class BaseTBSolver:
    """
    The base solver for all related tight-binding model of this project

    This class only defines the common operations and provide implementation
    to some of the functions. It is not meant to be instantiated directly.
    """

    # This attribute should be override by derived subclasses
    default_model_params = {}

    def __init__(self, orbit_num, e_num=13, min_num=100, numkx=500, numky=None):
        """
        Customize the newly created instance

        Prepare the k-points along the Gamma-M-K-Gamma path and the k-mesh in
        the equivalent First Brillouin Zone; Calculate the corresponding
        exponents exp(1j*k*r1), exp(1j*k*r2) and exp(1j*k*r3) for reuse.

        Parameters
        ----------
        orbit_num : int
            The number of orbits in the unit-cell
            The total number of single-particle states in the unit-cell is
            `2 * orbit_num` for spin-1/2 case
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

        assert isinstance(orbit_num, int) and orbit_num > 0
        assert isinstance(e_num, (int, float)) and (0 <= e_num <= 2 * orbit_num)
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
        self._orbit_num = orbit_num
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

    @property
    def orbit_num(self):
        """
        The `orbit_num` attribute
        """

        return self._orbit_num

    def _EnergyCore(self, exponents, *, return_vectors=True, **model_params):
        raise NotImplementedError(
            "This method should be implemented in the derived subclass."
        )

    @classmethod
    def _params2identifier(cls, **model_params):
        return tuple(
            model_params.get(name, cls.default_model_params[name])
            for name in sorted(cls.default_model_params)
        )

    def _TypicalSolver(self, **model_params):
        params_id = self._params2identifier(**model_params)
        if not hasattr(self, "_params_id") or self._params_id != params_id:
            GMKGPathEs = self._EnergyCore(
                self._GMKGPathExponents, return_vectors=False, **model_params
            )
            BZMeshEs, BZMeshVectors = self._EnergyCore(
                self._BZMeshExponents, **model_params
            )
            self._params_id = params_id
            self._GMKGPathEs = GMKGPathEs
            self._BZMeshEs = BZMeshEs.reshape((-1, ))
            self._BZMeshProbs = np.transpose(
                (BZMeshVectors * BZMeshVectors.conj()).real, axes=(0, 2, 1)
            ).reshape((-1, BZMeshVectors.shape[1]))

    def DOS(self, gamma=0.01, **model_params):
        """
        Calculate the orbital-projected densities of states

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
        projected_dos : array of floats with shape (N, orbit_num)
            The corresponding orbital-projected densities of states.
            Every column correspond to an orbit in the unit-cell
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
        ) / (self._numkx * self._numky)
        return omegas, projected_dos

    def AverageElectronNumber(self, **model_params):
        """
        Calculate the averaged number of electrons on each orbit

        Parameters
        ----------
        model_params : other key-word arguments
            Specifying the tight-binding model parameters

        Returns
        -------
        GE : float
            The ground state energy per Star-of-David of the system
        mu : float
            The chemical potential
        avg_electron_nums : array
            The averaged number of electrons on each orbit
        """

        self._TypicalSolver(**model_params)
        kth = self._total_electron_num // 2
        partition_indices = np.argpartition(self._BZMeshEs, kth=[kth-1, kth])
        GE = 2 * np.sum(self._BZMeshEs[partition_indices[0:kth]])
        avg_electron_nums = 2 * np.sum(
            self._BZMeshProbs[partition_indices[0:kth]], axis=0
        )

        # The total number of electrons is odd
        if self._total_electron_num % 2:
            mu = self._BZMeshEs[partition_indices[kth]]
            avg_electron_nums += self._BZMeshProbs[partition_indices[kth]]
            GE += mu
        else:
            index0, index1 = partition_indices[kth-1:kth+1]
            mu = (self._BZMeshEs[index0] + self._BZMeshEs[index1]) / 2
        GE /= (self._numkx * self._numky)
        avg_electron_nums /= (self._numkx * self._numky)
        return GE, mu, avg_electron_nums
