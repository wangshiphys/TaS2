"""
Some commonly used data of this project
"""


import numpy as np


__all__ = [
    "COORDS_CELL", "COORDS_STAR",

    "As_CELL", "Bs_CELL", "Rs_CELL",

    "As_STAR", "Bs_STAR", "Rs_STAR",

    "Gamma", "Ms_CELL", "Ks_CELL", "Ms_STAR", "Ks_STAR",

    "POINT_TYPE_A", "POINT_TYPE_B", "POINT_TYPE_C",

    "BOND_TYPE_A", "BOND_TYPE_B", "BOND_TYPE_C",
    "BOND_TYPE_D", "BOND_TYPE_E", "BOND_TYPE_F",
    "BONDS_INTRA", "BONDS_INTER", "ALL_BONDS",

    "KPath",
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


# Different types of points in the David-Star. The points belong to different
# types are specified by their indices. See also `COORDS_STAR` for the indices
# definition
POINT_TYPE_A = (0, )
POINT_TYPE_B = (1, 2, 3, 4, 5, 6)
POINT_TYPE_C = (7, 8, 9, 10, 11, 12)

# Different types of bonds in and between the David-Star.
# The bonds are classified according to their length
# The bonds are specified by two points and every points are specified by the
# David-Star they belong and their indices in the David-Star
BOND_TYPE_A = (
    (((0, 0), 0), ((0, 0), 1)),
    (((0, 0), 0), ((0, 0), 2)),
    (((0, 0), 0), ((0, 0), 3)),
    (((0, 0), 0), ((0, 0), 4)),
    (((0, 0), 0), ((0, 0), 5)),
    (((0, 0), 0), ((0, 0), 6)),
)
BOND_TYPE_B = (
    (((0, 0), 1), ((0, 0), 2)),
    (((0, 0), 2), ((0, 0), 3)),
    (((0, 0), 3), ((0, 0), 4)),
    (((0, 0), 4), ((0, 0), 5)),
    (((0, 0), 5), ((0, 0), 6)),
    (((0, 0), 6), ((0, 0), 1)),
)
BOND_TYPE_C = (
    (((0, 0), 1), ((0, 0), 7)),
    (((0, 0), 7), ((0, 0), 2)),
    (((0, 0), 2), ((0, 0), 8)),
    (((0, 0), 8), ((0, 0), 3)),
    (((0, 0), 3), ((0, 0), 9)),
    (((0, 0), 9), ((0, 0), 4)),
    (((0, 0), 4), ((0, 0), 10)),
    (((0, 0), 10), ((0, 0), 5)),
    (((0, 0), 5), ((0, 0), 11)),
    (((0, 0), 11), ((0, 0), 6)),
    (((0, 0), 6), ((0, 0), 12)),
    (((0, 0), 12), ((0, 0), 1)),
)
BOND_TYPE_D = (
    (((0, 0), 9), ((1, 0), 12)),
    (((0, 0), 10), ((0, 1), 7)),
    (((0, 0), 11), ((-1, 1), 8)),
)
BOND_TYPE_E = (
    (((0, 0), 8), ((1, 0), 12)),
    (((0, 0), 9), ((1, 0), 11)),
    (((0, 0), 9), ((0, 1), 7)),
    (((0, 0), 10), ((0, 1), 12)),
    (((0, 0), 10), ((-1, 1), 8)),
    (((0, 0), 11), ((-1, 1), 7)),
)
BOND_TYPE_F = (
    (((0, 0), 3), ((1, 0), 12)),
    (((0, 0), 9), ((1, 0), 6)),
    (((0, 0), 4), ((0, 1), 7)),
    (((0, 0), 10), ((0, 1), 1)),
    (((0, 0), 5), ((-1, 1), 8)),
    (((0, 0), 11), ((-1, 1), 2)),
)

# Bonds intra and inter the David-Star
BONDS_INTRA = (BOND_TYPE_A, BOND_TYPE_B, BOND_TYPE_C)
BONDS_INTER = (BOND_TYPE_D, BOND_TYPE_E, BOND_TYPE_F)
ALL_BONDS = (
    BOND_TYPE_A, BOND_TYPE_B, BOND_TYPE_C,
    BOND_TYPE_D, BOND_TYPE_E, BOND_TYPE_F,
)
################################################################################


# Generate k-points in the k-space
def KPath(points, min_num=200, loop=True):
    """
    Generating k-points on the path that specified by the given `points`

    If `loop` is set to `False`, the k-path is generated as follow:
        points[0] ->  ... -> points[i] -> ... -> points[N-1]
    If `loop` is set to `True`, the k-path is generated as follow:
        points[0] -> ... -> points[i] -> ... -> points[N-1] -> points[0]
    The k-points between the given `points` is generated linearly

    Parameters
    ----------
    points : 2D array with shape (N, 2) or (N, 3)
        Special points on the k-path
        Every row represent the coordinate of a k-point
        It is assumed that there are no identical adjacent points in the
        given `points` parameter
    min_num : int, optional
        The number of k-point on the shortest k-path segment
        The number of k-point on other k-path segments are scaled according
        to their length
        Default: 200
    loop : boolean, optional
        Whether to generate a k-loop or not
        Default: True

    Returns
    -------
    kpoints : 2D array with shape (N, 2) or (N, 3)
        A collection k-points on the path that specified by the given `points`
    indices : list
        The indices of the given `points` in the returned `kpoints` array
    """

    assert len(points) > 1, "At least two points are required"
    points = np.concatenate((points, points), axis=0)
    assert points.ndim == 2 and points.shape[1] in (2, 3)
    assert isinstance(min_num, int) and min_num >= 1

    point_num = points.shape[0] // 2
    end = (point_num + 1) if loop else point_num
    dRs = points[1:end] - points[0:end-1]
    lengths = np.linalg.norm(dRs, axis=-1)

    min_length = np.min(lengths)
    if min_length < 1e-4:
        raise ValueError("Adjacent points are identical!")

    sampling_nums = [
        int(min_num * length / min_length) for length in lengths
    ]
    kpoints = []
    segment_num = dRs.shape[0]
    for i in range(segment_num):
        endpoint = False if i != (segment_num - 1) else True
        ratios = np.linspace(0, 1, num=sampling_nums[i], endpoint=endpoint)
        kpoints.append(ratios[:, np.newaxis] * dRs[i] + points[i])
    kpoints = np.concatenate(kpoints, axis=0)
    indices = [0, *np.cumsum(sampling_nums)]
    indices[-1] -= 1
    return kpoints, indices


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kpoints, indices = KPath([Gamma, Ms_CELL[0], Ks_CELL[0]], loop=True)
    labels = [r"$\Gamma$", "$M$", "$K$", r"$\Gamma$"]

    # Show the First Brillouin Zone and the Gamma-M-K-Gamma path
    fig, (ax0, ax1) = plt.subplots(1, 2)
    Ks = np.append(Ks_CELL, Ks_CELL[[0]], axis=0)
    ax0.plot(Ks[:, 0], Ks[:, 1], marker="o", ms=15)
    ax0.plot(kpoints[:, 0], kpoints[:, 1])
    ax0.set_aspect("equal")
    ax0.set_axis_off()

    Es = -2 * np.sum(np.cos(np.dot(kpoints, Rs_CELL.T)), axis=-1)
    ax1.plot(Es)
    ax1.set_xlim(0, Es.shape[0])
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="both")
    plt.show()
