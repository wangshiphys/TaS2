import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import eigsh

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.hilbertspace import base_vectors
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.lattice import lattice_generator
from HamiltonianPy.termofH import AoC, StateID

from TaS2DataBase import KPath, Gamma, Ms_CELL, Ks_CELL


ORBITS = (0, 1, 2, 3, 4, 5, 6)
SPINS = (SPIN_DOWN, SPIN_UP)
ORBIT_NUM = len(ORBITS)
SPIN_NUM = len(SPINS)

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
    -180: INTER_TERMS_A, 0: INTER_TERMS_A,
    -120: INTER_TERMS_B, 60: INTER_TERMS_B,
    -60: INTER_TERMS_C, 120: INTER_TERMS_C,
}


def HTermGenerator(cluster, **kwargs):
    onsite_term_coeffs = [
        kwargs["mu_c"] / 2, kwargs["t_sc"],
        kwargs["t_ss1"], kwargs["t_ss3"], kwargs["t_ss5"],
    ]
    inter_terms_coeff = [
        kwargs["t_ss2"], kwargs["t_ss4"], kwargs["t_ss6"], kwargs["t_ss6"],
    ]

    HTerms = []
    for point in cluster.points:
        for spin in SPINS:
            for coeff, terms in zip(onsite_term_coeffs, ONSITE_TERMS):
                for orbit0, orbit1 in terms:
                    c = AoC(CREATION, site=point, spin=spin, orbit=orbit0)
                    a = AoC(ANNIHILATION, site=point, spin=spin, orbit=orbit1)
                    HTerms.append(coeff * c * a)

    bulk_bonds, boundary_bonds = cluster.bonds(nth=1)
    for bond in bulk_bonds + boundary_bonds:
        inter_terms = INTER_TERMS[bond.getAzimuth(ndigits=0)]
        p0, p1 = bond.getEndpoints()
        for spin in SPINS:
            for coeff, (orbit0, orbit1) in zip(inter_terms_coeff, inter_terms):
                c = AoC(CREATION, site=p0, spin=spin, orbit=orbit0)
                a = AoC(ANNIHILATION, site=p1, spin=spin, orbit=orbit1)
                HTerms.append(coeff * c * a)

    return HTerms


def RealSpaceSolver(numx=2, numy=2, e_num=13, **kwargs):
    site_num = numx * numy
    state_num = site_num * ORBIT_NUM * SPIN_NUM
    tmp = site_num * e_num
    assert tmp == int(tmp)
    e_num_total = int(tmp)

    cluster = lattice_generator("triangle", num0=numx, num1=numy)
    HTerms = HTermGenerator(cluster, **kwargs)
    state_indices_table = IndexTable(
        StateID(site=point, spin=spin, orbit=orbit)
        for point in cluster.points for spin in SPINS for orbit in ORBITS
    )

    HM = np.zeros((state_num, state_num), dtype=np.float64)
    for term in HTerms:
        coeff = term.coeff
        c, a = term.components
        p0, trash = cluster.decompose(c.site)
        p1, trash = cluster.decompose(a.site)
        c_eqv = c.derive(site=p0)
        a_eqv = a.derive(site=p1)
        row = c_eqv.getStateIndex(state_indices_table)
        col = a_eqv.getStateIndex(state_indices_table)
        HM[row, col] += coeff
    HM += HM.T.conj()
    values, vectors = np.linalg.eigh(HM)
    GE = np.sum(values[0:e_num_total]/site_num)
    print("The ground state energy: {0}".format(GE))

    avg_num_array = np.zeros((ORBIT_NUM, site_num), dtype=np.float64)
    probabilities = (vectors * vectors.conj()).real
    for row, orbit in enumerate(ORBITS):
        for col, point in enumerate(cluster.points):
            avg_num = 0
            for spin in SPINS:
                which = StateID(point, spin=spin, orbit=orbit).getIndex(
                        state_indices_table
                    )
                avg_num += np.sum(probabilities[which, 0:e_num_total])
            avg_num_array[row, col] = avg_num

    avg_num_site = np.sum(avg_num_array, axis=0)
    msg = "The averaged number of electrons on the {0}th site is: {1}"
    for index, point in enumerate(cluster.points):
        print(msg.format(index, avg_num_site[index]))
    avg_num_orbit = np.sum(avg_num_array, axis=1)
    msg = "The averaged number of electrons on the {0}th orbit is : {1}"
    for index, orbit in enumerate(ORBITS):
        print(msg.format(index, avg_num_orbit[index]))
    avg_num_per_site = np.sum(avg_num_array) / site_num
    msg = "The averaged number of electrons per-site: {0}"
    print(msg.format(avg_num_per_site))
    print("=" * 80)


def OccupationSolver(numx=2, numy=2, e_num=13, **kwargs):
    site_num = numx * numy
    state_num = site_num * ORBIT_NUM * SPIN_NUM
    tmp = site_num * e_num
    assert tmp == int(tmp)
    e_num_total = int(tmp)

    cluster = lattice_generator("triangle", num0=numx, num1=numy)
    HTerms = HTermGenerator(cluster, **kwargs)
    state_indices_table = IndexTable(
        StateID(site=point, spin=spin, orbit=orbit)
        for point in cluster.points for spin in SPINS for orbit in ORBITS
    )

    HM = 0.0
    right_bases = base_vectors((state_num, e_num_total))
    for term in HTerms:
        c, a = term.components
        p0, trash = cluster.decompose(c.site)
        p1, trash = cluster.decompose(a.site)
        c_eqv = c.derive(site=p0)
        a_eqv = a.derive(site=p1)
        HM += (term.coeff * c_eqv * a_eqv).matrix_repr(
            state_indices_table, right_bases
        )
    HM += HM.getH()
    if HM.shape[0] > 5000:
        values, vectors = eigsh(HM, k=1, which="SA")
    else:
        values, vectors = np.linalg.eigh(HM.toarray())
    GE = values[0] / site_num
    print("The ground state energy: {0}".format(GE))

    avg_num_array = np.zeros((ORBIT_NUM, site_num), dtype=np.float64)
    for row, orbit in enumerate(ORBITS):
        for col, point in enumerate(cluster.points):
            NM = 0.0
            for spin in SPINS:
                c = AoC(CREATION, site=point, spin=spin, orbit=orbit)
                a = AoC(ANNIHILATION, site=point, spin=spin, orbit=orbit)
                NM += (c * a).matrix_repr(
                    state_indices_table, right_bases
                )
            avg_num_array[row, col] = np.vdot(
                vectors[:, 0], NM.dot(vectors[:, 0])
            )
    avg_num_site = np.sum(avg_num_array, axis=0)
    msg = "The averaged number of electrons on the {0}th site is: {1}"
    for index, point in enumerate(cluster.points):
        print(msg.format(index, avg_num_site[index]))
    avg_num_orbit = np.sum(avg_num_array, axis=1)
    msg = "The averaged number of electrons on the {0}th orbit is : {1}"
    for index, orbit in enumerate(ORBITS):
        print(msg.format(index, avg_num_orbit[index]))
    avg_num_per_site = np.sum(avg_num_array) / site_num
    msg = "The averaged number of electrons per-site: {0}"
    print(msg.format(avg_num_per_site))
    print("=" * 80)


def KSpaceSolver(**kwargs):
    state_num = ORBIT_NUM * SPIN_NUM
    cluster = lattice_generator("triangle", num0=1, num1=1)
    HTerms = HTermGenerator(cluster, **kwargs)
    state_indices_table = IndexTable(
        StateID(site=point, spin=spin, orbit=orbit)
        for spin in SPINS for point in cluster.points  for orbit in ORBITS
    )
    GMKGPath, GMKGIndices = KPath([Gamma, Ms_CELL[0], Ks_CELL[0]], min_num=500)
    shape = (GMKGPath.shape[0], state_num, state_num)
    HM = np.zeros(shape, dtype=np.complex128)
    for term in HTerms:
        c, a = term.components
        p0, dR0 = cluster.decompose(c.site)
        p1, dR1 = cluster.decompose(a.site)
        c_eqv = c.derive(site=p0)
        a_eqv = a.derive(site=p1)
        index0 = c_eqv.getStateIndex(state_indices_table)
        index1 = a_eqv.getStateIndex(state_indices_table)
        HM[:, index0, index1] += term.coeff * np.exp(
            1j * np.dot(GMKGPath, dR1 - dR0)
        )
    HM += np.transpose(HM.conj(), axes=(0, 2, 1))
    values, vectors = np.linalg.eigh(HM)

    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_xlim(0, len(values)-1)
    ax.set_xticks(GMKGIndices)
    ax.set_xticklabels([r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    ax.grid(axis="x", ls="dashed")
    plt.show()


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

    RealSpaceSolver(**model_params)
    KSpaceSolver(**model_params)
    OccupationSolver(**model_params)
