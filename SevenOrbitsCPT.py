"""
Solving the Seven-Orbits Hubbard model using cluster perturbation theory
"""


import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import *

from SevenOrbitsModel import *


def VMatrixTemplate(HTerms_Inter, cluster, row_aoc_indices_table):
    template = []
    for term in HTerms_Inter:
        c, a = term.components
        p0, dR0 = cluster.decompose(c.site)
        p1, dR1 = cluster.decompose(a.site)
        c_eqv = c.derive(site=p0)
        a_eqv = a.derive(site=p1)
        index0 = c_eqv.getIndex(row_aoc_indices_table)
        index1 = a_eqv.dagger().getIndex(row_aoc_indices_table)
        template.append((index0, index1, term.coeff, dR0 - dR1))
    return template

def VMatrixGenerator(kpoint, template, shape):
    VM = np.zeros(shape, dtype=np.complex128)
    for row, col, coeff, dR in template:
        VM[row, col] += coeff * np.exp(-1j * np.dot(kpoint, dR))
    VM += VM.T.conj()
    return VM


def CoreFunc(
        numx=1, numy=1, e_num_per_unit_cell=13, lanczos=False, **model_params,
):
    site_num = numx * numy
    state_num = site_num * ORBIT_NUM * SPIN_NUM
    e_num_total = site_num * e_num_per_unit_cell
    assert e_num_total == int(e_num_total)
    e_num_total = int(e_num_total)

    bases = (
        base_vectors((state_num, e_num_total - 1)),
        base_vectors((state_num, e_num_total)),
        base_vectors((state_num, e_num_total + 1)),
    )

    cluster = lattice_generator("triangle", num0=numx, num1=numy)
    Interactions, Hopping_Intra, Hopping_Inter = HTermGenerator(
        cluster, **model_params
    )
    HTerms_Intra = Hopping_Intra + Interactions

    state_indices_table_cluster = IndexTable(
        StateID(site=site, spin=spin, orbit=orbit)
        for site in cluster.points for spin in SPINS for orbit in ORBITS
    )
    row_aoc_indices_table_cluster = IndexTable(
        AoC(CREATION, site=site, spin=spin, orbit=orbit)
        for site in cluster.points for spin in SPINS for orbit in ORBITS
    )

    v_matrix_template = VMatrixTemplate(
        Hopping_Inter, cluster, row_aoc_indices_table_cluster
    )
    cluster_gf_solver = ClusterGFSolver(
        HTerms_Intra, state_indices_table_cluster, bases,
        row_aoc_indices_table_cluster, lanczos=lanczos, conserved=True
    )
    return v_matrix_template, cluster_gf_solver


def CPTEB(
        omegas, kpath, numx=1, numy=1, e_num_per_unit_cell=13, lanczos=False,
        eta=0.01j, **model_params
):
    site_num = numx * numy
    state_num = site_num * SPIN_NUM * ORBIT_NUM
    v_matrix_shape = (state_num, state_num)
    v_matrix_template, cluster_gf_solver = CoreFunc(
        numx, numy, e_num_per_unit_cell, lanczos=lanczos, **model_params
    )

    cluster_gfs_inv = np.linalg.inv(
        [cluster_gf_solver(omega) for omega in (omegas + eta)]
    )

    VMatrices = np.array(
        [
            VMatrixGenerator(kpoint, v_matrix_template, v_matrix_shape)
            for kpoint in kpath
        ]
    )

    spectrums = -np.array([
        np.trace(np.linalg.inv(cluster_gf_inv - VMatrices), axis1=1, axis2=2)
        for cluster_gf_inv in cluster_gfs_inv
    ]).imag / site_num / np.pi

    fig, ax = plt.subplots()
    cs = ax.contourf(
        range(len(kpath)), omegas, spectrums, levels=200, cmap="hot"
    )
    ax.grid(axis="x", ls="dashed")
    fig.colorbar(cs, ax=ax)
    plt.show()
    plt.close("all")


def CPTDOS(
        omegas, numx=1, numy=1, e_num_per_unit_cell=13, lanczos=False, numk=200,
        eta=0.01j, **model_params
):
    site_num = numx * numy
    state_num = site_num * SPIN_NUM * ORBIT_NUM
    v_matrix_shape = (state_num, state_num)
    v_matrix_template, cluster_gf_solver = CoreFunc(
        numx, numy, e_num_per_unit_cell, lanczos=lanczos, **model_params
    )

    cluster_gfs_inv = np.linalg.inv(
        [cluster_gf_solver(omega) for omega in (omegas + eta)]
    )

    cluster = lattice_generator("triangle", num0=numx, num1=numy)
    ratio = np.linspace(0, 1, num=numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    BZMesh = np.dot(ratio_mesh, cluster.bs)
    VMatrices = np.array(
        [
            VMatrixGenerator(kpoint, v_matrix_template, v_matrix_shape)
            for kpoint in BZMesh
        ]
    )
    del cluster, ratio, ratio_mesh, BZMesh

    dos = (-np.pi / numk / numk) * np.array(
        [
            np.sum(
                np.trace(
                    np.linalg.inv(cluster_gf_inv - VMatrices), axis1=1, axis2=2
                ).imag
            ) for cluster_gf_inv in cluster_gfs_inv
        ]
    )

    file_name = "data/" + ",".join(
        "{0}={1:.3f}".format(key, value)
        for key,value in model_params.items()
    )
    np.savez(file_name, omegas=omegas, dos=dos)

    fig, ax = plt.subplots()
    ax.plot(omegas, dos)
    ax.set_xlim(omegas[0], omegas[-1])
    plt.show()
    fig.savefig(file_name + ".jpg", dpi=300)
    plt.close("all")
