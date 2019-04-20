"""
Solve the Seven-Orbits Hubbard model based on cluster perturbation theory(CPT)
"""


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from HamiltonianPy import *
from scipy.optimize import brentq

from SevenOrbits.SevenOrbitsModel import *


def VMatrixGenerator(kpoints, VTerms, cluster, row_aoc_indices_table):
    kpoints_num = len(kpoints)
    aocs_num = len(row_aoc_indices_table)
    VM = np.zeros((kpoints_num, aocs_num, aocs_num), dtype=np.complex128)
    for term in VTerms:
        c, a = term.components
        p0, dR0 = cluster.decompose(c.site)
        p1, dR1 = cluster.decompose(a.site)
        row_index = row_aoc_indices_table(c.derive(site=p0))
        col_index = row_aoc_indices_table(a.derive(site=p1).dagger())
        VM[:, row_index, col_index] += term.coeff * np.exp(
            1j * np.dot(kpoints, dR1 - dR0)
        )
    VM += np.transpose(VM.conj(), axes=(0, 2, 1))
    return VM


def _MuCore(mu, dos, omegas, occupied_num=13, total_num=14, reverse=False):
    delta_omega = omegas[1] - omegas[0]
    if reverse:
        num = total_num - occupied_num
        indices = omegas > mu
    else:
        num = occupied_num
        indices = omegas < mu
    return np.sum(dos[indices]) * delta_omega - num

def Mu(dos, omegas, occupied_num=13, total_num=14, reverse=False):
    args = (dos, omegas, occupied_num, total_num, reverse)
    return brentq(_MuCore, a=omegas[0], b=omegas[-1], args=args)


def CPTEB(
        omegas, kpath, numx=1, numy=1, e_num=13,
        lanczos=False, eta=0.01, **model_params
):
    site_num = numx * numy
    state_num_total = site_num * STATE_NUM
    e_num_total = site_num * e_num
    assert e_num_total == int(e_num_total)
    e_num_total = int(e_num_total)

    bases_pack = (
        base_vectors((state_num_total, e_num_total - 1)),
        base_vectors((state_num_total, e_num_total)),
        base_vectors((state_num_total, e_num_total + 1)),
    )

    cell = lattice_generator("triangle", num0=1, num1=1)
    cluster = lattice_generator("triangle", num0=numx, num1=numy)
    state_indices_table_cluster = IndexTable(
        StateID(site=site, spin=spin, orbit=orbit)
        for site in cluster.points for orbit in ORBITS for spin in SPINS
    )
    row_aoc_indices_table_cell = IndexTable(
        AoC(CREATION, site=site, spin=spin, orbit=orbit)
        for site in cell.points for orbit in ORBITS for spin in SPINS
    )
    row_aoc_indices_table_cluster = IndexTable(
        AoC(CREATION, site=site, spin=spin, orbit=orbit)
        for site in cluster.points for orbit in ORBITS for spin in SPINS
    )

    HTermsIntra, HTermsInter = HoppingTerms(cluster, **model_params)
    HTermsIntra += OnSiteInteraction(cluster, **model_params)

    cluster_gf_solver = ClusterGFSolver(
        HTermsIntra, state_indices_table_cluster, bases_pack,
        row_aoc_indices_table_cluster, lanczos=lanczos, conserved=True
    )
    cluster_gfs_inv = np.linalg.inv(
        [cluster_gf_solver(omega) for omega in (omegas + eta * 1j)]
    )

    VMatrices = VMatrixGenerator(
        kpath, HTermsInter, cluster, row_aoc_indices_table_cluster
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
        omegas, numk=200, numx=1, numy=1, e_num=13,
        lanczos=False, eta=0.01, save_data=True, **model_params
):
    site_num = numx * numy
    state_num_total = site_num * STATE_NUM
    e_num_total = site_num * e_num
    assert e_num_total == int(e_num_total)
    e_num_total = int(e_num_total)

    bases_pack = (
        base_vectors((state_num_total, e_num_total - 1)),
        base_vectors((state_num_total, e_num_total)),
        base_vectors((state_num_total, e_num_total + 1)),
    )

    cluster = lattice_generator("triangle", num0=numx, num1=numy)
    state_indices_table_cluster = IndexTable(
        StateID(site=site, spin=spin, orbit=orbit)
        for site in cluster.points for orbit in ORBITS for spin in SPINS
    )
    row_aoc_indices_table_cluster = IndexTable(
        AoC(CREATION, site=site, spin=spin, orbit=orbit)
        for site in cluster.points for orbit in ORBITS for spin in SPINS
    )

    HTermsIntra, HTermsInter = HoppingTerms(cluster, **model_params)
    HTermsIntra += OnSiteInteraction(cluster, **model_params)

    cluster_gf_solver = ClusterGFSolver(
        HTermsIntra, state_indices_table_cluster, bases_pack,
        row_aoc_indices_table_cluster, lanczos=lanczos, conserved=True
    )
    cluster_gfs_inv = np.linalg.inv(
        [cluster_gf_solver(omega) for omega in (omegas + eta * 1j)]
    )

    ratio = np.linspace(0, 1, num=numk, endpoint=False)
    ratio_mesh = np.stack(
        np.meshgrid(ratio, ratio, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    BZMesh = np.dot(ratio_mesh, cluster.bs)
    del ratio, ratio_mesh

    VMatrices = VMatrixGenerator(
        BZMesh, HTermsInter, cluster, row_aoc_indices_table_cluster
    )
    del BZMesh

    dos = -np.array(
        [
            np.sum(
                np.trace(
                    np.linalg.inv(cluster_gf_inv - VMatrices), axis1=1, axis2=2
                ).imag
            ) for cluster_gf_inv in cluster_gfs_inv
        ]
    ) / (np.pi * numk * numk)
    dos_sum = np.sum(dos) * (omegas[1] - omegas[0])
    print("The integration of DoS: {0}".format(dos_sum))
    mup = Mu(dos, omegas, e_num_total, state_num_total, reverse=False)
    muh = Mu(dos, omegas, e_num_total, state_num_total, reverse=True)
    print("The chemical potential from particle point of view: {0}".format(mup))
    print("The chemical potential from hole point of view: {0}".format(muh))

    fig, ax = plt.subplots()
    ax.plot(omegas, dos)
    ax.axvline(mup, ls="dashed", color="gray")
    ax.axvline(muh, ls="dotted", color="gray")
    ax.set_xlim(omegas[0], omegas[-1])

    plt.show()
    if save_data:
        data_path = Path("data/eta={0:.3f}/".format(eta))
        fig_path = Path("fig/eta={0:.3f}/".format(eta))
        data_path.mkdir(parents=True, exist_ok=True)
        fig_path.mkdir(parents=True, exist_ok=True)
        file_name = "DOS at " + ",".join(
            "{0}={1:.3f}".format(key, model_params[key])
            for key in sorted(model_params)
        )
        np.savez(data_path / (file_name + ".npz"), omegas=omegas, dos=dos)
        fig.savefig(fig_path / (file_name + ".jpg"), dpi=300)
    plt.close("all")
