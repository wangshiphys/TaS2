"""
Construct the Seven-Orbits model Hamiltonian proposed by Shuang Qiao et al.

See also:
    Shuang Qiao et al., Phys. Rev. X 7, 041054 (2017)
"""


from itertools import product

from HamiltonianPy import AoC, ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP

__all__ = [
    "ORBITS", "ORBIT_NUM",
    "SPINS", "SPIN_NUM",
    "OnSiteInteraction",
    "HoppingTerms",
    "HTermGenerator",
]


# All possible orbits and spin-flavors on a lattice site
ORBITS = (0, 1, 2, 3, 4, 5, 6)
SPINS = (SPIN_DOWN, SPIN_UP)
ORBIT_NUM = len(ORBITS)
SPIN_NUM = len(SPINS)


# Global variables that describe the hopping and interaction terms
# The integer numbers are the indices of these orbits
ONSITE_TERMS = {
    "Muc": [(0, 0)],
    # Hubbard interaction for electrons on the 0th orbit
    "Uc": [(0, 0)],
    # Coulomb interaction for electrons on different orbits
    "Usc": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
    "tsc": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
    "tss1": [(1, 4), (2, 5), (3, 6)],
    "tss3": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)],
    "tss5": [(1, 3), (2, 4), (3, 5), (4, 6), (5, 1), (6, 2)],
}
INTER_TERMS_TSS2 = {
    -180: [(3, 6)], 0  : [(3, 6)],
    -120: [(2, 5)], 60 : [(2, 5)],
    -60 : [(1, 4)], 120: [(1, 4)],
}
INTER_TERMS_TSS4 = {
    -180: [(1, 4)], 0  : [(1, 4)],
    -120: [(6, 3)], 60 : [(6, 3)],
    -60 : [(5, 2)], 120: [(5, 2)],
}
INTER_TERMS_TSS6 = {
    -180: [(3, 1), (4, 6)], 0  : [(3, 1), (4, 6)],
    -120: [(2, 6), (3, 5)], 60 : [(2, 6), (3, 5)],
    -60 : [(1, 5), (2, 4)], 120: [(1, 5), (2, 4)],
}
INTER_TERMS = {
    "tss2": INTER_TERMS_TSS2,
    "tss4": INTER_TERMS_TSS4,
    "tss6": INTER_TERMS_TSS6,
}


# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "Muc": 0.212,
    "tsc": 0.162,
    "Uc": 0.0,
    "Usc": 0.0,
    "tss1": 0.150,
    "tss2": 0.091,
    "tss3": 0.072,
    "tss4": 0.050,
    "tss5": 0.042,
    "tss6": 0.042,
}


def update_parameters(**model_params):
    """
    Generate model parameters from the given `model_params`

    For these model parameters not specified, the default value are used

    Parameters
    ----------
    model_params: model parameters
        Currently recognized keyword arguments are:
          Muc, tsc, Uc, Usc, tss1, tss2, tss3, tss4, tss5, tss6

    Returns
    -------
    res : dict
        Model parameters composed of the given values and defaults
    """

    new_model_params = dict(DEFAULT_MODEL_PARAMS)
    new_model_params.update(model_params)
    return new_model_params


def NumberOperator(site, spin=0, orbit=0):
    """
    Create particle-number-operator for the state specified by the given
    parameters

    Parameters
    ----------
    site : np.ndarray
        The coordinate of the localized single-particle state
    spin : int, optional
        The spin index of the single-particle state
        default: 0
    orbit : int, optional
        The orbit index of the single-particle state
        default: 0

    Returns
    -------
    res : ParticleTerm
        The particle-number operator
    """

    C = AoC(CREATION, site=site, spin=spin, orbit=orbit)
    A = AoC(ANNIHILATION, site=site, spin=spin, orbit=orbit)
    return C * A


def OnSiteInteraction(cluster, **model_params):
    """
    Generate on-site interaction terms

    Current implemented terms:
      1. The Hubbard interaction for electrons on the 0th orbit |0>;
      2. The Coulomb interaction between electrons on the 0th orbit and other
      six orbits.

    Parameters
    ----------
    cluster : Lattice
        The lattice based on which this model is defined
    model_params : Necessary model parameters

    Returns
    -------
    terms : list of ParticleTerm
        A collection of on-site interaction terms
    """

    terms = []
    model_params = update_parameters(**model_params)
    for key in ["Uc", "Usc"]:
        coeff = model_params[key] / 2.0
        if coeff == 0.0:
            continue

        orbital_pairs = ONSITE_TERMS[key]
        for site in cluster.points:
            for orbit0, orbit1 in orbital_pairs:
                N0_UP = NumberOperator(site, SPIN_UP, orbit0)
                N0_DOWN = NumberOperator(site, SPIN_DOWN, orbit0)
                if orbit0 == orbit1:
                    terms.append(coeff * N0_UP * N0_DOWN)
                else:
                    N1_UP = NumberOperator(site, SPIN_UP, orbit1)
                    N1_DOWN = NumberOperator(site, SPIN_DOWN, orbit1)
                    for N0, N1 in product([N0_UP, N0_DOWN], [N1_UP, N1_DOWN]):
                        terms.append(coeff * N0 * N1)
    return terms


def HoppingTerms(cluster, **model_params):
    """
    Generate hopping terms

    Current implemented terms:
      1. The on-site chemical potential term on the 0th orbit;
      2. The on-site hopping terms between the 0th orbit and other six orbits;
      3. Three types of on-site hopping terms between the six extended orbits;
      4. Three types of inter-site hopping terms between the six extended
      orbits.

    Parameters
    ----------
    cluster : Lattice
        The lattice based on which this model is defined
    model_params : Necessary model parameters

    Returns
    -------
    terms_intra : list of ParticleTerm
        A collection of hopping terms intra the cluster
    terms_inter : list of ParticleTerm
        A collection of hopping terms between the cluster
    """

    hopping_terms_intra = []
    hopping_terms_inter = []
    model_params = update_parameters(**model_params)
    for key in ["Muc", "tsc", "tss1", "tss3", "tss5"]:
        coeff = model_params[key]
        if key == "Muc":
            coeff /= 2.0
        if coeff == 0.0:
            continue

        orbital_pairs = ONSITE_TERMS[key]
        for site in cluster.points:
            for spin in SPINS:
                for orbit0, orbit1 in orbital_pairs:
                    C = AoC(CREATION, site=site, spin=spin, orbit=orbit0)
                    A = AoC(ANNIHILATION, site=site, spin=spin, orbit=orbit1)
                    hopping_terms_intra.append(coeff * C * A)

    bulk_bonds, boundary_bonds = cluster.bonds(nth=1)
    containers = [hopping_terms_intra, hopping_terms_inter]
    all_bonds = [bulk_bonds, boundary_bonds]
    for key in ["tss2", "tss4", "tss6"]:
        coeff = model_params[key]
        if coeff == 0.0:
            continue

        for container, bonds in zip(containers, all_bonds):
            for bond in bonds:
                p0, p1 = bond.getEndpoints()
                orbital_pairs = INTER_TERMS[key][bond.getAzimuth(ndigits=0)]
                for spin in SPINS:
                    for orbit0, orbit1 in orbital_pairs:
                        C = AoC(CREATION, site=p0, spin=spin, orbit=orbit0)
                        A = AoC(ANNIHILATION, site=p1, spin=spin, orbit=orbit1)
                        container.append(coeff * C * A)
    return hopping_terms_intra, hopping_terms_inter


def HTermGenerator(cluster, **model_params):
    """
    Generate terms of the model Hamiltonian

    Current implemented terms:
      See the documents of the `OnSiteInteraction` and `HoppingTerms`

    Parameters
    ----------
    cluster : Lattice
        The lattice based on which this model is defined
    model_params : Necessary model parameters

    Returns
    -------
    terms_interaction : list of ParticleTerm
        A collection of on-site interaction terms
    terms_intra : list of ParticleTerm
        A collection of hopping terms intra the cluster
    terms_inter : list of ParticleTerm
        A collection of hopping terms between the cluster
    """

    terms_interaction = OnSiteInteraction(cluster, **model_params)
    terms_intra, terms_inter = HoppingTerms(cluster, **model_params)
    return terms_interaction, terms_intra, terms_inter


if __name__ == "__main__":
    from HamiltonianPy import lattice_generator
    cluster = lattice_generator("triangle", num0=2, num1=2)
    interactions, intra, inter = HTermGenerator(cluster, Uc=1.0, Usc=0.5)
    assert len(interactions) == 100
    assert len(intra) == 216
    assert len(inter) == 56
