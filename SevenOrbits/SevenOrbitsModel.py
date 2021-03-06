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
    "STATE_NUM",
    "OnSiteInteraction",
    "HoppingTerms",
    "HTermGenerator",
]


# All possible orbits and spin-flavors on a lattice site
# The number of single-particle state on a lattice site
ORBITS = (0, 1, 2, 3, 4, 5, 6)
SPINS = (SPIN_DOWN, SPIN_UP)
ORBIT_NUM = len(ORBITS)
SPIN_NUM = len(SPINS)
STATE_NUM = ORBIT_NUM * SPIN_NUM

# Global variables that describe the hopping and interaction terms
# The integer numbers are the indices of these orbits
ONSITE_TERMS = {
    # The on-site energy difference between the central orbit and the six
    # surrounding orbits
    "Muc": [(0, 0)],
    # Hubbard interaction for electrons on the central orbit
    "Uc": [(0, 0)],
    # Hubbard interaction for electrons on the six surrounding orbits
    "Us": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)],
    # Coulomb interaction between electrons on the central orbit and the six
    # surrounding orbits
    "Usc": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
    # The hopping terms between the central orbit and the six surrounding orbits
    "tsc": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
    # Three types of intra-cluster hopping terms between the six surrounding
    # orbits
    "tss1": [(1, 4), (2, 5), (3, 6)],
    "tss3": [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)],
    "tss5": [(1, 3), (2, 4), (3, 5), (4, 6), (5, 1), (6, 2)],
}

# Three types of inter-cluster hopping terms between the six surrounding orbits
# The hopping term is bond dependent
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
    # The inter-cluster hopping terms between the central orbits
    "tcc": [(0, 0)],
    "tss2": INTER_TERMS_TSS2,
    "tss4": INTER_TERMS_TSS4,
    "tss6": INTER_TERMS_TSS6,
}


# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "Muc" : 0.2120,
    "Uc"  : 0.0000,
    "Us"  : 0.0000,
    "Usc" : 0.0000,
    "tcc" : 0.0000,
    "tsc" : 0.1620,
    "tss1": 0.1500,
    "tss2": 0.0910,
    "tss3": 0.0720,
    "tss4": 0.0500,
    "tss5": 0.0420,
    "tss6": 0.0420,
}


def update_parameters(**model_params):
    """
    Generate model parameters from the given `model_params`

    For these model parameters not specified, the default value are used

    Parameters
    ----------
    model_params: model parameters
        Currently recognized keyword arguments are:
            Muc, Uc, Us, Usc, tcc, tsc, tss1, tss2, tss3, tss4, tss5, tss6

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

    Currently implemented terms:
      1. The Hubbard interaction for electrons on the central orbit: Uc-term;
      2. The Hubbard interaction for electrons on the six surrounding orbits:
      Us-term;
      3. The Coulomb interaction between electrons on the central orbit and
      the six surrounding orbits: Usc-term;

    Parameters
    ----------
    cluster : Lattice
        The lattice based on which this model is defined
    model_params : Necessary model parameters
        Currently recognized keyword arguments are: Uc, Us, Usc;
        Other keyword arguments are ignored silently.

    Returns
    -------
    terms : list of ParticleTerm
        A collection of on-site interaction terms
    """

    terms = []
    model_params = update_parameters(**model_params)
    for key in ["Uc", "Us", "Usc"]:
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

    Currently implemented terms:
      1. The on-site energy difference between the central orbit and the six
      surrounding orbits: Muc-term;
      2. The on-site hopping terms between electrons on the central orbit and
      the six surrounding orbits: tsc-term;
      3. Three types of on-site hopping terms between the six surrounding
      orbits: tss1-, tss3-, tss5-term;
      4. The inter-site hopping terms between electrons on the central orbits:
      tcc-term;
      5. Three types of inter-site hopping terms between the six
      surrounding orbits: tss2-, tss4-, tss6-term;

    Parameters
    ----------
    cluster : Lattice
        The lattice based on which this model is defined
    model_params : Necessary model parameters
        Currently recognized keyword arguments are: Muc, tcc, tsc, tss1,
        tss2, tss3, tss4, tss5, tss6; Other keyword arguments are ignored
        silently.

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
    for key in ["tcc", "tss2", "tss4", "tss6"]:
        coeff = model_params[key]
        if coeff == 0.0:
            continue

        for container, bonds in zip(containers, all_bonds):
            for bond in bonds:
                p0, p1 = bond.getEndpoints()
                if key == "tcc":
                    orbital_pairs = INTER_TERMS[key]
                else:
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

    Currently implemented terms:
      See the documents of the `OnSiteInteraction` and `HoppingTerms`

    Parameters
    ----------
    cluster : Lattice
        The lattice based on which this model is defined
    model_params : Necessary model parameters
        Currently recognized keyword arguments are: Muc, tcc, tsc, tss1,
        tss2, tss3, tss4, tss5, tss6, Uc, Us, Usc; Other keyword arguments are
        ignored silently.

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

    numx = 2
    numy = 2
    site_num = numx * numy
    cluster = lattice_generator("triangle", num0=numx, num1=numy)

    Uc_num = site_num
    Us_num = 6 * site_num
    Usc_num = 4 * 6 * site_num

    Muc_num = SPIN_NUM * site_num
    tsc_num = 6 * SPIN_NUM * site_num
    tss1_num = 3 * SPIN_NUM * site_num
    tss3_num = 6 * SPIN_NUM * site_num
    tss5_num = 6 * SPIN_NUM * site_num

    tcc_intra_num = 5 * 2
    tcc_inter_num = 7 * 2
    tss2_intra_num = (2 * 1 + 2 * 1 + 1 * 1) * SPIN_NUM
    tss2_inter_num = (2 * 1 + 2 * 1 + 3 * 1) * SPIN_NUM
    tss4_intra_num = (2 * 1 + 2 * 1 + 1 * 1) * SPIN_NUM
    tss4_inter_num = (2 * 1 + 2 * 1 + 3 * 1) * SPIN_NUM
    tss6_intra_num = (2 * 2 + 2 * 2 + 1 * 2) * SPIN_NUM
    tss6_inter_num = (2 * 2 + 2 * 2 + 3 * 2) * SPIN_NUM

    assert len(OnSiteInteraction(cluster, Uc=1.0, Us=0.0, Usc=0.0)) == Uc_num
    assert len(OnSiteInteraction(cluster, Uc=0.0, Us=1.0, Usc=0.0)) == Us_num
    assert len(OnSiteInteraction(cluster, Uc=0.0, Us=0.0, Usc=1.0)) == Usc_num

    model_params = {
        "Muc" : 0.0,
        "Uc"  : 0.0,
        "Us"  : 0.0,
        "Usc" : 0.0,
        "tcc" : 0.0,
        "tsc" : 0.0,
        "tss1": 0.0,
        "tss2": 0.0,
        "tss3": 0.0,
        "tss4": 0.0,
        "tss5": 0.0,
        "tss6": 0.0,
    }

    for num, key in zip(
        [Muc_num, tsc_num, tss1_num, tss3_num, tss5_num],
        ["Muc", "tsc", "tss1", "tss3", "tss5"]
    ):
        new_model_params = dict(model_params)
        new_model_params[key] = 1.0
        intra_terms, inter_terms = HoppingTerms(cluster, **new_model_params)
        assert len(intra_terms) == num and len(inter_terms) == 0

    for num_intra, num_inter, key in zip(
        [tcc_intra_num, tss2_intra_num, tss4_intra_num, tss6_intra_num],
        [tcc_inter_num, tss2_inter_num, tss4_inter_num, tss6_inter_num],
        ["tcc", "tss2", "tss4", "tss6"]
    ):
        new_model_params = dict(model_params)
        new_model_params[key] = 1.0
        intra_terms, inter_terms = HoppingTerms(cluster, **new_model_params)
        assert len(intra_terms) == num_intra and len(inter_terms) == num_inter
