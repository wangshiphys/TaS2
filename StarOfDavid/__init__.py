"""
The Star-of-David tight-binding model was proposed to simulate the band
structure of the commensurate charge-density-wave(CCDW) phase of 1T-TaS2. At
low temperatures, 1T-TaS2 undergoes a commensurate reconstruction into a
Star-of-David unit-cell containing 13 Ta atoms. There are three inequivalent
types of Ta atoms in the Star-of-David unit-cell designated as "A", "B",
and "C". In this script, the three types of Ta atoms are represented by three
different COLORS: "blue", "orange" and "green". The Star-of-David unit-cell
are centered on atoms of type "A" and characterized by a shrinkage of the
"AB", "BC" and "AC" inter-atomic distances by 6.4, 3.2 and 4.28%,
respectively. The super-lattice has 6-fold rotation symmetry.

After the reconstruction, the original nearest-neighbor bonds are of
different length, so does the hopping amplitude. There are two kinds of bond
length within the unit-cells and three kinds between the adjacent unit-cells.
We also treat the center Ta atoms specially, so hopping on the "AB" bonds are
taken to be different from that on the "BB" bonds. The hopping amplitudes for
six types of bonds are t0, t1, t2, t3, t4, t5 respectively.
"""


# Different types of points in the Star-of-David unit-cell
# The points belong to different types are specified by their indices
# See also `COORDS_STAR` in the `TaS2DataBase` module for the indices definition
POINT_TYPE_A = (0, )
POINT_TYPE_B = (1, 2, 3, 4, 5, 6)
POINT_TYPE_C = (7, 8, 9, 10, 11, 12)

# Different types of bonds in and between the Star-of-David unit-cells
# The bonds are classified according to their length
# The bonds are specified by two points and every points are specified by the
# Star-of-David unit-cell they belong and their indices in the Star-of-David
# unit-cell
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

# Bonds intra and inter the Star-of-David unit-cell
BONDS_INTRA = (BOND_TYPE_A, BOND_TYPE_B, BOND_TYPE_C)
BONDS_INTER = (BOND_TYPE_D, BOND_TYPE_E, BOND_TYPE_F)
ALL_BONDS = (
    BOND_TYPE_A, BOND_TYPE_B, BOND_TYPE_C,
    BOND_TYPE_D, BOND_TYPE_E, BOND_TYPE_F,
)
