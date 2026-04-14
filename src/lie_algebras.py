"""
lie_algebras.py
===============
Pre-defined structure constants for Lie algebras appearing in
Thurston's geometrization program and classical Lie theory.

All indices are **0-based**.  The structure constants c^k_{ij} satisfy

    [e_i, e_j] = sum_k  c^k_{ij}  e_k,

and are encoded as sparse dicts  {(i, j, k): value}.

The antisymmetry  c^k_{ij} = -c^k_{ji}  is included explicitly
(both signs are listed) so that the tensor is self-consistent.

References
----------
- P. Scott, "The Geometries of 3-Manifolds," Bull. London Math. Soc.
  15 (1983) 401–487.
- W. Thurston, "Three-Dimensional Geometry and Topology," Vol. 1,
  Princeton University Press, 1997.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Union

StructureConstants = Dict[Tuple[int, int, int], Union[int, float]]


# ---------------------------------------------------------------------------
#  Helper
# ---------------------------------------------------------------------------

def _antisymmetrize(sc_half: StructureConstants) -> StructureConstants:
    """
    Given structure constants for i < j only, return the full antisymmetric
    dict including the i > j entries.
    """
    full: StructureConstants = {}
    for (i, j, k), v in sc_half.items():
        full[(i, j, k)] = v
        full[(j, i, k)] = -v
    return full


# ---------------------------------------------------------------------------
#  Classical 3-dimensional Lie algebras
# ---------------------------------------------------------------------------

def sl2(field: str = "R") -> Tuple[StructureConstants, int, List[str]]:
    """
    sl(2, K)  for K = R or C.

    Basis:  h (e_0),  e (e_1),  f (e_2).
    Commutation relations:
        [h, e] =  2e,   i.e.  c^1_{01} = 2
        [h, f] = -2f,   i.e.  c^2_{02} = -2
        [e, f] =  h,    i.e.  c^0_{12} = 1

    This is also the Lie algebra of SL(2, K) and PSL(2, K).
    """
    sc = _antisymmetrize({
        (0, 1, 1):  2,   # [h, e] = 2e
        (0, 2, 2): -2,   # [h, f] = -2f
        (1, 2, 0):  1,   # [e, f] = h
    })
    return sc, 3, ["h", "e", "f"]


def so3() -> Tuple[StructureConstants, int, List[str]]:
    """
    so(3) ~ su(2).

    Basis:  e_0, e_1, e_2  (or L_x, L_y, L_z in physics notation).
    Commutation relations:
        [e_0, e_1] = e_2
        [e_1, e_2] = e_0
        [e_2, e_0] = e_1

    This is the Lie algebra of SO(3) and SU(2).
    The Killing form is negative-definite (compact type).
    """
    sc = _antisymmetrize({
        (0, 1, 2): 1,
        (1, 2, 0): 1,
        (2, 0, 1): 1,
    })
    return sc, 3, ["e_0", "e_1", "e_2"]


def sol() -> Tuple[StructureConstants, int, List[str]]:
    """
    sol  — the Lie algebra of the Sol geometry.

    Basis:  e_1 (e_0), e_2 (e_1), e_3 (e_2)  (using 0-based indices).
    Commutation relations (Scott's conventions):
        [e_1, e_3] =  e_1,   i.e.  c^0_{02} = 1
        [e_2, e_3] = -e_2,   i.e.  c^1_{12} = -1

    Remark: The Sol geometry does NOT admit an Ad-invariant
    non-degenerate symmetric bilinear form; the computation
    confirms this (det = 0 identically).
    """
    sc = _antisymmetrize({
        (0, 2, 0):  1,   # [e_1, e_3] = e_1
        (1, 2, 1): -1,   # [e_2, e_3] = -e_2
    })
    return sc, 3, ["e_1", "e_2", "e_3"]


# ---------------------------------------------------------------------------
#  Four-dimensional Lie algebras (Thurston geometries)
# ---------------------------------------------------------------------------

def nil_geometry() -> Tuple[StructureConstants, int, List[str]]:
    """
    Lie algebra of the Nil geometry (= Heisenberg group x R).

    Basis:  X (e_0), Y (e_1), Z (e_2), R (e_3).
    Commutation relations:
        [X, Y] =  Z,    i.e.  c^2_{01} = 1
        [R, X] =  Y,    i.e.  c^1_{30} = 1
        [R, Y] = -X,    i.e.  c^0_{31} = -1

    The isometry group of Nil has Lie algebra  nil x R,
    where nil is the 3-dimensional Heisenberg algebra.
    """
    sc = _antisymmetrize({
        (0, 1, 2):  1,   # [X, Y] = Z
        (3, 0, 1):  1,   # [R, X] = Y
        (3, 1, 0): -1,   # [R, Y] = -X
    })
    return sc, 4, ["X", "Y", "Z", "R"]


def h2_times_r() -> Tuple[StructureConstants, int, List[str]]:
    """
    Lie algebra of the H²×R geometry.

    The isometry group of H²×R has Lie algebra  sl(2,R) x R.
    We write the basis as  e_0, e_1, e_2  (sl(2,R) part) and
    e_3  (R part).

    Commutation relations for sl(2, R):
        [e_0, e_1] =  e_2,   c^2_{01} = 1
        [e_1, e_2] =  e_0,   c^0_{12} = 1
        [e_2, e_0] = -e_1,   c^1_{20} = -1
      (These correspond to the so(2,1) ~ sl(2,R) presentation
       via  h = e_0-e_2,  e = e_0+e_2, f = e_1.)

    e_3 commutes with everything (central R factor).

    Note: the computation in the Mathematica notebook uses the
    structure constants
        c^2_{01}=1, c^0_{12}=1, c^1_{20}=-1
    which give so(2,1) ~ sl(2,R).  The result has a 2-dimensional
    space of invariant forms: one from sl(2,R) and one from R.
    """
    sc = _antisymmetrize({
        (0, 1, 2):  1,   # [e_0, e_1] = e_2
        (1, 2, 0):  1,   # [e_1, e_2] = e_0
        (2, 0, 1): -1,   # [e_2, e_0] = -e_1
    })
    return sc, 4, ["e_0", "e_1", "e_2", "e_3"]


def s2_times_r() -> Tuple[StructureConstants, int, List[str]]:
    """
    Lie algebra of the S²×R geometry.

    The isometry group of S²×R has Lie algebra  so(3) x R.
    Basis:  e_0, e_1, e_2  (so(3) part), e_3 (R part).
    Commutation relations:
        [e_0, e_1] = e_2
        [e_1, e_2] = e_0
        [e_2, e_0] = e_1
    e_3 commutes with everything.
    """
    sc = _antisymmetrize({
        (0, 1, 2): 1,
        (1, 2, 0): 1,
        (2, 0, 1): 1,
    })
    return sc, 4, ["e_0", "e_1", "e_2", "e_3"]


def s3_geometry() -> Tuple[StructureConstants, int, List[str]]:
    """
    CS gauge algebra for the S³ geometry: su(2) (dim = 3).

    **Caution — gauge group vs. isometry group:**
    The full isometry group is Isom_0^+(S³) = SO(4), whose Lie algebra
    is so(4) ≅ su(2) ⊕ su(2)  (dim = 6); see so4().
    In Chern–Simons theory for S³-geometric manifolds the gauge group is
    conventionally taken to be SU(2) (or SO(3)), not SO(4).
    This function returns the 3-dimensional su(2) ≅ so(3) algebra
    appropriate for that CS computation.
    """
    sc = _antisymmetrize({
        (0, 1, 2): 1,
        (1, 2, 0): 1,
        (2, 0, 1): 1,
    })
    return sc, 3, ["e_0", "e_1", "e_2"]


def so4() -> Tuple[StructureConstants, int, List[str]]:
    """
    so(4) — Lie algebra of SO(4) = Isom_0^+(S³)  (dim = 6).

    so(4) ≅ su(2) ⊕ su(2) as a direct sum of two simple ideals.

    Basis:  {a_0, a_1, a_2} for the first su(2) factor,
            {b_0, b_1, b_2} for the second su(2) factor.
    Commutation relations:
        First  su(2):  [a_0,a_1]=a_2,  [a_1,a_2]=a_0,  [a_2,a_0]=a_1
        Second su(2):  [b_0,b_1]=b_2,  [b_1,b_2]=b_0,  [b_2,b_0]=b_1
        Cross  terms:  all zero (direct sum decomposition).

    The space of Ad-invariant symmetric bilinear forms is 2-dimensional
    (one free parameter per simple factor).
    """
    sc = _antisymmetrize({
        (0, 1, 2): 1,  (1, 2, 0): 1,  (2, 0, 1): 1,   # first  su(2)
        (3, 4, 5): 1,  (4, 5, 3): 1,  (5, 3, 4): 1,   # second su(2)
    })
    return sc, 6, ["a_0", "a_1", "a_2", "b_0", "b_1", "b_2"]


def sl2r_times_r() -> Tuple[StructureConstants, int, List[str]]:
    """
    sl(2,R) ⊕ R — Lie algebra of Isom_0^+(SL~_2 R)  (dim = 4).

    Isom_0^+(widetilde{SL}_2 R) ≅ (widetilde{SL}_2 R × R) / Z,
    where the discrete quotient Z does not affect the Lie algebra.
    Hence the Lie algebra is sl(2,R) ⊕ R, the same as for H²×R.

    Basis:  e_0, e_1, e_2  (sl(2,R) part),  e_3  (R part).
    Commutation relations: identical to h2_times_r().

    Note: sl2r_times_r() and h2_times_r() return the same structure
    constants; they are listed separately to make the geometry–algebra
    correspondence explicit.
    """
    sc = _antisymmetrize({
        (0, 1, 2):  1,
        (1, 2, 0):  1,
        (2, 0, 1): -1,
    })
    return sc, 4, ["e_0", "e_1", "e_2", "e_3"]


def sl2c_real() -> Tuple[StructureConstants, int, List[str]]:
    """
    sl(2,C)_R  ≅  so(1,3) — the Lie algebra of Isom_0^+(H³) = PSL(2,C)
    viewed as a **real** Lie algebra of real dimension 6.

    Basis:  {J_0, J_1, J_2}  (rotation generators),
            {K_0, K_1, K_2}  (boost generators).
    (0-based indices: 0–2 = J, 3–5 = K.)

    Commutation relations of so(1,3):
        [J_i, J_j] =  eps_{ijk} J_k   (so(3) subalgebra)
        [J_i, K_j] =  eps_{ijk} K_k   (boosts transform as vectors)
        [K_i, K_j] = -eps_{ijk} J_k   (Lorentz signature)

    **Remark on the invariant form space.**
    As a *real* Lie algebra, sl(2,C)_R has a **2-dimensional** space of
    Ad-invariant symmetric bilinear forms spanned by
        B_1(X,Y) = Re Tr(XY)   and   B_2(X,Y) = Im Tr(XY)
    (equivalently, the real and imaginary parts of the Killing form of
    sl(2,C) over C).  This is because the complexification satisfies
    (sl(2,C)_R)_C ≅ sl(2,C) ⊕ sl(2,C), which has a 2-dimensional
    space of invariant forms.
    As a *complex* Lie algebra, sl(2,C) is simple and has only a
    1-dimensional space of invariant forms.

    The non-degeneracy condition det(G) = -(t_0^2 + t_1^2)^3 ≠ 0
    holds for all (t_0, t_1) ≠ (0, 0).
    """
    sc = _antisymmetrize({
        # [J0,J1]=J2, [J1,J2]=J0, [J2,J0]=J1
        (0, 1, 2):  1,
        (1, 2, 0):  1,
        (2, 0, 1):  1,
        # [J0,K1]=K2, [J1,K2]=K0, [J2,K0]=K1
        (0, 4, 5):  1,
        (1, 5, 3):  1,
        (2, 3, 4):  1,
        # [J0,K2]=-K1, [J1,K0]=-K2, [J2,K1]=-K0
        (0, 5, 4): -1,
        (1, 3, 5): -1,
        (2, 4, 3): -1,
        # [K0,K1]=-J2, [K1,K2]=-J0, [K2,K0]=-J1
        (3, 4, 2): -1,
        (4, 5, 0): -1,
        (5, 3, 1): -1,
    })
    return sc, 6, ["J_0", "J_1", "J_2", "K_0", "K_1", "K_2"]


def e3_geometry() -> Tuple[StructureConstants, int, List[str]]:
    """
    Lie algebra of the Euclidean geometry E³.

    The isometry group SE(3) has Lie algebra se(3),
    spanned by translations {p_1, p_2, p_3} and rotations {L_1, L_2, L_3}.
    Basis (0-based):
        e_0 = p_1, e_1 = p_2, e_2 = p_3,
        e_3 = L_1, e_4 = L_2, e_5 = L_3.

    Commutation relations of se(3):
        [L_i, L_j] = epsilon_{ijk} L_k   (so(3) subalgebra)
        [L_i, p_j] = epsilon_{ijk} p_k   (action on translations)
        [p_i, p_j] = 0                   (translations commute)

    In 0-based index notation:
        so(3) part (indices 3, 4, 5):
            [e_3, e_4] = e_5,  [e_4, e_5] = e_3,  [e_5, e_3] = e_4
        action on translations:
            [e_3, e_1] = e_2,  [e_3, e_2] = -e_1
            [e_4, e_2] = e_0,  [e_4, e_0] = -e_2
            [e_5, e_0] = e_1,  [e_5, e_1] = -e_0
    """
    sc = _antisymmetrize({
        # so(3) part
        (3, 4, 5): 1,
        (4, 5, 3): 1,
        (5, 3, 4): 1,
        # action of rotations on translations
        (3, 1, 2):  1,
        (3, 2, 1): -1,
        (4, 2, 0):  1,
        (4, 0, 2): -1,
        (5, 0, 1):  1,
        (5, 1, 0): -1,
    })
    return sc, 6, ["p_1", "p_2", "p_3", "L_1", "L_2", "L_3"]


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

BUILTIN_ALGEBRAS = {
    "sl2":        sl2,
    "so3":        so3,
    "su2":        so3,           # su(2) ~ so(3)
    "sol":        sol,
    "nil":        nil_geometry,
    "h2xr":       h2_times_r,
    "s2xr":       s2_times_r,
    "s3":         s3_geometry,   # CS gauge algebra su(2), dim=3
    "so4":        so4,           # Lie algebra of Isom_0^+(S³), dim=6
    "sl2r_times_r": sl2r_times_r,  # Lie algebra of Isom_0^+(SL~_2 R), dim=4
    "sl2c":       sl2c_real,     # Lie algebra of Isom_0^+(H³), dim=6
    "e3":         e3_geometry,
}


def list_algebras() -> List[str]:
    """Return the names of all built-in Lie algebras."""
    return sorted(BUILTIN_ALGEBRAS.keys())


def get_algebra(name: str) -> Tuple[StructureConstants, int, List[str]]:
    """
    Retrieve a built-in Lie algebra by name.

    Parameters
    ----------
    name : str
        One of the keys returned by `list_algebras()`.

    Returns
    -------
    (sc, dim, basis_labels)

    Raises
    ------
    KeyError
        If `name` is not found in the registry.
    """
    key = name.lower()
    if key not in BUILTIN_ALGEBRAS:
        raise KeyError(
            f"Unknown algebra '{name}'. "
            f"Available: {list_algebras()}"
        )
    return BUILTIN_ALGEBRAS[key]()
