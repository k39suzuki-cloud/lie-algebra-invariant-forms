"""
invariant_forms.py
==================
Compute the most general Ad-invariant, symmetric, non-degenerate
bilinear form on a finite-dimensional Lie algebra.

Mathematical background
-----------------------
Let g be a Lie algebra with basis {e_1, ..., e_n} and structure constants
c^k_{ij} defined by

    [e_i, e_j] = sum_k  c^k_{ij}  e_k.

A symmetric bilinear form  <·,·> : g x g -> K  with Gram matrix
G = (g_{ij}),  g_{ij} = <e_i, e_j>,  is **Ad-invariant** if and only if

    <[A, B], C> + <B, [A, C]> = 0   for all  A, B, C in g.

In terms of structure constants this becomes (for every triple i, j, k):

    sum_l  c^l_{ij} g_{lk}  +  sum_m  c^m_{ik} g_{jm}  =  0.        (*)

The form is **symmetric** iff  g_{ij} = g_{ji}  for all i, j.
The form is **non-degenerate** iff  det(G) != 0.

The function `find_ad_invariant_form` solves the linear system (*) together
with the symmetry constraints and returns the most general solution.

Usage
-----
>>> import numpy as np
>>> from src.invariant_forms import find_ad_invariant_form, print_result

# Define structure constants as a dict {(i, j, k): value}
# using 0-based indices.
# Example: sl(2, R)  with basis {h, e, f}
# [h, e] = 2e,  [h, f] = -2f,  [e, f] = h
sc_sl2 = {
    (0, 1, 1):  2, (1, 0, 1): -2,   # [h, e] = 2e
    (0, 2, 2): -2, (2, 0, 2):  2,   # [h, f] = -2f
    (1, 2, 0):  1, (2, 1, 0): -1,   # [e, f] = h
}
result = find_ad_invariant_form(sc_sl2, dim=3)
print_result(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import sympy as sp
from sympy import Matrix, Symbol, symbols, det, zeros, Rational, latex


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

StructureConstants = Dict[Tuple[int, int, int], Union[int, float, sp.Expr]]


@dataclass
class BilinearFormResult:
    """
    Container for the output of `find_ad_invariant_form`.

    Attributes
    ----------
    dim : int
        Dimension of the Lie algebra.
    gram_matrix : Matrix
        The Gram matrix G of the bilinear form, expressed in terms of
        free parameters (sympy symbols).
    free_params : list[Symbol]
        List of free parameters that remain unconstrained by the
        Ad-invariance and symmetry conditions.
    determinant : sp.Expr
        Determinant of the Gram matrix (as a sympy expression).
    is_always_degenerate : bool
        True iff the determinant is identically zero, meaning no choice
        of free parameters makes the form non-degenerate.
    nondeg_condition : str
        Human-readable condition "det(G) != 0" for non-degeneracy
        (empty string if always degenerate).
    basis_labels : list[str]
        Labels for the basis elements (e_0, e_1, ..., or user-supplied).
    """

    dim: int
    gram_matrix: Matrix
    free_params: List[Symbol]
    determinant: sp.Expr
    is_always_degenerate: bool
    nondeg_condition: str
    basis_labels: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _build_structure_tensor(
    sc: StructureConstants, dim: int
) -> List[List[List[sp.Expr]]]:
    """
    Convert a sparse dict of structure constants to a dense 3-tensor.

    Parameters
    ----------
    sc  : dict with keys (i, j, k) (0-based) and numeric / sympy values.
    dim : dimension of the Lie algebra.

    Returns
    -------
    c[i][j][k]  =  c^k_{ij}   (0-based indices).
    """
    c: List[List[List[sp.Expr]]] = [
        [[sp.Integer(0)] * dim for _ in range(dim)] for _ in range(dim)
    ]
    for (i, j, k), val in sc.items():
        if not (0 <= i < dim and 0 <= j < dim and 0 <= k < dim):
            raise ValueError(
                f"Index ({i}, {j}, {k}) is out of range for dim={dim}. "
                "Indices must be 0-based and in [0, dim)."
            )
        c[i][j][k] = sp.sympify(val)
    return c


def _antisymmetry_check(
    c: List[List[List[sp.Expr]]], dim: int
) -> List[str]:
    """
    Verify  c^k_{ij} + c^k_{ji} = 0  for all i, j, k.
    Returns a list of warning strings for any violations.
    """
    warnings: List[str] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            for k in range(dim):
                s = sp.simplify(c[i][j][k] + c[j][i][k])
                if s != 0:
                    warnings.append(
                        f"  Antisymmetry violated: c[{i}][{j}][{k}] + "
                        f"c[{j}][{i}][{k}] = {s}  (expected 0)"
                    )
    return warnings


def _jacobi_check(
    c: List[List[List[sp.Expr]]], dim: int
) -> List[str]:
    """
    Verify the Jacobi identity
        sum_{l} (c^l_{ij} c^m_{lk} + c^l_{jk} c^m_{li} + c^l_{ki} c^m_{lj}) = 0
    for all i < j < k and all m.
    Returns a list of warning strings for any violations.
    """
    warnings: List[str] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            for k in range(j + 1, dim):
                for m in range(dim):
                    val = sum(
                        c[i][j][l] * c[l][k][m]
                        + c[j][k][l] * c[l][i][m]
                        + c[k][i][l] * c[l][j][m]
                        for l in range(dim)
                    )
                    val = sp.simplify(val)
                    if val != 0:
                        warnings.append(
                            f"  Jacobi violated at (i,j,k,m) = "
                            f"({i},{j},{k},{m}): residual = {val}"
                        )
    return warnings


def find_ad_invariant_form(
    sc: StructureConstants,
    dim: int,
    basis_labels: Optional[Sequence[str]] = None,
    check_jacobi: bool = True,
    check_antisymmetry: bool = True,
    verbose: bool = False,
) -> BilinearFormResult:
    """
    Find the most general Ad-invariant symmetric bilinear form on a
    Lie algebra given by its structure constants.

    Parameters
    ----------
    sc : dict
        Structure constants  c^k_{ij}  as a sparse dictionary with
        keys (i, j, k) (0-based integer indices) and numeric or sympy
        values.  Only non-zero entries need to be specified.
        The antisymmetry  c^k_{ij} = -c^k_{ji}  is *not* enforced
        automatically; include both signs explicitly.
    dim : int
        Dimension of the Lie algebra.
    basis_labels : list[str], optional
        Names for the basis elements.  Defaults to
        ['e_0', 'e_1', ..., 'e_{dim-1}'].
    check_jacobi : bool, default True
        If True, verify the Jacobi identity and print warnings.
    check_antisymmetry : bool, default True
        If True, verify antisymmetry of structure constants and warn.
    verbose : bool, default False
        If True, print intermediate steps to stdout.

    Returns
    -------
    BilinearFormResult
        See the docstring of `BilinearFormResult` for field descriptions.

    Mathematical details
    --------------------
    The function sets up a **linear** system in the dim^2 unknowns
    G[i, j] = <e_i, e_j>.

    Condition 1 — Ad-invariance (equation (*) in module docstring):
        For each triple (i, j, k) in {0,...,dim-1}^3:
            sum_{l=0}^{dim-1}  c^l_{ij} * G[l, k]
          + sum_{m=0}^{dim-1}  c^m_{ik} * G[j, m]  =  0.

    Condition 2 — Symmetry:
        G[i, j] = G[j, i]   for all  i < j.

    These give a homogeneous linear system  A * vec(G) = 0.
    `sympy.solve` extracts the solution space; the free variables are
    the undetermined parameters.

    Raises
    ------
    ValueError
        If any index in `sc` is out of range, or if dim <= 0.
    RuntimeError
        If the linear system has no solution (should not occur for a
        valid Lie algebra, but guards against degenerate input).
    """
    if dim <= 0:
        raise ValueError(f"dim must be a positive integer, got {dim}.")

    # --- Default basis labels ---
    if basis_labels is None:
        basis_labels = [f"e_{i}" for i in range(dim)]
    else:
        basis_labels = list(basis_labels)
        if len(basis_labels) != dim:
            raise ValueError(
                f"len(basis_labels) = {len(basis_labels)} != dim = {dim}."
            )

    # --- Step 1: build structure constant tensor ---
    c = _build_structure_tensor(sc, dim)

    # --- Optional sanity checks ---
    if check_antisymmetry:
        warns = _antisymmetry_check(c, dim)
        if warns:
            print("[WARNING] Antisymmetry violations detected:")
            for w in warns:
                print(w)

    if check_jacobi:
        warns = _jacobi_check(c, dim)
        if warns:
            print("[WARNING] Jacobi identity violations detected:")
            for w in warns:
                print(w)

    # --- Step 2: define symbolic Gram matrix G ---
    # G_sym[i][j] is the sympy symbol G_{i,j}
    G_sym = [[Symbol(f"G_{i}_{j}") for j in range(dim)] for i in range(dim)]
    all_vars = [G_sym[i][j] for i in range(dim) for j in range(dim)]

    if verbose:
        print("Step 2: Gram matrix (symbolic):")
        for row in G_sym:
            print(" ", [str(x) for x in row])

    # --- Step 3a: Ad-invariance equations ---
    # For each (i, j, k): sum_l c[i][j][l] * G[l][k] + sum_m c[i][k][m] * G[j][m] = 0
    ad_eqns: List[sp.Expr] = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                lhs = (
                    sum(c[i][j][l] * G_sym[l][k] for l in range(dim))
                    + sum(c[i][k][m] * G_sym[j][m] for m in range(dim))
                )
                lhs = sp.expand(lhs)
                if lhs != 0:
                    ad_eqns.append(lhs)

    if verbose:
        print(f"Step 3a: {len(ad_eqns)} non-trivial Ad-invariance equations.")

    # --- Step 3b: Symmetry equations ---
    symm_eqns: List[sp.Expr] = []
    for i in range(dim):
        for j in range(i + 1, dim):
            expr = sp.expand(G_sym[i][j] - G_sym[j][i])
            if expr != 0:
                symm_eqns.append(expr)

    if verbose:
        print(f"Step 3b: {len(symm_eqns)} symmetry equations.")

    # --- Step 4: Solve the linear system ---
    all_eqns = ad_eqns + symm_eqns

    # Build a matrix equation  M * x = 0  where x = vec(G)
    # using sympy's linear algebra solver for robustness
    A, _ = sp.linear_eq_to_matrix(all_eqns, all_vars)
    sol_space = A.nullspace()

    # Reconstruct the general solution
    # Free parameters: one per nullspace basis vector
    n_free = len(sol_space)
    free_params = [Symbol(f"t_{k}") for k in range(n_free)]

    # General solution vector (dim^2 entries)
    sol_vec = sp.Matrix([0] * (dim * dim))
    for param, basis_vec in zip(free_params, sol_space):
        sol_vec += param * basis_vec

    # Map back to matrix form
    gram = Matrix(dim, dim, lambda i, j: sp.simplify(sol_vec[i * dim + j]))

    if verbose:
        print("Step 5: Gram matrix solution:")
        sp.pprint(gram)

    # --- Step 5: Determinant and non-degeneracy ---
    det_G = sp.simplify(gram.det())

    is_always_degenerate = (det_G == 0)
    nondeg_condition = "" if is_always_degenerate else f"det(G) = {det_G} ≠ 0"

    return BilinearFormResult(
        dim=dim,
        gram_matrix=gram,
        free_params=free_params,
        determinant=det_G,
        is_always_degenerate=is_always_degenerate,
        nondeg_condition=nondeg_condition,
        basis_labels=basis_labels,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_result(result: BilinearFormResult, latex_output: bool = False) -> None:
    """
    Pretty-print the result of `find_ad_invariant_form`.

    Parameters
    ----------
    result : BilinearFormResult
    latex_output : bool
        If True, also print the Gram matrix in LaTeX format.
    """
    sep = "=" * 60
    print(sep)
    print(f"  Lie algebra dimension : {result.dim}")
    print(f"  Basis                 : {result.basis_labels}")
    print(sep)
    print("\nGram matrix G  (rows/columns indexed by basis elements):\n")
    sp.pprint(result.gram_matrix, use_unicode=True)

    if latex_output:
        print("\nLaTeX representation:")
        print(r"G = " + latex(result.gram_matrix))

    print(f"\nFree parameters : {result.free_params}")
    print(f"Determinant     : det(G) = {result.determinant}")

    if result.is_always_degenerate:
        print(
            "\n[RESULT] The form is ALWAYS DEGENERATE (det = 0 identically).\n"
            "         No Ad-invariant non-degenerate symmetric bilinear\n"
            "         form exists on this Lie algebra."
        )
    else:
        print(
            f"\n[RESULT] The form is non-degenerate if and only if:\n"
            f"         {result.nondeg_condition}"
        )
    print(sep + "\n")


def result_to_dict(result: BilinearFormResult) -> dict:
    """
    Serialize a `BilinearFormResult` to a plain dictionary (useful for
    testing and logging).
    """
    return {
        "dim": result.dim,
        "basis_labels": result.basis_labels,
        "gram_matrix": [[str(result.gram_matrix[i, j]) for j in range(result.dim)]
                        for i in range(result.dim)],
        "free_params": [str(p) for p in result.free_params],
        "determinant": str(result.determinant),
        "is_always_degenerate": result.is_always_degenerate,
        "nondeg_condition": result.nondeg_condition,
    }
