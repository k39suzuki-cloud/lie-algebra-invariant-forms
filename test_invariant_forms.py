"""
tests/test_invariant_forms.py
=============================
Unit tests for the invariant_forms and lie_algebras modules.

Run with:
    pytest tests/
or:
    python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sympy as sp
import pytest
from src import find_ad_invariant_form, get_algebra, list_algebras


# ---------------------------------------------------------------------------
#  Helper: verify Ad-invariance of a concrete Gram matrix
# ---------------------------------------------------------------------------

def verify_ad_invariance(gram: sp.Matrix, sc: dict, dim: int) -> bool:
    """
    Plug a *concrete* (numerical) Gram matrix into the Ad-invariance
    condition and check all equations.
    """
    # Build dense tensor
    c = [[[sp.Integer(0)] * dim for _ in range(dim)] for _ in range(dim)]
    for (i, j, k), v in sc.items():
        c[i][j][k] = sp.sympify(v)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                val = (
                    sum(c[i][j][l] * gram[l, k] for l in range(dim))
                    + sum(c[i][k][m] * gram[j, m] for m in range(dim))
                )
                if sp.simplify(val) != 0:
                    return False
    return True


# ---------------------------------------------------------------------------
#  Tests for sl(2, R)
# ---------------------------------------------------------------------------

class TestSl2:
    def setup_method(self):
        sc, dim, basis = get_algebra("sl2")
        self.result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)
        self.sc = sc
        self.dim = dim

    def test_dim(self):
        assert self.result.dim == 3

    def test_one_free_parameter(self):
        # sl(2,R) has a 1-dimensional space of invariant forms (up to scale)
        assert len(self.result.free_params) == 1

    def test_not_always_degenerate(self):
        assert not self.result.is_always_degenerate

    def test_gram_matrix_shape(self):
        G = self.result.gram_matrix
        assert G.shape == (3, 3)

    def test_symmetry(self):
        G = self.result.gram_matrix
        for i in range(3):
            for j in range(3):
                assert sp.simplify(G[i, j] - G[j, i]) == 0

    def test_ad_invariance_on_killing_normalization(self):
        # Substitute t_0 = 1 (Killing-form normalization up to scale)
        G_conc = self.result.gram_matrix.subs(self.result.free_params[0], 1)
        assert verify_ad_invariance(G_conc, self.sc, self.dim)


# ---------------------------------------------------------------------------
#  Tests for so(3)
# ---------------------------------------------------------------------------

class TestSo3:
    def setup_method(self):
        sc, dim, basis = get_algebra("so3")
        self.result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)
        self.sc = sc
        self.dim = dim

    def test_one_free_parameter(self):
        assert len(self.result.free_params) == 1

    def test_gram_is_scalar_multiple_of_identity(self):
        G = self.result.gram_matrix
        t = self.result.free_params[0]
        expected = t * sp.eye(3)
        assert sp.simplify(G - expected) == sp.zeros(3, 3)

    def test_not_always_degenerate(self):
        assert not self.result.is_always_degenerate


# ---------------------------------------------------------------------------
#  Tests for Sol (no non-degenerate form)
# ---------------------------------------------------------------------------

class TestSol:
    def setup_method(self):
        sc, dim, basis = get_algebra("sol")
        self.result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)

    def test_always_degenerate(self):
        assert self.result.is_always_degenerate

    def test_det_zero(self):
        assert self.result.determinant == 0


# ---------------------------------------------------------------------------
#  Tests for Nil geometry
# ---------------------------------------------------------------------------

class TestNil:
    def setup_method(self):
        sc, dim, basis = get_algebra("nil")
        self.result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)
        self.sc = sc
        self.dim = dim

    def test_dim(self):
        assert self.result.dim == 4

    def test_two_free_parameters(self):
        assert len(self.result.free_params) == 2

    def test_not_always_degenerate(self):
        assert not self.result.is_always_degenerate

    def test_ad_invariance(self):
        # Substitute concrete values for free params
        subs = {p: (i + 1) for i, p in enumerate(self.result.free_params)}
        G_conc = self.result.gram_matrix.subs(subs)
        assert verify_ad_invariance(G_conc, self.sc, self.dim)


# ---------------------------------------------------------------------------
#  Tests for H²×R geometry
# ---------------------------------------------------------------------------

class TestH2xR:
    def setup_method(self):
        sc, dim, basis = get_algebra("h2xr")
        self.result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)

    def test_dim(self):
        assert self.result.dim == 4

    def test_two_free_parameters(self):
        assert len(self.result.free_params) == 2

    def test_not_always_degenerate(self):
        assert not self.result.is_always_degenerate


# ---------------------------------------------------------------------------
#  Tests for S²×R geometry
# ---------------------------------------------------------------------------

class TestS2xR:
    def setup_method(self):
        sc, dim, basis = get_algebra("s2xr")
        self.result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)

    def test_two_free_parameters(self):
        assert len(self.result.free_params) == 2

    def test_not_always_degenerate(self):
        assert not self.result.is_always_degenerate


# ---------------------------------------------------------------------------
#  Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_algebras_listed(self):
        names = list_algebras()
        for name in ["sl2", "so3", "sol", "nil", "h2xr", "s2xr", "s3", "e3"]:
            assert name in names

    def test_unknown_algebra_raises(self):
        with pytest.raises(KeyError):
            get_algebra("nonexistent_algebra_xyz")

    def test_all_built_ins_run_without_error(self):
        for name in list_algebras():
            sc, dim, basis = get_algebra(name)
            result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)
            assert result.gram_matrix.shape == (dim, dim)


# ---------------------------------------------------------------------------
#  Input validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_negative_dim_raises(self):
        with pytest.raises(ValueError):
            find_ad_invariant_form({}, dim=-1)

    def test_zero_dim_raises(self):
        with pytest.raises(ValueError):
            find_ad_invariant_form({}, dim=0)

    def test_out_of_range_index_raises(self):
        sc = {(0, 1, 3): 1, (1, 0, 3): -1}   # index 3 >= dim=3
        with pytest.raises(ValueError):
            find_ad_invariant_form(sc, dim=3)

    def test_wrong_basis_label_length_raises(self):
        sc, dim, _ = get_algebra("sl2")
        with pytest.raises(ValueError):
            find_ad_invariant_form(sc, dim=dim, basis_labels=["a", "b"])
