# lie-algebra-invariant-forms

A Python toolkit for computing **Ad-invariant symmetric bilinear forms** on
finite-dimensional Lie algebras, with built-in support for all Lie algebras
arising in **Thurston's eight model geometries**.

This tool is intended for researchers working on Chern–Simons theory,
geometric topology, and related areas where invariant polynomials on Lie
algebras play a central role.

---

## Mathematical Background

Let $\mathfrak{g}$ be a Lie algebra with basis $\{e_1, \ldots, e_n\}$ and
structure constants $c^k_{ij}$ defined by

$$[e_i, e_j] = \sum_k c^k_{ij} e_k.$$

A symmetric bilinear form $\langle\cdot,\cdot\rangle : \mathfrak{g} \times \mathfrak{g} \to \mathbb{K}$
with Gram matrix $G = (g_{ij})$, $g_{ij} = \langle e_i, e_j \rangle$, is
**Ad-invariant** if and only if

$$\langle [A, B], C \rangle + \langle B, [A, C] \rangle = 0
\quad \text{for all } A, B, C \in \mathfrak{g}.$$

In terms of structure constants, this reads (for every triple $i, j, k$):

$$\sum_l c^l_{ij} g_{lk} + \sum_m c^m_{ik} g_{jm} = 0. \tag{$*$}$$

The toolkit solves $(*)$ together with the symmetry constraint $g_{ij} = g_{ji}$
as a **homogeneous linear system** over $\mathbb{Q}$, using
[SymPy](https://www.sympy.org/)'s exact arithmetic.  The output is the
most general solution, parameterised by free scalar variables $t_0, t_1, \ldots$

### Connection to Chern–Simons theory

Ad-invariant symmetric bilinear forms are precisely the **invariant
polynomials** (degree-2 Ad-invariant maps $\mathfrak{g} \to \mathbb{K}$)
used to define Chern–Simons forms:

$$\mathrm{CS}_{\mathfrak{p}}(\omega_1, \omega_0)
= 2\int_0^1 \langle \omega_1 - \omega_0, \Omega_t \rangle  dt
\in \mathcal{A}^3(M;\mathbb{K}),$$

where $\Omega_t$ is the curvature of the interpolating connection
$\omega_t = (1-t)\omega_0 + t\omega_1$.  Knowing the space of such forms
is the first step in computing Chern–Simons invariants of geometric
3-manifolds.

---

## Thurston's Eight Geometries

The following table summarises the results for the Lie algebras of the
isometry groups of Thurston's eight model geometries.

| Geometry | Lie algebra $\mathfrak{g}$ | dim | Free params | Non-degenerate? |
|---|---|---|---|---|
| $S^3$ | $\mathfrak{su}(2) \cong \mathfrak{so}(3)$ | 3 | 1 | Yes ($t_0 \neq 0$) |
| $\mathbb{E}^3$ | $\mathfrak{se}(3)$ | 6 | — | — |
| $\mathbb{H}^3$ | $\mathfrak{sl}(2,\mathbb{R})$ | 3 | 1 | Yes ($t_0 \neq 0$) |
| $S^2 \times \mathbb{R}$ | $\mathfrak{so}(3) \oplus \mathbb{R}$ | 4 | 2 | Yes ($t_0, t_1 \neq 0$) |
| $\mathbb{H}^2 \times \mathbb{R}$ | $\mathfrak{sl}(2,\mathbb{R}) \oplus \mathbb{R}$ | 4 | 2 | Yes ($t_0, t_1 \neq 0$) |
| $\mathrm{Nil}$ | $\mathfrak{nil} \oplus \mathbb{R}$ | 4 | 2 | Yes ($t_0 \neq 0$) |
| $\mathrm{Sol}$ | $\mathfrak{sol}$ | 3 | — | **No** (always degenerate) |
| $\widetilde{\mathrm{SL}}_2\mathbb{R}$ | $\mathfrak{sl}(2,\mathbb{R})$ | 3 | 1 | Yes ($t_0 \neq 0$) |

> **Note on Sol:** The Sol geometry is the only one among Thurston's eight
> for which no Ad-invariant non-degenerate symmetric bilinear form exists.
> This is reflected in the more delicate structure of Chern–Simons theory
> for Sol manifolds.

---

## Installation

**Requirements:** Python ≥ 3.9, [SymPy](https://www.sympy.org/) ≥ 1.12.

```bash
git clone https://github.com/<your-username>/lie-algebra-invariant-forms.git
cd lie-algebra-invariant-forms
pip install -r requirements.txt
```

No installation step is required beyond cloning; all modules live in `src/`.

---

## Quick Start

### Using a built-in algebra

```python
from src import get_algebra, find_ad_invariant_form, print_result

# Retrieve sl(2, R)
sc, dim, basis = get_algebra("sl2")

# Compute the most general Ad-invariant symmetric bilinear form
result = find_ad_invariant_form(sc, dim=dim, basis_labels=basis)

# Display the result
print_result(result)
```

**Output:**
```
============================================================
  Lie algebra dimension : 3
  Basis                 : ['h', 'e', 'f']
============================================================

Gram matrix G  (rows/columns indexed by basis elements):

⎡2⋅t₀  0   0 ⎤
⎢             ⎥
⎢ 0    0   t₀ ⎥
⎢             ⎥
⎣ 0   t₀   0 ⎦

Free parameters : [t_0]
Determinant     : det(G) = -2*t_0**3

[RESULT] The form is non-degenerate if and only if:
         det(G) = -2*t_0**3 ≠ 0
```

This is the Killing form of $\mathfrak{sl}(2,\mathbb{R})$ up to the scale
$t_0$.

### List all built-in algebras

```python
from src import list_algebras
print(list_algebras())
# ['e3', 'h2xr', 'nil', 's2xr', 's3', 'sl2', 'so3', 'sol', 'su2']
```

### Defining a custom Lie algebra

Structure constants are supplied as a sparse dictionary
`{(i, j, k): value}` using **0-based indices**.  Both signs of the
antisymmetry $c^k_{ij} = -c^k_{ji}$ must be included explicitly.

```python
from src import find_ad_invariant_form, print_result

# Heisenberg algebra:  [e_0, e_1] = e_2,  all others zero
sc_heis = {
    (0, 1, 2):  1,
    (1, 0, 2): -1,
}
result = find_ad_invariant_form(sc_heis, dim=3, basis_labels=["X", "Y", "Z"])
print_result(result)
```

### LaTeX output

```python
print_result(result, latex_output=True)
```

---

## Running the Examples

```bash
# Run all built-in Lie algebras
python examples/run_all.py

# Same, with LaTeX output for Gram matrices
python examples/run_all.py --latex

# Custom algebra example (aff(R) x aff(R))
python examples/custom_algebra.py
```

---

## Running the Tests

```bash
pip install pytest
pytest tests/ -v
```

All 27 tests should pass.  The tests verify:
- Correctness of the Gram matrix (symmetry and Ad-invariance)
- Number of free parameters for each built-in algebra
- Non-degeneracy / degeneracy classification
- Input validation (out-of-range indices, wrong dimensions, etc.)

---

## Project Structure

```
lie-algebra-invariant-forms/
├── src/
│   ├── __init__.py          # Public API
│   ├── invariant_forms.py   # Core solver (find_ad_invariant_form)
│   └── lie_algebras.py      # Built-in structure constants
├── examples/
│   ├── run_all.py           # Run all built-in algebras
│   └── custom_algebra.py    # Custom algebra demo
├── tests/
│   └── test_invariant_forms.py
├── requirements.txt
└── README.md
```

---

## API Reference

### `find_ad_invariant_form(sc, dim, ...)`

```
find_ad_invariant_form(
    sc            : dict[(i,j,k) -> value],
    dim           : int,
    basis_labels  : list[str]   = None,
    check_jacobi  : bool        = True,
    check_antisymmetry : bool   = True,
    verbose       : bool        = False,
) -> BilinearFormResult
```

Solves the Ad-invariance and symmetry constraints and returns a
`BilinearFormResult` with fields:

| Field | Type | Description |
|---|---|---|
| `dim` | `int` | Dimension of the Lie algebra |
| `gram_matrix` | `sympy.Matrix` | Gram matrix in terms of free parameters |
| `free_params` | `list[Symbol]` | Undetermined scalar parameters |
| `determinant` | `sympy.Expr` | $\det(G)$ as a sympy expression |
| `is_always_degenerate` | `bool` | True iff $\det(G) \equiv 0$ |
| `nondeg_condition` | `str` | Human-readable non-degeneracy condition |

### `get_algebra(name)`

Returns `(sc, dim, basis_labels)` for a built-in Lie algebra.
Available names: see `list_algebras()`.

### `print_result(result, latex_output=False)`

Pretty-prints a `BilinearFormResult`; optionally also prints the Gram
matrix in LaTeX format.

---

## Mathematical Notes

### Why is Sol special?

The Sol Lie algebra has the property that every Ad-invariant symmetric
bilinear form is degenerate.  Concretely, if the basis is
$\{e_1, e_2, e_3\}$ with $[e_1, e_3] = e_1$ and $[e_2, e_3] = -e_2$,
then the only solution to $(*)$ has $g_{11} = g_{12} = g_{13} = g_{22} = g_{23} = 0$,
so $\det(G) = 0$ identically regardless of $g_{33}$.

### Relation to the Killing form

For semisimple Lie algebras (e.g.\ $\mathfrak{sl}(2,\mathbb{R})$,
$\mathfrak{so}(3)$), the space of Ad-invariant symmetric bilinear forms
is 1-dimensional, spanned by the Killing form
$\kappa(X, Y) = \mathrm{Tr}(\mathrm{ad}_X \circ \mathrm{ad}_Y)$.
The free parameter $t_0$ is precisely the overall scale.

For reductive Lie algebras $\mathfrak{g} = \mathfrak{s} \oplus \mathfrak{z}$
(semisimple part $\oplus$ centre), the space is 2-dimensional:
one parameter for $\mathfrak{s}$ and one for $\mathfrak{z}$.
This is why $H^2 \times \mathbb{R}$ and $S^2 \times \mathbb{R}$ each have
two free parameters.

---

## License

MIT License.  See [LICENSE](LICENSE).

---

## Citation

If you use this tool in research, please cite the source code and the
relevant mathematical references:

- P. Scott, *The Geometries of 3-Manifolds*, Bull. London Math. Soc. **15** (1983), 401–487.
- W. Thurston, *Three-Dimensional Geometry and Topology*, Vol. 1, Princeton University Press, 1997.
- D. Freed, *Classical Chern–Simons Theory, Part 2*, Houston J. Math. **28** (2002), 293–310.
