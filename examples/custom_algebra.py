"""
examples/custom_algebra.py
==========================
Demonstrates how to define a custom Lie algebra by supplying
structure constants directly, and compute its Ad-invariant form.

We use the Lie algebra  g = aff(R) x aff(R),  where
aff(R) is the 2-dimensional affine Lie algebra:
    basis:  a, b  with  [a, b] = b.

The direct sum g = aff(R) x aff(R) has dimension 4 with basis
    e_0 = a_1,  e_1 = b_1,  e_2 = a_2,  e_3 = b_2
and commutation relations
    [e_0, e_1] = e_1,    [e_2, e_3] = e_3.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import find_ad_invariant_form, print_result

# Structure constants (0-based indices, antisymmetric)
sc_aff2 = {
    (0, 1, 1):  1,   # [a_1, b_1] = b_1
    (1, 0, 1): -1,
    (2, 3, 3):  1,   # [a_2, b_2] = b_2
    (3, 2, 3): -1,
}

result = find_ad_invariant_form(
    sc_aff2,
    dim=4,
    basis_labels=["a_1", "b_1", "a_2", "b_2"],
)

print("Custom Lie algebra:  aff(R) x aff(R)")
print_result(result, latex_output=True)
