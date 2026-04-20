"""
examples/run_all.py
===================
Run the Ad-invariant bilinear form computation for all built-in
Lie algebras and display the results.

Usage
-----
From the repository root:

    python examples/run_all.py

or, to also write LaTeX output:

    python examples/run_all.py --latex
"""

import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import find_ad_invariant_form, print_result, get_algebra, list_algebras

SECTION_LABELS = {
    "sl2":  "sl(2, R)  — Lie algebra of SL(2, R) / PSL(2, R)",
    "so3":  "so(3) ~ su(2)  — Lie algebra of SO(3) / SU(2)  [S³ geometry]",
    "sol":  "sol  — Sol geometry  (NO non-degenerate invariant form)",
    "nil":  "nil ⋊ R  — Nil geometry  (semidirect product)",
    "h2xr": "sl(2, R) × R  — H²×R geometry",
    "s2xr": "so(3) × R  — S²×R geometry",
    "s3":   "su(2) ~ so(3)  — S³ geometry (3-dim piece)",
    "e3":   "se(3)  — Euclidean geometry E³",
}

def main(latex_output: bool = False) -> None:
    print("=" * 70)
    print("  Ad-invariant Symmetric Bilinear Forms on Lie Algebras")
    print("  (Thurston's Eight Geometries and Classical Examples)")
    print("=" * 70)

    for name in list_algebras():
        label = SECTION_LABELS.get(name, name)
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        sc, dim, basis = get_algebra(name)
        result = find_ad_invariant_form(
            sc, dim=dim, basis_labels=basis,
            check_jacobi=True, check_antisymmetry=True,
            verbose=False,
        )
        print_result(result, latex_output=latex_output)


if __name__ == "__main__":
    latex = "--latex" in sys.argv
    main(latex_output=latex)
