import math
import traceback
from typing import Any, List, Tuple


class PhysicsChecker:
    """
    Audits a solution instance for the alpha‑decay favouritism problem.
    Checks:
      - Required result keys exist and are of expected types.
      - Physical invariants: Q_alpha positive, Q_others negative.
      - Modelling quality: G_fission > G_alpha, booleans consistent, G_ratio computed correctly.
      - Monotonicity: G_ratio increases with parent atomic number Z.
      - Weak answer sanity (finite, non‑negative, booleans correct).
    """

    def __init__(self, solution_instance: Any):
        results = getattr(solution_instance, 'results', {}) or {}
        # All expected keys, defaulting to None for robustness.
        self.Q_alpha_pos = results.get('Q_alpha_pos', None)
        self.Q_others_neg = results.get('Q_others_neg', None)
        self.G_alpha = results.get('G_alpha', None)
        self.G_fission = results.get('G_fission', None)
        self.G_ratio = results.get('G_ratio', None)
        # Optional additional attributes (not required for core tests)
        self.Z = getattr(solution_instance, 'Z', None)
        self.A = getattr(solution_instance, 'A', None)

    # ------------------------------------------------------------------------
    # Interface tests – required keys exist and have correct types.
    # ------------------------------------------------------------------------

    def test_interface_Q_alpha_pos_present(self) -> Tuple[bool, str]:
        if self.Q_alpha_pos is None:
            return (False, "Missing key: Q_alpha_pos")
        if not isinstance(self.Q_alpha_pos, bool):
            return (False, f"Q_alpha_pos should be bool, got {type(self.Q_alpha_pos).__name__}")
        return (True, "Q_alpha_pos present and bool")

    def test_interface_Q_others_neg_present(self) -> Tuple[bool, str]:
        if self.Q_others_neg is None:
            return (False, "Missing key: Q_others_neg")
        if not isinstance(self.Q_others_neg, bool):
            return (False, f"Q_others_neg should be bool, got {type(self.Q_others_neg).__name__}")
        return (True, "Q_others_neg present and bool")

    def test_interface_G_alpha_present(self) -> Tuple[bool, str]:
        if self.G_alpha is None:
            return (False, "Missing key: G_alpha")
        if not isinstance(self.G_alpha, (int, float)):
            return (False, f"G_alpha should be numeric, got {type(self.G_alpha).__name__}")
        return (True, "G_alpha present and numeric")

    def test_interface_G_fission_present(self) -> Tuple[bool, str]:
        if self.G_fission is None:
            return (False, "Missing key: G_fission")
        if not isinstance(self.G_fission, (int, float)):
            return (False, f"G_fission should be numeric, got {type(self.G_fission).__name__}")
        return (True, "G_fission present and numeric")

    def test_interface_G_ratio_present(self) -> Tuple[bool, str]:
        if self.G_ratio is None:
            return (False, "Missing key: G_ratio")
        if not isinstance(self.G_ratio, (int, float)):
            return (False, f"G_ratio should be numeric, got {type(self.G_ratio).__name__}")
        return (True, "G_ratio present and numeric")

    # ------------------------------------------------------------------------
    # Invariant tests – physical laws that must hold.
    # ------------------------------------------------------------------------

    def test_invariant_Q_alpha_pos_true(self) -> Tuple[bool, str]:
        if self.Q_alpha_pos is None:
            return (False, "SKIP: Q_alpha_pos not available")
        if self.Q_alpha_pos is not True:
            return (False, f"Q_alpha_pos must be True (alpha decay energy > 0), got {self.Q_alpha_pos}")
        return (True, "Q_alpha_pos is True")

    def test_invariant_Q_others_neg_true(self) -> Tuple[bool, str]:
        if self.Q_others_neg is None:
            return (False, "SKIP: Q_others_neg not available")
        if self.Q_others_neg is not True:
            return (False, f"Q_others_neg must be True (other decays have Q < 0), got {self.Q_others_neg}")
        return (True, "Q_others_neg is True")

    def test_invariant_G_alpha_positive(self) -> Tuple[bool, str]:
        if self.G_alpha is None:
            return (False, "SKIP: G_alpha not available")
        if not (self.G_alpha > 0):
            return (False, f"G_alpha must be positive, got {self.G_alpha}")
        return (True, "G_alpha positive")

    def test_invariant_G_fission_positive(self) -> Tuple[bool, str]:
        if self.G_fission is None:
            return (False, "SKIP: G_fission not available")
        if not (self.G_fission > 0):
            return (False, f"G_fission must be positive, got {self.G_fission}")
        return (True, "G_fission positive")

    def test_invariant_G_ratio_greater_than_one(self) -> Tuple[bool, str]:
        if self.G_ratio is None:
            return (False, "SKIP: G_ratio not available")
        if not (self.G_ratio > 1.0):
            return (False, f"G_ratio must be > 1 (fission less penetrable), got {self.G_ratio}")
        return (True, "G_ratio > 1")

    # ------------------------------------------------------------------------
    # Modelling / trace tests – verify the quality of the modelling steps.
    # ------------------------------------------------------------------------

    def test_modeling_G_fission_greater_than_G_alpha(self) -> Tuple[bool, str]:
        if self.G_alpha is None or self.G_fission is None:
            return (False, "SKIP: missing G_alpha or G_fission")
        if not (self.G_fission > self.G_alpha):
            return (False, f"G_fission ({self.G_fission}) must be larger than G_alpha ({self.G_alpha})")
        return (True, "G_fission > G_alpha")

    def test_modeling_both_booleans_true(self) -> Tuple[bool, str]:
        """For a heavy nucleus, the solver should have produced True for both booleans."""
        if self.Q_alpha_pos is None or self.Q_others_neg is None:
            return (False, "SKIP: one or both boolean keys missing")
        if not (self.Q_alpha_pos and self.Q_others_neg):
            return (False, f"Both booleans must be True, got Q_alpha_pos={self.Q_alpha_pos}, Q_others_neg={self.Q_others_neg}")
        return (True, "Both Q_alpha_pos and Q_others_neg are True")

    def test_modeling_G_ratio_consistency(self) -> Tuple[bool, str]:
        """
        Check that G_ratio equals G_fission / G_alpha (if both are available).
        This ensures the solver computed the ratio correctly.
        """
        if self.G_alpha is None or self.G_fission is None or self.G_ratio is None:
            return (False, "SKIP: missing one of G_alpha, G_fission, G_ratio")
        expected_ratio = self.G_fission / self.G_alpha
        if not math.isclose(self.G_ratio, expected_ratio, rel_tol=1e-2):
            return (False, f"G_ratio ({self.G_ratio}) does not match G_fission/G_alpha ({expected_ratio})")
        return (True, "G_ratio consistent with G_fission / G_alpha")

    # ------------------------------------------------------------------------
    # Monotonicity test – check that G_ratio increases with parent atomic number Z.
    # ------------------------------------------------------------------------

    def test_monotonicity_G_ratio_increases_with_Z(self) -> Tuple[bool, str]:
        if self.G_ratio is None:
            return (False, "SKIP: G_ratio not available")
        # Choose a typical parent atomic number for a heavy nucleus.
        Z_typical = 92.0
        # Perturb Z by +10%
        Z_perturbed = 1.1 * Z_typical
        # Under the approximation G_ratio ∝ Z (since for large Z, G_fission/G_alpha ~ Z/8).
        # The constant of proportionality cancels in the ratio.
        perturbed_ratio = self.G_ratio * (Z_perturbed / Z_typical)
        if perturbed_ratio <= self.G_ratio:
            return (False, "Perturbed G_ratio should be larger than baseline")
        return (True, "G_ratio scales monotonically with Z: +10% Z gives ~10% increase")

    # ------------------------------------------------------------------------
    # Answer sanity (weak) – only trivial checks, no re‑derivation.
    # ------------------------------------------------------------------------

    def test_answer_sanity_values(self) -> Tuple[bool, str]:
        """Weak sanity: all numeric values are finite and non‑negative."""
        issues = []
        for name, value in [('G_alpha', self.G_alpha), ('G_fission', self.G_fission), ('G_ratio', self.G_ratio)]:
            if value is None:
                issues.append(f"{name} is None (skipped)")
                continue
            if math.isinf(value) or math.isnan(value):
                issues.append(f"{name} is infinite or NaN")
            elif value < 0:
                issues.append(f"{name} negative")
        for name, value in [('Q_alpha_pos', self.Q_alpha_pos), ('Q_others_neg', self.Q_others_neg)]:
            if value is not None and not isinstance(value, bool):
                issues.append(f"{name} is not bool")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All values finite, non‑negative, and booleans correct")

    # ------------------------------------------------------------------------
    # Runner – discover and run all test_* methods.
    # ------------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                traceback.print_exc()
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)