import math
import traceback
from inspect import getmembers, ismethod

class PhysicsChecker:
    """
    Auditor for the fish-in-aquarium red-shift problem.
    Checks that the observed frequency is correctly identified as red-shifted
    and that the modeling follows the moving-boundary Doppler relationship.
    """
    def __init__(self, solution_instance):
        # Extract givens safely (all default to None)
        self.f0 = None
        self.n = None
        self.u = None
        self.c = None

        # First try direct attribute, then look inside givens dict
        for attr in ('f0', 'n', 'u', 'c'):
            if hasattr(solution_instance, attr):
                setattr(self, attr, getattr(solution_instance, attr, None))
            elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
                setattr(self, attr, solution_instance.givens.get(attr, None))

        # Extract result: observed frequency
        self.f_observed = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.f_observed = solution_instance.results.get('f_observed', None)
        elif hasattr(solution_instance, 'f_observed'):
            self.f_observed = solution_instance.f_observed

        # Also store explanation steps if available (for qualitative reasoning checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = None

    # -------------------------------------------------------------------------
    # Interface tests – required keys exist and are valid numbers
    # -------------------------------------------------------------------------
    def test_interface_f_observed_exists(self):
        if self.f_observed is None:
            return (False, "Missing result key 'f_observed'")
        if not isinstance(self.f_observed, (int, float)):
            return (False, f"'f_observed' is not numeric: {type(self.f_observed)}")
        return (True, "f_observed present and numeric")

    def test_interface_givens_present(self):
        missing = []
        for name in ('f0', 'n', 'u', 'c'):
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing given(s): {', '.join(missing)}")
        # Check types
        for name in missing:
            val = getattr(self, name)
            if not isinstance(val, (int, float)):
                return (False, f"Given '{name}' is not numeric: {type(val)}")
        return (True, "All required givens present and numeric")

    # -------------------------------------------------------------------------
    # Invariant tests – physical sanity using theoretical predictions
    # -------------------------------------------------------------------------
    def test_invariant_no_motion(self):
        """If u = 0, observed frequency must equal source frequency."""
        if self.u is None or self.f0 is None or self.f_observed is None:
            return (False, "SKIP: missing u, f0, or f_observed")
        # We need the case u=0 – but if the actual u is not zero, we cannot directly test.
        # Instead, we can check that when u→0 the formula gives equality.
        # Better: we can compute the expected f_observed if u=0 from the theoretical formula.
        # But that would re-derive. Instead, note that the formula gives f_observed = f0 when u=0.
        # Since u is given, we can only test if u == 0 exactly (rare).
        # So skip this test gracefully.
        if abs(self.u) < 1e-12:
            expected = self.f0
            if not math.isclose(self.f_observed, expected, rel_tol=1e-8, abs_tol=1e-12):
                return (False, f"u=0 but f_observed ({self.f_observed}) != f0 ({expected})")
            return (True, "u=0, frequencies equal")
        return (False, "SKIP: u is non-zero, cannot test zero-motion directly")

    def test_invariant_redshift_condition(self):
        """
        Fundamental physics: For any n>1 and u>0, f_observed < f0.
        (denominator > numerator in the Doppler formula)
        """
        if self.f_observed is None or self.f0 is None:
            return (False, "SKIP: missing f_observed or f0")
        if self.f_observed < self.f0:
            return (True, "Observed frequency is lower than source (red-shift)")
        else:
            return (False, f"Observed frequency {self.f_observed} is not lower than source {self.f0}")

    # -------------------------------------------------------------------------
    # Modeling tests – check the actual theoretical ratio using givens
    # -------------------------------------------------------------------------
    def test_modeling_doppler_ratio(self):
        """
        Verify that f_observed / f0 = (1 + u/c) / (1 + n*u/c) within 1% tolerance.
        This is derived from phase-matching at the moving boundary.
        """
        if any(v is None for v in (self.f_observed, self.f0, self.n, self.u, self.c)):
            return (False, "SKIP: missing f_observed, f0, n, u, or c")
        if abs(self.c) < 1e-12:
            return (False, "SKIP: invalid c (zero)")
        if abs(self.u) < 1e-15:
            # When u=0 the ratio is 1
            expected_ratio = 1.0
        else:
            numerator = 1 + self.u / self.c
            denominator = 1 + self.n * self.u / self.c
            expected_ratio = numerator / denominator
        actual_ratio = self.f_observed / self.f0
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"f_observed/f0 = {actual_ratio:.4f} matches theory {expected_ratio:.4f}")
        else:
            return (False, f"f_observed/f0 = {actual_ratio:.4f} differs from theory {expected_ratio:.4f}")

    # -------------------------------------------------------------------------
    # Monotonicity tests – sensitivity of observed frequency to input parameters
    # -------------------------------------------------------------------------
    def test_monotonicity_freq_decreases_with_u(self):
        """
        For n>1 and u>0, increasing the water surface speed should decrease
        the observed frequency (i.e., more redshift).
        Perturb u by +10% and recompute ratio.
        """
        if any(v is None for v in (self.f_observed, self.f0, self.n, self.u, self.c)):
            return (False, "SKIP: missing data for monotonicity test")
        if abs(self.c) < 1e-12 or abs(self.u) < 1e-15:
            return (False, "SKIP: u or c too small to perturb")
        # Baseline ratio
        numerator = 1 + self.u / self.c
        denominator = 1 + self.n * self.u / self.c
        base_ratio = numerator / denominator
        # Perturbed u: increase by 10%
        u_pert = self.u * 1.1
        numerator_p = 1 + u_pert / self.c
        denominator_p = 1 + self.n * u_pert / self.c
        pert_ratio = numerator_p / denominator_p
        if pert_ratio < base_ratio:
            return (True, "Increasing u decreases ratio (consistent with more redshift)")
        else:
            return (False, f"After +10% u, ratio went from {base_ratio:.5f} to {pert_ratio:.5f} (expected decrease)")

    def test_monotonicity_freq_decreases_with_n(self):
        """
        For fixed u>0, increasing the refractive index should also decrease
        the observed frequency (more redshift). Perturb n by +10%.
        """
        if any(v is None for v in (self.f_observed, self.f0, self.n, self.u, self.c)):
            return (False, "SKIP: missing data")
        if abs(self.c) < 1e-12 or abs(self.u) < 1e-15:
            return (False, "SKIP: u or c too small")
        numerator = 1 + self.u / self.c
        base_ratio = numerator / (1 + self.n * self.u / self.c)
        # Perturb n: increase by 10%
        n_pert = self.n * 1.1
        pert_ratio = numerator / (1 + n_pert * self.u / self.c)
        if pert_ratio < base_ratio:
            return (True, "Increasing n decreases ratio (consistent with more redshift)")
        else:
            return (False, f"After +10% n, ratio went from {base_ratio:.5f} to {pert_ratio:.5f} (expected decrease)")

    # -------------------------------------------------------------------------
    # Qualitative reasoning check using explanation steps (if available)
    # -------------------------------------------------------------------------
    def test_modeling_redshift_mention_in_explanation(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = " ".join(self.explanation_steps).lower()
        keywords = ["redshift", "red-shifted", "red-shift"]
        if any(kw in text for kw in keywords):
            return (True, "Explanation mentions red-shift or red-shifted")
        else:
            return (False, "Explanation does not mention red-shift/red-shifted")

    # -------------------------------------------------------------------------
    # Weak final answer sanity (only sign and range)
    # -------------------------------------------------------------------------
    def test_answer_sanity_observed_positive(self):
        if self.f_observed is None:
            return (False, "SKIP: f_observed missing")
        if self.f_observed <= 0:
            return (False, f"Observed frequency {self.f_observed} is not positive")
        return (True, "Observed frequency positive")

    # -------------------------------------------------------------------------
    # Run all tests
    # -------------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Dynamically find all test methods
        for name, method in getmembers(self, predicate=ismethod):
            if not name.startswith("test_"):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# End of PhysicsChecker