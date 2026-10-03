import math
import traceback
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits a solution to the charged particle in crossed E and B fields problem.
    Provides multiple small tests based on physics invariants, modeling,
    monotonicity, and a weak answer sanity check.
    """

    def __init__(self, solution_instance: Any):
        # ── Extract givens (using None defaults) ──
        # Attempt to get from attributes or from results/givens dictionary
        self.q = getattr(solution_instance, 'q', None)
        self.m = getattr(solution_instance, 'm', None)
        self.E = getattr(solution_instance, 'E', None)
        self.B = getattr(solution_instance, 'B', None)
        self.c = getattr(solution_instance, 'c', None)

        # If not found as attributes, try 'givens' dictionary
        if self.q is None and hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            self.q = givens.get('q', None)
            self.m = givens.get('m', None)
            self.E = givens.get('E', None)
            self.B = givens.get('B', None)
            self.c = givens.get('c', None)

        # ── Extract results (from results dict or attributes) ──
        # Expected keys from pseudocode emit contract
        self.invariant = None
        self.R = None
        self.p_x0 = None
        self.E_prime = None

        if hasattr(solution_instance, 'results'):
            res = solution_instance.results
            if isinstance(res, dict):
                self.invariant = res.get('invariant', None)
                self.R = res.get('R', None)
                self.p_x0 = res.get('p_x0', None)
                self.E_prime = res.get('E_prime', None)
            else:
                # fallback to attributes if results is not a dict
                self.invariant = getattr(solution_instance, 'invariant', None)
                self.R = getattr(solution_instance, 'R', None)
                self.p_x0 = getattr(solution_instance, 'p_x0', None)
                self.E_prime = getattr(solution_instance, 'E_prime', None)
        else:
            self.invariant = getattr(solution_instance, 'invariant', None)
            self.R = getattr(solution_instance, 'R', None)
            self.p_x0 = getattr(solution_instance, 'p_x0', None)
            self.E_prime = getattr(solution_instance, 'E_prime', None)

        # ── Derived quantities (computed inside checker for checks) ──
        self.invariant_theory = None
        if self.E is not None and self.B is not None and self.c is not None:
            self.invariant_theory = self.E**2 - (self.c**2)*(self.B**2)

    # ── Interface tests ──────────────────────────────────────
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present and not None."""
        missing = []
        for name, val in [('invariant', self.invariant),
                           ('R', self.R),
                           ('p_x0', self.p_x0),
                           ('E_prime', self.E_prime)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing key(s): {missing}")
        return (True, "All required keys present and non-None.")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that essential givens are available for physics checks."""
        missing = []
        for name, val in [('q', self.q), ('m', self.m),
                          ('E', self.E), ('B', self.B), ('c', self.c)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All essential givens present.")

    # ── Invariant tests (theorem-based) ──────────────────────
    def test_invariant_definition(self) -> Tuple[bool, str]:
        """Check that invariant = E^2 - c^2 B^2."""
        if self.invariant is None or self.invariant_theory is None:
            return (False, "SKIP: missing invariant or givens")
        if math.isclose(self.invariant, self.invariant_theory, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"invariant matches theory: {self.invariant} ~ {self.invariant_theory}")
        else:
            return (False, f"invariant {self.invariant} differs from theory {self.invariant_theory}")

    def test_invariant_sign_for_e_lt_cb(self) -> Tuple[bool, str]:
        """If E < cB, invariant should be negative (timelike fields)."""
        if self.E is None or self.B is None or self.c is None:
            return (False, "SKIP: missing givens")
        if self.E < self.c * self.B:
            if self.invariant is not None:
                if self.invariant < 0:
                    return (True, f"E < cB, invariant={self.invariant} < 0")
                else:
                    return (False, f"E < cB but invariant={self.invariant} >= 0")
            else:
                return (False, "SKIP: invariant missing")
        else:
            return (False, "SKIP: condition E < cB not satisfied")

    def test_invariant_sign_for_e_gt_cb(self) -> Tuple[bool, str]:
        """If E > cB, invariant should be positive (spacelike fields)."""
        if self.E is None or self.B is None or self.c is None:
            return (False, "SKIP: missing givens")
        if self.E > self.c * self.B:
            if self.invariant is not None:
                if self.invariant > 0:
                    return (True, f"E > cB, invariant={self.invariant} > 0")
                else:
                    return (False, f"E > cB but invariant={self.invariant} <= 0")
            else:
                return (False, "SKIP: invariant missing")
        else:
            return (False, "SKIP: condition E > cB not satisfied")

    # ── Modeling/trace tests ─────────────────────────────────
    def test_modeling_gyroradius_R(self) -> Tuple[bool, str]:
        """
        Check that R = m c^2 E / (q (c^2 B^2 - E^2)) when E < cB.
        Derive expected R from givens and compare with stored R.
        """
        if any(x is None for x in [self.q, self.m, self.E, self.B, self.c]):
            return (False, "SKIP: missing givens")
        if self.E >= self.c * self.B:
            return (False, "SKIP: condition E < cB not met (R formula not valid)")
        denom = self.q * (self.c**2 * self.B**2 - self.E**2)
        if denom == 0:
            return (False, "Denominator zero unphysical")
        expected_R = (self.m * self.c**2 * self.E) / denom
        if self.R is None:
            return (False, "SKIP: R missing")
        if math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R matches expected: {self.R} ~ {expected_R}")
        else:
            return (False, f"R={self.R} differs from expected {expected_R}")

    def test_modeling_p_x0(self) -> Tuple[bool, str]:
        """
        Check that p_x0 = m c^2 B / sqrt(E^2 - c^2 B^2) when E > cB.
        """
        if any(x is None for x in [self.q, self.m, self.E, self.B, self.c]):
            return (False, "SKIP: missing givens")
        if self.E <= self.c * self.B:
            return (False, "SKIP: condition E > cB not met (p_x0 formula not valid)")
        denominator = math.sqrt(self.E**2 - self.c**2 * self.B**2)
        if denominator == 0:
            return (False, "Denominator zero unphysical")
        expected_p_x0 = (self.m * self.c**2 * self.B) / denominator
        if self.p_x0 is None:
            return (False, "SKIP: p_x0 missing")
        if math.isclose(self.p_x0, expected_p_x0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_x0 matches expected: {self.p_x0} ~ {expected_p_x0}")
        else:
            return (False, f"p_x0={self.p_x0} differs from expected {expected_p_x0}")

    def test_modeling_E_prime(self) -> Tuple[bool, str]:
        """
        Check that E_prime = sqrt(E^2 - c^2 B^2) when E > cB.
        """
        if self.E is None or self.B is None or self.c is None:
            return (False, "SKIP: missing givens")
        if self.E <= self.c * self.B:
            return (False, "SKIP: condition E > cB not met (E_prime formula not valid)")
        expected_E_prime = math.sqrt(self.E**2 - self.c**2 * self.B**2)
        if self.E_prime is None:
            return (False, "SKIP: E_prime missing")
        if math.isclose(self.E_prime, expected_E_prime, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_prime matches expected: {self.E_prime} ~ {expected_E_prime}")
        else:
            return (False, f"E_prime={self.E_prime} differs from expected {expected_E_prime}")

    # ── Monotonicity tests (perturbative check) ──────────────
    def test_monotonicity_p_x0_increases_with_B(self) -> Tuple[bool, str]:
        """
        For fixed E > cB, increasing B should increase p_x0.
        Derive p_x0 for original and for B+10%, then compare.
        """
        if any(x is None for x in [self.m, self.E, self.B, self.c, self.p_x0]):
            return (False, "SKIP: missing givens or p_x0")
        if self.E <= self.c * self.B:
            return (False, "SKIP: condition E > cB not met")
        # Perturb B by +10%
        B_pert = self.B * 1.1
        if self.E <= self.c * B_pert:  # ensure condition still holds
            return (False, "SKIP: perturbed B violates E > cB")
        denominator_pert = math.sqrt(self.E**2 - self.c**2 * B_pert**2)
        p_x0_pert = (self.m * self.c**2 * B_pert) / denominator_pert
        if p_x0_pert > self.p_x0:
            return (True, f"p_x0 increases with B: {self.p_x0} -> {p_x0_pert}")
        else:
            return (False, f"p_x0 did not increase: original {self.p_x0}, perturbed {p_x0_pert}")

    def test_monotonicity_R_increases_with_E(self) -> Tuple[bool, str]:
        """
        For fixed B with E < cB, increasing E should increase R.
        Derive R for original and for E+10%, then compare.
        """
        if any(x is None for x in [self.q, self.m, self.E, self.B, self.c, self.R]):
            return (False, "SKIP: missing givens or R")
        if self.E >= self.c * self.B:
            return (False, "SKIP: condition E < cB not met")
        E_pert = self.E * 1.1
        if E_pert >= self.c * self.B:  # ensure condition still holds
            return (False, "SKIP: perturbed E violates E < cB")
        denom_pert = self.q * (self.c**2 * self.B**2 - E_pert**2)
        R_pert = (self.m * self.c**2 * E_pert) / denom_pert
        if R_pert > self.R:
            return (True, f"R increases with E: {self.R} -> {R_pert}")
        else:
            return (False, f"R did not increase: original {self.R}, perturbed {R_pert}")

    # ── Weak answer sanity test (only one) ───────────────────
    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        """
        Weak sanity: All results should be positive (or non-negative) and finite.
        Does NOT re-derive any answer or use math.isclose.
        """
        results = {
            'invariant': self.invariant,
            'R': self.R,
            'p_x0': self.p_x0,
            'E_prime': self.E_prime
        }
        for name, val in results.items():
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or Inf: {val}")
            # For invariant, it can be negative, but we just check finiteness.
            if name != 'invariant' and val < 0:
                return (False, f"{name} is negative: {val}")
        return (True, "All available results are finite and positive (invariant allowed negative).")

    # ── Runner ───────────────────────────────────────────────
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods and execute them, returning overall status and logs."""
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)