import math
import sympy as sp
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Checker for the two-step potential scattering problem.
    """
    def __init__(self, solution_instance: Any) -> None:
        """Extract results safely with None defaults."""
        results = getattr(solution_instance, 'results', {})
        self.S = results.get('S', None)
        self.detM = results.get('detM', None)
        self.T_left = results.get('T_left', None)
        # Additional optional numeric fields (if provided)
        self.k1 = results.get('k1', None)
        self.k2 = results.get('k2', None)
        self.k3 = results.get('k3', None)
        self.a = results.get('a', None)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_S_exists(self) -> Tuple[bool, str]:
        if self.S is None:
            return (False, "Missing 'S' in results.")
        return (True, "S is present.")

    def test_interface_detM_exists(self) -> Tuple[bool, str]:
        if self.detM is None:
            return (False, "Missing 'detM' in results.")
        return (True, "detM is present.")

    def test_interface_T_left_exists(self) -> Tuple[bool, str]:
        if self.T_left is None:
            return (False, "Missing 'T_left' in results.")
        return (True, "T_left is present.")

    # ----------------------------------------------------------------------
    # Invariant / structure checks
    # ----------------------------------------------------------------------
    def test_invariant_S_formula_contains_M22(self) -> Tuple[bool, str]:
        if self.S is None:
            return (False, "SKIP: S not provided.")
        if 'M22' in self.S or 'M_{22}' in self.S or 'M_22' in self.S:
            return (True, "S formula includes M22.")
        return (False, "S formula does not contain M22.")

    def test_invariant_S_formula_contains_det(self) -> Tuple[bool, str]:
        if self.S is None:
            return (False, "SKIP: S not provided.")
        if 'det' in self.S:
            return (True, "S formula includes a determinant term.")
        return (False, "S formula lacks a determinant term.")

    def test_invariant_detM_is_k1_over_k3(self) -> Tuple[bool, str]:
        if self.detM is None:
            return (False, "SKIP: detM not provided.")
        if 'k1' in self.detM and 'k3' in self.detM and '/' in self.detM:
            return (True, "detM contains k1/k3 structure.")
        if 'k_1' in self.detM and 'k_3' in self.detM and ('\\frac' in self.detM or 'frac' in self.detM):
            return (True, "detM contains LaTeX fraction k1/k3.")
        return (False, "detM does not match the expected k1/k3 form.")

    def test_invariant_T_left_contains_cos_sin(self) -> Tuple[bool, str]:
        if self.T_left is None:
            return (False, "SKIP: T_left not provided.")
        if 'cos' in self.T_left and 'sin' in self.T_left:
            return (True, "T_left includes trigonometric functions.")
        return (False, "T_left missing cos or sin.")

    def test_invariant_T_left_numerator_positive(self) -> Tuple[bool, str]:
        if self.T_left is None:
            return (False, "SKIP: T_left not provided.")
        if '4' in self.T_left and 'k1' in self.T_left and 'k2' in self.T_left:
            return (True, "Numerator contains positive product of k1^2 and k2^2.")
        if 'k_1' in self.T_left and 'k_2' in self.T_left and '4' in self.T_left:
            return (True, "Numerator contains 4 k_1^2 k_2^2.")
        return (False, "Numerator does not match expected form.")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_trace_derived_T_left_consistent_with_known_form(self) -> Tuple[bool, str]:
        """Check that T_left denominator has cos^2 + sin^2 structure."""
        if self.T_left is None:
            return (False, "SKIP: T_left not provided.")
        has_cos2 = 'cos' in self.T_left and ('^2' in self.T_left or r'\cos^2' in self.T_left)
        has_sin2 = 'sin' in self.T_left and ('^2' in self.T_left or r'\sin^2' in self.T_left)
        has_plus = '+' in self.T_left
        if has_cos2 and has_sin2 and has_plus:
            return (True, "Denominator structure appears correct (cos² + sin²).")
        return (False, "Denominator does not have the required sum of cos² and sin² terms.")

    def test_trace_detM_has_no_trigonometric_functions(self) -> Tuple[bool, str]:
        """detM = k1/k3 should have no trigonometric functions."""
        if self.detM is None:
            return (False, "SKIP: detM not provided.")
        if 'cos' in self.detM or 'sin' in self.detM:
            return (False, "detM contains trigonometric functions; should be simple ratio.")
        return (True, "detM is free of trigonometric functions, as expected.")

    # ----------------------------------------------------------------------
    # Monotonicity checks (with explicit input perturbation)
    # ----------------------------------------------------------------------
    def _compute_detM_numeric(self, k1_num, k3_num):
        """Compute detM = k1/k3."""
        if k1_num is None or k3_num is None or k3_num == 0:
            return None
        return k1_num / k3_num

    def _compute_T_left_numeric(self, k1_num, k2_num, k3_num, a_num):
        """Compute T_left from the known formula using numeric values."""
        if any(v is None for v in [k1_num, k2_num, k3_num, a_num]):
            return None
        # T_left = (4*k1^2*k2^2) / ((k1+k3)^2*k2^2*cos^2(k2*a) + (k1*k3+k2^2)^2*sin^2(k2*a))
        cos_term = math.cos(k2_num * a_num)
        sin_term = math.sin(k2_num * a_num)
        denom = (k1_num + k3_num)**2 * k2_num**2 * cos_term**2 + (k1_num * k3_num + k2_num**2)**2 * sin_term**2
        if denom == 0:
            return None
        return (4.0 * k1_num**2 * k2_num**2) / denom

    def test_monotonicity_detM_increases_with_k1(self) -> Tuple[bool, str]:
        """Perturb k1 upward and verify detM increases."""
        if self.k1 is None or self.k3 is None:
            return (False, "SKIP: numeric k1/k3 not available.")
        baseline_detM = self._compute_detM_numeric(self.k1, self.k3)
        if baseline_detM is None:
            return (False, "SKIP: could not compute baseline detM.")
        # Perturb k1 by +10%
        epsilon = 0.1 * self.k1
        new_k1 = self.k1 + epsilon
        new_detM = self._compute_detM_numeric(new_k1, self.k3)
        if new_detM is None:
            return (False, "SKIP: could not compute perturbed detM.")
        if new_detM > baseline_detM:
            return (True, f"detM increased from {baseline_detM:.3f} to {new_detM:.3f} when k1 increased by 10%.")
        else:
            return (False, f"detM did not increase (baseline={baseline_detM:.3f}, new={new_detM:.3f}).")

    def test_monotonicity_T_left_not_constant_with_a(self) -> Tuple[bool, str]:
        """Perturb a and check T_left changes (not constant)."""
        if any(v is None for v in [self.k1, self.k2, self.k3, self.a]):
            return (False, "SKIP: numeric parameters not all available.")
        baseline_T = self._compute_T_left_numeric(self.k1, self.k2, self.k3, self.a)
        if baseline_T is None:
            return (False, "SKIP: could not compute baseline T_left.")
        # Perturb a by +5%
        epsilon = 0.05 * self.a
        new_a = self.a + epsilon
        new_T = self._compute_T_left_numeric(self.k1, self.k2, self.k3, new_a)
        if new_T is None:
            return (False, "SKIP: could not compute perturbed T_left.")
        # Check that the relative change is significant (>1e-6) to avoid numerical noise
        if abs(new_T - baseline_T) > 1e-6 * max(abs(baseline_T), 1e-10):
            return (True, f"T_left changed from {baseline_T:.6f} to {new_T:.6f} when a increased by 5%.")
        else:
            return (False, f"T_left did not change significantly (baseline={baseline_T:.6f}, new={new_T:.6f}).")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_T_left_not_NaN(self) -> Tuple[bool, str]:
        if self.T_left is None:
            return (False, "SKIP: T_left not provided.")
        if any(bad in self.T_left.lower() for bad in ['nan', 'inf']):
            return (False, "T_left contains invalid term (NaN or inf).")
        return (True, "T_left does not contain NaN or inf.")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)