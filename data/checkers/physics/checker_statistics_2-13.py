import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    """
    Audits a student's solution for the Frenkel defect problem.
    Checks interfaces, physical invariants, modeling quality, and monotonic trends.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract results dictionary
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.U = results.get('U', None)
        self.S = results.get('S', None)
        self.n_eq = results.get('n_eq', None)

        # Safely extract givens dictionary (supports both dict and direct attributes)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.N = givens.get('N', None) or getattr(solution_instance, 'N', None)
        self.E = givens.get('E', None) or getattr(solution_instance, 'E', None)
        self.T = givens.get('T', None) or getattr(solution_instance, 'T', None)
        self.k = (givens.get('k', None) or givens.get('k_B', None) or
                  getattr(solution_instance, 'k', None) or getattr(solution_instance, 'k_B', None))

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_U_present(self) -> Tuple[bool, str]:
        if self.U is None:
            return False, "U not found in results"
        return True, "U is present"

    def test_interface_S_present(self) -> Tuple[bool, str]:
        if self.S is None:
            return False, "S not found in results"
        return True, "S is present"

    def test_interface_n_eq_present(self) -> Tuple[bool, str]:
        if self.n_eq is None:
            return False, "n_eq not found in results"
        return True, "n_eq is present"

    def test_interface_U_numeric(self) -> Tuple[bool, str]:
        if self.U is None:
            return (False, "SKIP: U not available")
        if isinstance(self.U, (int, float)):
            return True, "U is numeric"
        return False, f"U has type {type(self.U)}"

    def test_interface_S_numeric(self) -> Tuple[bool, str]:
        if self.S is None:
            return (False, "SKIP: S not available")
        if isinstance(self.S, (int, float)):
            return True, "S is numeric"
        return False, f"S has type {type(self.S)}"

    def test_interface_n_eq_numeric(self) -> Tuple[bool, str]:
        if self.n_eq is None:
            return (False, "SKIP: n_eq not available")
        if isinstance(self.n_eq, (int, float)):
            return True, "n_eq is numeric"
        return False, f"n_eq has type {type(self.n_eq)}"

    # ------------------------------------------------------------------
    # Invariant tests (physics sanity with tolerances)
    # ------------------------------------------------------------------
    def test_invariant_energy_defect_relation(self) -> Tuple[bool, str]:
        if any(v is None for v in (self.U, self.n_eq, self.E)):
            return (False, "SKIP: missing U, n_eq, or E")
        if math.isclose(self.E, 0.0, rel_tol=1e-9):
            return (False, "SKIP: E is zero, cannot divide")
        expected_U = self.n_eq * self.E
        if math.isclose(self.U, expected_U, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"U ≈ n_eq*E  (U={self.U}, expected={expected_U})"
        return False, f"U={self.U} does not match n_eq*E={expected_U}"

    def test_invariant_entropy_stirling_form(self) -> Tuple[bool, str]:
        if any(v is None for v in (self.S, self.N, self.n_eq, self.k)):
            return (False, "SKIP: missing S, N, n_eq, or k")
        if self.n_eq <= 0 or self.N - self.n_eq <= 0:
            return (False, "SKIP: n_eq or N-n_eq non-positive for Stirling")
        try:
            ln_N = math.log(self.N)
            ln_n = math.log(self.n_eq)
            ln_Nminus = math.log(self.N - self.n_eq)
            expected_S = 2.0 * self.k * (self.N * ln_N - self.n_eq * ln_n - (self.N - self.n_eq) * ln_Nminus)
            if math.isclose(self.S, expected_S, rel_tol=1e-2, abs_tol=1e-2):
                return True, "S close to Stirling form"
            return False, f"S={self.S} far from Stirling expected {expected_S}"
        except Exception as e:
            return (False, f"Error in Stirling check: {e}")

    def test_invariant_n_eq_formula(self) -> Tuple[bool, str]:
        if any(v is None for v in (self.n_eq, self.N, self.E, self.k, self.T)):
            return (False, "SKIP: missing n_eq, N, E, k, or T")
        if self.k == 0 or self.T == 0:
            return (False, "SKIP: k or T zero")
        try:
            expected_n = self.N / (1.0 + math.exp(self.E / (2.0 * self.k * self.T)))
            if math.isclose(self.n_eq, expected_n, rel_tol=1e-2, abs_tol=1e-2):
                return True, "n_eq close to N/(1+exp(E/2kT))"
            return False, f"n_eq={self.n_eq} far from expected {expected_n}"
        except Exception as e:
            return (False, f"Error in n_eq formula check: {e}")

    def test_invariant_n_eq_bounds(self) -> Tuple[bool, str]:
        if self.n_eq is None or self.N is None:
            return (False, "SKIP: missing n_eq or N")
        if 0 <= self.n_eq <= self.N + 1e-9:
            return True, f"n_eq={self.n_eq} within [0, N]"
        return False, f"n_eq={self.n_eq} out of [0, N]"

    def test_invariant_energy_nonnegative(self) -> Tuple[bool, str]:
        if self.U is None:
            return (False, "SKIP: no U")
        if self.U >= -1e-12:
            return True, f"U = {self.U} is effectively nonnegative"
        return False, f"U = {self.U} is negative"

    def test_invariant_entropy_nonnegative(self) -> Tuple[bool, str]:
        if self.S is None:
            return (False, "SKIP: no S")
        if self.S >= -1e-12:
            return True, f"S = {self.S} is effectively nonnegative"
        return False, f"S = {self.S} is negative"

    # ------------------------------------------------------------------
    # Modeling / Trace tests (checking the modeling steps)
    # ------------------------------------------------------------------
    def test_modeling_n_eq_large(self) -> Tuple[bool, str]:
        """Check that the assumption n>>1 is satisfied."""
        if self.n_eq is None:
            return (False, "SKIP: n_eq not available")
        # Reasonable threshold: n_eq > 10 to be considered >> 1
        if self.n_eq > 10:
            return True, f"n_eq = {self.n_eq} > 10, consistent with n>>1 assumption"
        return False, f"n_eq = {self.n_eq} is not >>1, modeling assumption violated"

    def test_modeling_N_minus_n_eq_large(self) -> Tuple[bool, str]:
        """Check that (N-n) is also large, as required for Stirling of second term."""
        if self.n_eq is None or self.N is None:
            return (False, "SKIP: missing n_eq or N")
        diff = self.N - self.n_eq
        if diff > 10:
            return True, f"N - n_eq = {diff} > 10, Stirling valid for both terms"
        return False, f"N - n_eq = {diff} is not large, Stirling approximation may be inaccurate"

    def test_modeling_equilibrium_condition(self) -> Tuple[bool, str]:
        """
        Re-derive the equilibrium condition: from dF/dn=0 we obtain
        (N-n)/n = exp(E/(2kT)).  Check that this holds for the given n_eq.
        """
        if any(v is None for v in (self.n_eq, self.N, self.E, self.k, self.T)):
            return (False, "SKIP: missing required quantities")
        if self.n_eq <= 0 or self.n_eq >= self.N:
            return (False, "SKIP: n_eq not strictly inside (0, N)")
        try:
            lhs = (self.N - self.n_eq) / self.n_eq
            rhs = math.exp(self.E / (2.0 * self.k * self.T))
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"(N-n_eq)/n_eq ≈ exp(E/2kT): {lhs:.6f} vs {rhs:.6f}"
            return False, f"Equilibrium condition not satisfied: (N-n)/n = {lhs}, exp(E/2kT) = {rhs}"
        except Exception as e:
            return (False, f"Error in equilibrium condition check: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare outputs)
    # ------------------------------------------------------------------
    def test_monotonicity_n_eq_increasing_T(self) -> Tuple[bool, str]:
        if any(v is None for v in (self.N, self.E, self.k, self.T)):
            return (False, "SKIP: missing N, E, k, or T")
        if self.T <= 0 or self.k <= 0:
            return (False, "SKIP: T or k non-positive")
        try:
            base = self.N / (1.0 + math.exp(self.E / (2.0 * self.k * self.T)))
            delta = 0.01 * abs(self.T)
            T_pert = self.T + delta
            perturbed = self.N / (1.0 + math.exp(self.E / (2.0 * self.k * T_pert)))
            if perturbed > base - 1e-12:
                return True, f"n_eq increases with T: base={base:.4f}, perturbed={perturbed:.4f}"
            return False, f"n_eq does not increase with T: base={base}, perturbed={perturbed}"
        except Exception as e:
            return (False, f"Error in monotonicity T test: {e}")

    def test_monotonicity_n_eq_decreasing_E(self) -> Tuple[bool, str]:
        if any(v is None for v in (self.N, self.E, self.k, self.T)):
            return (False, "SKIP: missing N, E, k, or T")
        if self.k <= 0 or self.T <= 0:
            return (False, "SKIP: k or T non-positive")
        try:
            base = self.N / (1.0 + math.exp(self.E / (2.0 * self.k * self.T)))
            delta = 0.01 * abs(self.E) if self.E != 0 else 0.01
            E_pert = self.E + delta
            perturbed = self.N / (1.0 + math.exp(E_pert / (2.0 * self.k * self.T)))
            if perturbed < base + 1e-12:
                return True, f"n_eq decreases with E: base={base:.4f}, perturbed={perturbed:.4f}"
            return False, f"n_eq does not decrease with E: base={base}, perturbed={perturbed}"
        except Exception as e:
            return (False, f"Error in monotonicity E test: {e}")

    # ------------------------------------------------------------------
    # Weak final-answer sanity (at most one, no hardcoded target values)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        if self.n_eq is None:
            return (False, "SKIP: n_eq not available")
        if math.isfinite(self.n_eq) and self.n_eq > 0:
            return True, f"n_eq = {self.n_eq} is positive and finite"
        return False, f"n_eq = {self.n_eq} is not a positive finite number"

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs