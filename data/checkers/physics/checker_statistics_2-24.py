import math
import traceback
from typing import Tuple, List, Optional

# Try to import sympy; if not available, set a flag
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

class PhysicsChecker:
    """
    Physics checker for the two-level atoms problem (glass with two energy levels).
    Checks the derived heat capacities for identical atoms (Q1) and for a uniform
    distribution of gaps (Q2) at low temperature.
    """

    def __init__(self, solution_instance):
        # Safely extract givens
        self.N = getattr(solution_instance, 'N', None)
        self.delta = getattr(solution_instance, 'delta', None)          # for Q1
        self.delta0 = getattr(solution_instance, 'delta0', None)       # for Q2
        self.k = getattr(solution_instance, 'k', None)
        self.T = getattr(solution_instance, 'T', None)

        # Additional givens might be stored in a 'givens' dict; try that too
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
            if self.N is None and 'N' in givens:
                self.N = givens['N']
            if self.delta is None and 'delta' in givens:
                self.delta = givens['delta']
            if self.delta0 is None and 'delta0' in givens:
                self.delta0 = givens['delta0']
            if self.k is None and 'k' in givens:
                self.k = givens['k']
            if self.T is None and 'T' in givens:
                self.T = givens['T']

        # Safely extract results (using None defaults)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.C_total = self.results.get('C_total', None)
        self.c_avg_exact = self.results.get('c_avg_exact', None)
        self.c_avg_lowT = self.results.get('c_avg_lowT', None)

        # Explanation steps (optional, for planning tests)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_has_C_total(self) -> Tuple[bool, str]:
        """Check that C_total is present and not None."""
        if self.C_total is None:
            return (False, "C_total is missing or None")
        return (True, "C_total is present")

    def test_interface_has_c_avg_exact(self) -> Tuple[bool, str]:
        """Check that c_avg_exact is present and not None."""
        if self.c_avg_exact is None:
            return (False, "c_avg_exact is missing or None")
        return (True, "c_avg_exact is present")

    def test_interface_has_c_avg_lowT(self) -> Tuple[bool, str]:
        """Check that c_avg_lowT is present and not None."""
        if self.c_avg_lowT is None:
            return (False, "c_avg_lowT is missing or None")
        return (True, "c_avg_lowT is present")

    def test_interface_required_givens_available(self) -> Tuple[bool, str]:
        """Check that all necessary givens (N, delta, k, T, delta0) are available."""
        missing = []
        if self.N is None:
            missing.append("N")
        if self.delta is None:
            missing.append("delta")
        if self.k is None:
            missing.append("k")
        if self.T is None:
            missing.append("T")
        if self.delta0 is None:
            missing.append("delta0")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens available")

    # ----------------------------------------------------------------------
    # Invariant / theorem-based tests
    # ----------------------------------------------------------------------
    def test_invariant_partition_function(self) -> Tuple[bool, str]:
        """Check that the partition function for one atom is 2 cosh(beta*delta)."""
        if self.delta is None or self.k is None or self.T is None:
            return (False, "SKIP: missing delta, k, or T")
        if self.T == 0.0:
            return (False, "SKIP: T=0 leads to infinite beta")
        beta = 1.0 / (self.k * self.T)
        # We do not have Z directly; we only have derived quantities.
        # But we can check the internal consistency: if we have C_total and N,
        # we can derive beta*delta and verify the C_total formula.
        # Instead, we compute the expected C_total for given delta, T, N and compare.
        if self.C_total is not None and self.N is not None:
            x = beta * self.delta
            expected_per_atom = self.k * x**2 * (1.0 / math.cosh(x))**2
            expected_total = self.N * expected_per_atom
            if math.isclose(self.C_total, expected_total, rel_tol=1e-2, abs_tol=1e-8):
                return (True, "C_total matches partition function derived formula")
            else:
                return (False, f"C_total {self.C_total} does not match expected {expected_total}")
        else:
            return (False, "SKIP: need C_total and N for this check")

    def test_invariant_specific_heat_form(self) -> Tuple[bool, str]:
        """Check that c_avg_exact has the form of an integral weighted by 1/delta0."""
        if self.c_avg_exact is None or self.k is None or self.T is None or self.delta0 is None:
            return (False, "SKIP: missing c_avg_exact, k, T, or delta0")
        # The exact average is an integral from 0 to delta0 of the single-atom heat capacity.
        # We can numerically compute a simple Riemann sum and compare.
        if self.T == 0.0:
            return (False, "SKIP: T=0")
        beta = 1.0 / (self.k * self.T)
        # Use a coarse numerical integration to check consistency
        n_steps = 1000
        d_delta = self.delta0 / n_steps
        integral = 0.0
        for i in range(n_steps):
            d = (i + 0.5) * d_delta
            x = beta * d
            if x > 700:   # avoid overflow in cosh
                c = 0.0
            else:
                c = self.k * x**2 / (math.cosh(x)**2)
            integral += c * d_delta
        expected_avg = integral / self.delta0
        if math.isclose(self.c_avg_exact, expected_avg, rel_tol=1e-2, abs_tol=1e-12):
            return (True, "c_avg_exact consistent with numerical integration")
        else:
            return (False, f"c_avg_exact {self.c_avg_exact} does not match numerical {expected_avg}")

    def test_invariant_lowT_approx_form(self) -> Tuple[bool, str]:
        """Check that c_avg_lowT is linear in T and of the form (k^2 T)/delta0."""
        if self.c_avg_lowT is None or self.k is None or self.T is None or self.delta0 is None:
            return (False, "SKIP: missing c_avg_lowT, k, T, or delta0")
        if self.k == 0.0 or self.delta0 == 0.0:
            return (False, "SKIP: Division by zero avoided")
        expected = (self.k**2 * self.T) / self.delta0
        # Allow some tolerance due to approximations in the derivation.
        if math.isclose(self.c_avg_lowT, expected, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"c_avg_lowT matches expected linear term {expected}")
        else:
            return (False, f"c_avg_lowT {self.c_avg_lowT} not close to expected {expected}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_lowT_consistency_with_exact(self) -> Tuple[bool, str]:
        """For low enough T (kT << delta0), c_avg_lowT should approximate c_avg_exact."""
        if self.c_avg_exact is None or self.c_avg_lowT is None:
            return (False, "SKIP: missing c_avg_exact or c_avg_lowT")
        if self.T is None or self.delta0 is None or self.k is None:
            return (False, "SKIP: missing T, delta0, or k")
        # Check low T condition: kT << delta0
        if self.delta0 == 0.0:
            return (False, "SKIP: delta0=0")
        if not (self.k * self.T < 0.1 * self.delta0):
            return (True, "SKIP: not in low temperature regime, test not applicable")
        # Compare the two values relative to the exact one
        if abs(self.c_avg_exact) < 1e-30:
            relative_diff = abs(self.c_avg_lowT - self.c_avg_exact)
        else:
            relative_diff = abs(self.c_avg_lowT - self.c_avg_exact) / abs(self.c_avg_exact)
        if relative_diff < 0.2:  # Allow up to 20% deviation due to approximation
            return (True, f"Low T approximation consistent with exact: relative diff {relative_diff:.4f}")
        else:
            return (False, f"Low T approximation deviates too much: relative diff {relative_diff:.4f}")

    def test_modeling_exact_integral_rho_uniform(self) -> Tuple[bool, str]:
        """Check that c_avg_exact is derived from integrating the single-atom heat capacity over uniform distribution."""
        # This is more of a structural check: the explanation steps should mention integration.
        # We check if any step describes the integral.
        if not self.explanation_steps:
            return (True, "SKIP: no explanation steps available")
        integration_mentioned = any("integral" in step.lower() or "∫" in step for step in self.explanation_steps)
        if integration_mentioned:
            return (True, "Integration over uniform distribution is mentioned in explanation")
        else:
            return (False, "Integration over uniform distribution not mentioned in explanation steps; modeling may be incomplete")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_c_avg_lowT_increases_with_T(self) -> Tuple[bool, str]:
        """c_avg_lowT should increase linearly with T (for positive T)."""
        if self.c_avg_lowT is None or self.T is None or self.k is None or self.delta0 is None:
            return (False, "SKIP: missing required data")
        if self.T <= 0.0:
            return (False, "SKIP: T is non-positive")
        # Perturb T upward by 10% and compute the expected low-T value for the perturbed T.
        T2 = self.T * 1.1
        expected_perturbed = (self.k**2 * T2) / self.delta0
        # The baseline is the solution's c_avg_lowT at original T.
        # For increasing function, perturbed should be larger than baseline.
        if expected_perturbed > self.c_avg_lowT:
            return (True, f"c_avg_lowT increases with T: perturbed {expected_perturbed:.4e} > baseline {self.c_avg_lowT:.4e}")
        else:
            return (False, f"c_avg_lowT does not increase as T increases: perturbed {expected_perturbed:.4e} <= baseline {self.c_avg_lowT:.4e}")

    def test_monotonicity_C_total_increases_with_T(self) -> Tuple[bool, str]:
        """For low temperatures (kT << delta), C_total should increase with T."""
        if self.C_total is None or self.T is None or self.N is None or self.delta is None or self.k is None:
            return (False, "SKIP: missing data")
        if self.T <= 0.0:
            return (False, "SKIP: non-positive T")
        # Ensure we are in the low temperature regime (so that function is increasing)
        beta = 1.0 / (self.k * self.T)
        x = beta * self.delta
        # Check that x is large enough (>~2) to be on the increasing side
        if x < 2.0:
            return (True, "SKIP: T is not low enough to guarantee monotonic increase (may be past Schottky peak)")
        # Baseline: the solution's C_total at original T
        base = self.C_total
        # Perturb T upward by 10% and compute expected C_total for that perturbed T.
        T2 = self.T * 1.1
        beta2 = 1.0 / (self.k * T2)
        x2 = beta2 * self.delta
        # Use the formula C_total = N * k * (x)^2 * sech^2(x)
        c_perturbed = self.N * self.k * x2**2 * (1.0 / math.cosh(x2))**2
        if c_perturbed > base:
            return (True, f"C_total increases with T at low T: perturbed {c_perturbed:.4e} > base {base:.4e}")
        else:
            return (False, f"C_total does not increase with T: perturbed {c_perturbed:.4e} <= base {base:.4e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity tests (only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_positivity(self) -> Tuple[bool, str]:
        """All result quantities must be non-negative and finite."""
        if self.C_total is not None and (self.C_total < 0 or not math.isfinite(self.C_total)):
            return (False, f"C_total is invalid: {self.C_total}")
        if self.c_avg_exact is not None and (self.c_avg_exact < 0 or not math.isfinite(self.c_avg_exact)):
            return (False, f"c_avg_exact is invalid: {self.c_avg_exact}")
        if self.c_avg_lowT is not None and (self.c_avg_lowT < 0 or not math.isfinite(self.c_avg_lowT)):
            return (False, f"c_avg_lowT is invalid: {self.c_avg_lowT}")
        return (True, "All results are non-negative and finite")

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)