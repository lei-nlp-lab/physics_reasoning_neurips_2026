import math
import traceback
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Physics checker for the adsorption problem:
    fraction of molecules with exactly one occupied site.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all required quantities using safe defaults (None).
        sources: solution_instance.results dict, solution_instance.givens dict,
                 and direct attributes of solution_instance.
        """
        # Result key
        self.f: float | None = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}
        self.f = results.get('f', None)

        # Given quantities (look in multiple places)
        self.T: float | None = None
        self.P: float | None = None
        self.k: float | None = None
        self.epsilon0: float | None = None
        self.p0: float | None = None

        # Try to extract from a 'givens' attribute dict
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
            self.T = givens.get('T', getattr(solution_instance, 'T', None))
            self.P = givens.get('P', getattr(solution_instance, 'P', None))
            self.k = givens.get('k', getattr(solution_instance, 'k', None))
            self.epsilon0 = givens.get('epsilon0', getattr(solution_instance, 'epsilon0', None))
            self.p0 = givens.get('p0', getattr(solution_instance, 'p0', None))
        else:
            # Fallback to direct attributes
            self.T = getattr(solution_instance, 'T', None)
            self.P = getattr(solution_instance, 'P', None)
            self.k = getattr(solution_instance, 'k', None)
            self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
            self.p0 = getattr(solution_instance, 'p0', None)

    # ------------------------------------------------------------------------
    # Helper: compute expected fraction using correct physics
    # ------------------------------------------------------------------------
    def _compute_f(self, P: float, T: float) -> float:
        """
        Compute f = 2θ(1-θ) where θ is the occupancy per site,
        using the grand partition function (correct expression).
        Formula: f = 2 p0 P (1+exp(-ε0/(2kT))) /
                       (p0 + P (1+exp(-ε0/(2kT))))**2
        """
        if any(x is None for x in (self.epsilon0, self.k, self.p0)):
            raise ValueError("Missing required constants (epsilon0, k, p0) for f calculation.")
        beta = 1.0 / (self.k * T)
        exp_factor = math.exp(-self.epsilon0 * beta / 2.0)
        numerator = 2.0 * self.p0 * P * (1.0 + exp_factor)
        denominator = self.p0 + P * (1.0 + exp_factor)
        return numerator / (denominator ** 2)

    # ------------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------------
    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that all required given variables are present and positive."""
        required = {
            'T': self.T,
            'P': self.P,
            'k': self.k,
            'epsilon0': self.epsilon0,
            'p0': self.p0,
        }
        missing = [name for name, val in required.items() if val is None]
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        # check positivity
        non_positive = [name for name, val in required.items() if val <= 0.0]
        if non_positive:
            return (False, f"Non-positive givens: {', '.join(non_positive)}")
        return (True, f"All required givens present and positive: T={self.T}, P={self.P}, k={self.k}, ε0={self.epsilon0}, p0={self.p0}")

    def test_interface_results_f(self) -> Tuple[bool, str]:
        """Check that result key 'f' exists and is numeric."""
        if self.f is None:
            return (False, "Missing result key 'f' (None)")
        if not isinstance(self.f, (int, float)):
            return (False, f"f is not numeric, got type {type(self.f).__name__}")
        if math.isnan(self.f) or math.isinf(self.f):
            return (False, f"f is NaN or infinite: {self.f}")
        return (True, f"f is present and numeric: {self.f}")

    def test_invariant_f_range(self) -> Tuple[bool, str]:
        """
        f = 2θ(1-θ) must be between 0 and 0.5 (inclusive).
        """
        if self.f is None:
            return (False, "SKIP: f not available")
        if not (0.0 <= self.f <= 0.5):
            return (False, f"f out of physical range [0, 0.5]: {self.f}")
        return (True, f"f={self.f} is within [0, 0.5]")

    def test_invariant_f_finite(self) -> Tuple[bool, str]:
        """f must be a finite number."""
        if self.f is None:
            return (False, "SKIP: f not available")
        if not math.isfinite(self.f):
            return (False, f"f is not finite: {self.f}")
        return (True, f"f={self.f} is finite")

    def test_invariant_dimensionless(self) -> Tuple[bool, str]:
        """
        f is a probability, so it must be dimensionless.
        Since we cannot check units directly, we verify it's a plain float/int.
        """
        if self.f is None:
            return (False, "SKIP: f not available")
        if isinstance(self.f, (int, float)):
            return (True, f"f={self.f} is dimensionless (numeric)")
        return (False, f"f has non-numeric type: {type(self.f).__name__}")

    def test_modeling_f_via_grand_partition(self) -> Tuple[bool, str]:
        """
        Re-derive f from the grand partition function using the given constants
        and compare with the solution's f using fuzzy matching.
        """
        if any(x is None for x in (self.f, self.T, self.P, self.k, self.epsilon0, self.p0)):
            return (False, "SKIP: insufficient data to compute expected f")
        try:
            expected = self._compute_f(self.P, self.T)
        except Exception as e:
            return (False, f"Error computing expected f: {e}")
        rel_tol = 1e-2
        abs_tol = 1e-2
        if not math.isclose(self.f, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            return (False,
                    f"f={self.f} differs from expected={expected} "
                    f"(rel_tol={rel_tol}, abs_tol={abs_tol})")
        return (True, f"f={self.f} matches expected={expected} within tolerances")

    def test_modeling_f_via_theta(self) -> Tuple[bool, str]:
        """
        Compute θ = occupancy per site from the grand partition,
        then f = 2θ(1-θ) and compare with solution's f.
        This tests the core relation f = 2θ(1-θ).
        """
        if any(x is None for x in (self.f, self.T, self.P, self.k, self.epsilon0, self.p0)):
            return (False, "SKIP: insufficient data")
        # Compute θ = (P/p0)(1+exp(-ε0/(2kT))) / [1 + (P/p0)(1+exp(-ε0/(2kT)))]
        beta = 1.0 / (self.k * self.T)
        exp_factor = math.exp(-self.epsilon0 * beta / 2.0)
        KP = self.P / self.p0 * (1.0 + exp_factor)
        if KP < 0:
            return (False, f"Computed KP={KP} is negative, implementation error?")
        theta = KP / (1.0 + KP)
        f_from_theta = 2.0 * theta * (1.0 - theta)
        rel_tol = 1e-2
        abs_tol = 1e-2
        if not math.isclose(self.f, f_from_theta, rel_tol=rel_tol, abs_tol=abs_tol):
            return (False,
                    f"f={self.f} differs from f_from_theta={f_from_theta} "
                    f"(rel_tol={rel_tol}, abs_tol={abs_tol})")
        return (True, f"f={self.f} agrees with f=2θ(1-θ) = {f_from_theta}")

    def test_monotonicity_f_decreases_with_p0(self) -> Tuple[bool, str]:
        """
        As p0 increases, the fraction f should strictly decrease
        (for fixed P, T, ε0, k).
        Perturb p0 by +10% and check that f decreases.
        """
        if any(x is None for x in (self.f, self.T, self.P, self.k, self.epsilon0, self.p0)):
            return (False, "SKIP: insufficient data")
        try:
            f_baseline = self._compute_f(self.P, self.T)
        except Exception:
            return (False, "SKIP: cannot compute baseline f")
        # Increase p0 by 10%
        p0_new = self.p0 * 1.1
        # Need to temporarily change self.p0 for computation
        old_p0 = self.p0
        self.p0 = p0_new
        try:
            f_new = self._compute_f(self.P, self.T)
        except Exception as e:
            self.p0 = old_p0
            return (False, f"Error computing f with increased p0: {e}")
        self.p0 = old_p0
        if f_new >= f_baseline:
            return (False, f"f did not decrease when p0 increased: baseline={f_baseline}, new={f_new}")
        return (True, f"f decreased from {f_baseline} to {f_new} after p0 increase by 10%")

    def test_monotonicity_f_increases_with_P(self) -> Tuple[bool, str]:
        """
        At very low P, f ~ 2*(P/p0)*(1+exp(...)) / 1  (linear)
        So f should increase with P for small P.
        Perturb P by +10% and check f increases.
        """
        if any(x is None for x in (self.f, self.T, self.P, self.k, self.epsilon0, self.p0)):
            return (False, "SKIP: insufficient data")
        try:
            f_baseline = self._compute_f(self.P, self.T)
        except Exception:
            return (False, "SKIP: cannot compute baseline f")
        # Increase P by 10%
        P_new = self.P * 1.1
        old_P = self.P
        self.P = P_new
        try:
            f_new = self._compute_f(self.P, self.T)
        except Exception as e:
            self.P = old_P
            return (False, f"Error computing f with increased P: {e}")
        self.P = old_P
        if f_new <= f_baseline:
            return (False, f"f did not increase when P increased: baseline={f_baseline}, new={f_new}")
        return (True, f"f increased from {f_baseline} to {f_new} after P increase by 10%")

    def test_answer_sanity_weak_f(self) -> Tuple[bool, str]:
        """
        Weak sanity check: f must be a sensible number.
        No comparison to expected values.
        """
        if self.f is None:
            return (False, "SKIP: f not available")
        if math.isnan(self.f) or math.isinf(self.f):
            return (False, f"f is NaN or inf: {self.f}")
        # Must be between 0 and 1 (actually <=0.5, but we allow up to 1 for safety)
        if self.f < 0.0 or self.f > 1.0:
            return (False, f"f={self.f} outside reasonable bounds [0,1]")
        return (True, f"f={self.f} passes weak sanity check")

    # ------------------------------------------------------------------------
    # run_all_tests: discover and run all test_ methods
    # ------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all test methods and return (all_passed, log_messages)."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)