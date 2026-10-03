import math
import traceback
from typing import Tuple, Any

class PhysicsChecker:
    """
    Audits a solution for the work required to bring a point charge to the center
    of a neutral conducting spherical shell through a tiny hole.
    """

    def __init__(self, solution_instance: object):
        # Extract givens safely with None defaults
        self.q = getattr(solution_instance, 'q', None)
        self.R = getattr(solution_instance, 'R', None)
        self.t = getattr(solution_instance, 't', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Also try from .givens dict
        givens = getattr(solution_instance, 'givens', {})
        if givens:
            self.q = givens.get('q', self.q)
            self.R = givens.get('R', self.R)
            self.t = givens.get('t', self.t)
            self.epsilon0 = givens.get('epsilon0', self.epsilon0)

        # Extract the result key 'W' from .results
        results = getattr(solution_instance, 'results', {})
        self.W = results.get('W', None)

        # Store explanation steps for reference (not used in tests)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Private constants
        self._pi = math.pi

    def _is_valid_number(self, x) -> bool:
        """Check if x is a finite number (int or float, not None, not NaN, not inf)."""
        if x is None:
            return False
        if isinstance(x, (int, float)):
            return math.isfinite(x)
        return False

    def _missing_givens(self) -> bool:
        """Return True if any required given is missing or not a finite number."""
        return not all(self._is_valid_number(v) for v in [self.q, self.R, self.t, self.epsilon0])

    def _theoretical_work(self, q, R, t, epsilon0) -> float:
        """Compute the theoretical work using the energy method: W = q^2/(8πϵ0)*(1/R - 1/(R+t))."""
        return (q**2) / (8 * self._pi * epsilon0) * (1.0 / R - 1.0 / (R + t))

    def _small_t_approximation(self, q, R, t, epsilon0) -> float:
        """First-order approximation when t << R: W ≈ q^2 * t / (8πϵ0 * R^2)."""
        return (q**2) * t / (8 * self._pi * epsilon0 * (R**2))

    # --------------------------------------------------------------
    # INTERFACE TESTS
    # --------------------------------------------------------------
    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that all four givens (q, R, t, epsilon0) are present and finite."""
        if self._missing_givens():
            return (False, "SKIP: one or more required givens missing or non-finite")
        return (True, "All required givens present and finite")

    def test_interface_W_present(self) -> Tuple[bool, str]:
        """Check that the result key 'W' is present and is a finite number."""
        if self.W is None:
            return (False, "SKIP: W is None")
        if not self._is_valid_number(self.W):
            return (False, "W is not a finite number")
        return (True, "W is present and finite")

    # --------------------------------------------------------------
    # INVARIANT TESTS (theorem-based physical sanity)
    # --------------------------------------------------------------
    def test_invariant_W_positive(self) -> Tuple[bool, str]:
        """Work required must be positive (external agent does positive work)."""
        if self.W is None:
            return (False, "SKIP: W not available")
        if not self._is_valid_number(self.W):
            return (False, "W is not a finite number")
        if self.W <= 0:
            return (False, f"W = {self.W} is not positive")
        return (True, "W is positive")

    def test_invariant_W_finite(self) -> Tuple[bool, str]:
        """Work must be finite."""
        if self.W is None:
            return (False, "SKIP: W not available")
        if not self._is_valid_number(self.W):
            return (False, "W is not a finite number")
        return (True, "W is finite")

    def test_invariant_zero_thickness_limit(self) -> Tuple[bool, str]:
        """If t=0 (zero-thickness shell), the work should be zero."""
        if self._missing_givens():
            return (False, "SKIP: givens missing")
        if not self._is_valid_number(self.t) or self.t != 0:
            return (False, "SKIP: t is not zero")
        if self.W is None:
            return (False, "SKIP: W not available")
        expected = 0.0
        if not math.isclose(self.W, expected, rel_tol=1e-9, abs_tol=1e-12):
            return (False, f"Expected W=0 for t=0, got {self.W}")
        return (True, "W=0 for t=0 as expected")

    # --------------------------------------------------------------
    # MODELING / TRACE TESTS
    # --------------------------------------------------------------
    def test_modeling_energy_method_formula(self) -> Tuple[bool, str]:
        """Verify that the provided work matches the theoretical energy method result."""
        if self._missing_givens():
            return (False, "SKIP: required givens missing")
        if self.W is None:
            return (False, "SKIP: W not available")
        try:
            expected = self._theoretical_work(self.q, self.R, self.t, self.epsilon0)
        except (ZeroDivisionError, OverflowError) as e:
            return (False, f"Error computing expected work: {e}")
        if not math.isclose(self.W, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W = {self.W} does not match expected {expected} (rel_tol=1e-2)")
        return (True, "W matches the theoretical energy method formula")

    def test_modeling_small_thickness_approximation(self) -> Tuple[bool, str]:
        """If t is small relative to R, the work should be close to the first-order approximation."""
        if self._missing_givens():
            return (False, "SKIP: givens missing")
        if self.W is None:
            return (False, "SKIP: W not available")
        # Condition: t / R < 0.1 to consider it small
        if self.t / self.R > 0.1:
            return (False, "SKIP: t is not small enough (t/R > 0.1)")
        try:
            approx = self._small_t_approximation(self.q, self.R, self.t, self.epsilon0)
            exact = self._theoretical_work(self.q, self.R, self.t, self.epsilon0)
        except (ZeroDivisionError, OverflowError) as e:
            return (False, f"Error computing approximation: {e}")
        # Check that the exact and approximation are close (relative difference < 5%)
        if not math.isclose(exact, approx, rel_tol=1e-2, abs_tol=1e-12):
            return (False, f"Exact W = {exact} and small-t approximation = {approx} disagree")
        # Now check that the solution's W is also close to the approximation (should be because it matches exact)
        if not math.isclose(self.W, approx, rel_tol=5e-2, abs_tol=1e-12):
            return (False, f"Solution W = {self.W} does not match small-t approximation = {approx}")
        return (True, "For small t, work is consistent with linear approximation")

    # --------------------------------------------------------------
    # MONOTONICITY TESTS (perturb input, check output trend)
    # --------------------------------------------------------------
    def test_monotonicity_R_increases_decreases_work(self) -> Tuple[bool, str]:
        """Increasing the inner radius R should decrease the required work."""
        if self._missing_givens():
            return (False, "SKIP: givens missing")
        if self.R <= 0:
            return (False, "SKIP: R must be positive")
        if self.t < 0:
            return (False, "SKIP: t must be non-negative")
        try:
            baseline = self._theoretical_work(self.q, self.R, self.t, self.epsilon0)
            # Perturb R by +10%
            R2 = self.R * 1.10
            perturbed = self._theoretical_work(self.q, R2, self.t, self.epsilon0)
        except (ZeroDivisionError, OverflowError) as e:
            return (False, f"Error during computation: {e}")
        if perturbed >= baseline:
            return (False, f"Increasing R should decrease work, but perturbed work {perturbed} >= baseline {baseline}")
        return (True, "Increasing R decreases work as expected")

    def test_monotonicity_t_increases_increases_work(self) -> Tuple[bool, str]:
        """Increasing the shell thickness t should increase the required work."""
        if self._missing_givens():
            return (False, "SKIP: givens missing")
        if self.t < 0:
            return (False, "SKIP: t must be non-negative")
        try:
            baseline = self._theoretical_work(self.q, self.R, self.t, self.epsilon0)
            # Perturb t by +10% (if t=0, use small positive perturbation)
            if self.t == 0:
                dt = 0.01 * self.R  # small relative to R
            else:
                dt = 0.10 * self.t
            t2 = self.t + dt
            perturbed = self._theoretical_work(self.q, self.R, t2, self.epsilon0)
        except (ZeroDivisionError, OverflowError) as e:
            return (False, f"Error during computation: {e}")
        if perturbed <= baseline:
            return (False, f"Increasing t should increase work, but perturbed work {perturbed} <= baseline {baseline}")
        return (True, "Increasing t increases work as expected")

    # --------------------------------------------------------------
    # WEAK ANSWER SANITY (at most one)
    # --------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Weak sanity: W must be finite, positive, and less than q^2/(4πϵ0 R) (an upper bound)."""
        if self.W is None:
            return (False, "SKIP: W not available")
        if not self._is_valid_number(self.W):
            return (False, "W is not a finite number")
        if self.W <= 0:
            return (False, f"W = {self.W} is not positive")
        # Upper bound estimate: 1/R - 1/(R+t) < 1/R, so W < q^2/(4πϵ0 R)
        if self._missing_givens():
            return (True, "W is finite and positive (remaining bounds require givens)")
        try:
            upper_bound = (self.q**2) / (4 * self._pi * self.epsilon0 * self.R)
        except (ZeroDivisionError, OverflowError):
            return (True, "W is finite and positive (upper bound not computable)")
        if self.W >= upper_bound:
            return (False, f"W = {self.W} exceeds upper bound q^2/(4πϵ0 R) = {upper_bound}")
        return (True, "W passes weak sanity checks")

    # --------------------------------------------------------------
    # RUN ALL TESTS
    # --------------------------------------------------------------
    def run_all_tests(self):
        """Discover, execute all test_* methods, return (all_pass, logs)."""
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