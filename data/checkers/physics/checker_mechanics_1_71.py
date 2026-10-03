import math
import traceback
from typing import Any, Dict, Optional, Tuple, List

class PhysicsChecker:
    """
    Audits a solution for the precession of a periastron problem.
    The planet experiences an additional repulsive force F = A * r.
    Expected result key: 'omega_p' (angular velocity of precession).
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens – assume they are stored in a 'givens' dict or as attributes
        givens: Optional[Dict[str, Any]] = getattr(solution_instance, 'givens', None)
        if givens is None:
            # Fallback: try to get from instance attributes directly
            givens = {}

        self.G = givens.get('G', getattr(solution_instance, 'G', None))
        self.M = givens.get('M', getattr(solution_instance, 'M', None))
        self.m = givens.get('m', getattr(solution_instance, 'm', None))
        self.R = givens.get('R', getattr(solution_instance, 'R', None))
        self.A = givens.get('A', getattr(solution_instance, 'A', None))
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Extract results – assume stored in a 'results' dict
        results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.omega_p = results.get('omega_p', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, return (success, logs)."""
        logs: List[str] = []
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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the result key 'omega_p' exists and is a finite number."""
        if self.omega_p is None:
            return False, "Missing result key 'omega_p'."
        if not isinstance(self.omega_p, (int, float)):
            return False, f"omega_p is not numeric, got {type(self.omega_p)}"
        if math.isnan(self.omega_p):
            return False, "omega_p is NaN."
        if math.isinf(self.omega_p):
            return False, "omega_p is infinite."
        return True, "omega_p present and finite."

    def test_interface_inputs_provided(self) -> Tuple[bool, str]:
        """Check that all required givens are present and positive."""
        missing = []
        for name, val in [("G", self.G), ("M", self.M), ("m", self.m),
                          ("R", self.R), ("A", self.A)]:
            if val is None:
                missing.append(name)
            elif val <= 0:
                return False, f"Given '{name}' must be positive, got {val}"
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All givens present and positive."

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based sanity)
    # ------------------------------------------------------------------
    def test_invariant_radial_equation(self) -> Tuple[bool, str]:
        """
        Check the circular orbit condition from the effective potential.
        Equation: -L^2/(m R^3) + G M m / R^2 + A * R = 0
        Using L = m R^2 ω0 and ω0^2 = G M / R^3 - A/m .
        Derived condition reduces to identity, so we check that the
        given inputs satisfy ω0^2 = G M / R^3 - A/m exactly (within tolerance),
        which is a direct consequence.
        """
        if any(v is None for v in (self.G, self.M, self.m, self.R, self.A)):
            return False, "SKIP: missing givens"

        # Compute expected ω0^2 from the derived circular orbit condition
        term = self.G * self.M / self.R**3 - self.A / self.m
        if term <= 0:
            return False, f"GM/R^3 - A/m is not positive (={term}), no real orbit"

        # If we had the solution's ω0, we would compare, but we don't.
        # Instead, we verify that the derivative condition is satisfied.
        # Use L = m R^2 ω0, compute derivative numerically.
        # ω0 = sqrt(term)
        omega0 = math.sqrt(term)
        L = self.m * self.R**2 * omega0

        # dU_eff/dr = -L^2/(m r^3) + G M m / r^2 + A r
        dU_dr = -L**2 / (self.m * self.R**3) + self.G * self.m * self.M / self.R**2 + self.A * self.R
        if not math.isclose(dU_dr, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Effective potential derivative not zero: {dU_dr}"
        return True, "Circular orbit condition satisfied (dU_eff/dr ≈ 0)"

    def test_invariant_omega_r_squared_positive(self) -> Tuple[bool, str]:
        """
        Check that ω_r^2 = GM/R^3 - 4A/m is positive (required for real radial oscillations).
        """
        if any(v is None for v in (self.G, self.M, self.m, self.R, self.A)):
            return False, "SKIP: missing givens"

        omega_r2 = self.G * self.M / self.R**3 - 4 * self.A / self.m
        if omega_r2 <= 0:
            return False, f"ω_r^2 not positive (={omega_r2}); no oscillatory radial motion"
        return True, "ω_r^2 > 0"

    def test_invariant_precession_small_compared_to_orbital(self) -> Tuple[bool, str]:
        """
        For a nearly circular orbit, the precession frequency must be smaller
        than the orbital frequency ω0. Check that ω_p < ω0.
        """
        if any(v is None for v in (self.G, self.M, self.m, self.R, self.A, self.omega_p)):
            return False, "SKIP: missing data"

        omega0 = math.sqrt(self.G * self.M / self.R**3 - self.A / self.m)
        if self.omega_p >= omega0:
            return False, f"Precession {self.omega_p} not smaller than orbital ω0 = {omega0}"
        return True, "ω_p < ω0"

    # ------------------------------------------------------------------
    # Modeling / trace tests (consistency with derived formulas)
    # ------------------------------------------------------------------
    def test_modeling_precession_definition(self) -> Tuple[bool, str]:
        """
        Verify that ω_p = ω0 - ω_r, using derived expressions for ω0 and ω_r.
        This tests the core physical relationship from the explanation.
        """
        if any(v is None for v in (self.G, self.M, self.m, self.R, self.A, self.omega_p)):
            return False, "SKIP: missing data"

        omega0 = math.sqrt(self.G * self.M / self.R**3 - self.A / self.m)
        omega_r = math.sqrt(self.G * self.M / self.R**3 - 4 * self.A / self.m)
        expected_omega_p = omega0 - omega_r

        if not math.isclose(self.omega_p, expected_omega_p, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"ω_p mismatch: got {self.omega_p}, expected {expected_omega_p}"
        return True, "ω_p = ω0 - ω_r holds within tolerance"

    def test_modeling_omega_r_from_second_derivative(self) -> Tuple[bool, str]:
        """
        Check that ω_r^2 derived from the second derivative of U_eff equals
        GM/R^3 - 4A/m, using the effective potential expression.
        """
        if any(v is None for v in (self.G, self.M, self.m, self.R, self.A)):
            return False, "SKIP: missing givens"

        omega0 = math.sqrt(self.G * self.M / self.R**3 - self.A / self.m)
        L = self.m * self.R**2 * omega0
        # d^2U_eff/dr^2 = 3L^2/(m r^4) - 2GMm/r^3 - A
        d2U_dr2 = 3 * L**2 / (self.m * self.R**4) - 2 * self.G * self.m * self.M / self.R**3 - self.A
        omega_r2_from_deriv = d2U_dr2 / self.m
        expected_omega_r2 = self.G * self.M / self.R**3 - 4 * self.A / self.m

        if not math.isclose(omega_r2_from_deriv, expected_omega_r2, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"ω_r^2 from derivative = {omega_r2_from_deriv}, expected {expected_omega_r2}"
        return True, "Second derivative yields correct ω_r^2"

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks)
    # ------------------------------------------------------------------
    def test_monotonicity_precession_decreases_with_R(self) -> Tuple[bool, str]:
        """
        Check that increasing the orbital radius R (while keeping others constant)
        decreases the precession frequency ω_p.
        """
        if any(v is None for v in (self.G, self.M, self.m, self.R, self.A)):
            return False, "SKIP: missing givens"

        # Baseline ω_p
        omega0 = math.sqrt(self.G * self.M / self.R**3 - self.A / self.m)
        omega_r = math.sqrt(self.G * self.M / self.R**3 - 4 * self.A / self.m)
        baseline = omega0 - omega_r

        # Increase R by 10%
        R_new = self.R * 1.10
        omega0_new = math.sqrt(self.G * self.M / R_new**3 - self.A / self.m)
        omega_r_new = math.sqrt(self.G * self.M / R_new**3 - 4 * self.A / self.m)
        new_value = omega0_new - omega_r_new

        if new_value >= baseline:
            return False, f"ω_p increased (new={new_value}) instead of decreased (baseline={baseline})"
        return True, "ω_p monotonically decreases with increasing R"

    def test_monotonicity_precession_increase_with_A_near_zero(self) -> Tuple[bool, str]:
        """
        For very small A, ω_p increases as A increases (since derivative is positive at A=0).
        Check that if we slightly increase A from zero, ω_p increases.
        """
        if any(v is None for v in (self.G, self.M, self.m, self.R)):
            return False, "SKIP: missing givens"

        # Use a small A = 0 for baseline, then small positive A2
        A_baseline = 0.0
        A_small = 1e-3 * (self.G * self.M / self.R**3) * self.m  # small fraction
        omega0_base = math.sqrt(self.G * self.M / self.R**3 - A_baseline / self.m)
        omega_r_base = math.sqrt(self.G * self.M / self.R**3 - 4 * A_baseline / self.m)
        baseline = omega0_base - omega_r_base  # should be 0

        omega0_new = math.sqrt(self.G * self.M / self.R**3 - A_small / self.m)
        omega_r_new = math.sqrt(self.G * self.M / self.R**3 - 4 * A_small / self.m)
        new_value = omega0_new - omega_r_new

        if new_value <= baseline:
            return False, f"For small A, ω_p did not increase (new={new_value}, baseline={baseline})"
        return True, "ω_p increases for small positive A (as expected)"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, very broad)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """
        Weak check: ω_p should be a positive number less than the typical orbital frequency.
        No exact comparison; just sign and order-of-magnitude.
        """
        if self.omega_p is None:
            return False, "SKIP: no omega_p"
        if not isinstance(self.omega_p, (int, float)):
            return False, "omega_p is not numeric"
        if math.isnan(self.omega_p) or math.isinf(self.omega_p):
            return False, "omega_p is NaN or infinite"

        # A plausible range: should not exceed 10^10 rad/s (very fast precession)
        if self.omega_p <= 0:
            return False, f"omega_p must be positive, got {self.omega_p}"
        if self.omega_p > 1e10:
            return False, f"omega_p unreasonably large: {self.omega_p}"
        return True, "omega_p is positive and within a broad plausible range"