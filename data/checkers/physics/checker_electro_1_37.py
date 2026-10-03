import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Physics auditor for the spherical condenser contraction problem.
    Checks invariants, modeling, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # ----- Extract givens and constants -----
        # We assume solution_instance may have attributes or a .results dict.
        # Try attributes first, then .results as fallback.
        def _get(name: str, default=None):
            val = getattr(solution_instance, name, None)
            if val is None:
                results = getattr(solution_instance, 'results', {})
                val = results.get(name, default)
            return val

        self.Q = _get('Q')
        self.a = _get('a')
        self.b = _get('b')
        self.a_prime = _get('a_prime') or _get("a'")  # allow both keys
        # eps0: standard value if not provided (universal constant)
        eps0_attr = _get('eps0')
        if eps0_attr is None:
            self.eps0 = 8.854187817e-12  # F/m, SI value
        else:
            self.eps0 = eps0_attr

        # ----- Extract main result -----
        self.W = _get('W')
        # Optional intermediate energies (not required but useful for checks)
        self.U_i = _get('U_i')
        self.U_f = _get('U_f')

        # Derived for internal calculations
        self.pi = math.pi

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that all required givens are present and have valid types."""
        missing = []
        if self.Q is None:
            missing.append("Q (charge)")
        if self.a is None:
            missing.append("a (initial outer radius)")
        if self.b is None:
            missing.append("b (inner radius)")
        if self.a_prime is None:
            missing.append("a_prime (final outer radius)")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        # Type checks
        if not isinstance(self.Q, (int, float)):
            return (False, "Q must be numeric")
        if not isinstance(self.a, (int, float)):
            return (False, "a must be numeric")
        if not isinstance(self.b, (int, float)):
            return (False, "b must be numeric")
        if not isinstance(self.a_prime, (int, float)):
            return (False, "a_prime must be numeric")
        return (True, "All required givens present and numeric.")

    def test_interface_result_W(self) -> Tuple[bool, str]:
        """Check that the result key 'W' exists and is numeric."""
        if self.W is None:
            return (False, "Missing result: W")
        if not isinstance(self.W, (int, float)):
            return (False, "W must be numeric")
        return (True, "W present and numeric.")

    # ----------------------------------------------------------------------
    # Invariant / theorem-based checks
    # ----------------------------------------------------------------------
    def test_invariant_work_equals_energy_decrease(self) -> Tuple[bool, str]:
        """Check that W = U_initial - U_final using the field energy formula."""
        if any(v is None for v in [self.Q, self.a, self.b, self.a_prime, self.W]):
            return (False, "SKIP: missing givens or result W")
        # Compute initial and final field energies: U = Q^2/(8 pi eps0) * (1/b - 1/r)
        try:
            U_i_calc = (self.Q**2) / (8 * self.pi * self.eps0) * (1/self.b - 1/self.a)
            U_f_calc = (self.Q**2) / (8 * self.pi * self.eps0) * (1/self.b - 1/self.a_prime)
            W_expected = U_i_calc - U_f_calc
        except ZeroDivisionError:
            return (False, "Division by zero: check radii values")
        if math.isclose(self.W, W_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"W matches energy decrease: |W - expected| = {abs(self.W - W_expected):.2e}")
        else:
            return (False, f"W does not match expected {W_expected:.4e}, got {self.W:.4e}")

    def test_invariant_work_positive(self) -> Tuple[bool, str]:
        """Work done by electric force should be positive (contraction reduces energy)."""
        if self.W is None:
            return (False, "SKIP: W not available")
        if self.W > 0:
            return (True, f"W = {self.W:.4e} > 0 (correct sign)")
        else:
            return (False, f"W must be positive, got {self.W:.4e}")

    def test_invariant_geometry_a_prime_less_than_a(self) -> Tuple[bool, str]:
        """Check contraction: a' < a."""
        if self.a is None or self.a_prime is None:
            return (False, "SKIP: missing a or a_prime")
        if self.a_prime < self.a:
            return (True, f"a' = {self.a_prime} < a = {self.a}")
        else:
            return (False, f"Contraction violated: a' = {self.a_prime} >= a = {self.a}")

    def test_invariant_geometry_a_prime_greater_than_b(self) -> Tuple[bool, str]:
        """Check that final radius is larger than inner sphere radius."""
        if self.b is None or self.a_prime is None:
            return (False, "SKIP: missing b or a_prime")
        if self.a_prime > self.b:
            return (True, f"a' = {self.a_prime} > b = {self.b}")
        else:
            return (False, f"Geometry invalid: a' = {self.a_prime} <= b = {self.b}")

    def test_invariant_field_integral_consistency(self) -> Tuple[bool, str]:
        """Check that the derived formula W = Q^2(a-a')/(8 pi eps0 a a') holds."""
        if any(v is None for v in [self.Q, self.a, self.a_prime, self.W]):
            return (False, "SKIP: missing Q, a, a_prime, or W")
        # Compute expected W from the compact formula
        try:
            W_formula = (self.Q**2 * (self.a - self.a_prime)) / (8 * self.pi * self.eps0 * self.a * self.a_prime)
        except ZeroDivisionError:
            return (False, "Division by zero: a or a_prime zero")
        if math.isclose(self.W, W_formula, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "W matches compact formula")
        else:
            return (False, f"W = {self.W:.4e}, formula gives {W_formula:.4e}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_energy_difference_sign(self) -> Tuple[bool, str]:
        """Check that U_i > U_f, so work is positive."""
        # We'll compute U_i and U_f from formula if not directly provided
        if any(v is None for v in [self.Q, self.a, self.b, self.a_prime]):
            return (False, "SKIP: missing givens")
        try:
            U_i = (self.Q**2) / (8 * self.pi * self.eps0) * (1/self.b - 1/self.a)
            U_f = (self.Q**2) / (8 * self.pi * self.eps0) * (1/self.b - 1/self.a_prime)
        except ZeroDivisionError:
            return (False, "Division by zero: check radii")
        if U_i > U_f:
            return (True, "Initial energy > final energy, consistent with positive work")
        else:
            return (False, f"Initial energy {U_i:.4e} not greater than final {U_f:.4e}")

    def test_modeling_capacitance_formula(self) -> Tuple[bool, str]:
        """Optional: Check that capacitance formula is consistent (if provided)."""
        # This test only runs if the solution provides C_i or C_f
        if hasattr(self, 'C_i') and self.C_i is not None:
            C_i_expected = 4 * self.pi * self.eps0 * self.a * self.b / (self.a - self.b)
            if math.isclose(self.C_i, C_i_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Initial capacitance consistent")
            else:
                return (False, f"C_i mismatch: {self.C_i} vs expected {C_i_expected}")
        return (True, "SKIP: no capacitance data provided")

    # ----------------------------------------------------------------------
    # Monotonicity checks
    # ----------------------------------------------------------------------
    def test_monotonicity_work_increases_with_contraction(self) -> Tuple[bool, str]:
        """
        Check monotonicity: if a' decreases (more contraction), work increases.
        We perturb the given a' by -1% and compute expected W.
        """
        if any(v is None for v in [self.a, self.a_prime, self.Q, self.W]):
            return (False, "SKIP: missing a, a_prime, Q, or W")
        if self.a_prime >= self.a:
            return (False, "SKIP: not a contraction (a' >= a)")
        # Compute W for a slightly smaller a''
        delta = 0.01 * self.a_prime
        a_new = self.a_prime - delta  # more contraction
        if a_new <= self.b:
            return (True, "SKIP: perturbation would violate geometry")
        # Expected W_new using the same Q, a, b, eps0
        try:
            W_new = (self.Q**2 * (self.a - a_new)) / (8 * self.pi * self.eps0 * self.a * a_new)
        except ZeroDivisionError:
            return (False, "Division by zero in perturbed calculation")
        if W_new > self.W:
            return (True, f"Work increases when a' decreases: W = {self.W:.4e}, W_new = {W_new:.4e}")
        else:
            return (False, f"Monotonicity violated: W_new = {W_new:.4e} <= W = {self.W:.4e}")

    def test_monotonicity_work_increases_with_charge(self) -> Tuple[bool, str]:
        """Check that increasing Q increases work (W ~ Q^2)."""
        if self.Q is None or self.W is None:
            return (False, "SKIP: missing Q or W")
        # Perturb Q by +5%
        Q_new = self.Q * 1.05
        # Recompute W using the same other givens (unchanged)
        if any(v is None for v in [self.a, self.a_prime]):
            return (False, "SKIP: missing a or a_prime")
        try:
            W_new = (Q_new**2 * (self.a - self.a_prime)) / (8 * self.pi * self.eps0 * self.a * self.a_prime)
        except ZeroDivisionError:
            return (False, "Division by zero")
        if W_new > self.W:
            return (True, f"Work increases with Q: W = {self.W:.4e}, Q+5% gives {W_new:.4e}")
        else:
            return (False, f"Monotonicity violated: Q+5% gives W = {W_new:.4e} <= original {self.W:.4e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one, no isclose, no re-derivation of answer)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """Check that W is finite, positive, and not NaN."""
        if self.W is None:
            return (False, "SKIP: W not provided")
        if math.isnan(self.W) or math.isinf(self.W):
            return (False, f"W is NaN or infinite: {self.W}")
        if self.W <= 0:
            return (False, f"W must be positive, got {self.W}")
        # Broad magnitude check: W should be roughly Q^2/(8 pi eps0) * (a)/(b^2)? Not exact.
        # Just ensure it's not absurdly large (e.g., > 1e20) for typical scale
        # Use typical values: Q=1e-6, a=0.1, b=0.05 => W ~ 1e-6 J. 1e20 is unrealistic.
        if self.W > 1e20:
            return (False, f"W too large ({self.W:.2e}) - likely unphysical")
        return (True, "W is finite, positive, and plausible in magnitude")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, return success flag and logs."""
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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)