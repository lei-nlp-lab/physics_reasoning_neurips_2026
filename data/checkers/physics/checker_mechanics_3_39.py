import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Checker for the relativistic charged particle in a B field problem.
    Implements multiple unit-test-like methods for auditing physics consistency.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens from the solution instance (could be attributes or results dict)
        self.q = None
        self.m = None
        self.R = None
        self.omega = None
        self.beta = None
        self.c = None
        self.tau = None  # proper time

        # Try to get from instance attributes first, then from results dict
        if hasattr(solution_instance, 'givens'):
            g = solution_instance.givens
            self.q = g.get('q')
            self.m = g.get('m')
            self.R = g.get('R')
            self.omega = g.get('omega')
            self.beta = g.get('beta')
            self.c = g.get('c')
            self.tau = g.get('tau')
        else:
            self.q = getattr(solution_instance, 'q', None)
            self.m = getattr(solution_instance, 'm', None)
            self.R = getattr(solution_instance, 'R', None)
            self.omega = getattr(solution_instance, 'omega', None)
            self.beta = getattr(solution_instance, 'beta', None)
            self.c = getattr(solution_instance, 'c', None)
            self.tau = getattr(solution_instance, 'tau', None)

        # Extract results dict; if missing, use empty dict
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Specific result keys from pseudocode contract
        self.B = self.results.get('B', None)
        self.u_prime_0 = self.results.get('u_prime_0', None)
        self.du_prime_0_dtau = self.results.get('du_prime_0_dtau', None)
        self.dE_dtau = self.results.get('dE_dtau', None)

    def _safe_gamma(self) -> Optional[float]:
        """Compute gamma = 1/sqrt(1 - (omega*R/c)^2) if all needed values present."""
        if None in (self.omega, self.R, self.c):
            return None
        ratio = self.omega * self.R / self.c
        if ratio >= 1:
            return None  # invalid physical regime
        try:
            return 1.0 / math.sqrt(1 - ratio ** 2)
        except (ValueError, ZeroDivisionError):
            return None

    def _safe_gamma_b(self) -> Optional[float]:
        """Compute gamma_b = 1/sqrt(1 - beta^2)."""
        if self.beta is None:
            return None
        if abs(self.beta) >= 1:
            return None
        try:
            return 1.0 / math.sqrt(1 - self.beta ** 2)
        except (ValueError, ZeroDivisionError):
            return None

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, run them, collect logs."""
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

    # ------------------------------------------------------------------
    # Interface checks: required keys exist and are non-None
    # ------------------------------------------------------------------
    def test_interface_B_exists(self) -> Tuple[bool, str]:
        if self.B is None:
            return (False, "Missing result key 'B'")
        return (True, "B is present")

    def test_interface_u_prime_0_exists(self) -> Tuple[bool, str]:
        if self.u_prime_0 is None:
            return (False, "Missing result key 'u_prime_0'")
        return (True, "u_prime_0 is present")

    def test_interface_du_prime_0_dtau_exists(self) -> Tuple[bool, str]:
        if self.du_prime_0_dtau is None:
            return (False, "Missing result key 'du_prime_0_dtau'")
        return (True, "du_prime_0_dtau is present")

    def test_interface_dE_dtau_exists(self) -> Tuple[bool, str]:
        if self.dE_dtau is None:
            return (False, "Missing result key 'dE_dtau'")
        return (True, "dE_dtau is present")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        needed = ['q', 'm', 'R', 'omega', 'beta', 'c']
        missing = [var for var in needed if getattr(self, var) is None]
        if missing:
            return (False, f"Missing given parameters: {missing}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant checks: physics consistency using derived formulas
    # Use math.isclose with rel_tol=1e-2, abs_tol=1e-2
    # ------------------------------------------------------------------
    def test_invariant_B_consistency(self) -> Tuple[bool, str]:
        """Check that B matches q v B = gamma m v^2 / R with v=omega*R."""
        if any(var is None for var in [self.q, self.m, self.R, self.omega, self.c]):
            return (False, "SKIP: missing givens")
        gamma = self._safe_gamma()
        if gamma is None:
            return (False, "Cannot compute gamma (invalid ratio)")
        expected_B = gamma * self.m * self.omega / self.q
        if not math.isfinite(expected_B):
            return (False, f"Expected B not finite: {expected_B}")
        if not math.isfinite(self.B):
            return (False, f"Result B not finite: {self.B}")
        if math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"B matches: expected {expected_B:.6e}, got {self.B:.6e}")
        else:
            return (False, f"B mismatch: expected {expected_B:.6e}, got {self.B:.6e}")

    def test_invariant_u_prime_0_initial_value(self) -> Tuple[bool, str]:
        """Check that at tau=0, u_prime_0 = gamma_b * gamma * c."""
        if any(var is None for var in [self.u_prime_0, self.beta, self.c]):
            return (False, "SKIP: missing u_prime_0, beta, or c")
        gamma = self._safe_gamma()
        gamma_b = self._safe_gamma_b()
        if gamma is None or gamma_b is None:
            return (False, "Cannot compute gamma or gamma_b")
        # Expected u_prime_0 at tau=0 (with phi=0): gamma_b * gamma * c
        expected = gamma_b * gamma * self.c
        # We need to evaluate u_prime_0 at tau=0. Since it may be a symbolic expression,
        # we attempt to substitute tau=0. If it's numeric, we assume it corresponds to tau=0.
        # For simplicity, we treat as numeric value given at tau=0 (as per pseudocode).
        # Real integration would require a function; we assume the result is evaluated at tau=0.
        # This is a weak check but acceptable for invariant.
        if hasattr(self.u_prime_0, 'subs'):
            # Sympy expression
            try:
                val = float(self.u_prime_0.subs(self.tau if self.tau is not None else 0, 0))
            except:
                val = float(self.u_prime_0)  # fallback
        else:
            val = float(self.u_prime_0)
        if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"u_prime_0(0) matches: expected {expected:.6e}, got {val:.6e}")
        else:
            return (False, f"u_prime_0(0) mismatch: expected {expected:.6e}, got {val:.6e}")

    def test_invariant_du_prime_0_dtau_form(self) -> Tuple[bool, str]:
        """Check the derivative amplitude at tau=0 against expected expression."""
        # Expected: du'^0/dtau = (gamma^2 * omega^2 * R * beta * gamma_b) / c^2 * cos(...)
        # At tau=0, cos = 1.
        if any(var is None for var in [self.du_prime_0_dtau, self.omega, self.R,
                                        self.beta, self.c, self.q, self.m, self.B]):
            return (False, "SKIP: missing data for du_prime_0_dtau check")
        gamma = self._safe_gamma()
        gamma_b = self._safe_gamma_b()
        if gamma is None or gamma_b is None:
            return (False, "Cannot compute gamma or gamma_b")
        # Derive amplitude using provided formula: A = (q/(m*c^2)) * gamma_b * beta * B * gamma * omega * R
        # Since B = gamma * m * omega / q, substitute:
        # A = (q/(m*c^2)) * gamma_b * beta * (gamma*m*omega/q) * gamma * omega * R
        #   = (gamma_b * beta * gamma^2 * omega^2 * R) / c^2
        # which matches expected amplitude.
        amplitude_expected = (gamma_b * self.beta * (gamma**2) * (self.omega**2) * self.R) / (self.c**2)
        if not math.isfinite(amplitude_expected):
            return (False, f"Expected amplitude not finite: {amplitude_expected}")
        # Extract amplitude from result at tau=0 (cos = 1)
        if hasattr(self.du_prime_0_dtau, 'subs'):
            try:
                val_at_zero = float(self.du_prime_0_dtau.subs(
                    self.tau if self.tau is not None else 0, 0))
            except:
                val_at_zero = float(self.du_prime_0_dtau)
        else:
            val_at_zero = float(self.du_prime_0_dtau)
        if math.isclose(abs(val_at_zero), abs(amplitude_expected), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"du_prime_0_dtau amplitude matches: expected {amplitude_expected:.6e}, got {val_at_zero:.6e}")
        else:
            return (False, f"du_prime_0_dtau amplitude mismatch: expected {amplitude_expected:.6e}, got {val_at_zero:.6e}")

    def test_invariant_dE_dtau_relation(self) -> Tuple[bool, str]:
        """Check that dE/dtau = c * dp'^0/dtau, where dp'^0/dtau = m * du'^0/dtau."""
        if any(var is None for var in [self.dE_dtau, self.du_prime_0_dtau, self.m, self.c]):
            return (False, "SKIP: missing dE_dtau, du_prime_0_dtau, m, or c")
        # Compute expected dp'^0/dtau = m * du_prime_0_dtau
        dp0_dtau = self.m * self.du_prime_0_dtau
        expected_dE_dtau = self.c * dp0_dtau
        # Check numeric consistency (use isclose)
        if not math.isfinite(self.dE_dtau) or not math.isfinite(expected_dE_dtau):
            return (False, f"dE_dtau or expected not finite")
        if math.isclose(self.dE_dtau, expected_dE_dtau, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "dE/dtau = c * dp'^0/dtau holds")
        else:
            return (False, f"dE/dtau mismatch: expected {expected_dE_dtau:.6e}, got {self.dE_dtau:.6e}")

    # ------------------------------------------------------------------
    # Modeling checks: based on explanation steps, not hardcoding full solution
    # ------------------------------------------------------------------
    def test_modeling_B_expression_gamma_factor(self) -> Tuple[bool, str]:
        """Verify that B expression involves 1/sqrt(1 - (omega*R/c)^2) factor."""
        if self.B is None or self.omega is None or self.R is None or self.c is None:
            return (False, "SKIP: missing data")
        # Check that B is proportional to gamma (i.e., contains relativistic factor)
        # We can compare B vs non-relativistic value B0 = m*omega/q
        gamma = self._safe_gamma()
        if gamma is None:
            return (False, "Cannot compute gamma")
        B0 = self.m * self.omega / self.q if self.q != 0 else None
        if B0 is None or B0 == 0:
            return (False, "Cannot compute non-relativistic B")
        ratio = self.B / B0
        if math.isclose(ratio, gamma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"B contains gamma factor: B/B0 = {ratio:.4f}, gamma = {gamma:.4f}")
        else:
            return (False, f"B does not have correct gamma factor: ratio {ratio:.4f} vs gamma {gamma:.4f}")

    def test_modeling_u_prime_0_contains_sine_term(self) -> Tuple[bool, str]:
        """Check that u_prime_0 expression is not constant but contains sin term."""
        if self.u_prime_0 is None:
            return (False, "SKIP: missing u_prime_0")
        # Evaluate at two different tau values (if symbolic) to see if it varies.
        # If it's a numeric constant, it would fail.
        # We'll try to evaluate at tau=0 and tau=1 (if tau is known).
        # For symbolic, we check if it has 'sin' in string representation.
        # For numeric, we assume the user provided value at specific tau; we can't check variation.
        # We'll do a simple string check for 'sin' (weak but allowed).
        s = str(self.u_prime_0).lower()
        if 'sin' in s:
            return (True, "u_prime_0 expression contains sin term")
        else:
            # Possibly it's a numeric constant; we flag it as potentially wrong but not fail hard
            return (False, "u_prime_0 does not contain sin term; may be constant, which is incorrect")

    def test_modeling_du_prime_0_dtau_contains_cosine(self) -> Tuple[bool, str]:
        """Check that derivative contains cosine term."""
        if self.du_prime_0_dtau is None:
            return (False, "SKIP: missing du_prime_0_dtau")
        s = str(self.du_prime_0_dtau).lower()
        if 'cos' in s:
            return (True, "du_prime_0_dtau contains cos term")
        else:
            return (False, "du_prime_0_dtau does not contain cos term; may be constant")

    # ------------------------------------------------------------------
    # Monotonicity checks: perturb one input and see output trend
    # ------------------------------------------------------------------
    def test_monotonicity_B_increases_with_omega(self) -> Tuple[bool, str]:
        """Check that B increases when omega increases (for fixed q,m,R,c)."""
        if any(var is None for var in [self.q, self.m, self.R, self.omega, self.c, self.B]):
            return (False, "SKIP: missing data")
        # Baseline
        gamma_orig = self._safe_gamma()
        if gamma_orig is None:
            return (False, "Cannot compute gamma_orig")
        B_orig = self.B
        # Perturb omega by +5%
        omega_new = self.omega * 1.05
        ratio_new = omega_new * self.R / self.c
        if ratio_new >= 1:
            return (False, "Perturbed omega too high, would exceed c")
        gamma_new = 1.0 / math.sqrt(1 - ratio_new**2)
        B_new_expected = gamma_new * self.m * omega_new / self.q
        # Check direction: B_new > B_orig
        if B_new_expected > B_orig:
            return (True, f"B increases with omega: B_orig={B_orig:.6e}, B_new(1.05*omega)={B_new_expected:.6e}")
        else:
            return (False, f"B does not increase: B_orig={B_orig:.6e}, B_new={B_new_expected:.6e}")

    def test_monotonicity_u_prime_0_increases_with_beta(self) -> Tuple[bool, str]:
        """Check that u_prime_0 (at tau=0) increases when beta increases."""
        if any(var is None for var in [self.u_prime_0, self.beta, self.c, self.omega, self.R]):
            return (False, "SKIP: missing data")
        gamma = self._safe_gamma()
        if gamma is None:
            return (False, "Cannot compute gamma")
        # Baseline at beta = self.beta
        gamma_b_orig = self._safe_gamma_b()
        if gamma_b_orig is None:
            return (False, "Cannot compute gamma_b_orig")
        u0_orig = gamma_b_orig * gamma * self.c  # at tau=0
        # Increase beta by 10% (ensure still <1)
        beta_new = self.beta * 1.1
        if abs(beta_new) >= 1:
            return (False, "Perturbed beta too close to 1")
        gamma_b_new = 1.0 / math.sqrt(1 - beta_new**2)
        u0_new = gamma_b_new * gamma * self.c
        if u0_new > u0_orig:
            return (True, f"u_prime_0(0) increases with beta: orig={u0_orig:.6e}, new={u0_new:.6e}")
        else:
            return (False, f"u_prime_0(0) does not increase: orig={u0_orig:.6e}, new={u0_new:.6e}")

    # ------------------------------------------------------------------
    # Weak answer sanity check (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_positive(self) -> Tuple[bool, str]:
        """Weak sanity: all results are finite and B > 0."""
        results_list = [self.B, self.u_prime_0, self.du_prime_0_dtau, self.dE_dtau]
        for idx, res in enumerate(results_list):
            if res is None:
                return (False, f"Result {idx} is None")
            # Convert to float if possible (might be sympy expression)
            try:
                val = float(res)
            except (TypeError, ValueError):
                # If it's symbolic, we can't evaluate numeric here; skip for this weak test
                continue
            if not math.isfinite(val):
                return (False, f"Result {idx} is not finite (value: {val})")
        # Check B positive
        if self.B is not None:
            try:
                B_val = float(self.B)
                if B_val <= 0:
                    return (False, f"B is not positive: {B_val}")
            except:
                pass
        return (True, "All results finite and B positive (where evaluable)")