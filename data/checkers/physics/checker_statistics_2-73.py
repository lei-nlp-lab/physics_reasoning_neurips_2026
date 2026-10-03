import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution for computing the blackbody temperature required to generate
    a quasi‑monochromatic beam equivalent to a He‑Ne laser.
    """

    # Standard physical constants (used as fallback if not provided)
    _STANDARD_H = 6.62607015e-34    # J s
    _STANDARD_C = 299792458.0       # m/s
    _STANDARD_K = 1.380649e-23      # J/K

    def __init__(self, solution_instance: Any):
        """
        Extract all necessary inputs and results from the solution_instance.
        Use None for missing values – no numeric defaults.
        """
        # Store solution_instance for later extraction
        self.solution_instance = solution_instance

        # 1. Results dictionary (must contain 'T')
        self.results: Dict = getattr(solution_instance, 'results', None) or {}
        if not isinstance(self.results, dict):
            self.results = {}

        # 2. Givens dictionary
        self.givens: Dict = getattr(solution_instance, 'givens', None) or {}
        if not isinstance(self.givens, dict):
            self.givens = {}

        # 3. Explanation steps
        self.explanation_steps: List[str] = getattr(solution_instance, 'explanation_steps', None) or []

        # 4. Pseudocode
        self.pseudocode: str = getattr(solution_instance, 'pseudocode', None) or ""

        # 5. Extract constants (prefer solution_instance, then givens)
        self.h = self._extract('h')
        self.c = self._extract('c')
        self.k = self._extract('k')

        # 6. Extract problem parameters
        self.lambda_ = self._extract('lambda')
        self.delta_lambda = self._extract('delta_lambda')
        self.P = self._extract('P')
        self.A = self._extract('A')
        self.theta = self._extract('theta')

        # Derived solid angle
        self.Omega = None
        if self.theta is not None:
            self.Omega = math.pi * (self.theta ** 2)

        # 7. Result T
        self.T = self.results.get('T', None)

        # 8. Pre‑compute intermediate quantities for efficiency
        self._ln_arg = None
        self._wien_x = None
        if self._have_all_givens() and self.T is not None and self.T > 0:
            self._compute_intermediates()

    def _extract(self, name: str) -> Any:
        """Safely extract a single attribute, first from solution_instance, then from givens."""
        # try direct attribute on solution_instance
        val = getattr(self.solution_instance, name, None)
        if val is not None:
            return val
        # try key in givens dict
        val = self.givens.get(name, None)
        if val is not None:
            return val
        return None

    def _have_all_givens(self) -> bool:
        """Check that all required physical quantities are available."""
        needed = [self.h, self.c, self.k, self.lambda_, self.delta_lambda,
                  self.P, self.A, self.theta]
        return all(x is not None for x in needed)

    def _compute_intermediates(self) -> None:
        """Compute ln argument and Wien exponent needed by multiple tests."""
        numerator = 2 * self.h * self.c**2 * self.A * self.Omega * self.delta_lambda
        denominator = self.P * (self.lambda_ ** 5)
        if denominator > 0:
            self._ln_arg = numerator / denominator
        else:
            self._ln_arg = None

        x = (self.h * self.c) / (self.lambda_ * self.k * self.T)
        self._wien_x = x

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that essential keys exist in results and givens."""
        missing = []
        if self.T is None:
            missing.append("result['T']")
        if self.lambda_ is None:
            missing.append("given 'lambda'")
        if self.delta_lambda is None:
            missing.append("given 'delta_lambda'")
        if self.P is None:
            missing.append("given 'P' (power)")
        if self.A is None:
            missing.append("given 'A' (area)")
        if self.theta is None:
            missing.append("given 'theta' (divergence)")
        if self.h is None or self.c is None or self.k is None:
            missing.append("constants h, c, or k")
        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "All required keys present."

    def test_interface_results_types(self) -> Tuple[bool, str]:
        """Check that T is numeric and finite."""
        if self.T is None:
            return False, "T is missing, cannot check type."
        if not isinstance(self.T, (int, float)):
            return False, f"T has type {type(self.T).__name__}, expected numeric."
        if math.isnan(self.T) or math.isinf(self.T):
            return False, f"T is {self.T}, not finite."
        return True, "T is numeric and finite."

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_power_equation(self) -> Tuple[bool, str]:
        """
        Check that the given laser power matches the blackbody emission
        at the computed T using the exact Planck law:
        P = A * Omega * B_lambda(T) * Delta_lambda
        """
        if not self._have_all_givens():
            return False, "SKIP: missing givens for power equation."
        if self.T is None:
            return False, "SKIP: T is missing."

        h = self.h; c = self.c; k = self.k; lam = self.lambda_; T = self.T
        exponent = (h * c) / (lam * k * T)
        if exponent > 700:   # avoid overflow; use Wien approximation
            B_lambda = (2*h*c**2 / lam**5) * math.exp(-exponent)
        else:
            try:
                B_lambda = (2*h*c**2 / lam**5) * 1.0 / (math.exp(exponent) - 1)
            except (OverflowError, ZeroDivisionError):
                return False, "Numerical error in Planck calculation."

        computed_power = self.A * self.Omega * B_lambda * self.delta_lambda
        if computed_power <= 0:
            return False, f"Computed power is {computed_power}, not positive."

        if not math.isclose(computed_power, self.P, rel_tol=1e-2, abs_tol=1e-10):
            return False, (f"Power mismatch: given P={self.P:.3e} W, "
                           f"computed P={computed_power:.3e} W "
                           f"(rel diff {abs(computed_power-self.P)/self.P:.2%})")
        return True, "Power equation satisfied within 1% tolerance."

    def test_invariant_planck_equation(self) -> Tuple[bool, str]:
        """
        Check that the quantity Q = (P * lambda^5) / (A * Omega * Delta_lambda)
        equals exactly 2 h c^2 / (exp(hc/(lambda k T)) - 1).
        This is the fundamental Planck constraint.
        """
        if not self._have_all_givens():
            return False, "SKIP: missing givens for Planck equation."
        if self.T is None:
            return False, "SKIP: T is missing."

        # Compute left-hand side Q
        denominator_lhs = self.A * self.Omega * self.delta_lambda
        if denominator_lhs <= 0:
            return False, f"Denominator (A*Omega*Δλ) = {denominator_lhs}, must be > 0."
        Q = (self.P * self.lambda_**5) / denominator_lhs

        # Compute right-hand side (using exact Planck)
        h = self.h; c = self.c; k = self.k; lam = self.lambda_; T = self.T
        exponent = (h * c) / (lam * k * T)
        if exponent > 700:
            rhs = (2*h*c**2) * math.exp(-exponent)
        else:
            try:
                rhs = (2*h*c**2) / (math.exp(exponent) - 1)
            except (OverflowError, ZeroDivisionError):
                return False, "Numerical error in Planck RHS."

        if not math.isclose(Q, rhs, rel_tol=1e-2, abs_tol=1e-10):
            return False, (f"Planck equation mismatch: LHS={Q:.3e}, RHS={rhs:.3e} "
                           f"(rel diff {abs(Q-rhs)/rhs:.2%})")
        return True, "Planck equation satisfied within 1% tolerance."

    # ----------------------------------------------------------------------
    # Modeling & trace tests
    # ----------------------------------------------------------------------
    def test_modeling_wien_approximation_validity(self) -> Tuple[bool, str]:
        """
        Check that the Wien approximation is justified:
        hc/(lambda k T) >> 1 (typically > 5).
        """
        if self._wien_x is None:
            return False, "SKIP: cannot compute Wien parameter."
        if self._wien_x <= 5:
            return False, (f"Wien parameter x = hc/(λkT) = {self._wien_x:.3f} "
                           f"(needs > 5 for valid approximation).")
        return True, f"Wien approximation valid (x = {self._wien_x:.3f} > 5)."

    def test_modeling_ln_argument_greater_than_one(self) -> Tuple[bool, str]:
        """
        The argument of the logarithm in the Wien‑based formula must be > 1.
        argument = (2 h c^2 A Omega Δλ) / (P λ^5)
        """
        if self._ln_arg is None:
            return False, "SKIP: cannot compute ln argument."
        if self._ln_arg <= 1:
            return False, f"Ln argument = {self._ln_arg:.3e}, must be > 1."
        return True, f"Ln argument = {self._ln_arg:.3e} > 1."

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_T_with_Omega(self) -> Tuple[bool, str]:
        """
        If the divergence angle θ increases (hence Ω increases), the required
        blackbody temperature should decrease.
        Recomputes T using the Wien formula and compares for a 10% increase in Ω.
        """
        if not self._have_all_givens():
            return False, "SKIP: missing givens for monotonicity test."
        if self.T is None:
            return False, "SKIP: T is missing."

        # Baseline T computed from given Ω
        T0 = self._compute_T_from_givens(Omega_override=None)
        if T0 is None:
            return False, "Could not recompute T from original givens."

        # Increase Ω by 10%
        Omega_new = self.Omega * 1.1
        T1 = self._compute_T_from_givens(Omega_override=Omega_new)
        if T1 is None:
            return False, "Could not recompute T with increased Omega."

        if T1 > T0:
            return False, (f"Monotonicity violation: T(Ω) = {T0:.3e} K, "
                           f"T(1.1Ω) = {T1:.3e} K, expected decrease.")
        return True, (f"Monotonicity satisfied: T decreased from {T0:.3e} K "
                      f"to {T1:.3e} K when Ω increased by 10%.")

    def _compute_T_from_givens(self, Omega_override: Optional[float] = None) -> Optional[float]:
        """
        Recomputed T using the Wien‑based formula:
        T = (h c) / (λ k) / ln( (2 h c^2 A Ω Δλ) / (P λ^5) )
        Uses the instance's constants and givens, optionally overriding Ω.
        """
        h = self.h; c = self.c; k = self.k
        lam = self.lambda_; dlam = self.delta_lambda
        P = self.P; A = self.A
        omega = Omega_override if Omega_override is not None else self.Omega
        if any(x is None for x in [h, c, k, lam, dlam, P, A, omega]):
            return None

        numerator = 2 * h * c**2 * A * omega * dlam
        denominator = P * (lam**5)
        if denominator <= 0 or numerator <= 0:
            return None
        arg = numerator / denominator
        if arg <= 1:
            return None
        try:
            T = (h * c) / (lam * k) / math.log(arg)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        return T

    # ----------------------------------------------------------------------
    # Answer sanity (weak – only sign, finite, broad range)
    # ----------------------------------------------------------------------
    def test_answer_sanity_T_range(self) -> Tuple[bool, str]:
        """
        Broad‑range sanity: T must be positive, finite, and within
        1e8 K to 1e11 K (the target is ~6e9 K).
        """
        if self.T is None:
            return False, "T missing."
        if not math.isfinite(self.T):
            return False, f"T is not finite ({self.T})."
        if self.T <= 0:
            return False, f"T = {self.T} ≤ 0."
        if self.T < 1e8 or self.T > 1e11:
            return False, f"T = {self.T:.3e} K outside plausible range [1e8, 1e11]."
        return True, f"T = {self.T:.3e} K within plausible range."

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run every method whose name starts with 'test_'.
        Collect results in logs. Return (all_pass, logs).
        """
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs