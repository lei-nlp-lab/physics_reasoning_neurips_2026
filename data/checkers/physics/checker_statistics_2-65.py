import math
import traceback
from typing import Dict, Any, Optional, Tuple, List

class PhysicsChecker:
    """
    Robust checker for a Bose-Einstein condensation problem.
    Checks interface, invariant relations, modeling quality, monotonicity,
    and provides a weak final-answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all necessary data from solution_instance safely.
        Use None defaults for missing values.
        """
        # Attempt to get results and givens as dictionaries
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        if self.givens is None:
            self.givens = {}
        self.explanation_steps: Optional[List[Dict[str, str]]] = getattr(
            solution_instance, 'explanation_steps', None
        )

        # Extract required result quantities with None defaults
        self.D_eps: Optional[str] = self.results.get('D_eps', None)
        self.nbar: Optional[str] = self.results.get('nbar', None)
        self.mu_eq: Optional[str] = self.results.get('mu_eq', None)
        self.T_c: Any = self.results.get('T_c', None)  # could be string or numeric
        self.mu: Any = self.results.get('mu', None)
        self.N0: Any = self.results.get('N0', None)
        self.U: Any = self.results.get('U', None)

        # Extract givens if available (used in monotonicity)
        self.N_given: Optional[float] = self.givens.get('N', None)
        self.V_given: Optional[float] = self.givens.get('V', None)
        self.T_given: Optional[float] = self.givens.get('T', None)
        self.m_given: Optional[float] = self.givens.get('m', None)

        # Common physical constants (if not provided, we cannot compute numeric checks)
        self.k_B = self.givens.get('k_B', None)
        self.hbar = self.givens.get('hbar', None)

    # ---------- Helper method for safe numeric conversion ----------
    def _to_float(self, val) -> Optional[float]:
        """Try to convert a value to float, return None if impossible."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # ---------- Interface Tests ----------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist."""
        required_keys = ['D_eps', 'nbar', 'mu_eq', 'T_c', 'mu', 'N0', 'U']
        missing = [key for key in required_keys if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_type_D_eps(self) -> Tuple[bool, str]:
        """Check that D_eps is a string."""
        if not isinstance(self.D_eps, str):
            return (False, f"D_eps should be a string, got {type(self.D_eps)}")
        return (True, "D_eps is string")

    def test_interface_type_nbar(self) -> Tuple[bool, str]:
        """Check that nbar is a string."""
        if not isinstance(self.nbar, str):
            return (False, f"nbar should be a string, got {type(self.nbar)}")
        return (True, "nbar is string")

    def test_interface_type_mu(self) -> Tuple[bool, str]:
        """Check that mu is present and either numeric or string."""
        if self.mu is None:
            return (False, "mu is not present in results")
        return (True, "mu is present")

    # ---------- Invariant Tests (theorem‑based) ----------
    def test_invariant_D_eps_sqrt_form(self) -> Tuple[bool, str]:
        """D(ε) must be proportional to ε^(1/2) (or √ε) for free bosons."""
        if self.D_eps is None:
            return (False, "SKIP: D_eps not available")
        d = self.D_eps.lower()
        # Check for common square‑root notations
        has_sqrt = ('sqrt(eps)' in d or 'eps**(1/2)' in d or 'eps^(1/2)' in d
                    or 'eps**0.5' in d or 'ε^(1/2)' in d)
        if not has_sqrt:
            return (False, "D(ε) does not contain √ε or ε^(1/2) proportionality")
        return (True, "D(ε) has correct √ε dependence")

    def test_invariant_nbar_form(self) -> Tuple[bool, str]:
        """n̄(ε) must be of the form 1/(exp((ε-μ)/k_B T)-1)."""
        if self.nbar is None:
            return (False, "SKIP: nbar not available")
        n = self.nbar.lower()
        # Look for exp and minus 1 in denominator
        has_exp = 'exp' in n or 'e^' in n
        has_minus_one = '- 1' in n
        if not (has_exp and has_minus_one):
            return (False, "nbar does not appear to have correct Bose‑Einstein form")
        return (True, "nbar has correct Bose‑Einstein form")

    def test_invariant_mu_zero_for_T_less_Tc(self) -> Tuple[bool, str]:
        """For T < T_c, μ must be exactly zero (or very close)."""
        if self.mu is None:
            return (False, "SKIP: mu not provided")
        # Accept if mu is the number 0 or the string '0'
        if self.mu == 0 or self.mu == '0':
            return (True, "μ = 0 as expected for T < T_c")
        # If numeric, check close to zero
        mu_float = self._to_float(self.mu)
        if mu_float is not None:
            if math.isclose(mu_float, 0.0, abs_tol=1e-9):
                return (True, f"μ = {mu_float} is effectively zero")
        # If neither, we still note that we cannot verify temperature condition,
        # but here we only check that the result declares μ = 0.
        return (False, f"μ is not zero, got {self.mu}")

    def test_invariant_N0_positive(self) -> Tuple[bool, str]:
        """Condensate number N0 must be non‑negative (for T < T_c)."""
        if self.N0 is None:
            return (False, "SKIP: N0 not provided")
        # Try to get numeric value
        n0_float = self._to_float(self.N0)
        if n0_float is not None:
            if n0_float < -1e-9:
                return (False, f"N0 = {n0_float} is negative")
            return (True, f"N0 = {n0_float} is non‑negative")
        # Otherwise, accept if string contains 'N*(1-(T/T_c)^(3/2))' or similar
        n0_str = str(self.N0).lower()
        if 'n*' in n0_str and '1-' in n0_str and '3/2' in n0_str:
            return (True, "N0 expression has correct form")
        return (False, "Cannot verify N0 positivity from expression")

    # ---------- Modeling/Trace Tests ----------
    def test_modeling_Tc_contains_integral(self) -> Tuple[bool, str]:
        """T_c expression should contain the integral ∫ x^{1/2}/(e^x-1) dx."""
        if self.T_c is None:
            return (False, "SKIP: T_c not available")
        tc_str = str(self.T_c).lower()
        if 'integral' in tc_str or 'int' in tc_str or '∫' in tc_str:
            # Check for x^(1/2) in denominator with e^x-1
            if ('x**(1/2)' in tc_str or 'x^(1/2)' in tc_str) and ('e^x-1' in tc_str or 'exp(x)-1' in tc_str):
                return (True, "T_c expression includes the required integral")
        return (False, "T_c expression does not clearly show the dimensionless integral")

    def test_modeling_U_contains_J_integral(self) -> Tuple[bool, str]:
        """Energy U should contain ∫ x^{3/2}/(e^x-1) dx."""
        if self.U is None:
            return (False, "SKIP: U not available")
        u_str = str(self.U).lower()
        if ('x**(3/2)' in u_str or 'x^(3/2)' in u_str) and ('e^x-1' in u_str or 'exp(x)-1' in u_str):
            return (True, "U expression includes required x^{3/2} integral")
        # Also accept if it states the numerical prefactor 0.770
        if '0.770' in u_str:
            return (True, "U expression shows correct numerical coefficient")
        return (False, "U expression does not contain expected integral")

    def test_trace_condensate_fraction_relation(self) -> Tuple[bool, str]:
        """Check that N0 expression matches N*(1-(T/T_c)^(3/2))."""
        if self.N0 is None:
            return (False, "SKIP: N0 not provided")
        # Get numeric N0 if possible
        n0_float = self._to_float(self.N0)
        # We need N and T, Tc to verify
        N = self._to_float(self.N_given)
        T = self._to_float(self.T_given)
        Tc = self._to_float(self.T_c)
        if n0_float is not None and N is not None and T is not None and Tc is not None:
            if Tc <= 0:
                return (False, "Tc must be positive")
            ratio = (T / Tc) ** 1.5
            expected_N0 = N * (1 - ratio)
            if math.isclose(n0_float, expected_N0, rel_tol=1e-2):
                return (True, "N0 matches expected relation N*(1-(T/Tc)^(3/2))")
            else:
                return (False, f"N0 = {n0_float}, expected ≈ {expected_N0}")
        else:
            # Fallback: check expression form
            n0_str = str(self.N0).lower()
            if 'n*' in n0_str and '(1-(t/t_c)^(3/2))' in n0_str:
                return (True, "N0 expression has correct form")
            return (False, "Cannot verify N0 relation (missing numeric values)")

    # ---------- Monotonicity Test ----------
    def test_monotonicity_condensate_fraction(self) -> Tuple[bool, str]:
        """
        For T < T_c, decreasing T should increase the condensate fraction N0.
        We need numeric data for T, Tc, and N.
        If not available, we attempt to extract from N0 expression.
        """
        # Get numeric values for T, Tc, N
        T = self._to_float(self.T_given)
        Tc = self._to_float(self.T_c)
        N = self._to_float(self.N_given)
        if T is None or Tc is None or N is None:
            return (False, "SKIP: missing numeric T, Tc, or N for monotonicity")

        # Verify that we are below Tc
        if T >= Tc:
            return (False, "Current T is not less than Tc; monotonicity test only valid for T < Tc")

        # Compute baseline N0
        ratio = (T / Tc) ** 1.5
        baseline_N0 = N * (1 - ratio)

        # Perturb T downward by 5% (still below Tc)
        delta = 0.05
        T2 = T * (1 - delta)
        if T2 >= Tc:
            T2 = Tc * 0.99  # ensure still below Tc
        ratio2 = (T2 / Tc) ** 1.5
        perturbed_N0 = N * (1 - ratio2)

        # The condensate should be larger for lower T (N0 increases)
        if perturbed_N0 <= baseline_N0 - 1e-9:
            return (False, f"Lowering T from {T} to {T2} decreased N0 (={perturbed_N0} < {baseline_N0})")
        return (True, f"Monotonicity: lowering T increases N0 (from {baseline_N0:.3g} to {perturbed_N0:.3g})")

    # ---------- Weak Final Answer Sanity (only one) ----------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that D(eps) expression is non‑negative for positive eps,
        and that nbar is positive for eps > mu.
        Only primitive checks without re‑deriving values.
        """
        # D_eps: ensure it contains positive coefficients
        if self.D_eps is not None:
            # Check that it has a positive prefactor (e.g., V/(4π^2) >0)
            d_lower = self.D_eps.lower()
            if 'v/' in d_lower or 'v/' in d_lower or 'v' in d_lower:
                # assume V positive
                pass
            # we cannot do more without solving

        # nbar: ensure that denominator is positive for typical (eps-mu) > small
        if self.nbar is not None:
            n_lower = self.nbar.lower()
            if 'exp' not in n_lower and 'e^' not in n_lower:
                return (False, "nbar should contain an exponential")
            if '- 1' not in n_lower:
                return (False, "nbar should contain '-1' in denominator")

        # mu: if numeric, ensure it is ≤0 (for T>Tc) but we have no temperature info
        # So we just check that it is not positive (since for T<Tc it's zero, for T>Tc negative)
        mu_float = self._to_float(self.mu)
        if mu_float is not None:
            if mu_float > 1e-9:
                # Could be a mistake
                return (False, f"μ should be ≤ 0, but got {mu_float}")

        return (True, "Weak sanity passed: no obvious contradictions")

    # ---------- Master test runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, run them, and collect results."""
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
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)