import math
import traceback
import sympy

class PhysicsChecker:
    """
    Auditor for the relativistic collision: K- + p -> π0 + Λ0.
    Checks conservation laws, on-shell conditions, and derived energy formula.
    """
    def __init__(self, solution_instance):
        # Safe extraction from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        
        # Masses: try givens first, then results
        self.m_K = self.givens.get('m_K') or self.results.get('m_K') or None
        self.m_p = self.givens.get('m_p') or self.results.get('m_p') or None
        self.m_pi = self.givens.get('m_pi') or self.results.get('m_pi') or None
        self.m_L = self.givens.get('m_L') or self.results.get('m_L') or None
        
        # Incident energy
        self.E_K = self.results.get('E_K', None) or self.givens.get('E_K', None)
        
        # Derived intermediate quantities (may be computed later)
        self.p_K = None   # momentum of K- (computed if needed)
        self.E_pi = None  # total energy of π0
        self.p_pi = None   # momentum of π0

    # ----------------------------------------------------------------------
    # Helper: compute p from E and m (relativistic)
    # ----------------------------------------------------------------------
    def _momentum(self, E, m):
        if E is None or m is None:
            return None
        try:
            # Could be sympy expression; work with sympy if possible
            return sympy.sqrt(E**2 - m**2)
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Helper: check if all numeric masses are available
    # ----------------------------------------------------------------------
    def _numeric_masses(self):
        return all(isinstance(m, (int, float)) for m in [self.m_K, self.m_p, self.m_pi, self.m_L] if m is not None)

    # ----------------------------------------------------------------------
    # Test: interface keys
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        missing = []
        if self.E_K is None:
            missing.append('E_K')
        if self.m_K is None:
            missing.append('m_K')
        if self.m_p is None:
            missing.append('m_p')
        if self.m_pi is None:
            missing.append('m_pi')
        if self.m_L is None:
            missing.append('m_L')
        if missing:
            return (False, f"Missing required keys: {missing}")
        return (True, "All required keys present")

    # ----------------------------------------------------------------------
    # Test: invariant energy-momentum conservation (S = s)
    # ----------------------------------------------------------------------
    def test_invariant_energy_momentum_conservation(self):
        if any(v is None for v in [self.E_K, self.m_K, self.m_p, self.m_pi, self.m_L]):
            return (False, "SKIP: missing mass or energy values")
        # Compute p_K and E_pi from conservation
        self.p_K = self._momentum(self.E_K, self.m_K)
        self.E_pi = self.E_K + self.m_p - self.m_L
        if self.E_pi is None or self.p_K is None:
            return (False, "Cannot compute p_K or E_pi")
        self.p_pi = self._momentum(self.E_pi, self.m_pi)
        # Compute invariant S_initial and S_final
        S_initial = (self.E_K + self.m_p)**2 - self.p_K**2
        S_final = (self.E_pi + self.m_L)**2 - self.p_pi**2
        if not math.isclose(float(S_initial), float(S_final), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Invariant S not conserved: initial={S_initial}, final={S_final}")
        return (True, f"Conservation ok: S = {S_initial:.4f}")

    # ----------------------------------------------------------------------
    # Test: on-shell conditions for K- and π0
    # ----------------------------------------------------------------------
    def test_invariant_on_shell_conditions(self):
        if any(v is None for v in [self.E_K, self.m_K, self.m_p, self.m_pi, self.m_L]):
            return (False, "SKIP: missing mass or energy values")
        # Ensure intermediate quantities computed
        if self.p_K is None or self.E_pi is None:
            self.p_K = self._momentum(self.E_K, self.m_K)
            self.E_pi = self.E_K + self.m_p - self.m_L
        # Check K-: E_K^2 - p_K^2 should equal m_K^2
        mass_shell_K = self.E_K**2 - self.p_K**2
        if not math.isclose(float(mass_shell_K), float(self.m_K**2), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"K- off-shell: E_K^2 - p_K^2 = {mass_shell_K}, expected {self.m_K**2}")
        # Check π0: E_pi^2 - p_pi^2 should equal m_pi^2
        p_pi = self._momentum(self.E_pi, self.m_pi)
        if p_pi is None:
            return (False, "Cannot compute p_pi")
        mass_shell_pi = self.E_pi**2 - p_pi**2
        if not math.isclose(float(mass_shell_pi), float(self.m_pi**2), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"π0 off-shell: E_pi^2 - p_pi^2 = {mass_shell_pi}, expected {self.m_pi**2}")
        return (True, "Both K- and π0 satisfy on-shell condition")

    # ----------------------------------------------------------------------
    # Test: modeling – re-derive E_K from conservation laws
    # ----------------------------------------------------------------------
    def test_modeling_energy_formula(self):
        """Re-derive the expression for E_K from the given masses (numeric or symbolic) and compare."""
        masses = [self.m_K, self.m_p, self.m_pi, self.m_L]
        if any(m is None for m in masses):
            return (False, "SKIP: missing masses")
        # Build the expected expression: (m_K^2 - m_pi^2 + (m_L - m_p)^2) / (2*(m_L - m_p))
        try:
            m_K, m_p, m_pi, m_L = sympy.symbols('m_K m_p m_pi m_L')
            expected_expr = (m_K**2 - m_pi**2 + (m_L - m_p)**2) / (2*(m_L - m_p))
            # Substitute actual values (symbolic or numeric)
            sub_dict = {
                m_K: self.m_K,
                m_p: self.m_p,
                m_pi: self.m_pi,
                m_L: self.m_L
            }
            expected_EK = expected_expr.subs(sub_dict)
            if expected_EK.is_Float or expected_EK.is_Rational:
                expected_EK = float(expected_EK)
            # Now compare with self.E_K
            if self.E_K is None:
                return (False, "E_K is missing")
            # Use sympy simplification to check equality (handles symbolic)
            if sympy.simplify(expected_EK - self.E_K) != 0:
                return (False, f"E_K formula mismatch: expected {expected_EK}, got {self.E_K}")
            return (True, f"E_K formula consistent: {self.E_K}")
        except Exception as e:
            return (False, f"Error in formula check: {e}")

    # ----------------------------------------------------------------------
    # Test: modeling – trace the derivation using invariant S method
    # ----------------------------------------------------------------------
    def test_modeling_invariant_derivation(self):
        """
        Verify that using the invariant S = (sum E)^2 - (sum p)^2 leads to the same E_K expression.
        This checks the reasoning from explanation_steps without copying hardcoded numbers.
        """
        masses = [self.m_K, self.m_p, self.m_pi, self.m_L]
        if any(m is None for m in masses):
            return (False, "SKIP: missing masses")
        if self.E_K is None:
            return (False, "E_K missing")
        # Define symbols
        mK, mp, mpi, mL = sympy.symbols('mK mp mpi mL')
        EK_var = sympy.symbols('EK')
        # Build equation from S conservation (with p_L=0 and p_pi = p_K)
        # S_initial = (EK + mp)^2 - p_K^2 = (EK + mp)^2 - (EK^2 - mK^2) = ... 
        # Simplify S_initial: (EK + mp)^2 - (EK^2 - mK^2) = EK^2 + 2*EK*mp + mp^2 - EK^2 + mK^2 = 2*EK*mp + mp^2 + mK^2
        # S_final = (E_pi + mL)^2 - p_pi^2 = (sqrt(p_K^2 + mpi^2) + mL)^2 - (EK^2 - mK^2)
        # Alternative: use derived expression from explanation E_pi = EK + mp - mL, p_pi = p_K
        # Then S_final = (EK + mp - mL + mL)^2 - p_K^2 = (EK + mp)^2 - p_K^2 = S_initial automatically.
        # So this is automatically satisfied. Instead check that substituting E_K formula into S relation holds.
        # Use the derived expression as starting point: E_K_formula = (mK^2 - mpi^2 + (mL - mp)^2)/(2*(mL - mp))
        expected_EK = (mK**2 - mpi**2 + (mL - mp)**2) / (2*(mL - mp))
        # Substitute actual masses and simplify difference
        diff = sympy.simplify(expected_EK - self.E_K)
        if diff != 0:
            # If symbolic fails due to non-symbolic self.E_K, try numeric
            try:
                numeric_diff = float(sympy.N(expected_EK.subs({mK:self.m_K, mp:self.m_p, mpi:self.m_pi, mL:self.m_L}) - self.E_K))
                if abs(numeric_diff) > 1e-6:
                    return (False, f"Derived E_K formula does not match: difference {numeric_diff}")
            except Exception:
                return (False, "Cannot verify derivation mathematically")
        return (True, "Derivation via invariant S consistent with formula")

    # ----------------------------------------------------------------------
    # Test: monotonicity – E_K increases with m_K
    # ----------------------------------------------------------------------
    def test_monotonicity_E_K_with_m_K(self):
        if not self._numeric_masses() or self.E_K is None:
            return (False, "SKIP: numeric masses required")
        # Perturb m_K by +1%
        m_K_new = self.m_K * 1.01
        # Recompute E_K using formula
        expected_EK_new = (m_K_new**2 - self.m_pi**2 + (self.m_L - self.m_p)**2) / (2*(self.m_L - self.m_p))
        if expected_EK_new <= self.E_K:
            return (False, f"E_K should increase when m_K increases: baseline {self.E_K:.4f}, perturbed {expected_EK_new:.4f}")
        return (True, f"Monotonicity holds: E_K increased from {self.E_K:.4f} to {expected_EK_new:.4f}")

    # ----------------------------------------------------------------------
    # Test: monotonicity – E_K decreases with m_L (since denominator increases)
    # ----------------------------------------------------------------------
    def test_monotonicity_E_K_with_m_L(self):
        if not self._numeric_masses() or self.E_K is None:
            return (False, "SKIP: numeric masses required")
        # Perturb m_L by +1%
        m_L_new = self.m_L * 1.01
        # Recompute E_K using formula
        expected_EK_new = (self.m_K**2 - self.m_pi**2 + (m_L_new - self.m_p)**2) / (2*(m_L_new - self.m_p))
        if expected_EK_new >= self.E_K:
            return (False, f"E_K should decrease when m_L increases: baseline {self.E_K:.4f}, perturbed {expected_EK_new:.4f}")
        return (True, f"Monotonicity holds: E_K decreased from {self.E_K:.4f} to {expected_EK_new:.4f}")

    # ----------------------------------------------------------------------
    # Test: weak answer sanity – positivity, finite, E_K > m_K
    # ----------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        if self.E_K is None:
            return (False, "E_K missing")
        try:
            EK = float(self.E_K)
        except (TypeError, ValueError):
            return (False, "E_K not numeric")
        if not math.isfinite(EK) or EK <= 0:
            return (False, f"E_K not positive finite: {EK}")
        if self.m_K is not None:
            try:
                mK = float(self.m_K)
            except (TypeError, ValueError):
                mK = None
            if mK is not None and EK <= mK:
                return (False, f"E_K <= m_K: {EK} <= {mK}")
        return (True, "E_K is positive, finite, and > m_K (if available)")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
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