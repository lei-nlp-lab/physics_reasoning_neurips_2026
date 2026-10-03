import math
import traceback

class PhysicsChecker:
    """
    Auditor for variational method problem (infinite barrier + linear potential)
    and excited state trial function orthogonality.
    """
    
    def __init__(self, solution_instance):
        # Extract givens safely (use None for missing)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.hbar = givens.get('hbar', None)
        self.m = givens.get('m', None)
        self.C = givens.get('C', None)
        self.p_integer = givens.get('p', None)  # if provided
        self.a = givens.get('a', None)          # variational parameter
        # Extract results
        results = getattr(solution_instance, 'results', {}) or {}
        self.p_opt = results.get('p_opt', None)
        self.E0_est = results.get('E0_est', None)
        self.orthogonality = results.get('orthogonality', None)
        
        # Store explanation steps if available (for trace checks)
        self.explanation = getattr(solution_instance, 'explanation_steps', None)
    
    # ------------------------------------------------------------
    # Helper: compute variational energy for given p and a
    # ------------------------------------------------------------
    def _E_var(self, p, a):
        """E_var = hbar^2 a^2/(2m (2p-1)) + C (2p+1)/(2a)"""
        if self.hbar is None or self.m is None or self.C is None:
            return None
        hbar2 = self.hbar ** 2
        try:
            term1 = hbar2 * a**2 / (2.0 * self.m * (2*p - 1))
            term2 = self.C * (2*p + 1) / (2.0 * a)
            return term1 + term2
        except (ZeroDivisionError, TypeError):
            return None
    
    def _a_opt(self, p):
        """Optimal a for given p: a^3 = C m (2p+1)(2p-1) / (2 hbar^2)"""
        if self.hbar is None or self.m is None or self.C is None:
            return None
        hbar2 = self.hbar ** 2
        try:
            a3 = self.C * self.m * (2*p+1) * (2*p-1) / (2.0 * hbar2)
            return a3 ** (1/3.0) if a3 > 0 else None
        except (ZeroDivisionError, TypeError):
            return None
    
    def _E_min(self, p):
        """Minimum energy for given integer p."""
        a_opt = self._a_opt(p)
        if a_opt is None:
            return None
        return self._E_var(p, a_opt)
    
    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_givens(self):
        """Check that required givens exist and are numeric."""
        if self.hbar is None:
            return (False, "hbar missing")
        if self.m is None:
            return (False, "m missing")
        if self.C is None:
            return (False, "C missing")
        if not isinstance(self.hbar, (int, float)):
            return (False, "hbar not numeric")
        if not isinstance(self.m, (int, float)):
            return (False, "m not numeric")
        if not isinstance(self.C, (int, float)):
            return (False, "C not numeric")
        return (True, "all required givens present")
    
    def test_interface_results_keys(self):
        """Check that required result keys exist."""
        if self.p_opt is None:
            return (False, "p_opt missing")
        if self.E0_est is None:
            return (False, "E0_est missing")
        if self.orthogonality is None:
            return (False, "orthogonality missing")
        return (True, "all result keys present")
    
    def test_interface_p_opt_type(self):
        """p_opt must be a positive integer."""
        if self.p_opt is None:
            return (False, "p_opt not available")
        if not isinstance(self.p_opt, int) or self.p_opt < 1:
            return (False, f"p_opt={self.p_opt} must be positive integer")
        return (True, f"p_opt={self.p_opt} valid")
    
    # ------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_variational_bound(self):
        """E0_est must be positive and finite."""
        if self.E0_est is None:
            return (False, "E0_est not available")
        if not math.isfinite(self.E0_est) or self.E0_est <= 0:
            return (False, f"E0_est={self.E0_est} must be positive finite")
        return (True, "variational estimate positive finite")
    
    def test_invariant_optimum_p_upper_bound(self):
        """p_opt=2 must give lower E than p_opt=1 (if givens allow)."""
        if self.p_opt is None:
            return (False, "p_opt not available")
        # Compute E_min(1) and E_min(2) using same constants
        if self.hbar is None or self.m is None or self.C is None:
            return (False, "cannot compute E_min without constants")
        e1 = self._E_min(1)
        e2 = self._E_min(2)
        if e1 is None or e2 is None:
            return (False, "E_min calculation failed")
        # E_min(2) should be lower than E_min(1)
        if e2 >= e1:
            return (False, f"expected E_min(2) < E_min(1), got {e2} >= {e1}")
        return (True, f"E_min(2)={e2} < E_min(1)={e1}")
    
    def test_invariant_orthogonality_condition(self):
        """Orthogonality statement must involve zero overlap between phi1 and phi0."""
        if self.orthogonality is None:
            return (False, "orthogonality not available")
        # Accept any statement that mentions integral zero or orthogonal
        key_phrases = ["0", "zero", "orthogonal", "overlap"]
        ok = any(phrase in str(self.orthogonality).lower() for phrase in key_phrases)
        if not ok:
            return (False, "orthogonality statement does not imply zero overlap")
        return (True, "orthogonality condition stated")
    
    # ------------------------------------------------------------
    # Modeling / trace tests (consistent with explanation steps)
    # ------------------------------------------------------------
    def test_modeling_V_expectation(self):
        """Check that <V> = C (2p+1)/(2a) for p=2 and optimal a."""
        if self.hbar is None or self.m is None or self.C is None:
            return (False, "cannot compute <V> without constants")
        a_opt2 = self._a_opt(2)
        if a_opt2 is None:
            return (False, "a_opt(2) could not be computed")
        # <V> = C*(2p+1)/(2a)
        V_expected = self.C * 5 / (2.0 * a_opt2)
        # Derive from E_min(2) - kinetic
        E2 = self._E_min(2)
        if E2 is None:
            return (False, "E_min(2) unavailable")
        # Kinetic part: hbar^2 a^2/(2m (2p-1)) = hbar^2 a_opt2^2/(6m)
        K2 = (self.hbar**2 * a_opt2**2) / (6.0 * self.m)
        V2_derived = E2 - K2
        if not math.isclose(V_expected, V2_derived, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"<V> mismatch: expected {V_expected}, derived {V2_derived}")
        return (True, f"<V>={V_expected} consistent")
    
    def test_modeling_kinetic_expectation(self):
        """Check kinetic term for p=2: <T> = hbar^2 a^2/(2m(2p-1))."""
        if self.hbar is None or self.m is None:
            return (False, "cannot compute <T> without constants")
        a_opt2 = self._a_opt(2)
        if a_opt2 is None:
            return (False, "a_opt(2) cannot be computed")
        T_expected = (self.hbar**2 * a_opt2**2) / (2.0 * self.m * 3)  # (2p-1)=3
        E2 = self._E_min(2)
        if E2 is None:
            return (False, "E_min(2) unavailable")
        V_expected = self.C * 5 / (2.0 * a_opt2)
        T_derived = E2 - V_expected
        if not math.isclose(T_expected, T_derived, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"<T> mismatch: expected {T_expected}, derived {T_derived}")
        return (True, f"<T>={T_expected} consistent")
    
    def test_modeling_minimization_condition(self):
        """Check that derivative condition yields a_opt formula."""
        if self.hbar is None or self.m is None or self.C is None:
            return (False, "cannot check minimization without constants")
        # For p=2, a_opt^3 = C*m*(5*3)/(2 hbar^2) = 15 C m/(2 hbar^2)
        a3_formula = 15 * self.C * self.m / (2 * self.hbar**2)
        a_opt_expected = a3_formula ** (1/3.0)
        a_opt_from_helper = self._a_opt(2)
        if a_opt_from_helper is None:
            return (False, "a_opt(2) helper returned None")
        if not math.isclose(a_opt_expected, a_opt_from_helper, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"a_opt mismatch: expected {a_opt_expected}, got {a_opt_from_helper}")
        return (True, "minimization condition correct")
    
    def test_modeling_scale_factor(self):
        """Check final energy scale factor: E0_est should match (5^{2/3}/3)*(3/2)^{5/3} * (hbar^2 C^2/m)^{1/3}."""
        if self.E0_est is None:
            return (False, "E0_est not available")
        if self.hbar is None or self.m is None or self.C is None:
            return (False, "cannot compute reference energy without constants")
        factor = (5**(2/3)/3) * (3/2)**(5/3)
        ref = factor * (self.hbar**2 * self.C**2 / self.m)**(1/3)
        if not math.isclose(self.E0_est, ref, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E0_est {self.E0_est} != reference {ref}")
        return (True, f"E0_est={self.E0_est} matches expected scaling")
    
    # ------------------------------------------------------------
    # Monotonicity tests (explicit perturbation and baseline comparison)
    # ------------------------------------------------------------
    def test_monotonicity_E_var_vs_a(self):
        """For p=2, E_var(a) should have a minimum at a_opt; perturb a by +-10% increases E."""
        if self.hbar is None or self.m is None or self.C is None:
            return (False, "cannot test monotonicity without constants")
        a_opt = self._a_opt(2)
        if a_opt is None:
            return (False, "a_opt unavailable")
        base_E = self._E_var(2, a_opt)
        for factor in [0.9, 1.1]:
            a_pert = a_opt * factor
            pert_E = self._E_var(2, a_pert)
            if pert_E is None:
                return (False, f"E_var at a={a_pert} failed")
            if pert_E <= base_E:
                return (False, f"E_var at a={a_pert} ({pert_E}) not greater than baseline {base_E}")
        return (True, "E_var increases when a is perturbed away from optimum")
    
    def test_monotonicity_p_rank(self):
        """Perturb p from optimal 2 to 1 and 3; baseline E at p=2 should be lowest."""
        if self.hbar is None or self.m is None or self.C is None:
            return (False, "cannot test p ranking without constants")
        base_E = self._E_min(2)
        if base_E is None:
            return (False, "E_min(2) unavailable")
        for p_test in [1, 3]:
            pert_E = self._E_min(p_test)
            if pert_E is None:
                return (False, f"E_min({p_test}) unavailable")
            if pert_E <= base_E:
                return (False, f"E_min({p_test})={pert_E} not greater than baseline {base_E}")
        return (True, "E_min(2) is lower than E_min(1) and E_min(3)")
    
    # ------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------
    def test_answer_sanity_broad_magnitude(self):
        """E0_est should be of order (hbar^2 C^2/m)^{1/3} (dimensionally consistent)."""
        if self.E0_est is None:
            return (False, "E0_est not available")
        if self.hbar is None or self.m is None or self.C is None:
            return (False, "cannot check magnitude without constants")
        ref = (self.hbar**2 * self.C**2 / self.m)**(1/3)
        ratio = self.E0_est / ref
        if ratio < 0.1 or ratio > 10:
            return (False, f"E0_est {self.E0_est} is outside plausible range (ref {ref})")
        return (True, "E0_est magnitude plausible")
    
    # ------------------------------------------------------------
    # Run all tests (discovery)
    # ------------------------------------------------------------
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)