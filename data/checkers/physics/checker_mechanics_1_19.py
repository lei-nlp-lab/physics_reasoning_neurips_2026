import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction with None defaults (no numeric defaults)
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        
        # Extract key quantities from results or givens
        self.L = self.results.get('L', None)
        self.m = self.givens.get('m', None) or self.results.get('m', None)
        self.lam = self.givens.get('lambda', None) or self.givens.get('lam', None) or self.results.get('lambda', None)
        self.J0 = self.givens.get('J0', None) or self.results.get('J0', None)
        
        # Also try direct attribute access as fallback
        if self.m is None:
            self.m = getattr(solution_instance, 'm', None)
        if self.lam is None:
            self.lam = getattr(solution_instance, 'lambda', None)
        if self.J0 is None:
            self.J0 = getattr(solution_instance, 'J0', None)
        
        # Define symbol for time
        self.t = sp.Symbol('t', positive=True)
        
    # ----------------------------------------------------------------------
    # Interface Tests
    # ----------------------------------------------------------------------
    def test_interface_L_present(self):
        """Check that angular momentum L is present in results."""
        if self.L is None:
            return (False, "SKIP: 'L' not found in results")
        return (True, "L is present")
    
    def test_interface_givens_present(self):
        """Check that essential givens m, lambda, J0 are available."""
        missing = []
        if self.m is None:
            missing.append('m')
        if self.lam is None:
            missing.append('lambda')
        if self.J0 is None:
            missing.append('J0')
        if missing:
            return (False, f"Missing givens: {missing} (tests requiring them will be skipped)")
        return (True, "All givens present")
    
    # ----------------------------------------------------------------------
    # Invariant Tests (physical law checks, symbolic when possible)
    # ----------------------------------------------------------------------
    def test_invariant_ode_satisfied(self):
        """Check that L(t) satisfies dL/dt = -(lambda/m) L."""
        if self.L is None or self.m is None or self.lam is None:
            return (False, "SKIP: missing L, m, or lambda")
        # Try to derive dL/dt symbolically
        try:
            L_expr = sp.sympify(self.L)
            dL_dt = sp.diff(L_expr, self.t)
            lhs = dL_dt
            rhs = - (self.lam / self.m) * L_expr
            diff = sp.simplify(lhs - rhs)
            if diff == 0:
                return (True, "dL/dt == -(lambda/m) L holds symbolically")
            else:
                # Fallback: numeric check at a sample point using isclose
                m_val = 1.0
                lam_val = 0.5
                J0_val = 2.0
                t_val = 1.0
                L_num = float(L_expr.subs({self.t: t_val, self.m: m_val, self.lam: lam_val, 'J0': J0_val}))
                rhs_num = - (lam_val / m_val) * L_num
                dt = 1e-6
                t1 = t_val + dt
                L1 = float(L_expr.subs({self.t: t1, self.m: m_val, self.lam: lam_val, 'J0': J0_val}))
                numeric_dLdt = (L1 - L_num) / dt
                if math.isclose(numeric_dLdt, rhs_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "dL/dt == -(lambda/m) L holds numerically (symbolic derivation failed)")
                else:
                    return (False, f"ODE not satisfied: symbolic difference = {diff}, numeric mismatch")
        except Exception as e:
            return (False, f"ODE test crashed: {e}")
    
    def test_invariant_torque_equation_sign(self):
        """Check that the torque is dissipative (sign of derivative matches)."""
        if self.L is None or self.m is None or self.lam is None:
            return (False, "SKIP: missing L, m, or lambda")
        try:
            L_expr = sp.sympify(self.L)
            dL_dt = sp.diff(L_expr, self.t)
            m_val = 1.0
            lam_val = 0.5
            J0_val = 1.0
            t_val = 0.1
            L_val = float(L_expr.subs({self.t: t_val, self.m: m_val, self.lam: lam_val, 'J0': J0_val}))
            dL_val = float(dL_dt.subs({self.t: t_val, self.m: m_val, self.lam: lam_val, 'J0': J0_val}))
            if L_val > 0 and dL_val < 0:
                return (True, "Torque is dissipative: dL/dt negative when L positive")
            else:
                return (False, f"Sign check failed: L_val={L_val}, dL_val={dL_val}")
        except Exception as e:
            return (False, f"Sign test crashed: {e}")
    
    # ----------------------------------------------------------------------
    # Modeling / Trace Tests (check quality of the solution form)
    # ----------------------------------------------------------------------
    def test_modeling_exponential_form(self):
        """Check that L(t) has the form L0 * exp(- (lambda/m) t)."""
        if self.L is None or self.m is None or self.lam is None or self.J0 is None:
            return (False, "SKIP: missing L, m, lambda, or J0")
        try:
            L_expr = sp.sympify(self.L)
            expected = self.J0 * sp.exp(- (self.lam / self.m) * self.t)
            diff = sp.simplify(L_expr - expected)
            if diff == 0:
                return (True, "L(t) exactly equals J0 * exp(-lambda/m t)")
            else:
                # Numeric fallback
                m_val = 1.2
                lam_val = 0.3
                J0_val = 5.0
                t_val = 2.0
                L_num = float(L_expr.subs({self.t: t_val, self.m: m_val, self.lam: lam_val, 'J0': J0_val}))
                expected_num = J0_val * math.exp(- (lam_val / m_val) * t_val)
                if math.isclose(L_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "L(t) numerically matches expected exponential decay")
                else:
                    return (False, f"Exponential form not matched: L_num={L_num}, expected={expected_num}")
        except Exception as e:
            return (False, f"Exponential form test crashed: {e}")
    
    def test_modeling_initial_condition(self):
        """Check that L(0) = J0."""
        if self.L is None or self.J0 is None:
            return (False, "SKIP: missing L or J0")
        try:
            L_expr = sp.sympify(self.L)
            L0 = L_expr.subs(self.t, 0)
            if sp.simplify(L0 - self.J0) == 0:
                return (True, "L(0) = J0 holds symbolically")
            else:
                m_val = 1.0
                lam_val = 0.5
                J0_val = float(self.J0) if isinstance(self.J0, (int, float)) else 3.0
                L0_num = float(L_expr.subs({self.t: 0, self.m: m_val, self.lam: lam_val, 'J0': J0_val}))
                if math.isclose(L0_num, J0_val, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "L(0) = J0 holds numerically")
                else:
                    return (False, f"Initial condition not satisfied: L(0)={L0_num}, J0={J0_val}")
        except Exception as e:
            return (False, f"Initial condition test crashed: {e}")
    
    # ----------------------------------------------------------------------
    # Monotonicity Tests (explicit perturbation vs baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_decay_with_lambda(self):
        """Check that increasing lambda by a small percentage reduces L at fixed time."""
        if self.L is None or self.m is None or self.lam is None or self.J0 is None:
            return (False, "SKIP: missing L, m, lambda, or J0")
        try:
            L_expr = sp.sympify(self.L)
            # Baseline input
            m_val = 1.0
            lam_baseline = 1.0
            J0_val = 2.0
            t_val = 0.5
            # Perturb lambda by +10%
            lam_perturbed = lam_baseline * 1.10
            # Compute outputs
            L_baseline = float(L_expr.subs({self.t: t_val, self.m: m_val, self.lam: lam_baseline, 'J0': J0_val}))
            L_perturbed = float(L_expr.subs({self.t: t_val, self.m: m_val, self.lam: lam_perturbed, 'J0': J0_val}))
            if L_perturbed < L_baseline:
                return (True, f"L decreases with 10% increase in lambda: {L_perturbed} < {L_baseline}")
            else:
                return (False, f"Monotonicity violated: L(lambda+10%)={L_perturbed} >= L(original)={L_baseline}")
        except Exception as e:
            return (False, f"Monotonicity lambda test crashed: {e}")
    
    def test_monotonicity_decay_with_mass(self):
        """Check that increasing mass by a small percentage increases L (slower decay)."""
        if self.L is None or self.m is None or self.lam is None or self.J0 is None:
            return (False, "SKIP: missing L, m, lambda, or J0")
        try:
            L_expr = sp.sympify(self.L)
            lam_val = 1.0
            m_baseline = 1.0
            J0_val = 2.0
            t_val = 0.5
            # Perturb mass by +10%
            m_perturbed = m_baseline * 1.10
            L_baseline = float(L_expr.subs({self.t: t_val, self.m: m_baseline, self.lam: lam_val, 'J0': J0_val}))
            L_perturbed = float(L_expr.subs({self.t: t_val, self.m: m_perturbed, self.lam: lam_val, 'J0': J0_val}))
            if L_perturbed > L_baseline:
                return (True, f"L increases with 10% increase in mass: {L_perturbed} > {L_baseline}")
            else:
                return (False, f"Monotonicity violated: L(m+10%)={L_perturbed} <= L(original)={L_baseline}")
        except Exception as e:
            return (False, f"Monotonicity mass test crashed: {e}")
    
    # ----------------------------------------------------------------------
    # Weak Answer Sanity (one test only, no re-deriving or isclose)
    # ----------------------------------------------------------------------
    def test_answer_sanity_sign_finite(self):
        """Check that L(t) is positive and finite for positive t, J0>0."""
        if self.L is None or self.J0 is None:
            return (False, "SKIP: missing L or J0")
        try:
            L_expr = sp.sympify(self.L)
            m_val = 1.0
            lam_val = 0.5
            J0_val = 2.0  # assume positive
            t_vals = [0.0, 0.5, 1.0, 5.0]
            for t_val in t_vals:
                L_val = float(L_expr.subs({self.t: t_val, self.m: m_val, self.lam: lam_val, 'J0': J0_val}))
                if not math.isfinite(L_val):
                    return (False, f"L(t={t_val}) is not finite")
                # We don't check exact value, just positivity
                if J0_val > 0 and L_val <= 0:
                    return (False, f"L(t={t_val}) <= 0 for positive J0")
            return (True, "L(t) is finite and positive for t>=0")
        except Exception as e:
            return (False, f"Sanity test crashed: {e}")
    
    # ----------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)