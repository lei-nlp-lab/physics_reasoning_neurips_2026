import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.Q = givens.get('Q', None)
        self.R = givens.get('R', None)
        self.epsilon0 = givens.get('epsilon0', None) or givens.get('eps0', None)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.E_inside = self._to_expr(results.get('E_inside', None))
        self.E_outside = self._to_expr(results.get('E_outside', None))
        self.V_inside = self._to_expr(results.get('V_inside', None))
        self.V_outside = self._to_expr(results.get('V_outside', None))
        
        # Define symbol for radial coordinate
        self.r = sp.Symbol('r', real=True, positive=True)
        
        # Precompute some derived quantities if givens are present
        if self.Q is not None and self.R is not None:
            self.rho = 3 * self.Q / (4 * sp.pi * self.R**3)
        else:
            self.rho = None

    def _to_expr(self, val):
        """Convert a value to a sympy expression, return None if impossible."""
        if val is None:
            return None
        try:
            # If it's already a sympy expression, keep it
            if isinstance(val, sp.Basic):
                return val
            # If it's a string, parse it
            if isinstance(val, str):
                return sp.sympify(val)
            # Otherwise try sympify
            return sp.sympify(val)
        except:
            return None

    def _eval_at(self, expr, r_val):
        """Evaluate a symbolic expression at a given r (numeric). Returns None on failure."""
        if expr is None:
            return None
        try:
            subs_dict = {self.r: r_val}
            numeric = sp.N(expr.subs(subs_dict))
            return float(numeric)
        except:
            return None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All four result keys must be present and not None."""
        missing = []
        for name, val in [('E_inside', self.E_inside),
                          ('E_outside', self.E_outside),
                          ('V_inside', self.V_inside),
                          ('V_outside', self.V_outside)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens(self):
        """Check that required givens exist and are positive."""
        if self.Q is None:
            return (False, "Missing givens: Q")
        if self.R is None or self.R <= 0:
            return (False, "Missing or non-positive R")
        if self.epsilon0 is None or self.epsilon0 <= 0:
            return (False, "Missing or non-positive epsilon0")
        return (True, "Givens present and valid")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use isclose)
    # ------------------------------------------------------------------
    def test_invariant_gauss_outside(self):
        """For r > R: E_outside * 4π r² = Q/ε0"""
        if self.E_outside is None:
            return (False, "SKIP: E_outside not available")
        if self.Q is None or self.epsilon0 is None:
            return (False, "SKIP: missing Q or epsilon0")
        
        r_test = 2.0 * self.R if self.R else 1.0
        E_val = self._eval_at(self.E_outside, r_test)
        if E_val is None:
            return (False, "Cannot evaluate E_outside numerically")
        
        lhs = E_val * 4 * math.pi * r_test**2
        rhs = float(self.Q / self.epsilon0)
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Gauss outside satisfied: {lhs:.4e} ≈ {rhs:.4e}")
        else:
            return (False, f"Gauss outside violated: {lhs:.4e} ≠ {rhs:.4e}")

    def test_invariant_gauss_inside(self):
        """For r < R: E_inside * 4π r² = (ρ * (4/3)π r³)/ε0"""
        if self.E_inside is None or self.rho is None:
            return (False, "SKIP: missing E_inside or rho")
        if self.epsilon0 is None:
            return (False, "SKIP: missing epsilon0")
        
        r_test = 0.5 * self.R if self.R else 0.5
        E_val = self._eval_at(self.E_inside, r_test)
        if E_val is None:
            return (False, "Cannot evaluate E_inside numerically")
        
        lhs = E_val * 4 * math.pi * r_test**2
        rho_val = float(self.rho)
        rhs = (rho_val * (4.0/3.0) * math.pi * r_test**3) / float(self.epsilon0)
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Gauss inside satisfied: {lhs:.4e} ≈ {rhs:.4e}")
        else:
            return (False, f"Gauss inside violated: {lhs:.4e} ≠ {rhs:.4e}")

    def test_invariant_continuity_E(self):
        """E must be continuous at r=R: E_inside(R) = E_outside(R)"""
        if self.E_inside is None or self.E_outside is None:
            return (False, "SKIP: missing E expressions")
        if self.R is None:
            return (False, "SKIP: missing R")
        
        r_test = float(self.R)
        E_in = self._eval_at(self.E_inside, r_test)
        E_out = self._eval_at(self.E_outside, r_test)
        if E_in is None or E_out is None:
            return (False, "Cannot evaluate at r=R")
        
        if math.isclose(E_in, E_out, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E continuous at R: {E_in:.4e} ≈ {E_out:.4e}")
        else:
            return (False, f"E not continuous: inside {E_in:.4e}, outside {E_out:.4e}")

    def test_invariant_continuity_V(self):
        """Potential must be continuous at r=R: V_inside(R) = V_outside(R)"""
        if self.V_inside is None or self.V_outside is None:
            return (False, "SKIP: missing V expressions")
        if self.R is None:
            return (False, "SKIP: missing R")
        
        r_test = float(self.R)
        V_in = self._eval_at(self.V_inside, r_test)
        V_out = self._eval_at(self.V_outside, r_test)
        if V_in is None or V_out is None:
            return (False, "Cannot evaluate at r=R")
        
        if math.isclose(V_in, V_out, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V continuous at R: {V_in:.4e} ≈ {V_out:.4e}")
        else:
            return (False, f"V not continuous: inside {V_in:.4e}, outside {V_out:.4e}")

    def test_invariant_potential_infinity(self):
        """V_outside(r) → 0 as r → ∞"""
        if self.V_outside is None:
            return (False, "SKIP: V_outside not available")
        
        # Evaluate at a large r (e.g., 1000*R)
        r_large = 1000 * (float(self.R) if self.R else 1.0)
        V_val = self._eval_at(self.V_outside, r_large)
        if V_val is None:
            return (False, "Cannot evaluate V_outside at large r")
        
        if math.isclose(V_val, 0.0, abs_tol=1e-4):
            return (True, f"V_outside(∞) ≈ 0: {V_val:.4e}")
        else:
            return (False, f"V_outside(∞) not zero: {V_val:.4e}")

    # ------------------------------------------------------------------
    # Modelling / trace checks (re-derive from physics)
    # ------------------------------------------------------------------
    def test_modeling_inside_field_form(self):
        """E_inside should be (Q * r) / (4 π ε0 R³)"""
        if self.E_inside is None:
            return (False, "SKIP: E_inside missing")
        if self.Q is None or self.R is None or self.epsilon0 is None:
            return (False, "SKIP: missing givens")
        
        expected = (self.Q * self.r) / (4 * sp.pi * self.epsilon0 * self.R**3)
        for r_frac in [0.2, 0.5, 0.8]:
            r_val = r_frac * float(self.R)
            e_actual = self._eval_at(self.E_inside, r_val)
            e_exp = float(expected.subs({self.r: r_val}))
            if e_actual is None or not math.isclose(e_actual, e_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Mismatch at r={r_val}: actual {e_actual}, expected {e_exp}")
        return (True, "E_inside matches expected form")

    def test_modeling_outside_field_form(self):
        """E_outside should be Q / (4 π ε0 r²)"""
        if self.E_outside is None:
            return (False, "SKIP: E_outside missing")
        if self.Q is None or self.epsilon0 is None:
            return (False, "SKIP: missing givens")
        
        expected = self.Q / (4 * sp.pi * self.epsilon0 * self.r**2)
        for r_frac in [1.5, 2.0, 5.0]:
            r_val = r_frac * float(self.R)
            e_actual = self._eval_at(self.E_outside, r_val)
            e_exp = float(expected.subs({self.r: r_val}))
            if e_actual is None or not math.isclose(e_actual, e_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Mismatch at r={r_val}: actual {e_actual}, expected {e_exp}")
        return (True, "E_outside matches expected form")

    def test_modeling_inside_potential_form(self):
        """V_inside should be Q/(8π ε0 R) * (3 - r²/R²)"""
        if self.V_inside is None:
            return (False, "SKIP: V_inside missing")
        if self.Q is None or self.R is None or self.epsilon0 is None:
            return (False, "SKIP: missing givens")
        
        expected = self.Q / (8 * sp.pi * self.epsilon0 * self.R) * (3 - self.r**2 / self.R**2)
        for r_frac in [0.2, 0.5, 0.8]:
            r_val = r_frac * float(self.R)
            v_actual = self._eval_at(self.V_inside, r_val)
            v_exp = float(expected.subs({self.r: r_val}))
            if v_actual is None or not math.isclose(v_actual, v_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Mismatch at r={r_val}: actual {v_actual}, expected {v_exp}")
        return (True, "V_inside matches expected form")

    def test_modeling_outside_potential_form(self):
        """V_outside should be Q/(4 π ε0 r)"""
        if self.V_outside is None:
            return (False, "SKIP: V_outside missing")
        if self.Q is None or self.epsilon0 is None:
            return (False, "SKIP: missing givens")
        
        expected = self.Q / (4 * sp.pi * self.epsilon0 * self.r)
        for r_frac in [1.5, 2.0, 5.0]:
            r_val = r_frac * float(self.R)
            v_actual = self._eval_at(self.V_outside, r_val)
            v_exp = float(expected.subs({self.r: r_val}))
            if v_actual is None or not math.isclose(v_actual, v_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Mismatch at r={r_val}: actual {v_actual}, expected {v_exp}")
        return (True, "V_outside matches expected form")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_E_outside(self):
        """E_outside should decrease as r increases (perturb r from r1 to r2)."""
        if self.E_outside is None:
            return (False, "SKIP: E_outside missing")
        if self.R is None:
            return (False, "SKIP: R missing")
        
        # Baseline at r1, perturbed at r2 (both > R)
        r1 = 2.0 * float(self.R)
        r2 = 3.0 * float(self.R)
        E1 = self._eval_at(self.E_outside, r1)
        E2 = self._eval_at(self.E_outside, r2)
        if E1 is None or E2 is None:
            return (False, "Cannot evaluate E_outside")
        if E2 < E1:
            return (True, f"E_outside decreases: E({r1})={E1:.4e} > E({r2})={E2:.4e}")
        else:
            return (False, f"E_outside does not decrease: E({r1})={E1:.4e} not > E({r2})={E2:.4e}")

    def test_monotonicity_E_inside(self):
        """E_inside should increase as r increases (perturb r from r1 to r2)."""
        if self.E_inside is None:
            return (False, "SKIP: E_inside missing")
        if self.R is None:
            return (False, "SKIP: R missing")
        
        # Baseline at r1, perturbed at r2 (both < R)
        r1 = 0.2 * float(self.R)
        r2 = 0.8 * float(self.R)
        E1 = self._eval_at(self.E_inside, r1)
        E2 = self._eval_at(self.E_inside, r2)
        if E1 is None or E2 is None:
            return (False, "Cannot evaluate E_inside")
        if E2 > E1:
            return (True, f"E_inside increases: E({r1})={E1:.4e} < E({r2})={E2:.4e}")
        else:
            return (False, f"E_inside does not increase: E({r1})={E1:.4e} not < E({r2})={E2:.4e}")

    def test_monotonicity_V_outside(self):
        """V_outside should decrease as r increases (perturb r from r1 to r2)."""
        if self.V_outside is None:
            return (False, "SKIP: V_outside missing")
        if self.R is None:
            return (False, "SKIP: R missing")
        
        # Baseline at r1, perturbed at r2 (both > R)
        r1 = 2.0 * float(self.R)
        r2 = 3.0 * float(self.R)
        V1 = self._eval_at(self.V_outside, r1)
        V2 = self._eval_at(self.V_outside, r2)
        if V1 is None or V2 is None:
            return (False, "Cannot evaluate V_outside")
        if V2 < V1:
            return (True, f"V_outside decreases: V({r1})={V1:.4e} > V({r2})={V2:.4e}")
        else:
            return (False, f"V_outside does not decrease: V({r1})={V1:.4e} not > V({r2})={V2:.4e}")

    def test_monotonicity_V_inside(self):
        """V_inside should decrease as r increases (perturb r from r1 to r2)."""
        if self.V_inside is None:
            return (False, "SKIP: V_inside missing")
        if self.R is None:
            return (False, "SKIP: R missing")
        
        # Baseline at r1, perturbed at r2 (both < R)
        r1 = 0.2 * float(self.R)
        r2 = 0.8 * float(self.R)
        V1 = self._eval_at(self.V_inside, r1)
        V2 = self._eval_at(self.V_inside, r2)
        if V1 is None or V2 is None:
            return (False, "Cannot evaluate V_inside")
        if V2 < V1:
            return (True, f"V_inside decreases: V({r1})={V1:.4e} > V({r2})={V2:.4e}")
        else:
            return (False, f"V_inside does not decrease: V({r1})={V1:.4e} not > V({r2})={V2:.4e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (single method)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """For positive Q, fields and potentials should be positive; finite at r=0."""
        if self.Q is not None and self.Q <= 0:
            # If Q is negative, sign test would be opposite; just skip this test
            return (True, "Q non-positive – sign test not applicable")
        
        # Positive Q assumed
        fail_msgs = []
        if self.E_inside is not None:
            e_in_0 = self._eval_at(self.E_inside, 0.0)
            if e_in_0 is None or e_in_0 < 0:
                fail_msgs.append("E_inside negative or non-finite at r=0")
        if self.E_outside is not None:
            e_out = self._eval_at(self.E_outside, 2*float(self.R))
            if e_out is None or e_out < 0:
                fail_msgs.append("E_outside negative or non-finite")
        if self.V_inside is not None:
            v_in_0 = self._eval_at(self.V_inside, 0.0)
            if v_in_0 is None or v_in_0 < 0:
                fail_msgs.append("V_inside negative or non-finite at r=0")
            # Also check finite
            if v_in_0 is not None and (math.isnan(v_in_0) or math.isinf(v_in_0)):
                fail_msgs.append("V_inside infinite at r=0")
        if self.V_outside is not None:
            v_out = self._eval_at(self.V_outside, 2*float(self.R))
            if v_out is None or v_out < 0:
                fail_msgs.append("V_outside negative or non-finite")
        if fail_msgs:
            return (False, "; ".join(fail_msgs))
        return (True, "Signs and finiteness are reasonable")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all methods starting with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith('test_'):
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