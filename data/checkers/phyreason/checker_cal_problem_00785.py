import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs
        self.a = self._safe_get(solution_instance, 'a', default=sp.symbols('a', positive=True))
        self.c = self._safe_get(solution_instance, 'c', default=sp.symbols('c', positive=True))
        
        # Extract results with fallback to results dictionary
        results = getattr(solution_instance, 'results', {})
        self.t = self._safe_get(solution_instance, 't', results.get('t', None))
        self.l_cx = self._safe_get(solution_instance, 'l_cx', results.get('l_cx', None))
        self.theta = self._safe_get(solution_instance, 'theta', results.get('theta', None))
        self.v = self._safe_get(solution_instance, 'v', results.get('v', None))
        self.l_BE = self._safe_get(solution_instance, 'l_BE', results.get('l_BE', None))
        self.C = self._safe_get(solution_instance, 'C', results.get('C', None))
        
        # Expected symbolic expressions
        self.expected_t = 2*sp.sqrt(6)*self.a/(3*self.c)
        self.expected_l_cx = sp.sqrt(3)*self.a/3
        
        # Prepare substitution dictionary for numeric evaluation
        self.subs_dict = {}
        if isinstance(self.a, sp.Symbol):
            self.subs_dict[self.a] = 1.0
        else:
            self.subs_dict[self.a] = float(self.a)
        if isinstance(self.c, sp.Symbol):
            self.subs_dict[self.c] = 1.0
        else:
            self.subs_dict[self.c] = float(self.c)
        
        self.logs = []

    def _safe_get(self, obj, attr, default=None):
        """Safely retrieve attribute from object or its results dictionary."""
        val = getattr(obj, attr, None)
        if val is None and hasattr(obj, 'results'):
            val = obj.results.get(attr, None)
        return val if val is not None else default

    def _numeric_value(self, expr):
        """Convert expression to numeric float using substitution if possible."""
        if expr is None:
            return None
        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, sp.Expr):
            try:
                # Substitute known symbols
                subs = {s: val for s, val in self.subs_dict.items() if s in expr.free_symbols}
                if subs:
                    expr_sub = expr.subs(subs)
                    if expr_sub.is_number:
                        return float(expr_sub.evalf())
                # Direct evaluation if no substitution needed
                return float(expr.evalf())
            except:
                return None
        return None

    def _is_close(self, a, b):
        """Robust fuzzy comparison with 2% tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)

    def test_invariants(self):
        """Check fundamental physics laws and geometric relations."""
        logs = []
        
        # 1. Snell's law: n = sin(45°)/sin(θ) = √2
        if self.theta is not None:
            theta_val = self._numeric_value(self.theta)
            if theta_val is not None:
                theta_rad = math.radians(theta_val)
                n_computed = math.sin(math.radians(45)) / math.sin(theta_rad)
                if not self._is_close(n_computed, math.sqrt(2)):
                    logs.append(f"FAIL: Snell's law violated: sin(45°)/sin(θ) = {n_computed:.4f}, expected √2 ≈ {math.sqrt(2):.4f}")
                else:
                    logs.append("PASS: Snell's law holds.")
            else:
                logs.append("SKIP: θ cannot be evaluated numerically.")
        else:
            logs.append("SKIP: θ not provided.")
        
        # 2. Speed in medium: v = c / √2
        if self.v is not None and self.c is not None:
            v_val = self._numeric_value(self.v)
            c_val = self._numeric_value(self.c)
            if v_val is not None and c_val is not None:
                expected_v = c_val / math.sqrt(2)
                if not self._is_close(v_val, expected_v):
                    logs.append(f"FAIL: Speed mismatch: v = {v_val:.4f}, expected c/√2 = {expected_v:.4f}")
                else:
                    logs.append("PASS: Speed relation v = c/√2 holds.")
            else:
                logs.append("SKIP: v or c cannot be evaluated numerically.")
        else:
            logs.append("SKIP: v or c not provided.")
        
        # 3. Critical angle: sin(C) = 1/√2 → C = 45°
        if self.C is not None:
            C_val = self._numeric_value(self.C)
            if C_val is not None:
                sin_C = math.sin(math.radians(C_val))
                if not self._is_close(sin_C, 1/math.sqrt(2)):
                    logs.append(f"FAIL: Critical angle incorrect: sin(C) = {sin_C:.4f}, expected 1/√2 ≈ {1/math.sqrt(2):.4f}")
                else:
                    logs.append("PASS: Critical angle correct.")
            else:
                logs.append("SKIP: C cannot be evaluated numerically.")
        else:
            logs.append("SKIP: C not provided.")
        
        # 4. Geometry: tan(θ) = l_BE / a
        if self.theta is not None and self.l_BE is not None and self.a is not None:
            theta_val = self._numeric_value(self.theta)
            l_BE_val = self._numeric_value(self.l_BE)
            a_val = self._numeric_value(self.a)
            if theta_val is not None and l_BE_val is not None and a_val is not None:
                expected_l_BE = a_val * math.tan(math.radians(theta_val))
                if not self._is_close(l_BE_val, expected_l_BE):
                    logs.append(f"FAIL: Geometry mismatch: l_BE = {l_BE_val:.4f}, expected a·tan(θ) = {expected_l_BE:.4f}")
                else:
                    logs.append("PASS: Geometry relation tan(θ) = l_BE/a holds.")
            else:
                logs.append("SKIP: θ, l_BE, or a cannot be evaluated numerically.")
        else:
            logs.append("SKIP: θ, l_BE, or a not provided.")
        
        # 5. Path length and time consistency: t = (a/cosθ) / v
        if self.t is not None and self.theta is not None and self.v is not None and self.a is not None:
            t_val = self._numeric_value(self.t)
            theta_val = self._numeric_value(self.theta)
            v_val = self._numeric_value(self.v)
            a_val = self._numeric_value(self.a)
            if t_val is not None and theta_val is not None and v_val is not None and a_val is not None:
                cos_theta = math.cos(math.radians(theta_val))
                path_length = a_val / cos_theta
                expected_t = path_length / v_val
                if not self._is_close(t_val, expected_t):
                    logs.append(f"FAIL: Time inconsistency: t = {t_val:.4f}, expected (a/cosθ)/v = {expected_t:.4f}")
                else:
                    logs.append("PASS: Time consistent with path length and speed.")
            else:
                logs.append("SKIP: t, θ, v, or a cannot be evaluated numerically.")
        else:
            logs.append("SKIP: t, θ, v, or a not provided.")
        
        self.logs.extend(logs)
        return all('FAIL' not in log for log in logs)

    def test_trace_consistency(self):
        """Verify final answers match expected values and intermediate steps."""
        logs = []
        
        # Check time t
        if self.t is not None:
            t_val = self._numeric_value(self.t)
            expected_t_val = self._numeric_value(self.expected_t)
            if t_val is not None and expected_t_val is not None:
                if not self._is_close(t_val, expected_t_val):
                    logs.append(f"FAIL: Time t = {t_val:.4f}, expected {expected_t_val:.4f}")
                else:
                    logs.append("PASS: Time t matches expected value.")
            else:
                logs.append("SKIP: t or expected_t cannot be evaluated numerically.")
        else:
            logs.append("SKIP: t not provided.")
        
        # Check width l_cx
        if self.l_cx is not None:
            l_cx_val = self._numeric_value(self.l_cx)
            expected_l_cx_val = self._numeric_value(self.expected_l_cx)
            if l_cx_val is not None and expected_l_cx_val is not None:
                if not self._is_close(l_cx_val, expected_l_cx_val):
                    logs.append(f"FAIL: Width l_cx = {l_cx_val:.4f}, expected {expected_l_cx_val:.4f}")
                else:
                    logs.append("PASS: Width l_cx matches expected value.")
            else:
                logs.append("SKIP: l_cx or expected_l_cx cannot be evaluated numerically.")
        else:
            logs.append("SKIP: l_cx not provided.")
        
        self.logs.extend(logs)
        return all('FAIL' not in log for log in logs)

    def test_monotonicity(self):
        """Check scaling behavior with respect to parameters a and c."""
        logs = []
        
        # Time t should be proportional to a and inversely proportional to c
        if self.t is not None:
            try:
                if isinstance(self.t, sp.Expr) and self.a in self.t.free_symbols and self.c in self.t.free_symbols:
                    t_func = sp.lambdify((self.a, self.c), self.t, 'math')
                    a_base, c_base = 1.0, 1.0
                    t_base = t_func(a_base, c_base)
                    
                    # Perturb a
                    a_pert = 1.1 * a_base
                    t_a = t_func(a_pert, c_base)
                    if not self._is_close(t_a / t_base, a_pert / a_base):
                        logs.append(f"FAIL: t does not scale linearly with a: ratio = {t_a/t_base:.4f}, expected {a_pert/a_base:.4f}")
                    else:
                        logs.append("PASS: t scales linearly with a.")
                    
                    # Perturb c (inverse proportionality)
                    c_pert = 1.1 * c_base
                    t_c = t_func(a_base, c_pert)
                    if not self._is_close(t_c / t_base, c_base / c_pert):
                        logs.append(f"FAIL: t does not scale inversely with c: ratio = {t_c/t_base:.4f}, expected {c_base/c_pert:.4f}")
                    else:
                        logs.append("PASS: t scales inversely with c.")
                else:
                    logs.append("SKIP: t is not a symbolic expression in a and c.")
            except Exception as e:
                logs.append(f"SKIP: Monotonicity check for t failed: {e}")
        else:
            logs.append("SKIP: t not provided.")
        
        # Width l_cx should be proportional to a
        if self.l_cx is not None:
            try:
                if isinstance(self.l_cx, sp.Expr) and self.a in self.l_cx.free_symbols:
                    l_cx_func = sp.lambdify(self.a, self.l_cx, 'math')
                    a_base = 1.0
                    l_base = l_cx_func(a_base)
                    a_pert = 1.1 * a_base
                    l_pert = l_cx_func(a_pert)
                    if not self._is_close(l_pert / l_base, a_pert / a_base):
                        logs.append(f"FAIL: l_cx does not scale linearly with a: ratio = {l_pert/l_base:.4f}, expected {a_pert/a_base:.4f}")
                    else:
                        logs.append("PASS: l_cx scales linearly with a.")
                else:
                    logs.append("SKIP: l_cx is not a symbolic expression in a.")
            except Exception as e:
                logs.append(f"SKIP: Monotonicity check for l_cx failed: {e}")
        else:
            logs.append("SKIP: l_cx not provided.")
        
        self.logs.extend(logs)
        return all('FAIL' not in log for log in logs)

    def run_all_tests(self):
        """Execute all checks and return overall success and logs."""
        self.logs = []
        success = True
        try:
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self.logs.append(f"ERROR during test execution: {traceback.format_exc()}")
            success = False
        return success, self.logs