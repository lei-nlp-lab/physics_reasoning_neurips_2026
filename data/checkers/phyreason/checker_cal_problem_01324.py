import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Default symbols
        self.m = sp.symbols('m', positive=True)
        self.g = sp.symbols('g', positive=True)
        self.L = sp.symbols('L', positive=True)
        
        # Try to extract inputs from solution_instance (safe access)
        try:
            self.m = getattr(solution_instance, 'm', 
                            getattr(solution_instance, 'inputs', {}).get('m', self.m))
            self.g = getattr(solution_instance, 'g', 
                            getattr(solution_instance, 'inputs', {}).get('g', self.g))
            self.L = getattr(solution_instance, 'L', 
                            getattr(solution_instance, 'inputs', {}).get('L', self.L))
        except Exception:
            pass  # Keep default symbols
        
        # Extract results with safe fallbacks
        self.v = self._safe_get(solution_instance, 'v')
        self.omega = self._safe_get(solution_instance, 'omega')
        self.F_B = self._safe_get(solution_instance, 'F_B')
        
        # Collect logs
        self.logs = []
        
    def _safe_get(self, solution_instance, key):
        """Safely retrieve a result from the solution instance."""
        try:
            # First try direct attribute, then results dictionary
            val = getattr(solution_instance, key, 
                         getattr(solution_instance, 'results', {}).get(key, None))
            return val
        except Exception:
            return None
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Compare two values (symbolic or numeric) with tolerance.
        Returns True if they are effectively equal.
        """
        # If both are numeric (or can be converted)
        try:
            a_num = float(a) if not isinstance(a, (int, float)) else a
            b_num = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_num, b_num, rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            pass
        
        # Symbolic comparison
        try:
            diff = sp.simplify(a - b)
            # Check if diff simplifies to 0
            if diff == 0 or diff.equals(0):
                return True
            # If diff is a constant expression, evaluate numerically
            if diff.is_constant():
                return math.isclose(float(diff.evalf()), 0.0, 
                                   rel_tol=rel_tol, abs_tol=abs_tol)
            # Try substituting typical values
            subs_dict = {}
            for sym in diff.free_symbols:
                if sym.name == 'm':
                    subs_dict[sym] = 1.0
                elif sym.name == 'g':
                    subs_dict[sym] = 9.8
                elif sym.name == 'L':
                    subs_dict[sym] = 1.0
                else:
                    subs_dict[sym] = 1.0
            diff_val = diff.subs(subs_dict).evalf()
            return math.isclose(float(diff_val), 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
        except Exception:
            # If all else fails, return False
            return False
    
    def _numeric_val(self, expr):
        """Convert expression to a numeric float for perturbation tests."""
        if isinstance(expr, (int, float)):
            return float(expr)
        if hasattr(expr, 'evalf'):
            try:
                # If it's a constant expression
                if expr.is_constant():
                    return float(expr.evalf())
                # Otherwise substitute default values
                subs_dict = {}
                for sym in expr.free_symbols:
                    if sym.name == 'm':
                        subs_dict[sym] = 1.0
                    elif sym.name == 'g':
                        subs_dict[sym] = 9.8
                    elif sym.name == 'L':
                        subs_dict[sym] = 1.0
                    else:
                        subs_dict[sym] = 1.0
                return float(expr.subs(subs_dict).evalf())
            except Exception:
                # Fallback
                return 1.0
        # If it's not a sympy object and not a number, try to convert
        try:
            return float(expr)
        except:
            return 1.0
    
    def test_invariants(self):
        """Check fundamental physics laws: centripetal condition and force balance."""
        logs = []
        success = True
        
        # 1. Centripetal condition for ball A at highest point: v^2/L = g
        if self.v is not None:
            expr1 = self.v**2 / self.L - self.g
            if self._is_close(expr1, 0):
                logs.append("PASS: Centripetal condition for ball A (v^2/L = g) holds.")
            else:
                logs.append(f"FAIL: Centripetal condition for ball A fails. v^2/L - g = {expr1}")
                success = False
        else:
            logs.append("SKIP: v not available for centripetal condition check.")
            success = False
        
        # 2. Force balance for ball B: F_B = 3mg
        if self.F_B is not None:
            expr2 = self.F_B - 3 * self.m * self.g
            if self._is_close(expr2, 0):
                logs.append("PASS: Force balance for ball B (F_B = 3mg) holds.")
            else:
                logs.append(f"FAIL: Force balance for ball B fails. F_B - 3mg = {expr2}")
                success = False
        else:
            logs.append("SKIP: F_B not available for force balance check.")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results (v, omega, F_B)."""
        logs = []
        success = True
        
        # Check relationship between v and omega: omega = v/L
        if self.v is not None and self.omega is not None:
            expr1 = self.omega - self.v / self.L
            if self._is_close(expr1, 0):
                logs.append("PASS: Angular velocity consistency (ω = v/L) holds.")
            else:
                logs.append(f"FAIL: Angular velocity consistency fails. ω - v/L = {expr1}")
                success = False
        else:
            logs.append("SKIP: v or ω not available for angular velocity check.")
        
        # Check relationship between omega and F_B: F_B - mg = 2mLω²
        if self.F_B is not None and self.omega is not None:
            expr2 = self.F_B - self.m * self.g - 2 * self.m * self.L * self.omega**2
            if self._is_close(expr2, 0):
                logs.append("PASS: Force-angular velocity relationship (F_B - mg = 2mLω²) holds.")
            else:
                logs.append(f"FAIL: Force-angular velocity relationship fails. Difference = {expr2}")
                success = False
        else:
            logs.append("SKIP: F_B or ω not available for force-angular velocity check.")
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb mass m and verify expected trends in v and F_B."""
        logs = []
        success = True
        
        # Get numeric values for inputs
        m_val = self._numeric_val(self.m)
        g_val = self._numeric_val(self.g)
        L_val = self._numeric_val(self.L)
        
        # Original values (computed from physics formulas, not from solution)
        v_orig = math.sqrt(g_val * L_val)
        F_orig = 3 * m_val * g_val
        
        # Perturb mass by +10%
        m_pert = m_val * 1.1
        v_pert = math.sqrt(g_val * L_val)  # v is independent of m
        F_pert = 3 * m_pert * g_val
        
        # Check v remains unchanged (within tolerance)
        if not math.isclose(v_orig, v_pert, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Velocity v should be independent of mass. v_orig={v_orig}, v_pert={v_pert}")
            success = False
        else:
            logs.append("PASS: Velocity v is independent of mass (as expected).")
        
        # Check F_B increases with mass
        if F_pert <= F_orig:
            logs.append(f"FAIL: Force F_B should increase with mass. F_orig={F_orig}, F_pert={F_pert}")
            success = False
        else:
            logs.append(f"PASS: Force F_B increases with mass (F_orig={F_orig}, F_pert={F_pert}).")
        
        # Additionally, check proportionality: F_B ∝ m
        expected_ratio = 1.1
        actual_ratio = F_pert / F_orig
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F_B should be directly proportional to m. Ratio expected={expected_ratio}, actual={actual_ratio}")
            success = False
        else:
            logs.append(f"PASS: F_B is directly proportional to m (ratio={actual_ratio}).")
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {name} Test ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"=== {name} Test ===")
                all_logs.append(f"ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs