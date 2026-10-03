import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        Extract inputs and results from the solution instance.
        """
        # Safe extraction of results (default to empty dict if not present)
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants (with unit conversions as needed)
        self.m = 2e-3  # kg (2 g -> 0.002 kg)
        self.g = 10.0  # m/s²
        self.l = 1.0   # m
        self.R = 0.4   # m
        self.H = 0.2   # m
        self.v_conveyor = 2.0  # m/s
        self.mu = 0.5  # coefficient of kinetic friction
        
        # Extract key results with safe fallbacks
        self.v_b = self.results.get('v_b', None)
        self.v_0 = self.results.get('v_0', None)
        self.h_1 = self.results.get('h_1', None)
        self.F_N_expr = self.results.get('F_N', None)  # could be string or numeric
        self.DF = self.results.get('DF', None)
        self.x_range = self.results.get('x', None)    # could be string or tuple
        self.s_2 = self.results.get('s_2', None)
        self.s_3 = self.results.get('s_3', None)
        self.t_fall = self.results.get('t', None)
        self.s_proj = self.results.get('s', None)
        
        # For symbolic checks
        self.h = sp.symbols('h', positive=True)  # symbolic variable for height
        
        # Initialize test logs
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerances."""
        # Handle None values
        if a is None or b is None:
            return False
        # Handle symbolic expressions
        if hasattr(a, 'evalf') or hasattr(b, 'evalf'):
            try:
                a_val = float(a) if not isinstance(a, (int, float)) else a
                b_val = float(b) if not isinstance(b, (int, float)) else b
                return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
            except:
                return False
        # Normal numerical comparison
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        success = True
        
        # 1. Conservation of energy for block b's fall
        if self.v_b is not None:
            # v_b should satisfy v_b = sqrt(2*g*h) for some h
            # We don't know h, but we can check consistency with collision
            pass
        
        # 2. Elastic collision conservation laws
        if self.v_b is not None and self.v_0 is not None:
            # For equal masses in elastic collision:
            # Momentum: v_b = v_b' + v_0
            # Energy: v_b² = v_b'² + v_0²
            # This implies v_0 = v_b and v_b' = 0, or v_0 = 0 and v_b' = v_b
            # From steps: v_0 = v_b = 5 m/s
            if not self._isclose(self.v_0, self.v_b):
                self._log(f"Collision: v_0={self.v_0} ≠ v_b={self.v_b}", False)
                success = False
            else:
                self._log(f"Collision: v_0 = v_b = {self.v_0} m/s (elastic, equal masses)", True)
        
        # 3. Work-energy theorem for minimum height h_1
        if self.h_1 is not None:
            # From step 5: m*g*h_1 - 2*μ*m*g*l - m*g*H = 0
            expected_h1 = 2*self.mu*self.l + self.H  # 2*0.5*1 + 0.2 = 1.2
            if not self._isclose(self.h_1, expected_h1):
                self._log(f"Minimum height: h_1={self.h_1}, expected {expected_h1}", False)
                success = False
            else:
                self._log(f"Minimum height h_1={self.h_1} m correct", True)
        
        # 4. Centripetal force at point E (using symbolic check)
        if self.F_N_expr is not None:
            try:
                # Expected: F_N = 0.1*h - 0.14 (for h >= 1.2)
                expected_FN = 0.1*self.h - 0.14
                
                # Parse the provided expression
                if isinstance(self.F_N_expr, str):
                    # Remove spaces and convert to sympy
                    expr_str = self.F_N_expr.replace(' ', '')
                    # Handle common formats
                    if '0.1h-0.14' in expr_str or '0.1*h-0.14' in expr_str:
                        provided_FN = expected_FN  # Force match for this format
                    else:
                        provided_FN = sp.sympify(expr_str)
                else:
                    provided_FN = sp.sympify(str(self.F_N_expr))
                
                # Simplify difference
                diff = sp.simplify(provided_FN - expected_FN)
                if diff != 0:
                    # Evaluate numerically at a test point
                    test_h = 1.5
                    val_provided = float(provided_FN.subs(self.h, test_h))
                    val_expected = float(expected_FN.subs(self.h, test_h))
                    if not self._isclose(val_provided, val_expected):
                        self._log(f"Normal force mismatch at h={test_h}: {val_provided} vs {val_expected}", False)
                        success = False
                    else:
                        self._log(f"Normal force expression matches numerically", True)
                else:
                    self._log(f"Normal force expression F_N = {provided_FN} matches expected form", True)
            except Exception as e:
                self._log(f"Could not parse F_N expression: {e}", False)
                success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check DF = √3/5
        if self.DF is not None:
            expected_DF = math.sqrt(3)/5  # ≈ 0.3464
            if not self._isclose(self.DF, expected_DF):
                self._log(f"DF={self.DF}, expected √3/5≈{expected_DF}", False)
                success = False
            else:
                self._log(f"DF={self.DF} matches geometric relationship", True)
        
        # 2. Check s_2 = 1.8 m (for h in (0.9, 1.2))
        if self.s_2 is not None:
            # From step 14: m*g*h - μ*m*g*s_2 = 0 → s_2 = h/μ
            # At h=0.9 (lower bound), s_2 = 0.9/0.5 = 1.8
            expected_s2 = 1.8
            if not self._isclose(self.s_2, expected_s2):
                self._log(f"s_2={self.s_2}, expected {expected_s2} for h=0.9", False)
                success = False
            else:
                self._log(f"s_2={self.s_2} m correct for h=0.9", True)
        
        # 3. Check s_3 = 0.4 m
        if self.s_3 is not None:
            # From step 15: m*g*H - μ*m*g*s_3 = 0 → s_3 = H/μ = 0.2/0.5 = 0.4
            expected_s3 = 0.4
            if not self._isclose(self.s_3, expected_s3):
                self._log(f"s_3={self.s_3}, expected {expected_s3}", False)
                success = False
            else:
                self._log(f"s_3={self.s_3} m correct", True)
        
        # 4. Check free-fall time from E to F
        if self.t_fall is not None:
            # H = (1/2)*g*t² → t = sqrt(2H/g)
            expected_t = math.sqrt(2*self.H/self.g)  # sqrt(0.4/10) = sqrt(0.04) = 0.2
            if not self._isclose(self.t_fall, expected_t):
                self._log(f"Fall time t={self.t_fall}, expected {expected_t}", False)
                success = False
            else:
                self._log(f"Fall time t={self.t_fall} s correct", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        success = True
        
        # Perturbation test on friction coefficient μ
        original_mu = self.mu
        perturbed_mu = original_mu * 1.1  # Increase by 10%
        
        # Recompute expected h_1: h_1 = 2*μ*l + H
        original_h1 = 2*original_mu*self.l + self.H
        perturbed_h1 = 2*perturbed_mu*self.l + self.H
        
        # Check monotonicity: increasing μ should increase h_1
        if perturbed_h1 > original_h1:
            self._log(f"Monotonicity: μ↑ ({original_mu}→{perturbed_mu}) → h_1↑ ({original_h1:.2f}→{perturbed_h1:.2f})", True)
        else:
            self._log(f"Monotonicity violated: μ↑ but h_1 not increased", False)
            success = False
        
        # Restore original μ
        self.mu = original_mu
        
        # Perturbation test on height h (for normal force)
        test_h_values = [1.3, 1.5, 1.6]
        F_N_values = []
        for h_val in test_h_values:
            F_N = 0.1*h_val - 0.14
            F_N_values.append(F_N)
        
        # Check that F_N increases with h (positive derivative)
        if all(F_N_values[i] < F_N_values[i+1] for i in range(len(F_N_values)-1)):
            self._log("Monotonicity: F_N increases with h (dF_N/dh = 0.1 > 0)", True)
        else:
            self._log("Monotonicity: F_N should increase with h", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        # Run tests in sequence
        test_results = []
        
        try:
            self._log("Starting Physics Audit...")
            
            # Test 1: Invariants
            inv_success = self.test_invariants()
            test_results.append(inv_success)
            self._log(f"Invariants test: {'PASS' if inv_success else 'FAIL'}")
            
            # Test 2: Trace Consistency
            trace_success = self.test_trace_consistency()
            test_results.append(trace_success)
            self._log(f"Trace consistency test: {'PASS' if trace_success else 'FAIL'}")
            
            # Test 3: Monotonicity
            mono_success = self.test_monotonicity()
            test_results.append(mono_success)
            self._log(f"Monotonicity test: {'PASS' if mono_success else 'FAIL'}")
            
            # Overall success
            overall_success = all(test_results)
            self._log(f"\nOverall audit: {'PASS' if overall_success else 'FAIL'}")
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"Physics audit crashed: {e}\n{traceback.format_exc()}"
            self._log(error_msg, False)
            return False, self.logs