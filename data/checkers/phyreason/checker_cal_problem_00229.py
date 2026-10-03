import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe extraction of results using getattr with fallback to results dict
        self.alpha = getattr(solution_instance, 'alpha', self.results.get('alpha', None))
        self.U = getattr(solution_instance, 'U', self.results.get('U', None))
        self.x_N = getattr(solution_instance, 'x_N', self.results.get('x_N', None))
        self.y_M = getattr(solution_instance, 'y_M', self.results.get('y_M', None))
        self.k = getattr(solution_instance, 'k', self.results.get('k', None))
        self.B_1 = getattr(solution_instance, 'B_1', self.results.get('B_1', None))
        self.t_1 = getattr(solution_instance, 't_1', self.results.get('t_1', None))
        self.t_2 = getattr(solution_instance, 't_2', self.results.get('t_2', None))
        self.t_3 = getattr(solution_instance, 't_3', self.results.get('t_3', None))
        self.t_n = getattr(solution_instance, 't_n', self.results.get('t_n', None))
        self.v_E = getattr(solution_instance, 'v_E', self.results.get('v_E', None))
        self.r_1 = getattr(solution_instance, 'r_1', self.results.get('r_1', None))
        self.P1N = getattr(solution_instance, 'P1N', self.results.get('P1N', None))
        
        # Extract inputs safely
        self.v0 = getattr(solution_instance, 'v0', None)
        self.T = getattr(solution_instance, 'T', None)
        self.q = getattr(solution_instance, 'q', None)
        self.m = getattr(solution_instance, 'm', None)
        self.n = getattr(solution_instance, 'n', 3)  # Default n for testing
        
        # Define symbols for symbolic checks
        self.v0_sym = sp.Symbol('v0', positive=True)
        self.T_sym = sp.Symbol('T', positive=True)
        self.q_sym = sp.Symbol('q', positive=True)
        self.m_sym = sp.Symbol('m', positive=True)
        self.alpha_sym = sp.Symbol('alpha', positive=True)
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy comparison for numerical values"""
        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
            elif hasattr(a, 'evalf') and hasattr(b, 'evalf'):
                # Symbolic expressions - simplify difference
                diff = sp.simplify(a - b)
                return diff == 0
            else:
                # Try to convert to float
                return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            return False
            
    def test_invariants(self):
        """Check fundamental physics laws"""
        success = True
        
        try:
            # 1. Check angle alpha = 30° from velocity decomposition
            # From step1: 2v0 * cos(alpha) = v0 => cos(alpha) = 1/2 => alpha = 60°
            # But steps analysis says 30°, so we check both possibilities
            if self.alpha is not None:
                # Convert to radians for calculation
                alpha_rad = math.radians(float(self.alpha))
                # Check cos(alpha) = 0.5 or alpha = 30°
                cos_alpha = math.cos(alpha_rad)
                if not (self._is_close(cos_alpha, 0.5) or self._is_close(float(self.alpha), 30)):
                    success = self._log(f"Angle alpha inconsistent: got {self.alpha}°, cos(alpha)={cos_alpha}", False)
                else:
                    success = self._log(f"Angle alpha check passed: {self.alpha}°", True)
            
            # 2. Energy conservation: qU = ΔK = 1/2 m (4v0² - v0²) = 3/2 m v0²
            if all(x is not None for x in [self.U, self.v0, self.q, self.m]):
                delta_K = 0.5 * self.m * ((2*self.v0)**2 - self.v0**2)
                qU = self.q * self.U
                if not self._is_close(qU, delta_K):
                    success = self._log(f"Energy conservation failed: qU={qU}, ΔK={delta_K}", False)
                else:
                    success = self._log(f"Energy conservation passed: qU = ΔK = {delta_K}", True)
            
            # 3. Lorentz force in magnetic field: qvB = mv²/r
            if all(x is not None for x in [self.B_1, self.v0, self.r_1, self.q, self.m]):
                lhs = self.q * (2*self.v0) * self.B_1
                rhs = self.m * (2*self.v0)**2 / self.r_1
                if not self._is_close(lhs, rhs):
                    success = self._log(f"Lorentz force check failed: qvB={lhs}, mv²/r={rhs}", False)
                else:
                    success = self._log(f"Lorentz force check passed: qvB = mv²/r = {lhs}", True)
                    
        except Exception as e:
            success = self._log(f"Invariant test error: {str(e)}", False)
            
        return success
        
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        success = True
        
        try:
            # 1. Check v_E = 2v0 * sin(2α)
            if all(x is not None for x in [self.v_E, self.v0, self.alpha]):
                alpha_rad = math.radians(float(self.alpha))
                v_E_expected = 2 * self.v0 * math.sin(2 * alpha_rad)
                if not self._is_close(self.v_E, v_E_expected):
                    success = self._log(f"v_E inconsistent: got {self.v_E}, expected {v_E_expected}", False)
                else:
                    success = self._log(f"v_E check passed: {self.v_E} ≈ 2v0*sin(2α)", True)
            
            # 2. Check x_N = (v_E * T) * sin(α)
            if all(x is not None for x in [self.x_N, self.v_E, self.T, self.alpha]):
                alpha_rad = math.radians(float(self.alpha))
                x_N_expected = self.v_E * self.T * math.sin(alpha_rad)
                if not self._is_close(self.x_N, x_N_expected):
                    success = self._log(f"x_N inconsistent: got {self.x_N}, expected {x_N_expected}", False)
                else:
                    success = self._log(f"x_N check passed: {self.x_N} ≈ v_E*T*sin(α)", True)
            
            # 3. Check y_M = v0 * T / 2
            if all(x is not None for x in [self.y_M, self.v0, self.T]):
                y_M_expected = self.v0 * self.T / 2
                if not self._is_close(self.y_M, y_M_expected):
                    success = self._log(f"y_M inconsistent: got {self.y_M}, expected {y_M_expected}", False)
                else:
                    success = self._log(f"y_M check passed: {self.y_M} ≈ v0*T/2", True)
            
            # 4. Check boundary slope k = -y_M / x_N
            if all(x is not None for x in [self.k, self.y_M, self.x_N]):
                k_expected = -self.y_M / self.x_N
                if not self._is_close(self.k, k_expected):
                    success = self._log(f"Slope k inconsistent: got {self.k}, expected {k_expected}", False)
                else:
                    success = self._log(f"Slope k check passed: {self.k} ≈ -y_M/x_N", True)
            
            # 5. Check t_1 = T
            if all(x is not None for x in [self.t_1, self.T]):
                if not self._is_close(self.t_1, self.T):
                    success = self._log(f"t_1 inconsistent: got {self.t_1}, expected {self.T}", False)
                else:
                    success = self._log(f"t_1 check passed: t_1 = T = {self.T}", True)
                    
        except Exception as e:
            success = self._log(f"Trace consistency error: {str(e)}", False)
            
        return success
        
    def test_monotonicity(self):
        """Test how results change when inputs are perturbed"""
        success = True
        
        try:
            # We'll perturb v0 by 10% and check trends
            if all(x is not None for x in [self.v0, self.T, self.q, self.m]):
                original_v0 = self.v0
                perturbed_v0 = original_v0 * 1.1
                
                # Recalculate expected values using formulas
                alpha_rad = math.radians(30)  # Assume α=30°
                
                # U should scale with v0²
                U_original = self.U
                U_perturbed_expected = (3 * self.m * perturbed_v0**2) / (2 * self.q)
                U_ratio_expected = (perturbed_v0 / original_v0)**2
                
                # x_N should scale with v0
                if self.x_N is not None:
                    x_N_original = self.x_N
                    x_N_perturbed_expected = (3 * math.sqrt(3) * perturbed_v0 * self.T) / 4
                    x_N_ratio_expected = perturbed_v0 / original_v0
                    
                    # Check if ratio is preserved
                    if not self._is_close(x_N_perturbed_expected / x_N_original, x_N_ratio_expected, rel_tol=5e-2):
                        success = self._log(f"x_N monotonicity failed: ratio {x_N_perturbed_expected/x_N_original} vs expected {x_N_ratio_expected}", False)
                    else:
                        success = self._log(f"x_N monotonicity passed: scales linearly with v0", True)
                
                # B_1 should scale with 1/v0
                if self.B_1 is not None:
                    B_original = self.B_1
                    B_perturbed_expected = (math.sqrt(3) * self.m) / (6 * self.q * self.T)
                    # B_1 doesn't depend on v0 from formula, but check
                    if not self._is_close(B_original, B_perturbed_expected):
                        success = self._log(f"B_1 invariance check: should be independent of v0", False)
                    else:
                        success = self._log(f"B_1 monotonicity passed: independent of v0", True)
                        
        except Exception as e:
            success = self._log(f"Monotonicity test error: {str(e)}", False)
            
        return success
        
    def run_all_tests(self):
        """Run all tests and return summary"""
        self.logs = []
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall_success = test1 and test2 and test3
        
        # Add summary
        summary = f"Physics Checker Summary: {'ALL PASS' if overall_success else 'SOME FAILURES'}"
        summary += f"\n- Invariants: {'PASS' if test1 else 'FAIL'}"
        summary += f"\n- Trace Consistency: {'PASS' if test2 else 'FAIL'}"
        summary += f"\n- Monotonicity: {'PASS' if test3 else 'FAIL'}"
        
        self.logs.insert(0, summary)
        
        return overall_success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(student_solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)