import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction: use getattr with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (with safe defaults)
        self.m = getattr(solution_instance, 'm', 50.0)
        self.t1 = getattr(solution_instance, 't1', 0.6)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin53 = getattr(solution_instance, 'sin53', 0.8)
        self.cos53 = getattr(solution_instance, 'cos53', 0.6)
        
        # Extract results with safe fallbacks
        self.H = self.results.get('H', 0.0)
        self.x3 = self.results.get('x_3', 0.0)
        self.vy = self.results.get('v_y', 0.0)
        self.v0 = self.results.get('v_0', 0.0)
        self.y3 = self.results.get('y_3', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.y1 = self.results.get('y_1', 0.0)
        self.y2 = self.results.get('y_2', 0.0)
        
        # Derived constants
        self.tan53 = self.sin53 / self.cos53  # = 0.8/0.6 = 4/3 ≈ 1.333
        
        # For symbolic checks
        self.symbols = sp.symbols('v0 vy t1 g y1 y2 H x2 t2')
        self.logs = []
    
    def _log(self, msg, success=None):
        """Helper to log messages and success status."""
        entry = {'message': msg}
        if success is not None:
            entry['success'] = success
        self.logs.append(entry)
    
    def test_invariants(self):
        """Check fundamental physical laws and step equations."""
        success = True
        
        # 1. Free fall: v_y = g * t1
        expected_vy = self.g * self.t1
        if not math.isclose(self.vy, expected_vy, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Free fall invariant failed: v_y = {self.vy}, expected g*t1 = {expected_vy}", False)
            success = False
        else:
            self._log(f"Free fall invariant passed: v_y = {self.vy} ≈ g*t1 = {expected_vy}", True)
        
        # 2. Perpendicular landing condition: v0 = v_y * tan53
        expected_v0 = self.vy * self.tan53
        if not math.isclose(self.v0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Perpendicular condition failed: v0 = {self.v0}, expected v_y*tan53 = {expected_v0}", False)
            success = False
        else:
            self._log(f"Perpendicular condition passed: v0 = {self.v0} ≈ v_y*tan53 = {expected_v0}", True)
        
        # 3. Vertical displacement A->B: y1 = 0.5 * g * t1^2
        expected_y1 = 0.5 * self.g * self.t1**2
        if not math.isclose(self.y1, expected_y1, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Vertical displacement y1 failed: y1 = {self.y1}, expected 0.5*g*t1^2 = {expected_y1}", False)
            success = False
        else:
            self._log(f"Vertical displacement y1 passed: y1 = {self.y1} ≈ 0.5*g*t1^2 = {expected_y1}", True)
        
        # 4. Second jump projectile motion (B->C)
        #    Derive t2 from the condition tan53 = y2 / x2
        #    x2 = (3/8)*v0 * t2, y2 = 0.5*g*t2^2
        #    => tan53 = (0.5*g*t2^2) / ((3/8)*v0*t2) = (4*g*t2) / (3*v0)
        #    => t2 = (3*v0*tan53) / (4*g)
        t2_calc = (3 * self.v0 * self.tan53) / (4 * self.g)
        # Now compute expected x2 and y2
        expected_x2 = (3/8) * self.v0 * t2_calc
        expected_y2 = 0.5 * self.g * t2_calc**2
        
        # Check y2/x2 = tan53
        if self.x2 != 0 and not math.isclose(self.y2 / self.x2, self.tan53, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Slope condition y2/x2 failed: y2/x2 = {self.y2/self.x2}, tan53 = {self.tan53}", False)
            success = False
        else:
            self._log(f"Slope condition y2/x2 passed: y2/x2 ≈ tan53 = {self.tan53}", True)
        
        # Check x2 and y2 individually (if provided)
        if self.x2 != 0 and not math.isclose(self.x2, expected_x2, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"x2 failed: x2 = {self.x2}, expected (3/8)*v0*t2 = {expected_x2}", False)
            success = False
        else:
            self._log(f"x2 passed: x2 = {self.x2} ≈ (3/8)*v0*t2 = {expected_x2}", True)
        
        if self.y2 != 0 and not math.isclose(self.y2, expected_y2, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"y2 failed: y2 = {self.y2}, expected 0.5*g*t2^2 = {expected_y2}", False)
            success = False
        else:
            self._log(f"y2 passed: y2 = {self.y2} ≈ 0.5*g*t2^2 = {expected_y2}", True)
        
        # 5. Total height: H = y1 + y2
        expected_H = self.y1 + self.y2
        if not math.isclose(self.H, expected_H, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Total height H failed: H = {self.H}, expected y1+y2 = {expected_H}", False)
            success = False
        else:
            self._log(f"Total height H passed: H = {self.H} ≈ y1+y2 = {expected_H}", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check that v0 matches the target answer (8 m/s)
        if not math.isclose(self.v0, 8.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"v0 consistency failed: v0 = {self.v0}, target = 8 m/s", False)
            success = False
        else:
            self._log(f"v0 consistency passed: v0 = {self.v0} ≈ 8 m/s", True)
        
        # Check that H matches the target answer (5 m)
        if not math.isclose(self.H, 5.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"H consistency failed: H = {self.H}, target = 5 m", False)
            success = False
        else:
            self._log(f"H consistency passed: H = {self.H} ≈ 5 m", True)
        
        # Verify that y1 and y2 sum to H
        if not math.isclose(self.y1 + self.y2, self.H, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"H = y1+y2 consistency failed: y1={self.y1}, y2={self.y2}, sum={self.y1+self.y2}, H={self.H}", False)
            success = False
        else:
            self._log(f"H = y1+y2 consistency passed", True)
        
        # Verify that v0, vy, and tan53 are consistent
        if not math.isclose(self.v0, self.vy * self.tan53, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"v0 = vy*tan53 consistency failed", False)
            success = False
        else:
            self._log(f"v0 = vy*tan53 consistency passed", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb t1 and verify that H changes as expected."""
        success = True
        original_t1 = self.t1
        
        try:
            # Perturb t1 by +1%
            perturbed_t1 = original_t1 * 1.01
            self._log(f"Perturbing t1 from {original_t1} to {perturbed_t1}", None)
            
            # Recompute quantities using the same physical formulas
            perturbed_vy = self.g * perturbed_t1
            perturbed_v0 = perturbed_vy * self.tan53
            perturbed_y1 = 0.5 * self.g * perturbed_t1**2
            # For the second jump, t2 = v0/g (derived from tan53 condition)
            perturbed_t2 = perturbed_v0 / self.g
            perturbed_y2 = 0.5 * self.g * perturbed_t2**2
            perturbed_H = perturbed_y1 + perturbed_y2
            
            # Expected H from the symbolic formula: H = 0.5*g*t1^2 * (1 + tan53^2)
            expected_perturbed_H = 0.5 * self.g * perturbed_t1**2 * (1 + self.tan53**2)
            
            # Check that perturbed H is greater than original H (since t1 increased)
            if not perturbed_H > self.H:
                self._log(f"Monotonicity failed: H did not increase when t1 increased", False)
                success = False
            else:
                self._log(f"Monotonicity passed: H increased from {self.H} to {perturbed_H} when t1 increased", True)
            
            # Check that the recomputed H matches the expected formula
            if not math.isclose(perturbed_H, expected_perturbed_H, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Perturbed H formula failed: computed {perturbed_H}, expected {expected_perturbed_H}", False)
                success = False
            else:
                self._log(f"Perturbed H formula passed: {perturbed_H} ≈ 0.5*g*t1^2*(1+tan53^2)", True)
                
        except Exception as e:
            self._log(f"Monotonicity test error: {traceback.format_exc()}", False)
            success = False
        finally:
            # Restore original t1
            self.t1 = original_t1
            self._log(f"Restored t1 to {original_t1}", None)
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        overall = invariants_ok and trace_ok and monotonicity_ok
        
        # Summary log
        self._log(f"All tests completed. Invariants: {invariants_ok}, Trace: {trace_ok}, Monotonicity: {monotonicity_ok}. Overall: {overall}", None)
        
        return overall, self.logs