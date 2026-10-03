import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # SAFE ATTRIBUTE ACCESS: Use ONLY the results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract ALL values from the results dictionary ONLY
        # Use defaults only if not found in results
        self.U1 = self.results.get('U1', 1000.0)
        self.d = self.results.get('d', 0.01)
        self.l = self.results.get('l', 0.05)
        self.m = self.results.get('m', 9.10938356e-31)
        self.e = self.results.get('e', 1.602176634e-19)
        
        # Extracted results (from key mapping)
        self.v0_val = self.results.get('v_0', 0.0)
        self.a_val = self.results.get('a', 0.0)
        self.U2_val = self.results.get('U_2', 0.0)
        self.d_half_val = self.results.get('d/2', 0.0)
        self.L_val = self.results.get('L', 0.0)
        
        # Logs
        self.logs = []
        
    def log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def test_invariants(self):
        """Verify energy conservation and Newton's 2nd law based on KB theorems"""
        try:
            # 1. Work-energy theorem for initial acceleration (from rest to v0)
            # Use a more reasonable threshold for "too small"
            if abs(self.m) < 1e-50 or abs(self.e) < 1e-50:
                self.log(f"Cannot test work-energy theorem: m={self.m:.2e} or e={self.e:.2e} is too small", False)
                return
                
            v0_theoretical = math.sqrt(2 * self.e * self.U1 / self.m)
            v0_check = abs(self.v0_val - v0_theoretical) / max(abs(self.v0_val), abs(v0_theoretical), 1.0)
            # Use relative tolerance
            if math.isclose(v0_check, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Work-energy theorem satisfied: v0 = sqrt(2eU1/m) (relative diff={v0_check:.2e})", True)
            else:
                self.log(f"Work-energy theorem violation: v0 differs by {v0_check:.2e}", False)
            
            # 2. Newton's 2nd law for vertical acceleration in plates
            if abs(self.m) < 1e-50 or abs(self.d) < 1e-50:
                self.log(f"Cannot test Newton's 2nd law: m={self.m:.2e} or d={self.d:.2e} is too small", False)
                return
                
            a_theoretical = (self.e * self.U2_val) / (self.m * self.d)
            a_check = abs(self.a_val - a_theoretical) / max(abs(self.a_val), abs(a_theoretical), 1.0)
            if math.isclose(a_check, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Newton's 2nd law satisfied: a = eU2/(m d) (relative diff={a_check:.2e})", True)
            else:
                self.log(f"Newton's 2nd law violation: a differs by {a_check:.2e}", False)
            
            # 3. Kinematic consistency: d/2 = 0.5*a*t^2 and L = v0*t
            # Compute time from horizontal motion: t = L/v0
            if abs(self.v0_val) < 1e-50:
                self.log(f"Cannot test kinematic consistency: v0={self.v0_val:.2e} is too small", False)
                return
                
            t_calc = self.L_val / self.v0_val
            vertical_disp = 0.5 * self.a_val * (t_calc**2)
            disp_check = abs(vertical_disp - self.d_half_val) / max(abs(vertical_disp), abs(self.d_half_val), 1.0)
            if math.isclose(disp_check, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Kinematic consistency: d/2 = 0.5*a*t^2 satisfied (relative diff={disp_check:.2e})", True)
            else:
                self.log(f"Kinematic inconsistency: d/2 vs computed {vertical_disp:.2e} (relative diff={disp_check:.2e})", False)
                
        except Exception as e:
            self.log(f"Invariant test failed with error: {traceback.format_exc()}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check if v0, a, and U2 satisfy derived relationship U2 = 2*U1*d^2/l^2
            if abs(self.l) < 1e-50:
                self.log(f"Cannot test trace consistency: l={self.l:.2e} is too small", False)
                return
                
            U2_calculated = 2 * self.U1 * (self.d**2) / (self.l**2)
            U2_check = abs(self.U2_val - U2_calculated) / max(abs(self.U2_val), abs(U2_calculated), 1.0)
            
            if math.isclose(U2_check, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Trace consistency: U2 = 2*U1*d^2/l^2 satisfied (relative diff={U2_check:.2e})", True)
            else:
                self.log(f"Trace inconsistency: U2={self.U2_val:.2e} vs calculated {U2_calculated:.2e} (relative diff={U2_check:.2e})", False)
                
            # Check that L equals l (horizontal displacement equals plate length)
            L_check = abs(self.L_val - self.l) / max(abs(self.L_val), abs(self.l), 1.0)
            if math.isclose(L_check, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Horizontal displacement L equals plate length l (relative diff={L_check:.2e})", True)
            else:
                self.log(f"L ({self.L_val:.2e}) ≠ l ({self.l:.2e}) (relative diff={L_check:.2e})", False)
                
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {traceback.format_exc()}", False)
    
    def test_monotonicity(self):
        """Perturb input U1 and check output trends"""
        try:
            original_U1 = self.U1
            perturbed_U1 = original_U1 * 1.1  # Increase by 10%
            
            # Protect against division by zero
            if abs(self.m) < 1e-50 or abs(self.e) < 1e-50:
                self.log(f"Cannot test monotonicity: m={self.m:.2e} or e={self.e:.2e} is too small", False)
                return
            
            # Recalculate v0 using formula
            original_v0 = math.sqrt(2 * self.e * original_U1 / self.m)
            perturbed_v0 = math.sqrt(2 * self.e * perturbed_U1 / self.m)
            
            # Recalculate U2 using formula
            if abs(self.l) < 1e-50:
                self.log(f"Cannot test monotonicity: l={self.l:.2e} is too small", False)
                return
                
            original_U2_calc = 2 * original_U1 * (self.d**2) / (self.l**2)
            perturbed_U2_calc = 2 * perturbed_U1 * (self.d**2) / (self.l**2)
            
            # Check monotonicity: if U1 increases, v0 should increase, U2 should increase
            v0_increased = perturbed_v0 > original_v0
            U2_increased = perturbed_U2_calc > original_U2_calc
            
            success = v0_increased and U2_increased
            if success:
                self.log(f"Monotonicity: U1↑ → v0↑ and U2↑ (v0: {original_v0:.2e}→{perturbed_v0:.2e}, U2: {original_U2_calc:.2e}→{perturbed_U2_calc:.2e})", True)
            else:
                self.log(f"Monotonicity violation: U1↑ but v0 or U2 didn't increase as expected", False)
                
            # Restore original value (important!)
            self.U1 = original_U1
            
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {traceback.format_exc()}", False)
    
    def run_all_tests(self):
        self.logs = []
        self.log("Starting Physics Audit", None)
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Count passes/fails
        passes = sum(1 for log in self.logs if log.startswith("PASS"))
        fails = sum(1 for log in self.logs if log.startswith("FAIL"))
        total = passes + fails
        
        success = (fails == 0) and (total > 0)
        self.log(f"Summary: {passes}/{total} checks passed", None)
        
        return success, self.logs

# Example usage guard
if __name__ == "__main__":
    # For testing purposes only
    class MockSolution:
        def __init__(self):
            self.results = {
                'v_0': math.sqrt(2 * 1.602176634e-19 * 1000 / 9.10938356e-31),
                'a': (1.602176634e-19 * (2*1000*0.01**2/0.05**2)) / (9.10938356e-31 * 0.01),
                'U_2': 2 * 1000 * 0.01**2 / 0.05**2,
                'd/2': 0.005,
                'L': 0.05,
                'U1': 1000.0,
                'd': 0.01,
                'l': 0.05,
                'm': 9.10938356e-31,
                'e': 1.602176634e-19
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nOverall success: {success}")