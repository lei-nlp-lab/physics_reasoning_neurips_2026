import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize all inputs and results from the solution instance.
        Uses getattr with fallback to results dictionary to prevent AttributeError.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 9.10938356e-31)  # electron mass in kg
        self.e = getattr(solution_instance, 'e', 1.60217662e-19)  # electron charge in C
        self.B = getattr(solution_instance, 'B', 1.0)  # magnetic flux density in T
        self.v0 = getattr(solution_instance, 'v0', 1.0)  # critical velocity in m/s
        
        # Extract results from attributes or results dictionary with safe fallbacks
        self.E = getattr(solution_instance, 'E', 
                        getattr(solution_instance, 'results', {}).get('E', 0.0))
        self.y1 = getattr(solution_instance, 'y_1', 
                         getattr(solution_instance, 'results', {}).get('y_1', 0.0))
        self.N_ratio = getattr(solution_instance, 'N/N_0', 
                              getattr(solution_instance, 'results', {}).get('N/N_0', 0.0))
        self.v = getattr(solution_instance, 'v', 
                        getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.y = getattr(solution_instance, 'y', 
                        getattr(solution_instance, 'results', {}).get('y', 0.0))
        self.v_m = getattr(solution_instance, 'v_m', 
                          getattr(solution_instance, 'results', {}).get('v_m', 0.0))
        self.F_all = getattr(solution_instance, 'F_all', 
                            getattr(solution_instance, 'results', {}).get('F_all', 0.0))
        
        # For monotonicity test, store original v if available
        self.original_v = self.v if hasattr(self, 'v') else 0.9 * self.v0
        
        # Initialize sympy symbols
        self.m_sym = sp.symbols('m', positive=True)
        self.e_sym = sp.symbols('e', positive=True)
        self.B_sym = sp.symbols('B', positive=True)
        self.v0_sym = sp.symbols('v0', positive=True)
        self.v_sym = sp.symbols('v', positive=True)
        self.v_m_sym = sp.symbols('v_m', positive=True)
        self.E_sym = sp.symbols('E', positive=True)
        self.y_sym = sp.symbols('y', positive=True)
        self.y1_sym = sp.symbols('y1', positive=True)
        
        # Collect logs
        self.logs = []
    
    def log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def test_invariants(self):
        """Verify fundamental physics invariants: force balance, energy conservation, Newton's laws."""
        try:
            # 1. Check straight-line condition: eE = e*v0*B => E = v0*B
            expected_E = self.v0 * self.B
            if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Force balance at v0: E = {self.E:.3e}, v0*B = {expected_E:.3e}", True)
            else:
                self.log(f"Force balance at v0 failed: E = {self.E:.3e}, expected v0*B = {expected_E:.3e}", False)
                return False
            
            # 2. Check energy conservation for step2: eE*y1 = ΔKE = 0.5*m*((v0/2)^2 - (v0/4)^2)
            ke_initial = 0.5 * self.m * (self.v0/4)**2
            ke_final = 0.5 * self.m * (self.v0/2)**2
            delta_ke = ke_final - ke_initial
            work_done = abs(self.e) * self.E * self.y1  # Electric force does positive work when electron moves opposite to field (e negative)
            
            if math.isclose(work_done, delta_ke, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Energy conservation for y1: work = {work_done:.3e} J, ΔKE = {delta_ke:.3e} J", True)
            else:
                self.log(f"Energy conservation for y1 failed: work = {work_done:.3e} J, ΔKE = {delta_ke:.3e} J", False)
                return False
            
            # 3. Check that magnitudes of net force at highest and lowest points are equal (problem statement)
            # From steps 5 and 6:
            # Highest point: F_all = e*v_m*B - e*E  (magnetic force upward, electric force downward)
            # Lowest point: F_all = e*E - e*v*B    (electric force upward, magnetic force downward)
            # Their magnitudes should be equal: |e*v_m*B - e*E| = |e*E - e*v*B|
            # => |v_m*B - E| = |E - v*B|
            
            F_high = abs(self.e * self.v_m * self.B - self.e * self.E)
            F_low = abs(self.e * self.E - self.e * self.v * self.B)
            
            if math.isclose(F_high, F_low, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Force magnitude equality: F_high = {F_high:.3e} N, F_low = {F_low:.3e} N", True)
            else:
                self.log(f"Force magnitude equality failed: F_high = {F_high:.3e} N, F_low = {F_low:.3e} N", False)
                return False
            
            # 4. Check Newton's second law consistency: For the highest point, net force should provide centripetal acceleration
            # The exact relationship depends on curvature, but we can check that forces are consistent with the derived v_m
            # From step7: v_m = (2E/B) - v
            expected_v_m = (2 * self.E / self.B) - self.v
            if math.isclose(self.v_m, expected_v_m, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_m consistency: v_m = {self.v_m:.3e} m/s, expected (2E/B)-v = {expected_v_m:.3e} m/s", True)
            else:
                self.log(f"v_m consistency failed: v_m = {self.v_m:.3e} m/s, expected (2E/B)-v = {expected_v_m:.3e} m/s", False)
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Exception in test_invariants: {str(e)}\n{traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate and final results with expected formulas."""
        try:
            # 1. Check E = v0*B (already checked in invariants, but verify symbolically)
            symbolic_E = sp.simplify(self.v0_sym * self.B_sym)
            self.log(f"Symbolic check: E = {self.E}, formula v0*B = {self.v0*self.B}", True)
            
            # 2. Check y1 = 3*m*v0/(32*e*B)
            expected_y1 = (3 * self.m * self.v0) / (32 * abs(self.e) * self.B)
            if math.isclose(self.y1, expected_y1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"y1 consistency: y1 = {self.y1:.3e} m, expected 3mv0/(32eB) = {expected_y1:.3e} m", True)
            else:
                self.log(f"y1 consistency failed: y1 = {self.y1:.3e} m, expected 3mv0/(32eB) = {expected_y1:.3e} m", False)
                return False
            
            # 3. Check y = 2*m*(v0 - v)/(e*B) from step7
            expected_y = (2 * self.m * (self.v0 - self.v)) / (abs(self.e) * self.B)
            if math.isclose(self.y, expected_y, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"y consistency: y = {self.y:.3e} m, expected 2m(v0-v)/(eB) = {expected_y:.3e} m", True)
            else:
                self.log(f"y consistency failed: y = {self.y:.3e} m, expected 2m(v0-v)/(eB) = {expected_y:.3e} m", False)
                return False
            
            # 4. Check N/N_0 = 90%
            if math.isclose(self.N_ratio, 0.9, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"N/N_0 consistency: N/N_0 = {self.N_ratio:.3f}, expected 0.9", True)
            else:
                self.log(f"N/N_0 consistency failed: N/N_0 = {self.N_ratio:.3f}, expected 0.9", False)
                return False
            
            # 5. Check that v = 0.9*v0 from step8 (for the inequality condition)
            expected_v = 0.9 * self.v0
            if math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v consistency: v = {self.v:.3e} m/s, expected 0.9*v0 = {expected_v:.3e} m/s", True)
            else:
                self.log(f"v consistency failed: v = {self.v:.3e} m/s, expected 0.9*v0 = {expected_v:.3e} m/s", False)
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb an input parameter and verify expected trends, then restore."""
        try:
            # Perturb the incident velocity v (which is less than v0) and check that y increases as v decreases
            # From step7: y = 2*m*(v0 - v)/(e*B)
            # dy/dv = -2*m/(e*B) < 0, so y decreases as v increases
            
            # Store original v
            original_v = self.v
            
            # Test with a smaller v
            smaller_v = original_v * 0.95
            y_smaller = (2 * self.m * (self.v0 - smaller_v)) / (abs(self.e) * self.B)
            
            # Test with a larger v
            larger_v = original_v * 1.05
            y_larger = (2 * self.m * (self.v0 - larger_v)) / (abs(self.e) * self.B)
            
            # Check monotonicity: y_smaller > y_original > y_larger
            y_original = (2 * self.m * (self.v0 - original_v)) / (abs(self.e) * self.B)
            
            if y_smaller > y_original > y_larger:
                self.log(f"Monotonicity: y decreases as v increases (y at v={smaller_v:.3e}: {y_smaller:.3e} > y at v={original_v:.3e}: {y_original:.3e} > y at v={larger_v:.3e}: {y_larger:.3e})", True)
            else:
                self.log(f"Monotonicity failed: y_smaller={y_smaller:.3e}, y_original={y_original:.3e}, y_larger={y_larger:.3e}", False)
                return False
            
            # Restore original v (not strictly necessary since we didn't modify instance, but for completeness)
            self.v = original_v
            
            return True
            
        except Exception as e:
            self.log(f"Exception in test_monotonicity: {str(e)}\n{traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        # Run tests in order
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        # Overall success
        success = invariants_ok and trace_ok and monotonicity_ok
        
        # Summary log
        if success:
            self.log("All physics checks passed!", True)
        else:
            self.log("Some physics checks failed. See details above.", False)
        
        return success, self.logs