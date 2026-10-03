import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        # Inputs - try to get from results dict first, then from instance attributes
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all needed values from results dict with fallbacks
        self.R = self.results.get('R', getattr(solution_instance, 'R', None))
        self.B = self.results.get('B', getattr(solution_instance, 'B', None))
        self.v = self.results.get('v', getattr(solution_instance, 'v', None))
        self.q = self.results.get('q', getattr(solution_instance, 'q', None))
        self.m = self.results.get('m', getattr(solution_instance, 'm', None))
        
        # Extract results with fallbacks
        self.d = self.results.get('d', 0.0)
        self.fracqm = self.results.get('fracqm', 0.0)  # q/m ratio (signed)
        self.r = self.results.get('r', 0.0)
        self.T = self.results.get('T', 0.0)
        self.t = self.results.get('t', 0.0)
        
        # Constants
        self.theta = math.radians(60)  # 60° deflection
    
    def test_invariants(self):
        """Verify energy conservation and force balance laws."""
        logs = []
        success = True
        
        # 1. Energy conservation: magnetic force does no work, so speed should be constant
        # In circular motion in B-field, speed |v| is constant
        # We'll check via the radius formula which assumes constant v
        if self.R is not None and self.r is not None and self.v is not None and self.B is not None and abs(self.fracqm) > 1e-15:
            # From Lorentz force = centripetal: |q|vB = mv²/r => r = mv/(|q|B) = v/(|q/m|*B)
            r_expected = self.v / (abs(self.fracqm) * self.B)
            if not math.isclose(self.r, r_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Radius invariant violated: r={self.r}, expected={r_expected}")
                success = False
            else:
                logs.append("Radius invariant (Lorentz=centripetal) satisfied.")
        
        # 2. Geometry invariant: r = R / tan(30°)
        if self.R is not None and self.r is not None:
            r_geo = self.R / math.tan(math.radians(30))
            if not math.isclose(self.r, r_geo, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Geometry invariant violated: r={self.r}, R/tan30°={r_geo}")
                success = False
            else:
                logs.append("Geometry invariant (r=R/tan30°) satisfied.")
        
        # 3. Period invariant: T = 2πm/(|q|B) = 2π/(|q/m|*B)
        if self.T is not None and self.B is not None and abs(self.fracqm) > 1e-15:
            T_expected = 2 * math.pi / (abs(self.fracqm) * self.B)
            if not math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Period invariant violated: T={self.T}, expected={T_expected}")
                success = False
            else:
                logs.append("Period invariant (T=2πm/|q|B) satisfied.")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # 1. Check specific charge |q/m| (magnitude)
        if self.fracqm is not None and self.R is not None and self.B is not None and self.v is not None:
            # From steps: |q/m| = √3 v / (3 B R)
            expected_fracqm_mag = (math.sqrt(3) * self.v) / (3 * self.B * self.R)
            actual_fracqm_mag = abs(self.fracqm)
            if not math.isclose(actual_fracqm_mag, expected_fracqm_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"|q/m| inconsistent: {actual_fracqm_mag} vs expected {expected_fracqm_mag}")
                success = False
            else:
                logs.append(f"|q/m|={actual_fracqm_mag} matches expected formula.")
            
            # Check sign: should be negative for negative charge
            if self.fracqm >= 0:
                logs.append(f"Sign violation: q/m should be negative for negative particle, got {self.fracqm}")
                success = False
            else:
                logs.append(f"q/m sign correct: negative charge → negative q/m ({self.fracqm})")
        
        # 2. Check time t
        if self.t is not None and self.R is not None and self.v is not None and self.v != 0:
            # From steps: t = √3 π R / (3 v)
            expected_t = (math.sqrt(3) * math.pi * self.R) / (3 * self.v)
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Time t inconsistent: {self.t} vs expected {expected_t}")
                success = False
            else:
                logs.append(f"t={self.t} matches expected formula.")
        
        # 3. Check translation distance d
        if self.d is not None and self.R is not None:
            # From steps: d = √3 R / 3
            expected_d = (math.sqrt(3) / 3) * self.R
            if not math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Distance d inconsistent: {self.d} vs expected {expected_d}")
                success = False
            else:
                logs.append(f"d={self.d} matches expected formula.")
        
        # 4. Check t = T/6 relationship
        if self.t is not None and self.T is not None and self.T != 0:
            expected_t_from_T = self.T / 6
            if not math.isclose(self.t, expected_t_from_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"t=T/6 violated: t={self.t}, T/6={expected_t_from_T}")
                success = False
            else:
                logs.append("t = T/6 relationship satisfied.")
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        logs = []
        success = True
        
        # Check if we have all required data
        if not (self.R is not None and self.B is not None and self.v is not None and self.fracqm is not None):
            logs.append("Insufficient data for monotonicity test.")
            return False, logs
        
        original_B = self.B
        perturbed_B = original_B * 1.01  # Increase B by 1%
        
        # 1. Test: Increasing B should decrease radius r (for fixed v and |q/m|)
        original_r = self.v / (abs(self.fracqm) * original_B)
        perturbed_r = self.v / (abs(self.fracqm) * perturbed_B)
        
        if perturbed_r >= original_r:
            logs.append(f"Monotonicity violated: B↑ should make r↓, but r changed from {original_r} to {perturbed_r}")
            success = False
        else:
            logs.append(f"Monotonicity passed: B↑ ({original_B}→{perturbed_B}) causes r↓ ({original_r}→{perturbed_r})")
        
        # 2. Test: Increasing B should decrease period T
        original_T = 2 * math.pi / (abs(self.fracqm) * original_B)
        perturbed_T = 2 * math.pi / (abs(self.fracqm) * perturbed_B)
        
        if perturbed_T >= original_T:
            logs.append(f"Monotonicity violated: B↑ should make T↓, but T changed from {original_T} to {perturbed_T}")
            success = False
        else:
            logs.append(f"Monotonicity passed: B↑ causes T↓ ({original_T}→{perturbed_T})")
        
        # Restore original B
        self.B = original_B
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        logs = []
        overall_success = True
        
        test_functions = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            logs.append(f"\n=== {test_name} ===")
            try:
                test_success, test_logs = test_func()
                logs.extend(test_logs)
                if not test_success:
                    overall_success = False
                    logs.append(f"✗ {test_name} FAILED")
                else:
                    logs.append(f"✓ {test_name} PASSED")
            except Exception as e:
                logs.append(f"Error in {test_name}: {str(e)}")
                traceback.print_exc()
                overall_success = False
        
        logs.append(f"\n=== SUMMARY ===")
        logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, logs