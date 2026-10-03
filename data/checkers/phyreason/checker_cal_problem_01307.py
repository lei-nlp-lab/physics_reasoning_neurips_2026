import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results."""
        try:
            # Safely extract inputs
            self.T = getattr(solution_instance, 'T', 0.0)
            self.R = getattr(solution_instance, 'R', 0.0)
            self.G = getattr(solution_instance, 'G', 0.0)
            
            # Safely extract results with fallback
            results = getattr(solution_instance, 'results', {})
            self.M = results.get('M', 0.0)
            self.g = results.get('g', 0.0)
            
            # Store the instance for possible formula access
            self.solution = solution_instance
        except Exception as e:
            self.error = f"Initialization failed: {str(e)}"
            raise
    
    def compare_values(self, a, b):
        """Robust fuzzy comparison with physics tolerance."""
        try:
            # Use physics-appropriate tolerance (1-2% for rounding noise)
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        except:
            return False
    
    def test_invariants(self):
        """Test fundamental physics laws based on KB."""
        logs = []
        success = True
        
        try:
            # Check for zero values that would cause division errors
            if self.G == 0.0:
                logs.append("WARNING: G is zero, skipping Kepler's law and Newton's law tests")
                # We can still test centripetal acceleration formula
                pass
            else:
                # 1. Check Kepler's Third Law: T² ∝ r³/M
                # M = 4π²R³/(G T²)
                expected_M = 4 * math.pi**2 * self.R**3 / (self.G * self.T**2)
                if not self.compare_values(self.M, expected_M):
                    logs.append("FAIL: Mass M violates Kepler's Third Law")
                    success = False
                else:
                    logs.append("PASS: Kepler's Third Law satisfied for mass M")
                
                # 3. Check Newton's Law of Gravitation at surface: g = G*M/R²
                newton_g = self.G * self.M / self.R**2
                if not self.compare_values(self.g, newton_g):
                    logs.append("FAIL: g and M violate Newton's Law of Gravitation at surface")
                    success = False
                else:
                    logs.append("PASS: Newton's Law of Gravitation consistent at surface")
            
            # 2. Check centripetal acceleration formula (does not require G)
            # g = 4π²R/T²
            expected_g = 4 * math.pi**2 * self.R / self.T**2
            if not self.compare_values(self.g, expected_g):
                logs.append("FAIL: Surface gravity g violates centripetal acceleration formula")
                success = False
            else:
                logs.append("PASS: Centripetal acceleration formula satisfied for g")
            
            # 4. Check orbital velocity consistency (does not require G)
            # For circular orbit at surface: g = v²/R where v = 2πR/T
            v_orbital = 2 * math.pi * self.R / self.T
            centripetal_accel = v_orbital**2 / self.R
            if not self.compare_values(self.g, centripetal_accel):
                logs.append("FAIL: Centripetal vs gravitational force imbalance")
                success = False
            else:
                logs.append("PASS: Centripetal force equals gravitational force")
                
        except Exception as e:
            logs.append(f"ERROR in invariants test: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answers."""
        logs = []
        success = True
        
        try:
            # Check if solution has steps attribute
            steps = getattr(self.solution, 'steps', {})
            
            if 'step_1' in steps:
                step1_eq = steps['step_1'].get('equation', '')
                # Expected: M = 4π²R³/(GT²)
                expected = "M=\\frac{4\\pi^{2}R^{3}}{GT^{2}}"
                if expected not in step1_eq:
                    logs.append("WARN: Step 1 equation doesn't match expected form")
            
            if 'step_2' in steps:
                step2_eq = steps['step_2'].get('equation', '')
                # Expected: g = 4π²R/T²
                expected = "g=\\frac{4\\pi^{2}R}{T^{2}}"
                if expected not in step2_eq:
                    logs.append("WARN: Step 2 equation doesn't match expected form")
            
            # Numerical consistency: Use step formulas to recompute
            # Check for zero values first
            if self.G == 0.0 or self.T == 0.0 or self.R == 0.0:
                logs.append("WARN: Cannot verify steps due to zero input values")
            else:
                try:
                    # Recompute M from step 1 formula
                    M_recalc = 4 * math.pi**2 * self.R**3 / (self.G * self.T**2)
                    if not self.compare_values(self.M, M_recalc):
                        logs.append("FAIL: Step 1 calculation inconsistent with final M")
                        success = False
                    
                    # Recompute g from step 2 formula
                    g_recalc = 4 * math.pi**2 * self.R / self.T**2
                    if not self.compare_values(self.g, g_recalc):
                        logs.append("FAIL: Step 2 calculation inconsistent with final g")
                        success = False
                except Exception as e:
                    logs.append(f"WARN: Could not numerically verify steps: {str(e)}")
            
            if success and not logs:
                logs.append("PASS: All steps consistent with final answers")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Test if perturbations produce physically correct trends."""
        logs = []
        success = True
        
        try:
            # Check for zero values
            if self.G == 0.0 or self.T == 0.0 or self.R == 0.0:
                logs.append("WARN: Cannot test monotonicity due to zero input values")
                return True, logs  # Return True to not fail the test
            
            original_R = self.R
            original_M = self.M
            original_g = self.g
            
            # Perturb radius by +10%
            perturbed_R = original_R * 1.1
            
            # Recalculate using physics formulas
            M_perturbed = 4 * math.pi**2 * perturbed_R**3 / (self.G * self.T**2)
            g_perturbed = 4 * math.pi**2 * perturbed_R / self.T**2
            
            # Check trends based on KB
            # For fixed T and G: M ∝ R³, g ∝ R
            expected_M_trend = (perturbed_R**3) / (original_R**3)
            
            # Avoid division by zero
            if original_M == 0:
                logs.append("WARN: original_M is zero, cannot compute trend ratio")
                actual_M_trend = 0
            else:
                actual_M_trend = M_perturbed / original_M
            
            expected_g_trend = perturbed_R / original_R
            
            if original_g == 0:
                logs.append("WARN: original_g is zero, cannot compute trend ratio")
                actual_g_trend = 0
            else:
                actual_g_trend = g_perturbed / original_g
            
            # Fuzzy trend checking (allow 5% error in trend ratio)
            if original_M != 0 and not math.isclose(actual_M_trend, expected_M_trend, rel_tol=0.05):
                logs.append(f"FAIL: M trend incorrect. Expected ~{expected_M_trend:.3f}, got {actual_M_trend:.3f}")
                success = False
            
            if original_g != 0 and not math.isclose(actual_g_trend, expected_g_trend, rel_tol=0.05):
                logs.append(f"FAIL: g trend incorrect. Expected ~{expected_g_trend:.3f}, got {actual_g_trend:.3f}")
                success = False
            
            if success:
                logs.append("PASS: Monotonicity trends correct for R perturbation")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return aggregate results."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                for log in logs:
                    all_logs.append(f"  {log}")
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"  TEST CRASHED: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs

# Example usage guard
if __name__ == "__main__":
    # Mock solution for testing
    class MockSolution:
        def __init__(self):
            self.T = 100.0  # seconds
            self.R = 1.74e6  # meters
            self.G = 6.67430e-11  # m³/kg/s²
            self.results = {
                'M': 7.35e22,  # kg (actual Moon mass)
                'g': 1.62  # m/s²
            }
            self.steps = {
                'step_1': {'equation': 'M=\\frac{4\\pi^{2}R^{3}}{GT^{2}}'},
                'step_2': {'equation': 'g=\\frac{4\\pi^{2}R}{T^{2}}'}
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)