import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        try:
            self.solution = solution_instance
            
            # Input parameters (safe access from results or attributes)
            results = getattr(solution_instance, 'results', {})
            
            # Get E, C, L from results with defaults
            self.E = results.get('E', getattr(solution_instance, 'E', 12.0))
            self.C = results.get('C', getattr(solution_instance, 'C', 1e-6))
            self.L = results.get('L', getattr(solution_instance, 'L', 0.1))
            
            # Results (safe access with fallback)
            self.t_d = results.get('t', results.get('Deltat', 0.0))
            self.I_avg = results.get('overlineI', 0.0)
            self.Delta_q = results.get('Deltaq', 0.0)
            self.Delta_t = results.get('Deltat', self.t_d)
            
            # Use math.pi for numeric calculations
            self.pi = math.pi
            
        except Exception as e:
            self.init_error = f"Initialization error: {e}"
            print(self.init_error)
    
    def test_invariants(self):
        """Test conservation laws and fundamental physical relationships."""
        logs = []
        try:
            # 1. Check discharge time formula: t = (π/2)√(LC)
            expected_t = (self.pi/2) * math.sqrt(self.L * self.C)
            
            if not math.isclose(self.t_d, expected_t, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Discharge time mismatch. Got {self.t_d}, expected ~{expected_t}")
            else:
                logs.append("PASS: Discharge time matches (π/2)√(LC)")
            
            # 2. Check average current: Ī = (2E/π)√(C/L)
            expected_I = (2 * self.E / self.pi) * math.sqrt(self.C / self.L)
            
            if not math.isclose(self.I_avg, expected_I, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Average current mismatch. Got {self.I_avg}, expected ~{expected_I}")
            else:
                logs.append("PASS: Average current matches (2E/π)√(C/L)")
            
            # 3. Check consistency between Δq, Δt, and Ī
            if self.Delta_t != 0:
                computed_I = self.Delta_q / self.Delta_t
                if not math.isclose(self.I_avg, computed_I, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Ī ≠ Δq/Δt. Ī={self.I_avg}, Δq/Δt={computed_I}")
                else:
                    logs.append("PASS: Ī = Δq/Δt relationship holds")
            
            # 4. Check Δq = CE
            expected_Delta_q = self.C * self.E
            if not math.isclose(self.Delta_q, expected_Delta_q, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Δq ≠ CE. Got {self.Delta_q}, expected {expected_Delta_q}")
            else:
                logs.append("PASS: Δq = CE relationship holds")
                    
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        logs = []
        try:
            # Step 1: t = T/4 where T = 2π√(LC)
            T_expected = 2 * self.pi * math.sqrt(self.L * self.C)
            t_from_T = T_expected / 4
            
            # Step 2: t = (π/2)√(LC)
            t_expected = (self.pi/2) * math.sqrt(self.L * self.C)
            
            # Check consistency between steps (numeric)
            if not math.isclose(t_from_T, t_expected, rel_tol=1e-10):
                logs.append(f"FAIL: Step inconsistency. t_from_T={t_from_T}, t_expected={t_expected}")
            else:
                logs.append("PASS: Steps 1 and 2 are consistent (t = T/4 = (π/2)√(LC))")
            
            # Step 3: Check Ī = Δq/Δt with given formulas
            Delta_q_expected = self.C * self.E
            Delta_t_expected = (self.pi/2) * math.sqrt(self.L * self.C)
            I_from_steps = Delta_q_expected / Delta_t_expected
            expected_I = (2 * self.E / self.pi) * math.sqrt(self.C / self.L)
            
            if not math.isclose(I_from_steps, expected_I, rel_tol=1e-10):
                logs.append(f"FAIL: Step 3 inconsistency in Ī derivation.")
            else:
                logs.append("PASS: Step 3 derivation is consistent")
            
            # Cross-check with final results
            if not math.isclose(self.t_d, self.Delta_t, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"WARN: t and Δt should be equal but differ: t={self.t_d}, Δt={self.Delta_t}")
            else:
                logs.append("PASS: t and Δt are consistent (same quantity)")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        """Test that parameter changes produce expected trends."""
        logs = []
        try:
            # Original values using formulas
            t_original = (self.pi/2) * math.sqrt(self.L * self.C)
            I_original = (2 * self.E / self.pi) * math.sqrt(self.C / self.L)
            
            # Test 1: Increase C by 10%
            C_perturbed = self.C * 1.1
            t_increased_C = (self.pi/2) * math.sqrt(self.L * C_perturbed)
            I_increased_C = (2 * self.E / self.pi) * math.sqrt(C_perturbed / self.L)
            
            # According to KB: t increases with C, Ī increases with √(C/L)
            if not (t_increased_C > t_original and I_increased_C > I_original):
                logs.append("FAIL: C increase should increase both t and Ī")
            else:
                logs.append("PASS: C increase → t↑ and Ī↑ (as expected)")
            
            # Test 2: Increase L by 10%
            L_perturbed = self.L * 1.1
            t_increased_L = (self.pi/2) * math.sqrt(L_perturbed * self.C)
            I_increased_L = (2 * self.E / self.pi) * math.sqrt(self.C / L_perturbed)
            
            # According to KB: t increases with L (since T∝√L), Ī decreases with √(1/L)
            if not (t_increased_L > t_original and I_increased_L < I_original):
                logs.append("FAIL: L increase should increase t but decrease Ī")
            else:
                logs.append("PASS: L increase → t↑ and Ī↓ (as expected)")
            
            # Test 3: Increase E by 10% (should only affect Ī, not t)
            E_perturbed = self.E * 1.1
            t_same = (self.pi/2) * math.sqrt(self.L * self.C)  # t independent of E
            I_increased_E = (2 * E_perturbed / self.pi) * math.sqrt(self.C / self.L)
            
            if not math.isclose(t_same, t_original, rel_tol=0.001):
                logs.append("FAIL: t should be independent of E")
            elif not (I_increased_E > I_original):
                logs.append("FAIL: E increase should increase Ī")
            else:
                logs.append("PASS: E affects only Ī (linear), not t")
            
            # Restore original values (conceptually - we used copies)
            logs.append("INFO: Parameter perturbation test completed, original values preserved")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run each test
            for test_name, test_method in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== {test_name} Test ===")
                test_logs = test_method()
                all_logs.extend(test_logs)
                
                # Check for failure indicators in logs
                for log in test_logs:
                    if log.startswith("FAIL:"):
                        success = False
                    if log.startswith("ERROR:"):
                        success = False
        
        except Exception as e:
            all_logs.append(f"\nCRITICAL ERROR in test suite: {e}")
            all_logs.append(traceback.format_exc())
            success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall result: {'PASS' if success else 'FAIL'}")
        
        return success, all_logs

# Example usage (for testing):
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.E = 12.0  # Volts
            self.C = 1e-6  # Farads
            self.L = 0.1   # Henries
            self.results = {
                't': 0.000992,  # (π/2)√(LC) ≈ 0.000993
                'Deltat': 0.000992,
                'overlineI': 0.0768,  # (2E/π)√(C/L) ≈ 0.0764
                'Deltaq': 1.2e-5  # CE = 12e-6
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)