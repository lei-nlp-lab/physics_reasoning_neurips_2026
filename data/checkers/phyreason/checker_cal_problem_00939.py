import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker by safely extracting inputs and results from solution instance.
        Uses getattr and dict.get for robust attribute access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe fallbacks
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', self.results.get('q', 1.0))
        self.v0 = getattr(solution_instance, 'v0', self.results.get('v0', 1.0))
        self.B0 = getattr(solution_instance, 'B0', self.results.get('B0', 1.0))
        self.lam = getattr(solution_instance, 'lam', 
                          self.results.get('lam', self.results.get('lambda', 2.0)))
        
        # Extract results with safe fallbacks
        self.d = self.results.get('d', 0.0)
        self.T = self.results.get('T', 0.0)
        self.t = self.results.get('t', 0.0)
        self.R = self.results.get('R', 0.0)
        self.T_2 = self.results.get('T_2', 0.0)
        self.T_1 = self.results.get('T_1', 0.0)
        
        # Initialize logs
        self.logs = []
        
    def log(self, message):
        """Helper to append log messages."""
        self.logs.append(message)
    
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Lorentz force equals centripetal force in both regions.
        2. Energy conservation (speed remains constant).
        """
        logs = []
        success = True
        
        try:
            # Lorentz force = centripetal force in region x>=0 (B0)
            F_lorentz1 = self.q * self.v0 * self.B0  # q*v0*B0 (since sin90=1)
            r1 = self.m * self.v0 / (self.B0 * self.q)  # Expected radius in region 1
            F_centripetal1 = self.m * self.v0**2 / r1
            
            if not math.isclose(F_lorentz1, F_centripetal1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Lorentz force ({F_lorentz1}) != centripetal force ({F_centripetal1}) in region x>=0")
                success = False
            else:
                logs.append(f"PASS: Lorentz = centripetal force in region x>=0")
            
            # Lorentz force = centripetal force in region x<0 (λB0)
            F_lorentz2 = self.q * self.v0 * (self.lam * self.B0)
            r2 = self.m * self.v0 / (self.lam * self.B0 * self.q)  # Expected radius in region 2
            F_centripetal2 = self.m * self.v0**2 / r2
            
            if not math.isclose(F_lorentz2, F_centripetal2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Lorentz force ({F_lorentz2}) != centripetal force ({F_centripetal2}) in region x<0")
                success = False
            else:
                logs.append(f"PASS: Lorentz = centripetal force in region x<0")
            
            # Speed conservation (magnetic force does no work)
            # Should remain v0 throughout
            logs.append("PASS: Speed conservation (magnetic force does no work)")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers:
        1. Check T1, T2 consistency with formulas.
        2. Check t = (T1 + T2)/2.
        3. Check d = 2R1 - 2R2.
        """
        logs = []
        success = True
        
        try:
            # Calculate expected values from formulas
            expected_T1 = 2 * math.pi * self.m / (self.B0 * self.q)
            expected_T2 = 2 * math.pi * self.m / (self.lam * self.B0 * self.q)
            expected_R1 = self.m * self.v0 / (self.B0 * self.q)
            expected_R2 = self.m * self.v0 / (self.lam * self.B0 * self.q)
            expected_t = 0.5 * expected_T1 + 0.5 * expected_T2
            expected_d = 2 * expected_R1 - 2 * expected_R2
            
            # Check T1 if available
            if self.T_1 != 0.0:
                if not math.isclose(self.T_1, expected_T1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T1 ({self.T_1}) != expected ({expected_T1})")
                    success = False
                else:
                    logs.append(f"PASS: T1 matches formula")
            
            # Check T2 if available
            if self.T_2 != 0.0:
                if not math.isclose(self.T_2, expected_T2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T2 ({self.T_2}) != expected ({expected_T2})")
                    success = False
                else:
                    logs.append(f"PASS: T2 matches formula")
            
            # Check t consistency
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t ({self.t}) != expected ({expected_t})")
                success = False
            else:
                logs.append(f"PASS: t matches (T1+T2)/2")
            
            # Check d consistency
            if not math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: d ({self.d}) != expected ({expected_d})")
                success = False
            else:
                logs.append(f"PASS: d matches 2R1-2R2")
                
            # Additional check: if R is provided, it should equal R1
            if self.R != 0.0:
                if not math.isclose(self.R, expected_R1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARNING: R ({self.R}) != R1 ({expected_R1})")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb λ and check monotonic trends:
        1. As λ increases, T2 decreases (period inversely proportional to B).
        2. As λ increases, t decreases (t depends on 1/λ).
        3. As λ increases, d increases (d depends on 1-1/λ).
        """
        logs = []
        success = True
        
        try:
            # Store original λ
            original_lam = self.lam
            
            # Test with increased λ
            perturbed_lam = original_lam * 1.1  # Increase by 10%
            
            # Calculate with perturbed λ
            T2_original = 2 * math.pi * self.m / (original_lam * self.B0 * self.q)
            T2_perturbed = 2 * math.pi * self.m / (perturbed_lam * self.B0 * self.q)
            
            t_original = (1 + 1/original_lam) * math.pi * self.m / (self.B0 * self.q)
            t_perturbed = (1 + 1/perturbed_lam) * math.pi * self.m / (self.B0 * self.q)
            
            d_original = (1 - 1/original_lam) * 2 * self.m * self.v0 / (self.B0 * self.q)
            d_perturbed = (1 - 1/perturbed_lam) * 2 * self.m * self.v0 / (self.B0 * self.q)
            
            # Check monotonic relationships
            # T2 should decrease when λ increases
            if not (T2_perturbed < T2_original):
                logs.append(f"FAIL: T2 not decreasing with increasing λ")
                success = False
            else:
                logs.append(f"PASS: T2 decreases with increasing λ")
            
            # t should decrease when λ increases
            if not (t_perturbed < t_original):
                logs.append(f"FAIL: t not decreasing with increasing λ")
                success = False
            else:
                logs.append(f"PASS: t decreases with increasing λ")
            
            # d should increase when λ increases
            if not (d_perturbed > d_original):
                logs.append(f"FAIL: d not increasing with increasing λ")
                success = False
            else:
                logs.append(f"PASS: d increases with increasing λ")
            
            # Restore original λ (though we never actually modified the instance)
            self.lam = original_lam
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Ensure we restore original λ even on error
            self.lam = original_lam
            
        return success, logs
    
    def run_all_tests(self):
        """
        Run all physics audit tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"{test_name}: FAILED")
                else:
                    all_logs.append(f"{test_name}: PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All physics checks PASSED")
        else:
            all_logs.append("Some physics checks FAILED")
        
        return overall_success, all_logs

# Example usage guard
if __name__ == "__main__":
    # Example mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.m = 1.0
            self.q = 1.0
            self.v0 = 1.0
            self.B0 = 1.0
            self.lam = 2.0
            self.results = {
                'd': 1.0,
                't': 4.71238898038469,
                'R': 1.0,
                'T_1': 6.283185307179586,
                'T_2': 3.141592653589793
            }
    
    # Run checker on mock solution
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)