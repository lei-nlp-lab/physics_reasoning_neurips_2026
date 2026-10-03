import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely
        self.m = getattr(solution_instance, 'm', 2.0)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.theta_deg = getattr(solution_instance, 'theta', 37.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Convert angle to radians
        self.theta_rad = math.radians(self.theta_deg)
        
        # Extract results safely - using .results dictionary
        results = getattr(solution_instance, 'results', {})
        
        self.N1 = results.get('N_1', 0.0)
        self.N2 = results.get('N_2', 0.0)
        self.f_max1 = results.get('f_max', 0.0)
        self.f_max2 = results.get("f_max'", 0.0)  # Note: key has quote
        self.F_resultant1 = results.get('F_resultant', 0.0)
        self.F_all = results.get('F_all', 0.0)
        
        # Input forces from problem context
        self.F1 = 10.0  # N (first case)
        self.F2 = 30.0  # N (second case)
        
        # Trigonometric values (from problem)
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        self.logs = []
    
    def test_invariants(self):
        """Test fundamental physics laws."""
        logs = []
        
        try:
            # 1. Check normal force calculations
            N1_expected = self.m * self.g + self.F1 * self.sin37
            if not math.isclose(self.N1, N1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: N1 expected {N1_expected:.2f}, got {self.N1:.2f}")
            else:
                logs.append(f"PASS: N1 = {self.N1:.2f} N")
            
            N2_expected = self.m * self.g + self.F2 * self.sin37
            if not math.isclose(self.N2, N2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: N2 expected {N2_expected:.2f}, got {self.N2:.2f}")
            else:
                logs.append(f"PASS: N2 = {self.N2:.2f} N")
            
            # 2. Check maximum friction forces
            fmax1_expected = self.mu * self.N1
            if not math.isclose(self.f_max1, fmax1_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f_max1 expected {fmax1_expected:.2f}, got {self.f_max1:.2f}")
            else:
                logs.append(f"PASS: f_max1 = {self.f_max1:.2f} N")
            
            fmax2_expected = self.mu * self.N2
            if not math.isclose(self.f_max2, fmax2_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f_max2 expected {fmax2_expected:.2f}, got {self.f_max2:.2f}")
            else:
                logs.append(f"PASS: f_max2 = {self.f_max2:.2f} N")
            
            # 3. Check vertical force balance (no vertical acceleration)
            # For F1:
            F_vertical1 = self.F1 * self.sin37
            weight = self.m * self.g
            # N1 should balance weight + vertical component
            if not math.isclose(self.N1, weight + F_vertical1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical balance violated for F1: N1 != mg + Fsinθ")
            
            # For F2:
            F_vertical2 = self.F2 * self.sin37
            if not math.isclose(self.N2, weight + F_vertical2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical balance violated for F2: N2 != mg + Fsinθ")
            
            # 4. Check horizontal force balance (Newton's 2nd law)
            # For F1: Object at rest, horizontal forces balance
            F_horizontal1 = self.F1 * self.cos37
            # Friction force should equal horizontal component (since no motion)
            if self.F_resultant1 == 0:  # Object doesn't move
                if F_horizontal1 > self.f_max1:
                    logs.append(f"FAIL: F1 should not cause motion (F_horiz={F_horizontal1:.2f} > f_max={self.f_max1:.2f})")
                else:
                    logs.append(f"PASS: F1 doesn't overcome static friction")
            else:
                logs.append(f"FAIL: F_resultant1 should be 0 for F=10N, got {self.F_resultant1}")
            
            # For F2: Object moves
            F_horizontal2 = self.F2 * self.cos37
            if not math.isclose(F_horizontal2, self.f_max2 + self.F_all, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Horizontal forces don't balance for F2: {F_horizontal2:.2f} != {self.f_max2:.2f} + {self.F_all:.2f}")
            else:
                logs.append(f"PASS: Newton's 2nd law satisfied for F2")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('FAIL')]) == 0
    
    def test_trace_consistency(self):
        """Check that intermediate results lead to final answers."""
        logs = []
        
        try:
            # Check that F_resultant1 matches expected answer (0)
            if not math.isclose(self.F_resultant1, 0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_resultant1 should be 0, got {self.F_resultant1}")
            else:
                logs.append(f"PASS: F_resultant1 = 0 N matches target answer")
            
            # Check that F_all matches expected answer (5 N)
            if not math.isclose(self.F_all, 5.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_all should be 5 N, got {self.F_all}")
            else:
                logs.append(f"PASS: F_all = {self.F_all:.2f} N matches target answer")
            
            # Verify the chain of calculations
            # From N1 to f_max1
            if not math.isclose(self.f_max1, self.mu * self.N1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f_max1 inconsistent with N1")
            
            # From N2 to f_max2
            if not math.isclose(self.f_max2, self.mu * self.N2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f_max2 inconsistent with N2")
            
            # Check that F_all = F2*cos37 - f_max2
            F_horiz2 = self.F2 * self.cos37
            if not math.isclose(self.F_all, F_horiz2 - self.f_max2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_all doesn't match F2*cos37 - f_max2")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('FAIL')]) == 0
    
    def test_monotonicity(self):
        """Test that increasing F increases normal force and affects motion correctly."""
        logs = []
        
        try:
            # Save original state
            original_F2 = self.F2
            
            # Test 1: Increase F2 by 10%
            self.F2 = original_F2 * 1.1
            N2_new = self.m * self.g + self.F2 * self.sin37
            f_max2_new = self.mu * N2_new
            F_horiz_new = self.F2 * self.cos37
            F_all_new = F_horiz_new - f_max2_new
            
            # Check monotonic relationships
            if N2_new <= self.N2:
                logs.append(f"FAIL: Normal force should increase with increased F")
            if f_max2_new <= self.f_max2:
                logs.append(f"FAIL: Friction force should increase with increased F")
            
            # The net force should also increase if we're in the sliding regime
            if F_all_new <= self.F_all:
                logs.append(f"WARNING: Net force didn't increase with F (might be near transition)")
            
            # Test 2: Decrease F2 to below transition point (between 10N and 30N)
            self.F2 = 20.0  # Intermediate value
            N2_mid = self.m * self.g + self.F2 * self.sin37
            f_max2_mid = self.mu * N2_mid
            F_horiz_mid = self.F2 * self.cos37
            
            # Check if object would move at F=20N
            if F_horiz_mid > f_max2_mid:
                logs.append(f"INFO: Object would still move at F=20N")
            else:
                logs.append(f"INFO: Object would stop moving at F=20N")
            
            # Restore original
            self.F2 = original_F2
            
            logs.append(f"PASS: Monotonicity tests completed")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith('FAIL')]) == 0
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        
        # Run tests
        test1_ok = self.test_invariants()
        test2_ok = self.test_trace_consistency()
        test3_ok = self.test_monotonicity()
        
        # Summary
        success = test1_ok and test2_ok and test3_ok
        self.logs.append(f"\nSUMMARY: All tests passed? {success}")
        self.logs.append(f" - Invariants: {'PASS' if test1_ok else 'FAIL'}")
        self.logs.append(f" - Trace consistency: {'PASS' if test2_ok else 'FAIL'}")
        self.logs.append(f" - Monotonicity: {'PASS' if test3_ok else 'FAIL'}")
        
        return success, self.logs

# Optional: Add a simple test to verify the checker works
if __name__ == "__main__":
    # Create a mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'N_1': 26.0,
                'N_2': 38.0,
                'f_max': 13.0,
                "f_max'": 19.0,
                'F_resultant': 0.0,
                'F_all': 5.0
            }
    
    mock = MockSolution()
    checker = PhysicsChecker(mock)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)