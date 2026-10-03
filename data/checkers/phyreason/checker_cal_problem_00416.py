import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution_instance."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Safely extract all values with fallback defaults
        self.v0 = getattr(solution_instance, 'v_0', self.results.get('v_0', 0.0))
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        
        # Problem constants (from context)
        self.a = 0.2  # m/s^2 (given in problem)
        
        # Expected values from unit conversions (for trace consistency)
        self.expected_v0 = 30.0  # 108 km/h → m/s
        self.expected_v = 40.0   # 144 km/h → m/s
        
        # Store original values for monotonicity test
        self._orig_v0 = float(self.v0)
        
    def test_invariants(self):
        """Verify physics laws are satisfied."""
        logs = []
        
        # Convert to float for numerical checks
        try:
            v0 = float(self.v0)
            v = float(self.v)
            t = float(self.t)
        except (ValueError, TypeError) as e:
            logs.append(f"ERROR: Invalid numeric values - {e}")
            return False, logs
        
        # 1. Check kinematic equation: v = v0 + a*t
        expected_v = v0 + self.a * t
        if not math.isclose(v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Kinematic equation violated")
            logs.append(f"  v = {v}, but v0 + a*t = {v0} + {self.a}*{t} = {expected_v}")
            logs.append(f"  Difference: {abs(v - expected_v):.3f}")
            return False, logs
        
        logs.append("PASS: Kinematic equation v = v0 + a*t is satisfied")
        
        # 2. Verify acceleration sign consistency
        if v0 < v and self.a <= 0:
            logs.append(f"WARNING: Acceleration should be positive for speed increase")
        elif v0 > v and self.a >= 0:
            logs.append(f"WARNING: Acceleration should be negative for speed decrease")
        
        return True, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        try:
            v0 = float(self.v0)
            v = float(self.v)
            t = float(self.t)
        except (ValueError, TypeError) as e:
            logs.append(f"ERROR: Invalid numeric values - {e}")
            return False, logs
        
        # Check unit conversions match expected values
        v0_ok = math.isclose(v0, self.expected_v0, rel_tol=1e-2, abs_tol=1e-2)
        v_ok = math.isclose(v, self.expected_v, rel_tol=1e-2, abs_tol=1e-2)
        
        if not v0_ok:
            logs.append(f"FAIL: v0 = {v0} m/s, expected {self.expected_v0} m/s")
        else:
            logs.append(f"PASS: v0 conversion correct (108 km/h → {v0} m/s)")
        
        if not v_ok:
            logs.append(f"FAIL: v = {v} m/s, expected {self.expected_v} m/s")
        else:
            logs.append(f"PASS: v conversion correct (144 km/h → {v} m/s)")
        
        # Check time calculation consistency
        if v0_ok and v_ok:
            expected_t = (self.expected_v - self.expected_v0) / self.a
            t_ok = math.isclose(t, expected_t, rel_tol=1e-2, abs_tol=1e-2)
            
            if not t_ok:
                logs.append(f"FAIL: t = {t} s, expected {expected_t} s from converted speeds")
                return False, logs
            else:
                logs.append(f"PASS: Time calculation consistent t = {t} s")
        
        return v0_ok and v_ok and t_ok, logs
    
    def test_monotonicity(self):
        """Perturb initial speed and check time response."""
        logs = []
        
        try:
            v0 = float(self.v0)
            v = float(self.v)
        except (ValueError, TypeError) as e:
            logs.append(f"ERROR: Invalid numeric values - {e}")
            return False, logs
        
        # Store original
        original_v0 = v0
        
        # Test 1: Increase initial speed by 5%
        perturbed_v0 = v0 * 1.05
        t_perturbed = (v - perturbed_v0) / self.a
        
        # Time should DECREASE if initial speed increases (for same final speed)
        expected_trend = t_perturbed < float(self.t)
        
        if not expected_trend:
            logs.append(f"FAIL: Monotonicity violated")
            logs.append(f"  Original: v0={v0:.2f}, t={float(self.t):.2f}")
            logs.append(f"  Perturbed: v0={perturbed_v0:.2f}, t={t_perturbed:.2f}")
            logs.append(f"  t should decrease when v0 increases")
        else:
            logs.append(f"PASS: Time decreases when initial speed increases (as expected)")
        
        # Test 2: Decrease initial speed by 5%
        perturbed_v0 = v0 * 0.95
        t_perturbed = (v - perturbed_v0) / self.a
        
        # Time should INCREASE if initial speed decreases
        expected_trend = t_perturbed > float(self.t)
        
        if not expected_trend:
            logs.append(f"FAIL: Monotonicity violated")
            logs.append(f"  Original: v0={v0:.2f}, t={float(self.t):.2f}")
            logs.append(f"  Perturbed: v0={perturbed_v0:.2f}, t={t_perturbed:.2f}")
            logs.append(f"  t should increase when v0 decreases")
        else:
            logs.append(f"PASS: Time increases when initial speed decreases (as expected)")
        
        # Restore original value
        self.v0 = original_v0
        
        return True, logs
    
    def run_all_tests(self):
        """Execute all physics checks and return comprehensive results."""
        all_passed = True
        all_logs = ["=== Physics Audit Report ==="]
        
        tests = [
            ("Invariants Check", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                all_logs.append(f"\n--- {test_name} ---")
                passed, logs = test_func()
                all_logs.extend(logs)
                if not passed:
                    all_passed = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_passed = False
        
        # Final summary
        all_logs.append(f"\n=== Summary ===")
        all_logs.append(f"Overall: {'PASS' if all_passed else 'FAIL'}")
        all_logs.append(f"Target answer: 50 s")
        all_logs.append(f"Calculated time: {self.t} s")
        
        return all_passed, all_logs

# Optional: Example usage demonstration
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'v_0': 30.0,
                'v': 40.0,
                't': 50.0
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)