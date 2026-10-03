import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract all inputs and results safely from solution instance."""
        # Use safe attribute access with fallbacks
        self.v_0 = getattr(solution_instance, 'v_0', solution_instance.results.get('v_0', 0.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        self.v_1 = getattr(solution_instance, 'v_1', solution_instance.results.get('v_1', 0.0))
        self.v_3 = getattr(solution_instance, 'v_3', solution_instance.results.get('v_3', 0.0))
        
        # Store original for restoration
        self._original_v_0 = float(self.v_0)
        self.logs = []
    
    def test_invariants(self):
        """Check fundamental physics invariants."""
        logs = []
        success = True
        
        try:
            # 1. Sign invariant: deceleration must be negative (opposite to velocity)
            if not math.isclose(self.a, -6.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration a={self.a} should be -6 m/s² for braking")
                success = False
            elif self.a > 0:
                logs.append(f"FAIL: Acceleration a={self.a} should be negative for deceleration")
                success = False
            else:
                logs.append("PASS: Deceleration sign correct (negative)")
            
            # 2. Kinematic equation check for v_1 at t=1s
            # v_1 = v_0 + a*t1 with t1=1
            expected_v_1 = self.v_0 + self.a * 1.0
            if not math.isclose(self.v_1, expected_v_1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_1={self.v_1} violates v_1 = v_0 + a*t (expected {expected_v_1})")
                success = False
            else:
                logs.append(f"PASS: v_1={self.v_1} satisfies v_1 = v_0 + a*1")
            
            # 3. Stopping time invariant: t = -v_0/a (when v=0)
            if abs(self.a) > 1e-6:  # Avoid division by zero
                expected_t = -self.v_0 / self.a
                if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Stopping time t={self.t} violates t = -v_0/a (expected {expected_t})")
                    success = False
                else:
                    logs.append(f"PASS: Stopping time t={self.t} correct")
            
            # 4. v_3 must be 0 (car stops by 2s and stays stopped)
            if not math.isclose(self.v_3, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_3={self.v_3} should be 0 m/s after stopping")
                success = False
            else:
                logs.append("PASS: v_3=0 m/s correct")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # From steps analysis: t=2s, v_1=6m/s, v_3=0m/s
            # Check t=2s
            if not math.isclose(self.t, 2.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Inconsistent t={self.t}, should be 2.0s from step analysis")
                success = False
            
            # Check v_1=6m/s
            if not math.isclose(self.v_1, 6.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Inconsistent v_1={self.v_1}, should be 6.0 m/s from step analysis")
                success = False
            
            # Check v_3=0m/s
            if not math.isclose(self.v_3, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Inconsistent v_3={self.v_3}, should be 0.0 m/s from step analysis")
                success = False
            
            # Verify direction consistency: v_1 should have same sign as v_0
            if self.v_0 * self.v_1 < 0:
                logs.append(f"FAIL: v_1 direction inconsistent with v_0 (v_0={self.v_0}, v_1={self.v_1})")
                success = False
            
            if success:
                logs.append("PASS: All trace quantities consistent with steps analysis")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_monotonicity(self):
        """Perturb initial velocity and check expected trends."""
        logs = []
        success = True
        
        try:
            # Perturb v_0 by +10%
            original_v_0 = float(self.v_0)
            perturbed_v_0 = original_v_0 * 1.1
            
            # Expected trends from KB:
            # 1. Stopping time t = -v_0/a should increase when v_0 increases (a fixed negative)
            # 2. Velocity at 1s: v_1 = v_0 + a*1 should increase when v_0 increases
            
            # Calculate expected perturbed values using formulas
            if abs(self.a) > 1e-6:
                perturbed_t = -perturbed_v_0 / self.a
                perturbed_v_1 = perturbed_v_0 + self.a * 1.0
                
                # Original values
                original_t = -original_v_0 / self.a
                original_v_1 = original_v_0 + self.a * 1.0
                
                # Check trends
                if not (perturbed_t > original_t):
                    logs.append(f"FAIL: t should increase when v_0 increases (original t={original_t}, perturbed t={perturbed_t})")
                    success = False
                
                if not (perturbed_v_1 > original_v_1):
                    logs.append(f"FAIL: v_1 should increase when v_0 increases (original v_1={original_v_1}, perturbed v_1={perturbed_v_1})")
                    success = False
                
                if success:
                    logs.append(f"PASS: Monotonicity holds (t: {original_t:.2f}→{perturbed_t:.2f}, v_1: {original_v_1:.2f}→{perturbed_v_1:.2f})")
            else:
                logs.append("SKIP: Acceleration too small for monotonicity test")
            
            # Restore original value
            self.v_0 = original_v_0
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} Test ---")
            try:
                test_success, test_logs = test_func()
                if not test_success:
                    overall_success = False
                self.logs.extend(test_logs)
            except Exception as e:
                self.logs.append(f"CRITICAL: {test_name} test crashed: {str(e)}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        self.logs.insert(0, f"\n=== Physics Checker Results: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs