import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Safely extract all expected results with defaults
        self.v1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0))
        self.v2 = getattr(solution_instance, 'v_2', self.results.get('v_2', 0.0))
        self.delta_x = getattr(solution_instance, 'delta_x', self.results.get('delta_x', 0.0))
        self.K = getattr(solution_instance, 'K', self.results.get('K', 0.0))
        
        # Physical constants from problem statement
        self.g = 10.0  # m/s²
        
        # Time intervals from steps analysis
        self.t1 = 0.1  # s (first 0.1s interval)
        self.t2 = 0.2  # s (second 0.1s interval, total time)
        
        # Store logs for all tests
        self.logs = []
        
    def test_invariants(self):
        """Verify kinematic equations and conservation laws."""
        logs = []
        success = True
        
        try:
            # Test 1: v1 = g*t1 (free fall from rest)
            expected_v1 = self.g * self.t1
            if math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ v1 correct: {self.v1} ≈ g*t1 = {expected_v1} m/s")
            else:
                logs.append(f"✗ v1 incorrect: {self.v1} ≠ g*t1 = {expected_v1} m/s")
                success = False
            
            # Test 2: Δx = 0.5*g*(t2² - t1²) for second interval
            expected_delta_x = 0.5 * self.g * (self.t2**2 - self.t1**2)
            if math.isclose(self.delta_x, expected_delta_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Δx correct: {self.delta_x} ≈ 0.5*g*(t2²-t1²) = {expected_delta_x} m")
            else:
                logs.append(f"✗ Δx incorrect: {self.delta_x} ≠ 0.5*g*(t2²-t1²) = {expected_delta_x} m")
                success = False
            
            # Test 3: K = v2/v1 = √2 (from free fall equations)
            if self.v1 != 0:
                K_calculated = self.v2 / self.v1
            else:
                K_calculated = 0.0
                logs.append("⚠ v1 is zero, cannot calculate K ratio")
            
            # Check both the computed K and if it's approximately √2
            if math.isclose(self.K, math.sqrt(2), rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ K correct: {self.K} ≈ √2 = {math.sqrt(2)}")
            else:
                logs.append(f"✗ K incorrect: {self.K} ≠ √2 = {math.sqrt(2)}")
                success = False
            
            # Verify v1² = 2gh1 and v2² = 2gh2 implies v2²/v1² = 2
            # This is equivalent to K² = 2
            if math.isclose(self.K**2, 2.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ K² = 2 relation holds: {self.K**2} ≈ 2")
            else:
                logs.append(f"✗ K² = 2 relation fails: {self.K**2} ≠ 2")
                success = False
            
        except Exception as e:
            logs.append(f"✗ Exception in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check that v1 from step_1 is used consistently
            expected_v1 = self.g * self.t1
            if not math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✗ v1 inconsistency: stored {self.v1} ≠ kinematic {expected_v1}")
                success = False
            else:
                logs.append(f"✓ v1 consistent with kinematics")
            
            # Check that v2 can be derived from v1 and K
            if self.v1 != 0:
                v2_from_K = self.K * self.v1
                if math.isclose(self.v2, v2_from_K, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ v2 consistent: {self.v2} ≈ K*v1 = {v2_from_K}")
                else:
                    logs.append(f"✗ v2 inconsistent: {self.v2} ≠ K*v1 = {v2_from_K}")
                    success = False
            
            # Verify that all three target answers match
            target_v1 = 1.0
            target_delta_x = 0.15
            target_K = math.sqrt(2)
            
            all_targets_match = (
                math.isclose(self.v1, target_v1, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.delta_x, target_delta_x, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.K, target_K, rel_tol=1e-2, abs_tol=1e-2)
            )
            
            if all_targets_match:
                logs.append("✓ All results match target answers")
            else:
                logs.append("✗ Some results don't match target answers")
                success = False
                
        except Exception as e:
            logs.append(f"✗ Exception in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input parameters and check physical trends."""
        logs = []
        success = True
        
        try:
            # Original values for restoration
            original_g = self.g
            
            # Perturbation 1: Increase g by 10%
            self.g = original_g * 1.1
            
            # Recalculate using kinematic formulas
            perturbed_v1 = self.g * self.t1
            perturbed_delta_x = 0.5 * self.g * (self.t2**2 - self.t1**2)
            
            # Check monotonicity: v1 should increase with g
            if perturbed_v1 > self.v1:
                logs.append(f"✓ v1 increases with g: {perturbed_v1} > {self.v1}")
            else:
                logs.append(f"✗ v1 monotonicity violation: {perturbed_v1} ≤ {self.v1}")
                success = False
            
            # Δx should increase with g
            if perturbed_delta_x > self.delta_x:
                logs.append(f"✓ Δx increases with g: {perturbed_delta_x} > {self.delta_x}")
            else:
                logs.append(f"✗ Δx monotonicity violation: {perturbed_delta_x} ≤ {self.delta_x}")
                success = False
            
            # K should be independent of g (from free fall equations)
            # K = √(2gh2)/√(2gh1) = √(h2/h1) independent of g
            logs.append(f"✓ K remains {self.K} (independent of g)")
            
            # Restore original g
            self.g = original_g
            
            # Perturbation 2: Change time interval t1
            original_t1 = self.t1
            self.t1 = original_t1 * 1.2  # Increase by 20%
            
            perturbed_v1_new = self.g * self.t1
            
            # v1 should increase with t1
            if perturbed_v1_new > self.v1:
                logs.append(f"✓ v1 increases with t1: {perturbed_v1_new} > {self.v1}")
            else:
                logs.append(f"✗ v1 monotonicity violation with t1: {perturbed_v1_new} ≤ {self.v1}")
                success = False
            
            # Restore original t1
            self.t1 = original_t1
            
            # Perturbation 3: Check that v2 > v1 (since K = √2 > 1)
            if self.v2 > self.v1:
                logs.append(f"✓ v2 > v1 as expected: {self.v2} > {self.v1}")
            else:
                logs.append(f"✗ v2 ≤ v1: {self.v2} ≤ {self.v1}")
                success = False
            
        except Exception as e:
            logs.append(f"✗ Exception in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success status and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n{'='*60}")
            self.logs.append(f"Running {test_name}")
            self.logs.append(f"{'='*60}")
            
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                if success:
                    self.logs.append(f"✅ {test_name} PASSED")
                else:
                    self.logs.append(f"❌ {test_name} FAILED")
            except Exception as e:
                self.logs.append(f"💥 {test_name} CRASHED: {str(e)}")
                overall_success = False
        
        self.logs.append(f"\n{'='*60}")
        self.logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        self.logs.append(f"{'='*60}")
        
        return overall_success, self.logs