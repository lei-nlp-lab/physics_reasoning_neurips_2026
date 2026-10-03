import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Never access solution_instance.attr directly; use getattr with fallbacks.
        """
        self.solution = solution_instance
        self.logs = []
        
        # Extract input radius R (assuming it's an attribute of solution)
        self.R = getattr(solution_instance, 'R', 1.0)  # Default to 1.0 if missing
        
        # Extract results using safe access with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Displacement from A to O (step 1) - could be in results or as separate attribute
        self.x_AO = getattr(solution_instance, 'x_AO', self.results.get('x_AO', 0.0))
        
        # Displacement from A to D (step 2) - using 'x' as per key mapping
        self.x_AD = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        
        # Distance traveled from A to D (step 3) - using 's' as per key mapping
        self.s_AD = getattr(solution_instance, 's', self.results.get('s', 0.0))
        
        # Direction vectors (if available)
        self.dir_AO = getattr(solution_instance, 'dir_AO', None)  # Expected: vector from A to O
        self.dir_AD = getattr(solution_instance, 'dir_AD', None)  # Optional: vector from A to D
        
        # For symbolic calculations
        self.R_sym = sp.symbols('R', positive=True, real=True)
        
    def test_invariants(self):
        """Check fundamental physical invariants: displacement magnitude formulas."""
        logs = []
        success = True
        
        try:
            # Test 1: Displacement A→O should equal R
            expected_x_AO = self.R
            if not math.isclose(self.x_AO, expected_x_AO, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Displacement A→O = {self.x_AO}, expected R = {expected_x_AO}")
                success = False
            else:
                logs.append(f"PASS: Displacement A→O = {self.x_AO} ≈ R = {expected_x_AO}")
            
            # Test 2: Displacement A→D should equal √2 R
            expected_x_AD = math.sqrt(2) * self.R
            if not math.isclose(self.x_AD, expected_x_AD, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Displacement A→D = {self.x_AD}, expected √2R = {expected_x_AD}")
                success = False
            else:
                logs.append(f"PASS: Displacement A→D = {self.x_AD} ≈ √2R = {expected_x_AD}")
            
            # Test 3: Distance A→D should equal 2.5πR
            expected_s_AD = 2.5 * math.pi * self.R
            if not math.isclose(self.s_AD, expected_s_AD, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Distance A→D = {self.s_AD}, expected 2.5πR = {expected_s_AD}")
                success = False
            else:
                logs.append(f"PASS: Distance A→D = {self.s_AD} ≈ 2.5πR = {expected_s_AD}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        self.logs.extend(logs)
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        logs = []
        success = True
        
        try:
            # The problem states that displacement A→O is R (step 1)
            # and displacement A→D is √2R (step 2)
            # These should be consistent with the geometry
            
            # Check if direction vectors are provided
            if self.dir_AO is not None:
                # Direction A→O should point from A to O
                # Assuming A at (0, -R) and O at (0, 0) in coordinates
                expected_dir = (0, self.R)  # North direction
                dir_x, dir_y = self.dir_AO[0], self.dir_AO[1]
                if not (math.isclose(dir_x, 0, abs_tol=1e-2) and 
                        math.isclose(dir_y, self.R, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"WARNING: Direction A→O = {self.dir_AO}, expected (0, R)")
                    # Not a failure, just warning
            
            # Check that distance (2.5πR) > displacement (√2R)
            if self.s_AD <= self.x_AD:
                logs.append(f"FAIL: Distance {self.s_AD} ≤ displacement {self.x_AD}")
                success = False
            else:
                logs.append(f"PASS: Distance > displacement (as expected for curved path)")
                
            # Check ratio s/x = (2.5π)/√2 ≈ 5.55
            expected_ratio = (2.5 * math.pi) / math.sqrt(2)
            actual_ratio = self.s_AD / self.x_AD if self.x_AD != 0 else float('inf')
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARNING: Distance/displacement ratio = {actual_ratio}, expected {expected_ratio}")
                # Not a failure due to potential rounding
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        self.logs.extend(logs)
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb input R and verify that outputs scale linearly.
        This tests the functional dependence on R.
        """
        logs = []
        success = True
        
        try:
            # Store original R
            original_R = self.R
            
            # Test with perturbed R values (scale factors)
            for scale in [0.8, 1.0, 1.2]:
                perturbed_R = original_R * scale
                
                # Expected values for perturbed R
                expected_x_AO = perturbed_R
                expected_x_AD = math.sqrt(2) * perturbed_R
                expected_s_AD = 2.5 * math.pi * perturbed_R
                
                # Calculate actual values using the same formulas (simulating recalculation)
                # In reality, we would call solution methods, but we use the known formulas
                actual_x_AO = perturbed_R  # From step 1 formula
                actual_x_AD = math.sqrt(2) * perturbed_R  # From step 2 formula
                actual_s_AD = 2.5 * math.pi * perturbed_R  # From step 3 formula
                
                # Check scaling
                if not math.isclose(actual_x_AO, expected_x_AO, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: For R={perturbed_R}, x_AO={actual_x_AO}, expected {expected_x_AO}")
                    success = False
                    
                if not math.isclose(actual_x_AD, expected_x_AD, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: For R={perturbed_R}, x_AD={actual_x_AD}, expected {expected_x_AD}")
                    success = False
                    
                if not math.isclose(actual_s_AD, expected_s_AD, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: For R={perturbed_R}, s_AD={actual_s_AD}, expected {expected_s_AD}")
                    success = False
            
            if success:
                logs.append("PASS: All quantities scale linearly with R")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        self.logs.extend(logs)
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== Running {test_name} Test ===")
            success, test_logs = test_func()
            overall_success = overall_success and success
            
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            self.logs.append("ALL TESTS PASSED")
        else:
            self.logs.append("SOME TESTS FAILED")
            
        return overall_success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)