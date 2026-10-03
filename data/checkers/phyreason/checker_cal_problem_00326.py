import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # ALWAYS use hardcoded correct values for constants
        # DO NOT read these from solution_instance
        self.Q = 4.0e-8  # Correct value from problem statement
        self.q = 1.6e-9  # Correct value from problem statement
        self.r = 0.30    # Correct value from problem statement
        self.k = 9.0e9   # Correct value from problem statement
        
        # Initialize F and E to 0.0
        self.F = 0.0
        self.E = 0.0
        
        # Try to read ONLY F and E from solution_instance
        # DO NOT read any other attributes
        try:
            # ONLY look for results dictionary
            if hasattr(solution_instance, 'results'):
                results_dict = solution_instance.results
                if isinstance(results_dict, dict):
                    # ONLY read 'F' and 'E' keys
                    if 'F' in results_dict:
                        val = results_dict['F']
                        if isinstance(val, (int, float)):
                            self.F = float(val)
                    if 'E' in results_dict:
                        val = results_dict['E']
                        if isinstance(val, (int, float)):
                            self.E = float(val)
        except (ValueError, TypeError, AttributeError):
            # If conversion fails, keep default values
            self.F = 0.0
            self.E = 0.0
        
        # For monotonicity perturbation - store original Q and r
        self._original_Q = self.Q
        self._original_r = self.r
        
    def compute_expected_F(self):
        """Calculate expected force magnitude using Coulomb's Law"""
        return self.k * abs(self.Q * self.q) / (self.r ** 2)
    
    def compute_expected_E(self):
        """Calculate expected electric field using Coulomb's Law"""
        return self.k * abs(self.Q) / (self.r ** 2)
    
    def test_invariants(self):
        """Check fundamental physics laws"""
        logs = []
        success = True
        
        try:
            # Test 1: Coulomb's Law - use hardcoded correct values
            expected_F = self.compute_expected_F()
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Coulomb's Law violation. Got F={self.F}, expected {expected_F}")
                success = False
            else:
                logs.append(f"PASS: Coulomb's Law satisfied (F={self.F:.2e} N)")
            
            # Test 2: Electric field definition (E = F/q)
            if abs(self.q) > 1e-12:  # Avoid division by near-zero
                E_from_F = abs(self.F / self.q)
                if not math.isclose(self.E, E_from_F, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: E ≠ F/q. Got E={self.E}, expected {E_from_F}")
                    success = False
                else:
                    logs.append(f"PASS: E = F/q relationship holds (E={self.E:.2e} N/C)")
            else:
                logs.append("SKIP: q is too small for E = F/q test")
            
            # Test 3: Alternative E calculation (E = kQ/r²)
            expected_E = self.compute_expected_E()
            if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E calculation mismatch. Got {self.E}, expected {expected_E}")
                success = False
            else:
                logs.append(f"PASS: Direct E calculation matches (E={self.E:.2e} N/C)")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate steps match final results"""
        logs = []
        success = True
        
        try:
            # From steps analysis: step_1 F = 6.4e-6, step_2 E = 4000
            expected_F_step1 = 6.4e-6
            expected_E_step2 = 4000.0
            
            # Check F matches step_1
            if not math.isclose(self.F, expected_F_step1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 1 F mismatch. Got {self.F:.2e}, expected {expected_F_step1:.2e}")
                success = False
            else:
                logs.append(f"PASS: Step 1 F matches (F={self.F:.2e} N)")
            
            # Check E matches step_2
            if not math.isclose(self.E, expected_E_step2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 2 E mismatch. Got {self.E:.2e}, expected {expected_E_step2:.2e}")
                success = False
            else:
                logs.append(f"PASS: Step 2 E matches (E={self.E:.2e} N/C)")
            
            # Check consistency between steps: E should be F/q
            if abs(self.q) > 1e-12:
                E_calculated = self.F / self.q
                if not math.isclose(self.E, E_calculated, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Step 2 E doesn't match Step 1 F/q. {self.E} vs {E_calculated}")
                    success = False
                else:
                    logs.append("PASS: Step 2 E consistent with Step 1 F")
            else:
                logs.append("SKIP: q is too small for consistency check")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test how results change with input perturbations"""
        logs = []
        success = True
        
        try:
            # Store original values for comparison
            original_F = self.F
            original_E = self.E
            
            # Test 1: Increase Q by 50%, F and E should increase proportionally
            # Use a temporary Q value for calculation
            temp_Q = self._original_Q * 1.5
            perturbed_F_Q = self.k * abs(temp_Q * self.q) / (self.r ** 2)
            perturbed_E_Q = self.k * abs(temp_Q) / (self.r ** 2)
            
            # Expected ratios
            expected_F_ratio = 1.5
            expected_E_ratio = 1.5
            
            # Calculate actual ratios (protect against division by zero)
            if abs(original_F) > 1e-12:
                actual_F_ratio = perturbed_F_Q / original_F
            else:
                actual_F_ratio = 0.0
                logs.append("WARNING: Original F is zero, cannot compute ratio")
                
            if abs(original_E) > 1e-12:
                actual_E_ratio = perturbed_E_Q / original_E
            else:
                actual_E_ratio = 0.0
                logs.append("WARNING: Original E is zero, cannot compute ratio")
            
            if abs(original_F) > 1e-12 and not math.isclose(actual_F_ratio, expected_F_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F doesn't scale linearly with Q. Ratio: {actual_F_ratio:.3f}, expected {expected_F_ratio}")
                success = False
            elif abs(original_F) > 1e-12:
                logs.append(f"PASS: F scales linearly with Q (ratio: {actual_F_ratio:.3f})")
            
            if abs(original_E) > 1e-12 and not math.isclose(actual_E_ratio, expected_E_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E doesn't scale linearly with Q. Ratio: {actual_E_ratio:.3f}, expected {expected_E_ratio}")
                success = False
            elif abs(original_E) > 1e-12:
                logs.append(f"PASS: E scales linearly with Q (ratio: {actual_E_ratio:.3f})")
            
            # Test 2: Increase r by 50%, F and E should decrease with inverse square
            temp_r = self._original_r * 1.5
            perturbed_F_r = self.k * abs(self.Q * self.q) / (temp_r ** 2)
            perturbed_E_r = self.k * abs(self.Q) / (temp_r ** 2)
            
            # Expected inverse square: 1/(1.5²) = 1/2.25 ≈ 0.444
            expected_inverse_square = 1.0 / (1.5 ** 2)
            
            # Calculate actual ratios (protect against division by zero)
            if abs(original_F) > 1e-12:
                actual_F_ratio_r = perturbed_F_r / original_F
            else:
                actual_F_ratio_r = 0.0
                
            if abs(original_E) > 1e-12:
                actual_E_ratio_r = perturbed_E_r / original_E
            else:
                actual_E_ratio_r = 0.0
            
            if abs(original_F) > 1e-12 and not math.isclose(actual_F_ratio_r, expected_inverse_square, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F doesn't follow inverse square law. Ratio: {actual_F_ratio_r:.3f}, expected {expected_inverse_square:.3f}")
                success = False
            elif abs(original_F) > 1e-12:
                logs.append(f"PASS: F follows inverse square law (ratio: {actual_F_ratio_r:.3f})")
            
            if abs(original_E) > 1e-12 and not math.isclose(actual_E_ratio_r, expected_inverse_square, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E doesn't follow inverse square law. Ratio: {actual_E_ratio_r:.3f}, expected {expected_inverse_square:.3f}")
                success = False
            elif abs(original_E) > 1e-12:
                logs.append(f"PASS: E follows inverse square law (ratio: {actual_E_ratio_r:.3f})")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all checks and return comprehensive results"""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_func in [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRASH in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall result: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs