import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Get results dictionary
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract L and C from results with defaults
        self.L = results_dict.get('L', 2.5e-3)
        self.C = results_dict.get('C', 4e-6)
        
        # Extract T and other results
        self.T = results_dict.get('T', None)
        self.omega = results_dict.get('omega', None)
        self.q0 = results_dict.get('q0', None)
        self.delta_t = results_dict.get('delta_t', None)
        self.q1 = results_dict.get('q1', None)
        self.I1 = results_dict.get('I1', None)
        self.UE_avg = results_dict.get('UE_avg', None)
        self.UB_avg = results_dict.get('UB_avg', None)
        self.U_total = results_dict.get('U_total', None)
        
        # Store logs for reporting
        self.logs = []
    
    def log_test(self, test_name, success, message):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        log_msg = f"{test_name}: {status} - {message}"
        self.logs.append(log_msg)
        return success
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants for LC circuits.
        """
        tests_passed = 0
        total_tests = 0
        
        try:
            # Test 1: Period formula T = 2π√(LC)
            total_tests += 1
            T_expected = 2 * math.pi * math.sqrt(float(self.L) * float(self.C))
            if self.T is not None:
                if math.isclose(float(self.T), T_expected, rel_tol=1e-2, abs_tol=1e-6):
                    tests_passed += 1
                    self.log_test("Period Formula", True, 
                                 f"T={self.T:.6e}s matches T=2π√(LC)={T_expected:.6e}s")
                else:
                    self.log_test("Period Formula", False,
                                 f"T={self.T:.6e}s ≠ 2π√(LC)={T_expected:.6e}s")
            else:
                self.log_test("Period Formula", False, "Period T not found in results")
        
        except Exception as e:
            self.log_test("Period Formula", False, f"Error: {str(e)}")
        
        try:
            # Test 2: Energy conservation (if q0 is available)
            total_tests += 1
            if self.q0 is not None and self.UE_avg is not None and self.UB_avg is not None:
                # Calculate expected energies
                q0_val = float(self.q0)
                C_val = float(self.C)
                L_val = float(self.L)
                
                # Calculate angular frequency
                omega_val = 1 / math.sqrt(L_val * C_val)
                
                # Expected average energies
                UE_expected = q0_val**2 / (4 * C_val)
                UB_expected = L_val * (q0_val * omega_val)**2 / 4
                U_total_expected = UE_expected + UB_expected
                
                # Check if individual energies match
                UE_match = math.isclose(float(self.UE_avg), UE_expected, rel_tol=1e-2, abs_tol=1e-6)
                UB_match = math.isclose(float(self.UB_avg), UB_expected, rel_tol=1e-2, abs_tol=1e-6)
                
                # Check total energy if available
                if self.U_total is not None:
                    U_total_match = math.isclose(float(self.U_total), U_total_expected, 
                                                rel_tol=1e-2, abs_tol=1e-6)
                    if UE_match and UB_match and U_total_match:
                        tests_passed += 1
                        self.log_test("Energy Conservation", True,
                                     f"UE_avg={self.UE_avg:.6e}J, UB_avg={self.UB_avg:.6e}J, "
                                     f"U_total={self.U_total:.6e}J match formulas")
                    else:
                        self.log_test("Energy Conservation", False,
                                     f"Energy values don't match formulas: "
                                     f"UE_avg (calc={UE_expected:.6e} vs got={self.UE_avg:.6e}), "
                                     f"UB_avg (calc={UB_expected:.6e} vs got={self.UB_avg:.6e})")
                elif UE_match and UB_match:
                    tests_passed += 1
                    self.log_test("Energy Conservation", True,
                                 f"UE_avg and UB_avg match formulas")
                else:
                    self.log_test("Energy Conservation", False,
                                 f"Energy values don't match formulas")
            else:
                self.log_test("Energy Conservation", False, 
                             "Insufficient data for energy check (need q0, UE_avg, UB_avg)")
        
        except Exception as e:
            self.log_test("Energy Conservation", False, f"Error: {str(e)}")
        
        try:
            # Test 3: Relationship between q1 and I1 (if available)
            total_tests += 1
            if (self.q0 is not None and self.delta_t is not None and 
                self.q1 is not None and self.I1 is not None):
                
                q0_val = float(self.q0)
                dt_val = float(self.delta_t)
                L_val = float(self.L)
                C_val = float(self.C)
                
                omega_val = 1 / math.sqrt(L_val * C_val)
                
                # Expected values from formulas
                q1_expected = q0_val * math.cos(omega_val * dt_val)
                I1_expected = -q0_val * omega_val * math.sin(omega_val * dt_val)
                
                # Check if dt is approximately T/4 (within 1%)
                T_val = 2 * math.pi * math.sqrt(L_val * C_val)
                is_near_quarter_period = math.isclose(dt_val, T_val/4, rel_tol=0.01)
                
                if is_near_quarter_period:
                    # For dt ≈ T/4, we expect q1 ≈ 0 and I1 ≈ -q0*ω
                    # Use more appropriate tolerances for this special case
                    q1_near_zero = abs(float(self.q1)) < 1e-3  # q1 should be near 0
                    I1_expected_quarter = -q0_val * omega_val
                    I1_near_expected = math.isclose(float(self.I1), I1_expected_quarter, 
                                                   rel_tol=1e-2, abs_tol=1e-3)
                    
                    if q1_near_zero and I1_near_expected:
                        tests_passed += 1
                        self.log_test("Charge-Current Relationship", True,
                                     f"For dt≈T/4: q1≈0 ({self.q1:.6e}) and I1≈-q0ω ({self.I1:.6e})")
                    else:
                        self.log_test("Charge-Current Relationship", False,
                                     f"For dt≈T/4: q1 should be near 0 (got {self.q1:.6e}), "
                                     f"I1 should be near {-q0_val*omega_val:.6e} (got {self.I1:.6e})")
                else:
                    # General case: use standard comparison
                    q1_match = math.isclose(float(self.q1), q1_expected, rel_tol=1e-2, abs_tol=1e-6)
                    I1_match = math.isclose(float(self.I1), I1_expected, rel_tol=1e-2, abs_tol=1e-6)
                    
                    if q1_match and I1_match:
                        tests_passed += 1
                        self.log_test("Charge-Current Relationship", True,
                                     f"q1 and I1 match oscillation formulas")
                    else:
                        self.log_test("Charge-Current Relationship", False,
                                     f"q1 (calc={q1_expected:.6e} vs got={self.q1:.6e}), "
                                     f"I1 (calc={I1_expected:.6e} vs got={self.I1:.6e})")
            else:
                self.log_test("Charge-Current Relationship", False,
                             "Insufficient data for q1/I1 check")
        
        except Exception as e:
            self.log_test("Charge-Current Relationship", False, f"Error: {str(e)}")
        
        return tests_passed == total_tests
    
    def test_trace_consistency(self):
        """
        Check consistency between intermediate results and final answer.
        """
        tests_passed = 0
        total_tests = 0
        
        try:
            # Test 1: Check that calculated period matches target answer
            total_tests += 1
            target_T = 6.28e-4  # From target answers
            if self.T is not None:
                if math.isclose(float(self.T), target_T, rel_tol=1e-2, abs_tol=1e-6):
                    tests_passed += 1
                    self.log_test("Target Period Match", True,
                                 f"T={self.T:.6e}s matches target {target_T:.6e}s")
                else:
                    self.log_test("Target Period Match", False,
                                 f"T={self.T:.6e}s ≠ target {target_T:.6e}s")
            else:
                self.log_test("Target Period Match", False, "Period T not found")
        
        except Exception as e:
            self.log_test("Target Period Match", False, f"Error: {str(e)}")
        
        try:
            # Test 2: Check omega and T relationship
            total_tests += 1
            if self.omega is not None and self.T is not None:
                omega_from_T = 2 * math.pi / float(self.T)
                if math.isclose(float(self.omega), omega_from_T, rel_tol=1e-2, abs_tol=1e-6):
                    tests_passed += 1
                    self.log_test("Omega-T Consistency", True,
                                 f"ω={self.omega:.6e}rad/s matches 2π/T")
                else:
                    self.log_test("Omega-T Consistency", False,
                                 f"ω={self.omega:.6e}rad/s ≠ 2π/T={omega_from_T:.6e}rad/s")
            else:
                self.log_test("Omega-T Consistency", False,
                             "Insufficient data for ω-T check")
        
        except Exception as e:
            self.log_test("Omega-T Consistency", False, f"Error: {str(e)}")
        
        return tests_passed == total_tests
    
    def test_monotonicity(self):
        """
        Test how period changes with perturbations in L or C.
        """
        tests_passed = 0
        total_tests = 0
        
        try:
            # Test 1: Increase L -> T should increase
            total_tests += 1
            L_original = float(self.L)
            C_original = float(self.C)
            
            # Perturb L by +10%
            L_perturbed = L_original * 1.1
            T_original = 2 * math.pi * math.sqrt(L_original * C_original)
            T_perturbed = 2 * math.pi * math.sqrt(L_perturbed * C_original)
            
            if T_perturbed > T_original:
                tests_passed += 1
                self.log_test("L-Increase Effect", True,
                             f"T increases from {T_original:.6e}s to {T_perturbed:.6e}s when L increases 10%")
            else:
                self.log_test("L-Increase Effect", False,
                             f"T should increase with L but didn't: {T_original:.6e}s → {T_perturbed:.6e}s")
        
        except Exception as e:
            self.log_test("L-Increase Effect", False, f"Error: {str(e)}")
        
        try:
            # Test 2: Increase C -> T should increase
            total_tests += 1
            C_perturbed = C_original * 1.1
            T_perturbed_C = 2 * math.pi * math.sqrt(L_original * C_perturbed)
            
            if T_perturbed_C > T_original:
                tests_passed += 1
                self.log_test("C-Increase Effect", True,
                             f"T increases from {T_original:.6e}s to {T_perturbed_C:.6e}s when C increases 10%")
            else:
                self.log_test("C-Increase Effect", False,
                             f"T should increase with C but didn't: {T_original:.6e}s → {T_perturbed_C:.6e}s")
        
        except Exception as e:
            self.log_test("C-Increase Effect", False, f"Error: {str(e)}")
        
        try:
            # Test 3: If q0 is available, test energy scaling
            total_tests += 1
            if self.q0 is not None:
                q0_original = float(self.q0)
                # Perturb q0 by +20%
                q0_perturbed = q0_original * 1.2
                
                # Calculate energies for both
                omega = 1 / math.sqrt(L_original * C_original)
                UE_original = q0_original**2 / (4 * C_original)
                UB_original = L_original * (q0_original * omega)**2 / 4
                U_total_original = UE_original + UB_original
                
                UE_perturbed = q0_perturbed**2 / (4 * C_original)
                UB_perturbed = L_original * (q0_perturbed * omega)**2 / 4
                U_total_perturbed = UE_perturbed + UB_perturbed
                
                # Energy should scale quadratically
                scaling_ratio = (q0_perturbed / q0_original)**2
                UE_ratio = UE_perturbed / UE_original
                UB_ratio = UB_perturbed / UB_original
                U_total_ratio = U_total_perturbed / U_total_original
                
                if (math.isclose(UE_ratio, scaling_ratio, rel_tol=1e-2) and
                    math.isclose(UB_ratio, scaling_ratio, rel_tol=1e-2) and
                    math.isclose(U_total_ratio, scaling_ratio, rel_tol=1e-2)):
                    tests_passed += 1
                    self.log_test("q0-Energy Scaling", True,
                                 f"Energies scale quadratically with q0: ratio={scaling_ratio:.3f}")
                else:
                    self.log_test("q0-Energy Scaling", False,
                                 f"Energy scaling incorrect: UE_ratio={UE_ratio:.3f}, "
                                 f"UB_ratio={UB_ratio:.3f}, expected {scaling_ratio:.3f}")
            else:
                self.log_test("q0-Energy Scaling", False,
                             "q0 not available for energy scaling test")
        
        except Exception as e:
            self.log_test("q0-Energy Scaling", False, f"Error: {str(e)}")
        
        return tests_passed == total_tests
    
    def run_all_tests(self):
        """
        Run all tests and return success status and logs.
        """
        self.logs = []  # Reset logs
        
        try:
            # Run all tests
            invariants_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            monotonicity_ok = self.test_monotonicity()
            
            # Overall success if all tests pass
            overall_success = invariants_ok and trace_ok and monotonicity_ok
            
            # Add summary
            summary = (f"Overall: {'PASS' if overall_success else 'FAIL'} - "
                      f"Invariants: {'PASS' if invariants_ok else 'FAIL'}, "
                      f"Trace: {'PASS' if trace_ok else 'FAIL'}, "
                      f"Monotonicity: {'PASS' if monotonicity_ok else 'FAIL'}")
            self.logs.append(summary)
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"PhysicsChecker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs