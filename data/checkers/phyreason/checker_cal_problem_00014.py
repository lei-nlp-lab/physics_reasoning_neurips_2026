import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Uses getattr with fallbacks to avoid AttributeError.
        """
        # Extract inputs with safe defaults
        self.L = getattr(solution_instance, 'L', 1600.0)
        self.v1 = getattr(solution_instance, 'v1', 80.0)
        self.m = getattr(solution_instance, 'm', 1.0e4)
        self.h = getattr(solution_instance, 'h', 100.0)
        self.v2 = getattr(solution_instance, 'v2', 100.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results from results dictionary or attributes
        self.results = getattr(solution_instance, 'results', {})
        
        self.DeltaE = getattr(solution_instance, 'DeltaE', 
                             self.results.get('DeltaE', 0.0))
        self.t = getattr(solution_instance, 't', 
                        self.results.get('t', 0.0))
        self.a = getattr(solution_instance, 'a', 
                        self.results.get('a', 0.0))
        
        # Store logs for debugging
        self.logs = []
    
    def test_invariants(self):
        """Test physical invariants: kinematics and energy conservation."""
        logs = []
        success = True
        
        try:
            # 1. Check kinematic relationships for taxiing phase
            # From uniformly accelerated motion: v_avg = (0 + v1)/2 = L/t
            v_avg_expected = (0 + self.v1) / 2
            v_avg_calculated = self.L / self.t if self.t > 0 else float('inf')
            
            if not math.isclose(v_avg_expected, v_avg_calculated, 
                               rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Average velocity mismatch. Expected {v_avg_expected:.2f}, got {v_avg_calculated:.2f}")
                success = False
            else:
                logs.append("PASS: Average velocity consistent with kinematics")
            
            # 2. Check acceleration from v1 = a*t
            a_calculated = self.v1 / self.t if self.t > 0 else float('inf')
            if not math.isclose(self.a, a_calculated, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Acceleration mismatch. From v1/t: {a_calculated:.2f}, result: {self.a:.2f}")
                success = False
            else:
                logs.append("PASS: Acceleration consistent with v1 = a*t")
            
            # 3. Check displacement formula: L = 0.5*a*t^2
            L_calculated = 0.5 * self.a * self.t**2
            if not math.isclose(self.L, L_calculated, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Displacement mismatch. From 0.5*a*t^2: {L_calculated:.2f}, given: {self.L}")
                success = False
            else:
                logs.append("PASS: Displacement consistent with kinematics")
            
            # 4. Check energy formula: ΔE = 0.5*m*v2^2 + mgh - 0.5*m*v1^2
            DeltaE_calculated = (0.5 * self.m * self.v2**2 + 
                               self.m * self.g * self.h - 
                               0.5 * self.m * self.v1**2)
            
            if not math.isclose(self.DeltaE, DeltaE_calculated, 
                               rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Energy change mismatch. Calculated: {DeltaE_calculated:.2e}, result: {self.DeltaE:.2e}")
                success = False
            else:
                logs.append("PASS: Energy change formula verified")
            
            # 5. Symbolic check using sympy
            try:
                v1_s, t_s, L_s, a_s, m_s, v2_s, h_s, g_s = sp.symbols('v1 t L a m v2 h g')
                
                # Kinematic equation: L = (v1/2)*t
                eq1 = sp.Eq(L_s, (v1_s/2) * t_s)
                eq1_check = eq1.subs({L_s: self.L, v1_s: self.v1, t_s: self.t})
                if not eq1_check:
                    logs.append(f"WARNING: Symbolic kinematic check failed for L = (v1/2)*t")
                
                # Energy equation
                DeltaE_s = 0.5*m_s*v2_s**2 + m_s*g_s*h_s - 0.5*m_s*v1_s**2
                eq2 = sp.Eq(self.DeltaE, DeltaE_s.subs({
                    m_s: self.m, v2_s: self.v2, h_s: self.h, 
                    g_s: self.g, v1_s: self.v1
                }))
                if not math.isclose(float(self.DeltaE), float(DeltaE_s.subs({
                    m_s: self.m, v2_s: self.v2, h_s: self.h, 
                    g_s: self.g, v1_s: self.v1
                })), rel_tol=0.02, abs_tol=0.02):
                    logs.append("WARNING: Symbolic energy check failed")
                
            except Exception as e:
                logs.append(f"Symbolic check skipped due to error: {str(e)}")
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Step 1: Check taxiing time t = 2L/v1
            t_expected = 2 * self.L / self.v1
            if not math.isclose(self.t, t_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Time inconsistent. Expected t=2L/v1={t_expected:.2f}, got {self.t:.2f}")
                success = False
            else:
                logs.append(f"PASS: Time t={self.t:.2f}s matches 2L/v1")
            
            # Step 2: Check acceleration a = v1/t
            a_expected = self.v1 / t_expected if t_expected > 0 else float('inf')
            if not math.isclose(self.a, a_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Acceleration inconsistent. Expected a=v1/t={a_expected:.2f}, got {self.a:.2f}")
                success = False
            else:
                logs.append(f"PASS: Acceleration a={self.a:.2f}m/s^2 matches v1/t")
            
            # Step 3 & 4: Check ΔE calculation
            DeltaE_expected = (0.5 * self.m * self.v2**2 + 
                             self.m * self.g * self.h - 
                             0.5 * self.m * self.v1**2)
            
            if not math.isclose(self.DeltaE, DeltaE_expected, 
                               rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: ΔE inconsistent. Expected {DeltaE_expected:.2e}J, got {self.DeltaE:.2e}J")
                success = False
            else:
                logs.append(f"PASS: ΔE={self.DeltaE:.2e}J matches energy formula")
            
            # Verify all steps lead to the target answers
            target_a = 2.0
            target_t = 40.0
            target_DeltaE = 2.8e7
            
            if not (math.isclose(self.a, target_a, rel_tol=0.02, abs_tol=0.02) and
                    math.isclose(self.t, target_t, rel_tol=0.02, abs_tol=0.02)):
                logs.append(f"FAIL: Kinematic results don't match targets: a≈{self.a:.2f} vs {target_a}, t≈{self.t:.2f} vs {target_t}")
                success = False
            
            if not math.isclose(self.DeltaE, target_DeltaE, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: ΔE doesn't match target: {self.DeltaE:.2e} vs {target_DeltaE:.2e}")
                success = False
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Test how results change with perturbation of input parameters.
        Perturb mass m and check ΔE changes proportionally.
        """
        logs = []
        success = True
        
        try:
            # Save original values
            original_m = self.m
            original_DeltaE = self.DeltaE
            
            # Perturb mass by +1%
            perturbation = 0.01
            self.m = original_m * (1 + perturbation)
            
            # Recalculate ΔE using the formula
            DeltaE_perturbed = (0.5 * self.m * self.v2**2 + 
                              self.m * self.g * self.h - 
                              0.5 * self.m * self.v1**2)
            
            # Expected change: ΔE should scale linearly with m
            expected_ratio = 1 + perturbation
            actual_ratio = DeltaE_perturbed / original_DeltaE if original_DeltaE != 0 else float('inf')
            
            # Check if ratio is close to expected (within 2%)
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Monotonicity test failed. Expected ΔE ratio {expected_ratio:.4f}, got {actual_ratio:.4f}")
                success = False
            else:
                logs.append(f"PASS: ΔE scales linearly with mass (ratio: {actual_ratio:.4f})")
            
            # Additional test: perturb v1 (should affect ΔE negatively)
            original_v1 = self.v1
            self.v1 = original_v1 * 1.01  # +1%
            
            DeltaE_v1_perturbed = (0.5 * self.m * self.v2**2 + 
                                 self.m * self.g * self.h - 
                                 0.5 * self.m * self.v1**2)
            
            # Since ΔE = ... - 0.5*m*v1^2, increasing v1 should decrease ΔE
            if not (DeltaE_v1_perturbed < original_DeltaE):
                logs.append("WARNING: Increasing v1 should decrease ΔE, but didn't")
            
            # Restore original values
            self.m = original_m
            self.v1 = original_v1
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            # Try to restore values even on error
            try:
                self.m = original_m
                self.v1 = original_v1
            except:
                pass
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                all_logs.append(f"Test {test_name} crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        return overall_success, all_logs