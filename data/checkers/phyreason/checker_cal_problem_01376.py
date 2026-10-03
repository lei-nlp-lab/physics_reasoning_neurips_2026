import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results
        self.m = 2e4  # kg
        self.P0 = 600e3  # W (600 kW)
        self.g = 10  # m/s^2
        
        # Extract results with safe fallbacks - ONLY from results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Get all values as floats from results dict
        self.f = self.results.get('f')
        self.F = self.results.get('F')
        self.twoF = self.results.get('2F')
        self.t1 = self.results.get('t_1')
        self.v1 = self.results.get('v_1')
        self.vm = self.results.get('v_m')
        
        self.logs = []
        
    def log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Resistance force definition
            expected_f = 0.01 * self.m * self.g  # 2000 N
            if self.f is not None:
                f_check = math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Resistance force: expected {expected_f:.0f} N, got {self.f:.0f} N", f_check)
            else:
                self.log("Missing resistance force f", False)
                return False
            
            # 2. Maximum speed from force balance (zero net force at v_m)
            # At v_m: total driving force = total resistance
            # Driving force from both motors: F_drive = 2*P0 / v_m
            # Total resistance: 8 * f
            if self.vm is not None and self.f is not None:
                F_drive = 2 * self.P0 / self.vm
                F_res_total = 8 * self.f
                balance_check = math.isclose(F_drive, F_res_total, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Force balance at v_m: F_drive={F_drive:.0f} N, F_res_total={F_res_total:.0f} N", balance_check)
            else:
                self.log("Missing v_m or f for force balance check", False)
                return False
            
            # 3. Newton's 2nd law for whole train during acceleration
            # 2F - 8f = 8m * a, where a = v1 / t1
            if all(x is not None for x in [self.twoF, self.f, self.v1, self.t1]):
                a = self.v1 / self.t1
                lhs = self.twoF - 8 * self.f
                rhs = 8 * self.m * a
                newton_check = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Newton's 2nd law (whole train): {lhs:.0f} N = {rhs:.0f} N", newton_check)
            else:
                self.log("Missing data for Newton's 2nd law (whole train)", False)
            
            # 4. Newton's 2nd law for last 3 cars (step 3)
            # F - 3f = 3m * a
            if all(x is not None for x in [self.F, self.f, self.v1, self.t1]):
                a = self.v1 / self.t1
                lhs = self.F - 3 * self.f
                rhs = 3 * self.m * a
                newton_part_check = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Newton's 2nd law (last 3 cars): {lhs:.0f} N = {rhs:.0f} N", newton_part_check)
            else:
                self.log("Missing data for Newton's 2nd law (last 3 cars)", False)
            
            return True
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results."""
        try:
            checks = []
            
            # 1. Power-velocity relationship (step 5)
            if all(x is not None for x in [self.twoF, self.v1]):
                expected_power = self.twoF * self.v1
                actual_power = 2 * self.P0
                power_check = math.isclose(expected_power, actual_power, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Power equation: {expected_power:.0f} W = {actual_power:.0f} W", power_check)
                checks.append(power_check)
            
            # 2. Kinematic equation v1 = a * t1 (step 6)
            if all(x is not None for x in [self.v1, self.t1]):
                # Compute acceleration from whole train force balance
                if self.twoF is not None and self.f is not None:
                    a_calc = (self.twoF - 8 * self.f) / (8 * self.m)
                    v1_calc = a_calc * self.t1
                    kinematic_check = math.isclose(self.v1, v1_calc, rel_tol=1e-2, abs_tol=1e-2)
                    self.log(f"Kinematic: v1={self.v1:.2f} m/s, calculated={v1_calc:.2f} m/s", kinematic_check)
                    checks.append(kinematic_check)
            
            # 3. Check F (force between 5th and 6th) from step 3
            if all(x is not None for x in [self.F, self.f, self.v1, self.t1]):
                a = self.v1 / self.t1
                F_calc = 3 * self.f + 3 * self.m * a
                F_check = math.isclose(self.F, F_calc, rel_tol=1e-2, abs_tol=1e-2)
                self.log(f"Force F: {self.F:.0f} N, calculated={F_calc:.0f} N", F_check)
                checks.append(F_check)
            
            return all(checks) if checks else False
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check trends."""
        try:
            # Save original values
            original_m = self.m
            
            # Test 1: Increase mass by 10%
            self.m = original_m * 1.1
            f_perturbed = 0.01 * self.m * self.g
            
            # At maximum speed: v_m = 2*P0 / (8*f)
            v_m_original = 2 * self.P0 / (8 * (0.01 * original_m * self.g))
            v_m_perturbed = 2 * self.P0 / (8 * f_perturbed)
            
            # Check: increased mass -> increased resistance -> decreased max speed
            mass_check = v_m_perturbed < v_m_original
            self.log(f"Mass ↑ {original_m:.0f}→{self.m:.0f} kg: v_m {v_m_original:.1f}→{v_m_perturbed:.1f} m/s (should decrease)", 
                    mass_check)
            
            # Restore original mass
            self.m = original_m
            
            # Test 2: Increase power by 10%
            original_P0 = self.P0
            self.P0 = original_P0 * 1.1
            v_m_original = 2 * original_P0 / (8 * self.f) if self.f else None
            v_m_perturbed = 2 * self.P0 / (8 * self.f) if self.f else None
            
            if v_m_original and v_m_perturbed:
                power_check = v_m_perturbed > v_m_original
                self.log(f"Power ↑ {original_P0/1e3:.0f}→{self.P0/1e3:.0f} kW: v_m {v_m_original:.1f}→{v_m_perturbed:.1f} m/s (should increase)", 
                        power_check)
            else:
                power_check = False
                self.log("Missing f for power perturbation test", False)
            
            # Restore original power
            self.P0 = original_P0
            
            return mass_check and power_check
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all checks and return (success, logs)."""
        self.logs = []
        overall_success = True
        
        try:
            self.log("=== Starting Physics Audit ===")
            
            # Test 1: Fundamental invariants
            inv_success = self.test_invariants()
            overall_success = overall_success and inv_success
            
            # Test 2: Trace consistency
            trace_success = self.test_trace_consistency()
            overall_success = overall_success and trace_success
            
            # Test 3: Monotonicity
            mono_success = self.test_monotonicity()
            overall_success = overall_success and mono_success
            
            # Final summary
            self.log(f"=== Audit Complete: {'PASS' if overall_success else 'FAIL'} ===")
            
        except Exception as e:
            self.log(f"Critical error in run_all_tests: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs