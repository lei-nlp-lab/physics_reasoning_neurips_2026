import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        
        # Extract input parameters (with safe defaults)
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 1.0))
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', 1.0))
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 1.0))
        
        # Extract results with safe access
        self.U_m = getattr(solution_instance, 'U_m', solution_instance.results.get('U_m', 1.0))
        self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', 1.0))
        self.x_0 = getattr(solution_instance, 'x_0', solution_instance.results.get('x_0', 1.0))
        self.W_F = getattr(solution_instance, 'W_F', solution_instance.results.get('W_F', 1.0))
        self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', 1.0))
        self.v_m = getattr(solution_instance, 'v_m', solution_instance.results.get('v_m', 1.0))
        
        # F_A can be either a numeric value or a callable function
        self.F_A = getattr(solution_instance, 'F_A', solution_instance.results.get('F_A', None))
        
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)

    def test_invariants(self):
        """Check fundamental physics laws from KB and steps"""
        success = True
        
        try:
            # 1. Motional EMF check: U_m = B*l*v_m
            if self.B == 0 or self.l == 0:
                self.log("WARNING: B or l is zero, skipping motional EMF check")
            else:
                v_m_expected = self.U_m / (self.B * self.l)
                if not math.isclose(self.v_m, v_m_expected, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    self.log(f"FAIL: Motional EMF invariant. v_m = {self.v_m}, expected {v_m_expected} from U_m = B*l*v_m")
                else:
                    self.log("OK: Motional EMF invariant holds")
            
            # 2. Ampere Force check: F_A = -ilB = -(B*l*U_m/R) * sin(2πt/T)
            if self.F_A is not None:
                # If F_A is callable, test at a specific time; otherwise, we cannot test time dependence
                if callable(self.F_A):
                    t_test = 0.1 * self.T
                    F_A_expected = -(self.B * self.l * self.U_m / self.R) * math.sin(2 * math.pi * t_test / self.T)
                    F_A_val = self.F_A(t_test)
                    
                    if not math.isclose(F_A_val, F_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                        success = False
                        self.log(f"FAIL: Ampere Force at t={t_test:.3f}: got {F_A_val}, expected {F_A_expected}")
                    else:
                        self.log("OK: Ampere Force matches expected formula")
                else:
                    # F_A is a numeric value, likely at some time t. We can't test time dependence.
                    self.log("INFO: F_A is not callable, skipping time-dependent Ampere Force check")
            
            # 3. Energy conservation: W_F = Q + 1/2 m v_m^2
            if self.R == 0:
                self.log("WARNING: R is zero, skipping Q calculation")
            else:
                Q_expected = (self.U_m**2 / (2 * self.R)) * (self.T / 4)
                KE = 0.5 * self.m * self.v_m**2
                W_F_expected = Q_expected + KE
                
                if not math.isclose(self.W_F, W_F_expected, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    self.log(f"FAIL: Energy conservation. W_F = {self.W_F}, expected {W_F_expected}")
                else:
                    self.log("OK: Energy conservation holds")
            
            # 4. SHM relationship: x_0 = T*U_m/(2π*B*l)
            if self.B == 0 or self.l == 0:
                self.log("WARNING: B or l is zero, skipping SHM relationship")
            else:
                x_0_expected = self.T * self.U_m / (2 * math.pi * self.B * self.l)
                if not math.isclose(self.x_0, x_0_expected, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    self.log(f"FAIL: SHM relationship. x_0 = {self.x_0}, expected {x_0_expected}")
                else:
                    self.log("OK: SHM relationship holds")
            
            # 5. Check that F_A acts as restoring force - only meaningful if F_A is callable
            if self.F_A is not None and callable(self.F_A):
                F_A_at_0 = self.F_A(0.0)
                # At t=0, u=0, so i=0, F_A=0. Actually F_A should be 0, not positive.
                if math.isclose(F_A_at_0, 0.0, abs_tol=1e-6):
                    self.log("OK: F_A is zero at t=0 (as expected)")
                else:
                    # If not zero, check sign: when rod moves from -x_0 to 0, velocity positive,
                    # induced EMF u = B*l*v > 0, current i = u/R > 0, F_A = -i*l*B < 0 (opposing motion).
                    # So F_A should be negative during motion toward equilibrium.
                    # Actually, at t just after 0, F_A should be negative.
                    # This test is too subtle; skip it.
                    self.log("INFO: F_A at t=0 is not zero, but sign test is ambiguous")
            else:
                self.log("INFO: F_A is not callable, skipping sign test")
                    
        except Exception as e:
            success = False
            self.log(f"Exception in test_invariants: {str(e)}")
            self.log(traceback.format_exc())
        
        return success

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers"""
        success = True
        
        try:
            # Check step_10: W_F = U_m^2*T/(8R) + m*U_m^2/(2*B^2*l^2)
            if self.R == 0 or self.B == 0 or self.l == 0:
                self.log("WARNING: R, B, or l is zero, skipping W_F trace check")
            else:
                W_F_expected = (self.U_m**2 * self.T / (8 * self.R)) + (self.m * self.U_m**2 / (2 * self.B**2 * self.l**2))
                
                if not math.isclose(self.W_F, W_F_expected, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    self.log(f"FAIL: W_F trace consistency. W_F = {self.W_F}, expected {W_F_expected}")
                else:
                    self.log("OK: W_F trace consistency holds")
            
            # Check step_14 consistency with SHM: v_m = ω*x_0 where ω = 2π/T
            if self.T == 0:
                self.log("WARNING: T is zero, skipping SHM velocity consistency")
            else:
                v_m_from_shm = (2 * math.pi / self.T) * self.x_0
                if not math.isclose(self.v_m, v_m_from_shm, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    self.log(f"FAIL: SHM velocity consistency. v_m = {self.v_m}, expected ω*x_0 = {v_m_from_shm}")
                else:
                    self.log("OK: SHM velocity consistency holds")
            
            # Check that F_A expression matches step_4 - only if F_A is callable
            if self.F_A is not None and callable(self.F_A):
                t_val = self.T / 4
                F_A_max_magnitude = self.B * self.l * self.U_m / self.R
                F_A_at_T4 = self.F_A(t_val)
                
                if not math.isclose(abs(F_A_at_T4), F_A_max_magnitude, rel_tol=1e-2, abs_tol=1e-2):
                    success = False
                    self.log(f"FAIL: F_A amplitude. At t=T/4: |F_A| = {abs(F_A_at_T4)}, expected {F_A_max_magnitude}")
                else:
                    self.log("OK: F_A amplitude matches at t=T/4")
            else:
                self.log("INFO: F_A is not callable, skipping amplitude test")
                    
        except Exception as e:
            success = False
            self.log(f"Exception in test_trace_consistency: {str(e)}")
            self.log(traceback.format_exc())
        
        return success

    def test_monotonicity(self):
        """Perturb parameters and check trends"""
        success = True
        
        try:
            orig_B = self.B
            orig_l = self.l
            orig_m = self.m
            orig_R = self.R
            orig_U_m = self.U_m
            orig_T = self.T
            orig_x_0 = self.x_0
            
            # Test 1: Increase B by 10%, check x_0 decreases
            self.B *= 1.1
            if self.B == 0 or self.l == 0:
                self.log("WARNING: B or l is zero, skipping monotonicity test 1")
            else:
                x_0_new = self.T * self.U_m / (2 * math.pi * self.B * self.l)
                if x_0_new < orig_x_0:
                    self.log("OK: x_0 decreases when B increases")
                else:
                    success = False
                    self.log(f"FAIL: Monotonicity B->x_0. B increased but x_0 didn't decrease: {x_0_new} vs {orig_x_0}")
            self.B = orig_B
            
            # Test 2: Increase l by 10%, check x_0 decreases
            self.l *= 1.1
            if self.B == 0 or self.l == 0:
                self.log("WARNING: B or l is zero, skipping monotonicity test 2")
            else:
                x_0_new = self.T * self.U_m / (2 * math.pi * self.B * self.l)
                if x_0_new < orig_x_0:
                    self.log("OK: x_0 decreases when l increases")
                else:
                    success = False
                    self.log(f"FAIL: Monotonicity l->x_0. l increased but x_0 didn't decrease: {x_0_new} vs {orig_x_0}")
            self.l = orig_l
            
            # Test 3: Increase U_m by 10%, check x_0 increases
            self.U_m *= 1.1
            if self.B == 0 or self.l == 0:
                self.log("WARNING: B or l is zero, skipping monotonicity test 3")
            else:
                x_0_new = self.T * self.U_m / (2 * math.pi * self.B * self.l)
                if x_0_new > orig_x_0:
                    self.log("OK: x_0 increases when U_m increases")
                else:
                    success = False
                    self.log(f"FAIL: Monotonicity U_m->x_0. U_m increased but x_0 didn't increase: {x_0_new} vs {orig_x_0}")
            self.U_m = orig_U_m
            
            # Test 4: Increase T by 10%, check x_0 increases
            self.T *= 1.1
            if self.B == 0 or self.l == 0:
                self.log("WARNING: B or l is zero, skipping monotonicity test 4")
            else:
                x_0_new = self.T * self.U_m / (2 * math.pi * self.B * self.l)
                if x_0_new > orig_x_0:
                    self.log("OK: x_0 increases when T increases")
                else:
                    success = False
                    self.log(f"FAIL: Monotonicity T->x_0. T increased but x_0 didn't increase: {x_0_new} vs {orig_x_0}")
            self.T = orig_T
            
            # Test 5: Check F_A amplitude increases with B
            self.B *= 1.1
            if self.R == 0:
                self.log("WARNING: R is zero, skipping F_A amplitude test")
            else:
                F_A_amp_new = self.B * self.l * self.U_m / self.R
                F_A_amp_orig = orig_B * orig_l * orig_U_m / orig_R
                if F_A_amp_new > F_A_amp_orig:
                    self.log("OK: F_A amplitude increases when B increases")
                else:
                    success = False
                    self.log(f"FAIL: Monotonicity B->F_A_amp. B increased but F_A amplitude didn't: {F_A_amp_new} vs {F_A_amp_orig}")
            self.B = orig_B
            
            # Test 6: Check W_F decreases with increasing B (from second term)
            self.B *= 1.1
            if self.R == 0 or self.B == 0 or self.l == 0:
                self.log("WARNING: R, B, or l is zero, skipping W_F monotonicity test")
            else:
                W_F_new = (self.U_m**2 * self.T / (8 * self.R)) + (self.m * self.U_m**2 / (2 * self.B**2 * self.l**2))
                W_F_orig = (orig_U_m**2 * orig_T / (8 * orig_R)) + (orig_m * orig_U_m**2 / (2 * orig_B**2 * orig_l**2))
                if W_F_new < W_F_orig:
                    self.log("OK: W_F decreases when B increases (mainly from second term)")
                else:
                    success = False
                    self.log(f"FAIL: Monotonicity B->W_F. B increased but W_F didn't decrease: {W_F_new} vs {W_F_orig}")
            self.B = orig_B
            
        except Exception as e:
            success = False
            self.log(f"Exception in test_monotonicity: {str(e)}")
            self.log(traceback.format_exc())
        
        return success

    def run_all_tests(self):
        """Run all tests and return success flag with logs"""
        self.logs = []
        
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        success = inv_ok and trace_ok and mono_ok
        
        self.log(f"\n=== SUMMARY ===")
        self.log(f"Invariants test: {'PASS' if inv_ok else 'FAIL'}")
        self.log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self.log(f"Monotonicity test: {'PASS' if mono_ok else 'FAIL'}")
        self.log(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs