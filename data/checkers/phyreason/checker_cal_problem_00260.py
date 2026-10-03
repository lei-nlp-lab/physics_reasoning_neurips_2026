import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.d_2 = self.results.get('d_2', 0.0)
        self.T_2 = self.results.get('T_2', 0.0)
        self.T_1 = self.results.get('T_1', 0.0)
        self.t_2 = self.results.get('t_2', 0.0)
        self.Deltat = self.results.get('Deltat', 0.0)  # Note: key is 'Deltat' not 'Δ t'
        self.t_2_prime = self.results.get('t_2\'', 0.0)  # t_2'
        self.t_1 = self.results.get('t_1', 0.0)
        self.t_1_prime = self.results.get('t_1\'', 0.0)  # t_1'
        self.d_1 = self.results.get('d_1', 0.0)
        self.x_1 = self.results.get('x_1', 0.0)
        self.DeltaT = self.results.get('DeltaT', 0.0)   # ΔT
        
        # Physical constants from problem statement (safe extraction if needed)
        self.v0 = 30.0  # m/s
        self.v1 = 5.0   # m/s
        self.v2 = 10.0  # m/s (inferred from steps: d=10m, t_2'=1s => v2 = d/t_2' = 10 m/s)
        self.d = 10.0   # m
        self.a = 1.0    # m/s² (magnitude)
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for test results
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append((success, message))
        
    def _check_kinematic_eq(self, v0, v1, displacement, acceleration, description):
        """Check v1^2 = v0^2 + 2*a*displacement."""
        lhs = v1**2
        rhs = v0**2 + 2 * acceleration * displacement
        if not math.isclose(lhs, rhs, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"{description}: kinematic equation violated. LHS={lhs:.2f}, RHS={rhs:.2f}", False)
            return False
        self._log(f"{description}: kinematic equation satisfied.", True)
        return True
        
    def test_invariants(self):
        """Verify kinematic equations and physical consistency."""
        success = True
        
        # Step 1: Deceleration from v0 to v1 (a = -1)
        if not self._check_kinematic_eq(self.v0, self.v1, self.d_1, -self.a, 
                                         "Step 1 (deceleration to v1)"):
            success = False
            
        # Step 4: Deceleration from v0 to v2 (a = -1)
        if not self._check_kinematic_eq(self.v0, self.v2, self.d_2, -self.a,
                                         "Step 4 (deceleration to v2)"):
            success = False
            
        # Step 2: Total displacement x1 = 2*d_1 + d
        x1_calc = 2 * self.d_1 + self.d
        if not math.isclose(self.x_1, x1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 2: x1 mismatch. Expected {x1_calc:.2f}, got {self.x_1:.2f}", False)
            success = False
        else:
            self._log(f"Step 2: x1 = 2*d1 + d satisfied.", True)
            
        # Step 5: Time t1 from average velocity: d1 = (v0+v1)/2 * t1
        t1_calc = 2 * self.d_1 / (self.v0 + self.v1)
        if not math.isclose(self.t_1, t1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 5: t1 mismatch. Expected {t1_calc:.2f}, got {self.t_1:.2f}", False)
            success = False
        else:
            self._log(f"Step 5: t1 from average velocity satisfied.", True)
            
        # Step 6: Time t1' from constant velocity: d = v1 * t1'
        t1_prime_calc = self.d / self.v1
        if not math.isclose(self.t_1_prime, t1_prime_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 6: t1' mismatch. Expected {t1_prime_calc:.2f}, got {self.t_1_prime:.2f}", False)
            success = False
        else:
            self._log(f"Step 6: t1' from constant motion satisfied.", True)
            
        # Step 7: Total time T1 = 2*t1 + t1'
        T1_calc = 2 * self.t_1 + self.t_1_prime
        if not math.isclose(self.T_1, T1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 7: T1 mismatch. Expected {T1_calc:.2f}, got {self.T_1:.2f}", False)
            success = False
        else:
            self._log(f"Step 7: T1 = 2*t1 + t1' satisfied.", True)
            
        # Step 8: Time t2 from average velocity: d2 = (v0+v2)/2 * t2
        t2_calc = 2 * self.d_2 / (self.v0 + self.v2)
        if not math.isclose(self.t_2, t2_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 8: t2 mismatch. Expected {t2_calc:.2f}, got {self.t_2:.2f}", False)
            success = False
        else:
            self._log(f"Step 8: t2 from average velocity satisfied.", True)
            
        # Step 9: Time t2' from constant velocity: d = v2 * t2'
        t2_prime_calc = self.d / self.v2
        if not math.isclose(self.t_2_prime, t2_prime_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 9: t2' mismatch. Expected {t2_prime_calc:.2f}, got {self.t_2_prime:.2f}", False)
            success = False
        else:
            self._log(f"Step 9: t2' from constant motion satisfied.", True)
            
        # Step 10: Δt = (d1 - d2) / v0
        Deltat_calc = (self.d_1 - self.d_2) / self.v0
        if not math.isclose(self.Deltat, Deltat_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 10: Δt mismatch. Expected {Deltat_calc:.2f}, got {self.Deltat:.2f}", False)
            success = False
        else:
            self._log(f"Step 10: Δt = (d1-d2)/v0 satisfied.", True)
            
        # Step 11: T2 = 2*t2 + t2' + 2*Δt
        T2_calc = 2 * self.t_2 + self.t_2_prime + 2 * self.Deltat
        if not math.isclose(self.T_2, T2_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 11: T2 mismatch. Expected {T2_calc:.2f}, got {self.T_2:.2f}", False)
            success = False
        else:
            self._log(f"Step 11: T2 = 2*t2 + t2' + 2Δt satisfied.", True)
            
        # Step 12: ΔT = T1 - T2
        DeltaT_calc = self.T_1 - self.T_2
        if not math.isclose(self.DeltaT, DeltaT_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 12: ΔT mismatch. Expected {DeltaT_calc:.2f}, got {self.DeltaT:.2f}", False)
            success = False
        else:
            self._log(f"Step 12: ΔT = T1 - T2 satisfied.", True)
            
        return success
    
    def test_trace_consistency(self):
        """Cross-check that intermediate results are consistent with final answers."""
        success = True
        
        # Final answers from problem: x1=885 m, ΔT=8.5 s
        if not math.isclose(self.x_1, 885.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Trace consistency: x1 should be 885 m, but got {self.x_1:.2f} m", False)
            success = False
        else:
            self._log("Trace consistency: x1 matches target 885 m.", True)
            
        if not math.isclose(self.DeltaT, 8.5, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Trace consistency: ΔT should be 8.5 s, but got {self.DeltaT:.2f} s", False)
            success = False
        else:
            self._log("Trace consistency: ΔT matches target 8.5 s.", True)
            
        # Additionally, check that d_1 and d_2 are positive and d_1 > d_2 (since v1 < v2)
        if self.d_1 <= 0 or self.d_2 <= 0:
            self._log(f"Trace consistency: displacements d1={self.d_1:.2f}, d2={self.d_2:.2f} should be positive.", False)
            success = False
        if self.d_1 <= self.d_2:
            self._log(f"Trace consistency: d1 should be greater than d2 (since v1<v2), but d1={self.d_1:.2f}, d2={self.d_2:.2f}", False)
            success = False
            
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input parameter (v2) and verify that ΔT changes monotonically.
        Increase v2 → T2 should decrease → ΔT should increase.
        """
        success = True
        
        # Store original v2
        v2_original = self.v2
        
        try:
            # Perturb v2 by +1 m/s (to 11 m/s)
            v2_perturbed = v2_original + 1.0
            
            # Recalculate T2 and ΔT using the same formulas as in steps
            # 1. Deceleration distance for perturbed v2
            d2_perturbed = (v2_perturbed**2 - self.v0**2) / (2 * -self.a)  # a = -1
            # 2. Time for deceleration/acceleration phases
            t2_perturbed = 2 * d2_perturbed / (self.v0 + v2_perturbed)
            # 3. Time for constant velocity section
            t2_prime_perturbed = self.d / v2_perturbed
            # 4. Δt = (d1 - d2_perturbed)/v0
            Deltat_perturbed = (self.d_1 - d2_perturbed) / self.v0
            # 5. T2_perturbed
            T2_perturbed = 2 * t2_perturbed + t2_prime_perturbed + 2 * Deltat_perturbed
            # 6. ΔT_perturbed = T1 - T2_perturbed
            DeltaT_perturbed = self.T_1 - T2_perturbed
            
            # Check monotonicity: ΔT should increase when v2 increases
            if DeltaT_perturbed <= self.DeltaT:
                self._log(f"Monotonicity failed: ΔT decreased or stayed same when v2 increased. "
                          f"Original ΔT={self.DeltaT:.2f}, perturbed ΔT={DeltaT_perturbed:.2f}", False)
                success = False
            else:
                self._log(f"Monotonicity passed: ΔT increased from {self.DeltaT:.2f} to {DeltaT_perturbed:.2f} "
                          f"when v2 increased from {v2_original} to {v2_perturbed}.", True)
                
        except Exception as e:
            self._log(f"Monotonicity test error: {traceback.format_exc()}", False)
            success = False
            
        # Restore v2 (not strictly needed as we didn't modify instance, but for clarity)
        self.v2 = v2_original
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs = []  # Reset logs
        
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            overall_success = inv_ok and trace_ok and mono_ok
            
            # Summary log
            if overall_success:
                self._log("All tests passed.", True)
            else:
                self._log("Some tests failed.", False)
                
        except Exception as e:
            self._log(f"Unexpected error during testing: {traceback.format_exc()}", False)
            overall_success = False
            
        return overall_success, self.logs