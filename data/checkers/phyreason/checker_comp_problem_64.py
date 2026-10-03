import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs from results dictionary (they should be stored there)
        self.results = getattr(solution_instance, 'results', {})
        
        # Get input parameters from results (with defaults)
        self.H = self.results.get('H', 0.0)
        self.V = self.results.get('V', 0.0)
        self.m = self.results.get('m', 0.0)
        self.p = self.results.get('p', 0.0)
        self.T = self.results.get('T', 0.0)
        self.g = self.results.get('g', 9.8)
        
        # Compute x = (m * g * H) / (p * V)
        if self.p * self.V != 0:
            self.x = (self.m * self.g * self.H) / (self.p * self.V)
        else:
            self.x = 0.0
        
        # Extract required output variables
        self.V0 = self.results.get('V_0', 0.0)
        self.T1 = self.results.get('T_1', 0.0)
        self.T2 = self.results.get('T_2', 0.0)
        self.Delta_p = self.results.get('Deltap', 0.0)
        self.p1_prime = self.results.get('p_1\'', 0.0)
        self.p2_prime = self.results.get('p_2\'', 0.0)
        self.T1_prime = self.results.get('T_1\'', 0.0)
        self.T2_prime = self.results.get('T_2\'', 0.0)
        
        # Constants
        self.R = 8.314
        self.gamma = 5/3
        
        self.logs = []
    
    def _log(self, message):
        self.logs.append(message)
    
    def test_invariants(self):
        """Check physical invariants based on KB and steps analysis."""
        success = True
        try:
            # 1. Check Δp = p * x
            expected_Delta_p = self.p * self.x
            if not math.isclose(self.Delta_p, expected_Delta_p, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Δp = {self.Delta_p} ≠ p*x = {expected_Delta_p}")
                success = False
            else:
                self._log(f"OK: Δp = p*x (within tolerance)")
            
            # 2. Check V0 = V * (1+x)^{-3/5}
            expected_V0 = self.V * (1 + self.x) ** (-3/5)
            if not math.isclose(self.V0, expected_V0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: V0 = {self.V0} ≠ V*(1+x)^{-3/5} = {expected_V0}")
                success = False
            else:
                self._log(f"OK: V0 = V*(1+x)^{-3/5}")
            
            # 3. Check adiabatic condition for lower chamber when leak starts: V^5 p^3 = V0^5 (p+Δp)^3
            left1 = (self.V ** 5) * (self.p ** 3)
            right1 = (self.V0 ** 5) * ((self.p + self.Delta_p) ** 3)
            if not math.isclose(left1, right1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: V^5 p^3 = {left1} ≠ V0^5 (p+Δp)^3 = {right1}")
                success = False
            else:
                self._log(f"OK: Adiabatic condition V^5 p^3 = V0^5 (p+Δp)^3 holds")
            
            # 4. Check adiabatic condition for lower chamber when piston touches: (V+V0)^5 (p+Δp)^3 = V^5 (p2')^3
            if self.p2_prime != 0:
                left2 = ((self.V + self.V0) ** 5) * ((self.p + self.Delta_p) ** 3)
                right2 = (self.V ** 5) * (self.p2_prime ** 3)
                if not math.isclose(left2, right2, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: (V+V0)^5 (p+Δp)^3 = {left2} ≠ V^5 (p2')^3 = {right2}")
                    success = False
                else:
                    self._log(f"OK: Adiabatic condition (V+V0)^5 (p+Δp)^3 = V^5 (p2')^3 holds")
            
            # 5. Check pressure difference when piston touches: p1' = p2' - Δp
            if self.p1_prime != 0 and self.p2_prime != 0:
                expected_p1_prime = self.p2_prime - self.Delta_p
                if not math.isclose(self.p1_prime, expected_p1_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: p1' = {self.p1_prime} ≠ p2' - Δp = {expected_p1_prime}")
                    success = False
                else:
                    self._log(f"OK: p1' = p2' - Δp")
            
            # 6. Check T1 formula: T1 = (T/2) * ( (1+(1+x)^{3/5})^{5/3} - x )
            if self.T1 != 0:
                term = (1 + (1 + self.x) ** (3/5)) ** (5/3)
                expected_T1 = (self.T / 2) * (term - self.x)
                if not math.isclose(self.T1, expected_T1, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: T1 = {self.T1} ≠ (T/2)*((1+(1+x)^(3/5))^(5/3)-x) = {expected_T1}")
                    success = False
                else:
                    self._log(f"OK: T1 matches formula")
            
            # 7. Check T2 formula: T2 = T * (1+(1+x)^{3/5})^{2/3}
            if self.T2 != 0:
                expected_T2 = self.T * (1 + (1 + self.x) ** (3/5)) ** (2/3)
                if not math.isclose(self.T2, expected_T2, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: T2 = {self.T2} ≠ T*(1+(1+x)^(3/5))^(2/3) = {expected_T2}")
                    success = False
                else:
                    self._log(f"OK: T2 matches formula")
            
            # 8. Check adiabatic temperature-pressure relation for lower chamber: T^5 p^{-2} = (T2')^5 (p2')^{-2}
            if self.T2_prime != 0 and self.p2_prime != 0:
                left3 = (self.T ** 5) * (self.p ** (-2))
                right3 = (self.T2_prime ** 5) * (self.p2_prime ** (-2))
                if not math.isclose(left3, right3, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"FAIL: T^5 p^{-2} = {left3} ≠ (T2')^5 (p2')^{-2} = {right3}")
                    success = False
                else:
                    self._log(f"OK: Adiabatic T-p relation for lower chamber holds")
            
        except Exception as e:
            self._log(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Check consistency of intermediate results with final answers."""
        success = True
        try:
            # 1. Sanity: positive values
            if self.V0 <= 0 or self.T1 <= 0 or self.T2 <= 0:
                self._log(f"FAIL: Non-positive physical quantities: V0={self.V0}, T1={self.T1}, T2={self.T2}")
                success = False
            else:
                self._log(f"OK: V0, T1, T2 are positive")
            
            # 2. Check V0 < V (since x>0)
            if not (self.V0 < self.V):
                self._log(f"FAIL: V0 = {self.V0} not less than V = {self.V} (expected due to x>0)")
                success = False
            else:
                self._log(f"OK: V0 < V as expected")
            
            # 3. Check T2 > T (since (1+(1+x)^{3/5})^{2/3} > 1)
            if not (self.T2 > self.T):
                self._log(f"FAIL: T2 = {self.T2} not greater than T = {self.T}")
                success = False
            else:
                self._log(f"OK: T2 > T as expected")
            
            # 4. Check that T1 is reasonable (maybe between 0 and T2, but not required)
            if self.T1 > 2 * self.T or self.T1 < 0:
                self._log(f"WARN: T1 = {self.T1} seems out of plausible range")
                # Not a failure, just a warning
            
            # 5. Check p1' > 0 (if available)
            if self.p1_prime != 0 and self.p1_prime <= 0:
                self._log(f"FAIL: p1' = {self.p1_prime} is not positive")
                success = False
            elif self.p1_prime != 0:
                self._log(f"OK: p1' is positive")
            
            # 6. Verify that T1 and T1' match (if T1' is given)
            if self.T1_prime != 0 and not math.isclose(self.T1, self.T1_prime, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: T1 = {self.T1} ≠ T1' = {self.T1_prime}")
                success = False
            elif self.T1_prime != 0:
                self._log(f"OK: T1 = T1'")
            
            # 7. Verify that T2 and T2' match (if T2' is given)
            if self.T2_prime != 0 and not math.isclose(self.T2, self.T2_prime, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: T2 = {self.T2} ≠ T2' = {self.T2_prime}")
                success = False
            elif self.T2_prime != 0:
                self._log(f"OK: T2 = T2'")
            
        except Exception as e:
            self._log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb input x and check expected trends in outputs."""
        success = True
        try:
            original_x = self.x
            # Perturb x by 1% increase
            perturbed_x = original_x * 1.01
            
            # Recompute expected values using formulas
            V0_original = self.V * (1 + original_x) ** (-3/5)
            V0_perturbed = self.V * (1 + perturbed_x) ** (-3/5)
            
            T1_original = (self.T / 2) * ((1 + (1 + original_x) ** (3/5)) ** (5/3) - original_x)
            T1_perturbed = (self.T / 2) * ((1 + (1 + perturbed_x) ** (3/5)) ** (5/3) - perturbed_x)
            
            T2_original = self.T * (1 + (1 + original_x) ** (3/5)) ** (2/3)
            T2_perturbed = self.T * (1 + (1 + perturbed_x) ** (3/5)) ** (2/3)
            
            # Check trends
            # V0 should decrease as x increases (since derivative negative)
            if not (V0_perturbed < V0_original):
                self._log(f"FAIL: V0 did not decrease when x increased")
                success = False
            else:
                self._log(f"OK: V0 decreases with increasing x")
            
            # T2 should increase as x increases (derivative positive)
            if not (T2_perturbed > T2_original):
                self._log(f"FAIL: T2 did not increase when x increased")
                success = False
            else:
                self._log(f"OK: T2 increases with increasing x")
            
            # T1: we just log the change
            delta_T1 = T1_perturbed - T1_original
            self._log(f"INFO: T1 change with +1% x: delta = {delta_T1}")
            
            # Restore x (no actual change to instance, just for clarity)
            self.x = original_x
            
        except Exception as e:
            self._log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success and logs."""
        self.logs = []
        overall_success = True
        
        self._log("=== Starting PhysicsChecker ===")
        
        inv_success = self.test_invariants()
        if not inv_success:
            overall_success = False
        self._log(f"test_invariants: {'PASS' if inv_success else 'FAIL'}")
        
        trace_success = self.test_trace_consistency()
        if not trace_success:
            overall_success = False
        self._log(f"test_trace_consistency: {'PASS' if trace_success else 'FAIL'}")
        
        mono_success = self.test_monotonicity()
        if not mono_success:
            overall_success = False
        self._log(f"test_monotonicity: {'PASS' if mono_success else 'FAIL'}")
        
        self._log(f"=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs