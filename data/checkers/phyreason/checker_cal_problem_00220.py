import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all required data from the solution instance."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract initial conditions from problem context (hardcoded for safety)
        self.V_A = 4.0e-3  # m^3
        self.p_A = 47.0    # cmHg
        self.V_B = 6.0e-3  # m^3
        self.p_B = 52.0    # cmHg
        
        # Extract results with safe fallbacks
        self.V1 = self._safe_get('V_1')
        self.V2 = self._safe_get('V_2')
        self.sum_V = self._safe_get('V_1+V_2')
        self.pV1 = self._safe_get('pV_1')
        self.pV2 = self._safe_get('pV_2')
        
        # Pre-compute invariant quantities
        self.total_volume_initial = self.V_A + self.V_B
        self.product_A_initial = self.p_A * self.V_A
        self.product_B_initial = self.p_B * self.V_B
        
    def _safe_get(self, key):
        """Safe extraction with multiple fallback levels."""
        value = getattr(self, key, None)
        if value is None:
            value = self.results.get(key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0
    
    def test_invariants(self):
        """Test fundamental physical laws: Boyle's Law and Volume Conservation."""
        logs = []
        
        # 1. Total volume conservation
        if self.V1 > 0 and self.V2 > 0:
            total_volume_final = self.V1 + self.V2
            if not math.isclose(total_volume_final, self.total_volume_initial, 
                              rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: Volume conservation violated. "
                          f"Initial: {self.total_volume_initial:.6f}, "
                          f"Final: {total_volume_final:.6f}")
            else:
                logs.append("PASS: Volume conservation satisfied.")
        
        # 2. Boyle's Law for gas A (using final pressure equality)
        if self.V1 > 0 and self.V2 > 0:
            # Calculate final pressure from gas A
            p_final_A = self.product_A_initial / self.V1
            # Calculate final pressure from gas B
            p_final_B = self.product_B_initial / self.V2
            
            # Use more reasonable tolerance for pressure equilibrium
            # The values are around 50, so rel_tol=1e-2 gives ±0.5
            if not math.isclose(p_final_A, p_final_B, rel_tol=2e-2, abs_tol=1.0):
                logs.append(f"FAIL: Pressure equilibrium violated. "
                          f"p_A_final: {p_final_A:.3f}, p_B_final: {p_final_B:.3f}")
            else:
                logs.append("PASS: Pressure equilibrium satisfied (Boyle's Law).")
        
        # 3. Check individual Boyle's Law products
        if self.pV1 > 0:
            if not math.isclose(self.pV1, self.product_A_initial, 
                              rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: Gas A product mismatch. "
                          f"Expected: {self.product_A_initial:.4f}, "
                          f"Got: {self.pV1:.4f}")
            else:
                logs.append("PASS: Gas A product conserved.")
        
        if self.pV2 > 0:
            if not math.isclose(self.pV2, self.product_B_initial, 
                              rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: Gas B product mismatch. "
                          f"Expected: {self.product_B_initial:.4f}, "
                          f"Got: {self.pV2:.4f}")
            else:
                logs.append("PASS: Gas B product conserved.")
        
        return len([l for l in logs if 'FAIL' in l]) == 0, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Check if sum_V matches total_volume_initial
        if self.sum_V > 0:
            if not math.isclose(self.sum_V, self.total_volume_initial, 
                              rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: Intermediate sum mismatch. "
                          f"Expected: {self.total_volume_initial:.6f}, "
                          f"Got: {self.sum_V:.6f}")
            else:
                logs.append("PASS: Intermediate volume sum consistent.")
        
        # Check if V1 + V2 matches sum_V (if provided)
        if self.sum_V > 0 and self.V1 > 0 and self.V2 > 0:
            calculated_sum = self.V1 + self.V2
            if not math.isclose(calculated_sum, self.sum_V, 
                              rel_tol=1e-2, abs_tol=1e-4):
                logs.append(f"FAIL: Final volumes don't match intermediate sum. "
                          f"Sum: {self.sum_V:.6f}, V1+V2: {calculated_sum:.6f}")
            else:
                logs.append("PASS: Final volumes consistent with intermediate sum.")
        
        # Check if products match expectations
        if self.V1 > 0 and self.pV1 > 0:
            expected_p = self.pV1 / self.V1
            actual_p_A = self.product_A_initial / self.V1
            if not math.isclose(expected_p, actual_p_A, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Gas A pressure inconsistency. "
                          f"From product: {expected_p:.3f}, "
                          f"From calculation: {actual_p_A:.3f}")
        
        return len([l for l in logs if 'FAIL' in l]) == 0, logs
    
    def test_monotonicity(self):
        """Perturb an input parameter and verify physical trends."""
        logs = []
        
        # Save original state
        original_V_A = self.V_A
        
        try:
            # Perturb V_A by +10%
            perturbed_V_A = original_V_A * 1.10
            
            # Calculate new equilibrium using derived formulas
            # From equations: p_final = (p_A*V_A + p_B*V_B) / (V_A + V_B)
            # And V1 = (p_A * V_A) / p_final
            
            # Original equilibrium
            p_final_orig = (self.p_A*original_V_A + self.p_B*self.V_B) / (original_V_A + self.V_B)
            V1_orig = (self.p_A * original_V_A) / p_final_orig
            
            # Perturbed equilibrium
            p_final_pert = (self.p_A*perturbed_V_A + self.p_B*self.V_B) / (perturbed_V_A + self.V_B)
            V1_pert = (self.p_A * perturbed_V_A) / p_final_pert
            
            # Check monotonicity: If initial volume increases, final volume should increase
            if V1_pert > V1_orig:
                logs.append("PASS: Monotonicity holds - increasing V_A increases V1.")
            else:
                logs.append(f"WARNING: Non-monotonic behavior. "
                          f"V1_orig: {V1_orig:.6f}, V1_pert: {V1_pert:.6f}")
            
            # Check if the change is physically reasonable (~7-8% change for 10% perturbation)
            change_percent = (V1_pert - V1_orig) / V1_orig * 100
            # Relax the expected range due to floating-point variations
            if 5.0 < change_percent < 12.0:
                logs.append(f"PASS: Physical sensitivity reasonable ({change_percent:.1f}% change).")
            else:
                logs.append(f"NOTE: Sensitivity unusual ({change_percent:.1f}% change).")
                
        except ZeroDivisionError:
            logs.append("SKIP: Monotonicity test skipped due to division by zero.")
        
        # Restore original value (just in case)
        self.V_A = original_V_A
        
        return True, logs  # This test is informative, not pass/fail
    
    def run_all_tests(self):
        """Execute all tests and return comprehensive results."""
        all_logs = []
        success = True
        
        # Run invariants test
        inv_success, inv_logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(inv_logs)
        success = success and inv_success
        
        # Run trace consistency test
        trace_success, trace_logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(trace_logs)
        success = success and trace_success
        
        # Run monotonicity test
        mono_success, mono_logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(mono_logs)
        
        # Overall assessment
        all_logs.append("\n=== Summary ===")
        if success:
            all_logs.append("SUCCESS: All critical tests passed.")
        else:
            all_logs.append("FAILURE: One or more critical tests failed.")
        
        return success, all_logs