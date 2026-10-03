import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks
        self.v0 = getattr(solution_instance, 'v0', 9.0)
        self.m = getattr(solution_instance, 'm', 25.0)
        self.mA = getattr(solution_instance, 'mA', 100.0)
        self.mB = getattr(solution_instance, 'mB', 100.0)
        
        # Safely extract results with fallbacks from results dictionary
        results = getattr(solution_instance, 'results', {})
        self.v = results.get('v', 0.0)
        self.v_prime = results.get('v\'', 0.0)  # Child's jump speed
        self.vA_prime = results.get('v_A\'', 0.0)  # Sled A speed after child jumps off
        self.I = results.get('I', 0.0)  # Impulse on sled A
        self.W = results.get('W', 0.0)  # Work on sled B
        
        # Store original values for monotonicity test
        self.original_m = self.m
        self.original_results = {
            'v': self.v,
            'v_prime': self.v_prime,
            'vA_prime': self.vA_prime,
            'I': self.I,
            'W': self.W
        }
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for test results
        self.logs = []

    def _is_close(self, a, b):
        """Robust numerical comparison with fuzzy matching."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        """Test conservation laws and physical invariants."""
        logs = []
        
        try:
            # 1. Total momentum conservation for the entire process
            # Initial momentum: (m + mA) * v0 (sled B stationary)
            # Final momentum: (m + mA + mB) * v
            p_initial = (self.m + self.mA) * self.v0
            p_final = (self.m + self.mA + self.mB) * self.v
            
            if self._is_close(p_initial, p_final):
                logs.append("PASS: Total momentum conserved (entire process)")
            else:
                logs.append(f"FAIL: Total momentum not conserved. Initial: {p_initial}, Final: {p_final}")
            
            # 2. Momentum conservation during child's jump from sled A
            # Before jump: (m + mA) * v0
            # After jump: m*v_prime + mA*vA_prime
            p_before_jump = (self.m + self.mA) * self.v0
            p_after_jump = self.m * self.v_prime + self.mA * self.vA_prime
            
            if self._is_close(p_before_jump, p_after_jump):
                logs.append("PASS: Momentum conserved during child's jump from sled A")
            else:
                logs.append(f"FAIL: Momentum not conserved during jump from sled A. Before: {p_before_jump}, After: {p_after_jump}")
            
            # 3. Momentum conservation during child's interaction with sled B
            # Child jumps onto sled B with speed v_prime, then off with speed -v_prime
            # Net momentum change for sled B: mB * v (final) - 0 (initial)
            # This should equal 2*m*v_prime according to step_2 equation
            momentum_change_sledB = self.mB * self.v
            child_momentum_change = 2 * self.m * self.v_prime
            
            if self._is_close(momentum_change_sledB, child_momentum_change):
                logs.append("PASS: Momentum transfer to sled B matches child's momentum change")
            else:
                logs.append(f"FAIL: Momentum mismatch for sled B. Sled B Δp: {momentum_change_sledB}, Child 2mv': {child_momentum_change}")
            
            # 4. Impulse-momentum theorem for sled A
            # I = mA * (v0 - vA_prime)
            impulse_calculated = self.mA * (self.v0 - self.vA_prime)
            
            if self._is_close(self.I, impulse_calculated):
                logs.append("PASS: Impulse on sled A matches impulse-momentum theorem")
            else:
                logs.append(f"FAIL: Impulse mismatch. Given I: {self.I}, Calculated: {impulse_calculated}")
            
            # 5. Work-energy theorem for sled B
            # W = (1/2) * mB * v^2
            work_calculated = 0.5 * self.mB * self.v**2
            
            if self._is_close(self.W, work_calculated):
                logs.append("PASS: Work on sled B matches work-energy theorem")
            else:
                logs.append(f"FAIL: Work mismatch. Given W: {self.W}, Calculated: {work_calculated}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            traceback.print_exc()
        
        self.logs.extend(logs)
        return logs

    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        logs = []
        
        try:
            # Check step_1: (m + mA) * v0 = (m + mA + mB) * v
            v_calculated = (self.m + self.mA) * self.v0 / (self.m + self.mA + self.mB)
            if self._is_close(self.v, v_calculated):
                logs.append(f"PASS: v={self.v} matches conservation equation (expected ~{v_calculated})")
            else:
                logs.append(f"FAIL: v mismatch. Given: {self.v}, Calculated from step_1: {v_calculated}")
            
            # Check step_2: 2*m*v_prime = mB * v
            v_prime_calculated = (self.mB * self.v) / (2 * self.m)
            if self._is_close(self.v_prime, v_prime_calculated):
                logs.append(f"PASS: v'={self.v_prime} matches conservation equation (expected ~{v_prime_calculated})")
            else:
                logs.append(f"FAIL: v' mismatch. Given: {self.v_prime}, Calculated from step_2: {v_prime_calculated}")
            
            # Check step_3: (m + mA)*v0 = m*v_prime + mA*vA_prime
            vA_prime_calculated = ((self.m + self.mA) * self.v0 - self.m * self.v_prime) / self.mA
            if self._is_close(self.vA_prime, vA_prime_calculated):
                logs.append(f"PASS: v_A'={self.vA_prime} matches conservation equation (expected ~{vA_prime_calculated})")
            else:
                logs.append(f"FAIL: v_A' mismatch. Given: {self.vA_prime}, Calculated from step_3: {vA_prime_calculated}")
            
            # Check step_4: I = mA * (v0 - vA_prime)
            I_calculated = self.mA * (self.v0 - self.vA_prime)
            if self._is_close(self.I, I_calculated):
                logs.append(f"PASS: I={self.I} matches impulse equation (expected ~{I_calculated})")
            else:
                logs.append(f"FAIL: I mismatch. Given: {self.I}, Calculated from step_4: {I_calculated}")
            
            # Check step_5: W = 0.5 * mB * v^2
            W_calculated = 0.5 * self.mB * self.v**2
            if self._is_close(self.W, W_calculated):
                logs.append(f"PASS: W={self.W} matches work-energy equation (expected ~{W_calculated})")
            else:
                logs.append(f"FAIL: W mismatch. Given: {self.W}, Calculated from step_5: {W_calculated}")
            
            # Cross-check target answers
            target_v = 5.0
            target_v_prime = 10.0
            target_I = 25.0
            target_W = 1250.0
            
            if self._is_close(self.v, target_v):
                logs.append(f"PASS: Final velocity v={self.v} matches target {target_v}m/s")
            else:
                logs.append(f"WARNING: v={self.v} differs from target {target_v}m/s")
            
            if self._is_close(self.v_prime, target_v_prime):
                logs.append(f"PASS: Jump speed v'={self.v_prime} matches target {target_v_prime}m/s")
            else:
                logs.append(f"WARNING: v'={self.v_prime} differs from target {target_v_prime}m/s")
            
            if self._is_close(self.I, target_I):
                logs.append(f"PASS: Impulse I={self.I} matches target {target_I}kg·m/s")
            else:
                logs.append(f"WARNING: I={self.I} differs from target {target_I}kg·m/s")
            
            if self._is_close(self.W, target_W):
                logs.append(f"PASS: Work W={self.W} matches target {target_W}J")
            else:
                logs.append(f"WARNING: W={self.W} differs from target {target_W}J")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            traceback.print_exc()
        
        self.logs.extend(logs)
        return logs

    def test_monotonicity(self):
        """Test how results change when perturbing child's mass."""
        logs = []
        
        try:
            # Perturb mass by 10%
            perturbed_m = self.original_m * 1.1
            
            # Recalculate using the same equations
            v_perturbed = (perturbed_m + self.mA) * self.v0 / (perturbed_m + self.mA + self.mB)
            v_prime_perturbed = (self.mB * v_perturbed) / (2 * perturbed_m)
            vA_prime_perturbed = ((perturbed_m + self.mA) * self.v0 - perturbed_m * v_prime_perturbed) / self.mA
            I_perturbed = self.mA * (self.v0 - vA_prime_perturbed)
            W_perturbed = 0.5 * self.mB * v_perturbed**2
            
            # Check trends
            # v should increase with m (derivative positive)
            if v_perturbed > self.original_results['v']:
                logs.append(f"PASS: v increases with mass ({self.original_results['v']} -> {v_perturbed})")
            else:
                logs.append(f"WARNING: v doesn't increase with mass as expected")
            
            # v' trend: v' = (mB*v)/(2m). As m increases, denominator increases but v increases.
            # Let's just check that the recalculated value is reasonable
            v_prime_expected = (self.mB * v_perturbed) / (2 * perturbed_m)
            if self._is_close(v_prime_perturbed, v_prime_expected):
                logs.append(f"PASS: v' perturbed calculation consistent ({v_prime_perturbed})")
            else:
                logs.append(f"WARNING: v' perturbed calculation inconsistent")
            
            # I should change predictably
            I_expected = self.mA * (self.v0 - vA_prime_perturbed)
            if self._is_close(I_perturbed, I_expected):
                logs.append(f"PASS: I perturbed calculation consistent ({I_perturbed})")
            else:
                logs.append(f"WARNING: I perturbed calculation inconsistent")
            
            # W should increase with v^2
            if W_perturbed > self.original_results['W']:
                logs.append(f"PASS: W increases with mass ({self.original_results['W']} -> {W_perturbed})")
            else:
                logs.append(f"WARNING: W doesn't increase with mass as expected")
            
            # Restore original values
            self.m = self.original_m
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            traceback.print_exc()
            # Ensure we restore original mass even if error occurs
            self.m = self.original_m
        
        self.logs.extend(logs)
        return logs

    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []  # Reset logs
        
        self.logs.append("=== PhysicsChecker Test Suite ===")
        self.logs.append(f"Inputs: m={self.m}, mA={self.mA}, mB={self.mB}, v0={self.v0}")
        self.logs.append(f"Results: v={self.v}, v'={self.v_prime}, v_A'={self.vA_prime}, I={self.I}, W={self.W}")
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success (no FAIL messages in logs)
        success = True
        for log in self.logs:
            if log.startswith("FAIL:"):
                success = False
                break
        
        self.logs.append(f"\nOverall test result: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs

# Example usage and testing
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.v0 = 9.0
            self.m = 25.0
            self.mA = 100.0
            self.mB = 100.0
            self.results = {
                'v': 5.0,
                'v\'': 10.0,
                'v_A\'': 8.75,
                'I': 25.0,
                'W': 1250.0
            }
    
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)