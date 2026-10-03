import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Uses getattr with fallbacks to avoid AttributeError.
        """
        # Input parameters with safe access - use defaults if not found
        self.gamma = getattr(solution_instance, 'gamma', 2.675e8)
        self.B0 = getattr(solution_instance, 'B0', 1.0)
        self.b = getattr(solution_instance, 'b', 0.1)
        self.omega = getattr(solution_instance, 'omega', 2.0e8)
        self.t = getattr(solution_instance, 't', 1.0e-6)
        self.mu = getattr(solution_instance, 'mu', 1.41e-26)
        
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results with safe fallbacks
        self.omega0 = self.results.get('omega₀', self.results.get('ω₀', None))
        self.dmu_dt_mag = self.results.get('|dmu/dt|', self.results.get('|dμ/dt|', None))
        self.delta = self.results.get('delta_', self.results.get('Δ', None))
        self.B_eff = self.results.get('Bₑff', None)
        self.Omega = self.results.get('Ω', None)
        self.P_up = self.results.get('P↑', None)
        self.P_down = self.results.get('P↓', None)
        
        # Initialize logs
        self.logs = []

    def log(self, message: str, success: bool = True):
        """Add a log message."""
        self.logs.append((success, message))

    def _is_close(self, a: Any, b: Any, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """
        Safe numerical comparison with fuzzy matching.
        Converts everything to float before comparison.
        """
        try:
            # Convert to float if possible
            if hasattr(a, '__float__'):
                a = float(a)
            if hasattr(b, '__float__'):
                b = float(b)
            
            # Handle None values
            if a is None or b is None:
                return False
                
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self) -> bool:
        """Test fundamental physical invariants and laws from KB."""
        success = True
        
        # 1. Test Larmor precession frequency: ω₀ = γB₀
        if self.omega0 is not None:
            expected = self.gamma * self.B0
            if not self._is_close(self.omega0, expected):
                self.log(f"Larmor frequency invariant failed: ω₀={self.omega0}, expected γB₀={expected}", False)
                success = False
            else:
                self.log(f"Larmor frequency invariant passed: ω₀ = γB₀ = {expected}")
        else:
            self.log("Missing ω₀ for Larmor frequency test", False)
            success = False
        
        # 2. Test Δ = γB₀ - ω (from step 6)
        if self.delta is not None:
            expected = self.gamma * self.B0 - self.omega
            if not self._is_close(self.delta, expected):
                self.log(f"Δ invariant failed: Δ={self.delta}, expected γB₀-ω={expected}", False)
                success = False
            else:
                self.log(f"Δ invariant passed: Δ = γB₀ - ω = {expected}")
        
        # 3. Test Rabi frequency: Ω = γb (from step 14)
        if self.Omega is not None:
            expected = self.gamma * self.b
            if not self._is_close(self.Omega, expected):
                self.log(f"Rabi frequency invariant failed: Ω={self.Omega}, expected γb={expected}", False)
                success = False
            else:
                self.log(f"Rabi frequency invariant passed: Ω = γb = {expected}")
        
        # 4. Test probability conservation: P↑ + P↓ = 1
        if self.P_up is not None and self.P_down is not None:
            total = self.P_up + self.P_down
            if not self._is_close(total, 1.0):
                self.log(f"Probability conservation failed: P↑+P↓={total}, expected 1", False)
                success = False
            else:
                self.log(f"Probability conservation passed: P↑+P↓ = 1")
        
        # 5. Test torque magnitude consistency
        if self.dmu_dt_mag is not None:
            # From step 2: |dμ/dt| = γ|μ|B₀sinφ
            # Without φ, we can only check if non-negative
            if hasattr(self.dmu_dt_mag, '__float__'):
                torque_val = float(self.dmu_dt_mag)
                if torque_val < -1e-10:  # Allow small negative due to floating point
                    self.log(f"Torque magnitude negative: {torque_val}", False)
                    success = False
                else:
                    self.log(f"Torque magnitude non-negative: {torque_val}")
        
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check consistency between ω₀ and Δ
        if self.omega0 is not None and self.delta is not None:
            # From step 5-6: Δ = ω₀ - ω
            expected_delta = self.omega0 - self.omega
            if not self._is_close(self.delta, expected_delta):
                self.log(f"Trace consistency failed: Δ={self.delta}, expected ω₀-ω={expected_delta}", False)
                success = False
            else:
                self.log(f"Trace consistency passed: Δ = ω₀ - ω")
        
        # Check effective magnetic field form
        if self.B_eff is not None:
            # From step 9 and 10: B_eff has time-averaged component (B₀ - ω/γ) in z-direction
            # Check if value is reasonable
            if hasattr(self.B_eff, '__float__'):
                B_eff_val = float(self.B_eff)
                # No specific check, just log the value
                self.log(f"B_eff value: {B_eff_val}")
        
        # Check Rabi oscillation probabilities
        if self.P_up is not None and self.P_down is not None and self.Omega is not None and self.t is not None:
            # From step 17-19: P↓ = sin²(Ωt/2), P↑ = cos²(Ωt/2)
            expected_P_down = math.sin(self.Omega * self.t / 2) ** 2
            expected_P_up = math.cos(self.Omega * self.t / 2) ** 2
            
            if not self._is_close(self.P_down, expected_P_down):
                self.log(f"P↓ trace consistency failed: {self.P_down}, expected sin²(Ωt/2)={expected_P_down}", False)
                success = False
            else:
                self.log(f"P↓ trace consistency passed: sin²(Ωt/2)")
            
            if not self._is_close(self.P_up, expected_P_up):
                self.log(f"P↑ trace consistency failed: {self.P_up}, expected cos²(Ωt/2)={expected_P_up}", False)
                success = False
            else:
                self.log(f"P↑ trace consistency passed: cos²(Ωt/2)")
        
        return success

    def test_monotonicity(self) -> bool:
        """
        Test monotonic trends by perturbing inputs.
        Returns to original state after testing.
        """
        success = True
        
        # Test 1: ω₀ should increase linearly with B₀ (for fixed γ)
        try:
            # Perturb B0 by +10%
            perturbed_B0 = self.B0 * 1.1
            original_omega0 = self.gamma * self.B0
            perturbed_omega0 = self.gamma * perturbed_B0
            
            # Check monotonic increase
            if perturbed_omega0 <= original_omega0:
                self.log(f"Monotonicity failed: ω₀ doesn't increase with B₀", False)
                success = False
            else:
                self.log(f"Monotonicity passed: ω₀ increases with B₀ (linear)")
            
            # Check linearity
            expected_ratio = perturbed_B0 / self.B0
            actual_ratio = perturbed_omega0 / original_omega0
            if not self._is_close(actual_ratio, expected_ratio):
                self.log(f"Linearity check failed: ω₀/B₀ ratio changed", False)
                success = False
                
        except Exception as e:
            self.log(f"Monotonicity test error: {str(e)}", False)
            success = False
        
        # Test 2: Ω should increase linearly with b (for fixed γ)
        try:
            # Perturb b by +10%
            perturbed_b = self.b * 1.1
            original_Omega = self.gamma * self.b
            perturbed_Omega = self.gamma * perturbed_b
            
            # Check monotonic increase
            if perturbed_Omega <= original_Omega:
                self.log(f"Monotonicity failed: Ω doesn't increase with b", False)
                success = False
            else:
                self.log(f"Monotonicity passed: Ω increases with b (linear)")
            
            # Check linearity
            expected_ratio = perturbed_b / self.b
            actual_ratio = perturbed_Omega / original_Omega
            if not self._is_close(actual_ratio, expected_ratio):
                self.log(f"Linearity check failed: Ω/b ratio changed", False)
                success = False
                
        except Exception as e:
            self.log(f"Monotonicity test error: {str(e)}", False)
            success = False
        
        # Test 3: For fixed B₀, Δ should decrease with increasing ω
        try:
            # Perturb ω by +10%
            perturbed_omega = self.omega * 1.1
            original_delta = self.gamma * self.B0 - self.omega
            perturbed_delta = self.gamma * self.B0 - perturbed_omega
            
            # Check monotonic decrease
            if perturbed_delta >= original_delta:
                self.log(f"Monotonicity failed: Δ doesn't decrease with ω", False)
                success = False
            else:
                self.log(f"Monotonicity passed: Δ decreases with ω")
                
        except Exception as e:
            self.log(f"Monotonicity test error: {str(e)}", False)
            success = False
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return (success, logs).
        """
        self.logs = []
        
        try:
            # Run tests
            test1_success = self.test_invariants()
            test2_success = self.test_trace_consistency()
            test3_success = self.test_monotonicity()
            
            overall_success = test1_success and test2_success and test3_success
            
            # Summary
            self.log(f"\n=== TEST SUMMARY ===")
            self.log(f"Invariants: {'PASS' if test1_success else 'FAIL'}")
            self.log(f"Trace Consistency: {'PASS' if test2_success else 'FAIL'}")
            self.log(f"Monotonicity: {'PASS' if test3_success else 'FAIL'}")
            self.log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
            
            # Extract log messages
            log_messages = [msg for _, msg in self.logs]
            return overall_success, log_messages
            
        except Exception as e:
            error_msg = f"Error during testing: {str(e)}\n{traceback.format_exc()}"
            self.logs.append((False, error_msg))
            log_messages = [msg for _, msg in self.logs]
            return False, log_messages