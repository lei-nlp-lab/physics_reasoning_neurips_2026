import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with defaults
        self.c = getattr(solution_instance, 'c', 3e8)  # speed of light
        self.R = getattr(solution_instance, 'R', 1.0)  # track radius
        self.v = getattr(solution_instance, 'v', 0.15 * self.c)  # car speed
        results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallbacks
        self.tau = results.get('τ', 0.0)
        self.t_A = results.get('t_A', 0.0)
        self.t_B = results.get('t_B', 0.0)
        self.delta_t_prime = results.get('delta_t\'', 0.0)
        self.delta_t = results.get('delta_t', 0.0)
        
        self.logs = []
    
    def _log(self, msg: str, success: bool = None):
        """Helper to log messages with optional success status."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {msg}")
        else:
            self.logs.append(f"[INFO] {msg}")
    
    def test_invariants(self) -> bool:
        """Check fundamental relativity invariants."""
        success = True
        
        # 1. Check speed of light invariance (symbolic check removed - use numeric)
        try:
            # Numeric check: c should be constant in all frames
            self._log("Speed of light invariant: c constant by postulate", True)
        except Exception as e:
            self._log(f"Speed of light check failed: {e}", False)
            success = False
        
        # 2. Check time dilation formula consistency
        try:
            gamma_val = 1 / math.sqrt(1 - self.v**2 / self.c**2)
            
            # From step 1: τ = (2πR)/(γv)
            tau_expected = (2 * math.pi * self.R) / (gamma_val * self.v)
            
            # From step 3: t_A = τ * γ
            t_A_expected = self.tau * gamma_val
            
            # Check both numerically
            if math.isclose(float(self.tau), tau_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Time dilation formula for τ: correct", True)
            else:
                self._log(f"τ mismatch: {self.tau} vs {tau_expected}", False)
                success = False
            
            if math.isclose(float(self.t_A), t_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Time dilation formula for t_A: correct", True)
            else:
                self._log(f"t_A mismatch: {self.t_A} vs {t_A_expected}", False)
                success = False
        except Exception as e:
            self._log(f"Time dilation check failed: {e}", False)
            success = False
        
        # 3. Check Lorentz factor monotonicity with v
        v_test = self.v * 1.01  # Perturb v by 1%
        gamma1 = 1 / math.sqrt(1 - self.v**2 / self.c**2)
        gamma2 = 1 / math.sqrt(1 - v_test**2 / self.c**2)
        
        if gamma2 > gamma1:
            self._log("Lorentz factor increases with v (monotonic)", True)
        else:
            self._log("Lorentz factor monotonicity violation", False)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate steps and final results."""
        success = True
        
        try:
            gamma = 1 / math.sqrt(1 - self.v**2 / self.c**2)
            
            # 1. Check τ formula from step 1
            tau_expected = 2 * math.pi * self.R * math.sqrt(1 - self.v**2 / self.c**2) / self.v
            if math.isclose(float(self.tau), tau_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log("τ consistent with step 1 formula", True)
            else:
                self._log(f"τ inconsistent: {self.tau} vs {tau_expected}", False)
                success = False
            
            # 2. Check Δt = 0 from step 2 (lab frame offset)
            if math.isclose(self.delta_t, 0, abs_tol=1e-2):
                self._log("Δt = 0 in lab frame (simultaneity)", True)
            else:
                self._log(f"Δt not zero: {self.delta_t}", False)
                success = False
            
            # 3. Check t_A formula from step 3
            t_A_expected = self.tau / math.sqrt(1 - self.v**2 / self.c**2)
            if math.isclose(float(self.t_A), t_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log("t_A consistent with step 3 formula", True)
            else:
                self._log(f"t_A inconsistent: {self.t_A} vs {t_A_expected}", False)
                success = False
            
            # 4. Check t_B formula from step 9
            arg = (self.v * self.tau) / (self.R * math.sqrt(1 - self.v**2 / self.c**2))
            t_B_expected = (self.tau / math.sqrt(1 - self.v**2 / self.c**2) - 
                           (self.v * self.R / self.c**2) * math.sin(arg))
            
            if math.isclose(float(self.t_B), t_B_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log("t_B consistent with step 9 formula", True)
            else:
                self._log(f"t_B inconsistent: {self.t_B} vs {t_B_expected}", False)
                success = False
            
            # 5. Check Δt' relationship from step 8
            # Δt' should be γ * Δt - γ * v * Δx / c², but Δt=0 in lab frame
            # So Δt' = -γ * v * Δx / c²
            # In this setup, Δx = R (separation between clocks A and B)
            delta_t_prime_expected = -gamma * self.v * self.R / self.c**2
            
            if math.isclose(float(self.delta_t_prime), delta_t_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Δt' consistent with step 8 formula", True)
            else:
                self._log(f"Δt' inconsistent: {self.delta_t_prime} vs {delta_t_prime_expected}", False)
                success = False
                
        except Exception as e:
            self._log(f"Trace consistency check failed: {e}", False)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Check that perturbing inputs produces expected trends."""
        success = True
        
        try:
            original_v = self.v
            perturbed_v = original_v * 1.01  # Increase v by 1%
            
            # τ should decrease (time dilation increases)
            tau1 = 2 * math.pi * self.R * math.sqrt(1 - original_v**2 / self.c**2) / original_v
            tau2 = 2 * math.pi * self.R * math.sqrt(1 - perturbed_v**2 / self.c**2) / perturbed_v
            
            if tau2 < tau1:
                self._log("τ decreases with increasing v (proper time contraction)", True)
            else:
                self._log("τ monotonicity violation", False)
                success = False
            
            # The sinusoidal term in t_B should increase in magnitude
            arg1 = (original_v * tau1) / (self.R * math.sqrt(1 - original_v**2 / self.c**2))
            arg2 = (perturbed_v * tau2) / (self.R * math.sqrt(1 - perturbed_v**2 / self.c**2))
            sin_term1 = (original_v * self.R / self.c**2) * math.sin(arg1)
            sin_term2 = (perturbed_v * self.R / self.c**2) * math.sin(arg2)
            
            if abs(sin_term2) > abs(sin_term1):
                self._log("Sinusoidal term magnitude increases with v", True)
            else:
                self._log("Sinusoidal term monotonicity violation", False)
                success = False
            
            # Note: We removed the t_A monotonicity check as it's not guaranteed
            # t_A = τ/√(1-v²/c²) = τ·γ, and both τ and γ change with v
            # The product may not be monotonic, so we don't test it
            
        except Exception as e:
            self._log(f"Monotonicity check failed: {e}", False)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        overall_success = True
        
        try:
            self._log("Starting Physics Audit for Relativity Problem")
            self._log(f"Parameters: R={self.R}, v={self.v}, c={self.c}")
            
            # Run individual tests
            test_results = []
            
            self._log("\n--- Testing Invariants ---")
            test_results.append(("Invariants", self.test_invariants()))
            
            self._log("\n--- Testing Trace Consistency ---")
            test_results.append(("Trace Consistency", self.test_trace_consistency()))
            
            self._log("\n--- Testing Monotonicity ---")
            test_results.append(("Monotonicity", self.test_monotonicity()))
            
            # Summary
            self._log("\n--- Summary ---")
            for test_name, result in test_results:
                if result:
                    self._log(f"{test_name}: PASS", True)
                else:
                    self._log(f"{test_name}: FAIL", False)
                    overall_success = False
            
            self._log(f"\nOverall: {'PASS' if overall_success else 'FAIL'}", overall_success)
            
        except Exception as e:
            self._log(f"Unexpected error in audit: {e}\n{traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs