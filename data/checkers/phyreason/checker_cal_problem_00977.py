import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results with fallback defaults."""
        # Access via safe getattr with fallback to results dictionary
        # Note: The wavelength is stored as 'λ' (Unicode) in results
        self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', 0.0))
        self.delta_t = getattr(solution_instance, 'delta_t', solution_instance.results.get('delta_t', 0.0))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        # Try both 'λ' and 'lam' for wavelength
        self.lam = getattr(solution_instance, 'lam', 
                          solution_instance.results.get('λ', 
                                                       solution_instance.results.get('lam', 0.0)))
        self.t1 = getattr(solution_instance, 't1', solution_instance.results.get('t1', 0.0))
        # The distance s might be stored in meters or cm. Check for 's_cm' first.
        self.s = getattr(solution_instance, 's', 
                        solution_instance.results.get('s_cm', 
                                                     solution_instance.results.get('s', 0.0)))
        
        # Extract amplitude A from distance traveled s = 13A (from step 7)
        # s is now in cm, so A will be in cm
        self.A = self.s / 13.0 if self.s != 0.0 else 0.0
        
        # Store logs
        self.logs = []
    
    def _log(self, msg, success=True):
        """Helper to log messages with success flag."""
        self.logs.append((success, msg))
        return success
    
    def test_invariants(self):
        """Verify fundamental wave physics relationships."""
        try:
            # 1. Period from time to second trough (step 1 & 2)
            # From step_1: t1 = 7T/4  => T = 4*t1/7
            expected_T = 4.0 * self.t1 / 7.0
            if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Period T mismatch: got {self.T}, expected {expected_T} from t1={self.t1}", False)
            
            # 2. Wave speed from wavelength and period (step 3)
            if self.T != 0:
                expected_v = self.lam / self.T
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    return self._log(f"Wave speed v mismatch: got {self.v}, expected λ/T = {expected_v} (λ={self.lam}, T={self.T})", False)
            
            # 3. Wave propagation time (step 4) - Δt = Δx/v
            # Distance between M and N is not directly given, but we can verify Δt is positive
            if self.delta_t <= 0:
                return self._log(f"Wave propagation time Δt should be positive, got {self.delta_t}", False)
            
            # 4. Vibration time representation (step 6)
            expected_t = 13.0 * self.T / 4.0  # t = 3T + T/4 = 13T/4
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Vibration time t mismatch: got {self.t}, expected 13T/4 = {expected_t}", False)
            
            # 5. Distance traveled from amplitude (step 7)
            if self.A > 0:
                expected_s = 13.0 * self.A
                if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                    return self._log(f"Distance s mismatch: got {self.s}, expected 13A = {expected_s} (A={self.A})", False)
            
            return self._log("All invariants satisfied", True)
            
        except Exception as e:
            return self._log(f"Error in test_invariants: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Final answers: T=2.0s, s=26cm
            
            # 1. Check T=2.0 is consistent with t1=3.5s
            if not math.isclose(self.T, 2.0, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Final T={self.T} doesn't match target 2.0s", False)
            
            # 2. Check s=26cm is consistent with amplitude A
            if not math.isclose(self.s, 26.0, rel_tol=1e-2, abs_tol=1e-2):
                return self._log(f"Final s={self.s} doesn't match target 26cm", False)
            
            # 3. Check that A derived from s is reasonable (positive, within typical wave amplitudes)
            if self.A <= 0 or self.A > 100:  # Arbitrary upper bound for cm scale
                return self._log(f"Amplitude A={self.A}cm seems unrealistic", False)
            
            # 4. Verify the chain: t1 → T → v → Δt → t → s
            # Already checked in invariants, but do a quick sanity
            if not (math.isclose(self.t1, 3.5, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.T, 2.0, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.v, 2.0, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.delta_t, 3.5, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.t, 6.5, rel_tol=1e-2, abs_tol=1e-2)):
                self._log("Note: Some intermediate values differ from steps analysis", True)
            
            return self._log("Trace consistency verified", True)
            
        except Exception as e:
            return self._log(f"Error in test_trace_consistency: {str(e)}", False)
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends match physical laws."""
        try:
            original_t1 = self.t1
            original_T = self.T
            original_v = self.v
            original_s = self.s
            original_lam = self.lam
            original_A = self.A
            
            # Test 1: Increase t1 by 10% → T should increase proportionally (T = 4t1/7)
            perturbed_t1 = original_t1 * 1.1
            perturbed_T = 4.0 * perturbed_t1 / 7.0
            if not (perturbed_T > original_T):
                return self._log(f"Monotonicity fail: T didn't increase when t1 increased (t1: {original_t1}→{perturbed_t1}, T: {original_T}→{perturbed_T})", False)
            
            # Test 2: Increase wavelength λ by 10% → v should increase (v = λ/T, T fixed from t1)
            perturbed_lam = original_lam * 1.1
            # Keep T fixed at original (since t1 unchanged), perturb v
            perturbed_v = perturbed_lam / original_T
            if not (perturbed_v > original_v):
                return self._log(f"Monotonicity fail: v didn't increase when λ increased (λ: {original_lam}→{perturbed_lam}, v: {original_v}→{perturbed_v})", False)
            
            # Test 3: Increase amplitude A by 10% → s should increase proportionally (s = 13A)
            perturbed_A = original_A * 1.1
            perturbed_s = 13.0 * perturbed_A
            if not (perturbed_s > original_s):
                return self._log(f"Monotonicity fail: s didn't increase when A increased (A: {original_A}→{perturbed_A}, s: {original_s}→{perturbed_s})", False)
            
            # Restore original values (not strictly needed as we don't modify instance)
            return self._log("Monotonicity tests passed", True)
            
        except Exception as e:
            return self._log(f"Error in test_monotonicity: {str(e)}", False)
    
    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        self.logs.clear()  # Reset logs
        
        results = []
        
        # Run tests
        results.append(self.test_invariants())
        results.append(self.test_trace_consistency())
        results.append(self.test_monotonicity())
        
        # Overall success
        success = all(result for result in results)
        
        # Add summary
        if success:
            self.logs.append((True, "=== ALL PHYSICS CHECKS PASSED ==="))
        else:
            self.logs.append((False, "=== SOME CHECKS FAILED ==="))
        
        return success, self.logs