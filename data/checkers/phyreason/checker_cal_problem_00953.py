import math
import sympy as sp
from typing import List, Tuple, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with fallback to problem defaults
        self.n = getattr(solution_instance, 'n', None)
        if self.n is None:
            self.n = getattr(solution_instance, 'inputs', {}).get('n', 100.0)
        
        self.S = getattr(solution_instance, 'S', None)
        if self.S is None:
            self.S = getattr(solution_instance, 'inputs', {}).get('S', 0.040)
        
        self.r = getattr(solution_instance, 'r', None)
        if self.r is None:
            self.r = getattr(solution_instance, 'inputs', {}).get('r', 1.0)
        
        self.R = getattr(solution_instance, 'R', None)
        if self.R is None:
            self.R = getattr(solution_instance, 'inputs', {}).get('R', 9.0)
        
        # Safe extraction of results using mapping
        # Convert to float where possible, keep None if missing
        self.E_m = self._safe_get_result(solution_instance, 'E_m')
        self.T = self._safe_get_result(solution_instance, 'T')
        self.I_m = self._safe_get_result(solution_instance, 'I_m')
        self.I = self._safe_get_result(solution_instance, 'I')
        self.P = self._safe_get_result(solution_instance, 'P')
        self.overlineE = self._safe_get_result(solution_instance, 'overlineE')
        self.overlineI = self._safe_get_result(solution_instance, 'overlineI')
        self.Q = self._safe_get_result(solution_instance, 'Q')
        
        # Constants
        self.sqrt2 = math.sqrt(2)
        self.pi = math.pi
        
        # Logs storage
        self.logs: List[str] = []
        
    def _safe_get_result(self, solution_instance, key: str) -> Optional[float]:
        """Safely extract a result value, converting to float if possible."""
        # First try direct attribute
        val = getattr(solution_instance, key, None)
        if val is None:
            # Then try results dictionary
            val = getattr(solution_instance, 'results', {}).get(key, None)
        
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        return None
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _log(self, message: str, success: bool) -> None:
        """Add a log message with success indicator."""
        status = "✓" if success else "✗"
        self.logs.append(f"{status} {message}")
    
    def test_invariants(self) -> bool:
        """Test fundamental physical laws and relationships."""
        success = True
        
        # 1. Ohm's Law for maximum values: I_m = E_m / (R + r)
        if self.I_m is not None and self.E_m is not None:
            expected_I_m = self.E_m / (self.R + self.r)
            ohm_ok = self._is_close(self.I_m, expected_I_m)
            self._log(f"Ohm's Law (max): I_m = {self.I_m:.3f} A, expected {expected_I_m:.3f} A", ohm_ok)
            if not ohm_ok:
                success = False
        
        # 2. Relationship between maximum and effective current: I = I_m / √2
        if self.I is not None and self.I_m is not None:
            expected_I = self.I_m / self.sqrt2
            ac_rel_ok = self._is_close(self.I, expected_I)
            self._log(f"AC relation: I = {self.I:.3f} A, expected I_m/√2 = {expected_I:.3f} A", ac_rel_ok)
            if not ac_rel_ok:
                success = False
        
        # 3. Power formula: P = I² * R
        if self.P is not None and self.I is not None:
            expected_P = self.I**2 * self.R
            power_ok = self._is_close(self.P, expected_P)
            self._log(f"Power: P = {self.P:.3f} W, expected I²R = {expected_P:.3f} W", power_ok)
            if not power_ok:
                success = False
        
        # 4. Faraday's Law consistency: E_m = n * B_m * S * (2π/T)
        # We can't check directly without B_m, but we can check that T is consistent with given parameters
        if self.E_m is not None and self.T is not None:
            # Compute B_m from the formula (assuming it's positive)
            B_m = self.E_m / (self.n * self.S * (2 * self.pi / self.T))
            faraday_ok = B_m > 0 and not math.isnan(B_m) and not math.isinf(B_m)
            self._log(f"Faraday's Law: computed B_m = {B_m:.3f} T (should be positive)", faraday_ok)
            if not faraday_ok:
                success = False
        
        # 5. Charge formula: Q = n * S * ΔB / (R + r)
        if self.Q is not None:
            # Compute ΔB from Q formula (should be positive)
            delta_B = self.Q * (self.R + self.r) / (self.n * self.S)
            charge_ok = delta_B > 0 and not math.isnan(delta_B) and not math.isinf(delta_B)
            self._log(f"Charge formula: computed ΔB = {delta_B:.4f} T (should be positive)", charge_ok)
            if not charge_ok:
                success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate results and final answers."""
        success = True
        
        # Check step-by-step consistency from the steps analysis
        # Step 1 -> Step 2: I_m = E_m / (R + r)
        if self.I_m is not None and self.E_m is not None:
            step2_ok = self._is_close(self.I_m, self.E_m / (self.R + self.r))
            self._log(f"Step 1→2: I_m = E_m/(R+r): {self.I_m:.3f} A vs {self.E_m/(self.R+self.r):.3f} A", step2_ok)
            if not step2_ok:
                success = False
        
        # Step 2 -> Step 3: I = I_m / √2
        if self.I is not None and self.I_m is not None:
            step3_ok = self._is_close(self.I, self.I_m / self.sqrt2)
            self._log(f"Step 2→3: I = I_m/√2: {self.I:.3f} A vs {self.I_m/self.sqrt2:.3f} A", step3_ok)
            if not step3_ok:
                success = False
        
        # Step 3 -> Step 4: P = I² * R
        if self.P is not None and self.I is not None:
            step4_ok = self._is_close(self.P, self.I**2 * self.R)
            self._log(f"Step 3→4: P = I²R: {self.P:.3f} W vs {self.I**2*self.R:.3f} W", step4_ok)
            if not step4_ok:
                success = False
        
        # Check final answers against target values (from problem statement)
        target_E_m = 8.0
        target_P = 2.88
        target_Q = 0.004
        
        if self.E_m is not None:
            final_E_ok = self._is_close(self.E_m, target_E_m)
            self._log(f"Final E_m: {self.E_m:.2f} V vs target {target_E_m} V", final_E_ok)
            if not final_E_ok:
                success = False
        
        if self.P is not None:
            final_P_ok = self._is_close(self.P, target_P)
            self._log(f"Final P: {self.P:.3f} W vs target {target_P} W", final_P_ok)
            if not final_P_ok:
                success = False
        
        if self.Q is not None:
            final_Q_ok = self._is_close(self.Q, target_Q)
            self._log(f"Final Q: {self.Q:.4f} C vs target {target_Q} C", final_Q_ok)
            if not final_Q_ok:
                success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb an input and check if outputs change as expected."""
        success = True
        
        # Store original values
        n_orig = self.n
        perturb_factor = 1.1  # 10% increase
        
        try:
            # Perturb number of turns (n) by 10%
            self.n = n_orig * perturb_factor
            
            # Recompute key quantities using formulas from steps
            # 1. Recompute E_m: E_m = n * B_m * S * (2π/T)
            # We need B_m * (2π/T) from original data
            if self.E_m is not None:
                B_m_2pi_over_T = self.E_m / (n_orig * self.S)
                E_m_new = self.n * self.S * B_m_2pi_over_T
                
                # 2. Recompute I_m = E_m / (R + r)
                I_m_new = E_m_new / (self.R + self.r)
                
                # 3. Recompute I = I_m / √2
                I_new = I_m_new / self.sqrt2
                
                # 4. Recompute P = I² * R
                P_new = I_new**2 * self.R
                
                # 5. Recompute Q = n * S * ΔB / (R + r)
                # First compute ΔB from original Q
                if self.Q is not None:
                    delta_B = self.Q * (self.R + self.r) / (n_orig * self.S)
                    Q_new = self.n * self.S * delta_B / (self.R + self.r)
                    
                    # Check monotonicity: all should increase when n increases
                    monotonic_E = E_m_new > self.E_m
                    monotonic_I_m = I_m_new > self.I_m
                    monotonic_I = I_new > self.I
                    monotonic_P = P_new > self.P
                    monotonic_Q = Q_new > self.Q
                    
                    self._log(f"Monotonicity (n↑→E_m↑): {E_m_new:.3f} > {self.E_m:.3f} V", monotonic_E)
                    self._log(f"Monotonicity (n↑→I_m↑): {I_m_new:.3f} > {self.I_m:.3f} A", monotonic_I_m)
                    self._log(f"Monotonicity (n↑→I↑): {I_new:.3f} > {self.I:.3f} A", monotonic_I)
                    self._log(f"Monotonicity (n↑→P↑): {P_new:.3f} > {self.P:.3f} W", monotonic_P)
                    self._log(f"Monotonicity (n↑→Q↑): {Q_new:.4f} > {self.Q:.4f} C", monotonic_Q)
                    
                    if not all([monotonic_E, monotonic_I_m, monotonic_I, monotonic_P, monotonic_Q]):
                        success = False
                else:
                    self._log("Skipping Q monotonicity: original Q is None", False)
                    success = False
            else:
                self._log("Skipping monotonicity: original E_m is None", False)
                success = False
                
        finally:
            # Restore original value
            self.n = n_orig
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test failed with exception: {str(e)}", False)
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs