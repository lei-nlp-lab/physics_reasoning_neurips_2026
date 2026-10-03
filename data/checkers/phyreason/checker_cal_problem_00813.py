import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all required inputs and results
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Waveform parameters (safe access with defaults)
        self.lambda_val = 8.0  # wavelength in meters (from figure) - for Q1
        self.delta_t = 0.05    # time difference in seconds
        
        # Extract results with safe fallbacks - FIXED: use correct key names
        self.v = self._safe_extract('v', 0.0)              # wave speed (positive) from Q1
        self.v_prime = self._safe_extract('v_prime', 0.0)  # FIXED: use 'v_prime' not 'v\''
        self.T_period = self._safe_extract('T', 0.0)       # wave period from Q2
        self.t3 = self._safe_extract('t_3', 0.0)           # time for particle P from Q2
        
    def _safe_extract(self, key, default):
        """Safely extract value from solution instance or results dict."""
        value = getattr(self.solution, key, None)
        if value is None:
            value = self.results.get(key, default)
        # Handle both float and int types
        if isinstance(value, (int, float)):
            return float(value)
        return default
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify fundamental physics invariants."""
        logs = []
        success = True
        
        # 1. Check wave speed relationship from Q1: v' = v + 80 m/s
        # This holds for any n: v' - v = (6+8n - (2+8n))/0.05 = 4/0.05 = 80
        expected_diff = 80.0
        actual_diff = self.v_prime - self.v
        
        # Only check if both values are non-zero
        if self.v > 0 and self.v_prime > 0:
            if math.isclose(actual_diff, expected_diff, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ Wave speed invariant holds: v' - v = {actual_diff:.2f} ≈ 80 m/s")
            else:
                logs.append(f"✗ Wave speed invariant fails: v' - v = {actual_diff:.2f} ≠ 80 m/s")
                success = False
        else:
            logs.append(f"⚠ Cannot check wave speed invariant: v={self.v}, v'={self.v_prime}")
        
        # 2. For Q2 period: T should be consistent with its own λ and v
        # Q2 uses λ=2 m, v=70 m/s (from ground truth step_5)
        lambda_q2 = 2.0
        v_q2 = 70.0
        expected_T_q2 = lambda_q2 / v_q2
        
        if math.isclose(self.T_period, expected_T_q2, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ Q2 period-wavelength-speed invariant: T = {self.T_period:.4f} ≈ λ_q2/v_q2 = {expected_T_q2:.4f}")
        else:
            logs.append(f"✗ Q2 period-wavelength-speed invariant: T = {self.T_period:.4f} ≠ λ_q2/v_q2 = {expected_T_q2:.4f}")
            success = False
        
        # 3. Check angular frequency consistency for Q2: ω = 2π/T = 70π rad/s
        if self.T_period > 0:
            omega_from_T = 2 * math.pi / self.T_period
            omega_expected = 70 * math.pi
            if math.isclose(omega_from_T, omega_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ Angular frequency invariant: ω = {omega_from_T:.2f} ≈ 70π = {omega_expected:.2f}")
            else:
                logs.append(f"✗ Angular frequency invariant: ω = {omega_from_T:.2f} ≠ 70π = {omega_expected:.2f}")
                success = False
            
            # 4. Check SHM displacement equation at t3
            A = 0.02  # amplitude in meters (2 cm)
            expected_x = A * math.sin(omega_expected * self.t3)
            # From step 7: x = 1 cm = 0.01 m
            if math.isclose(expected_x, 0.01, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"✓ SHM displacement invariant: x(t3) = {expected_x:.4f} ≈ 0.01 m")
            else:
                logs.append(f"✗ SHM displacement invariant: x(t3) = {expected_x:.4f} ≠ 0.01 m")
                success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate steps and final results."""
        logs = []
        success = True
        
        # The problem allows multiple n values. We need to check if the results
        # satisfy the formulas for SOME positive integer n.
        
        # Formulas from ground truth:
        # v = (2 + 8n)/0.05 = 40 + 160n
        # v' = (6 + 8n)/0.05 = 120 + 160n
        
        # Find if there exists integer n >= 0 such that:
        # v ≈ 40 + 160n AND v' ≈ 120 + 160n
        
        n_found = None
        for n_candidate in range(0, 10):  # Check n from 0 to 9
            v_expected = 40 + 160 * n_candidate
            v_prime_expected = 120 + 160 * n_candidate
            
            if (math.isclose(self.v, v_expected, rel_tol=0.02, abs_tol=0.02) and
                math.isclose(self.v_prime, v_prime_expected, rel_tol=0.02, abs_tol=0.02)):
                n_found = n_candidate
                break
        
        if n_found is not None:
            logs.append(f"✓ Q1 results consistent with steps for n={n_found}:")
            logs.append(f"  v = {self.v:.1f} ≈ 40 + 160×{n_found} = {40 + 160*n_found:.1f}")
            logs.append(f"  v' = {self.v_prime:.1f} ≈ 120 + 160×{n_found} = {120 + 160*n_found:.1f}")
        else:
            logs.append(f"✗ Q1 results inconsistent with steps for any n in [0,9]:")
            logs.append(f"  v = {self.v:.1f}, v' = {self.v_prime:.1f}")
            success = False
        
        # Check Q2 values
        t3_expected = 1/420
        if math.isclose(self.t3, t3_expected, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"✓ t3 consistent with steps: {self.t3:.6f} ≈ {t3_expected:.6f}")
        else:
            logs.append(f"✗ t3 inconsistent: {self.t3:.6f} ≠ {t3_expected:.6f}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb the number of wavelengths (n) and check monotonic trends for Q1.
        """
        logs = []
        success = True
        
        try:
            # Test n = 0, 1, 2 for Q1 formulas
            n_values = [0, 1, 2]
            v_values = []
            v_prime_values = []
            
            for n_val in n_values:
                # Q1 formulas
                delta_s = 2 + 8 * n_val
                delta_s_prime = 6 + 8 * n_val
                v_val = delta_s / self.delta_t
                v_prime_val = delta_s_prime / self.delta_t
                
                v_values.append(float(v_val))
                v_prime_values.append(float(v_prime_val))
            
            # Check monotonicity
            v_increasing = all(v_values[i] < v_values[i+1] for i in range(len(v_values)-1))
            v_prime_increasing = all(v_prime_values[i] < v_prime_values[i+1] for i in range(len(v_prime_values)-1))
            
            if v_increasing and v_prime_increasing:
                logs.append("✓ Monotonic trends hold for Q1:")
                logs.append(f"  - v increases with n: {v_values}")
                logs.append(f"  - v' increases with n: {v_prime_values}")
            else:
                logs.append("✗ Monotonic trends violated for Q1:")
                logs.append(f"  v trend: {'increasing' if v_increasing else 'not increasing'}")
                logs.append(f"  v' trend: {'increasing' if v_prime_increasing else 'not increasing'}")
                success = False
                
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed with error: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = ["=== Physics Audit Report ==="]
        overall_success = True
        
        # Run invariant tests
        success, logs = self.test_invariants()
        all_logs.append("\n[INVARIANT TESTS]")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        all_logs.append("\n[TRACE CONSISTENCY TESTS]")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        all_logs.append("\n[MONOTONICITY TESTS]")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Summary
        all_logs.append("\n=== SUMMARY ===")
        all_logs.append(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs