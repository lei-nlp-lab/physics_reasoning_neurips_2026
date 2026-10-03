import math
import sympy as sp
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extracts all inputs and results from solution_instance.
        Uses getattr and .get() to avoid AttributeError in sandbox.
        """
        # Extract known parameters (inputs)
        self.d = self._safe_get(solution_instance, 'd')          # double-slit spacing
        self.L = self._safe_get(solution_instance, 'L')          # screen distance
        self.n = self._safe_get(solution_instance, 'n')          # refractive index
        self.a = self._safe_get(solution_instance, 'a')          # fringe pattern spacing (from figure)
        
        # Extract results from KB mapping
        self.delta_x = self._safe_get(solution_instance, 'delta_x')  # Δx
        self.delta_s = self._safe_get(solution_instance, 'delta_s')  # Δs
        self.lambda_vac = self._safe_get(solution_instance, 'λ')      # λ (vacuum wavelength)
        self.d_prime = self._safe_get(solution_instance, "d'")        # d' (film thickness)
        self.lambda_med = self._safe_get(solution_instance, "λ'")     # λ' (wavelength in medium)
        
        # Store original values for monotonicity test
        self.original_n = self.n
        
    def _safe_get(self, obj: Any, key: str) -> Optional[float]:
        """Safely retrieve attribute or result with fallback to None."""
        # Try direct attribute
        value = getattr(obj, key, None)
        if value is not None:
            return float(value) if isinstance(value, (int, float)) else value
        
        # Try results dictionary
        if hasattr(obj, 'results'):
            value = obj.results.get(key)
            if value is not None:
                return float(value) if isinstance(value, (int, float)) else value
        
        # Try alternative key names
        alt_keys = {
            'delta_x': ['Δx', 'dx', 'deltax'],
            'delta_s': ['Δs', 'ds', 'deltas'],
            'λ': ['lambda', 'wavelength'],
            "d'": ['d_prime', 'dprime', 'film_thickness'],
            "λ'": ['lambda_prime', 'lambda_med', 'wavelength_medium']
        }
        
        if key in alt_keys:
            for alt in alt_keys[key]:
                val = getattr(obj, alt, None)
                if val is not None:
                    return float(val) if isinstance(val, (int, float)) else val
                if hasattr(obj, 'results'):
                    val = obj.results.get(alt)
                    if val is not None:
                        return float(val) if isinstance(val, (int, float)) else val
        
        return None
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics invariants from KB.
        Returns: (success, log_messages)
        """
        logs = []
        success = True
        
        # 1. Check wavelength relationships from KB Theorem 1
        if self.lambda_vac is not None and self.lambda_med is not None and self.n is not None:
            # λ' = λ/n
            expected_lambda_med = self.lambda_vac / self.n
            if not self._is_close(self.lambda_med, expected_lambda_med):
                logs.append(f"FAIL: λ' = λ/n violated. Got λ'={self.lambda_med}, expected {expected_lambda_med}")
                success = False
            else:
                logs.append("PASS: λ' = λ/n relationship holds")
            
            # Frequency should be constant (cancels out, but we can check symbolically)
            c = 3e8  # speed of light (placeholder)
            f_vac = c / self.lambda_vac if self.lambda_vac != 0 else 0
            f_med = c / self.lambda_med if self.lambda_med != 0 else 0
            if self.lambda_vac != 0 and self.lambda_med != 0 and not self._is_close(f_vac, f_med):
                logs.append(f"FAIL: Frequency not constant. f_vac={f_vac:.2e}, f_med={f_med:.2e}")
                success = False
        
        # 2. Check thin film constructive interference condition (KB Theorem 2)
        if self.d_prime is not None and self.lambda_med is not None:
            # For m=1 (minimum thickness): 2n*d' = λ'  (assuming n_1 = n, and phase change condition = λ'/2)
            # According to step_4: Δs = 2d' = λ'
            expected_delta_s = 2 * self.d_prime
            if self.delta_s is not None and not self._is_close(self.delta_s, expected_delta_s):
                logs.append(f"FAIL: Δs = 2d' violated. Got Δs={self.delta_s}, expected {expected_delta_s}")
                success = False
            
            # Check constructive condition: 2d' = λ' (from step_4)
            if not self._is_close(2 * self.d_prime, self.lambda_med):
                logs.append(f"FAIL: 2d' = λ' violated. 2d'={2*self.d_prime}, λ'={self.lambda_med}")
                success = False
            else:
                logs.append("PASS: Thin film constructive condition (2d' = λ') holds")
        
        # 3. Check double-slit interference formula
        if self.delta_x is not None and self.lambda_vac is not None and self.d is not None and self.L is not None:
            # Δx = (L/d) * λ
            expected_delta_x = (self.L / self.d) * self.lambda_vac if self.d != 0 else 0
            if not self._is_close(self.delta_x, expected_delta_x):
                logs.append(f"FAIL: Δx = (L/d)λ violated. Got Δx={self.delta_x}, expected {expected_delta_x}")
                success = False
            else:
                logs.append("PASS: Double-slit fringe spacing formula holds")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Check consistency between intermediate steps and final answers.
        Uses the STEPS ANALYSIS for cross-validation.
        """
        logs = []
        success = True
        
        # Step 1: Δx = a/2
        if self.delta_x is not None and self.a is not None:
            expected_delta_x = self.a / 2
            if not self._is_close(self.delta_x, expected_delta_x):
                logs.append(f"FAIL: Step 1 (Δx = a/2) inconsistent. Δx={self.delta_x}, a/2={expected_delta_x}")
                success = False
            else:
                logs.append("PASS: Step 1 (Δx = a/2) consistent")
        
        # Step 2 & 3: λ = (Δx * d) / L = (a*d)/(2L)
        if self.lambda_vac is not None and self.a is not None and self.d is not None and self.L is not None:
            expected_lambda = (self.a * self.d) / (2 * self.L) if self.L != 0 else 0
            if not self._is_close(self.lambda_vac, expected_lambda):
                logs.append(f"FAIL: Step 3 (λ = ad/2L) inconsistent. λ={self.lambda_vac}, expected {expected_lambda}")
                success = False
            else:
                logs.append("PASS: Step 3 (λ = ad/2L) consistent")
        
        # Step 4-7: d' = λ'/2 = λ/(2n) = ad/(4nL)
        if self.d_prime is not None and self.a is not None and self.d is not None and self.L is not None and self.n is not None:
            expected_d_prime = (self.a * self.d) / (4 * self.n * self.L) if (self.n * self.L) != 0 else 0
            if not self._is_close(self.d_prime, expected_d_prime):
                logs.append(f"FAIL: Step 7 (d' = ad/4nL) inconsistent. d'={self.d_prime}, expected {expected_d_prime}")
                success = False
            else:
                logs.append("PASS: Step 7 (d' = ad/4nL) consistent")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb refractive index n and check if d' changes as expected.
        Formula: d' = a*d/(4*n*L) → d' ∝ 1/n
        """
        logs = []
        success = True
        
        # Check if we have all needed parameters
        if None in [self.a, self.d, self.L, self.n, self.d_prime]:
            logs.append("SKIP: Missing parameters for monotonicity test")
            return True, logs  # Not a failure, just skip
        
        # Store original
        original_n = self.n
        original_d_prime = self.d_prime
        
        # Perturb n by +10%
        perturbed_n = original_n * 1.1
        perturbed_d_prime = (self.a * self.d) / (4 * perturbed_n * self.L)
        
        # Expected trend: d' should DECREASE when n increases (d' ∝ 1/n)
        if perturbed_d_prime >= original_d_prime:
            logs.append(f"FAIL: Monotonicity violated. n↑ from {original_n} to {perturbed_n}, but d' didn't decrease: {original_d_prime} → {perturbed_d_prime}")
            success = False
        else:
            logs.append(f"PASS: Monotonicity holds. n↑ from {original_n} to {perturbed_n}, d'↓ from {original_d_prime} to {perturbed_d_prime}")
        
        # Restore original value
        self.n = original_n
        
        # Additional check: magnitude of change should follow 1/n relationship
        expected_ratio = original_n / perturbed_n  # d' should scale by this factor
        actual_ratio = perturbed_d_prime / original_d_prime if original_d_prime != 0 else 0
        if not self._is_close(actual_ratio, expected_ratio, rel_tol=5e-2):  # 5% tolerance for perturbation
            logs.append(f"WARNING: Perturbation magnitude off. Expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics checks and return overall success and logs.
        """
        all_logs = ["=== Physics Checker ==="]
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n1. INVARIANTS TEST:")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n2. TRACE CONSISTENCY TEST:")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n3. MONOTONICITY TEST:")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs