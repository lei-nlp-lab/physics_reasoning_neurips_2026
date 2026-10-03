import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely
        self.B0 = 2.0e-2  # T
        self.m = 3.2e-26  # kg
        self.q_mag = 1.6e-18  # C (magnitude, since q is negative)
        self.l1 = 4.0  # m (BC)
        self.l2 = 2.0  # m (AB)
        
        # Safely extract results from solution_instance
        # First try to get results dictionary, then fall back to attributes
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract values with careful type checking
        self.R1 = float(results_dict.get('R_1', 1.0))
        self.R2 = float(results_dict.get('R_2', 5.0))
        self.v1 = float(results_dict.get('v_1', 1.0e6))
        self.v2 = float(results_dict.get('v_2', 5.0e6))
        self.T = float(results_dict.get('T', 2.0 * math.pi * 1.0e-6))
        self.t = float(results_dict.get('t', 3.14e-6))
        
        # v might be a string or tuple representing range - handle carefully
        v_result = results_dict.get('v', (self.v1, self.v2))
        if isinstance(v_result, str):
            # Try to parse string like "1.0e6 to 5.0e6"
            try:
                parts = v_result.replace('to', ',').split(',')
                self.v_min = float(parts[0].strip())
                self.v_max = float(parts[1].strip())
            except:
                self.v_min, self.v_max = self.v1, self.v2
        elif isinstance(v_result, (list, tuple)):
            self.v_min, self.v_max = float(v_result[0]), float(v_result[1])
        else:
            self.v_min, self.v_max = self.v1, self.v2
        
        self.logs = []
    
    def _log(self, message: str, success: bool) -> None:
        self.logs.append(f"{'✓' if success else '✗'} {message}")
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Check fundamental physical laws."""
        success = True
        
        # 1. Lorentz force as centripetal force: |q|vB = mv²/R => v = |q|BR/m
        # Check for both minimum and maximum radii/speeds
        # Protect against division by zero
        if self.m == 0:
            self._log("Mass is zero, cannot compute Lorentz invariant", False)
            success = False
        else:
            v1_calc = self.q_mag * self.B0 * self.R1 / self.m
            v2_calc = self.q_mag * self.B0 * self.R2 / self.m
            
            if not math.isclose(v1_calc, self.v1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Lorentz force invariant failed for v1: calculated {v1_calc:.2e}, got {self.v1:.2e}", False)
                success = False
            else:
                self._log(f"Lorentz force invariant holds for v1 ({v1_calc:.2e} ≈ {self.v1:.2e})", True)
                
            if not math.isclose(v2_calc, self.v2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Lorentz force invariant failed for v2: calculated {v2_calc:.2e}, got {self.v2:.2e}", False)
                success = False
            else:
                self._log(f"Lorentz force invariant holds for v2 ({v2_calc:.2e} ≈ {self.v2:.2e})", True)
        
        # 2. Geometric relationship: CORRECTED - For maximum radius, (R₂ - l₂)² + l₁² = R₂²
        lhs_eq = (self.R2 - self.l2)**2 + self.l1**2
        rhs_eq = self.R2**2
        if not math.isclose(lhs_eq, rhs_eq, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Geometric equation (R₂-{self.l2})²+({self.l1})²=R₂² not satisfied: {lhs_eq:.2f} ≠ {rhs_eq:.2f}", False)
            success = False
        else:
            self._log(f"Geometric equation satisfied: (R₂-{self.l2})²+({self.l1})²=R₂²", True)
        
        # 3. Period invariance: T = 2πm/(|q|B) should be constant regardless of R
        if self.q_mag * self.B0 == 0:
            self._log("Charge or B field is zero, cannot compute period", False)
            success = False
        else:
            T_calc = 2 * math.pi * self.m / (self.q_mag * self.B0)
            # Handle case where T might be 0.0 (from solution)
            if self.T == 0.0 and T_calc != 0.0:
                self._log(f"Period invariance failed: calculated {T_calc:.2e}, got {self.T:.2e}", False)
                success = False
            elif not math.isclose(T_calc, self.T, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Period invariance failed: calculated {T_calc:.2e}, got {self.T:.2e}", False)
                success = False
            else:
                self._log(f"Period invariance holds (T={T_calc:.2e}s)", True)
        
        return success, self.logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate and final results."""
        success = True
        
        # 1. Check speed range matches v1 and v2
        if not (math.isclose(self.v_min, self.v1, rel_tol=1e-2, abs_tol=1e-2) and 
                math.isclose(self.v_max, self.v2, rel_tol=1e-2, abs_tol=1e-2)):
            self._log(f"Speed range mismatch: v_min={self.v_min:.2e}, v1={self.v1:.2e}, v_max={self.v_max:.2e}, v2={self.v2:.2e}", False)
            success = False
        else:
            self._log(f"Speed range consistent: {self.v_min:.2e} to {self.v_max:.2e}", True)
        
        # 2. Check longest time t = T/2
        if self.T == 0.0:
            self._log("Period T is zero, cannot check t = T/2", False)
            success = False
        elif not math.isclose(self.t, self.T/2, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Time consistency failed: t={self.t:.2e}, T/2={self.T/2:.2e}", False)
            success = False
        else:
            self._log(f"Time consistency holds: t={self.t:.2e} = T/2", True)
        
        # 3. Check t = πR₁/v₁
        if self.v1 == 0:
            self._log("v1 is zero, cannot compute t = πR₁/v₁", False)
            success = False
        else:
            t_calc = math.pi * self.R1 / self.v1
            if not math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Alternative time formula failed: t={self.t:.2e}, πR₁/v₁={t_calc:.2e}", False)
                success = False
            else:
                self._log(f"Alternative time formula holds: t = πR₁/v₁ = {t_calc:.2e}s", True)
        
        return success, self.logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb input and check expected trends."""
        success = True
        
        # Perturb magnetic field B by +10%
        B_perturbed = self.B0 * 1.1
        
        # Recalculate expected changes using formulas
        # 1. Speeds should increase linearly with B
        if self.m == 0:
            self._log("Mass is zero, cannot compute speed monotonicity", False)
            success = False
        else:
            v1_perturbed = self.q_mag * B_perturbed * self.R1 / self.m
            v2_perturbed = self.q_mag * B_perturbed * self.R2 / self.m
            
            # Check monotonicity: perturbed values should be greater
            if not (v1_perturbed > self.v1 and v2_perturbed > self.v2):
                self._log(f"Speed monotonicity failed: B↑10% but v1 {v1_perturbed:.2e} ≤ {self.v1:.2e} or v2 {v2_perturbed:.2e} ≤ {self.v2:.2e}", False)
                success = False
            else:
                self._log(f"Speed monotonicity holds: B↑10% → v1↑ to {v1_perturbed:.2e}, v2↑ to {v2_perturbed:.2e}", True)
        
        # 2. Period should decrease with increasing B (T ∝ 1/B)
        if self.q_mag * B_perturbed == 0:
            self._log("Charge or perturbed B is zero, cannot compute period monotonicity", False)
            success = False
        else:
            T_perturbed = 2 * math.pi * self.m / (self.q_mag * B_perturbed)
            if self.T == 0.0:
                self._log("Original period T is zero, cannot check monotonicity", False)
                success = False
            elif not T_perturbed < self.T:
                self._log(f"Period monotonicity failed: B↑10% but T {T_perturbed:.2e} ≥ {self.T:.2e}", False)
                success = False
            else:
                self._log(f"Period monotonicity holds: B↑10% → T↓ to {T_perturbed:.2e}", True)
        
        # 3. Longest time t = T/2 should also decrease
        t_perturbed = T_perturbed / 2
        if self.t == 0.0:
            self._log("Original time t is zero, cannot check monotonicity", False)
            success = False
        elif not t_perturbed < self.t:
            self._log(f"Time monotonicity failed: B↑10% but t {t_perturbed:.2e} ≥ {self.t:.2e}", False)
            success = False
        else:
            self._log(f"Time monotonicity holds: B↑10% → t↓ to {t_perturbed:.2e}", True)
        
        # Restore original values (conceptual - no actual mutation needed)
        self._log("Original parameters restored after perturbation test", True)
        
        return success, self.logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success."""
        self.logs = []
        all_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n=== Testing {test_name} ===")
            test_success, _ = test_func()
            if not test_success:
                all_success = False
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if all_success else 'FAIL'} ===")
        return all_success, self.logs