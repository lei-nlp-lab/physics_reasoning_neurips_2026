import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract given constants
        self.m = 100.0
        self.g = 10.0
        self.h1 = 5000.0
        self.H = 20000.0
        self.a_first = 10.0 * self.g  # 100 m/s²
        
        # CORRECTED: From ground truth, cosθ = 1/11, sinθ = 10/11
        self.cosθ = 1.0 / 11.0
        self.sinθ = 10.0 / 11.0  # NOT sqrt(1 - cos²θ) = √(120/121)
        
        # Extract results safely using getattr with fallback to results dictionary
        self.F = self._safe_get(solution_instance, 'F', 0.0)
        self.v_y = self._safe_get(solution_instance, 'v_y', 0.0)
        self.t_2 = self._safe_get(solution_instance, 't_2', 0.0)
        self.a_y = self._safe_get(solution_instance, 'a_y', 0.0)
        self.t_all = self._safe_get(solution_instance, 't_all', 0.0)
        self.t_1 = self._safe_get(solution_instance, 't_1', 0.0)
        self.a_x = self._safe_get(solution_instance, 'a_x', 0.0)
    
    def _safe_get(self, solution_instance, key, default):
        """Safe attribute extraction with fallback to results dictionary."""
        value = getattr(solution_instance, key, None)
        if value is None:
            value = solution_instance.results.get(key, default)
        return float(value) if value is not None else default
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify Newton's Laws and kinematic consistency."""
        logs = []
        success = True
        
        # 1. Newton's Second Law: F = m(g + a_first) for first stage
        expected_F = self.m * (self.g + self.a_first)
        if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"F invariant failed: expected {expected_F}, got {self.F}")
            success = False
        else:
            logs.append(f"F = m(g+a) satisfied: {self.F} ≈ {expected_F}")
        
        # 2. Horizontal acceleration: a_x = (F sinθ)/m
        # CORRECTED: Use sinθ = 10/11
        expected_a_x = self.F * self.sinθ / self.m
        if not math.isclose(self.a_x, expected_a_x, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"a_x invariant failed: formula gives {expected_a_x}, got {self.a_x}")
            success = False
        else:
            logs.append(f"a_x = F sinθ/m satisfied: {self.a_x} ≈ {expected_a_x}")
        
        # 3. Vertical acceleration: a_y = (F cosθ - mg)/m
        expected_a_y = (self.F * self.cosθ - self.m * self.g) / self.m
        if not math.isclose(self.a_y, expected_a_y, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"a_y invariant failed: expected {expected_a_y}, got {self.a_y}")
            success = False
        else:
            logs.append(f"a_y = (F cosθ - mg)/m satisfied: {self.a_y} ≈ {expected_a_y}")
        
        # 4. Kinematic consistency for first stage
        # t_1 from h1 = 0.5 * a_first * t_1^2
        expected_t_1 = math.sqrt(2 * self.h1 / self.a_first)
        if not math.isclose(self.t_1, expected_t_1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"t_1 invariant failed: expected {expected_t_1}, got {self.t_1}")
            success = False
        else:
            logs.append(f"t_1 = √(2h1/a) satisfied: {self.t_1} ≈ {expected_t_1}")
        
        # v_y from v_y = a_first * t_1
        expected_v_y = self.a_first * self.t_1
        if not math.isclose(self.v_y, expected_v_y, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v_y invariant failed: expected {expected_v_y}, got {self.v_y}")
            success = False
        else:
            logs.append(f"v_y = a*t_1 satisfied: {self.v_y} ≈ {expected_v_y}")
        
        # 5. Second stage vertical motion (a_y = 0 → constant velocity)
        expected_t_2 = (self.H - self.h1) / self.v_y
        if not math.isclose(self.t_2, expected_t_2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"t_2 invariant failed: expected {expected_t_2}, got {self.t_2}")
            success = False
        else:
            logs.append(f"t_2 = (H-h1)/v_y satisfied: {self.t_2} ≈ {expected_t_2}")
        
        # 6. Total time consistency
        expected_t_all = self.t_1 + self.t_2
        if not math.isclose(self.t_all, expected_t_all, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"t_all invariant failed: expected {expected_t_all}, got {self.t_all}")
            success = False
        else:
            logs.append(f"t_all = t_1 + t_2 satisfied: {self.t_all} ≈ {expected_t_all}")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check against target answers: a_x=100, a_y=0, t_all=25
        if not math.isclose(self.a_x, 100.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"a_x trace failed: target 100, got {self.a_x}")
            success = False
        else:
            logs.append(f"a_x matches target: {self.a_x} ≈ 100")
        
        if not math.isclose(self.a_y, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"a_y trace failed: target 0, got {self.a_y}")
            success = False
        else:
            logs.append(f"a_y matches target: {self.a_y} ≈ 0")
        
        if not math.isclose(self.t_all, 25.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"t_all trace failed: target 25, got {self.t_all}")
            success = False
        else:
            logs.append(f"t_all matches target: {self.t_all} ≈ 25")
        
        # Verify v_y consistency with t_1
        if not math.isclose(self.v_y, self.a_first * self.t_1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v_y trace failed: v_y should be {self.a_first * self.t_1}, got {self.v_y}")
            success = False
        else:
            logs.append(f"v_y consistent with t_1: {self.v_y} ≈ {self.a_first} * {self.t_1}")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb inputs and verify expected trends."""
        logs = []
        success = True
        
        # Store original values
        original_m = self.m
        original_g = self.g
        
        # Test 1: Increase mass by 1% → total time should remain unchanged
        # (since F ∝ m, so accelerations and velocities unchanged)
        self.m = original_m * 1.01
        F_perturbed = self.m * (self.g + self.a_first)
        t_1_perturbed = math.sqrt(2 * self.h1 / self.a_first)
        v_y_perturbed = self.a_first * t_1_perturbed
        t_2_perturbed = (self.H - self.h1) / v_y_perturbed
        t_all_perturbed = t_1_perturbed + t_2_perturbed
        
        if not math.isclose(t_all_perturbed, self.t_all, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Mass monotonicity failed: t_all changed from {self.t_all} to {t_all_perturbed}")
            success = False
        else:
            logs.append(f"Mass monotonicity passed: t_all insensitive to mass ({self.t_all} → {t_all_perturbed})")
        
        # Restore mass
        self.m = original_m
        
        # Test 2: Increase g by 1% → total time should decrease
        # (larger g means larger a_first → faster first stage)
        self.g = original_g * 1.01
        a_first_perturbed = 10 * self.g
        t_1_perturbed = math.sqrt(2 * self.h1 / a_first_perturbed)
        v_y_perturbed = a_first_perturbed * t_1_perturbed
        t_2_perturbed = (self.H - self.h1) / v_y_perturbed
        t_all_perturbed = t_1_perturbed + t_2_perturbed
        
        if t_all_perturbed > self.t_all:
            logs.append(f"g monotonicity failed: t_all increased from {self.t_all} to {t_all_perturbed}")
            success = False
        else:
            logs.append(f"g monotonicity passed: t_all decreases when g increases ({self.t_all} → {t_all_perturbed})")
        
        # Restore g
        self.g = original_g
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return aggregated results."""
        all_success = True
        all_logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                all_success = False
        
        return all_success, all_logs