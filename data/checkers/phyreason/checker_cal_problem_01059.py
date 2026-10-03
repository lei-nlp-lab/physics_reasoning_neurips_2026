import math
import sympy as sp
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    """Robust physics auditor for sprint photo problem."""
    
    def __init__(self, solution_instance: Any):
        """Extract inputs and results safely."""
        # Problem inputs (fixed)
        self.actual_height = 1.70          # m
        self.photo_height = 1.7e-2         # m
        self.blur_width_photo = 2e-3       # m
        self.exposure_time = 1.0 / 60.0    # s
        
        # Extract results from solution instance with safe fallbacks
        self.scale_ratio = getattr(solution_instance, 'scale_ratio', 
                                 getattr(solution_instance, 'results', {}).get('scale_ratio', 0.0))
        self.v_avg = getattr(solution_instance, 'v_avg', 
                           getattr(solution_instance, 'results', {}).get('v_avg', 0.0))
        self.x = getattr(solution_instance, 'x', 
                       getattr(solution_instance, 'results', {}).get('x', 0.0))
        self.v = getattr(solution_instance, 'v', 
                       getattr(solution_instance, 'results', {}).get('v', 0.0))
        
        # Store original values for monotonicity test
        self._original_x = self.x
        self._original_v = self.v
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def _isclose(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physics laws and internal consistency."""
        logs = []
        success = True
        
        # 1. Check scale ratio: actual_height / photo_height
        expected_scale = self.actual_height / self.photo_height
        if not self._isclose(self.scale_ratio, expected_scale):
            logs.append(f"FAIL: scale_ratio = {self.scale_ratio}, expected ≈ {expected_scale}")
            success = False
        else:
            logs.append(f"PASS: scale_ratio = {self.scale_ratio} ≈ {expected_scale}")
        
        # 2. Check displacement: x = blur_width_photo * scale_ratio
        expected_x = self.blur_width_photo * self.scale_ratio
        if not self._isclose(self.x, expected_x):
            logs.append(f"FAIL: displacement x = {self.x}, expected ≈ {expected_x}")
            success = False
        else:
            logs.append(f"PASS: displacement x = {self.x} ≈ {expected_x}")
        
        # 3. Check average velocity: v_avg = x / exposure_time
        expected_v_avg = self.x / self.exposure_time
        if not self._isclose(self.v_avg, expected_v_avg):
            logs.append(f"FAIL: v_avg = {self.v_avg}, expected ≈ {expected_v_avg}")
            success = False
        else:
            logs.append(f"PASS: v_avg = {self.v_avg} ≈ {expected_v_avg}")
        
        # 4. Check instantaneous velocity approximation: v ≈ v_avg
        if not self._isclose(self.v, self.v_avg):
            logs.append(f"FAIL: v = {self.v} ≠ v_avg = {self.v_avg}")
            success = False
        else:
            logs.append(f"PASS: v = {self.v} ≈ v_avg = {self.v_avg}")
        
        # 5. Symbolic verification (optional fallback)
        try:
            scale_sym = sp.symbols('scale_sym')
            x_sym = sp.symbols('x_sym')
            v_avg_sym = sp.symbols('v_avg_sym')
            v_sym = sp.symbols('v_sym')
            
            eq1 = sp.Eq(scale_sym, self.actual_height / self.photo_height)
            eq2 = sp.Eq(x_sym, self.blur_width_photo * scale_sym)
            eq3 = sp.Eq(v_avg_sym, x_sym / self.exposure_time)
            eq4 = sp.Eq(v_sym, v_avg_sym)
            
            # Check if numerical values satisfy symbolic equations
            sol = {scale_sym: self.scale_ratio, x_sym: self.x, 
                   v_avg_sym: self.v_avg, v_sym: self.v}
            if (eq1.subs(sol).simplify() and eq2.subs(sol).simplify() and 
                eq3.subs(sol).simplify() and eq4.subs(sol).simplify()):
                logs.append("PASS: All symbolic equations satisfied")
            else:
                logs.append("WARN: Symbolic verification inconclusive")
        except Exception as e:
            logs.append(f"INFO: Symbolic check skipped due to {type(e).__name__}")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Target answers from problem statement
        target_x = 0.2  # m
        target_v = 12.0  # m/s
        
        # Check against computed values
        if not self._isclose(self.x, target_x):
            logs.append(f"FAIL: displacement x = {self.x}, target = {target_x}")
            success = False
        else:
            logs.append(f"PASS: displacement x matches target {target_x} m")
        
        if not self._isclose(self.v, target_v):
            logs.append(f"FAIL: velocity v = {self.v}, target = {target_v}")
            success = False
        else:
            logs.append(f"PASS: velocity v matches target {target_v} m/s")
        
        # Consistency between steps (using steps analysis)
        # Step 1 -> Step 2: x = blur_width_photo * scale_ratio
        computed_x = self.blur_width_photo * self.scale_ratio
        if not self._isclose(self.x, computed_x):
            logs.append(f"FAIL: Step 2 inconsistency: x = {self.x} ≠ {computed_x}")
            success = False
        
        # Step 2 -> Step 3: v_avg = x / exposure_time
        computed_v_avg = self.x / self.exposure_time
        if not self._isclose(self.v_avg, computed_v_avg):
            logs.append(f"FAIL: Step 3 inconsistency: v_avg = {self.v_avg} ≠ {computed_v_avg}")
            success = False
        
        # Step 3 -> Step 4: v = v_avg
        if not self._isclose(self.v, self.v_avg):
            logs.append(f"FAIL: Step 4 inconsistency: v = {self.v} ≠ v_avg = {self.v_avg}")
            success = False
        
        if success:
            logs.append("PASS: All trace steps are consistent")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb an input and verify expected trend."""
        logs = []
        success = True
        
        # Perturb blur width by +10%
        perturbed_blur = self.blur_width_photo * 1.1
        
        # Recalculate using formulas from steps analysis
        # Step 1: scale_ratio unchanged (depends on heights)
        # Step 2: x' = perturbed_blur * scale_ratio
        new_x = perturbed_blur * self.scale_ratio
        # Step 3: v_avg' = x' / exposure_time
        new_v_avg = new_x / self.exposure_time
        # Step 4: v' = v_avg'
        new_v = new_v_avg
        
        # Expected trend: increasing blur width -> increasing displacement and velocity
        if not (new_x > self._original_x and new_v > self._original_v):
            logs.append(f"FAIL: Monotonicity violated. Original x={self._original_x}, v={self._original_v}. "
                       f"After +10% blur: x'={new_x}, v'={new_v}")
            success = False
        else:
            logs.append(f"PASS: Monotonicity holds. Increasing blur width increases x and v.")
        
        # Restore original values (though we didn't modify instance variables)
        # No need as we used local variables
        
        # Additional check: perturb exposure time (increase by 10%)
        perturbed_time = self.exposure_time * 1.1
        new_v_avg_time = self.x / perturbed_time  # x unchanged
        if not (new_v_avg_time < self.v_avg):
            logs.append(f"FAIL: Monotonicity violated for exposure time. "
                       f"Original v_avg={self.v_avg}, after +10% time: v_avg'={new_v_avg_time}")
            success = False
        else:
            logs.append(f"PASS: Increasing exposure time decreases average velocity.")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and collect logs."""
        all_logs = ["=== Physics Checker Started ==="]
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n--- Invariants Test ---")
        all_logs.extend(logs1)
        overall_success &= success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n--- Trace Consistency Test ---")
        all_logs.extend(logs2)
        overall_success &= success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n--- Monotonicity Test ---")
        all_logs.extend(logs3)
        overall_success &= success3
        
        # Summary
        all_logs.append("\n=== Summary ===")
        if overall_success:
            all_logs.append("SUCCESS: All physics checks passed.")
        else:
            all_logs.append("FAILURE: One or more checks failed.")
        
        return overall_success, all_logs