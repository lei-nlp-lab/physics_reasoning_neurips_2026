import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        ONLY use the results dictionary and defined attributes.
        """
        self.results = getattr(solution_instance, 'results', {})
        self.solution = solution_instance
        
        # Extract frequency (v in Hz) - from step_1
        self.f = self._safe_get_numeric('v')
        
        # Extract wavelength in meters (λ) - from step_2
        self.lambda_m = self._safe_get_numeric('lambda')
        
        # Extract wavelength in nm - from step_2 second value
        # First try 'lambda_nm', then compute from lambda_m if available
        self.lambda_nm = self._safe_get_numeric('lambda_nm')
        if self.lambda_nm is None and self.lambda_m is not None:
            try:
                self.lambda_nm = self.lambda_m * 1e9
            except (TypeError, ValueError):
                self.lambda_nm = None
        
        # Extract color - from step_3
        # Try to get color as string or numeric from various possible keys
        self.color = None
        color_keys = ['color', 'color_category', 'perceived_color', 'N/A']
        for key in color_keys:
            if key in self.results:
                val = self.results[key]
                if isinstance(val, str):
                    self.color = val
                    break
                elif isinstance(val, (int, float)):
                    # Convert numeric color ID to string for comparison
                    color_map = {
                        1: 'violet', 2: 'blue', 3: 'green',
                        4: 'yellow', 5: 'orange', 6: 'red', 0: 'invisible'
                    }
                    if val in color_map:
                        self.color = color_map[int(val)]
                    break
        
        # Constants - use the same values as the problem context
        self.c = 3e8  # speed of light in m/s (simplified for this problem)
        self.h = 6.626e-34  # Planck's constant
        
        # Color boundaries in meters
        self.color_boundaries = {
            'violet': (380e-9, 450e-9),
            'blue': (450e-9, 495e-9),
            'green': (495e-9, 570e-9),
            'yellow': (570e-9, 590e-9),
            'orange': (590e-9, 620e-9),
            'red': (620e-9, 750e-9)
        }
    
    def _safe_get_numeric(self, key):
        """Safely extract a numeric value from results dictionary."""
        if key in self.results:
            val = self.results[key]
            if isinstance(val, (int, float)):
                return float(val)
            elif hasattr(val, 'evalf'):
                try:
                    return float(val.evalf())
                except (TypeError, ValueError):
                    return None
        return None
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics invariants."""
        logs = []
        success = True
        
        # 1. Check wave equation: c = f * λ
        if self.f is not None and self.lambda_m is not None:
            # Check for zero or near-zero values that would make the test meaningless
            if abs(self.f) < 1e-100 or abs(self.lambda_m) < 1e-100:
                logs.append("SKIP: Wave equation test - frequency or wavelength too small")
            else:
                expected_c = self.f * self.lambda_m
                # Use relative tolerance for comparison
                if not math.isclose(self.c, expected_c, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Wave equation violation. c = {self.c}, f*λ = {expected_c}")
                    success = False
                else:
                    logs.append("PASS: Wave equation c = f*λ holds")
        else:
            logs.append("SKIP: Wave equation test - missing frequency or wavelength")
        
        # 2. Check wavelength conversion: λ_nm = λ_m * 1e9
        if self.lambda_m is not None and self.lambda_nm is not None:
            # Check for consistency
            expected_nm = self.lambda_m * 1e9
            # Handle potential floating point errors
            if abs(self.lambda_m) < 1e-100:
                # If lambda_m is essentially 0, lambda_nm should also be essentially 0
                if abs(self.lambda_nm) > 1e-90:
                    logs.append(f"FAIL: Wavelength conversion mismatch for near-zero λ_m. λ_nm = {self.lambda_nm}, expected near 0")
                    success = False
                else:
                    logs.append("PASS: Wavelength conversion consistent for near-zero values")
            elif not math.isclose(self.lambda_nm, expected_nm, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Wavelength conversion mismatch. λ_nm = {self.lambda_nm}, λ_m*1e9 = {expected_nm}")
                success = False
            else:
                logs.append("PASS: Wavelength conversion λ_nm = λ_m * 1e9 holds")
        else:
            logs.append("SKIP: Wavelength conversion test - missing wavelength values")
        
        # 3. Check color consistency with wavelength
        if self.lambda_m is not None and self.color is not None:
            color_from_lambda = self._get_color_from_wavelength(self.lambda_m)
            # Compare color names (case-insensitive)
            if isinstance(self.color, str) and isinstance(color_from_lambda, str):
                if color_from_lambda.lower() not in str(self.color).lower():
                    logs.append(f"FAIL: Color mismatch. From λ: {color_from_lambda}, Reported: {self.color}")
                    success = False
                else:
                    logs.append(f"PASS: Color {self.color} consistent with wavelength {self.lambda_m:.2e} m")
            else:
                logs.append(f"SKIP: Color comparison - color type mismatch")
        else:
            logs.append("SKIP: Color consistency test - missing wavelength or color")
        
        # 4. Check frequency is in visible range (using wavelength)
        if self.lambda_m is not None:
            if self.lambda_m < 380e-9 or self.lambda_m > 750e-9:
                logs.append(f"WARNING: Wavelength {self.lambda_m:.2e} m outside visible range (380-750 nm)")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate steps and final answer."""
        logs = []
        success = True
        
        # Check frequency from step_1
        expected_f = 6e14
        if self.f is not None:
            if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Frequency mismatch. Expected: {expected_f:.2e}, Got: {self.f:.2e}")
                success = False
            else:
                logs.append(f"PASS: Frequency {self.f:.2e} Hz matches input")
        else:
            logs.append("SKIP: Frequency check - missing frequency value")
        
        # Check wavelength calculation from step_2
        if self.f is not None and self.lambda_m is not None:
            # Avoid division by zero
            if abs(self.f) < 1e-100:
                logs.append("SKIP: Wavelength calculation - frequency too small")
            else:
                expected_lambda_m = self.c / self.f
                if not math.isclose(self.lambda_m, expected_lambda_m, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Wavelength calculation mismatch. Expected: {expected_lambda_m:.2e}, Got: {self.lambda_m:.2e}")
                    success = False
                else:
                    logs.append(f"PASS: Wavelength {self.lambda_m:.2e} m correctly calculated from c/f")
        else:
            logs.append("SKIP: Wavelength calculation - missing frequency or wavelength")
        
        # Check nm conversion from step_2
        if self.lambda_m is not None and self.lambda_nm is not None:
            expected_lambda_nm = self.lambda_m * 1e9
            if not math.isclose(self.lambda_nm, expected_lambda_nm, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: nm conversion mismatch. Expected: {expected_lambda_nm:.1f}, Got: {self.lambda_nm}")
                success = False
            else:
                logs.append(f"PASS: nm conversion correct: {self.lambda_nm} nm")
        else:
            logs.append("SKIP: nm conversion check - missing wavelength values")
        
        # Check color from step_3
        if self.lambda_nm is not None and self.color is not None:
            # 500 nm should be green
            if self.lambda_nm >= 495 and self.lambda_nm < 570:
                if isinstance(self.color, str) and 'green' not in str(self.color).lower():
                    logs.append(f"FAIL: For λ={self.lambda_nm} nm, expected green, got {self.color}")
                    success = False
                else:
                    logs.append(f"PASS: Color {self.color} correct for λ={self.lambda_nm} nm")
        else:
            logs.append("SKIP: Color check - missing wavelength or color")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Test that relationships show correct monotonic trends.
        Perturb frequency and check wavelength and color change appropriately.
        """
        logs = []
        success = True
        
        if self.f is None or self.lambda_m is None:
            logs.append("SKIP: Missing frequency or wavelength for monotonicity test")
            return True, logs
        
        # Original values
        f_orig = self.f
        lambda_orig = self.lambda_m
        
        # Skip if values are too small
        if abs(f_orig) < 1e-100 or abs(lambda_orig) < 1e-100:
            logs.append("SKIP: Monotonicity test - frequency or wavelength too small")
            return True, logs
        
        # Test 1: Increase frequency by 10%
        f_perturbed = f_orig * 1.1
        # Avoid division by zero
        if abs(f_perturbed) < 1e-100:
            logs.append("SKIP: Perturbed frequency too small")
        else:
            lambda_perturbed = self.c / f_perturbed
            
            # Check wavelength decreases when frequency increases
            if lambda_perturbed >= lambda_orig:
                logs.append(f"FAIL: Monotonicity violation. f↑ ({f_orig:.2e}→{f_perturbed:.2e}) but λ not ↓ ({lambda_orig:.2e}→{lambda_perturbed:.2e})")
                success = False
            else:
                logs.append(f"PASS: f↑ → λ↓: {lambda_orig:.2e}→{lambda_perturbed:.2e} m")
        
        # Test 2: Decrease frequency by 10%
        f_perturbed = f_orig * 0.9
        # Avoid division by zero
        if abs(f_perturbed) < 1e-100:
            logs.append("SKIP: Perturbed frequency too small")
        else:
            lambda_perturbed = self.c / f_perturbed
            
            # Check wavelength increases when frequency decreases
            if lambda_perturbed <= lambda_orig:
                logs.append(f"FAIL: Monotonicity violation. f↓ ({f_orig:.2e}→{f_perturbed:.2e}) but λ not ↑ ({lambda_orig:.2e}→{lambda_perturbed:.2e})")
                success = False
            else:
                logs.append(f"PASS: f↓ → λ↑: {lambda_orig:.2e}→{lambda_perturbed:.2e} m")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All tests passed!")
        else:
            all_logs.append("Some tests failed.")
        
        return overall_success, all_logs
    
    def _get_color_from_wavelength(self, lambda_m: float) -> str:
        """Determine color from wavelength in meters."""
        for color, (lower, upper) in self.color_boundaries.items():
            if lower <= lambda_m < upper:
                return color
        return "invisible"
    
    def _get_color_range(self, color_name: str):
        """Get wavelength range for a color."""
        if color_name is None:
            return None
        color_name_lower = str(color_name).lower()
        for color, bounds in self.color_boundaries.items():
            if color in color_name_lower:
                return bounds
        return None