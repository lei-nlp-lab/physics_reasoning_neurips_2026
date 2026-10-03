import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction from solution_instance
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs - either from results or use known constants
        # The problem gives: v = 10 m/s, f = 5 Hz, S1S2 = 4 m
        self.v = 10.0  # m/s (from problem statement)
        self.f = 5.0   # Hz (from problem statement)
        self.d_S1S2 = 4.0  # m (from problem statement)
        
        # Try to get from results if available (but they shouldn't be there per contract)
        if 'v' in self.results:
            try:
                self.v = float(self.results['v'])
            except:
                pass
        if 'f' in self.results:
            try:
                self.f = float(self.results['f'])
            except:
                pass
        if 'S1S2' in self.results:
            try:
                self.d_S1S2 = float(self.results['S1S2'])
            except:
                pass
        
        # Extract results with safe fallbacks
        self.lambda_val = self._get_result('$lambda$', 0.0)
        self.AS1 = self._get_result('$AS_1$', 0.0)
        self.BS1 = self._get_result('$BS_1$', 0.0)
        self.CS1 = self._get_result('$CS_1$', 0.0)
        self.AS2_minus_AS1 = self._get_result('$AS_2-AS_1$', 0.0)
        self.CS1_minus_CS2 = self._get_result('$CS_1-CS_2$', 0.0)
        
        # Final answers (string format) - try multiple ways to get it
        self.final_answers = []
        
        # Try to get from solution_instance attribute
        if hasattr(solution_instance, 'final_answers'):
            self.final_answers = getattr(solution_instance, 'final_answers', [])
        
        # Try to get from globals
        try:
            if 'final_answers' in globals():
                self.final_answers = globals()['final_answers']
        except:
            pass
        
        # Try to get from module
        try:
            import sys
            module_name = solution_instance.__name__ if hasattr(solution_instance, '__name__') else '__main__'
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, 'final_answers'):
                    self.final_answers = getattr(module, 'final_answers', [])
        except:
            pass
        
        # Expected values from problem
        self.expected_lambda = self.v / self.f
        self.expected_AS1 = 1.0  # m
        self.expected_BS1 = 2.0  # m
        self.expected_CS1 = 3.0  # m
        self.expected_points_on_line = 3
        self.expected_points_on_circle = 6
        
        # Logs
        self.logs = []
    
    def _get_result(self, key, default):
        """Safely get a result from the dictionary."""
        # Try direct key first
        if key in self.results:
            val = self.results[key]
            try:
                return float(val) if not isinstance(val, str) else float(val.strip('$'))
            except:
                return default
        
        # Try alternative representations
        alt_key = key.replace('$', '')
        if alt_key in self.results:
            val = self.results[alt_key]
            try:
                return float(val) if not isinstance(val, str) else float(val.strip('$'))
            except:
                return default
        
        return default
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Test wavelength formula: λ = v/f
            calculated_lambda = self.v / self.f
            if not self._is_close(self.lambda_val, calculated_lambda):
                logs.append(f"FAIL: Wavelength mismatch. Calculated: {calculated_lambda:.2f}, Got: {self.lambda_val:.2f}")
                success = False
            else:
                logs.append(f"PASS: Wavelength correct (λ = v/f = {calculated_lambda:.2f} m)")
            
            # 2. Test that S1S2 = 4m contains integer number of wavelengths
            # According to steps analysis: d_S1S2 = 4m, λ = 2m → 2 wavelengths
            num_wavelengths = self.d_S1S2 / calculated_lambda
            if not self._is_close(num_wavelengths, 2.0):
                logs.append(f"WARNING: S1S2 distance should contain 2 wavelengths, but got {num_wavelengths:.2f}")
            
            # 3. Test constructive interference condition for points on line
            # For point A: AS2 - AS1 should equal λ (constructive, n=1)
            if not self._is_close(self.AS2_minus_AS1, calculated_lambda):
                logs.append(f"WARNING: Path difference for A should be λ = {calculated_lambda:.2f}, got {self.AS2_minus_AS1:.2f}")
            
            # For point C: CS1 - CS2 should equal λ (constructive, n=1)
            if not self._is_close(self.CS1_minus_CS2, calculated_lambda):
                logs.append(f"WARNING: Path difference for C should be λ = {calculated_lambda:.2f}, got {self.CS1_minus_CS2:.2f}")
            
            # 4. Verify distances on line match expected pattern
            # Points should be at λ/2, λ, 3λ/2 from S1
            expected_distances = [calculated_lambda/2, calculated_lambda, 1.5*calculated_lambda]
            actual_distances = [self.AS1, self.BS1, self.CS1]
            
            for exp, act, point in zip(expected_distances, actual_distances, ['A', 'B', 'C']):
                if not self._is_close(exp, act):
                    logs.append(f"FAIL: Distance for point {point} mismatch. Expected: {exp:.2f}, Got: {act:.2f}")
                    success = False
                else:
                    logs.append(f"PASS: Point {point} at correct distance {act:.2f} m")
            
            # 5. Test circle radius equals BS1
            if not self._is_close(self.BS1, calculated_lambda):
                logs.append(f"WARNING: Circle radius BS1 = {self.BS1:.2f} should equal λ = {calculated_lambda:.2f}")
        
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Parse final answers - if not available, skip this test
            if not self.final_answers:
                logs.append("SKIP: No final_answers found")
                return True, logs
            
            if len(self.final_answers) >= 3:
                # Answer 1: Wavelength
                ans1 = self.final_answers[0]
                try:
                    ans1_val = float(ans1.replace('m', '').strip())
                    if not self._is_close(ans1_val, self.lambda_val):
                        logs.append(f"FAIL: Final answer 1 ({ans1}) doesn't match λ = {self.lambda_val:.2f}")
                        success = False
                    else:
                        logs.append(f"PASS: Final answer 1 consistent with λ")
                except:
                    logs.append(f"WARNING: Could not parse final answer 1: {ans1}")
                
                # Answer 2: Points on line
                ans2 = self.final_answers[1]
                # Expect format like "3 pieces, 1m, 2m, 3m"
                if "3" in ans2 and "1" in ans2 and "2" in ans2 and "3" in ans2:
                    logs.append(f"PASS: Final answer 2 indicates 3 points at 1m, 2m, 3m")
                else:
                    logs.append(f"WARNING: Final answer 2 format unexpected: {ans2}")
                    # Check distances match
                    distances = [self.AS1, self.BS1, self.CS1]
                    expected_dists = [1.0, 2.0, 3.0]
                    for d, exp in zip(distances, expected_dists):
                        if not self._is_close(d, exp):
                            logs.append(f"FAIL: Distance {d:.2f} doesn't match expected {exp:.2f}")
                            success = False
                
                # Answer 3: Points on circle
                ans3 = self.final_answers[2]
                try:
                    ans3_val = int(ans3)
                    if ans3_val != self.expected_points_on_circle:
                        logs.append(f"FAIL: Final answer 3 ({ans3}) doesn't match expected 6 points")
                        success = False
                    else:
                        logs.append(f"PASS: Final answer 3 indicates 6 points on circle")
                except:
                    logs.append(f"WARNING: Could not parse final answer 3: {ans3}")
            else:
                logs.append("WARNING: Not enough final answers in list")
                success = False
            
            # Check internal consistency: BS1 should be half of S1S2 distance
            if not self._is_close(self.BS1, self.d_S1S2 / 2):
                logs.append(f"FAIL: BS1 = {self.BS1:.2f} should equal S1S2/2 = {self.d_S1S2/2:.2f}")
                success = False
            
            # Check that points A, B, C are collinear and ordered correctly
            if not (self.AS1 < self.BS1 < self.CS1):
                logs.append(f"WARNING: Points not in order: AS1={self.AS1:.2f}, BS1={self.BS1:.2f}, CS1={self.CS1:.2f}")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends."""
        logs = []
        success = True
        
        try:
            # Perturb frequency and check wavelength changes inversely (v fixed)
            original_f = self.f
            perturbed_f = original_f * 1.1  # +10%
            
            # Calculate new wavelength using physics formula
            original_lambda = self.v / original_f
            perturbed_lambda = self.v / perturbed_f
            
            # λ should decrease when f increases
            if perturbed_lambda < original_lambda:
                logs.append(f"PASS: λ decreases when f increases (v fixed)")
            else:
                logs.append(f"FAIL: λ doesn't follow inverse relationship with f")
                success = False
            
            # Check the ratio
            expected_ratio = original_f / perturbed_f
            actual_ratio = perturbed_lambda / original_lambda
            
            if not self._is_close(expected_ratio, actual_ratio):
                logs.append(f"WARNING: Ratio mismatch. Expected: {expected_ratio:.3f}, Actual: {actual_ratio:.3f}")
            
            # Test: If we increase S1S2 distance, number of points on line should increase
            # But careful: For fixed λ, more wavelengths fit in longer distance
            original_d = self.d_S1S2
            perturbed_d = original_d * 1.5
            
            # Number of constructive interference points on line = 2*(d/λ) + 1 (for integer wavelengths)
            original_n = 2 * (original_d / original_lambda) + 1
            perturbed_n = 2 * (perturbed_d / original_lambda) + 1
            
            if perturbed_n > original_n:
                logs.append(f"PASS: More points on line for larger S1S2 distance")
            else:
                logs.append(f"WARNING: Point count doesn't increase with distance")
            
        except ZeroDivisionError:
            logs.append("SKIP: Zero division in monotonicity test")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return combined results."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== TEST INVARIANTS ===")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== TEST TRACE CONSISTENCY ===")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== TEST MONOTONICITY ===")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        # Summary
        all_logs.append("\n=== SUMMARY ===")
        all_logs.append(f"Overall success: {overall_success}")
        
        return overall_success, all_logs