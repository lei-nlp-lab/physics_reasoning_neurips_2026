import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.R = getattr(solution_instance, 'R', 1.0)  # radius in meters
        self.t = getattr(solution_instance, 't', 5.0)  # time in seconds
        
        # Extract results with safe access
        self.x = self.results.get('x', 0.0)  # displacement
        self.s = self.results.get('s', 0.0)  # distance traveled
        self.v_prime = self.results.get("v'", 0.0)  # average speed
        self.v_bar = self.results.get('overlinev', 0.0)  # average velocity
        
        # Expected values from problem analysis
        self.expected_x = 4.0 * self.R  # displacement = 4R
        self.expected_s = 2.0 * math.pi * self.R  # distance = 2πR
        self.expected_v_bar = self.expected_x / self.t if self.t != 0 else 0.0
        self.expected_v_prime = self.expected_s / self.t if self.t != 0 else 0.0
        
        # Initialize test logs
        self.logs = []
        self.success = True

    def _log(self, message: str, is_success: bool = True):
        """Helper method to log test results"""
        status = "PASS" if is_success else "FAIL"
        self.logs.append(f"{status}: {message}")
        if not is_success:
            self.success = False

    def test_invariants(self):
        """Test fundamental physical invariants"""
        try:
            # 1. Check displacement formula: x = 4R
            if math.isclose(self.x, self.expected_x, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Displacement x = {self.x}m matches expected 4R = {self.expected_x}m")
            else:
                self._log(f"Displacement mismatch: got {self.x}m, expected {self.expected_x}m", False)
            
            # 2. Check distance formula: s = 2πR
            if math.isclose(self.s, self.expected_s, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance s = {self.s}m matches expected 2πR ≈ {self.expected_s:.3f}m")
            else:
                self._log(f"Distance mismatch: got {self.s}m, expected {self.expected_s:.3f}m", False)
            
            # 3. Check average velocity definition: v̄ = x/t
            v_bar_calc = self.x / self.t if self.t != 0 else float('inf')
            if math.isclose(self.v_bar, v_bar_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Average velocity v̄ = {self.v_bar}m/s matches x/t = {v_bar_calc}m/s")
            else:
                self._log(f"Average velocity mismatch: got {self.v_bar}m/s, calculated {v_bar_calc}m/s", False)
            
            # 4. Check average speed definition: v' = s/t
            v_prime_calc = self.s / self.t if self.t != 0 else float('inf')
            if math.isclose(self.v_prime, v_prime_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Average speed v' = {self.v_prime}m/s matches s/t = {v_prime_calc}m/s")
            else:
                self._log(f"Average speed mismatch: got {self.v_prime}m/s, calculated {v_prime_calc}m/s", False)
            
            # 5. Check that average speed ≥ average velocity magnitude (physics constraint)
            if self.v_prime >= abs(self.v_bar) - 1e-2:  # Allow small tolerance
                self._log(f"Average speed ({self.v_prime}m/s) ≥ average velocity magnitude ({abs(self.v_bar):.3f}m/s)")
            else:
                self._log(f"Average speed ({self.v_prime}m/s) < average velocity magnitude ({abs(self.v_bar):.3f}m/s)", False)
                
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            self._log(traceback.format_exc(), False)

    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results"""
        try:
            # Get step results from solution if available
            steps = getattr(self.solution, 'steps', {})
            
            # Check step 1: displacement
            step1_val = None
            if 'step_1' in steps:
                step1_results = steps['step_1'].get('result_quantity', [])
                for rq in step1_results:
                    if rq.get('symbol') == 'x':
                        step1_val = float(rq.get('value', 0.0))
            if step1_val and math.isclose(self.x, step1_val, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Final displacement x={self.x}m matches step 1 value {step1_val}m")
            elif step1_val:
                self._log(f"Displacement mismatch: final {self.x}m vs step 1 {step1_val}m", False)
            
            # Check step 2: distance
            step2_val = None
            if 'step_2' in steps:
                step2_results = steps['step_2'].get('result_quantity', [])
                for rq in step2_results:
                    if rq.get('symbol') == 's':
                        step2_val = float(rq.get('value', 0.0))
            if step2_val and math.isclose(self.s, step2_val, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Final distance s={self.s}m matches step 2 value {step2_val}m")
            elif step2_val:
                self._log(f"Distance mismatch: final {self.s}m vs step 2 {step2_val}m", False)
            
            # Check step 3: average velocity
            step3_val = None
            if 'step_3' in steps:
                step3_results = steps['step_3'].get('result_quantity', [])
                for rq in step3_results:
                    if rq.get('symbol') == '\\overline{v}' or rq.get('symbol') == 'overlinev':
                        step3_val = float(rq.get('value', 0.0))
            if step3_val and math.isclose(self.v_bar, step3_val, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Final average velocity v̄={self.v_bar}m/s matches step 3 value {step3_val}m/s")
            elif step3_val:
                self._log(f"Average velocity mismatch: final {self.v_bar}m/s vs step 3 {step3_val}m/s", False)
            
            # Check step 4: average speed
            step4_val = None
            if 'step_4' in steps:
                step4_results = steps['step_4'].get('result_quantity', [])
                for rq in step4_results:
                    if rq.get('symbol') == "v'" or rq.get('symbol') == 'vprime':
                        step4_val = float(rq.get('value', 0.0))
            if step4_val and math.isclose(self.v_prime, step4_val, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Final average speed v'={self.v_prime}m/s matches step 4 value {step4_val}m/s")
            elif step4_val:
                self._log(f"Average speed mismatch: final {self.v_prime}m/s vs step 4 {step4_val}m/s", False)
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)

    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs"""
        try:
            # Store original values
            original_R = self.R
            perturbed_R = original_R * 1.1  # Increase radius by 10%
            
            # Calculate expected changes using formulas from KB
            # According to KB: if displacement increases (with fixed time), average velocity increases
            new_x = 4.0 * perturbed_R
            new_s = 2.0 * math.pi * perturbed_R
            new_v_bar = new_x / self.t if self.t != 0 else 0.0
            new_v_prime = new_s / self.t if self.t != 0 else 0.0
            
            # Test monotonicity for displacement and average velocity
            if new_x > self.x:
                self._log(f"Displacement increases with R: {self.x}m → {new_x:.3f}m when R={original_R}→{perturbed_R}m")
                if new_v_bar > self.v_bar:
                    self._log(f"Average velocity increases with displacement: {self.v_bar}m/s → {new_v_bar:.3f}m/s")
                else:
                    self._log(f"Average velocity should increase but got {self.v_bar}m/s → {new_v_bar:.3f}m/s", False)
            else:
                self._log(f"Displacement doesn't increase as expected", False)
            
            # Test monotonicity for distance and average speed
            if new_s > self.s:
                self._log(f"Distance increases with R: {self.s}m → {new_s:.3f}m when R={original_R}→{perturbed_R}m")
                if new_v_prime > self.v_prime:
                    self._log(f"Average speed increases with distance: {self.v_prime}m/s → {new_v_prime:.3f}m/s")
                else:
                    self._log(f"Average speed should increase but got {self.v_prime}m/s → {new_v_prime:.3f}m/s", False)
            else:
                self._log(f"Distance doesn't increase as expected", False)
            
            # Restore original R
            self.R = original_R
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs"""
        self.logs = []
        self.success = True
        
        self.logs.append("=" * 50)
        self.logs.append("PhysicsChecker: Testing particle motion along semicircular arcs")
        self.logs.append(f"Inputs: R={self.R}m, t={self.t}s")
        self.logs.append(f"Results: x={self.x}m, s={self.s}m, v̄={self.v_bar}m/s, v'={self.v_prime}m/s")
        self.logs.append("=" * 50)
        
        self.logs.append("\n1. Testing physical invariants:")
        self.test_invariants()
        
        self.logs.append("\n2. Testing trace consistency:")
        self.test_trace_consistency()
        
        self.logs.append("\n3. Testing monotonic relationships:")
        self.test_monotonicity()
        
        self.logs.append("\n" + "=" * 50)
        self.logs.append(f"Overall: {'PASS' if self.success else 'FAIL'}")
        self.logs.append("=" * 50)
        
        return self.success, self.logs

# Example usage pattern (commented out):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.R = 1.0
#             self.t = 5.0
#             self.results = {
#                 'x': 4.0,
#                 's': 6.283185307179586,
#                 'v\'': 1.2566370614359172,
#                 'overlinev': 0.8
#             }
#             self.steps = {
#                 'step_1': {'result_quantity': [{'symbol': 'x', 'value': '4'}]},
#                 'step_2': {'result_quantity': [{'symbol': 's', 'value': '6.28'}]},
#                 'step_3': {'result_quantity': [{'symbol': '\\overline{v}', 'value': '0.8'}]},
#                 'step_4': {'result_quantity': [{'symbol': "v'", 'value': '1.256'}]}
#             }
#     
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)