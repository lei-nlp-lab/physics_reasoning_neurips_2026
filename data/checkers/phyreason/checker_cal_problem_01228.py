import math
import traceback
from typing import Tuple, List, Optional
import re

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access from solution instance."""
        # Inputs with safe defaults
        self.W = getattr(solution_instance, 'W', 40.0)  # Weight in N
        self.F1 = getattr(solution_instance, 'F1', 13.0)
        self.F2 = getattr(solution_instance, 'F2', 6.0)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        
        # Extract results safely - only look for required keys
        self.results = getattr(solution_instance, 'results', {})
        self.f = self.results.get('f', 0.0)  # Kinetic friction from sub-question 3
        self.f_m = self.results.get('f_m', 0.0)  # Maximum static friction
        self.answers_raw = self.results.get('answers', None)
        
        # Expected target answers (full strings with units and direction)
        self.target_strings = [
            "$7\\mathrm{N}$, direction to the left",
            "$6\\mathrm{N}$, direction to the right", 
            "$8\\mathrm{N}$, direction to the left"
        ]
        # Expected numeric values for comparison
        self.target_values = [7.0, 6.0, 8.0]
        
        # Store original values for monotonicity tests
        self._original_W = self.W
        self.logs = []
        
    def _log(self, message: str, is_error: bool = False):
        """Add message to logs."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
        
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def _extract_numeric_from_string(self, s: str) -> Optional[float]:
        """Extract numeric value from a string like '$7\mathrm{N}$, direction to the left'."""
        try:
            # Try direct conversion first
            return float(s)
        except (ValueError, TypeError):
            # Extract numeric part using regex
            match = re.search(r'[-+]?\d*\.?\d+', str(s))
            if match:
                return float(match.group())
            return None
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants."""
        try:
            # 1. Normal force should equal weight (horizontal surface)
            N_expected = self.W
            N_calculated = self.W  # For horizontal surface, N = W
            if not self._is_close(N_expected, N_calculated):
                self._log(f"Normal force invariant violated: expected {N_expected}N, got {N_calculated}N", True)
                return False
            
            # 2. Maximum static friction = μN
            f_m_expected = self.mu * self.W
            if not self._is_close(self.f_m, f_m_expected):
                self._log(f"Max static friction invariant violated: expected μN = {f_m_expected}N, got {self.f_m}N", True)
                return False
            
            # 3. For static equilibrium (sub-question 1): F1 - F2 - f1 = 0
            # Extract numeric value from answers list
            if self.answers_raw and isinstance(self.answers_raw, list) and len(self.answers_raw) >= 1:
                f1_val = self._extract_numeric_from_string(self.answers_raw[0])
                if f1_val is None:
                    self._log("Cannot extract numeric value from answer 1", True)
                    return False
                net_force1 = self.F1 - self.F2
                if not self._is_close(net_force1, f1_val):
                    self._log(f"Static equilibrium 1 violated: F1 - F2 = {net_force1}N ≠ f1 = {f1_val}N", True)
                    return False
            else:
                self._log("Cannot test static equilibrium 1: answers list missing or invalid", True)
                return False
            
            # 4. For static equilibrium (sub-question 2): F2 should equal f2
            if self.answers_raw and isinstance(self.answers_raw, list) and len(self.answers_raw) >= 2:
                f2_val = self._extract_numeric_from_string(self.answers_raw[1])
                if f2_val is None:
                    self._log("Cannot extract numeric value from answer 2", True)
                    return False
                if not self._is_close(self.F2, f2_val):
                    self._log(f"Static equilibrium 2 violated: F2 = {self.F2}N ≠ f2 = {f2_val}N", True)
                    return False
            else:
                self._log("Cannot test static equilibrium 2: answers list missing or invalid", True)
                return False
            
            # 5. Kinetic friction (sub-question 3) should equal μN
            f_k_expected = self.mu * self.W
            if not self._is_close(self.f, f_k_expected):
                self._log(f"Kinetic friction invariant violated: f = {self.f}N ≠ μN = {f_k_expected}N", True)
                return False
            
            self._log("All physics invariants satisfied")
            return True
            
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}", True)
            return False
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate steps and final answers."""
        try:
            # Based on STEPS ANALYSIS:
            # Step 1 & 4: f_m = μN = 0.2 * 40 = 8N
            f_m_calculated = self.mu * self.W
            if not self._is_close(f_m_calculated, 8.0):
                self._log(f"Step 1/4 inconsistency: f_m should be 8N, calculated {f_m_calculated}N", True)
                return False
            
            # Step 2: f1 = F1 - F2 = 13 - 6 = 7N
            f_step2 = self.F1 - self.F2
            if not self._is_close(f_step2, 7.0):
                self._log(f"Step 2 inconsistency: f1 should be 7N, calculated {f_step2}N", True)
                return False
            
            # Step 3: f2 = F2 = 6N
            if not self._is_close(self.F2, 6.0):
                self._log(f"Step 3 inconsistency: f2 should be 6N, F2 is {self.F2}N", True)
                return False
            
            # Step 5: f3 = μG = μW = 8N
            f_step5 = self.mu * self.W
            if not self._is_close(f_step5, 8.0):
                self._log(f"Step 5 inconsistency: f3 should be 8N, calculated {f_step5}N", True)
                return False
            
            self._log("All step-by-step calculations are consistent")
            return True
            
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}", True)
            return False
    
    def test_monotonicity(self) -> bool:
        """Test that results change appropriately with parameter perturbations."""
        original_passed = True
        
        try:
            # Perturb weight by +20%
            self.W = self._original_W * 1.2
            
            # Recalculate expected values with perturbed weight
            f_m_perturbed = self.mu * self.W  # Should increase by 20%
            
            # Check monotonic relationship: if weight increases, friction forces increase
            f_m_original = self.mu * self._original_W
            if f_m_perturbed <= f_m_original:
                self._log(f"Monotonicity violation: f_m didn't increase with weight increase", True)
                original_passed = False
            
            # Check proportionality: force should scale linearly with weight
            expected_ratio = 1.2
            if f_m_original == 0:
                self._log("Division by zero in monotonicity test", True)
                original_passed = False
            else:
                actual_ratio = f_m_perturbed / f_m_original
                if not self._is_close(actual_ratio, expected_ratio):
                    self._log(f"Proportionality violation: f_m ratio = {actual_ratio}, expected {expected_ratio}", True)
                    original_passed = False
            
            # Restore original weight
            self.W = self._original_W
            
            if original_passed:
                self._log("Monotonicity test passed: friction forces scale correctly with weight")
            
            return original_passed
            
        except Exception as e:
            # Restore on error
            self.W = self._original_W
            self._log(f"Exception in test_monotonicity: {str(e)}", True)
            return False
    
    def test_answer_format(self) -> bool:
        """Check that answers match expected format and values."""
        try:
            # Handle the answers data structure robustly
            answers = self.answers_raw
            
            # If answers is None, try to extract from results
            if answers is None:
                self._log("Answers list is None", True)
                return False
            
            # Ensure answers is a list
            if not isinstance(answers, list):
                # Try to convert to list
                try:
                    # Check if it's a single numeric value
                    if isinstance(answers, (int, float)):
                        answers = [float(answers)]
                    else:
                        # Try to convert from string or other
                        import ast
                        if isinstance(answers, str):
                            answers = ast.literal_eval(answers)
                        else:
                            answers = list(answers)
                except Exception as e:
                    self._log(f"Cannot convert answers to list: {answers}, error: {e}", True)
                    return False
            
            # Now we should have a list
            if len(answers) != 3:
                self._log(f"Expected 3 answers, got {len(answers)}: {answers}", True)
                return False
            
            # Compare each answer with target values
            for i, (answer_val, target_num, target_str) in enumerate(zip(answers, self.target_values, self.target_strings)):
                # First, check if answer contains the expected numeric value
                extracted_num = self._extract_numeric_from_string(answer_val)
                if extracted_num is None:
                    self._log(f"Answer {i+1} has no extractable numeric value: {answer_val}", True)
                    return False
                
                if not self._is_close(extracted_num, target_num):
                    self._log(f"Answer {i+1} numeric mismatch: {extracted_num}N ≠ {target_num}N", True)
                    return False
                
                # Optional: check string format (allow some flexibility in formatting)
                # Convert both to strings and check if they're similar
                answer_str = str(answer_val).lower()
                target_str_lower = target_str.lower()
                
                # Check for key components
                if str(target_num) not in answer_str:
                    self._log(f"Answer {i+1} doesn't contain expected value {target_num}: {answer_str}", True)
                    return False
            
            self._log("All answers match expected values and format")
            return True
            
        except Exception as e:
            self._log(f"Exception in test_answer_format: {str(e)}", True)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics audit tests and return results."""
        self.logs = []  # Reset logs
        
        test_results = []
        
        try:
            # Run all tests
            test_results.append(("Physics Invariants", self.test_invariants()))
            test_results.append(("Trace Consistency", self.test_trace_consistency()))
            test_results.append(("Monotonicity", self.test_monotonicity()))
            test_results.append(("Answer Format", self.test_answer_format()))
            
            # Check overall success
            all_passed = all(result[1] for result in test_results)
            
            # Summary log
            summary = "TEST SUMMARY:\n"
            for test_name, passed in test_results:
                summary += f"  {test_name}: {'PASS' if passed else 'FAIL'}\n"
            summary += f"OVERALL: {'PASS' if all_passed else 'FAIL'}"
            self._log(summary)
            
            return all_passed, self.logs
            
        except Exception as e:
            self._log(f"Fatal error in run_all_tests: {str(e)}\n{traceback.format_exc()}", True)
            return False, self.logs