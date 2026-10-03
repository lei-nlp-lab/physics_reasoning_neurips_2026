import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract required quantities with safe defaults
        self.v_2 = self.results.get('v_2', 0.0)
        self.overlinea = self.results.get('overlinea', 0.0)
        self.overlinev = self.results.get('overlinev', 0.0)
        self.a = self.results.get('a', 0.0)
        
        # Expected values from steps analysis
        self.expected_v2 = 3.0
        self.expected_overlinea = 5.0 / 3.0  # ≈1.6667
        self.expected_overlinev = 4.0
        self.expected_a = 1.0
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []
    
    def _log(self, message: str):
        self.logs.append(message)
    
    def test_invariants(self) -> bool:
        """Check that the computed results match the expected values from the steps."""
        success = True
        
        # Check v_2
        if not math.isclose(self.v_2, self.expected_v2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"FAIL: v_2 = {self.v_2}, expected ≈ {self.expected_v2}")
            success = False
        else:
            self._log(f"PASS: v_2 = {self.v_2} ≈ {self.expected_v2}")
        
        # Check average acceleration in first 3 seconds
        if not math.isclose(self.overlinea, self.expected_overlinea, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"FAIL: overlinea = {self.overlinea}, expected ≈ {self.expected_overlinea}")
            success = False
        else:
            self._log(f"PASS: overlinea = {self.overlinea} ≈ {self.expected_overlinea}")
        
        # Check average velocity during 4th second
        if not math.isclose(self.overlinev, self.expected_overlinev, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"FAIL: overlinev = {self.overlinev}, expected ≈ {self.expected_overlinev}")
            success = False
        else:
            self._log(f"PASS: overlinev = {self.overlinev} ≈ {self.expected_overlinev}")
        
        # Check average acceleration during 4th second
        if not math.isclose(self.a, self.expected_a, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"FAIL: a = {self.a}, expected ≈ {self.expected_a}")
            success = False
        else:
            self._log(f"PASS: a = {self.a} ≈ {self.expected_a}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with the final answer string.
        Since the generated code doesn't produce a final_answer attribute,
        we'll check if the results dictionary contains all required keys
        with correct values.
        """
        required_keys = ['v_2', 'overlinea', 'overlinev', 'a']
        missing_keys = [key for key in required_keys if key not in self.results]
        
        if missing_keys:
            self._log(f"FAIL: Missing keys in results: {missing_keys}")
            return False
        
        # Check that values are numeric (not symbolic)
        for key in required_keys:
            value = self.results[key]
            if not isinstance(value, (int, float)):
                self._log(f"FAIL: Value for {key} is not numeric: {type(value)}")
                return False
        
        self._log("PASS: All required results present and numeric")
        return True
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input (if available) and check the trend.
        Since the problem has no explicit input parameters (like mass), we skip this test.
        """
        self._log("INFO: No tunable input parameter found for monotonicity test. Skipping.")
        return True
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs = []
        self._log("=== Physics Checker Started ===")
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        success = test1 and test2 and test3
        self._log(f"=== Physics Checker Finished: {'SUCCESS' if success else 'FAILED'} ===")
        
        return success, self.logs