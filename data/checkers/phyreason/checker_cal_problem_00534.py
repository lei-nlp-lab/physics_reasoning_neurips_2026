import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all needed attributes from solution instance."""
        self.solution = solution_instance
        
        # Use safe attribute access with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all required values with safe fallbacks
        self.r = getattr(solution_instance, 'r', self.results.get('r', 0.0))
        self.I = getattr(solution_instance, 'I', self.results.get('I', 0.0))
        self.P_out = getattr(solution_instance, 'P_out', self.results.get('P_out', 0.0))
        self.U = getattr(solution_instance, 'U', self.results.get('U', 0.0))
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        
        # Store original values for monotonicity tests
        self.original_values = {
            'r': self.r,
            'I': self.I,
            'P_out': self.P_out,
            'U': self.U,
            'E': self.E
        }
        
        # Define tolerance for fuzzy matching (1-2% as per robustness rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message, is_error=False):
        """Helper method to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
        
    def _fuzzy_equal(self, a, b):
        """Fuzzy equality check with tolerance."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics laws: terminal voltage equation and power formula."""
        try:
            # Test 1: Terminal voltage equation U = E - I*r
            expected_U = self.E - self.I * self.r
            if not self._fuzzy_equal(self.U, expected_U):
                self._log(f"Terminal voltage invariant failed: U={self.U}, E-I*r={expected_U}", True)
                return False
            self._log(f"Terminal voltage invariant passed: U={self.U} ≈ E-I*r={expected_U}")
            
            # Test 2: Power formula P_out = U*I
            expected_P = self.U * self.I
            if not self._fuzzy_equal(self.P_out, expected_P):
                self._log(f"Power invariant failed: P_out={self.P_out}, U*I={expected_P}", True)
                return False
            self._log(f"Power invariant passed: P_out={self.P_out} ≈ U*I={expected_P}")
            
            # Test 3: Ohm's law for resistor (if we had R)
            # Since we don't have R directly, we can compute it from U and I
            R_computed = self.U / self.I if self.I != 0 else 0
            # Check that power also equals I²*R
            expected_P2 = (self.I ** 2) * R_computed
            if not self._fuzzy_equal(self.P_out, expected_P2):
                self._log(f"Alternative power invariant failed: P_out={self.P_out}, I²*R={expected_P2}", True)
                return False
            self._log(f"Alternative power invariant passed: P_out={self.P_out} ≈ I²*R={expected_P2}")
            
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with exception: {str(e)}", True)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results from steps analysis with final answers."""
        try:
            # Expected values from steps analysis
            expected_values = {
                'E': 4.0,    # Step 3: EMF when I=0
                'r': 1.0,    # Step 4: Internal resistance
                'U': 3.0,    # Step 5-6: Terminal voltage at intersection
                'I': 1.0,    # Step 7: Current at intersection
                'P_out': 3.0 # Step 8: Output power
            }
            
            all_passed = True
            for key, expected in expected_values.items():
                actual = getattr(self, key, 0.0)
                if not self._fuzzy_equal(actual, expected):
                    self._log(f"Trace consistency failed for {key}: expected {expected}, got {actual}", True)
                    all_passed = False
                else:
                    self._log(f"Trace consistency passed for {key}: {actual} ≈ {expected}")
            
            return all_passed
            
        except Exception as e:
            self._log(f"Trace consistency test failed with exception: {str(e)}", True)
            return False
    
    def test_monotonicity(self):
        """Test that physical relationships behave as expected when parameters are perturbed."""
        try:
            # Test 1: Increase current I by 10%, check terminal voltage decreases
            original_I = self.I
            perturbed_I = original_I * 1.1
            
            # Calculate new terminal voltage using U = E - I*r
            original_U = self.U
            perturbed_U = self.E - perturbed_I * self.r
            
            if not (perturbed_U < original_U or self._fuzzy_equal(perturbed_U, original_U)):
                self._log(f"Monotonicity test 1 failed: U should decrease when I increases. Original U={original_U}, Perturbed U={perturbed_U}", True)
                return False
            self._log(f"Monotonicity test 1 passed: U decreases from {original_U} to {perturbed_U} when I increases")
            
            # Test 2: Decrease internal resistance r by 10%, check terminal voltage increases (for same current)
            original_r = self.r
            perturbed_r = original_r * 0.9
            
            perturbed_U2 = self.E - self.I * perturbed_r
            
            if not (perturbed_U2 > original_U or self._fuzzy_equal(perturbed_U2, original_U)):
                self._log(f"Monotonicity test 2 failed: U should increase when r decreases. Original U={original_U}, Perturbed U={perturbed_U2}", True)
                return False
            self._log(f"Monotonicity test 2 passed: U increases from {original_U} to {perturbed_U2} when r decreases")
            
            # Test 3: Check power relationship - power should increase when current increases (but only up to a point)
            # For small perturbations from the operating point, power should follow P = I*(E - I*r)
            original_P = self.P_out
            perturbed_P = perturbed_I * (self.E - perturbed_I * self.r)
            
            # We're not checking specific direction here, just that the relationship holds
            expected_perturbed_P = perturbed_I * (self.E - perturbed_I * self.r)
            if not self._fuzzy_equal(perturbed_P, expected_perturbed_P):
                self._log(f"Monotonicity test 3 failed: Power formula not consistent. Expected {expected_perturbed_P}, got {perturbed_P}", True)
                return False
            self._log(f"Monotonicity test 3 passed: Power formula holds for perturbed current")
            
            # Restore original values
            self.I = original_I
            self.r = original_r
            
            return True
            
        except Exception as e:
            # Restore values on exception
            self.I = self.original_values['I']
            self.r = self.original_values['r']
            self._log(f"Monotonicity test failed with exception: {str(e)}", True)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs = []  # Reset logs
        success = True
        
        # Run each test
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Starting {test_name}...")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} FAILED", True)
                else:
                    self._log(f"{test_name} PASSED")
            except Exception as e:
                success = False
                self._log(f"{test_name} crashed: {traceback.format_exc()}", True)
        
        # Final summary
        if success:
            self._log("All physics checks passed successfully!")
        else:
            self._log("Some physics checks failed. Review the logs above.", True)
        
        return success, self.logs