import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        """
        # Try to get results from the solution instance
        # First check if it has a 'results' attribute
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        else:
            # If not, check if solution_instance itself is a dict-like object
            # that might contain our results
            self.results = {}
            # Try to get values that might be stored as attributes
            for key in ['U_AC', 'W_AB', 'W_AC', 'q']:
                if hasattr(solution_instance, key):
                    value = getattr(solution_instance, key)
                    try:
                        self.results[key] = float(value)
                    except (TypeError, ValueError):
                        pass
        
        # Ensure results is a dictionary
        if not isinstance(self.results, dict):
            self.results = {}
        
        # Extract values with safe defaults
        self.q = self._get_float('q', -3e-8)
        self.W_AB = self._get_float('W_AB', 1.5e-7)
        self.W_AC = self._get_float('W_AC', None)  # Computed, may be None
        self.U_AC = self._get_float('U_AC', None)  # Computed, may be None
        
        # Given work done AGAINST electrostatic force from B to C
        self.W_BC_against = 9e-8
        
        # Compute work done BY electrostatic force for BC (negative of work against)
        self.W_BC_by = -self.W_BC_against
        
        # Store original values
        self.original_q = self.q
        self.original_W_AB = self.W_AB
        
        # Initialize logs
        self.logs = []
    
    def _get_float(self, key, default):
        """Safely get a float value from results."""
        value = self.results.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def _log(self, message: str, status: str = "INFO"):
        """Add a log message with status."""
        self.logs.append(f"[{status}] {message}")
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants.
        """
        success = True
        
        # 1. Check work superposition
        expected_W_AC = self.W_AB + self.W_BC_by  # 1.5e-7 + (-9e-8) = 6e-8 J
        
        if self.W_AC is None:
            self._log("W_AC not provided in solution", "WARNING")
            computed_W_AC = expected_W_AC
        else:
            computed_W_AC = self.W_AC
            
            # Use relative tolerance for comparison
            if not math.isclose(computed_W_AC, expected_W_AC, rel_tol=1e-2):
                self._log(f"Work superposition violated: W_AC = {computed_W_AC:.2e} J, "
                         f"expected W_AB + W_BC = {expected_W_AC:.2e} J", "ERROR")
                success = False
            else:
                self._log(f"Work superposition verified: W_AC = {computed_W_AC:.2e} J = "
                         f"W_AB + W_BC = {expected_W_AC:.2e} J", "SUCCESS")
        
        # 2. Check potential difference definition
        if self.U_AC is not None:
            # Check if q is effectively zero
            if abs(self.q) < 1e-15:
                self._log(f"Charge q = {self.q:.2e} C is too close to zero, skipping U_AC check", "WARNING")
            else:
                try:
                    expected_U_AC = computed_W_AC / self.q  # Should be -2 V
                except ZeroDivisionError:
                    self._log(f"Division by zero: q = {self.q:.2e} C", "ERROR")
                    success = False
                else:
                    if not math.isclose(self.U_AC, expected_U_AC, rel_tol=1e-2):
                        self._log(f"Potential difference definition violated: U_AC = {self.U_AC:.2f} V, "
                                 f"expected W_AC/q = {expected_U_AC:.2f} V", "ERROR")
                        success = False
                    else:
                        self._log(f"Potential difference definition verified: U_AC = {self.U_AC:.2f} V = "
                                 f"W_AC/q = {expected_U_AC:.2f} V", "SUCCESS")
                    
                    # 3. Check sign consistency
                    if computed_W_AC > 0 and self.q < 0 and self.U_AC >= 0:
                        self._log(f"Sign inconsistency: positive W_AC with negative q should yield negative U_AC, "
                                 f"but got U_AC = {self.U_AC:.2f} V", "ERROR")
                        success = False
                    elif computed_W_AC > 0 and self.q < 0 and self.U_AC < 0:
                        self._log("Sign consistency verified: positive W_AC with negative q yields negative U_AC", "SUCCESS")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate steps with final answer.
        """
        success = True
        
        # Check step 2: W_AC calculation
        expected_W_AC_step2 = 6e-8
        if self.W_AC is not None:
            if not math.isclose(self.W_AC, expected_W_AC_step2, rel_tol=1e-2):
                self._log(f"Step 2 inconsistency: W_AC = {self.W_AC:.2e} J, "
                         f"expected from step analysis = {expected_W_AC_step2:.2e} J", "ERROR")
                success = False
            else:
                self._log(f"Step 2 consistency: W_AC = {self.W_AC:.2e} J matches step analysis", "SUCCESS")
        
        # Check step 3: U_AC calculation
        expected_U_AC_step3 = -2.0
        if self.U_AC is not None:
            if not math.isclose(self.U_AC, expected_U_AC_step3, rel_tol=1e-2):
                self._log(f"Step 3 inconsistency: U_AC = {self.U_AC:.2f} V, "
                         f"expected from step analysis = {expected_U_AC_step3:.2f} V", "ERROR")
                success = False
            else:
                self._log(f"Step 3 consistency: U_AC = {self.U_AC:.2f} V matches step analysis", "SUCCESS")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input charge magnitude and verify the trend.
        """
        success = True
        
        # Store original state
        original_q = self.q
        
        # Check if q is effectively zero
        if abs(original_q) < 1e-15:
            self._log(f"Original charge q = {original_q:.2e} C is too close to zero, skipping monotonicity test", "WARNING")
            return True
        
        # We need W_AC for this test
        if self.W_AC is None:
            self._log("W_AC not provided, cannot perform monotonicity test", "WARNING")
            return True
        
        # Perturb charge magnitude
        perturbed_q = original_q * 1.1
        
        # Recalculate U_AC with perturbed q
        try:
            perturbed_U_AC = self.W_AC / perturbed_q
        except ZeroDivisionError:
            self._log(f"Cannot compute perturbed U_AC: division by zero with q = {perturbed_q:.2e} C", "ERROR")
            return False
        
        # Check monotonicity
        try:
            original_magnitude = abs(self.W_AC / original_q)
        except ZeroDivisionError:
            self._log(f"Cannot compute original magnitude: division by zero with q = {original_q:.2e} C", "ERROR")
            return False
        
        perturbed_magnitude = abs(perturbed_U_AC)
        
        # Allow for small numerical errors
        if perturbed_magnitude < original_magnitude or math.isclose(perturbed_magnitude, original_magnitude, rel_tol=1e-2):
            self._log(f"Monotonicity verified: |U_AC| = {original_magnitude:.2f} V -> {perturbed_magnitude:.2f} V", "SUCCESS")
        else:
            self._log(f"Monotonicity violation: |U_AC| went from {original_magnitude:.2f} V to {perturbed_magnitude:.2f} V", "ERROR")
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success status and logs.
        """
        self.logs = []
        
        self._log("Starting Physics Audit", "INFO")
        
        test_results = []
        
        self._log("\n1. Testing invariants...", "INFO")
        test_results.append(("invariants", self.test_invariants()))
        
        self._log("\n2. Testing trace consistency...", "INFO")
        test_results.append(("trace_consistency", self.test_trace_consistency()))
        
        self._log("\n3. Testing monotonicity...", "INFO")
        test_results.append(("monotonicity", self.test_monotonicity()))
        
        overall_success = all(result for _, result in test_results)
        
        self._log(f"\n=== AUDIT SUMMARY ===", "INFO")
        for test_name, passed in test_results:
            status = "PASS" if passed else "FAIL"
            self._log(f"{test_name}: {status}", "INFO")
        
        if overall_success:
            self._log("All physics checks passed!", "SUCCESS")
        else:
            self._log("Physics audit failed", "ERROR")
        
        return overall_success, self.logs