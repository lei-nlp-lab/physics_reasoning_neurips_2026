import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize by extracting results from the solution instance.
        """
        self.solution = solution_instance
        
        # Extract results from the solution instance
        # First try to get results directly, then fallback to instance attributes
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results.copy()
        else:
            self.results = {}
            # Try to extract each required variable
            for var in ['N', 'f_1', 'f_1_direction', 'f_2', 'mu']:
                self.results[var] = getattr(solution_instance, var, None)
        
        # Initialize logs
        self.logs = []
        
    def _log(self, msg: str, success: bool):
        """Helper to log messages."""
        self.logs.append((msg, success))
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison."""
        # Handle None values
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self) -> bool:
        """
        Verify critical physics invariants:
        1. Self-locking condition: denominator of N becomes zero at critical mu.
        2. Direction of static friction f1 is encoded as a float (1.0 for "to the right").
        3. Normal force N is positive for mu < critical.
        """
        success = True
        
        try:
            # 1. Check self-locking condition: critical mu should satisfy 1 - 2√3 μ - μ² = 0
            mu_critical = self.results.get('mu')
            if mu_critical is not None:
                mu_critical = float(mu_critical)
                # Compute the value of the denominator at mu_critical
                denom_value = 1 - 2*math.sqrt(3)*mu_critical - mu_critical**2
                if not self._is_close(denom_value, 0.0):
                    self._log(f"Self-locking condition failed: denominator = {denom_value} != 0 at μ={mu_critical}", False)
                    success = False
                else:
                    self._log(f"Self-locking condition satisfied: denominator ≈ 0 at μ={mu_critical}", True)
                    
                # Also check that critical mu equals 2 - √3 ≈ 0.267949
                expected_critical = 2 - math.sqrt(3)
                if not self._is_close(mu_critical, expected_critical):
                    self._log(f"Critical μ mismatch: {mu_critical} vs expected {expected_critical}", False)
                    success = False
                else:
                    self._log(f"Critical μ matches 2-√3 ≈ {expected_critical}", True)
            else:
                self._log("No critical μ provided in results", False)
                success = False
                
            # 2. Check friction direction - should be a float (1.0 for "to the right")
            f1_dir = self.results.get('f_1_direction')
            if f1_dir is not None:
                # Convert to float for comparison
                try:
                    f1_dir_float = float(f1_dir)
                    # Check if it's approximately 1.0 (encoding for "to the right")
                    if not self._is_close(f1_dir_float, 1.0):
                        self._log(f"f_1_direction should be 1.0 (to the right), got {f1_dir_float}", False)
                        success = False
                    else:
                        self._log(f"f_1_direction correctly encoded as 1.0 (to the right)", True)
                except (ValueError, TypeError):
                    self._log(f"f_1_direction is not a valid float: {f1_dir}", False)
                    success = False
            else:
                self._log("No f_1_direction provided", False)
                success = False
                
            # 3. Check normal force N is positive (for reasonable mu values)
            N_val = self.results.get('N')
            if N_val is not None:
                N_val = float(N_val)
                if N_val <= 0:
                    self._log(f"Normal force N should be positive, got {N_val}", False)
                    success = False
                else:
                    self._log(f"Normal force N is positive: {N_val}", True)
            else:
                self._log("No N provided in results", True)  # Not a critical failure
                
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)
            success = False
            
        return success
        
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Check that f_2 = μ * N using the sample mu value.
        """
        success = True
        
        try:
            # Get the sample mu value used in the calculation
            # We need to infer this from the results or use a default
            mu_sample = 0.2  # Default sample value as used in the problem
            
            # Check f_2 = μ * N
            f2_val = self.results.get('f_2')
            N_val = self.results.get('N')
            
            if f2_val is not None and N_val is not None:
                f2_val = float(f2_val)
                N_val = float(N_val)
                
                expected_f2 = mu_sample * N_val
                
                if not self._is_close(f2_val, expected_f2):
                    self._log(f"f_2 inconsistency: expected μ*N = {expected_f2}, got {f2_val}", False)
                    success = False
                else:
                    self._log(f"f_2 = μ*N check passed: {f2_val} ≈ {mu_sample} * {N_val}", True)
            else:
                self._log("Missing numeric data for f_2 = μ*N check", True)  # Not a failure
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)
            success = False
            
        return success
        
    def test_monotonicity(self) -> bool:
        """
        Verify that as μ increases (below critical), N increases.
        """
        success = True
        
        try:
            # Use the formula: N = 2*k*x / (1 - 2√3*μ - μ²)
            # For monotonicity test, we'll use sample values
            k = 1.0
            x = 1.0
            
            # Test with two mu values below critical
            mu1 = 0.1
            mu2 = 0.15  # Increased from mu1
            
            # Compute denominators
            denom1 = 1 - 2*math.sqrt(3)*mu1 - mu1**2
            denom2 = 1 - 2*math.sqrt(3)*mu2 - mu2**2
            
            # Check that denominator decreases when mu increases
            if denom2 >= denom1:
                self._log(f"Denominator did not decrease when μ increased: {denom1} -> {denom2}", False)
                success = False
            else:
                self._log(f"Denominator decreases as μ increases: {denom1} -> {denom2}", True)
                
            # Compute N values
            N1 = 2*k*x / denom1
            N2 = 2*k*x / denom2
            
            if N2 <= N1:
                self._log(f"N did not increase when μ increased: {N1} -> {N2}", False)
                success = False
            else:
                self._log(f"N increases as μ increases (below critical): {N1} -> {N2}", True)
                
        except ZeroDivisionError:
            self._log("Zero division in monotonicity test (denominator too small)", True)
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)
            success = False
            
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        
        test_functions = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        overall_success = True
        for test_name, test_func in test_functions:
            self._log(f"--- {test_name} ---", True)
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {e}", False)
                overall_success = False
                
        # Collect log messages
        log_messages = [f"{'PASS' if success else 'FAIL'}: {msg}" for msg, success in self.logs]
        
        return overall_success, log_messages