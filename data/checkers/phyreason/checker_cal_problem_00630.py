import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization with safe attribute access.
        Extracts all required inputs and results from solution_instance.
        """
        self.logs = []
        
        # Extract all values from results dictionary
        results = solution_instance.results
        
        # Get constants with sensible defaults
        self.G = results.get('G', 6.6743e-11)
        self.M_sun = results.get('M', 1.989e30)
        self.r1 = results.get('r1', 1.496e11)
        self.T1 = results.get('T1', 3.156e7)
        
        # Extract results safely
        self.k = results.get('k', None)
        self.omega = results.get('omega', None)
        self.M_land = results.get('M_land', None)
        self.N_A = results.get('N/A', None)  # Note: key is 'N/A' with slash
        
        # Define constants
        self.pi = math.pi
        
        # Validate critical values to avoid division by zero
        if self.G == 0:
            self._log("WARNING: G is zero, may cause division errors", True)
        if self.T1 == 0:
            self._log("WARNING: T1 is zero, may cause division errors", True)
        
    def _log(self, message: str, is_error: bool = False):
        """Helper to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def _is_close(self, a: Any, b: Any) -> bool:
        """
        Robust numerical comparison with tolerance.
        """
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Handle zero values specially to avoid division issues
            if a == 0 and b == 0:
                return True
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        # Handle None cases
        if a is None or b is None:
            return False
        # Try to convert to float if possible
        try:
            a_float = float(a)
            b_float = float(b)
            if a_float == 0 and b_float == 0:
                return True
            return math.isclose(a_float, b_float, rel_tol=1e-2, abs_tol=1e-2)
        except:
            return False
    
    def test_invariants(self) -> bool:
        """
        Test fundamental physics invariants:
        1. Kepler's Third Law derivation consistency
        2. Force balance (gravitational = centripetal)
        """
        success = True
        
        try:
            # 1. Check Kepler's Third Law constant derivation
            # Avoid division by zero
            if self.G == 0 or self.M_sun == 0:
                self._log("Skipping k check due to zero G or M_sun", True)
                success = False
            else:
                expected_k = (self.G * self.M_sun) / (4 * self.pi**2)
                
                if self.k is not None:
                    if not self._is_close(self.k, expected_k):
                        self._log(f"Kepler's constant mismatch: k={self.k}, expected={expected_k}", True)
                        success = False
                    else:
                        self._log("Kepler's constant k matches expected value")
                else:
                    self._log("Warning: k not found in results", True)
                    success = False
            
            # 2. Check angular velocity vs period relationship
            if self.omega is not None and self.T1 is not None:
                if self.T1 == 0:
                    self._log("Cannot compute expected omega: T1 is zero", True)
                    success = False
                else:
                    expected_omega = 2 * self.pi / self.T1
                    if not self._is_close(self.omega, expected_omega):
                        self._log(f"Angular velocity mismatch: ω={self.omega}, expected={expected_omega}", True)
                        success = False
                    else:
                        self._log("Angular velocity matches period relationship")
            
            # 3. Check N/A value (should be 1.0 for force equality)
            if self.N_A is not None:
                if not self._is_close(self.N_A, 1.0):
                    self._log(f"N/A value mismatch: {self.N_A}, expected 1.0", True)
                    success = False
                else:
                    self._log("Force balance ratio is 1.0 as expected")
                    
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", True)
            success = False
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Check consistency between intermediate steps and final results.
        """
        success = True
        
        try:
            # Check that M_land is consistent with orbital parameters
            if all(v is not None for v in [self.G, self.r1, self.T1, self.M_land]):
                if self.G == 0 or self.T1 == 0:
                    self._log("Cannot check M_land consistency: G or T1 is zero", True)
                    success = False
                else:
                    # Expected expression for M_land
                    expected_M = (4 * self.pi**2 * self.r1**3) / (self.G * self.T1**2)
                    
                    if not self._is_close(self.M_land, expected_M):
                        self._log(f"Mass mismatch: M_land={self.M_land}, expected={expected_M}", True)
                        success = False
                    else:
                        self._log("Mass calculation consistent with orbital parameters")
            
            # Check that k is consistent with Sun's mass
            if self.k is not None:
                if self.G == 0:
                    self._log("Cannot check k consistency: G is zero", True)
                    success = False
                else:
                    expected_k_from_sun = (self.G * self.M_sun) / (4 * self.pi**2)
                    if not self._is_close(self.k, expected_k_from_sun):
                        self._log(f"k doesn't match Sun's mass: k={self.k}, expected={expected_k_from_sun}", True)
                        success = False
                    else:
                        self._log("k consistent with Sun's mass")
                    
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", True)
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Test that perturbations in inputs produce expected changes in outputs.
        """
        success = True
        
        try:
            # Test 1: Perturb r1 and check effect on M_land
            if all(v is not None for v in [self.G, self.r1, self.T1, self.M_land]):
                if self.G == 0 or self.T1 == 0 or self.M_land == 0:
                    self._log("Skipping r1 perturbation test due to zero value", True)
                else:
                    original_r1 = float(self.r1)
                    original_M = float(self.M_land)
                    
                    # Perturb r1 by 1%
                    perturb_factor = 1.01
                    perturbed_r1 = original_r1 * perturb_factor
                    
                    # Calculate expected M_land change (M ∝ r^3)
                    expected_ratio = perturb_factor**3
                    
                    # Calculate new M using formula
                    new_M = (4 * self.pi**2 * perturbed_r1**3) / (self.G * self.T1**2)
                    actual_ratio = new_M / original_M
                    
                    if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"Monotonicity failed for r1 perturbation: ratio={actual_ratio}, expected={expected_ratio}", True)
                        success = False
                    else:
                        self._log(f"Monotonicity test passed for r1 perturbation (M ∝ r³)")
            
            # Test 2: Perturb T1 and check effect on M_land
            if all(v is not None for v in [self.G, self.r1, self.T1, self.M_land]):
                if self.G == 0 or self.T1 == 0 or self.M_land == 0:
                    self._log("Skipping T1 perturbation test due to zero value", True)
                else:
                    original_T1 = float(self.T1)
                    original_M = float(self.M_land)
                    
                    # Perturb T1 by 1%
                    perturb_factor = 1.01
                    perturbed_T1 = original_T1 * perturb_factor
                    
                    # Calculate expected M_land change (M ∝ 1/T^2)
                    expected_ratio = 1 / (perturb_factor**2)
                    
                    # Calculate new M using formula
                    new_M = (4 * self.pi**2 * self.r1**3) / (self.G * perturbed_T1**2)
                    actual_ratio = new_M / original_M
                    
                    if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"Monotonicity failed for T1 perturbation: ratio={actual_ratio}, expected={expected_ratio}", True)
                        success = False
                    else:
                        self._log(f"Monotonicity test passed for T1 perturbation (M ∝ 1/T²)")
            
            # Test 3: Check trend for k with respect to M_sun
            if self.k is not None and self.M_sun is not None:
                if self.G == 0 or self.k == 0 or self.M_sun == 0:
                    self._log("Skipping k monotonicity test due to zero value", True)
                else:
                    original_k = float(self.k)
                    original_M = float(self.M_sun)
                    
                    # Perturb M_sun by 1%
                    perturb_factor = 1.01
                    perturbed_M = original_M * perturb_factor
                    
                    # Calculate expected k change (k ∝ M)
                    expected_ratio = perturb_factor
                    new_k = (self.G * perturbed_M) / (4 * self.pi**2)
                    actual_ratio = new_k / original_k
                    
                    if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"k monotonicity failed: ratio={actual_ratio}, expected={expected_ratio}", True)
                        success = False
                    else:
                        self._log(f"k monotonicity test passed (k ∝ M)")
                    
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", True)
            success = False
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all physics consistency tests and return results.
        """
        self.logs = []  # Reset logs
        overall_success = True
        
        # Run individual tests
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"{test_name} failed", True)
                else:
                    self._log(f"{test_name} passed")
            except Exception as e:
                self._log(f"{test_name} crashed: {str(e)}", True)
                overall_success = False
        
        # Summary
        if overall_success:
            self._log("All physics checks passed!")
        else:
            self._log("Some physics checks failed", True)
        
        return overall_success, self.logs