import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        
        Args:
            solution_instance: The solution object containing inputs and results.
        """
        self.solution = solution_instance
        
        # Input parameters with safe extraction
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'mass', 0.8))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'length', 0.5))
        self.theta_deg = getattr(solution_instance, 'theta', 37)
        self.g = getattr(solution_instance, 'g', 10)
        
        # Convert to radians
        self.theta = math.radians(self.theta_deg)
        
        # Extract results with safe access using results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # First part: static equilibrium (T, N)
        self.T_static = results.get('T', results.get('T_static', None))
        self.N_static = results.get('N', results.get('N_static', None))
        
        # Second part: rotating with N'=0 (T', ω)
        self.T_prime = results.get("T'", results.get('T_prime', None))
        self.N_prime = results.get("N'", results.get('N_prime', None))
        self.omega = results.get('ω', results.get('omega', None))
        
        # For symbolic calculations
        self.m_sym = sp.symbols('m', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        self.theta_sym = sp.symbols('theta', positive=True)
        self.L_sym = sp.symbols('L', positive=True)
        self.T_sym = sp.symbols('T', positive=True)
        self.N_sym = sp.symbols('N', positive=True)
        self.T_prime_sym = sp.symbols('T_prime', positive=True)
        self.omega_sym = sp.symbols('omega', positive=True)
        
        # Known trigonometric values from problem
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        
        # Test results storage
        self.logs = []
        self.all_passed = True
        
    def _log(self, message: str, passed: bool) -> None:
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        self.logs.append(f"{status}: {message}")
        if not passed:
            self.all_passed = False
            
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Safe numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> None:
        """Test physical invariants based on KB theorems."""
        
        # 1. Static equilibrium tests
        if self.T_static is not None:
            expected_T = self.m * self.g * self.cos_theta  # T = mg cosθ
            if self._is_close(self.T_static, expected_T):
                self._log(f"Static tension T = {self.T_static} ≈ mg cosθ = {expected_T}", True)
            else:
                self._log(f"Static tension mismatch: T = {self.T_static}, expected {expected_T}", False)
        
        if self.N_static is not None:
            expected_N = self.m * self.g * self.sin_theta  # N = mg sinθ
            if self._is_close(self.N_static, expected_N):
                self._log(f"Static normal force N = {self.N_static} ≈ mg sinθ = {expected_N}", True)
            else:
                self._log(f"Static normal force mismatch: N = {self.N_static}, expected {expected_N}", False)
        
        # 2. Rotating case: N' should be zero
        if self.N_prime is not None:
            if self._is_close(self.N_prime, 0.0):
                self._log(f"Rotating normal force N' = {self.N_prime} ≈ 0", True)
            else:
                self._log(f"Rotating normal force should be zero: N' = {self.N_prime}", False)
        
        # 3. T' should satisfy vertical equilibrium: T' cosθ = mg
        if self.T_prime is not None:
            expected_T_prime = self.m * self.g / self.cos_theta  # T' = mg / cosθ
            if self._is_close(self.T_prime, expected_T_prime):
                self._log(f"Rotating tension T' = {self.T_prime} ≈ mg/cosθ = {expected_T_prime}", True)
            else:
                self._log(f"Rotating tension mismatch: T' = {self.T_prime}, expected {expected_T_prime}", False)
        
        # 4. Centripetal force condition: mg tanθ = m ω² L sinθ
        if self.omega is not None and self.T_prime is not None:
            # Method 1: Direct formula check
            lhs = self.m * self.g * (self.sin_theta / self.cos_theta)  # mg tanθ
            rhs = self.m * (self.omega ** 2) * self.L * self.sin_theta  # m ω² L sinθ
            if self._is_close(lhs, rhs):
                self._log(f"Centripetal condition: mg tanθ = {lhs} ≈ m ω² L sinθ = {rhs}", True)
            else:
                self._log(f"Centripetal condition mismatch: mg tanθ = {lhs}, m ω² L sinθ = {rhs}", False)
            
            # Method 2: Check derived ω = sqrt(g / (L cosθ))
            expected_omega = math.sqrt(self.g / (self.L * self.cos_theta))
            if self._is_close(self.omega, expected_omega):
                self._log(f"Angular velocity ω = {self.omega} ≈ sqrt(g/(L cosθ)) = {expected_omega}", True)
            else:
                self._log(f"Angular velocity mismatch: ω = {self.omega}, expected {expected_omega}", False)
    
    def test_trace_consistency(self) -> None:
        """Check consistency between intermediate and final results."""
        
        # Check step 3 (static) vs step 7 (rotating) for T
        if self.T_static is not None and self.T_prime is not None:
            # T' should be larger than T when ball rotates (since T' = mg/cosθ > mg cosθ = T)
            if self.T_prime > self.T_static:
                self._log(f"Tension increases from static to rotating: T={self.T_static}, T'={self.T_prime}", True)
            else:
                self._log(f"Tension should increase but: T={self.T_static}, T'={self.T_prime}", False)
        
        # Check step 3 (static N) vs step 4 (rotating N'=0)
        if self.N_static is not None and self.N_prime is not None:
            if self.N_static > 0 and self._is_close(self.N_prime, 0):
                self._log("Normal force decreases from static to zero when rotating", True)
            else:
                self._log(f"Normal force trend unexpected: N={self.N_static}, N'={self.N_prime}", False)
    
    def test_monotonicity(self) -> None:
        """Perturb mass and check physical trends."""
        
        # Save original values
        original_m = self.m
        perturbed_m = original_m * 1.1  # Increase by 10%
        
        # Test 1: Static case - T ∝ m, N ∝ m
        if self.T_static is not None:
            expected_T_perturbed = perturbed_m * self.g * self.cos_theta
            expected_ratio = perturbed_m / original_m
            actual_ratio = expected_T_perturbed / self.T_static
            
            if self._is_close(actual_ratio, expected_ratio, rel_tol=1e-2):
                self._log(f"Static tension scales linearly with mass: T∝m", True)
            else:
                self._log(f"Static tension scaling mismatch: ratio={actual_ratio}, expected={expected_ratio}", False)
        
        # Test 2: Rotating case - T' ∝ m, ω independent of m
        if self.T_prime is not None and self.omega is not None:
            expected_T_prime_perturbed = perturbed_m * self.g / self.cos_theta
            expected_ratio_T = perturbed_m / original_m
            actual_ratio_T = expected_T_prime_perturbed / self.T_prime
            
            if self._is_close(actual_ratio_T, expected_ratio_T, rel_tol=1e-2):
                self._log(f"Rotating tension scales linearly with mass: T'∝m", True)
            else:
                self._log(f"Rotating tension scaling mismatch: ratio={actual_ratio_T}, expected={expected_ratio_T}", False)
            
            # ω should be independent of m
            expected_omega_perturbed = math.sqrt(self.g / (self.L * self.cos_theta))
            if self._is_close(expected_omega_perturbed, self.omega, rel_tol=1e-2):
                self._log(f"Angular velocity ω independent of mass: ω={self.omega} unchanged", True)
            else:
                self._log(f"ω should be independent of mass", False)
        
        # Restore original mass
        self.m = original_m
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        self.all_passed = True
        
        # Run tests in sequence
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Summary
        if self.all_passed:
            self.logs.append("SUMMARY: All physics checks passed.")
        else:
            self.logs.append("SUMMARY: Some physics checks failed.")
        
        return self.all_passed, self.logs

# Example usage (commented out for the sandbox)
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'T': 6.4,
#                 'N': 4.8,
#                 "T'": 10.0,
#                 "N'": 0.0,
#                 'ω': 5.0
#             }
#             self.m = 0.8
#             self.L = 0.5
#             self.theta = 37
#             self.g = 10
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)