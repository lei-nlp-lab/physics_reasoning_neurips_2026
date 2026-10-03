import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from the solution instance or use defaults from problem context
        self.m = getattr(solution_instance, 'm', 3.0)
        self.F = getattr(solution_instance, 'F', 30.0)
        self.mu1 = getattr(solution_instance, 'mu1', 0.5)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.sin_theta = getattr(solution_instance, 'sin_theta', 0.6)
        self.cos_theta = getattr(solution_instance, 'cos_theta', 0.8)
        self.x1 = getattr(solution_instance, 'x1', 10.0)
        self.x2 = getattr(solution_instance, 'x2', 5.0)
        
        # Safely extract results from solution_instance.results dictionary
        results = getattr(solution_instance, 'results', {})
        self.a1 = results.get('a_1', None)
        self.t = results.get('t', None)
        self.a2 = results.get('a_2', None)
        self.v = results.get('v', None)
        self.mu2 = results.get('mu_2', None)
        
        # Store original values for monotonicity test
        self._original_m = self.m
        self._original_t = self.t
        self._original_mu2 = self.mu2
        
        # Tolerance for fuzzy numerical matching (1-2% difference)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for test results
        self.logs = []
    
    def _log(self, message: str):
        self.logs.append(message)
    
    def test_invariants(self) -> bool:
        """Verify Newton's Second Law and kinematic consistency."""
        success = True
        
        # 1. Check Newton's Second Law on segment AB
        expected_a1 = (self.F - self.mu1 * self.m * self.g) / self.m
        if self.a1 is not None:
            if not math.isclose(self.a1, expected_a1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Invariant failed: a1 from solution ({self.a1}) does not match Newton's Second Law ({expected_a1})")
                success = False
        else:
            self._log("Warning: a1 not found in results, skipping check.")
        
        # 2. Check kinematic relation on AB: x1 = 0.5 * a1 * t^2
        if self.a1 is not None and self.t is not None:
            expected_x1 = 0.5 * self.a1 * self.t**2
            if not math.isclose(self.x1, expected_x1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Invariant failed: x1 (given {self.x1}) does not match kinematic equation ({expected_x1})")
                success = False
        
        # 3. Check velocity at B: v = a1 * t
        if self.a1 is not None and self.t is not None and self.v is not None:
            expected_v = self.a1 * self.t
            if not math.isclose(self.v, expected_v, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Invariant failed: v from solution ({self.v}) does not match v = a1 * t ({expected_v})")
                success = False
        
        # 4. Check Newton's Second Law on segment BC
        if self.a2 is not None and self.mu2 is not None:
            expected_a2 = -self.g * self.sin_theta - self.mu2 * self.g * self.cos_theta
            if not math.isclose(self.a2, expected_a2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Invariant failed: a2 from solution ({self.a2}) does not match Newton's Second Law on incline ({expected_a2})")
                success = False
        
        # 5. Check kinematic relation on BC: 0 = v^2 + 2 * a2 * x2
        if self.v is not None and self.a2 is not None:
            expected_residual = self.v**2 + 2 * self.a2 * self.x2
            if not math.isclose(expected_residual, 0.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log(f"Invariant failed: kinematic equation on incline gives residual {expected_residual}, expected 0")
                success = False
        
        if success:
            self._log("All invariants passed.")
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with the final answer."""
        success = True
        
        # Recalculate mu2 from the velocity and incline parameters
        if self.v is not None:
            # Derived from: 0 = v^2 + 2*(-g*sinθ - mu2*g*cosθ)*x2
            # => mu2 = (v^2/(2*g*x2) - sinθ) / cosθ
            recalc_mu2 = (self.v**2 / (2 * self.g * self.x2) - self.sin_theta) / self.cos_theta
            if self.mu2 is not None:
                if not math.isclose(self.mu2, recalc_mu2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self._log(f"Trace inconsistency: mu2 from solution ({self.mu2}) does not match recalculated mu2 ({recalc_mu2})")
                    success = False
            else:
                self._log("Warning: mu2 not found in results, skipping trace check.")
        
        # Recalculate t from x1 and a1
        if self.a1 is not None:
            recalc_t = math.sqrt(2 * self.x1 / self.a1)
            if self.t is not None:
                if not math.isclose(self.t, recalc_t, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self._log(f"Trace inconsistency: t from solution ({self.t}) does not match recalculated t ({recalc_t})")
                    success = False
        
        if success:
            self._log("Trace consistency passed.")
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb mass m and check that t and mu2 change in the expected direction.
        Expected: increasing m -> increasing t, decreasing mu2.
        """
        success = True
        
        # Store original values
        original_m = self.m
        original_t = self.t
        original_mu2 = self.mu2
        
        # Perturb mass by +10%
        perturbed_m = original_m * 1.1
        
        # Recalculate using the physical formulas (independent of solution methods)
        # Step 1: Calculate a1 for perturbed mass
        perturbed_a1 = (self.F - self.mu1 * perturbed_m * self.g) / perturbed_m
        
        # Step 2: Calculate t for perturbed mass
        if perturbed_a1 <= 0:
            self._log("Monotonicity check failed: perturbed a1 is non-positive.")
            success = False
        else:
            perturbed_t = math.sqrt(2 * self.x1 / perturbed_a1)
            
            # Step 3: Calculate v for perturbed mass
            perturbed_v = perturbed_a1 * perturbed_t
            
            # Step 4: Calculate mu2 for perturbed mass
            perturbed_mu2 = (perturbed_v**2 / (2 * self.g * self.x2) - self.sin_theta) / self.cos_theta
            
            # Check trends
            if self.t is not None and self.mu2 is not None:
                # t should increase when m increases
                if not (perturbed_t > self.t):
                    self._log(f"Monotonicity failed: t did not increase when m increased. Original t={self.t}, perturbed t={perturbed_t}")
                    success = False
                
                # mu2 should decrease when m increases
                if not (perturbed_mu2 < self.mu2):
                    self._log(f"Monotonicity failed: mu2 did not decrease when m increased. Original mu2={self.mu2}, perturbed mu2={perturbed_mu2}")
                    success = False
            else:
                self._log("Warning: t or mu2 not available, skipping monotonicity check.")
                success = False
        
        # Restore original mass and results
        self.m = original_m
        self.t = original_t
        self.mu2 = original_mu2
        
        if success:
            self._log("Monotonicity check passed.")
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        
        # Run each test
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall_success = test1 and test2 and test3
        
        # Final summary
        if overall_success:
            self._log("All physics checks passed.")
        else:
            self._log("Physics checks failed. See logs above.")
        
        return overall_success, self.logs