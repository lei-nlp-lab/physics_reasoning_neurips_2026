import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Robust initialization with safe attribute access.
        Extracts physical parameters and results from the solution instance.
        """
        # Safe extraction of parameters with defaults
        self.m = getattr(solution_instance, 'm', 0.2)
        self.q = getattr(solution_instance, 'q', 4e-4)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.4)
        self.E = getattr(solution_instance, 'E', 1e3)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        self.v_result = getattr(solution_instance, 'v', results.get('v', None))
        self.x_result = getattr(solution_instance, 'x', results.get('x', None))
        
        self.logs = []
    
    def _log(self, message: str):
        """Helper to append log messages."""
        self.logs.append(message)
    
    def test_step1_energy(self) -> bool:
        """
        Check work-energy theorem from top to point A (bottom of arc).
        Equation from KB: mgR - qER = 1/2 * m * v^2
        """
        try:
            if self.v_result is None:
                self._log("Step 1 check failed: v_result is None")
                return False
            
            # Numerical check with tolerance
            lhs_num = self.m * self.g * self.R - self.q * self.E * self.R
            rhs_num = 0.5 * self.m * (self.v_result**2)
            numerical_match = math.isclose(lhs_num, rhs_num, rel_tol=1e-2, abs_tol=1e-2)
            
            if not numerical_match:
                self._log(f"Step 1 energy mismatch: LHS={lhs_num:.3f}, RHS={rhs_num:.3f}, diff={abs(lhs_num-rhs_num):.3f}")
                return False
            self._log("Step 1 energy conservation passed.")
            return True
        except Exception as e:
            self._log(f"Step 1 check failed with error: {e}")
            return False
    
    def test_step2_energy(self) -> bool:
        """
        Check work-energy from top to stopping point on horizontal surface.
        CORRECTED Equation: mgR - qE(R + x) - μmgx = 0
        Or equivalently: mgR - qER - qEx - μmgx = 0
        """
        try:
            if self.x_result is None:
                self._log("Step 2 check failed: x_result is None")
                return False
            
            # Numerical check with CORRECTED equation
            lhs_num = self.m * self.g * self.R - self.q * self.E * (self.R + self.x_result) - self.mu * self.m * self.g * self.x_result
            numerical_match = math.isclose(lhs_num, 0, rel_tol=1e-2, abs_tol=1e-2)
            
            if not numerical_match:
                self._log(f"Step 2 energy mismatch: total work={lhs_num:.3f} != 0")
                return False
            self._log("Step 2 energy conservation passed.")
            return True
        except Exception as e:
            self._log(f"Step 2 check failed with error: {e}")
            return False
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check that intermediate results (v) are consistent with final x.
        Using energy from point A to stop: 1/2 m v^2 = (qE + μmg)x
        """
        try:
            if self.v_result is None or self.x_result is None:
                self._log("Trace consistency skipped: missing v or x")
                return False
            
            # Energy at point A equals work done on horizontal surface
            ke_a = 0.5 * self.m * (self.v_result**2)
            work_horizontal = (self.q * self.E + self.mu * self.m * self.g) * self.x_result
            
            if not math.isclose(ke_a, work_horizontal, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace inconsistency: KE_A={ke_a:.3f}, work_horizontal={work_horizontal:.3f}")
                return False
            self._log("Trace consistency passed.")
            return True
        except Exception as e:
            self._log(f"Trace consistency check failed: {e}")
            return False
    
    def test_monotonicity(self) -> bool:
        """
        Perturb mass (increase by 1%), recalculate v and x using formulas,
        check that both increase (since larger mass means more gravitational energy).
        Restore original mass after check.
        """
        original_m = self.m
        perturbed_m = original_m * 1.01  # 1% increase
        
        try:
            # Recalculate v using step1 formula
            v_original = math.sqrt(2 * (self.g * self.R - self.q * self.E * self.R / original_m))
            v_perturbed = math.sqrt(2 * (self.g * self.R - self.q * self.E * self.R / perturbed_m))
            
            # Recalculate x using CORRECTED step2 formula
            # From: mgR - qE(R + x) - μmgx = 0
            # Solve for x: x = (mgR - qER) / (qE + μmg)
            if (self.q * self.E + self.mu * original_m * self.g) == 0:
                self._log("Monotonicity check failed: division by zero in x calculation")
                return False
                
            x_original = (original_m * self.g * self.R - self.q * self.E * self.R) / (self.q * self.E + self.mu * original_m * self.g)
            x_perturbed = (perturbed_m * self.g * self.R - self.q * self.E * self.R) / (self.q * self.E + self.mu * perturbed_m * self.g)
            
            # Check monotonic increase
            v_increases = v_perturbed > v_original
            x_increases = x_perturbed > x_original
            
            if not (v_increases and x_increases):
                self._log(f"Monotonicity failed: v_orig={v_original:.3f}, v_pert={v_perturbed:.3f}, "
                         f"x_orig={x_original:.3f}, x_pert={x_perturbed:.3f}")
                return False
            self._log("Monotonicity passed.")
            return True
        except Exception as e:
            self._log(f"Monotonicity check failed: {e}")
            return False
        finally:
            self.m = original_m  # Restore original mass
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all checks and return overall success and logs."""
        self.logs.clear()
        
        tests = [
            ("Step1 Energy", self.test_step1_energy),
            ("Step2 Energy", self.test_step2_energy),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            if not test_func():
                success = False
                self._log(f"{test_name} FAILED")
            else:
                self._log(f"{test_name} PASSED")
        
        return success, self.logs