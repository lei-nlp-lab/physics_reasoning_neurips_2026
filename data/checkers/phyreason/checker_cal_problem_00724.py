import sympy as sp
import math
import traceback
from copy import deepcopy

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Use getattr with fallbacks to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Extract inputs (with safe defaults from problem statement)
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 0.1))
        self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'inputs', {}).get('q', 0.01))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 10.0))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'inputs', {}).get('L', math.sqrt(3)))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        
        # Extract results from solution_instance.results (with fallback to 0.0)
        results = getattr(solution_instance, 'results', {})
        self.t_m = results.get('t_m', getattr(solution_instance, 't_m', 0.0))
        self.T = results.get('T', getattr(solution_instance, 'T', 0.0))
        self.t = results.get('t', getattr(solution_instance, 't', 0.0))
        self.theta = results.get('theta', getattr(solution_instance, 'theta', 0.0))
        self.v = results.get('v', getattr(solution_instance, 'v', 0.0))
        self.r_1 = results.get('r_1', getattr(solution_instance, 'r_1', 0.0))
        self.E = results.get('E', getattr(solution_instance, 'E', 0.0))
        
        # Store logs for test results
        self.logs = []
        
    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append({"test": message, "success": success})
        
    def test_invariants(self):
        """
        Test fundamental physical invariants:
        1. Force equilibrium: qE = mg (electric vs gravitational)
        2. Lorentz force as centripetal force: qvB = mv²/r₁
        3. Period consistency: T = 2πr₁/v = 2πm/(qB)
        4. Angle-time relation: t_m = (θ/(2π)) * T = θm/(qB)
        """
        try:
            # Test 1: Force equilibrium (qE = mg)
            expected_E = self.m * self.g / self.q
            if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Force equilibrium (qE = mg) satisfied", True)
            else:
                self._log(f"Force equilibrium failed: E={self.E}, expected {expected_E}", False)
            
            # Test 2: Lorentz centripetal force (qvB = mv²/r₁)
            # Rearrange to v = qB r₁ / m
            expected_v = self.q * self.B * self.r_1 / self.m
            if math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Lorentz force as centripetal force satisfied", True)
            else:
                self._log(f"Lorentz force failed: v={self.v}, expected {expected_v}", False)
            
            # Test 3: Period consistency
            T1 = 2 * math.pi * self.r_1 / self.v if self.v != 0 else 0
            T2 = 2 * math.pi * self.m / (self.q * self.B)
            
            # Check if T matches either formula (allow 2% tolerance)
            if math.isclose(self.T, T1, rel_tol=1e-2, abs_tol=1e-2) or \
               math.isclose(self.T, T2, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Period consistency (T = 2πr₁/v = 2πm/(qB)) satisfied", True)
            else:
                self._log(f"Period failed: T={self.T}, expected ~{T1} or ~{T2}", False)
            
            # Test 4: Angle-time relation
            expected_t_m = (self.theta / (2 * math.pi)) * self.T
            alt_t_m = self.theta * self.m / (self.q * self.B)  # Alternative formula
            
            if math.isclose(self.t_m, expected_t_m, rel_tol=1e-2, abs_tol=1e-2) or \
               math.isclose(self.t_m, alt_t_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Angle-time relation (t_m = (θ/2π)T = θm/(qB)) satisfied", True)
            else:
                self._log(f"Angle-time failed: t_m={self.t_m}, expected ~{expected_t_m}", False)
                
            # Additional invariant: Geometric radius r₁ = L * sin(60°)
            expected_r1 = self.L * math.sin(math.radians(60))
            if math.isclose(self.r_1, expected_r1, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Geometric radius (r₁ = L·sin60°) satisfied", True)
            else:
                self._log(f"Geometric radius failed: r₁={self.r_1}, expected {expected_r1}", False)
                
            # Check if angle is 4π/3 (≈4.18879 rad)
            expected_theta = 4 * math.pi / 3
            if math.isclose(self.theta, expected_theta, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Central angle θ = 4π/3 satisfied", True)
            else:
                self._log(f"Angle θ mismatch: {self.theta} vs expected 4π/3", False)
                
        except Exception as e:
            self._log(f"Invariant test crashed: {str(e)}", False)
            
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify the step-by-step logic from the KB.
        """
        try:
            # Step 1: E from equilibrium
            expected_E = self.m * self.g / self.q
            step1_ok = math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2)
            
            # Step 2: r₁ from geometry
            expected_r1 = self.L * math.sqrt(3) / 2  # sin60° = √3/2
            step2_ok = math.isclose(self.r_1, expected_r1, rel_tol=1e-2, abs_tol=1e-2)
            
            # Step 3: v from centripetal force
            expected_v = self.q * self.B * self.r_1 / self.m
            step3_ok = math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2)
            
            # Step 4: Period T (could be from either formula)
            T_formula1 = 2 * math.pi * self.r_1 / self.v
            T_formula2 = 2 * math.pi * self.m / (self.q * self.B)
            step4_ok = (math.isclose(self.T, T_formula1, rel_tol=1e-2, abs_tol=1e-2) or
                       math.isclose(self.T, T_formula2, rel_tol=1e-2, abs_tol=1e-2))
            
            # Step 6 & 7: Angle and time
            expected_theta = 4 * math.pi / 3
            step6_ok = math.isclose(self.theta, expected_theta, rel_tol=1e-2, abs_tol=1e-2)
            
            expected_t_m = self.theta * self.m / (self.q * self.B)
            step7_ok = math.isclose(self.t_m, expected_t_m, rel_tol=1e-2, abs_tol=1e-2)
            
            if all([step1_ok, step2_ok, step3_ok, step4_ok, step6_ok, step7_ok]):
                self._log("Trace consistency (all steps logical) satisfied", True)
            else:
                failed_steps = []
                if not step1_ok: failed_steps.append("Step1: E")
                if not step2_ok: failed_steps.append("Step2: r₁")
                if not step3_ok: failed_steps.append("Step3: v")
                if not step4_ok: failed_steps.append("Step4: T")
                if not step6_ok: failed_steps.append("Step6: θ")
                if not step7_ok: failed_steps.append("Step7: t_m")
                self._log(f"Trace consistency failed at: {', '.join(failed_steps)}", False)
                
        except Exception as e:
            self._log(f"Trace consistency test crashed: {str(e)}", False)
            
    def test_monotonicity(self):
        """
        Perturb mass m and check that results change in the expected direction.
        Then restore original values.
        """
        original_m = self.m
        original_E = self.E
        original_v = self.v
        original_T = self.T
        original_t_m = self.t_m
        
        try:
            # Increase mass by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected values using formulas
            new_E = perturbed_m * self.g / self.q  # E = mg/q
            new_v = self.q * self.B * self.r_1 / perturbed_m  # v = qB r₁ / m
            new_T = 2 * math.pi * perturbed_m / (self.q * self.B)  # T = 2πm/(qB)
            new_t_m = self.theta * perturbed_m / (self.q * self.B)  # t_m = θm/(qB)
            
            # Check monotonic trends
            trends_ok = True
            if not (new_E > original_E):
                self._log(f"Monotonicity: E should increase with m (was {original_E}, becomes {new_E})", False)
                trends_ok = False
            if not (new_v < original_v):
                self._log(f"Monotonicity: v should decrease with m (was {original_v}, becomes {new_v})", False)
                trends_ok = False
            if not (new_T > original_T):
                self._log(f"Monotonicity: T should increase with m (was {original_T}, becomes {new_T})", False)
                trends_ok = False
            if not (new_t_m > original_t_m):
                self._log(f"Monotonicity: t_m should increase with m (was {original_t_m}, becomes {new_t_m})", False)
                trends_ok = False
                
            if trends_ok:
                self._log("Monotonicity (sensitivity to mass perturbation) satisfied", True)
                
        except Exception as e:
            self._log(f"Monotonicity test crashed: {str(e)}", False)
        finally:
            # Restore original values
            self.m = original_m
            self.E = original_E
            self.v = original_v
            self.T = original_T
            self.t_m = original_t_m
            
    def run_all_tests(self):
        """Run all tests and return success boolean and logs."""
        self.logs = []
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success (all tests passed)
        success = all(log["success"] for log in self.logs)
        
        return success, self.logs

# Example usage:
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()