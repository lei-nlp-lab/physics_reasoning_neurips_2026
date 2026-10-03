import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Constants given in the problem (safe extraction with defaults)
        self.m = getattr(solution_instance, 'm', 2.0e4)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.F1 = getattr(solution_instance, 'F1', 4.0e4)   # catapult force
        self.F2 = getattr(solution_instance, 'F2', 1.2e5)   # jet thrust
        self.v2 = getattr(solution_instance, 'v2', 36.0)    # final takeoff speed
        self.dt = 4.0  # time difference: t2 = t1 + 4 s
        
        # Friction force: 0.2 * weight
        self.Ff = 0.2 * self.m * self.g
        
        # Extracted results (using safe access from solution_instance.results if available)
        # If not found, use 0.0 as default (but tests will likely fail)
        results = getattr(solution_instance, 'results', {})
        
        self.x = getattr(solution_instance, 'x', results.get('x', 0.0))
        self.a1 = getattr(solution_instance, 'a_1', results.get('a_1', 0.0))
        self.v1 = getattr(solution_instance, 'v_1', results.get('v_1', 0.0))
        self.x1 = getattr(solution_instance, 'x_1', results.get('x_1', 0.0))
        self.l = getattr(solution_instance, 'l', results.get('l', 0.0))
        self.a2 = getattr(solution_instance, 'a_2', results.get('a_2', 0.0))
        
        # For completeness, also store v2 (already stored) and any other needed
        # Note: v2 is given as input, but also might be in results; we prioritize given value
        if 'v_2' in results:
            self.v2 = results.get('v_2', self.v2)
        
        # Initialize test logs
        self.logs = []
    
    def _log(self, message, success):
        """Helper to append a test result to logs."""
        self.logs.append((success, message))
    
    def test_invariants(self):
        """Check fundamental physical laws: Newton's second law and kinematic equations."""
        try:
            # 1. Newton's second law for catapult phase: a1 = (F1 + F2 - Ff) / m
            expected_a1 = (self.F1 + self.F2 - self.Ff) / self.m
            if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's law (catapult phase) failed: a1={self.a1}, expected {expected_a1}", False)
            else:
                self._log("Newton's law (catapult phase) passed.", True)
            
            # 2. Newton's second law for latter phase: a2 = (F2 - Ff) / m
            expected_a2 = (self.F2 - self.Ff) / self.m
            if not math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's law (latter phase) failed: a2={self.a2}, expected {expected_a2}", False)
            else:
                self._log("Newton's law (latter phase) passed.", True)
            
            # 3. Kinematic: v1^2 = 2 * a1 * x  (from v^2 = v0^2 + 2 a Δx, with v0=0)
            lhs = self.v1**2
            rhs = 2 * self.a1 * self.x
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Kinematic eq (catapult) failed: v1^2={lhs}, 2*a1*x={rhs}", False)
            else:
                self._log("Kinematic eq (catapult) passed.", True)
            
            # 4. Kinematic: v2^2 = v1^2 + 2 * a2 * x1
            lhs = self.v2**2
            rhs = self.v1**2 + 2 * self.a2 * self.x1
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Kinematic eq (latter) failed: v2^2={lhs}, v1^2+2*a2*x1={rhs}", False)
            else:
                self._log("Kinematic eq (latter) passed.", True)
            
            # 5. Total length: l = x + x1
            if not math.isclose(self.l, self.x + self.x1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Total length mismatch: l={self.l}, x+x1={self.x+self.x1}", False)
            else:
                self._log("Total length consistency passed.", True)
            
            # 6. Time condition: t2 = t1 + 4
            t1 = self.v1 / self.a1 if self.a1 != 0 else 0.0
            t2 = (self.v2 - self.v1) / self.a2 if self.a2 != 0 else 0.0
            if not math.isclose(t2, t1 + self.dt, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Time condition failed: t2={t2}, t1+4={t1+self.dt}", False)
            else:
                self._log("Time condition passed.", True)
                
        except Exception as e:
            self._log(f"Exception in test_invariants: {traceback.format_exc()}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Recompute v1 using the time condition and given v2
            # From t1 = v1/a1, t2 = (v2 - v1)/a2, and t2 = t1 + 4
            # => (v2 - v1)/a2 = v1/a1 + 4
            # Solve for v1: v2/a2 - v1/a2 = v1/a1 + 4
            # => v2/a2 - 4 = v1*(1/a1 + 1/a2)
            # => v1 = (v2/a2 - 4) / (1/a1 + 1/a2)
            if self.a1 != 0 and self.a2 != 0:
                computed_v1 = (self.v2/self.a2 - self.dt) / (1/self.a1 + 1/self.a2)
                if not math.isclose(self.v1, computed_v1, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"v1 consistency failed: stored={self.v1}, computed={computed_v1}", False)
                else:
                    self._log("v1 trace consistency passed.", True)
            else:
                self._log("Skipping v1 trace check due to zero acceleration.", False)
            
            # Recompute x from v1 and a1: x = v1^2/(2*a1)
            if self.a1 != 0:
                computed_x = self.v1**2 / (2 * self.a1)
                if not math.isclose(self.x, computed_x, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"x consistency failed: stored={self.x}, computed={computed_x}", False)
                else:
                    self._log("x trace consistency passed.", True)
            
            # Recompute x1 from v1, v2, a2: x1 = (v2^2 - v1^2)/(2*a2)
            if self.a2 != 0:
                computed_x1 = (self.v2**2 - self.v1**2) / (2 * self.a2)
                if not math.isclose(self.x1, computed_x1, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"x1 consistency failed: stored={self.x1}, computed={computed_x1}", False)
                else:
                    self._log("x1 trace consistency passed.", True)
            
            # Recompute l from x and x1
            computed_l = self.x + self.x1
            if not math.isclose(self.l, computed_l, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"l consistency failed: stored={self.l}, computed={computed_l}", False)
            else:
                self._log("l trace consistency passed.", True)
                
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {traceback.format_exc()}", False)
    
    def test_monotonicity(self):
        """
        Perturb the mass m and recalculate key quantities to check that the model
        responds continuously and in a physically plausible way.
        """
        try:
            # Store original values
            original_m = self.m
            perturbed_m = original_m * 1.01  # increase by 1%
            
            # Recalculate for perturbed mass
            Ff_pert = 0.2 * perturbed_m * self.g
            a1_pert = (self.F1 + self.F2 - Ff_pert) / perturbed_m
            a2_pert = (self.F2 - Ff_pert) / perturbed_m
            
            # Solve for t1 using the time condition and given v2
            # t1 = (v2 - 4*a2_pert) / (a1_pert + a2_pert)
            t1_pert = (self.v2 - self.dt * a2_pert) / (a1_pert + a2_pert)
            v1_pert = a1_pert * t1_pert
            x_pert = 0.5 * a1_pert * t1_pert**2
            t2_pert = t1_pert + self.dt
            x1_pert = v1_pert * t2_pert + 0.5 * a2_pert * t2_pert**2
            l_pert = x_pert + x1_pert
            
            # Check that the recalculated values for the original mass (using the same formulas)
            # match the stored values (within tolerance). This validates the formulas.
            t1_orig = (self.v2 - self.dt * self.a2) / (self.a1 + self.a2)
            v1_orig_formula = self.a1 * t1_orig
            x_orig_formula = 0.5 * self.a1 * t1_orig**2
            t2_orig = t1_orig + self.dt
            x1_orig_formula = v1_orig_formula * t2_orig + 0.5 * self.a2 * t2_orig**2
            l_orig_formula = x_orig_formula + x1_orig_formula
            
            if not (math.isclose(self.v1, v1_orig_formula, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.x, x_orig_formula, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.l, l_orig_formula, rel_tol=1e-2, abs_tol=1e-2)):
                self._log("Monotonicity baseline failed: formulas do not match stored values.", False)
            else:
                self._log("Monotonicity baseline passed.", True)
            
            # Check that the change is small (since perturbation is small)
            rel_change_v1 = abs(v1_pert - self.v1) / self.v1
            rel_change_x = abs(x_pert - self.x) / self.x
            rel_change_l = abs(l_pert - self.l) / self.l
            
            # Expect relative changes on the order of 1% (since mass changed by 1%)
            # Allow some flexibility: between 0.5% and 2%
            if not (1e-3 < rel_change_v1 < 0.05 and
                    1e-3 < rel_change_x < 0.05 and
                    1e-3 < rel_change_l < 0.05):
                self._log(f"Monotonicity: relative changes out of expected range: v1 {rel_change_v1:.3f}, x {rel_change_x:.3f}, l {rel_change_l:.3f}", False)
            else:
                self._log("Monotonicity: relative changes are plausible.", True)
            
            # Additionally, check that the model is continuous: the sign of change in x and l
            # should be the same as the sign of change in m (since increasing mass increases
            # the required distance in this specific setup, as observed in the example).
            if perturbed_m > original_m:
                if x_pert > self.x and l_pert > self.l:
                    self._log("Monotonicity trend: increased mass leads to increased x and l (plausible).", True)
                else:
                    self._log(f"Monotonicity trend unexpected: x {x_pert} vs {self.x}, l {l_pert} vs {self.l}", False)
            
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {traceback.format_exc()}", False)
    
    def run_all_tests(self):
        """
        Execute all tests and return a tuple (success_bool, logs_list).
        """
        self.logs = []
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success: all tests must have succeeded (True)
        all_success = all(success for success, _ in self.logs)
        
        return all_success, self.logs