import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of all inputs
        self.L = getattr(solution_instance, 'L', 30.0)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.v = getattr(solution_instance, 'v', 4.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.8)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        
        # Safe extraction of results (using both direct attributes and results dictionary)
        self.results = getattr(solution_instance, 'results', {})
        
        # Primary target answers
        self.t_total = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.E_mech = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        self.Q_heat = getattr(solution_instance, 'Q', self.results.get('Q', 0.0))
        self.E_motor = getattr(solution_instance, 'E_textmotor', self.results.get('E_textmotor', 0.0))
        
        # Intermediate results (with fallback values from steps analysis)
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.t1 = getattr(solution_instance, 't_1', self.results.get('t_1', 0.0))
        self.x1 = getattr(solution_instance, 'x_1', self.results.get('x_1', 0.0))
        self.t2 = getattr(solution_instance, 't_2', self.results.get('t_2', 0.0))
        self.Ek = getattr(solution_instance, 'E_k', self.results.get('E_k', 0.0))
        self.Ep = getattr(solution_instance, 'E_P', self.results.get('E_P', 0.0))
        self.x_conv = getattr(solution_instance, 'x_textconveyorbelt', self.results.get('x_textconveyorbelt', 0.0))
        self.delta_x = getattr(solution_instance, 'Deltax', self.results.get('Deltax', 0.0))
        
        # Logging storage
        self.logs = []
        
    def _log(self, msg, success=True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {msg}")
        return success
    
    def _isclose(self, a, b):
        """Robust fuzzy comparison for physics values."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self):
        """Verify fundamental physical laws."""
        try:
            # 1. Newton's Second Law: a = μ g cosθ - g sinθ
            expected_a = self.mu * self.g * self.cos37 - self.g * self.sin37
            if not self._isclose(self.a, expected_a):
                self._log(f"Newton's 2nd Law violation: a={self.a}, expected={expected_a}", False)
            else:
                self._log(f"Newton's 2nd Law verified: a={self.a:.3f} m/s²")
            
            # 2. Energy Conservation: E_motor = E_mech + Q_heat
            if not self._isclose(self.E_motor, self.E_mech + self.Q_heat):
                self._log(f"Energy conservation violation: E_motor={self.E_motor}, E+Q={self.E_mech+self.Q_heat}", False)
            else:
                self._log(f"Energy conservation verified: E_motor={self.E_motor:.1f} J = E+Q")
            
            # 3. Friction work: Q = μ m g cosθ * Δx
            expected_Q = self.mu * self.m * self.g * self.cos37 * self.delta_x
            if not self._isclose(self.Q_heat, expected_Q):
                self._log(f"Friction work mismatch: Q={self.Q_heat}, μmgcosθΔx={expected_Q}", False)
            else:
                self._log(f"Friction work verified: Q={self.Q_heat:.1f} J")
            
            return all("FAIL" not in log for log in self.logs[-3:])
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Verify internal consistency of all intermediate calculations."""
        try:
            # 1. Acceleration phase time: t1 = v / a
            expected_t1 = self.v / self.a if abs(self.a) > 1e-9 else float('inf')
            if not self._isclose(self.t1, expected_t1):
                self._log(f"t1 inconsistent: {self.t1} vs v/a={expected_t1}", False)
            else:
                self._log(f"t1 consistent: {self.t1:.1f} s")
            
            # 2. Acceleration distance: x1 = v²/(2a)
            expected_x1 = self.v**2 / (2 * self.a) if abs(self.a) > 1e-9 else 0.0
            if not self._isclose(self.x1, expected_x1):
                self._log(f"x1 inconsistent: {self.x1} vs v²/(2a)={expected_x1}", False)
            else:
                self._log(f"x1 consistent: {self.x1:.1f} m")
            
            # 3. Uniform motion time: t2 = (L - x1) / v
            expected_t2 = (self.L - self.x1) / self.v
            if not self._isclose(self.t2, expected_t2):
                self._log(f"t2 inconsistent: {self.t2} vs (L-x1)/v={expected_t2}", False)
            else:
                self._log(f"t2 consistent: {self.t2:.1f} s")
            
            # 4. Total time: t = t1 + t2
            expected_t = self.t1 + self.t2
            if not self._isclose(self.t_total, expected_t):
                self._log(f"Total time inconsistent: {self.t_total} vs t1+t2={expected_t}", False)
            else:
                self._log(f"Total time consistent: {self.t_total:.1f} s")
            
            # 5. Kinetic energy: Ek = 0.5 * m * v²
            expected_Ek = 0.5 * self.m * self.v**2
            if not self._isclose(self.Ek, expected_Ek):
                self._log(f"Kinetic energy inconsistent: {self.Ek} vs 0.5mv²={expected_Ek}", False)
            else:
                self._log(f"Kinetic energy consistent: {self.Ek:.1f} J")
            
            # 6. Potential energy: Ep = m * g * L * sinθ
            height = self.L * self.sin37
            expected_Ep = self.m * self.g * height
            if not self._isclose(self.Ep, expected_Ep):
                self._log(f"Potential energy inconsistent: {self.Ep} vs mgh={expected_Ep}", False)
            else:
                self._log(f"Potential energy consistent: {self.Ep:.1f} J")
            
            # 7. Mechanical energy: E = Ek + Ep
            expected_E = self.Ek + self.Ep
            if not self._isclose(self.E_mech, expected_E):
                self._log(f"Mechanical energy inconsistent: {self.E_mech} vs Ek+Ep={expected_E}", False)
            else:
                self._log(f"Mechanical energy consistent: {self.E_mech:.1f} J")
            
            # 8. Conveyor displacement during acceleration: x_conv = v * t1
            expected_x_conv = self.v * self.t1
            if not self._isclose(self.x_conv, expected_x_conv):
                self._log(f"Conveyor displacement inconsistent: {self.x_conv} vs v*t1={expected_x_conv}", False)
            else:
                self._log(f"Conveyor displacement consistent: {self.x_conv:.1f} m")
            
            # 9. Relative displacement: Δx = x_conv - x1
            expected_delta_x = self.x_conv - self.x1
            if not self._isclose(self.delta_x, expected_delta_x):
                self._log(f"Relative displacement inconsistent: {self.delta_x} vs x_conv-x1={expected_delta_x}", False)
            else:
                self._log(f"Relative displacement consistent: {self.delta_x:.1f} m")
            
            # 10. Motor energy: E_motor = E + Q
            expected_E_motor = self.E_mech + self.Q_heat
            if not self._isclose(self.E_motor, expected_E_motor):
                self._log(f"Motor energy inconsistent: {self.E_motor} vs E+Q={expected_E_motor}", False)
            else:
                self._log(f"Motor energy consistent: {self.E_motor:.1f} J")
            
            return all("FAIL" not in log for log in self.logs[-10:])
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and verify expected trends."""
        original_m = self.m
        try:
            # Double the mass
            perturbed_m = 2.0 * original_m
            
            # Recalculate all quantities using formulas from steps analysis
            # Acceleration should remain same (independent of m)
            a_pert = self.mu * self.g * self.cos37 - self.g * self.sin37
            
            # Times and distances remain same (independent of m)
            t1_pert = self.v / a_pert
            x1_pert = self.v**2 / (2 * a_pert)
            t2_pert = (self.L - x1_pert) / self.v
            t_pert = t1_pert + t2_pert
            x_conv_pert = self.v * t1_pert
            delta_x_pert = x_conv_pert - x1_pert
            
            # Energy quantities scale linearly with m
            Ek_pert = 0.5 * perturbed_m * self.v**2
            Ep_pert = perturbed_m * self.g * (self.L * self.sin37)
            E_pert = Ek_pert + Ep_pert
            Q_pert = self.mu * perturbed_m * self.g * self.cos37 * delta_x_pert
            E_motor_pert = E_pert + Q_pert
            
            # Verify scaling factors (all should be 2.0 for linear scaling)
            scaling_ok = True
            
            # Acceleration unchanged
            if not self._isclose(a_pert, self.a):
                self._log(f"Acceleration changed with mass perturbation: {a_pert} vs {self.a}", False)
                scaling_ok = False
            
            # Times and distances unchanged
            for name, orig, pert in [("t1", self.t1, t1_pert), ("x1", self.x1, x1_pert),
                                    ("t2", self.t2, t2_pert), ("t_total", self.t_total, t_pert),
                                    ("x_conv", self.x_conv, x_conv_pert), ("delta_x", self.delta_x, delta_x_pert)]:
                if not self._isclose(orig, pert):
                    self._log(f"{name} changed with mass perturbation: {pert} vs {orig}", False)
                    scaling_ok = False
            
            # Energies scale by factor 2
            scale_factor = perturbed_m / original_m
            for name, orig, pert in [("Ek", self.Ek, Ek_pert), ("Ep", self.Ep, Ep_pert),
                                    ("E", self.E_mech, E_pert), ("Q", self.Q_heat, Q_pert),
                                    ("E_motor", self.E_motor, E_motor_pert)]:
                expected = orig * scale_factor
                if not self._isclose(pert, expected):
                    self._log(f"{name} scaling violation: {pert} vs {expected} (factor {scale_factor})", False)
                    scaling_ok = False
            
            if scaling_ok:
                self._log(f"Monotonicity test passed: All quantities scale correctly with mass")
            else:
                self._log(f"Monotonicity test failed: Some quantities don't scale as expected", False)
            
            return scaling_ok
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
        finally:
            # Restore original mass
            self.m = original_m
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        try:
            self._log("Starting Physics Audit")
            
            # Run individual tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            # Final summary
            if success:
                self._log("All physics checks passed!")
            else:
                self._log("Some physics checks failed - review required", False)
                
        except Exception as e:
            self._log(f"Unexpected error in test suite: {traceback.format_exc()}", False)
            success = False
        
        return success, self.logs

# Optional: Example solution class for testing (not part of the checker)
if __name__ == "__main__":
    class MockSolution:
        def __init__(self):
            # Inputs
            self.L = 30.0
            self.theta = 37.0
            self.v = 4.0
            self.m = 1.0
            self.mu = 0.8
            self.g = 10.0
            self.sin37 = 0.6
            self.cos37 = 0.8
            
            # Results (from steps analysis)
            self.results = {
                'a': 0.4,
                't_1': 10.0,
                'x_1': 20.0,
                't_2': 2.5,
                't': 12.5,
                'E_k': 8.0,
                'E_P': 180.0,
                'E': 188.0,
                'x_textconveyorbelt': 40.0,
                'Deltax': 20.0,
                'Q': 128.0,
                'E_textmotor': 316.0
            }
    
    # Test the checker
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    print(f"Success: {success}")
    for log in logs:
        print(log)