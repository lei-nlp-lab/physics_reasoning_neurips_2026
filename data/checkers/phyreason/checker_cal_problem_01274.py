import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs
        self.h = getattr(solution_instance, 'h', 1.25)
        self.v0 = getattr(solution_instance, 'v0', 5.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.R = getattr(solution_instance, 'R', 0.5)
        self.theta = getattr(solution_instance, 'theta', 37.0)  # in degrees
        self.l = getattr(solution_instance, 'l', 5.0)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.f_E = getattr(solution_instance, 'f_E', 40.0)  # friction at E
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        
        # Safely extract results from solution_instance.results dictionary
        results = getattr(solution_instance, 'results', {})
        self.F_N = results.get('F_N', 0.0)
        self.a_1 = results.get('a_1', 0.0)
        self.a_2 = results.get('a_2', 0.0)
        self.t_min = results.get('t_min', 0.0)
        self.t_max = results.get('t_max', 0.0)
        self.t_1 = results.get('t_1', 0.0)
        self.t_2 = results.get('t_2', 0.0)
        self.v_E = results.get('v_E', 0.0)
        self.v_belt = results.get('v', 0.0)  # conveyor belt speed
        self.x_BC = results.get('x_BC', 0.0)
        self.x_1 = results.get('x_1', 0.0)
        
        # Additional derived values for checking
        self.t_fall = math.sqrt(2 * self.h / self.g)
        self.v_C = math.sqrt(self.v0**2 + 2 * self.g * self.h)
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def test_invariants(self):
        """Check energy conservation and Newton's laws."""
        successes = []
        
        # 1. Projectile motion: time to fall
        t_calc = math.sqrt(2 * self.h / self.g)
        successes.append(self._log(
            f"Fall time: expected {t_calc:.3f} s, got {self.t_fall:.3f} s",
            math.isclose(self.t_fall, t_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        # 2. Horizontal displacement BC
        x_BC_calc = self.v0 * self.t_fall
        successes.append(self._log(
            f"x_BC: expected {x_BC_calc:.3f} m, got {self.x_BC:.3f} m",
            math.isclose(self.x_BC, x_BC_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        # 3. Energy conservation from A to C (no friction)
        E_A = 0.5 * self.m * self.v0**2 + self.m * self.g * self.h
        E_C = 0.5 * self.m * self.v_C**2
        successes.append(self._log(
            f"Energy A->C: {E_A:.3f} J vs {E_C:.3f} J",
            math.isclose(E_A, E_C, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        # 4. Normal force at E from given friction
        F_N_calc = self.f_E / self.mu
        successes.append(self._log(
            f"Normal force at E: expected {F_N_calc:.3f} N, got {self.F_N:.3f} N",
            math.isclose(self.F_N, F_N_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        # 5. Centripetal force at E (Newton's second law for circular motion)
        centripetal_term = self.m * self.v_E**2 / self.R
        weight_component = self.m * self.g * self.cos37
        F_N_from_motion = weight_component + centripetal_term
        successes.append(self._log(
            f"F_N from motion: {F_N_from_motion:.3f} N vs {self.F_N:.3f} N",
            math.isclose(self.F_N, F_N_from_motion, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        # 6. Accelerations on conveyor belt
        a1_calc = self.g * self.sin37 + self.mu * self.g * self.cos37
        successes.append(self._log(
            f"a_1: expected {a1_calc:.3f} m/s², got {self.a_1:.3f} m/s²",
            math.isclose(self.a_1, a1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        a2_calc = self.g * self.sin37 - self.mu * self.g * self.cos37
        successes.append(self._log(
            f"a_2: expected {a2_calc:.3f} m/s², got {self.a_2:.3f} m/s²",
            math.isclose(self.a_2, a2_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        successes = []
        
        # 1. v_E from centripetal equation (already checked in invariants)
        # 2. t_1 and t_2 for v_belt = 4 m/s
        if self.v_belt > 0:
            t1_calc = (self.v_E - self.v_belt) / self.a_1
            successes.append(self._log(
                f"t_1 for v={self.v_belt}: expected {t1_calc:.3f} s, got {self.t_1:.3f} s",
                math.isclose(self.t_1, t1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            ))
            
            x1_calc = (self.v_E**2 - self.v_belt**2) / (2 * self.a_1)
            successes.append(self._log(
                f"x_1 for v={self.v_belt}: expected {x1_calc:.3f} m, got {self.x_1:.3f} m",
                math.isclose(self.x_1, x1_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            ))
            
            # t_2 from remaining distance: l - x_1 = v_belt * t_2 - 0.5 * a_2 * t_2**2
            # Because acceleration a_2 is opposite to velocity (deceleration)
            # Rearrange: 0.5*a_2*t_2^2 - v_belt*t_2 + (l - x_1) = 0
            a, b, c = 0.5 * self.a_2, -self.v_belt, (self.l - self.x_1)
            disc = b**2 - 4*a*c
            if disc >= 0:
                # Two positive roots? Take the one that matches t_2 = v_belt / a_2 when x_1=0? Actually solve.
                t2_pos1 = (-b + math.sqrt(disc)) / (2*a)
                t2_pos2 = (-b - math.sqrt(disc)) / (2*a)
                # Choose the positive root that is close to the stored t_2
                if math.isclose(self.t_2, t2_pos1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    t2_calc = t2_pos1
                else:
                    t2_calc = t2_pos2
                successes.append(self._log(
                    f"t_2 calc: {t2_calc:.3f} s vs {self.t_2:.3f} s",
                    math.isclose(self.t_2, t2_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
                ))
            else:
                successes.append(self._log("t_2: discriminant negative", False))
        
        # 3. t_max = t_1 + t_2
        t_max_calc = self.t_1 + self.t_2
        successes.append(self._log(
            f"t_max: {t_max_calc:.3f} s vs {self.t_max:.3f} s",
            math.isclose(self.t_max, t_max_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        
        # 4. t_min from quadratic: l = v_E*t_min - 0.5*a_2*t_min^2
        # Rearrange: 0.5*a_2*t_min^2 - v_E*t_min + l = 0
        a, b, c = 0.5 * self.a_2, -self.v_E, self.l
        disc = b**2 - 4*a*c
        if disc >= 0:
            t_min_calc1 = (-b + math.sqrt(disc)) / (2*a)
            t_min_calc2 = (-b - math.sqrt(disc)) / (2*a)
            # Choose the smaller positive root
            t_min_calc = min(t_min_calc1, t_min_calc2)
            successes.append(self._log(
                f"t_min calc: {t_min_calc:.3f} s vs {self.t_min:.3f} s",
                math.isclose(self.t_min, t_min_calc, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            ))
        else:
            successes.append(self._log("t_min: no real root", False))
        
        # 5. Final answers consistency
        # Target 1: x_BC = 2.5 m
        successes.append(self._log(
            f"Final x_BC: {self.x_BC:.3f} m vs 2.5 m",
            math.isclose(self.x_BC, 2.5, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        # Target 2: v_E = 6 m/s
        successes.append(self._log(
            f"Final v_E: {self.v_E:.3f} m/s vs 6 m/s",
            math.isclose(self.v_E, 6.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        ))
        # Target 3: v_belt >= 4 m/s and t in [1, 2.2] s
        v_ok = self.v_belt >= 4.0 - self.abs_tol
        successes.append(self._log(f"v_belt >= 4: {self.v_belt:.3f} m/s", v_ok))
        t_min_ok = math.isclose(self.t_min, 1.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        successes.append(self._log(f"t_min ≈ 1 s: {self.t_min:.3f} s", t_min_ok))
        t_max_ok = math.isclose(self.t_max, 2.2, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
        successes.append(self._log(f"t_max ≈ 2.2 s: {self.t_max:.3f} s", t_max_ok))
        
        return all(successes)
    
    def test_monotonicity(self):
        """Perturb mass m and check trends."""
        original_m = self.m
        successes = []
        
        # Increase mass by 10%
        self.m = original_m * 1.1
        
        # Recompute expected changes
        # Normal force at E: F_N = m*g*cos37 + m*v_E^2/R, but v_E also depends on m
        # From F_N = f_E / mu (constant), we have:
        # v_E^2 = (F_N/m - g*cos37) * R
        # So if m increases, v_E decreases.
        F_N_new = self.f_E / self.mu  # unchanged
        v_E_new = math.sqrt(max(0, (F_N_new / self.m - self.g * self.cos37) * self.R))
        
        # Check that v_E decreased
        if self.v_E > 0:
            trend_ok = v_E_new < self.v_E
            successes.append(self._log(
                f"Mass increase: v_E decreased from {self.v_E:.3f} to {v_E_new:.3f} m/s",
                trend_ok
            ))
        
        # Restore original mass
        self.m = original_m
        
        # Perturb friction coefficient mu
        original_mu = self.mu
        self.mu = original_mu * 1.1  # increase by 10%
        
        # a_1 = g*sin37 + mu*g*cos37 should increase
        a1_new = self.g * self.sin37 + self.mu * self.g * self.cos37
        successes.append(self._log(
            f"mu increase: a_1 increased from {self.a_1:.3f} to {a1_new:.3f} m/s²",
            a1_new > self.a_1
        ))
        
        # a_2 = g*sin37 - mu*g*cos37 should decrease
        a2_new = self.g * self.sin37 - self.mu * self.g * self.cos37
        successes.append(self._log(
            f"mu increase: a_2 decreased from {self.a_2:.3f} to {a2_new:.3f} m/s²",
            a2_new < self.a_2
        ))
        
        # Restore mu
        self.mu = original_mu
        
        return all(successes) if successes else True
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        overall_success = True
        
        try:
            self.logs.append("=== PhysicsChecker Started ===")
            
            # Run each test
            for test_name, test_func in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                self.logs.append(f"\n--- {test_name} ---")
                try:
                    success = test_func()
                    overall_success = overall_success and success
                except Exception as e:
                    self.logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                    overall_success = False
            
            self.logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {traceback.format_exc()}")
            overall_success = False
        
        return overall_success, self.logs