import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        # Extract inputs (platform height and horizontal distance)
        self.h = getattr(solution_instance, 'h', 0.8)  # default from problem
        self.s = getattr(solution_instance, 's', 1.2)  # default from problem
        self.g = getattr(solution_instance, 'g', 10.0)  # gravitational acceleration
        
        # Extract results safely from results dictionary or attributes
        results = getattr(solution_instance, 'results', {})
        self.v0 = getattr(solution_instance, 'v_0', results.get('v_0', 0.0))
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.vy = getattr(solution_instance, 'v_y', results.get('v_y', 0.0))
        
        # Store for later restoration in monotonicity test
        self.original_h = self.h
        self.original_s = self.s
        self.original_results = {'v0': self.v0, 'v': self.v, 't': self.t, 'vy': self.vy}
        
        # Initialize logs
        self.logs = []

    def test_invariants(self):
        """
        Verify fundamental physical laws: 
        1. Energy conservation (mechanical energy for projectile motion)
        2. Kinematic consistency (free-fall time formula)
        3. Newton's 2nd Law in vertical direction (F_net = m*g)
        """
        try:
            # 1. Energy conservation: Initial total energy = Final kinetic energy
            # Initial: KE_horizontal + PE = (1/2)*m*v0^2 + m*g*h
            # Final: KE_total = (1/2)*m*v^2
            # Cancel m: v^2 = v0^2 + 2*g*h
            energy_check = self.v**2 - (self.v0**2 + 2*self.g*self.h)
            if not math.isclose(energy_check, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Energy conservation violated. v^2={self.v**2:.3f}, "
                               f"v0^2+2gh={self.v0**2 + 2*self.g*self.h:.3f}")
                return False
            
            # 2. Free-fall time from kinematics: t = sqrt(2h/g)
            t_expected = math.sqrt(2*self.h/self.g)
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Free-fall time inconsistent. t={self.t:.3f}, "
                               f"sqrt(2h/g)={t_expected:.3f}")
                return False
            
            # 3. Vertical motion: v_y = g*t (from constant acceleration)
            if not math.isclose(self.vy, self.g*self.t, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Vertical kinematics violated. vy={self.vy:.3f}, "
                               f"g*t={self.g*self.t:.3f}")
                return False
            
            # 4. Horizontal motion: s = v0*t
            if not math.isclose(self.s, self.v0*self.t, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Horizontal kinematics violated. s={self.s:.3f}, "
                               f"v0*t={self.v0*self.t:.3f}")
                return False
            
            self.logs.append("PASS: All physical invariants satisfied.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer using the steps analysis.
        """
        try:
            # Step 1: t from free fall
            t_calc = math.sqrt(2*self.h/self.g)
            if not math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 1 (time) inconsistent. t={self.t:.3f}, "
                               f"calculated={t_calc:.3f}")
                return False
            
            # Step 2: v0 from uniform motion
            v0_calc = self.s / self.t
            if not math.isclose(self.v0, v0_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 2 (v0) inconsistent. v0={self.v0:.3f}, "
                               f"calculated={v0_calc:.3f}")
                return False
            
            # Step 3: vy from free fall velocity
            vy_calc = self.g * self.t
            if not math.isclose(self.vy, vy_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 3 (vy) inconsistent. vy={self.vy:.3f}, "
                               f"calculated={vy_calc:.3f}")
                return False
            
            # Step 4: v from Pythagorean theorem
            v_calc = math.sqrt(self.v0**2 + self.vy**2)
            if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Step 4 (v) inconsistent. v={self.v:.3f}, "
                               f"calculated={v_calc:.3f}")
                return False
            
            self.logs.append("PASS: All trace steps are consistent.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """
        Perturb input h by +1%, recalculate using formulas, check expected trends, then restore.
        """
        try:
            # Perturb h upward by 1%
            perturbed_h = self.h * 1.01
            
            # Recalculate using the kinematic formulas (independent of student's method)
            t_new = math.sqrt(2*perturbed_h/self.g)
            v0_new = self.s / t_new
            vy_new = self.g * t_new
            v_new = math.sqrt(v0_new**2 + vy_new**2)
            
            # Expected trends (from physics):
            # 1. t increases with h (t ∝ √h)
            if not (t_new > self.t):
                self.logs.append(f"FAIL: Time t should increase with h. "
                               f"Old t={self.t:.3f}, new t={t_new:.3f}")
                return False
            
            # 2. v0 decreases with h (v0 = s/t, t increases)
            if not (v0_new < self.v0):
                self.logs.append(f"FAIL: v0 should decrease with h. "
                               f"Old v0={self.v0:.3f}, new v0={v0_new:.3f}")
                return False
            
            # 3. vy increases with h (vy = g*t, t increases)
            if not (vy_new > self.vy):
                self.logs.append(f"FAIL: vy should increase with h. "
                               f"Old vy={self.vy:.3f}, new vy={vy_new:.3f}")
                return False
            
            # 4. Final speed v: v^2 = v0^2 + vy^2. Since vy increases and v0 decreases, 
            # the trend is ambiguous. But from energy: v^2 = v0^2 + 2gh, so v should increase.
            if not (v_new > self.v):
                self.logs.append(f"FAIL: Speed v should increase with h (from energy). "
                               f"Old v={self.v:.3f}, new v={v_new:.3f}")
                return False
            
            self.logs.append("PASS: Monotonicity trends hold for perturbed h.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
        finally:
            # Restore original values (safety)
            self.h = self.original_h
            # Note: We don't need to restore results because we only read them

    def run_all_tests(self):
        """
        Execute all checks and return (success, logs).
        """
        self.logs = []  # Reset logs
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_pass = True
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} ---")
            if not test_func():
                all_pass = False
                self.logs.append(f"--- {test_name}: FAILED ---")
            else:
                self.logs.append(f"--- {test_name}: PASSED ---")
        
        return all_pass, self.logs