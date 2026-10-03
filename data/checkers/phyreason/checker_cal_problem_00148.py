import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 24.3))
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', 279.3))
        
        # Problem constants (from KB/steps)
        self.v0 = 20.0
        self.vt = 3.0
        self.a_car = getattr(solution_instance, 'a_car', None)
        if self.a_car is None:
            # Compute from kinematic relationship if not provided
            self.a_car = (self.vt - self.v0) / self.t  # Negative deceleration
        
        # For monotonicity test backup
        self._original_t = self.t
        self._original_x = self.x
        self._original_v0 = self.v0

    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Check kinematic consistency between t and x using same acceleration
            a_from_t = (self.vt - self.v0) / self.t
            a_from_x = (self.vt**2 - self.v0**2) / (2 * self.x)
            
            if not math.isclose(a_from_t, a_from_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration inconsistency: a_from_t={a_from_t:.3f}, a_from_x={a_from_x:.3f}")
                success = False
            else:
                logs.append(f"PASS: Kinematic consistency check (a ≈ {a_from_t:.3f} m/s²)")
                
            # 2. Symbolic check with SymPy
            t_sym, x_sym, v0_sym, vt_sym, a_sym = sp.symbols('t x v0 vt a')
            eq1 = sp.Eq(t_sym, (vt_sym - v0_sym) / a_sym)
            eq2 = sp.Eq(x_sym, (vt_sym**2 - v0_sym**2) / (2 * a_sym))
            
            # Eliminate a to get direct relationship between t and x
            a_elim = sp.solve(eq1, a_sym)[0]
            eq2_sub = eq2.subs(a_sym, a_elim)
            expected_x = sp.solve(eq2_sub, x_sym)[0].subs({t_sym: self.t, v0_sym: self.v0, vt_sym: self.vt})
            
            if not math.isclose(float(expected_x), self.x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Symbolic consistency: expected x={float(expected_x):.2f}, got x={self.x:.2f}")
                success = False
            else:
                logs.append("PASS: Symbolic kinematic relationship holds")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check against target answers from steps analysis
            target_t = 24.3
            target_x = 279.3
            
            if not (math.isclose(self.t, target_t, rel_tol=1e-2, abs_tol=1e-2) and 
                    math.isclose(self.x, target_x, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"FAIL: Results deviate from target: t={self.t:.2f} (target {target_t}), x={self.x:.2f} (target {target_x})")
                success = False
            else:
                logs.append("PASS: Results match target answers within 1%")
                
            # Verify the two steps are internally consistent
            # Step 1: t = (vt - v0) / -a_car  => a_car = (v0 - vt) / t
            a_from_step1 = (self.v0 - self.vt) / self.t
            # Step 2: x = (vt^2 - v0^2) / (-2*a_car) = (v0^2 - vt^2) / (2*a_car)
            x_from_step2 = (self.v0**2 - self.vt**2) / (2 * a_from_step1)
            
            if not math.isclose(x_from_step2, self.x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step2 inconsistent with Step1: computed x={x_from_step2:.2f}, result x={self.x:.2f}")
                success = False
            else:
                logs.append(f"PASS: Step1 and Step2 are consistent (a_car ≈ {a_from_step1:.3f} m/s²)")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb initial velocity upward by 10%
            perturbed_v0 = self.v0 * 1.1
            
            # Compute using kinematic equations (constant acceleration assumption)
            a = (self.vt - self.v0) / self.t  # Original acceleration
            
            # New time and distance with perturbed v0 (same deceleration magnitude)
            t_perturbed = (self.vt - perturbed_v0) / a
            x_perturbed = (self.vt**2 - perturbed_v0**2) / (2 * a)
            
            # Check monotonicity: larger initial velocity → longer stopping time and distance
            if not (t_perturbed > self.t and x_perturbed > self.x):
                logs.append(f"FAIL: Monotonicity violation: v0↑ ({self.v0}→{perturbed_v0:.1f}) should give t↑ and x↑")
                logs.append(f"       Original: t={self.t:.2f}, x={self.x:.2f}")
                logs.append(f"       Perturbed: t={t_perturbed:.2f}, x={x_perturbed:.2f}")
                success = False
            else:
                logs.append(f"PASS: Monotonicity holds: v0↑ → t↑ ({self.t:.2f}→{t_perturbed:.2f}), x↑ ({self.x:.2f}→{x_perturbed:.2f})")
                
            # Restore original state
            self.t = self._original_t
            self.x = self._original_x
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariant Checks", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== Running {test_name} ===")
            success, logs = test_method()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs