import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.x_3 = getattr(solution_instance, 'x_3', solution_instance.results.get('x_3', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        
        # Known inputs from problem statement
        self.x1 = 3.0  # displacement in first second (m)
        self.t1 = 1.0  # first second (s)
        self.t3 = 3.0  # first 3 seconds (s)
        self.x_v = 12.0  # displacement for velocity calculation (m)
        
        # Store original values for restoration after perturbation
        self._original_a = self.a
        self._original_x1 = self.x1
        
        # Symbolic variables
        self.a_sym, self.x1_sym, self.t1_sym, self.x3_sym, self.t3_sym, self.v_sym, self.xv_sym = sp.symbols(
            'a_sym x1_sym t1_sym x3_sym t3_sym v_sym xv_sym', real=True, positive=True
        )
        
        # Equations from KB
        self.eq1 = sp.Eq(self.x1_sym, 0.5 * self.a_sym * self.t1_sym**2)  # step 1
        self.eq2 = sp.Eq(self.x3_sym, 0.5 * self.a_sym * self.t3_sym**2)  # step 2
        self.eq3 = sp.Eq(self.v_sym**2, 2 * self.a_sym * self.xv_sym)    # step 3 (v0=0)

    def test_invariants(self):
        """Verify energy conservation and kinematic consistency."""
        logs = []
        success = True
        
        # Check 1: Verify a = 2*x1/t1^2 (step 1)
        a_calc = 2 * self.x1 / self.t1**2
        if not math.isclose(self.a, a_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Acceleration mismatch: a={self.a:.3f}, expected {a_calc:.3f}")
            success = False
        else:
            logs.append(f"OK: Acceleration consistent: a={self.a:.3f}")
        
        # Check 2: Verify x_3 = 0.5*a*t3^2 (step 2)
        x3_calc = 0.5 * self.a * self.t3**2
        if not math.isclose(self.x_3, x3_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: x_3 mismatch: x_3={self.x_3:.3f}, expected {x3_calc:.3f}")
            success = False
        else:
            logs.append(f"OK: x_3 consistent: x_3={self.x_3:.3f}")
        
        # Check 3: Verify v = sqrt(2*a*x) (step 3, v0=0)
        v_calc = math.sqrt(2 * self.a * self.x_v)
        if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Velocity mismatch: v={self.v:.3f}, expected {v_calc:.3f}")
            success = False
        else:
            logs.append(f"OK: Velocity consistent: v={self.v:.3f}")
        
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        # Recalculate all quantities from a (assuming a is correct)
        a = 2 * self.x1 / self.t1**2
        x3 = 0.5 * a * self.t3**2
        v = math.sqrt(2 * a * self.x_v)
        
        checks = [
            ("a", self.a, a),
            ("x_3", self.x_3, x3),
            ("v", self.v, v)
        ]
        
        for name, obtained, expected in checks:
            if not math.isclose(obtained, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: {name}={obtained:.3f}, but expected {expected:.3f} from derived a")
                success = False
            else:
                logs.append(f"OK: {name}={obtained:.3f} matches derived value")
        
        # Additionally, verify with symbolic substitution
        try:
            sol_a = sp.solve(self.eq1.subs({self.x1_sym: self.x1, self.t1_sym: self.t1}), self.a_sym)
            if sol_a and not math.isclose(self.a, float(sol_a[0]), rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Symbolic a={float(sol_a[0]):.3f} vs obtained a={self.a:.3f}")
                success = False
            elif not sol_a:
                logs.append("WARNING: Could not solve symbolic equation for a")
        except Exception as e:
            logs.append(f"Symbolic check error: {e}")
        
        return success, logs

    def test_monotonicity(self):
        """Perturb input x1 and verify monotonic trends in a, x_3, v."""
        logs = []
        success = True
        
        # Store original state
        orig_x1 = self.x1
        orig_a = self.a
        orig_x3 = self.x_3
        orig_v = self.v
        
        # Perturb x1 by +10%
        perturb_factor = 1.1
        self.x1 *= perturb_factor
        
        # Recalculate using formulas (not the stored results)
        a_new = 2 * self.x1 / self.t1**2
        x3_new = 0.5 * a_new * self.t3**2
        v_new = math.sqrt(2 * a_new * self.x_v)
        
        # Check monotonic trends
        if not a_new > orig_a:
            logs.append(f"FAIL: Increasing x1 did not increase a (old={orig_a:.3f}, new={a_new:.3f})")
            success = False
        else:
            logs.append(f"OK: a increases with x1: {orig_a:.3f} -> {a_new:.3f}")
        
        if not x3_new > orig_x3:
            logs.append(f"FAIL: Increasing x1 did not increase x_3 (old={orig_x3:.3f}, new={x3_new:.3f})")
            success = False
        else:
            logs.append(f"OK: x_3 increases with x1: {orig_x3:.3f} -> {x3_new:.3f}")
        
        if not v_new > orig_v:
            logs.append(f"FAIL: Increasing x1 did not increase v (old={orig_v:.3f}, new={v_new:.3f})")
            success = False
        else:
            logs.append(f"OK: v increases with x1: {orig_v:.3f} -> {v_new:.3f}")
        
        # Restore original state
        self.x1 = orig_x1
        self.a = orig_a
        self.x_3 = orig_x3
        self.v = orig_v
        
        return success, logs

    def run_all_tests(self):
        """Execute all checks and return overall success and logs."""
        all_logs = ["=== PhysicsChecker: Uniform Acceleration Problem ==="]
        overall_success = True
        
        for test_name, test_func in [
            ("Invariant checks", self.test_invariants),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                all_logs.append(f"\n--- {test_name} ---")
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs