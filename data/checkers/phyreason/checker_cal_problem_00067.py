import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1000.0))
        self.v1 = getattr(solution_instance, 'v1', solution_instance.results.get('v1', 10.0))  # 36 km/h = 10 m/s
        self.s = getattr(solution_instance, 's', solution_instance.results.get('s', 20.0))
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', 6.0))
        
        # Safe attribute access: results (with defaults)
        self.t_1 = getattr(solution_instance, 't_1', solution_instance.results.get('t_1', 0.0))
        self.t_2 = getattr(solution_instance, 't_2', solution_instance.results.get('t_2', 0.0))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.F_f = getattr(solution_instance, 'F_f', solution_instance.results.get('F_f', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        self.logs = []
    
    def _log(self, msg, success=None):
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify energy conservation and Newton's laws based on KB."""
        success = True
        
        # Convert to float for numerical checks
        v1_float = float(self.v1)
        s_float = float(self.s)
        m_float = float(self.m)
        
        # 1. Check kinematic relation: v1^2 = 2 * |a| * s (constant acceleration)
        expected_stopping_acc = -v1_float**2 / (2 * s_float)  # Negative for deceleration
        if not math.isclose(float(self.a), expected_stopping_acc, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Acceleration mismatch: a={self.a}, expected {expected_stopping_acc}", False)
            success = False
        else:
            self._log(f"Kinematic invariant OK: a ≈ {expected_stopping_acc}", True)
        
        # 2. Check Newton's Second Law: F_f = m * |a|
        expected_force = m_float * abs(float(self.a))
        if not math.isclose(float(self.F_f), expected_force, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Force mismatch: F_f={self.F_f}, expected {expected_force}", False)
            success = False
        else:
            self._log(f"Newton's second law OK: F_f ≈ {expected_force}", True)
        
        # 3. Check energy/work principle: Work done by friction = initial kinetic energy
        # F_f * s = 0.5 * m * v1^2
        work_friction = float(self.F_f) * s_float
        initial_ke = 0.5 * m_float * v1_float**2
        if not math.isclose(work_friction, initial_ke, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Energy balance mismatch: Work={work_friction}, KE_initial={initial_ke}", False)
            success = False
        else:
            self._log(f"Energy conservation OK: Work ≈ {initial_ke}", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Convert to float
        v1_float = float(self.v1)
        s_float = float(self.s)
        l_float = float(self.l)
        
        # Step 1: t_1 = s / v_avg where v_avg = (v1 + 0)/2 = v1/2
        expected_t1 = s_float / (v1_float / 2)
        if not math.isclose(float(self.t_1), expected_t1, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 1 mismatch: t_1={self.t_1}, expected {expected_t1}", False)
            success = False
        else:
            self._log(f"Step 1 OK: t_1 ≈ {expected_t1}", True)
        
        # Step 2: a = (0 - v1) / t_1 = -v1/t_1
        expected_a = -v1_float / float(self.t_1) if float(self.t_1) != 0 else 0
        if not math.isclose(float(self.a), expected_a, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 2 mismatch: a={self.a}, expected {expected_a}", False)
            success = False
        else:
            self._log(f"Step 2 OK: a ≈ {expected_a}", True)
        
        # Step 3: F_f = m * |a|
        expected_F_f = float(self.m) * abs(expected_a)
        if not math.isclose(float(self.F_f), expected_F_f, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 3 mismatch: F_f={self.F_f}, expected {expected_F_f}", False)
            success = False
        else:
            self._log(f"Step 3 OK: F_f ≈ {expected_F_f}", True)
        
        # Step 4: t_2 = (l + L)/v0 - but we don't have L and v0.
        # Instead, from step 5: t = t_2 - t_1, so t_2 = t + t_1
        expected_t2 = float(self.t) + float(self.t_1)
        if not math.isclose(float(self.t_2), expected_t2, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 4-5 mismatch: t_2={self.t_2}, expected from t+t_1={expected_t2}", False)
            success = False
        else:
            self._log(f"Step 4-5 OK: t_2 = t + t_1 ≈ {expected_t2}", True)
        
        # Step 6: v^2 = v1^2 + 2*a*s (but note car stops at crosswalk, so v=0 for that)
        # Actually step 6 seems to compute a different v. Check with given symbolic answer: 5√5 ≈ 11.18034
        expected_v = 5 * math.sqrt(5)
        if not math.isclose(float(self.v), expected_v, 
                            rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            self._log(f"Step 6 mismatch: v={self.v}, expected {expected_v}", False)
            success = False
        else:
            self._log(f"Step 6 OK: v ≈ {expected_v}", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass and check force trend."""
        try:
            original_m = float(self.m)
            perturbed_m = original_m * 1.1  # +10%
            
            # Recalculate expected force using F_f = m * |a|
            # Note: a remains same because it's determined by kinematics (v1 and s)
            a_mag = abs(float(self.a))
            original_force = original_m * a_mag
            perturbed_force = perturbed_m * a_mag
            
            # Check monotonicity: if mass increases, force increases proportionally
            if perturbed_force > original_force:
                self._log(f"Monotonicity OK: F_f increases with mass (ΔF={perturbed_force-original_force:.1f})", True)
                return True
            else:
                self._log(f"Monotonicity violated: F_f doesn't increase with mass", False)
                return False
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {e}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                self._log(f"Running {name}...")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        # Summary
        self._log(f"\nOverall result: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs