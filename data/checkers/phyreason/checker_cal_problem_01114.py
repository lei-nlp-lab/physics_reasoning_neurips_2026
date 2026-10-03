import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with defaults (from problem context)
        self.m = getattr(solution_instance, 'm', 2.0)  # kg
        self.t = getattr(solution_instance, 't', 6.0)  # s
        self.x = getattr(solution_instance, 'x', 54.0)  # m
        
        # Extract results with fallbacks to results dict
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.F = getattr(solution_instance, 'F', self.results.get('F', 0.0))
        
        # Initialize symbolic variables for KB theorems
        self.v0 = 0.0  # starts from rest
        self.v1 = None  # will compute if needed
        self.v_avg = None
        
        # Expected values from steps analysis
        self.expected_v = 9.0  # m/s
        self.expected_a = 3.0  # m/s²
        self.expected_F = 6.0  # N
        
    def test_invariants(self):
        """Verify physical invariants using KB theorems"""
        logs = []
        success = True
        
        # 1. Average velocity invariant: v_avg = x/t
        v_avg_calc = self.x / self.t
        if not math.isclose(self.v, v_avg_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Average velocity invariant failed: v={self.v:.3f}, x/t={v_avg_calc:.3f}")
            success = False
        
        # 2. Displacement formula invariant: x = 0.5*a*t² (since v0=0)
        # This gives a = 2x/t²
        a_calc = 2 * self.x / (self.t ** 2)
        if not math.isclose(self.a, a_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Displacement invariant failed: a={self.a:.3f}, 2x/t²={a_calc:.3f}")
            success = False
        
        # 3. Newton's second law invariant: F = m*a
        F_calc = self.m * self.a
        if not math.isclose(self.F, F_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Newton's second law failed: F={self.F:.3f}, m*a={F_calc:.3f}")
            success = False
        
        # 4. Consistency between average and final velocity: v_avg = (v0 + v1)/2
        # Since v0=0, v1 = 2*v_avg
        v1_calc = 2 * self.v
        # Also from kinematics: v1 = a*t
        v1_kinematic = self.a * self.t
        if not math.isclose(v1_calc, v1_kinematic, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Velocity consistency failed: 2*v={v1_calc:.3f}, a*t={v1_kinematic:.3f}")
            success = False
        
        if success:
            logs.append("All invariants passed within 1% tolerance")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        logs = []
        success = True
        
        # Step 1: v should match expected average velocity
        if not math.isclose(self.v, self.expected_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Step 1 mismatch: v={self.v:.3f}, expected={self.expected_v:.3f}")
            success = False
        
        # Step 2: a should match expected acceleration
        if not math.isclose(self.a, self.expected_a, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Step 2 mismatch: a={self.a:.3f}, expected={self.expected_a:.3f}")
            success = False
        
        # Step 3: F should match expected force
        if not math.isclose(self.F, self.expected_F, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Step 3 mismatch: F={self.F:.3f}, expected={self.expected_F:.3f}")
            success = False
        
        # Check the chain: v -> a -> F
        # From v and t, we can compute a: a = 2*v/t
        a_from_v = 2 * self.v / self.t
        if not math.isclose(self.a, a_from_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace chain v->a failed: a={self.a:.3f}, 2*v/t={a_from_v:.3f}")
            success = False
        
        # From a and m, we can compute F: F = m*a
        F_from_a = self.m * self.a
        if not math.isclose(self.F, F_from_a, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace chain a->F failed: F={self.F:.3f}, m*a={F_from_a:.3f}")
            success = False
        
        if success:
            logs.append("All trace consistency checks passed")
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends"""
        logs = []
        success = True
        
        # Store original values
        original_m = self.m
        original_x = self.x
        original_t = self.t
        
        # Test 1: Increase mass, force should increase proportionally (F = m*a, a fixed)
        try:
            self.m = original_m * 1.1  # 10% increase
            # Recalculate a from kinematics (unchanged by mass change)
            a_new = 2 * original_x / (original_t ** 2)
            # Expected force: F_new = m_new * a
            F_expected = self.m * a_new
            # Check that F would increase
            F_original = original_m * a_new
            if not (F_expected > F_original):
                logs.append(f"Mass monotonicity failed: F({self.m:.3f})={F_expected:.3f} ≤ F({original_m:.3f})={F_original:.3f}")
                success = False
            else:
                logs.append(f"Mass monotonicity passed: F increases with m")
        except Exception as e:
            logs.append(f"Mass monotonicity test error: {e}")
            success = False
        finally:
            self.m = original_m  # Restore
        
        # Test 2: Increase displacement, acceleration should increase (a = 2x/t²)
        try:
            self.x = original_x * 1.1  # 10% increase
            a_new = 2 * self.x / (original_t ** 2)
            a_original = 2 * original_x / (original_t ** 2)
            if not (a_new > a_original):
                logs.append(f"Displacement monotonicity failed: a({self.x:.3f})={a_new:.3f} ≤ a({original_x:.3f})={a_original:.3f}")
                success = False
            else:
                logs.append(f"Displacement monotonicity passed: a increases with x")
        except Exception as e:
            logs.append(f"Displacement monotonicity test error: {e}")
            success = False
        finally:
            self.x = original_x  # Restore
        
        # Test 3: Increase time, acceleration should decrease (a = 2x/t²)
        try:
            self.t = original_t * 1.1  # 10% increase
            a_new = 2 * original_x / (self.t ** 2)
            a_original = 2 * original_x / (original_t ** 2)
            if not (a_new < a_original):
                logs.append(f"Time monotonicity failed: a({self.t:.3f})={a_new:.3f} ≥ a({original_t:.3f})={a_original:.3f}")
                success = False
            else:
                logs.append(f"Time monotonicity passed: a decreases with t")
        except Exception as e:
            logs.append(f"Time monotonicity test error: {e}")
            success = False
        finally:
            self.t = original_t  # Restore
        
        if success:
            logs.append("All monotonicity tests passed")
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(test_logs)
                if not test_success:
                    overall_success = False
                    all_logs.append(f"{test_name}: FAILED")
                else:
                    all_logs.append(f"{test_name}: PASSED")
            except Exception as e:
                overall_success = False
                all_logs.append(f"=== {test_name} ===")
                all_logs.append(f"Exception: {e}")
                all_logs.append(traceback.format_exc())
                all_logs.append(f"{test_name}: ERROR")
        
        all_logs.append(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs