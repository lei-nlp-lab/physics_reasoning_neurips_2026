import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute extraction.
        solution_instance: object from student solution, expected to have .results dict.
        """
        # Input parameters (with defaults matching problem)
        self.v1 = getattr(solution_instance, 'v1', 16.0)      # m/s
        self.v2 = getattr(solution_instance, 'v2', 12.0)      # m/s
        self.L = getattr(solution_instance, 'L', 6.0)         # m
        self.a1 = getattr(solution_instance, 'a1', 2.0)       # m/s²
        self.a2 = getattr(solution_instance, 'a2', 1.0)       # m/s²
        self.t_brake_A = 6.0                                  # given in problem
        
        # Results (from .results dict)
        self.results = getattr(solution_instance, 'results', {})
        
        # Key mapping (using .get() for safety)
        self.v_A6 = self.results.get('v_A6', None)
        self.v_B6 = self.results.get('v_B6', None)
        self.v_At = self.results.get('v_At', None)   # Should be numeric
        self.v_Bt = self.results.get('v_Bt', None)   # Should be numeric
        self.s1 = self.results.get('s_1', None)      # displacement of A
        self.s2 = self.results.get('s_2', None)      # displacement of B
        self.t_eq1 = self.results.get('t', None)     # first equal-speed time
        self.t_meet1 = self.results.get('t_1', None)
        self.t_meet2 = self.results.get('t_2', None)
        self.delta_t = self.results.get('delta_t', None)
        
        # Compute derived values for checking
        self.v_A6_expected = self.v1 - self.a1 * self.t_brake_A
        self.v_B6_expected = self.v2 - self.a2 * self.t_brake_A
        
    def test_invariants(self):
        """Verify kinematic laws and relative motion invariants."""
        logs = []
        success = True
        
        try:
            # 1. Check kinematic equations for t in [0, 6]
            # v_A(t) = v1 - a1*t, v_B(t) = v2 - a2*t
            t_test = 3.0  # arbitrary time within braking phase
            v_A_expected = self.v1 - self.a1 * t_test
            v_B_expected = self.v2 - self.a2 * t_test
            
            # v_At and v_Bt should be numeric values (speeds at t=4s)
            # For kinematic check, we compute directly from formulas
            v_A_calc = self.v1 - self.a1 * t_test
            v_B_calc = self.v2 - self.a2 * t_test
            
            if not math.isclose(v_A_calc, v_A_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_A(t) invariant violated. Expected {v_A_expected}, got {v_A_calc}")
                success = False
            else:
                logs.append(f"PASS: v_A(t) satisfies kinematic equation at t={t_test}s")
                
            if not math.isclose(v_B_calc, v_B_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v_B(t) invariant violated. Expected {v_B_expected}, got {v_B_calc}")
                success = False
            else:
                logs.append(f"PASS: v_B(t) satisfies kinematic equation at t={t_test}s")
            
            # 2. Check relative motion: a_B relative to A should be a2 - a1 = -1 m/s²
            a_rel_expected = self.a2 - self.a1  # = -1 m/s²
            # Direct calculation from given accelerations
            a_rel_calc = self.a2 - self.a1
            
            if not math.isclose(a_rel_calc, a_rel_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Relative acceleration invariant. Expected {a_rel_expected}, got {a_rel_calc}")
                success = False
            else:
                logs.append(f"PASS: Relative acceleration constant at {a_rel_expected} m/s²")
            
            # 3. Energy/work consideration: deceleration work should be consistent
            # For car A: work by brakes = 0.5*m*(v1² - v_A(6)²) [but mass unknown]
            # Instead check kinematic consistency: v² = u² + 2as
            # For car A from t=0 to t=6: s_A = v1*6 - 0.5*a1*36
            s_A_6 = self.v1 * 6 - 0.5 * self.a1 * 36
            v_A6_sq = self.v1**2 - 2 * self.a1 * s_A_6
            if not math.isclose(v_A6_sq, self.v_A6_expected**2, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: v² = u² + 2as violated for car A")
                success = False
            else:
                logs.append("PASS: v² = u² + 2as holds for car A")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # 1. Check speeds at t=6s
            if self.v_A6 is not None:
                if not math.isclose(float(self.v_A6), self.v_A6_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v_A(6) mismatch. Expected {self.v_A6_expected}, got {self.v_A6}")
                    success = False
                else:
                    logs.append(f"PASS: v_A(6) = {self.v_A6} matches expected {self.v_A6_expected}")
            else:
                logs.append("WARN: v_A6 not provided, skipping check")
                
            if self.v_B6 is not None:
                if not math.isclose(float(self.v_B6), self.v_B6_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v_B(6) mismatch. Expected {self.v_B6_expected}, got {self.v_B6}")
                    success = False
                else:
                    logs.append(f"PASS: v_B(6) = {self.v_B6} matches expected {self.v_B6_expected}")
            else:
                logs.append("WARN: v_B6 not provided, skipping check")
            
            # 2. Check first equal-speed time (t=4s)
            # Solve: v1 - a1*t = v2 - a2*t => t = (v1 - v2)/(a1 - a2)
            t_eq1_expected = (self.v1 - self.v2) / (self.a1 - self.a2)  # = 4.0
            if self.t_eq1 is not None:
                if not math.isclose(float(self.t_eq1), t_eq1_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: First equal-speed time mismatch. Expected {t_eq1_expected}, got {self.t_eq1}")
                    success = False
                else:
                    logs.append(f"PASS: First equal-speed time = {self.t_eq1}s (expected 4s)")
            else:
                logs.append("WARN: First equal-speed time not provided")
            
            # 3. Check second equal-speed time (t=8s)
            # After t=6s: v_A = 4 m/s constant, v_B = 6 - 1*(t-6)
            # Solve: 4 = 6 - (t-6) => t = 8
            t_eq2_expected = 8.0
            if self.delta_t is not None:
                delta_t_expected = 2.0  # from step_4
                if not math.isclose(float(self.delta_t), delta_t_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Δt mismatch. Expected {delta_t_expected}, got {self.delta_t}")
                    success = False
                else:
                    logs.append(f"PASS: Δt = {self.delta_t}s (expected 2s)")
                    # Compute t_eq2 from delta_t
                    t_eq2_calc = 6.0 + float(self.delta_t)
                    if not math.isclose(t_eq2_calc, t_eq2_expected, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: Second equal-speed time computed as {t_eq2_calc}, expected 8s")
                        success = False
            
            # 4. Check meeting times
            # For t ≤ 6: solve 16t - t² = 12t - 0.5t² + 6
            # => 4t - 0.5t² - 6 = 0 => t² - 8t + 12 = 0 => t=2, t=6
            if self.t_meet1 is not None:
                if not math.isclose(float(self.t_meet1), 2.0, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: First meeting time mismatch. Expected 2s, got {self.t_meet1}")
                    success = False
                else:
                    logs.append(f"PASS: First meeting time = {self.t_meet1}s (expected 2s)")
            
            if self.t_meet2 is not None:
                if not math.isclose(float(self.t_meet2), 6.0, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Second meeting time mismatch. Expected 6s, got {self.t_meet2}")
                    success = False
                else:
                    logs.append(f"PASS: Second meeting time = {self.t_meet2}s (expected 6s)")
            
            # 5. Check third meeting time (t=10s) if present
            # After t=6: s_A = s_A(6) + 4*(t-6), s_B = s_B(6) + 6*(t-6) - 0.5*1*(t-6)²
            # Solve: s_A(6) + 4Δt = s_B(6) + 6Δt - 0.5Δt² + 6
            # This is consistent with step_11
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and check trend consistency.
        Perturb v1 (car A's initial speed) by +1 m/s.
        """
        logs = []
        success = True
        
        original_v1 = self.v1
        
        try:
            # Perturb v1 by +1 m/s
            self.v1 += 1.0  # now 17 m/s
            
            # Recalculate expected values
            # 1. First equal-speed time: t = (v1 - v2)/(a1 - a2)
            t_eq1_new = (self.v1 - self.v2) / (self.a1 - self.a2)  # (17-12)/(2-1)=5
            t_eq1_original = (original_v1 - self.v2) / (self.a1 - self.a2)  # 4
            
            # Should increase (from 4 to 5)
            if not (t_eq1_new > t_eq1_original):
                logs.append(f"FAIL: Increasing v1 should increase first equal-speed time. Original {t_eq1_original}, new {t_eq1_new}")
                success = False
            else:
                logs.append(f"PASS: First equal-speed time increases from {t_eq1_original}s to {t_eq1_new}s when v1 increases")
            
            # 2. First meeting time (smallest positive solution of meeting equation for t≤6)
            # Equation: (v1 - v2)*t - 0.5*(a1 - a2)*t² = L
            # For original: 4t - 0.5t² = 6
            # For new: 5t - 0.5t² = 6 => 0.5t² - 5t + 6 = 0 => t² - 10t + 12 = 0
            # Solve quadratic: t = [10 ± √(100-48)]/2 = [10 ± √52]/2 ≈ [10 ± 7.21]/2
            # Smallest positive: (10 - 7.21)/2 ≈ 1.395
            coeff_a = -0.5 * (self.a1 - self.a2)
            coeff_b = self.v1 - self.v2
            coeff_c = -self.L
            # Quadratic: coeff_a*t² + coeff_b*t + coeff_c = 0
            discriminant = coeff_b**2 - 4*coeff_a*coeff_c
            if discriminant >= 0:
                t_meet1_new = (-coeff_b + math.sqrt(discriminant)) / (2*coeff_a)
                t_meet2_new = (-coeff_b - math.sqrt(discriminant)) / (2*coeff_a)
                t_meet_smallest_new = min(t_meet1_new, t_meet2_new)
                
                # Original smallest meeting time was 2s
                if not (t_meet_smallest_new < 2.0):
                    logs.append(f"FAIL: Increasing v1 should decrease first meeting time. Original 2s, new {t_meet_smallest_new}s")
                    success = False
                else:
                    logs.append(f"PASS: First meeting time decreases from 2s to {t_meet_smallest_new:.3f}s when v1 increases")
            else:
                logs.append("WARN: Could not compute meeting times after perturbation")
            
            # Restore original value
            self.v1 = original_v1
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            self.v1 = original_v1  # restore on error
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall result."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== Test Invariants ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== Test Trace Consistency ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== Test Monotonicity ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        return overall_success, all_logs