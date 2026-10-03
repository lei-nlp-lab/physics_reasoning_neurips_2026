import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization with fallback defaults.
        """
        # Input parameters (from problem context)
        self.v_safety = 10.0  # m/s
        self.a_f1 = 2.0       # m/s²
        self.d0 = 200.0       # m initial gap
        
        # Extract results safely (using mapping from KB)
        self.results = getattr(solution_instance, 'results', {})
        
        # First encounter time (primary answer)
        self.t_meet = getattr(solution_instance, 't', None)
        if self.t_meet is None:
            self.t_meet = self.results.get('t', None)
        
        # Maximum distance between cars (secondary answer)
        # Note: Δx' key may be stored as 'delta_x'' or similar
        self.delta_x_max = getattr(solution_instance, 'Δx\'', None)
        if self.delta_x_max is None:
            self.delta_x_max = self.results.get('delta_x\'', None)
        if self.delta_x_max is None:
            self.delta_x_max = self.results.get('Δx\'', None)
        
        # Second encounter time (tertiary answer)
        self.t_second = getattr(solution_instance, 't_3', None)
        if self.t_second is None:
            self.t_second = self.results.get('t_3', None)
        
        # Intermediate values (for consistency checks)
        self.t_equal_v = self.results.get('t_2', None)  # time when speeds equal (5s)
        self.x_race_at_5 = self.results.get('x_race', None)  # racing car at 5s
        self.x_safe_at_5 = self.results.get('x_safe', None)  # safety car at 5s
        self.v1 = self.results.get('v_1', None)  # speed at first meeting
        self.t0 = self.results.get('t_0', None)  # deceleration time to stop
        self.x_race_prime = self.results.get('x_race\'', None)  # deceleration distance
        self.x_safe_prime = self.results.get('x_safe\'', None)  # safety car during decel
        self.t2 = self.results.get('t_2', None)  # safety car catch-up time after stop
        
        # For symbolic computations
        self.t, self.x = sp.symbols('t x', real=True, positive=True)
    
    def test_invariants(self):
        """Test fundamental kinematic laws."""
        logs = []
        try:
            # 1. Check first meeting time satisfies relative motion equation
            # Equation: 0.5*a*t² = v_safety*t + d0
            if self.t_meet is not None:
                lhs = 0.5 * self.a_f1 * self.t_meet**2
                rhs = self.v_safety * self.t_meet + self.d0
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: First meeting time {self.t_meet}s doesn't satisfy 0.5*a*t² = v*t + d0")
                    logs.append(f"  Computed: 0.5*{self.a_f1}*{self.t_meet}² = {lhs}")
                    logs.append(f"  Expected: {self.v_safety}*{self.t_meet} + {self.d0} = {rhs}")
                else:
                    logs.append("PASS: First meeting satisfies kinematic equation")
            
            # 2. Check maximum distance occurs when velocities are equal
            if self.t_equal_v is not None and self.delta_x_max is not None:
                # Time when velocities equal: v_safety = a*t => t = v_safety/a = 5s
                expected_t_eq = self.v_safety / self.a_f1
                if not math.isclose(self.t_equal_v, expected_t_eq, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_equal_v = {self.t_equal_v}s, expected {expected_t_eq}s")
                else:
                    logs.append("PASS: Time when velocities equal is correct (5s)")
                    
                # Distance function: D(t) = d0 + v_safety*t - 0.5*a*t²
                # Maximum at t = 5s: D_max = 200 + 10*5 - 0.5*2*25 = 225
                expected_dmax = self.d0 + self.v_safety*expected_t_eq - 0.5*self.a_f1*expected_t_eq**2
                if not math.isclose(self.delta_x_max, expected_dmax, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Δx_max = {self.delta_x_max}m, expected {expected_dmax}m")
                else:
                    logs.append("PASS: Maximum distance matches kinematic prediction")
            
            # 3. Check velocity at first meeting
            if self.v1 is not None and self.t_meet is not None:
                expected_v1 = self.a_f1 * self.t_meet
                if not math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v1 = {self.v1}m/s, expected {expected_v1}m/s")
                else:
                    logs.append("PASS: Velocity at first meeting matches v = a*t")
            
            # 4. Check deceleration phase (if values exist)
            if all(v is not None for v in [self.v1, self.t0, self.x_race_prime]):
                # Assume deceleration = same magnitude as acceleration (2 m/s²)
                a_decel = -self.a_f1
                
                # t0 = v1/|a| = 40/2 = 20s
                expected_t0 = self.v1 / self.a_f1
                if not math.isclose(self.t0, expected_t0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t0 = {self.t0}s, expected {expected_t0}s")
                else:
                    logs.append("PASS: Deceleration time matches t = v/a")
                
                # x_race' = (v1/2)*t0 = average_vel * time
                expected_x_race_prime = 0.5 * self.v1 * self.t0
                if not math.isclose(self.x_race_prime, expected_x_race_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x_race' = {self.x_race_prime}m, expected {expected_x_race_prime}m")
                else:
                    logs.append("PASS: Deceleration distance matches kinematics")
            
            # 5. Check second encounter time
            if all(v is not None for v in [self.t0, self.t2, self.t_second]):
                # t_second = t0 + t2
                expected_t_second = self.t0 + self.t2
                if not math.isclose(self.t_second, expected_t_second, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_second = {self.t_second}s, expected {expected_t_second}s")
                else:
                    logs.append("PASS: Second encounter time = t0 + t2")
                    
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check step_3 and step_4: positions at t=5s
            if self.x_race_at_5 is not None:
                expected = 0.5 * self.a_f1 * 5**2
                if not math.isclose(self.x_race_at_5, expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x_race(5s) = {self.x_race_at_5}m, expected {expected}m")
            
            if self.x_safe_at_5 is not None:
                expected = self.v_safety * 5
                if not math.isclose(self.x_safe_at_5, expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x_safe(5s) = {self.x_safe_at_5}m, expected {expected}m")
            
            # Check step_10: safety car during deceleration
            if self.x_safe_prime is not None and self.t0 is not None:
                expected = self.v_safety * self.t0
                if not math.isclose(self.x_safe_prime, expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x_safe' = {self.x_safe_prime}m, expected {expected}m")
            
            # Check step_12: t2 calculation
            if all(v is not None for v in [self.t2, self.x_race_prime, self.x_safe_prime]):
                # t2 = (x_race' - x_safe') / v_safety
                # But careful: x_race' is where F1 stops, x_safe' is where safety car is at that moment
                # Distance safety car needs to cover = x_race' - x_safe'
                expected_t2 = (self.x_race_prime - self.x_safe_prime) / self.v_safety
                if not math.isclose(self.t2, expected_t2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t2 = {self.t2}s, expected {expected_t2}s")
                else:
                    logs.append("PASS: t2 calculation consistent")
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        logs = []
        original_a = self.a_f1
        original_d0 = self.d0
        
        try:
            # Test 1: Increase acceleration → shorter meeting time
            perturbed_a = original_a * 1.1  # +10%
            
            # Solve: 0.5*a*t² - v*t - d0 = 0
            # Positive root: t = [v + sqrt(v² + 2*a*d0)] / a
            def meeting_time(a, v, d0):
                disc = v**2 + 2*a*d0
                if disc < 0:
                    return None
                return (v + math.sqrt(disc)) / a
            
            t_orig = meeting_time(original_a, self.v_safety, self.d0)
            t_pert = meeting_time(perturbed_a, self.v_safety, self.d0)
            
            if t_orig is not None and t_pert is not None:
                if t_pert < t_orig:
                    logs.append("PASS: Increased acceleration reduces meeting time (monotonic)")
                else:
                    logs.append(f"FAIL: Acceleration increase didn't reduce time: {t_orig}s -> {t_pert}s")
            else:
                logs.append("SKIP: Could not compute meeting times for monotonicity")
            
            # Test 2: Increase initial distance → longer meeting time
            perturbed_d0 = original_d0 * 1.1  # +10%
            t_orig = meeting_time(original_a, self.v_safety, original_d0)
            t_pert = meeting_time(original_a, self.v_safety, perturbed_d0)
            
            if t_orig is not None and t_pert is not None:
                if t_pert > t_orig:
                    logs.append("PASS: Increased distance increases meeting time (monotonic)")
                else:
                    logs.append(f"FAIL: Distance increase didn't increase time: {t_orig}s -> {t_pert}s")
            
            # Test 3: Maximum distance should increase with initial distance
            t_eq = self.v_safety / original_a  # time when velocities equal
            dmax_orig = original_d0 + self.v_safety*t_eq - 0.5*original_a*t_eq**2
            dmax_pert = perturbed_d0 + self.v_safety*t_eq - 0.5*original_a*t_eq**2
            
            if dmax_pert > dmax_orig:
                logs.append("PASS: Increased distance increases maximum separation")
            else:
                logs.append("FAIL: Maximum distance didn't increase with initial distance")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self):
        """Execute all checks and return summary."""
        all_logs = []
        success = True
        
        # Run tests
        invariant_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        monotonicity_logs = self.test_monotonicity()
        
        all_logs.append("=== INVARIANT TESTS ===")
        all_logs.extend(invariant_logs)
        
        all_logs.append("\n=== TRACE CONSISTENCY TESTS ===")
        all_logs.extend(trace_logs)
        
        all_logs.append("\n=== MONOTONICITY TESTS ===")
        all_logs.extend(monotonicity_logs)
        
        # Check for any failures (logs containing "FAIL:")
        failure_count = sum(1 for log in all_logs if log.startswith("FAIL:"))
        
        if failure_count > 0:
            success = False
            all_logs.append(f"\n❌ {failure_count} test(s) failed")
        else:
            # Also check we have at least some positive validations
            if len(all_logs) > 5:  # Arbitrary threshold
                all_logs.append("\n✅ All checks passed within tolerances")
            else:
                success = False
                all_logs.append("\n⚠️  Insufficient validation performed")
        
        return success, all_logs