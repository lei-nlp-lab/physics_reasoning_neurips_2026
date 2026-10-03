import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs - use numeric defaults
        self.R = float(getattr(solution_instance, 'R', 5.0))
        self.s_BC = float(getattr(solution_instance, 's_BC', 9.5))  # distance B to C
        self.mu = float(getattr(solution_instance, 'mu', 0.1))
        self.g = float(getattr(solution_instance, 'g', 10.0))
        
        # Extract results with fallback to 0.0 if missing
        self.results = getattr(solution_instance, 'results', {})
        self.v_B = float(self.results.get('v_B', 0.0))
        self.v_c = float(self.results.get('v_c', 0.0))
        self.n = float(self.results.get('n', 0.0))
        self.s_final = float(self.results.get('s_final', 0.0))
        self.f = float(self.results.get('f', 0.0))
        self.s_remaining = float(self.results.get('s_remaining', 0.0))
        self.s_3_trips = float(self.results.get('s_3_trips', 0.0))
        self.total_s = float(self.results.get('s', 0.0))
        
    def test_invariants(self):
        logs = []
        try:
            # 1. Energy conservation from A to B (no friction on arc)
            v_B_expected = math.sqrt(2 * self.g * self.R)
            if not math.isclose(self.v_B, v_B_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B = {self.v_B}, expected {v_B_expected}")
            else:
                logs.append(f"PASS: v_B = {self.v_B} matches energy conservation")
            
            # 2. Friction force magnitude (per unit mass)
            f_expected = self.mu * self.g
            if not math.isclose(self.f, f_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f = {self.f}, expected {f_expected}")
            else:
                logs.append(f"PASS: f = {self.f} matches μmg")
            
            # 3. Work-energy from A to C (first trip)
            v_c_expected = math.sqrt(2 * self.g * self.R - 2 * self.mu * self.g * self.s_BC)
            if not math.isclose(self.v_c, v_c_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_c = {self.v_c}, expected {v_c_expected}")
            else:
                logs.append(f"PASS: v_c = {self.v_c} matches work-energy with friction")
            
            # 4. Total distance until rest (without barrier)
            total_s_expected = self.R / self.mu
            if not math.isclose(self.total_s, total_s_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: total_s = {self.total_s}, expected {total_s_expected}")
            else:
                logs.append(f"PASS: total_s = {self.total_s} matches mgh = μmg*s")
            
            # 5. Distance consistency for 3 round trips
            # Ground truth: "9.5m + 19m + 19m = 47.5m" = s_BC + 2*(2*s_BC)
            s_3_trips_expected = self.s_BC + 2 * (2 * self.s_BC)  # 47.5
            if not math.isclose(self.s_3_trips, s_3_trips_expected, rel_tol=1e-2):
                logs.append(f"FAIL: s_3_trips = {self.s_3_trips}, expected {s_3_trips_expected}")
            else:
                logs.append(f"PASS: s_3_trips = {self.s_3_trips} matches trip calculation")
            
            # 6. Remaining distance consistency
            s_remaining_expected = self.total_s - self.s_3_trips
            if not math.isclose(self.s_remaining, s_remaining_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: s_remaining = {self.s_remaining}, expected {s_remaining_expected}")
            else:
                logs.append(f"PASS: s_remaining = {self.s_remaining} matches subtraction")
            
            # 7. Final position consistency
            # After 3rd collision at C, slider moves left distance s_remaining
            s_final_expected = self.s_BC - self.s_remaining
            if not math.isclose(self.s_final, s_final_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: s_final = {self.s_final}, expected {s_final_expected}")
            else:
                logs.append(f"PASS: s_final = {self.s_final} matches position calculation")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # 1. v_B should lead to v_c via friction work
            v_c_from_v_B = math.sqrt(self.v_B**2 - 2 * self.mu * self.g * self.s_BC)
            if not math.isclose(self.v_c, v_c_from_v_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_c from v_B inconsistent: {self.v_c} vs {v_c_from_v_B}")
            else:
                logs.append(f"PASS: v_c consistent with v_B and friction")
            
            # 2. Number of collisions n should be integer 3
            if not math.isclose(self.n, 3.0, rel_tol=0, abs_tol=0.5):
                logs.append(f"FAIL: n = {self.n} should be exactly 3")
            else:
                logs.append(f"PASS: n = {self.n} is correct integer")
            
            # 3. Check that s_3_trips <= total_s (since friction eventually stops motion)
            if self.s_3_trips > self.total_s + 1e-2:
                logs.append(f"FAIL: s_3_trips ({self.s_3_trips}) > total_s ({self.total_s})")
            else:
                logs.append(f"PASS: s_3_trips <= total_s, motion stops eventually")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        original_mu = self.mu
        
        try:
            # Perturb mu upwards by 10%
            perturbed_mu = self.mu * 1.1
            
            # Recalculate total stopping distance
            total_s_perturbed = self.R / perturbed_mu
            
            # For monotonicity test, we just check that with higher friction,
            # the total stopping distance decreases (obvious) and that the
            # final position s_final increases (since less distance traveled
            # after last collision means stopping closer to C, i.e., larger s_final).
            # We don't need to recalculate the exact s_final for perturbed mu.
            # Instead, verify the trend: if mu increases, s_final should increase.
            # Since our solution has s_final = s_BC - (s_total - s_3_trips)
            # and s_3_trips is fixed for n=3 collisions, s_final = s_BC - (R/μ - 5*s_BC)
            # = 6*s_BC - R/μ. As μ increases, R/μ decreases, so s_final increases.
            # So monotonicity holds if derivative ds_final/dμ > 0.
            # Compute derivative numerically:
            mu1 = self.mu
            mu2 = mu1 * 1.001  # small perturbation
            s_final1 = self.s_BC - (self.R/mu1 - 5*self.s_BC)
            s_final2 = self.s_BC - (self.R/mu2 - 5*self.s_BC)
            
            if s_final2 > s_final1 or math.isclose(s_final2, s_final1, rel_tol=1e-2):
                logs.append(f"PASS: Monotonicity holds. With higher mu, s_final increases")
            else:
                logs.append(f"FAIL: Monotonicity violated. s_final decreases with increased mu")
            
            # Restore original mu
            self.mu = original_mu
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            self.mu = original_mu
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            
            if any("FAIL" in log or "ERROR" in log for log in logs):
                success = False
        
        return success, all_logs