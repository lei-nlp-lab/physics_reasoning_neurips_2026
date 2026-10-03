import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract ALL needed values from results dictionary with sensible defaults
        # Input values should also be in results if needed
        self.v0 = solution_instance.results.get('v0', 10.0)  # Default example value
        self.v1 = solution_instance.results.get('v1', 2.0)
        self.v2 = solution_instance.results.get('v2', 3.0)
        self.dt0 = solution_instance.results.get('Δt0', solution_instance.results.get('dt0', 1.0))
        
        # Results
        self.s = solution_instance.results.get('s', 0.0)
        self.dt = solution_instance.results.get('Δt', solution_instance.results.get('delta_t', 0.0))
        self.s1 = solution_instance.results.get('s_1', 0.0)
        self.s2 = solution_instance.results.get('s_2', 0.0)
        self.s3 = solution_instance.results.get('s_3', 0.0)
        self.s4 = solution_instance.results.get('s_4', 0.0)
        
        # Store original for monotonicity test
        self._original_inputs = {
            'v0': self.v0,
            'v1': self.v1,
            'v2': self.v2,
            'dt0': self.dt0
        }
        
        # Logs
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify physical laws from KB using only numeric comparisons"""
        try:
            # Check uniform linear motion: s1 = v0 * dt0
            expected_s1 = self.v0 * self.dt0
            if not math.isclose(self.s1, expected_s1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL invariant s1: actual={self.s1}, expected={expected_s1}")
                return False
            
            # Check uniform linear motion: s2 = v1 * dt0
            expected_s2 = self.v1 * self.dt0
            if not math.isclose(self.s2, expected_s2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL invariant s2: actual={self.s2}, expected={expected_s2}")
                return False
            
            # Check vector addition (relative motion): s = s1 + s2
            expected_s_from_step3 = self.s1 + self.s2
            if not math.isclose(self.s, expected_s_from_step3, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL invariant s from step3: actual={self.s}, expected={expected_s_from_step3}")
                return False
            
            # Check uniform linear motion: s3 = v0 * dt
            expected_s3 = self.v0 * self.dt
            if not math.isclose(self.s3, expected_s3, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL invariant s3: actual={self.s3}, expected={expected_s3}")
                return False
            
            # Check uniform linear motion: s4 = v2 * dt
            expected_s4 = self.v2 * self.dt
            if not math.isclose(self.s4, expected_s4, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL invariant s4: actual={self.s4}, expected={expected_s4}")
                return False
            
            # Check relative motion: s = s3 - s4
            expected_s_from_step5 = self.s3 - self.s4
            if not math.isclose(self.s, expected_s_from_step5, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL invariant s from step5: actual={self.s}, expected={expected_s_from_step5}")
                return False
            
            # Check derived formula: dt = (v0+v1)*dt0/(v0-v2)
            # Avoid division by zero
            if abs(self.v0 - self.v2) < 1e-10:
                self.log("SKIP invariant dt: v0 - v2 is zero (would cause division by zero)")
                return True
            expected_dt = (self.v0 + self.v1) * self.dt0 / (self.v0 - self.v2)
            if not math.isclose(self.dt, expected_dt, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL invariant dt: actual={self.dt}, expected={expected_dt}")
                return False
            
            self.log("PASS all invariants")
            return True
            
        except Exception as e:
            self.log(f"ERROR in invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check that s computed from steps matches final s
            s_from_step3 = (self.v0 + self.v1) * self.dt0
            if not math.isclose(self.s, s_from_step3, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL trace s: step3={s_from_step3}, final={self.s}")
                return False
            
            # Check that dt computed from steps matches final dt
            # Avoid division by zero
            if abs(self.v0 - self.v2) < 1e-10:
                self.log("SKIP trace dt: v0 - v2 is zero")
                return True
            dt_from_step6 = (self.v0 + self.v1) * self.dt0 / (self.v0 - self.v2)
            if not math.isclose(self.dt, dt_from_step6, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL trace dt: step6={dt_from_step6}, final={self.dt}")
                return False
            
            # Check consistency between s from step3 and s from step5
            s_from_step5 = (self.v0 - self.v2) * self.dt
            if not math.isclose(s_from_step3, s_from_step5, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL trace s consistency: step3={s_from_step3}, step5={s_from_step5}")
                return False
            
            self.log("PASS trace consistency")
            return True
            
        except Exception as e:
            self.log(f"ERROR in trace: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends"""
        try:
            # Avoid division by zero in original calculation
            if abs(self.v0 - self.v2) < 1e-10:
                self.log("SKIP monotonicity: v0 - v2 is zero (would cause division by zero)")
                return True
            
            # Perturb v1 by 10%
            perturbed_v1 = self.v1 * 1.1
            
            # Recalculate s and dt using formulas
            new_s = (self.v0 + perturbed_v1) * self.dt0
            new_dt = (self.v0 + perturbed_v1) * self.dt0 / (self.v0 - self.v2)
            
            # Check trends
            if perturbed_v1 > self.v1:
                if new_s <= self.s:
                    self.log(f"FAIL monotonicity s: v1↑ from {self.v1} to {perturbed_v1} but s {self.s}→{new_s}")
                    return False
                if new_dt <= self.dt:
                    self.log(f"FAIL monotonicity dt: v1↑ from {self.v1} to {perturbed_v1} but dt {self.dt}→{new_dt}")
                    return False
            else:
                # For completeness (though we only test increase)
                if new_s >= self.s:
                    self.log(f"FAIL monotonicity s: v1↓ from {self.v1} to {perturbed_v1} but s {self.s}→{new_s}")
                    return False
                if new_dt >= self.dt:
                    self.log(f"FAIL monotonicity dt: v1↓ from {self.v1} to {perturbed_v1} but dt {self.dt}→{new_dt}")
                    return False
            
            self.log("PASS monotonicity")
            return True
            
        except Exception as e:
            self.log(f"ERROR in monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all checks and return success status with logs"""
        self.logs = []  # Reset logs
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            if not test_func():
                success = False
                self.log(f"{test_name} failed")
            else:
                self.log(f"{test_name} passed")
        
        return success, self.logs