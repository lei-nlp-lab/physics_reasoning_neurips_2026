import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the free-fall object problem.
    Checks invariants, trace consistency, and monotonicity.
    """
    
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Inputs (with fallback to problem defaults)
        self.t = getattr(solution_instance, 't', 0.2)                # time to pass the floor
        self.h = getattr(solution_instance, 'h', 3.0)                # height of one floor
        self.g = getattr(solution_instance, 'g', 10.0)               # gravitational acceleration
        
        # Results (from results dictionary ONLY)
        results = getattr(solution_instance, 'results', {})
        self.Floor = results.get('Floor', None)
        self.x = results.get('x', None)
        self.v = results.get('v', None)
        self.delta_h = results.get('delta_h', None)
        
    def test_invariants(self):
        """
        Verify fundamental physical invariants:
        1. Kinematic equation: h = v*t + 0.5*g*t^2
        2. Velocity-displacement relation: v^2 = 2*g*delta_h
        3. Floor number consistency: Floor = ceil(5 + delta_h/h)
        """
        logs = []
        success = True
        
        # 1. Check h = v*t + 0.5*g*t^2
        if self.v is not None:
            lhs = self.h
            rhs = self.v * self.t + 0.5 * self.g * self.t**2
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant violated: h = v*t + 0.5*g*t^2 | {lhs} != {rhs}")
                success = False
        else:
            logs.append("Missing velocity (v) for invariant check.")
            success = False
        
        # 2. Check v^2 = 2*g*delta_h
        if self.v is not None and self.delta_h is not None:
            lhs = self.v**2
            rhs = 2 * self.g * self.delta_h
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant violated: v^2 = 2*g*delta_h | {lhs} != {rhs}")
                success = False
        else:
            logs.append("Missing v or delta_h for v^2 = 2*g*delta_h check.")
            success = False
        
        # 3. Check Floor = ceil(5 + delta_h/h)
        if self.Floor is not None and self.delta_h is not None:
            expected_floor = math.ceil(5 + self.delta_h / self.h)
            if not math.isclose(self.Floor, expected_floor, rel_tol=0, abs_tol=0.1):
                logs.append(f"Floor number mismatch: got {self.Floor}, expected {expected_floor}")
                success = False
        else:
            logs.append("Missing Floor or delta_h for floor number check.")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Verify that intermediate results are consistent with each other
        and with the final answer.
        """
        logs = []
        success = True
        
        # Recompute delta_h from v and compare
        if self.v is not None and self.delta_h is not None:
            computed_dh = self.v**2 / (2 * self.g)
            if not math.isclose(computed_dh, self.delta_h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace inconsistency: delta_h from v is {computed_dh}, but stored delta_h is {self.delta_h}")
                success = False
        
        # Recompute v from delta_h and compare
        if self.delta_h is not None and self.v is not None:
            computed_v = math.sqrt(2 * self.g * self.delta_h)
            if not math.isclose(computed_v, self.v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace inconsistency: v from delta_h is {computed_v}, but stored v is {self.v}")
                success = False
        
        # Check that x (if present) is approximately delta_h/h + 5
        if self.x is not None and self.delta_h is not None:
            expected_x = 5 + self.delta_h / self.h
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace inconsistency: x should be ~{expected_x}, but got {self.x}")
                success = False
        
        # Check that Floor is integer (allow float that is close to integer)
        if self.Floor is not None:
            if not (isinstance(self.Floor, int) or (isinstance(self.Floor, float) and math.isclose(self.Floor, round(self.Floor), rel_tol=0, abs_tol=0.1))):
                logs.append(f"Floor should be integer or close to integer, but got {self.Floor}")
                success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb the time t by +1% and verify that v and delta_h change
        in the expected direction (decrease).
        """
        logs = []
        success = True
        
        # Store original values
        original_t = self.t
        perturbed_t = original_t * 1.01  # increase by 1%
        
        # Recalculate v using the kinematic formula: v = (h - 0.5*g*t^2)/t
        v_new = (self.h - 0.5 * self.g * perturbed_t**2) / perturbed_t
        
        # Recalculate delta_h from the new v: delta_h = v^2/(2g)
        delta_h_new = v_new**2 / (2 * self.g)
        
        # Check monotonicity: as t increases, v should decrease
        if self.v is not None:
            if not (v_new < self.v or math.isclose(v_new, self.v, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Monotonicity violated: t increased but v did not decrease. v_old={self.v}, v_new={v_new}")
                success = False
        
        # Check monotonicity: as t increases, delta_h should decrease
        if self.delta_h is not None:
            if not (delta_h_new < self.delta_h or math.isclose(delta_h_new, self.delta_h, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Monotonicity violated: t increased but delta_h did not decrease. delta_h_old={self.delta_h}, delta_h_new={delta_h_new}")
                success = False
        
        # Additionally, verify that the recalculated values satisfy the kinematic equation for the perturbed t
        h_from_perturbed = v_new * perturbed_t + 0.5 * self.g * perturbed_t**2
        if not math.isclose(h_from_perturbed, self.h, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Perturbed values violate kinematics: h from perturbed t is {h_from_perturbed}, expected {self.h}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all checks and return overall success and log messages.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            success, logs = test_func()
            if not success:
                overall_success = False
            all_logs.append(f"--- {name} ---")
            if logs:
                all_logs.extend(logs)
            else:
                all_logs.append("All checks passed.")
        
        return overall_success, all_logs