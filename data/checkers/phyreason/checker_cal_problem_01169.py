import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with safe defaults
        self.v1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 20.0))  # m/s
        self.v2 = getattr(solution_instance, 'v_2', self.results.get('v_2', 5.0))   # m/s
        self.x = 60.0  # m (distance from speed bump to toll island entrance)
        self.d = 40.0  # m (total length of toll island)
        self.t0 = 20.0 # s (payment time for car B)
        
        # Extract calculated results with safe defaults
        self.a = self.results.get('a', 0.0)
        self.t1 = self.results.get('t_1', 0.0)
        self.x1 = self.results.get('x_1', 0.0)
        self.t2 = self.results.get('t_2', 0.0)
        self.tA = self.results.get('t_A', 0.0)
        self.t3 = self.results.get('t_3', 0.0)
        self.tB = self.results.get('t_B', 0.0)
        self.Deltat = self.results.get('Deltat', 0.0)
        self.t4 = self.results.get('t_4', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.t5 = self.results.get('t_5', 0.0)
        self.x3 = self.results.get('x_3', 0.0)
        self.t_constant = self.results.get('t', 0.0)  # Using t_constant to avoid conflict with time variable
        self.x4 = self.results.get('x_4', 0.0)
        self.Deltax = self.results.get('Deltax', 0.0)
        
        # Store original values for perturbation tests
        self._original_v1 = self.v1
        self._original_results = self.results.copy()

    def test_invariants(self):
        logs = []
        
        # Check 1: Positive physical quantities
        if self.v1 <= 0 or self.v2 <= 0:
            logs.append("Invariant violated: velocities must be positive.")
        if self.a <= 0:
            logs.append("Invariant violated: acceleration magnitude must be positive.")
        if any(val < 0 for val in [self.t1, self.t2, self.tA, self.t3, self.tB, self.t4, self.t5, self.t_constant]):
            logs.append("Invariant violated: negative time encountered.")
        if any(val < 0 for val in [self.x1, self.x2, self.x3, self.x4, self.Deltax]):
            logs.append("Invariant violated: negative distance encountered.")
        
        # Check 2: Kinematic consistency for car B deceleration (from step_1)
        # v1^2 = 2*a*(x + d/2)
        expected_a = self.v1**2 / (2 * (self.x + self.d/2))
        if not math.isclose(self.a, expected_a, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: a = {self.a:.3f}, expected {expected_a:.3f} from v1^2 = 2a(x+d/2)")
        
        # Check 3: Time-distance consistency for car A deceleration (step_2 and step_3)
        # t1 = (v1 - v2)/a
        expected_t1 = (self.v1 - self.v2) / self.a
        if not math.isclose(self.t1, expected_t1, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: t1 = {self.t1:.3f}, expected {expected_t1:.3f} from (v1-v2)/a")
        
        # x1 = (v1 + v2)/2 * t1
        expected_x1 = (self.v1 + self.v2) / 2 * self.t1
        if not math.isclose(self.x1, expected_x1, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: x1 = {self.x1:.3f}, expected {expected_x1:.3f} from (v1+v2)/2 * t1")
        
        # Check 4: Car A constant velocity phase (step_4)
        # t2 = (x + d/2 - x1) / v2
        expected_t2 = (self.x + self.d/2 - self.x1) / self.v2
        if not math.isclose(self.t2, expected_t2, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: t2 = {self.t2:.3f}, expected {expected_t2:.3f} from (80 - x1)/v2")
        
        # Check 5: Time addition consistency (step_5 and step_7)
        expected_tA = self.t1 + self.t2
        if not math.isclose(self.tA, expected_tA, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: tA = {self.tA:.3f}, expected {expected_tA:.3f} from t1 + t2")
        
        expected_tB = self.t3 + self.t0
        if not math.isclose(self.tB, expected_tB, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: tB = {self.tB:.3f}, expected {expected_tB:.3f} from t3 + t0")
        
        # Check 6: Time difference (step_8)
        expected_Deltat = self.tB - self.tA
        if not math.isclose(self.Deltat, expected_Deltat, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: Δt = {self.Deltat:.3f}, expected {expected_Deltat:.3f} from tB - tA")
        
        # Check 7: Symmetry in acceleration/deceleration times (step_9 and step_11)
        # t4 should equal t3 (both = v1/a)
        expected_t4 = self.v1 / self.a
        if not math.isclose(self.t4, expected_t4, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: t4 = {self.t4:.3f}, expected {expected_t4:.3f} from v1/a")
        
        # t5 should equal t1 (both = (v1 - v2)/a)
        expected_t5 = (self.v1 - self.v2) / self.a
        if not math.isclose(self.t5, expected_t5, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: t5 = {self.t5:.3f}, expected {expected_t5:.3f} from (v1-v2)/a")
        
        # Check 8: Distance consistency (step_10 and step_12)
        # x2 = 0.5*a*t3^2
        expected_x2 = 0.5 * self.a * self.t3**2
        if not math.isclose(self.x2, expected_x2, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: x2 = {self.x2:.3f}, expected {expected_x2:.3f} from 0.5*a*t3^2")
        
        # x3 = (v1^2 - v2^2)/(2*a)
        expected_x3 = (self.v1**2 - self.v2**2) / (2 * self.a)
        if not math.isclose(self.x3, expected_x3, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: x3 = {self.x3:.3f}, expected {expected_x3:.3f} from (v1^2-v2^2)/(2a)")
        
        # Check 9: Constant velocity phase for distance calculation (step_13 and step_14)
        # t = Δt + t4 - t5
        expected_t_constant = self.Deltat + self.t4 - self.t5
        if not math.isclose(self.t_constant, expected_t_constant, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: t = {self.t_constant:.3f}, expected {expected_t_constant:.3f} from Δt + t4 - t5")
        
        # x4 = v1 * t
        expected_x4 = self.v1 * self.t_constant
        if not math.isclose(self.x4, expected_x4, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: x4 = {self.x4:.3f}, expected {expected_x4:.3f} from v1 * t")
        
        # Check 10: Maximum distance calculation (step_15)
        expected_Deltax = self.x3 + self.x4 - self.x2
        if not math.isclose(self.Deltax, expected_Deltax, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Invariant violated: Δx = {self.Deltax:.3f}, expected {expected_Deltax:.3f} from x3 + x4 - x2")
        
        return len(logs) == 0, logs

    def test_trace_consistency(self):
        logs = []
        
        # Check final answers against target values
        if not math.isclose(self.Deltat, 21.0, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Trace consistency: Δt = {self.Deltat:.3f} s does not match target 21 s")
        
        if not math.isclose(self.Deltax, 455.0, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Trace consistency: Δx = {self.Deltax:.3f} m does not match target 455 m")
        
        # Check intermediate consistency: all steps should lead to final answers
        # Recalculate Δt from basic principles
        recalc_tA = self.t1 + self.t2
        recalc_tB = self.t3 + self.t0
        recalc_Deltat = recalc_tB - recalc_tA
        
        if not math.isclose(self.Deltat, recalc_Deltat, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Trace consistency: Δt doesn't match intermediate calculations")
        
        # Recalculate Δx from basic principles
        recalc_x3 = (self.v1**2 - self.v2**2) / (2 * self.a)
        recalc_t_constant = self.Deltat + self.t4 - self.t5
        recalc_x4 = self.v1 * recalc_t_constant
        recalc_x2 = 0.5 * self.a * self.t3**2
        recalc_Deltax = recalc_x3 + recalc_x4 - recalc_x2
        
        if not math.isclose(self.Deltax, recalc_Deltax, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Trace consistency: Δx doesn't match intermediate calculations")
        
        return len(logs) == 0, logs

    def test_monotonicity(self):
        logs = []
        
        # Perturb initial velocity v1 by +1%
        perturbed_v1 = self.v1 * 1.01
        
        # Recalculate all quantities using the kinematic formulas
        # Step 1: Recalculate acceleration
        perturbed_a = perturbed_v1**2 / (2 * (self.x + self.d/2))
        
        # Step 2-5: Recalculate car A times
        perturbed_t1 = (perturbed_v1 - self.v2) / perturbed_a
        perturbed_x1 = (perturbed_v1 + self.v2) / 2 * perturbed_t1
        perturbed_t2 = (self.x + self.d/2 - perturbed_x1) / self.v2
        perturbed_tA = perturbed_t1 + perturbed_t2
        
        # Step 6-8: Recalculate car B times and Δt
        perturbed_t3 = perturbed_v1 / perturbed_a
        perturbed_tB = perturbed_t3 + self.t0
        perturbed_Deltat = perturbed_tB - perturbed_tA
        
        # Step 9-15: Recalculate distances and Δx
        perturbed_t4 = perturbed_v1 / perturbed_a
        perturbed_x2 = 0.5 * perturbed_a * perturbed_t3**2
        perturbed_t5 = (perturbed_v1 - self.v2) / perturbed_a
        perturbed_x3 = (perturbed_v1**2 - self.v2**2) / (2 * perturbed_a)
        perturbed_t_constant = perturbed_Deltat + perturbed_t4 - perturbed_t5
        perturbed_x4 = perturbed_v1 * perturbed_t_constant
        perturbed_Deltax = perturbed_x3 + perturbed_x4 - perturbed_x2
        
        # Check monotonic trends:
        # When v1 increases, Δt should decrease (car A benefits more from higher initial speed)
        if perturbed_Deltat > self.Deltat:
            logs.append("Monotonicity violated: Δt increased when v1 increased (expected decrease)")
        
        # When v1 increases, Δx should increase (car A gets further ahead)
        if perturbed_Deltax < self.Deltax:
            logs.append("Monotonicity violated: Δx decreased when v1 increased (expected increase)")
        
        # Check that changes are physically reasonable (not too large)
        rel_change_Deltat = abs(perturbed_Deltat - self.Deltat) / abs(self.Deltat)
        rel_change_Deltax = abs(perturbed_Deltax - self.Deltax) / abs(self.Deltax)
        
        # Relative changes should be roughly proportional to the 1% perturbation
        if rel_change_Deltat > 0.05:  # Allow 5% max change for 1% perturbation
            logs.append(f"Monotonicity: Δt change {rel_change_Deltat:.3%} too large for 1% v1 change")
        
        if rel_change_Deltax > 0.05:
            logs.append(f"Monotonicity: Δx change {rel_change_Deltax:.3%} too large for 1% v1 change")
        
        # Restore original v1 (not strictly necessary since we used a local variable)
        self.v1 = self._original_v1
        
        return len(logs) == 0, logs

    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            test_success, logs = test_func()
            if not test_success:
                success = False
                all_logs.append(f"=== {test_name} Test Failed ===")
                all_logs.extend(logs)
            else:
                all_logs.append(f"✓ {test_name} Test Passed")
        
        return success, all_logs