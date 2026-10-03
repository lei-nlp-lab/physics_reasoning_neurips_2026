import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract results from the solution instance.
        Uses getattr and .get() to avoid AttributeError in sandbox.
        """
        self.T_OA = getattr(solution_instance, 'T_OA', solution_instance.results.get('T_OA', 0.0))
        self.T_OB = getattr(solution_instance, 'T_OB', solution_instance.results.get('T_OB', 0.0))
        self.G = getattr(solution_instance, 'G', solution_instance.results.get('G', 0.0))  # maximum weight

        self.angle_OA_deg = 45
        self.angle_OA_rad = math.radians(self.angle_OA_deg)

    def test_invariants(self):
        logs = []
        success = True

        # 1. Equilibrium of forces at point O (first part, assuming weight = 3 N)
        # Horizontal: T_OA * sin(45) = T_OB
        T_OA_horiz = self.T_OA * math.sin(self.angle_OA_rad)
        if not math.isclose(T_OA_horiz, self.T_OB, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Horizontal equilibrium fails: T_OA*sin45 = {T_OA_horiz} ≠ T_OB = {self.T_OB}")
            success = False

        # Vertical: T_OA * cos(45) = weight (computed from T_OA)
        weight_from_T_OA = self.T_OA * math.cos(self.angle_OA_rad)
        # Since tan45=1, weight should also equal T_OB
        if not math.isclose(weight_from_T_OA, self.T_OB, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Vertical equilibrium fails: T_OA*cos45 = {weight_from_T_OA} ≠ T_OB = {self.T_OB}")
            success = False

        # 2. Check against target values for first part: T_OA=3√2 ≈4.2426, T_OB=3
        T_OA_target = 3 * math.sqrt(2)
        if not math.isclose(self.T_OA, T_OA_target, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"T_OA = {self.T_OA} not close to target 3√2 ≈ {T_OA_target}")
            success = False

        if not math.isclose(self.T_OB, 3.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"T_OB = {self.T_OB} not close to target 3")
            success = False

        # 3. Check maximum weight (second part)
        if not math.isclose(self.G, 6.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Maximum weight G = {self.G} not close to target 6")
            success = False

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Step 1: For weight = 3, T_OB = 3, T_OA = 3√2 (already checked in invariants)

        # Step 2: If OB reaches max tension 6 N, then T_OA = 6 / cos45 = 6√2 ≈ 8.485
        T_OA_if_OB_max = 6 / math.cos(self.angle_OA_rad)
        if T_OA_if_OB_max >= 10.0:
            logs.append(f"If OB is at max 6 N, T_OA = {T_OA_if_OB_max} ≥ 10 N (unsafe for OA)")
            success = False

        # Step 3: Max tension in OB is 6 N (given)

        # Step 4: Maximum gravitational force G_max = T_OB_max * tan45 = 6
        G_max_from_OB = 6 * math.tan(self.angle_OA_rad)  # tan45=1
        if not math.isclose(self.G, G_max_from_OB, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"G_max from OB limit = {G_max_from_OB} ≠ computed G_max = {self.G}")
            success = False

        # Cross-check: At G_max, T_OA = G_max / sin45 = 6√2 ≈ 8.485 < 10
        T_OA_at_max = self.G / math.sin(self.angle_OA_rad)
        if T_OA_at_max >= 10.0:
            logs.append(f"At G_max, T_OA = {T_OA_at_max} ≥ 10 N (exceeds OA limit)")
            success = False

        # Ensure T_OA_at_max matches 6√2
        if not math.isclose(T_OA_at_max, 6 * math.sqrt(2), rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"At G_max, T_OA = {T_OA_at_max} ≠ 6√2 ≈ {6*math.sqrt(2)}")
            success = False

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Compute the weight for the first part from extracted T_OA
        weight_first = self.T_OA * math.cos(self.angle_OA_rad)

        # Perturb weight upward by 1%
        weight_perturbed = weight_first * 1.01
        T_OA_perturbed = weight_perturbed / math.sin(self.angle_OA_rad)
        T_OB_perturbed = weight_perturbed  # since tan45=1

        # Check monotonic increase
        if not (T_OA_perturbed > self.T_OA):
            logs.append(f"With 1% weight increase, T_OA does not increase: {T_OA_perturbed} ≤ {self.T_OA}")
            success = False
        if not (T_OB_perturbed > self.T_OB):
            logs.append(f"With 1% weight increase, T_OB does not increase: {T_OB_perturbed} ≤ {self.T_OB}")
            success = False

        # Check that at G_max, tensions are at safe limits
        T_OA_at_max = self.G / math.sin(self.angle_OA_rad)
        T_OB_at_max = self.G  # tan45=1

        if T_OA_at_max >= 10.0:
            logs.append(f"At G_max, T_OA = {T_OA_at_max} ≥ 10 N (should be < 10)")
            success = False
        if T_OB_at_max > 6.0:
            logs.append(f"At G_max, T_OB = {T_OB_at_max} > 6 N (exceeds OB limit)")
            success = False

        # Boundary check: if weight slightly exceeds G_max, T_OB > 6
        weight_above = self.G * 1.01
        T_OB_above = weight_above
        if T_OB_above <= 6.0:
            logs.append(f"With weight 1% above G_max, T_OB = {T_OB_above} ≤ 6 (should exceed OB limit)")
            success = False

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ('invariants', self.test_invariants),
            ('trace_consistency', self.test_trace_consistency),
            ('monotonicity', self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                if success:
                    all_logs.append(f"{test_name}: PASSED")
                else:
                    overall_success = False
                    for log in logs:
                        all_logs.append(f"{test_name}: {log}")
            except Exception as e:
                overall_success = False
                all_logs.append(f"{test_name}: ERROR - {str(e)}")

        return overall_success, all_logs