import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with fallback to problem constants
        self.L1 = getattr(solution_instance, 'L1', 4.0)
        self.L2 = getattr(solution_instance, 'L2', 3.6)
        self.v = getattr(solution_instance, 'v', 1.2)
        self.D1 = getattr(solution_instance, 'D1', 4.0)
        self.L3 = getattr(solution_instance, 'L3', 40.0)
        self.v1 = getattr(solution_instance, 'v1', 12.0)
        self.D2 = getattr(solution_instance, 'D2', 6.0)
        self.v2 = getattr(solution_instance, 'v2', 10.0)
        self.d1 = getattr(solution_instance, 'd1', 2.5)

        # Extract results safely from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.d2 = results.get('d_2', 0.0)
        self.a1 = results.get('a_1', 0.0)
        self.x3 = results.get('x_3', 0.0)
        self.v2_double_prime = results.get("v_2''", 0.0)   # note key with double quotes
        self.x4 = results.get('x_4', 0.0)
        self.t2 = results.get('t_2', 0.0)
        self.a = results.get('a', 0.0)                     # may be a range string
        self.a_max = results.get('a_max', 0.0)
        self.t3 = results.get('t_3', 0.0)
        self.x6 = results.get('x_6', 0.0)
        self.a2 = results.get('a_2', 0.0)
        self.v2_prime = results.get("v_2'", 0.0)           # note key with single quote
        self.a_min = results.get('a_min', 0.0)
        self.t1 = results.get('t_1', 0.0)
        self.x5 = results.get('x_5', 0.0)
        self.x1 = results.get('x_1', 0.0)

    def is_close(self, a, b):
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    # Step-wise verification methods
    def test_step1(self):
        expected_t1 = self.L2 / self.v
        if self.is_close(self.t1, expected_t1):
            return True, f"Step 1: t1 correct: {self.t1} ≈ {expected_t1}"
        return False, f"Step 1: t1 incorrect: got {self.t1}, expected {expected_t1}"

    def test_step2(self):
        expected_x1 = self.L3 + self.L1 + self.D1
        if self.is_close(self.x1, expected_x1):
            return True, f"Step 2: x1 correct: {self.x1} ≈ {expected_x1}"
        return False, f"Step 2: x1 incorrect: got {self.x1}, expected {expected_x1}"

    def test_step3(self):
        if self.t1 == 0:
            return False, "Step 3: t1 is zero, cannot compute a1"
        expected_a1 = 2 * (self.x1 - self.v1 * self.t1) / (self.t1 ** 2)
        if self.is_close(self.a1, expected_a1):
            return True, f"Step 3: a1 correct: {self.a1} ≈ {expected_a1}"
        return False, f"Step 3: a1 incorrect: got {self.a1}, expected {expected_a1}"

    def test_step4(self):
        dist_truck = self.L3 + self.L1 + self.D2
        if self.t1 == 0:
            return False, "Step 4: t1 is zero"
        expected_a2 = 2 * (dist_truck - self.v2 * self.t1) / (self.t1 ** 2)
        if self.is_close(self.a2, expected_a2):
            return True, f"Step 4: a2 correct: {self.a2} ≈ {expected_a2}"
        return False, f"Step 4: a2 incorrect: got {self.a2}, expected {expected_a2}"

    def test_step5(self):
        expected_v2_prime = self.v2 + self.a2 * self.t1
        if self.is_close(self.v2_prime, expected_v2_prime):
            return True, f"Step 5: v2' correct: {self.v2_prime} ≈ {expected_v2_prime}"
        return False, f"Step 5: v2' incorrect: got {self.v2_prime}, expected {expected_v2_prime}"

    def test_step6(self):
        # Consistency check between step 6 and step 7
        if self.t2 == 0:
            return False, "Step 6: t2 is zero"
        a_cargo = (self.v2_prime - self.v2) / self.t2
        expected_x3 = (self.v2 + self.v2_prime) / 2 * self.t2
        if self.is_close(self.x3, expected_x3):
            return True, f"Step 6-7: x3 consistent: {self.x3} ≈ {expected_x3}, a_cargo={a_cargo:.2f}"
        return False, f"Step 6-7: x3 inconsistent: got {self.x3}, expected {expected_x3}"

    def test_step8(self):
        expected_x4 = (self.L3 + self.L1 + self.D2) + self.v2_prime * (self.t2 - self.t1)
        if self.is_close(self.x4, expected_x4):
            return True, f"Step 8: x4 correct: {self.x4} ≈ {expected_x4}"
        return False, f"Step 8: x4 incorrect: got {self.x4}, expected {expected_x4}"

    def test_step9(self):
        expected_d2 = self.x4 - self.x3
        if self.is_close(self.d2, expected_d2):
            return True, f"Step 9: d2 correct: {self.d2} ≈ {expected_d2}"
        return False, f"Step 9: d2 incorrect: got {self.d2}, expected {expected_d2}"

    def test_step10(self):
        expected_t3 = 3 * self.L2 / self.v
        if self.is_close(self.t3, expected_t3):
            return True, f"Step 10: t3 correct: {self.t3} ≈ {expected_t3}"
        return False, f"Step 10: t3 incorrect: got {self.t3}, expected {expected_t3}"

    def test_step11(self):
        expected_v2_double_prime = 2 * self.L3 / self.t3 - self.v2
        if self.is_close(self.v2_double_prime, expected_v2_double_prime):
            return True, f"Step 11: v2'' correct: {self.v2_double_prime} ≈ {expected_v2_double_prime}"
        return False, f"Step 11: v2'' incorrect: got {self.v2_double_prime}, expected {expected_v2_double_prime}"

    def test_step12(self):
        expected_a_min = self.v2 ** 2 / (2 * self.L3)
        if self.is_close(self.a_min, expected_a_min):
            return True, f"Step 12: a_min correct: {self.a_min} ≈ {expected_a_min}"
        return False, f"Step 12: a_min incorrect: got {self.a_min}, expected {expected_a_min}"

    def test_step14(self):
        # Verify a_max is positive and greater than a_min
        if not self.a_max > self.a_min:
            return False, f"Step 14: a_max <= a_min: {self.a_max} <= {self.a_min}"
        # Check typical expected value (5.0) with tolerance
        expected_a_max = 5.0
        if self.is_close(self.a_max, expected_a_max):
            return True, f"Step 14: a_max is {self.a_max} ≈ {expected_a_max}"
        return False, f"Step 14: a_max unexpected: got {self.a_max}, expected {expected_a_max}"

    def test_range(self):
        if self.a_min <= self.a_max:
            return True, f"Range: a_min={self.a_min}, a_max={self.a_max}, valid"
        return False, f"Range: a_min > a_max: {self.a_min} > {self.a_max}"

    def test_monotonicity(self):
        original_v = self.v
        perturbed_v = original_v * 1.1   # increase by 10%
        new_t1 = self.L2 / perturbed_v
        new_a1 = 2 * (self.x1 - self.v1 * new_t1) / (new_t1 ** 2)
        if new_a1 > self.a1:
            return True, f"Monotonicity: when v increases from {original_v} to {perturbed_v:.2f}, a1 increases from {self.a1:.2f} to {new_a1:.2f}"
        return False, f"Monotonicity: when v increases, a1 does not increase: {self.a1:.2f} -> {new_a1:.2f}"

    def test_invariants(self):
        # Check kinematic relation v_f^2 = v_i^2 + 2 a s for truck acceleration
        dist = self.L3 + self.L1 + self.D2
        v2_sq_expected = self.v2 ** 2 + 2 * self.a2 * dist
        if self.is_close(self.v2_prime ** 2, v2_sq_expected):
            return True, f"Energy invariant: v2'^2 = v2^2 + 2*a2*dist holds"
        return False, f"Energy invariant: v2'^2 = {self.v2_prime**2}, expected {v2_sq_expected}"

    def test_trace_consistency(self):
        tests = [self.test_step1, self.test_step2, self.test_step3, self.test_step4,
                 self.test_step5, self.test_step6, self.test_step8, self.test_step9,
                 self.test_step10, self.test_step11, self.test_step12, self.test_step14]
        logs = []
        all_ok = True
        for test in tests:
            try:
                ok, msg = test()
                logs.append(msg)
                if not ok:
                    all_ok = False
            except Exception as e:
                logs.append(f"Error in {test.__name__}: {e}")
                all_ok = False
        return all_ok, logs

    def run_all_tests(self):
        logs = []
        success = True

        # Trace consistency
        ok, msgs = self.test_trace_consistency()
        logs.extend(msgs)
        if not ok:
            success = False

        # Physical invariants
        try:
            ok, msg = self.test_invariants()
            logs.append(msg)
            if not ok:
                success = False
        except Exception as e:
            logs.append(f"Invariants test error: {e}")
            success = False

        # Monotonicity
        try:
            ok, msg = self.test_monotonicity()
            logs.append(msg)
            if not ok:
                success = False
        except Exception as e:
            logs.append(f"Monotonicity test error: {e}")
            success = False

        # Range validity
        try:
            ok, msg = self.test_range()
            logs.append(msg)
            if not ok:
                success = False
        except Exception as e:
            logs.append(f"Range test error: {e}")
            success = False

        return success, logs