import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Uses fallback defaults from the problem steps.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Given constant (distance at braking start)
        self.L = 1.5  # meters

        # Extract with safe fallbacks (using step values as defaults)
        self.v1_0 = self._get_value('v_1', 4.0)
        self.a1_mag = self._get_value('a_1', 0.5)
        self.v2_0 = self._get_value('v_2', 3.0)
        self.a2_mag = self._get_value('a_2', 0.25)
        self.t_collision = self._get_value('t', 2.0)
        self.t_equal = self._get_value('t\'', 4.0)  # time when speeds equal
        self.d = self._get_value('d', 2.0)

        # Optional intermediate results (for trace consistency)
        self.s1_collision = self._get_value('s_1', None)
        self.s2_collision = self._get_value('s_2', None)
        self.s1_prime = self._get_value('s_1\'', None)
        self.s2_prime = self._get_value('s_2\'', None)

        # Signed accelerations (deceleration is negative)
        self.a1 = -self.a1_mag
        self.a2 = -self.a2_mag

        # Symbol for time
        self.t_sym = sp.symbols('t')

    def _get_value(self, key, default):
        """
        Safely retrieve a value from solution attributes or results dict.
        Returns float if numeric, else default.
        """
        val = getattr(self.solution, key, self.results.get(key, default))
        if isinstance(val, (int, float)):
            return float(val)
        return default

    # Kinematic helpers
    def v1(self, t):
        return self.v1_0 + self.a1 * t

    def v2(self, t):
        return self.v2_0 + self.a2 * t

    def s1(self, t):
        return self.v1_0 * t + 0.5 * self.a1 * t**2

    def s2(self, t):
        return self.v2_0 * t + 0.5 * self.a2 * t**2

    def test_invariants(self):
        """
        Verify fundamental physics laws: collision condition, velocity equality,
        and that the chosen collision time is the physically meaningful root.
        """
        logs = []
        success = True

        # 1. Collision condition: s1(t) - s2(t) = L
        s1_coll = self.s1(self.t_collision)
        s2_coll = self.s2(self.t_collision)
        diff_coll = s1_coll - s2_coll
        if not math.isclose(diff_coll, self.L, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: At t={self.t_collision}, s1-s2 = {diff_coll:.3f} ≠ L={self.L}")
            success = False
        else:
            logs.append(f"PASS: Collision condition: s1-s2 = {diff_coll:.3f} ≈ L")

        # 2. Velocities at collision should be positive (still moving forward)
        v1_coll = self.v1(self.t_collision)
        v2_coll = self.v2(self.t_collision)
        if v1_coll <= 0:
            logs.append(f"WARNING: Motorcycle velocity at collision is {v1_coll:.3f} ≤ 0")
        if v2_coll <= 0:
            logs.append(f"WARNING: Car velocity at collision is {v2_coll:.3f} ≤ 0")

        # 3. Collision time is the smallest positive root of s1(t)-s2(t)=L
        A = 0.5 * (self.a1 - self.a2)
        B = self.v1_0 - self.v2_0
        C = -self.L
        t = sp.symbols('t')
        eq = sp.Eq(A * t**2 + B * t + C, 0)
        roots = sp.solve(eq, t)
        real_roots = [float(r) for r in roots if r.is_real and float(r) >= 0]
        real_roots.sort()
        if len(real_roots) >= 2:
            if not math.isclose(self.t_collision, real_roots[0], rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Collision time {self.t_collision} ≠ smallest root {real_roots[0]}")
                success = False
            else:
                logs.append(f"PASS: Collision time matches smallest positive root {real_roots[0]}")
        else:
            logs.append(f"INFO: Found {len(real_roots)} positive root(s): {real_roots}")

        # 4. At t_equal, velocities must be equal
        v1_eq = self.v1(self.t_equal)
        v2_eq = self.v2(self.t_equal)
        if not math.isclose(v1_eq, v2_eq, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: At t'={self.t_equal}, v1={v1_eq:.3f} ≠ v2={v2_eq:.3f}")
            success = False
        else:
            logs.append(f"PASS: At t'={self.t_equal}, v1 = v2 = {v1_eq:.3f}")

        # 5. Extra distance d = s1(t_equal) - s2(t_equal)
        s1_eq = self.s1(self.t_equal)
        s2_eq = self.s2(self.t_equal)
        d_computed = s1_eq - s2_eq
        if not math.isclose(d_computed, self.d, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: d computed = {d_computed:.3f} ≠ given d={self.d}")
            success = False
        else:
            logs.append(f"PASS: d = s1'-s2' = {d_computed:.3f} ≈ given d")

        if d_computed <= 0:
            logs.append(f"WARNING: Extra distance d is non-positive: {d_computed:.3f}")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results (if provided) with final answers.
        """
        logs = []
        success = True

        # Check s1 and s2 at collision
        if self.s1_collision is not None:
            s1_comp = self.s1(self.t_collision)
            if not math.isclose(self.s1_collision, s1_comp, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENCY: s1 from results {self.s1_collision} ≠ computed {s1_comp:.3f}")
                success = False
        if self.s2_collision is not None:
            s2_comp = self.s2(self.t_collision)
            if not math.isclose(self.s2_collision, s2_comp, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENCY: s2 from results {self.s2_collision} ≠ computed {s2_comp:.3f}")
                success = False

        # Check s1' and s2' at t_equal
        if self.s1_prime is not None:
            s1_prime_comp = self.s1(self.t_equal)
            if not math.isclose(self.s1_prime, s1_prime_comp, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENCY: s1' from results {self.s1_prime} ≠ computed {s1_prime_comp:.3f}")
                success = False
        if self.s2_prime is not None:
            s2_prime_comp = self.s2(self.t_equal)
            if not math.isclose(self.s2_prime, s2_prime_comp, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENCY: s2' from results {self.s2_prime} ≠ computed {s2_prime_comp:.3f}")
                success = False

        if success:
            logs.append("PASS: Trace consistency checks passed (or no data to check).")
        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (initial velocity v1_0 and distance L) and verify
        that collision time changes in the expected direction.
        """
        logs = []
        success = True

        # Perturb v1_0: increasing speed should decrease collision time
        original_v1_0 = self.v1_0
        perturbed_v1_0 = original_v1_0 * 1.1
        A = 0.5 * (self.a1 - self.a2)
        B_pert = perturbed_v1_0 - self.v2_0
        C = -self.L
        t = sp.symbols('t')
        eq1 = sp.Eq(A * t**2 + B_pert * t + C, 0)
        roots1 = sp.solve(eq1, t)
        real_roots1 = [float(r) for r in roots1 if r.is_real and float(r) >= 0]
        if real_roots1:
            new_t1 = min(real_roots1)
            if new_t1 >= self.t_collision:
                logs.append(f"FAIL: Increasing v1_0 did not decrease t. Original {self.t_collision}, new {new_t1}")
                success = False
            else:
                logs.append(f"PASS: v1_0↑ → t↓ ({self.t_collision} → {new_t1})")
        else:
            logs.append("WARNING: No positive root for perturbed v1_0.")
            success = False

        # Perturb L: increasing distance should increase collision time
        perturbed_L = self.L * 1.1
        B_orig = original_v1_0 - self.v2_0
        C_pert = -perturbed_L
        eq2 = sp.Eq(A * t**2 + B_orig * t + C_pert, 0)
        roots2 = sp.solve(eq2, t)
        real_roots2 = [float(r) for r in roots2 if r.is_real and float(r) >= 0]
        if real_roots2:
            new_t2 = min(real_roots2)
            if new_t2 <= self.t_collision:
                logs.append(f"FAIL: Increasing L did not increase t. Original {self.t_collision}, new {new_t2}")
                success = False
            else:
                logs.append(f"PASS: L↑ → t↑ ({self.t_collision} → {new_t2})")
        else:
            logs.append("WARNING: No positive root for perturbed L.")
            success = False

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test in tests:
            all_logs.append(f"\n=== {name} ===")
            try:
                test_success, logs = test()
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs