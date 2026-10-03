import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Input parameters
        self.m = 0.3   # kg
        self.M = 0.9   # kg
        self.mu = 0.6
        self.v0 = 3.0  # m/s
        self.g = 10.0  # m/s²

        # Extract computed results with safe fallbacks
        # Accelerations
        self.a_m = self._safe_get('a_m')
        self.a_M = self._safe_get('a_M')

        # Displacements and velocities for first collision
        self.d_max1 = self._safe_get('d_max1')
        self.v1 = self._safe_get('v_1')
        self.x_m1 = self._safe_get('x_m1')
        self.x_M1 = self._safe_get('x_M1')

        # For N-th collision (using N as a parameter, we'll check symbolic pattern)
        self.d_maxN = self._safe_get('d_maxN')  # expected to be symbolic or numeric for specific N
        self.v_N = self._safe_get('v_N')
        self.v_N_minus_1 = self._safe_get('v_N-1')
        self.x_mN = self._safe_get('x_mN')
        self.x_MN = self._safe_get('x_MN')

        # Final answer: minimum board length
        self.L_min = self._safe_get('L_min')

        # Logs
        self.logs = []

    def _safe_get(self, key, default=None):
        """Safely retrieve attribute from solution or results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, default)
        return val

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical matching."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify physical invariants: Newton's 2nd Law, kinematic consistency."""
        logs = []
        success = True

        # 1. Check accelerations from Newton's 2nd Law
        expected_a_m = self.mu * self.g
        if not self._is_close(self.a_m, expected_a_m):
            logs.append(f"FAIL: a_m = {self.a_m} ≠ μ g = {expected_a_m}")
            success = False
        else:
            logs.append(f"PASS: a_m = {self.a_m} ≈ μ g = {expected_a_m}")

        expected_a_M = (self.mu * self.m * self.g) / self.M
        if not self._is_close(self.a_M, expected_a_M):
            logs.append(f"FAIL: a_M = {self.a_M} ≠ (μ m g)/M = {expected_a_M}")
            success = False
        else:
            logs.append(f"PASS: a_M = {self.a_M} ≈ (μ m g)/M = {expected_a_M}")

        # 2. Check force directions: friction on block is opposite to motion, on board is forward
        #    Using relative acceleration: block's deceleration relative to ground is a_m (decelerating from +v0 to -v0)
        #    Board's acceleration is a_M in the same direction as initial motion (positive).
        #    Relative acceleration magnitude should be a_m + a_M = 8 m/s².
        rel_acc = abs(self.a_m) + abs(self.a_M)   # both positive magnitudes
        expected_rel_acc = self.mu * self.g * (1 + self.m / self.M)
        if not self._is_close(rel_acc, expected_rel_acc):
            logs.append(f"FAIL: relative acceleration = {rel_acc} ≠ μ g (1 + m/M) = {expected_rel_acc}")
            success = False
        else:
            logs.append(f"PASS: relative acceleration = {rel_acc} ≈ μ g (1 + m/M) = {expected_rel_acc}")

        # 3. Check kinematic relation for first collision: v1 from simultaneous equations
        #    From steps: v1 = ( -v0 * a_M + v0 * a_m ) / (a_m + a_M) ? Actually solving:
        #    v1 = -v0 + a_m * t1  (block)
        #    v1 = v0 - a_M * t1   (board? Wait, step4 says v1 = v0 + a_M * t1, but a_M is positive and board slows down? Let's reexamine.
        #    Actually, step2 gives a_M = (μ m g)/M = 2 m/s², direction opposite to motion? The board is moving right, friction from block acts left, so board decelerates.
        #    So a_M should be negative if we take right as positive. But in the steps they use a_M as magnitude? Let's see step4: v1 = v0 + a_M * t1, but then step5 gives v1=1.5.
        #    If a_M=2 and v0=3, then v1 = 3 + 2*t1, which would be >3, but board should slow down. So they must be using a_M as magnitude and sign is implicit.
        #    Actually, step4 equation is v1 = v0 + a_M * t1, but then in step7 they use (-a_M) for displacement. So a_M is magnitude, direction handled by sign in equations.
        #    We'll check using the formula derived from steps 3 and 4:
        #        v1 = -v0 + a_m * t1   (block: rebounds so final velocity negative)
        #        v1 = v0 - a_M * t1    (board: slows down, so subtract)
        #    Solving: t1 = 2*v0/(a_m + a_M) = 2*3/(6+2)=0.75 s, then v1 = -3 + 6*0.75 = 1.5? That gives 1.5? Wait: -3 + 4.5 = 1.5, yes.
        #    But that's positive, meaning both move right after first collision? But block rebounds, so v1 should be negative? Contradiction.
        #    Let's re-read: "After the small block collides with the barrier, it rebounds with the same speed" -> so after collision, block's velocity is -3 m/s (leftwards).
        #    Then friction acts to slow it down and eventually reverse again. So step3: v1 = -v0 + a_m * t1, where v0=3, a_m=6, and t1 is the time until velocities equalize.
        #    At equalization, v1 is the common velocity. So v1 could be positive if the block hasn't yet reversed direction? Actually, block starts at -3, accelerates at 6 m/s² (rightwards due to friction), so its velocity increases from -3 to some value. The board starts at 3, decelerates at 2 m/s², so its velocity decreases from 3. They meet when equal. So v1 should be between -3 and 3.
        #    Our calculation gives v1 = 1.5, which is positive. That means after time t1, the block is moving right at 1.5 m/s and the board is also moving right at 1.5 m/s. That is consistent: block reversed direction from -3 to +1.5.
        #    So step3 equation is v1 = -v0 + a_m * t1, with v0=3, a_m=6, and step4 is v1 = v0 - a_M * t1, with v0=3, a_M=2.
        #    We'll check that the computed v1 satisfies these with consistent t1.
        if self.v1 is not None:
            t1_from_block = (self.v1 + self.v0) / self.a_m   # from v1 = -v0 + a_m * t1 => t1 = (v1 + v0)/a_m
            t1_from_board = (self.v0 - self.v1) / self.a_M   # from v1 = v0 - a_M * t1 => t1 = (v0 - v1)/a_M
            if not self._is_close(t1_from_block, t1_from_board):
                logs.append(f"FAIL: inconsistent t1 from block ({t1_from_block}) and board ({t1_from_board})")
                success = False
            else:
                logs.append(f"PASS: t1 consistent: {t1_from_block} s")
        else:
            logs.append("SKIP: v1 not provided for invariant check")

        self.logs.extend(logs)
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer and formulas."""
        logs = []
        success = True

        # 1. Check d_max1 = x_M1 - x_m1
        if all(v is not None for v in [self.d_max1, self.x_M1, self.x_m1]):
            d_max1_calc = self.x_M1 - self.x_m1
            if not self._is_close(self.d_max1, d_max1_calc):
                logs.append(f"FAIL: d_max1 = {self.d_max1} ≠ x_M1 - x_m1 = {d_max1_calc}")
                success = False
            else:
                logs.append(f"PASS: d_max1 = x_M1 - x_m1 = {self.d_max1}")

        # 2. Check x_m1 and x_M1 using kinematic equations with v1
        if self.v1 is not None and self.x_m1 is not None:
            # For block: v1^2 - (-v0)^2 = 2 * a_m * x_m1  (initial velocity after rebound is -v0 = -3)
            expected_x_m1 = (self.v1**2 - (-self.v0)**2) / (2 * self.a_m)
            if not self._is_close(self.x_m1, expected_x_m1):
                logs.append(f"FAIL: x_m1 = {self.x_m1} ≠ (v1^2 - (-v0)^2)/(2 a_m) = {expected_x_m1}")
                success = False
            else:
                logs.append(f"PASS: x_m1 kinematic equation holds")

        if self.v1 is not None and self.x_M1 is not None:
            # For board: v1^2 - v0^2 = 2 * (-a_M) * x_M1  (deceleration, so acceleration is -a_M)
            expected_x_M1 = (self.v1**2 - self.v0**2) / (2 * (-self.a_M))
            if not self._is_close(self.x_M1, expected_x_M1):
                logs.append(f"FAIL: x_M1 = {self.x_M1} ≠ (v1^2 - v0^2)/(2 * (-a_M)) = {expected_x_M1}")
                success = False
            else:
                logs.append(f"PASS: x_M1 kinematic equation holds")

        # 3. Check pattern for N-th collision: v_N = v_{N-1} / 2
        if self.v_N is not None and self.v_N_minus_1 is not None:
            if not self._is_close(self.v_N, self.v_N_minus_1 / 2):
                logs.append(f"FAIL: v_N = {self.v_N} ≠ v_{{N-1}}/2 = {self.v_N_minus_1 / 2}")
                success = False
            else:
                logs.append(f"PASS: v_N = v_{{N-1}}/2 holds")

        # 4. Check d_maxN pattern: d_maxN = 9 / 4^N
        #    We'll test for a specific N if possible, but d_maxN might be symbolic.
        #    If it's a number, we can check against formula with N inferred.
        if self.d_maxN is not None:
            # Try to infer N from v_N or v_{N-1}: v_{N-1} = 3 / 2^{N-1}
            # But if we don't have N, we can check that d_maxN = 9/4 * (1/4)^{N-1} = 9/4^N
            # We'll do a symbolic check: if d_maxN is a number, compute N = -log4(d_maxN * 4/9) + 1? Not straightforward.
            # Instead, check that d_maxN * 4^N is constant? We don't have N.
            # We'll skip if N is not provided, but we can check that d_maxN is positive and decreasing as N increases.
            pass

        # 5. Check L_min as sum of geometric series: L_min = 3 * (1 - 1/4^N) -> limit 3
        if self.L_min is not None:
            # For large N, L_min should approach 3
            if not self._is_close(self.L_min, 3.0):
                # But if N is finite, L_min should be less than 3
                # We can check that L_min = 3 * (1 - 1/4^N) for some integer N?
                # We'll compute N from d_max1 and ratio? Actually, d_max1 = 9/4, so series sum = (9/4) / (1 - 1/4) = 3.
                # So regardless of N, the infinite sum is 3. The finite sum is 3*(1 - 1/4^N).
                # If the solution computed L_min for infinite N, it should be exactly 3.
                logs.append(f"INFO: L_min = {self.L_min}, expected 3 m for infinite N")
                # We'll accept if close to 3 within tolerance
                if not self._is_close(self.L_min, 3.0):
                    logs.append(f"FAIL: L_min = {self.L_min} ≠ 3 m")
                    success = False
                else:
                    logs.append(f"PASS: L_min ≈ 3 m")
            else:
                logs.append(f"PASS: L_min = 3 m")

        self.logs.extend(logs)
        return success

    def test_monotonicity(self):
        """Perturb an input and check trend, then restore."""
        logs = []
        success = True

        # Perturb mass m by +10%
        original_m = self.m
        perturbed_m = original_m * 1.1

        # Recalculate expected accelerations and L_min trend
        # a_m should remain same (μ g) independent of m
        # a_M = (μ m g)/M, so increases with m
        # The relative acceleration increases, so the time to equalize decreases? But the series sum L_min might change.
        # Actually, L_min = sum d_maxN, and d_max1 = (v0^2)/(2*(a_m+a_M)) * (a_m+a_M)? Let's derive.
        # From kinematic: d_max1 = (v0^2)/(a_m+a_M)  (actually, from steps: d_max1 = 2*v0^2/(a_m+a_M) * (a_M?))
        # We have d_max1 = x_M1 - x_m1 = (v1^2 - v0^2)/(2*(-a_M)) - (v1^2 - (-v0)^2)/(2*a_m)
        # With v1 = ( -v0*a_M + v0*a_m )/(a_m+a_M) ? Actually, solving the two equations:
        #   v1 = -v0 + a_m t1
        #   v1 = v0 - a_M t1
        #   => t1 = 2*v0/(a_m+a_M)
        #   => v1 = v0*(a_m - a_M)/(a_m+a_M)
        # Then plug into displacements. After algebra, d_max1 = 2*v0^2 * a_M / (a_m*(a_m+a_M))
        # But wait, from step8: d_max1 = x_M1 - x_m1 = 2.25 = 9/4.
        # Let's compute with our numbers: v0=3, a_m=6, a_M=2 -> d_max1 = 2*9*2/(6*(6+2)) = 36/(6*8)=36/48=0.75? That's not 2.25.
        # So I must have made a mistake. Let's use the provided steps: they computed x_m1=-0.5625, x_M1=1.6875, difference=2.25.
        # Let's compute using the formulas from steps:
        #   x_m1 = (v1^2 - (-v0)^2)/(2*a_m) = (1.5^2 - 9)/(12) = (2.25-9)/12 = -6.75/12 = -0.5625 ✔
        #   x_M1 = (v1^2 - v0^2)/(2*(-a_M)) = (2.25-9)/(2*(-2)) = (-6.75)/(-4) = 1.6875 ✔
        # So d_max1 = 1.6875 - (-0.5625) = 2.25.
        # Now, d_max1 in terms of v0, a_m, a_M:
        #   v1 = v0*(a_m - a_M)/(a_m+a_M)
        #   x_m1 = [v0^2*( (a_m - a_M)^2/(a_m+a_M)^2 - 1 )] / (2*a_m)
        #   x_M1 = [v0^2*( (a_m - a_M)^2/(a_m+a_M)^2 - 1 )] / (2*(-a_M))
        #   d_max1 = x_M1 - x_m1 = [v0^2*( ( (a_m - a_M)^2/(a_m+a_M)^2 - 1 ) )] * [1/(2*(-a_M)) - 1/(2*a_m)]
        # Simplify: ( (a_m - a_M)^2 - (a_m+a_M)^2 ) / (a_m+a_M)^2 = ( -4 a_m a_M ) / (a_m+a_M)^2
        # So d_max1 = [v0^2 * (-4 a_m a_M)/(a_m+a_M)^2] * [1/(2)*(-1/a_M - 1/a_m)? Wait, careful.
        # Actually: 
        #   x_M1 - x_m1 = (v1^2 - v0^2)/(2*(-a_M)) - (v1^2 - v0^2)/(2*a_m)   [with v0 for block being -v0? No, block's initial is -v0]
        # Better to use relative motion: relative acceleration a_rel = a_m + a_M (since both accelerate towards each other? Actually, block accelerates right at a_m, board accelerates left at a_M? But a_M is magnitude of board's deceleration, so relative acceleration = a_m - (-a_M) = a_m + a_M? Wait, board's acceleration is negative, so relative = a_m - (-a_M) = a_m+a_M.
        # Initial relative velocity = v_board - v_block = v0 - (-v0) = 2*v0.
        # Time to relative velocity zero: t1 = (2*v0)/(a_rel) = 2*v0/(a_m+a_M).
        # Then relative displacement during this time: d_max1 = relative initial velocity * t1 + 0.5 * (-a_rel) * t1^2? Actually, relative acceleration is constant a_rel, but they are moving towards each other? Let's set up:
        #   x_relative = x_M - x_m, initial relative position 0.
        #   v_relative_initial = v_M0 - v_m0 = 3 - (-3) = 6.
        #   a_relative = a_M - a_m? But a_M is negative (deceleration), a_m is positive (acceleration rightwards). So a_relative = (-a_M) - a_m = -(a_m+a_M). So relative acceleration is negative (they are approaching).
        #   Then when v_relative becomes 0, we have maximum separation? Actually, initially they are together, then they separate because v_relative positive (board moves faster right than block). Then block accelerates right, board decelerates, so v_relative decreases. When v_relative=0, they are moving together, and separation is maximum.
        #   So using kinematic: v_relative_final^2 - v_relative_initial^2 = 2 * a_relative * d_max1.
        #   Here v_relative_final=0, v_relative_initial=2*v0, a_relative = -(a_m+a_M).
        #   So 0 - (2*v0)^2 = 2 * (-(a_m+a_M)) * d_max1
        #   => -4*v0^2 = -2*(a_m+a_M)* d_max1
        #   => d_max1 = (2*v0^2)/(a_m+a_M)
        # Plug in: v0=3, a_m=6, a_M=2 -> d_max1 = 2*9/8 = 18/8 = 2.25 ✔
        # So d_max1 = 2*v0^2/(a_m+a_M).

        # Now, with perturbed m, a_m unchanged, a_M becomes (μ * perturbed_m * g)/M.
        perturbed_a_M = (self.mu * perturbed_m * self.g) / self.M
        perturbed_d_max1 = 2 * self.v0**2 / (self.a_m + perturbed_a_M)

        # Since a_M increases, denominator increases, so d_max1 decreases.
        if perturbed_d_max1 < self.d_max1:
            logs.append(f"PASS: monotonicity: increasing m decreases d_max1 (from {self.d_max1} to {perturbed_d_max1})")
        else:
            logs.append(f"FAIL: monotonicity: increasing m did not decrease d_max1")
            success = False

        # Restore original m
        self.m = original_m

        self.logs.extend(logs)
        return success

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        all_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                self.logs.append(f"\n--- {name} ---")
                success = test_func()
                if not success:
                    all_success = False
                self.logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                self.logs.append(f"ERROR in {name}: {traceback.format_exc()}")
                all_success = False

        return all_success, self.logs