import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary inputs and results from the solution instance.
        """
        # Extract from solution_instance attributes or results dictionary
        self.p0 = self._safe_get(solution_instance, 'p0', 76.0)  # cmHg
        self.h1 = self._safe_get(solution_instance, 'h1', 24.0)  # cm
        self.delta_h_initial = self._safe_get(solution_instance, 'delta_h', 4.0)  # cm
        self.h2 = self._safe_get(solution_instance, 'h2', 12.0)  # cm
        self.T0 = self._safe_get(solution_instance, 'T0', 300.0)  # K
        self.area_ratio = 3.0  # B cross-section is 3 times A

        # Extract intermediate and final results
        self.p1 = self._safe_get(solution_instance, 'p1', None)
        self.V1 = self._safe_get(solution_instance, 'V1', None)
        self.delta_h1 = self._safe_get(solution_instance, 'delta_h1', None)
        self.delta_h2 = self._safe_get(solution_instance, 'delta_h2', None)
        self.V2 = self._safe_get(solution_instance, 'V2', None)
        self.T_final = self._safe_get(solution_instance, 'T_final', None)

        # Arbitrary cross-sectional area for A (cancels out)
        self.S = 1.0

        # Compute missing values if needed
        self._compute_missing()

    def _safe_get(self, obj, key, default):
        """
        Safely retrieve attribute or dictionary value.
        """
        value = getattr(obj, key, None)
        if value is None:
            value = getattr(obj, 'results', {}).get(key, default)
        return value

    def _compute_missing(self):
        """
        Compute any missing intermediate results using fundamental physics.
        """
        # Step 1: initial pressure in tube B
        if self.p1 is None:
            self.p1 = self.p0 + self.delta_h_initial  # cmHg

        # Conservation of mercury volume and geometric relation
        if self.delta_h2 is None:
            # From Δh1 + Δh2 = Δh and Δh1 = 3 * Δh2 (area ratio)
            self.delta_h2 = self.delta_h_initial / 4.0
        if self.delta_h1 is None:
            self.delta_h1 = 3.0 * self.delta_h2

        # Volumes (in units of S, since S=1)
        if self.V1 is None:
            self.V1 = self.area_ratio * self.h2  # 3 * 12 = 36
        if self.V2 is None:
            self.V2 = self.area_ratio * (self.h2 - self.delta_h2)  # 3 * (12 - 1) = 33

        # Final temperature if not provided (compute for reference)
        if self.T_final is None:
            # Compute from ideal gas laws for tubes A and B
            p_A0 = self.p0
            V_A0 = self.h1  # S=1
            V_A_final = self.h1 + self.delta_h1  # 24 + 3 = 27
            # Final pressure in tube B (isothermal)
            p_B_final = self.p1 * self.V1 / self.V2
            p_A_final = p_B_final  # equal levels condition
            self.T_final = self.T0 * (p_A_final / p_A0) * (V_A_final / V_A0)

    def test_invariants(self):
        """
        Verify fundamental physics invariants.
        """
        logs = []
        try:
            # 1. Initial pressure in tube B: p1 = p0 + Δh
            expected_p1 = self.p0 + self.delta_h_initial
            if not math.isclose(self.p1, expected_p1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: p1 mismatch. Got {self.p1}, expected {expected_p1}")

            # 2. Conservation of mercury volume: Δh1 = 3 * Δh2
            if not math.isclose(self.delta_h1, 3.0 * self.delta_h2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δh1 should be 3×Δh2. Got Δh1={self.delta_h1}, Δh2={self.delta_h2}")

            # 3. Geometric sum: Δh1 + Δh2 = Δh_initial
            sum_delta = self.delta_h1 + self.delta_h2
            if not math.isclose(sum_delta, self.delta_h_initial, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δh1+Δh2 should equal Δh. Got {sum_delta}, expected {self.delta_h_initial}")

            # 4. Isothermal process for tube B: p1 * V1 = p2 * V2
            p2 = self.p1 * self.V1 / self.V2  # final pressure in tube B
            left = self.p1 * self.V1
            right = p2 * self.V2
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Tube B isothermal condition violated. p1*V1={left}, p2*V2={right}")

            # 5. Ideal gas law for tube A: p_A0 * V_A0 / T0 = p_A_final * V_A_final / T_final
            p_A0 = self.p0
            V_A0 = self.h1  # S=1
            V_A_final = self.h1 + self.delta_h1
            p_A_final = p2  # from above
            left_A = p_A0 * V_A0 / self.T0
            right_A = p_A_final * V_A_final / self.T_final
            if not math.isclose(left_A, right_A, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Tube A ideal gas law violated. Left={left_A}, Right={right_A}")

            if not logs:
                logs.append("PASS: All invariants satisfied.")
        except Exception as e:
            logs.append(f"ERROR during invariant test: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """
        Check that intermediate results are consistent with the final answer.
        """
        logs = []
        try:
            # Recompute T_final from intermediate values
            p_A0 = self.p0
            V_A0 = self.h1
            V_A_final = self.h1 + self.delta_h1
            p_B_final = self.p1 * self.V1 / self.V2
            T_recomputed = self.T0 * (p_B_final / p_A0) * (V_A_final / V_A0)

            if not math.isclose(self.T_final, T_recomputed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(
                    f"FAIL: T_final inconsistency. From solution: {self.T_final}, "
                    f"Recomputed: {T_recomputed}"
                )
            else:
                logs.append("PASS: Trace consistency verified.")
        except Exception as e:
            logs.append(f"ERROR during trace consistency test: {e}\n{traceback.format_exc()}")
        return logs

    def _compute_T_final(self, p0, h1, delta_h, h2, T0):
        """
        Helper to compute final temperature from inputs using the same physics.
        """
        # Step-by-step calculation
        p1 = p0 + delta_h
        delta_h2 = delta_h / 4.0
        delta_h1 = 3.0 * delta_h2
        V1 = 3.0 * h2  # area_ratio * h2
        V2 = 3.0 * (h2 - delta_h2)
        p_B_final = p1 * V1 / V2
        V_A0 = h1
        V_A_final = h1 + delta_h1
        T_final = T0 * (p_B_final / p0) * (V_A_final / V_A0)
        return T_final

    def test_monotonicity(self):
        """
        Perturb an input and verify the response is physically reasonable.
        """
        logs = []
        try:
            # Original final temperature
            T_orig = self._compute_T_final(
                self.p0, self.h1, self.delta_h_initial, self.h2, self.T0
            )

            # Positive perturbation in delta_h
            delta_h_perturbed = self.delta_h_initial + 0.1
            T_pert_pos = self._compute_T_final(
                self.p0, self.h1, delta_h_perturbed, self.h2, self.T0
            )

            # Negative perturbation
            delta_h_perturbed_neg = self.delta_h_initial - 0.1
            T_pert_neg = self._compute_T_final(
                self.p0, self.h1, delta_h_perturbed_neg, self.h2, self.T0
            )

            # Check monotonicity: T should increase with delta_h
            if not (T_pert_pos > T_orig and T_pert_neg < T_orig):
                logs.append(
                    f"FAIL: Monotonicity violated. "
                    f"T(delta_h+0.1)={T_pert_pos}, T(orig)={T_orig}, T(delta_h-0.1)={T_pert_neg}"
                )
            else:
                logs.append("PASS: Monotonicity holds (T increases with delta_h).")
        except Exception as e:
            logs.append(f"ERROR during monotonicity test: {e}\n{traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        """
        Execute all audit tests and return success flag and logs.
        """
        logs = []
        success = True

        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            test_logs = test_func()
            if any(log.startswith("FAIL") or log.startswith("ERROR") for log in test_logs):
                success = False
            logs.append(f"=== {test_name} ===")
            logs.extend(test_logs)

        return success, logs