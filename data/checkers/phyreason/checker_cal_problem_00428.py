import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        """
        self.g = 10.0  # given in the problem
        self.t_last = 4.0  # time for the last segment (given)
        self.delta_x_last = 200.0  # distance of the last segment (given)

        # Safely extract results from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract v_0 (initial velocity at start of last 4 seconds)
        self.v_0 = getattr(solution_instance, 'v_0', self.results.get('v_0', None))
        # Extract total time t_all
        self.t_all = getattr(solution_instance, 't_all', self.results.get('t_all', None))
        # Extract final velocity v
        self.v = getattr(solution_instance, 'v', self.results.get('v', None))

        # If any result is None, try to compute from others (fallback)
        if self.v_0 is None or self.t_all is None or self.v is None:
            self._compute_missing()

    def _compute_missing(self):
        """
        Compute missing quantities using kinematic equations as a fallback.
        """
        try:
            if self.v_0 is None:
                # Use delta_x = v_0 * t + 0.5 * g * t^2 for the last segment
                self.v_0 = (self.delta_x_last - 0.5 * self.g * self.t_last**2) / self.t_last
            if self.v is None:
                self.v = self.v_0 + self.g * self.t_last
            if self.t_all is None:
                self.t_all = self.v / self.g
        except Exception:
            pass  # If computation fails, keep as None

    def test_invariants(self):
        """
        Verify fundamental physical invariants: kinematic consistency and energy conservation.
        """
        logs = []
        try:
            # Check 1: Kinematic consistency for the last segment
            # Equation: delta_x_last = v_0 * t_last + 0.5 * g * t_last^2
            lhs = self.delta_x_last
            rhs = self.v_0 * self.t_last + 0.5 * self.g * self.t_last**2
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Last segment kinematics mismatch. Computed displacement: {rhs:.2f}, given: {lhs:.2f}")
            else:
                logs.append("PASS: Last segment kinematics consistent.")

            # Check 2: Final velocity from total time
            # v = g * t_all
            v_from_tall = self.g * self.t_all
            if not math.isclose(self.v, v_from_tall, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final velocity mismatch. From t_all: {v_from_tall:.2f}, given v: {self.v:.2f}")
            else:
                logs.append("PASS: Final velocity consistent with total time.")

            # Check 3: Energy conservation (total height derived from results)
            # Total height h = 0.5 * g * t_all^2
            h_total = 0.5 * self.g * self.t_all**2
            # Kinetic energy at bottom: 0.5 * v^2, potential energy loss: g * h_total
            # Should satisfy v^2 = 2 * g * h_total
            lhs_energy = self.v**2
            rhs_energy = 2 * self.g * h_total
            if not math.isclose(lhs_energy, rhs_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation mismatch. v^2 = {lhs_energy:.2f}, 2gh = {rhs_energy:.2f}")
            else:
                logs.append("PASS: Energy conservation holds.")

            # Check 4: Alternative kinematic relation for the last segment: v^2 = v_0^2 + 2*g*delta_x_last
            lhs_last = self.v**2
            rhs_last = self.v_0**2 + 2 * self.g * self.delta_x_last
            if not math.isclose(lhs_last, rhs_last, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Last segment v^2 relation mismatch. v^2 = {lhs_last:.2f}, v_0^2+2gΔx = {rhs_last:.2f}")
            else:
                logs.append("PASS: Last segment v^2 relation consistent.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")

        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer and input data.
        """
        logs = []
        try:
            # Step 1: v_0 computed from last segment should match the given v_0
            v_0_computed = (self.delta_x_last - 0.5 * self.g * self.t_last**2) / self.t_last
            if not math.isclose(self.v_0, v_0_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_0 mismatch. Computed: {v_0_computed:.2f}, given: {self.v_0:.2f}")
            else:
                logs.append("PASS: v_0 consistent with last segment data.")

            # Step 2: t_all should equal v/g
            t_all_computed = self.v / self.g
            if not math.isclose(self.t_all, t_all_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_all mismatch. Computed: {t_all_computed:.2f}, given: {self.t_all:.2f}")
            else:
                logs.append("PASS: t_all consistent with final velocity.")

            # Step 3: v should equal v_0 + g*t_last
            v_computed = self.v_0 + self.g * self.t_last
            if not math.isclose(self.v, v_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v mismatch. Computed: {v_computed:.2f}, given: {self.v:.2f}")
            else:
                logs.append("PASS: v consistent with v_0 and last segment time.")

            # Overall consistency: total height from two methods
            # Method 1: h_total = 0.5 * g * t_all^2
            h1 = 0.5 * self.g * self.t_all**2
            # Method 2: height fallen before last segment + last segment
            t_before = self.t_all - self.t_last
            h_before = 0.5 * self.g * t_before**2
            h2 = h_before + self.delta_x_last
            if not math.isclose(h1, h2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Total height mismatch. From t_all: {h1:.2f}, from segments: {h2:.2f}")
            else:
                logs.append("PASS: Total height consistent across methods.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")

        return logs

    def test_monotonicity(self):
        """
        Perturb an input (delta_x_last) and verify that results change in the expected direction.
        """
        logs = []
        try:
            # Original values
            v_0_orig = self.v_0
            t_all_orig = self.t_all
            v_orig = self.v

            # Perturb delta_x_last by +1%
            delta_x_perturbed = self.delta_x_last * 1.01
            # Recompute v_0 from the perturbed delta_x
            v_0_perturbed = (delta_x_perturbed - 0.5 * self.g * self.t_last**2) / self.t_last
            # Recompute v
            v_perturbed = v_0_perturbed + self.g * self.t_last
            # Recompute t_all
            t_all_perturbed = v_perturbed / self.g

            # Check monotonicity: increasing delta_x_last should increase v_0, v, and t_all
            if not (v_0_perturbed > v_0_orig):
                logs.append(f"WARNING: v_0 did not increase as expected. Original: {v_0_orig:.2f}, perturbed: {v_0_perturbed:.2f}")
            else:
                logs.append("PASS: v_0 increases with increased last segment distance.")

            if not (v_perturbed > v_orig):
                logs.append(f"WARNING: v did not increase as expected. Original: {v_orig:.2f}, perturbed: {v_perturbed:.2f}")
            else:
                logs.append("PASS: v increases with increased last segment distance.")

            if not (t_all_perturbed > t_all_orig):
                logs.append(f"WARNING: t_all did not increase as expected. Original: {t_all_orig:.2f}, perturbed: {t_all_perturbed:.2f}")
            else:
                logs.append("PASS: t_all increases with increased last segment distance.")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")

        return logs

    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        logs = []
        logs.append("=== PhysicsChecker started ===")
        
        # Check that all required quantities are available
        if None in (self.v_0, self.t_all, self.v):
            logs.append("ERROR: Missing one or more required results (v_0, t_all, v).")
            return False, logs
        
        # Run individual test suites
        logs.append("\n--- test_invariants ---")
        logs.extend(self.test_invariants())
        
        logs.append("\n--- test_trace_consistency ---")
        logs.extend(self.test_trace_consistency())
        
        logs.append("\n--- test_monotonicity ---")
        logs.extend(self.test_monotonicity())
        
        # Determine overall success: no FAIL or ERROR messages
        success = True
        for log in logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        logs.append(f"\n=== PhysicsChecker finished. Overall success: {success} ===")
        return success, logs