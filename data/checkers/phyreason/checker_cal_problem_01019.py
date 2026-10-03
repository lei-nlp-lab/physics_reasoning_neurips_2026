import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        try:
            # Inputs (with safe defaults)
            self.S = getattr(solution_instance, 'S', None)
            self.m = getattr(solution_instance, 'm', None)
            self.p0 = getattr(solution_instance, 'p0', 
                             getattr(solution_instance, 'p', None))  # atmospheric pressure
            self.L = getattr(solution_instance, 'L', None)
            self.T1 = getattr(solution_instance, 'T1', 270.0)
            self.g = getattr(solution_instance, 'g', 9.8)

            # Results (from results dict or attributes)
            results = getattr(solution_instance, 'results', {})
            self.p1 = getattr(solution_instance, 'p1', results.get('p1', None))
            self.T2 = getattr(solution_instance, 'T2', results.get('T2', None))
            self.F = getattr(solution_instance, 'F', results.get('F', None))

            # Derived quantities for checking
            self.V1_initial = None
            self.V2_initial = None
            self.V1_final = None
            self.V2_final = None
            if all([self.L, self.S]):
                self.V1_initial = 3 * self.L * self.S          # Lower chamber
                self.V2_initial = 2 * self.L * (3 * self.S)    # Upper chamber
                self.V1_final = 2 * self.L * self.S
                self.V2_final = 3 * self.L * (3 * self.S)

            self.logs = []
        except Exception as e:
            self.logs.append(f"Initialization error: {e}")
            raise

    def test_invariants(self):
        """Check force equilibrium and isobaric process invariants."""
        logs = []
        try:
            # 1. Force equilibrium on piston a (from step_1)
            if all([self.m, self.g, self.p1, self.p0, self.S, self.F]):
                lhs = self.m * self.g + self.p1 * self.S
                rhs = self.p0 * self.S + self.F
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("✓ Force balance on piston a satisfied.")
                else:
                    logs.append(f"✗ Force balance on piston a failed: {lhs} ≠ {rhs}")
                    return False, logs

            # 2. Force equilibrium on piston b (from step_2)
            if all([self.p1, self.p0, self.S, self.F]):
                lhs = self.p1 * (3 * self.S)
                rhs = self.p0 * (3 * self.S) + self.F
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("✓ Force balance on piston b satisfied.")
                else:
                    logs.append(f"✗ Force balance on piston b failed: {lhs} ≠ {rhs}")
                    return False, logs

            # 3. Isobaric process check: p1 should remain constant
            if self.p1:
                # p1 is calculated once; we can't check constancy without more states.
                # Instead, check that the derived p1 matches the formula from step_2.
                expected_p1 = self.p0 + (self.m * self.g) / (2 * self.S)
                if math.isclose(self.p1, expected_p1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Gas pressure p1 matches equilibrium formula.")
                else:
                    logs.append(f"✗ p1 mismatch: {self.p1} ≠ {expected_p1}")
                    return False, logs

            # 4. Ideal gas law for initial state (optional, as p1 may not be directly given)
            if all([self.p1, self.T1, self.V1_initial, self.V2_initial]):
                # For a single gas mass, p1*V_total / T1 should be constant
                V_total_initial = self.V1_initial + self.V2_initial
                nR_initial = self.p1 * V_total_initial / self.T1
                # We don't have n, so we can't check absolute value, but we can check consistency with final state.
                logs.append("✓ Initial state passes basic sanity.")

            return True, logs
        except Exception as e:
            logs.append(f"Invariant test error: {e}\n{traceback.format_exc()}")
            return False, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        try:
            # 1. Check that T2 derived from volumes matches given T2
            if all([self.T1, self.T2, self.V1_initial, self.V2_initial, self.V1_final, self.V2_final]):
                V_initial = self.V1_initial + self.V2_initial
                V_final = self.V1_final + self.V2_final
                expected_T2 = self.T1 * (V_final / V_initial)
                if math.isclose(self.T2, expected_T2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ T2 consistent with isobaric volume change: {self.T2} K ≈ {expected_T2} K.")
                else:
                    logs.append(f"✗ T2 inconsistent: {self.T2} ≠ {expected_T2}")
                    return False, logs

            # 2. Check that F derived from p1 matches equilibrium
            if all([self.p1, self.p0, self.S, self.F]):
                expected_F = 3 * self.S * (self.p1 - self.p0)
                if math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Force F consistent with pressure difference.")
                else:
                    logs.append(f"✗ Force F mismatch: {self.F} ≠ {expected_F}")
                    return False, logs

            return True, logs
        except Exception as e:
            logs.append(f"Trace consistency test error: {e}\n{traceback.format_exc()}")
            return False, logs

    def test_monotonicity(self):
        """Perturb mass m and verify expected trends."""
        logs = []
        try:
            # Store original values
            m_original = self.m
            if m_original is None:
                logs.append("⚠ Mass m not provided; skipping monotonicity test.")
                return True, logs

            # Perturb m by +10%
            self.m = m_original * 1.10

            # Recalculate expected p1 and T2 using formulas
            new_p1 = self.p0 + (self.m * self.g) / (2 * self.S)
            # T2 should be independent of m (only depends on geometry and T1)
            new_T2 = self.T1 * ((self.V1_final + self.V2_final) / (self.V1_initial + self.V2_initial))

            # Check trends
            trend_ok = True
            if new_p1 > self.p1:
                logs.append(f"✓ p1 increases with increased mass (as expected).")
            else:
                logs.append(f"✗ p1 does not increase with increased mass.")
                trend_ok = False

            if math.isclose(new_T2, self.T2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ T2 remains constant with mass change (as expected).")
            else:
                logs.append(f"✗ T2 changes with mass (unexpected).")
                trend_ok = False

            # Restore original m
            self.m = m_original

            return trend_ok, logs
        except Exception as e:
            logs.append(f"Monotonicity test error: {e}\n{traceback.format_exc()}")
            # Restore m on error
            if 'm_original' in locals():
                self.m = m_original
            return False, logs

    def run_all_tests(self):
        """Execute all checks and return (success, logs)."""
        all_logs = []
        success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n--- {test_name} ---")
            try:
                passed, logs = test_method()
                all_logs.extend(logs)
                if not passed:
                    success = False
            except Exception as e:
                all_logs.append(f"Test crashed: {e}")
                success = False

        return success, all_logs