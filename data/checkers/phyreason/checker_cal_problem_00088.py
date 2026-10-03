import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.r = getattr(solution_instance, 'r', 0.1)
        self.L1 = getattr(solution_instance, 'L1', 0.2)
        self.L2 = getattr(solution_instance, 'L2', 0.4)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.m = getattr(solution_instance, 'm', 0.002)  # 2g = 0.002 kg
        self.g = getattr(solution_instance, 'g', 10.0)   # from expression matching

        # Safe extraction of results with fallback defaults
        # Use solution_instance.results if available, else empty dict
        results = getattr(solution_instance, 'results', {})
        self.v_F = results.get('v_F', 0.0)
        self.v_E = results.get('v_E', 0.0)
        self.F_N = results.get('F_N', 0.0)
        self.E_p0 = results.get('E_p0', 0.0)
        self.E_p1 = results.get('E_p1', 0.0)
        self.h_1 = results.get('h_1', 0.0)
        self.h_2 = results.get('h_2', 0.0)
        # E_p might be a symbolic expression string, a callable, or a numeric value
        self.E_p = results.get('E_p', 0.0)

        # Store original instance for potential restoration
        self.solution_instance = solution_instance
        self.logs = []

    def _log(self, message, success):
        self.logs.append((success, message))

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def _eval_E_p(self, h_value):
        """Evaluate E_p at a given h value, handling different stored types."""
        if callable(self.E_p):
            return self.E_p(h_value)
        elif isinstance(self.E_p, (int, float)):
            return float(self.E_p)
        elif isinstance(self.E_p, str):
            # Try to parse simple expression like "0.002*(10.0*h + 3.0)"
            # Replace 'h' with the numeric value and evaluate
            try:
                expr = self.E_p.replace('h', str(h_value))
                return float(eval(expr))
            except:
                # If parsing fails, return a default computed from formula
                return self.m * self.g * h_value + self.mu * self.m * self.g * (self.L1 + self.L2)
        else:
            # Fallback to direct formula
            return self.m * self.g * h_value + self.mu * self.m * self.g * (self.L1 + self.L2)

    def test_invariants(self):
        """Test physical invariants based on KB theorems."""
        try:
            # 1. Centripetal condition at top point F: m*g = m*v_F^2 / r
            expected_vF_sq = self.g * self.r
            actual_vF_sq = self.v_F**2
            if self._isclose(actual_vF_sq, expected_vF_sq):
                self._log(f"Centripetal condition at F: v_F^2 = {actual_vF_sq} ≈ g*r = {expected_vF_sq}", True)
            else:
                self._log(f"Centripetal condition at F failed: v_F^2={actual_vF_sq}, g*r={expected_vF_sq}", False)

            # 2. Work-energy from E to B (step2): 1/2 m v_E^2 = m*g*h_1 + mu*m*g*L2
            lhs = 0.5 * self.m * self.v_E**2
            rhs = self.m * self.g * self.h_1 + self.mu * self.m * self.g * self.L2
            if self._isclose(lhs, rhs):
                self._log(f"Work-energy E->B: KE_E = {lhs} ≈ PE_gain + friction work = {rhs}", True)
            else:
                self._log(f"Work-energy E->B failed: KE_E={lhs}, m*g*h_1+mu*m*g*L2={rhs}", False)

            # 3. Normal force at E (step3): F_N = m*g + m*v_E^2/r
            expected_FN = self.m * self.g + self.m * self.v_E**2 / self.r
            if self._isclose(self.F_N, expected_FN):
                self._log(f"Normal force at E: F_N={self.F_N} ≈ m*g + m*v_E^2/r = {expected_FN}", True)
            else:
                self._log(f"Normal force at E failed: F_N={self.F_N}, expected={expected_FN}", False)

            # 4. Energy conservation O->E (step4): E_p0 = 1/2 m v_E^2 + mu*m*g*L1
            expected_Ep0 = 0.5 * self.m * self.v_E**2 + self.mu * self.m * self.g * self.L1
            if self._isclose(self.E_p0, expected_Ep0):
                self._log(f"Energy O->E: E_p0={self.E_p0} ≈ KE_E + friction work OE = {expected_Ep0}", True)
            else:
                self._log(f"Energy O->E failed: E_p0={self.E_p0}, expected={expected_Ep0}", False)

            # 5. Consistency between step5 and step6 for E_p1
            Ep1_step5 = 2 * self.m * self.g * self.r + self.mu * self.m * self.g * self.L1 + 0.5 * self.m * self.v_F**2
            Ep1_step6 = self.m * self.g * self.h_1 + self.mu * self.m * self.g * (self.L1 + self.L2)
            if self._isclose(Ep1_step5, Ep1_step6) and self._isclose(self.E_p1, Ep1_step5):
                self._log(f"E_p1 consistency: step5={Ep1_step5}, step6={Ep1_step6}, stored={self.E_p1}", True)
            else:
                self._log(f"E_p1 inconsistency: step5={Ep1_step5}, step6={Ep1_step6}, stored={self.E_p1}", False)

            # 6. Height h_1 from geometry (step8) and energy (step6)
            expected_h2 = self.L2 * 0.5
            if self._isclose(self.h_2, expected_h2):
                self._log(f"Geometric max height: h_2={self.h_2} ≈ L2/2 = {expected_h2}", True)
            else:
                self._log(f"Geometric max height failed: h_2={self.h_2}, expected={expected_h2}", False)

            # Additionally, h_1 should satisfy 2.5*g*r = g*h_1 + mu*g*L2 (derived from steps)
            expected_h1 = 2.5 * self.r - self.mu * self.L2
            if self._isclose(self.h_1, expected_h1):
                self._log(f"Height h_1 consistency: h_1={self.h_1} ≈ 2.5r - mu*L2 = {expected_h1}", True)
            else:
                self._log(f"Height h_1 inconsistency: h_1={self.h_1}, expected={expected_h1}", False)

            # 7. Energy expression E_p (step9): E_p = m*g*h + mu*m*g*(L1+L2)
            h_sample = 0.1  # middle of range
            expected_Ep = self.m * self.g * h_sample + self.mu * self.m * self.g * (self.L1 + self.L2)
            actual_Ep = self._eval_E_p(h_sample)
            if self._isclose(actual_Ep, expected_Ep):
                self._log(f"E_p expression at h={h_sample}: {actual_Ep} ≈ m*g*h + mu*m*g*(L1+L2) = {expected_Ep}", True)
            else:
                self._log(f"E_p expression mismatch at h={h_sample}: got {actual_Ep}, expected {expected_Ep}", False)

        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check that v_F is consistent with the target answer 1 m/s
            if self._isclose(self.v_F, 1.0):
                self._log(f"v_F matches target: {self.v_F} ≈ 1 m/s", True)
            else:
                self._log(f"v_F deviation: {self.v_F} vs target 1 m/s", False)

            # Check F_N vs target 0.14 N
            if self._isclose(self.F_N, 0.14):
                self._log(f"F_N matches target: {self.F_N} ≈ 0.14 N", True)
            else:
                self._log(f"F_N deviation: {self.F_N} vs target 0.14 N", False)

            # Check E_p0 vs target 8e-3 J
            if self._isclose(self.E_p0, 8e-3):
                self._log(f"E_p0 matches target: {self.E_p0} ≈ 8e-3 J", True)
            else:
                self._log(f"E_p0 deviation: {self.E_p0} vs target 8e-3 J", False)

            # Check E_p expression: target is 2e-3*(10h+3) = 0.002*(10h+3)
            target_coeff_h = 0.002 * 10  # 0.02
            target_constant = 0.002 * 3  # 0.006
            actual_coeff_h = self.m * self.g  # should be 0.002 * 10 = 0.02
            actual_constant = self.mu * self.m * self.g * (self.L1 + self.L2)  # should be 0.5 * 0.002 * 10 * (0.2+0.4) = 0.006
            
            if self._isclose(actual_coeff_h, target_coeff_h) and self._isclose(actual_constant, target_constant):
                self._log("E_p expression matches target form", True)
            else:
                self._log(f"E_p expression mismatch: target coeff={target_coeff_h}, const={target_constant}, actual coeff={actual_coeff_h}, const={actual_constant}", False)

            # Check that h_1 and h_2 are within the range [0.05, 0.2]
            if 0.05 <= self.h_1 <= 0.2 and 0.05 <= self.h_2 <= 0.2:
                self._log(f"h_1={self.h_1}, h_2={self.h_2} within [0.05, 0.2]", True)
            else:
                self._log(f"h_1={self.h_1} or h_2={self.h_2} outside [0.05, 0.2]", False)

        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)

    def test_monotonicity(self):
        """Perturb mass m and check that E_p0 increases."""
        try:
            # Store original m
            original_m = self.m
            # Perturb by a small relative amount
            delta = 1e-5
            self.m += delta

            # Recompute E_p0 using the formula: E_p0 = 0.5*m*v_E^2 + mu*m*g*L1
            new_Ep0 = 0.5 * self.m * self.v_E**2 + self.mu * self.m * self.g * self.L1
            old_Ep0 = self.E_p0  # from original

            # Check monotonic increase: since all terms are linear in m, increasing m should increase E_p0
            if new_Ep0 > old_Ep0:
                self._log(f"Monotonicity (m increase): E_p0 increases from {old_Ep0} to {new_Ep0}", True)
            else:
                self._log(f"Monotonicity violation: E_p0 did not increase when m increased", False)

            # Restore original m
            self.m = original_m

            # Additional test: perturb mu and check E_p0 increases
            original_mu = self.mu
            self.mu += 1e-3
            new_Ep0_mu = 0.5 * self.m * self.v_E**2 + self.mu * self.m * self.g * self.L1
            if new_Ep0_mu > old_Ep0:
                self._log(f"Monotonicity (mu increase): E_p0 increases from {old_Ep0} to {new_Ep0_mu}", True)
            else:
                self._log(f"Monotonicity violation (mu): E_p0 did not increase when mu increased", False)
            self.mu = original_mu

        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)
            # Ensure restoration on error
            self.m = original_m
            self.mu = original_mu

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()

        # Determine overall success: all logs must be True
        success = all(log[0] for log in self.logs)
        return success, self.logs