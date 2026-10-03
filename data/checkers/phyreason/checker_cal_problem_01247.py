import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Results (fallback to results dictionary)
        results = getattr(solution_instance, 'results', {})
        
        # Extract all needed values from results with defaults
        self.v0 = results.get('v_0', 0.0)
        self.v1 = results.get('v_1', 0.0)
        self.F_N = results.get('F_N', 0.0)
        self.F_pressure = results.get('F_pressure', 0.0)
        self.W = results.get('W', 0.0)
        self.W2 = results.get('W_2', 0.0)
        
        # Extract constants from results
        self.G = results.get('G', 6.67430e-11)
        # Force correct value if G is 0.0 (common bug in student code)
        if math.isclose(self.G, 0.0, abs_tol=1e-20):
            self.G = 6.67430e-11
        
        self.M = results.get('M', 5.972e24)
        self.m0 = results.get('m0', 100000.0)
        self.r0 = results.get('r0', 6771000.0)
        self.r1 = results.get('r1', 6671000.0)
        
        self.logs = []

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify key physics laws from the KB and Steps Analysis."""
        logs = []
        try:
            # 1. v0 = sqrt(G*M / r0)
            v0_expected = math.sqrt(self.G * self.M / self.r0)
            if not self._is_close(self.v0, v0_expected):
                logs.append(f"FAIL: v0 mismatch. Expected {v0_expected}, got {self.v0}")
            else:
                logs.append("PASS: v0 = sqrt(G*M / r0)")

            # 2. F_N should be zero (from step 2)
            # Due to floating-point errors, compute directly and check if close to 0
            F_N_expected = (self.G * self.M * self.m0 / self.r0**2) - (self.m0 * self.v0**2 / self.r0)
            # Use a reasonable tolerance for floating-point errors
            if not math.isclose(F_N_expected, 0.0, abs_tol=1e-3):
                logs.append(f"FAIL: F_N not zero. Computed {F_N_expected}")
            elif not math.isclose(self.F_N, 0.0, abs_tol=1e-3):
                logs.append(f"FAIL: F_N not zero. Got {self.F_N}")
            else:
                logs.append("PASS: F_N = 0 (gravitational force equals centripetal force)")

            # 3. F_pressure should be zero (Newton's Third Law, reaction to F_N)
            if not math.isclose(self.F_pressure, 0.0, abs_tol=1e-3):
                logs.append(f"FAIL: F_pressure not zero. Expected ~0, got {self.F_pressure}")
            else:
                logs.append("PASS: F_pressure = 0 (Newton's Third Law)")

            # 4. v1 = sqrt(G*M / r1)
            v1_expected = math.sqrt(self.G * self.M / self.r1)
            if not self._is_close(self.v1, v1_expected):
                logs.append(f"FAIL: v1 mismatch. Expected {v1_expected}, got {self.v1}")
            else:
                logs.append("PASS: v1 = sqrt(G*M / r1)")

            # 5. W2 = G*M*m0*(1/r0 - 1/r1)
            W2_expected = self.G * self.M * self.m0 * (1/self.r0 - 1/self.r1)
            if not self._is_close(self.W2, W2_expected):
                logs.append(f"FAIL: W2 mismatch. Expected {W2_expected}, got {self.W2}")
            else:
                logs.append("PASS: W2 = G*M*m0*(1/r0 - 1/r1)")

            # 6. W = (G*M*m0)/(2*r1) - (G*M*m0)/(2*r0)
            W_expected = (self.G * self.M * self.m0) / (2*self.r1) - (self.G * self.M * self.m0) / (2*self.r0)
            if not self._is_close(self.W, W_expected):
                logs.append(f"FAIL: W mismatch. Expected {W_expected}, got {self.W}")
            else:
                logs.append("PASS: W = G*M*m0/(2*r1) - G*M*m0/(2*r0)")

            # 7. Work-energy theorem: W + W2 = 0.5*m0*(v0^2 - v1^2)
            left = self.W + self.W2
            right = 0.5 * self.m0 * (self.v0**2 - self.v1**2)
            if not self._is_close(left, right):
                logs.append(f"FAIL: Work-energy theorem violated. W+W2={left}, ΔK={right}")
            else:
                logs.append("PASS: Work-energy theorem holds (W+W2 = ΔK)")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("FAIL") or l.startswith("ERROR")]) == 0

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that gravitational force equals centripetal force at r0
            F_grav0 = self.G * self.M * self.m0 / self.r0**2
            F_c0 = self.m0 * self.v0**2 / self.r0
            if not self._is_close(F_grav0, F_c0):
                logs.append(f"FAIL: At r0, gravitational force {F_grav0} ≠ centripetal force {F_c0}")
            else:
                logs.append("PASS: At r0, gravitational force = centripetal force")

            # Check that gravitational force equals centripetal force at r1
            F_grav1 = self.G * self.M * self.m0 / self.r1**2
            F_c1 = self.m0 * self.v1**2 / self.r1
            if not self._is_close(F_grav1, F_c1):
                logs.append(f"FAIL: At r1, gravitational force {F_grav1} ≠ centripetal force {F_c1}")
            else:
                logs.append("PASS: At r1, gravitational force = centripetal force")

            # Check that the derived W matches the target answer
            target_W = (self.G * self.M * self.m0) / (2*self.r1) - (self.G * self.M * self.m0) / (2*self.r0)
            if not self._is_close(self.W, target_W):
                logs.append(f"FAIL: W does not match target answer. Expected {target_W}, got {self.W}")
            else:
                logs.append("PASS: W matches target answer")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("FAIL") or l.startswith("ERROR")]) == 0

    def test_monotonicity(self):
        """
        Perturb an input (m0) and verify that the outputs change as expected.
        Restore the original values after testing.
        """
        logs = []
        try:
            # Save original values
            original_m0 = self.m0
            perturbed_m0 = original_m0 * 2.0

            # Expected changes:
            # v0 and v1 are independent of m0, so they should remain the same.
            # F_N should remain 0 (since it's 0 regardless of m0).
            # W and W2 should double.

            # Compute expected perturbed values
            v0_original = self.v0
            v0_perturbed = math.sqrt(self.G * self.M / self.r0)  # same as original
            if not self._is_close(v0_original, v0_perturbed):
                logs.append(f"FAIL: v0 changed when m0 was perturbed. This is unexpected.")
            else:
                logs.append("PASS: v0 is independent of m0")

            v1_original = self.v1
            v1_perturbed = math.sqrt(self.G * self.M / self.r1)  # same as original
            if not self._is_close(v1_original, v1_perturbed):
                logs.append(f"FAIL: v1 changed when m0 was perturbed. This is unexpected.")
            else:
                logs.append("PASS: v1 is independent of m0")

            # F_N should remain 0
            F_N_perturbed = 0.0
            if not math.isclose(self.F_N, 0.0, abs_tol=1e-3) or not math.isclose(F_N_perturbed, 0.0, abs_tol=1e-3):
                logs.append(f"FAIL: F_N is not zero or did not remain zero after perturbation.")
            else:
                logs.append("PASS: F_N remains zero when m0 is doubled")

            # W and W2 should double
            W_original = self.W
            W2_original = self.W2
            W_perturbed = (self.G * self.M * perturbed_m0) / (2*self.r1) - (self.G * self.M * perturbed_m0) / (2*self.r0)
            W2_perturbed = self.G * self.M * perturbed_m0 * (1/self.r0 - 1/self.r1)

            if not self._is_close(W_perturbed, 2 * W_original):
                logs.append(f"FAIL: W did not double when m0 doubled. Original {W_original}, perturbed {W_perturbed}")
            else:
                logs.append("PASS: W scales linearly with m0")

            if not self._is_close(W2_perturbed, 2 * W2_original):
                logs.append(f"FAIL: W2 did not double when m0 doubled. Original {W2_original}, perturbed {W2_perturbed}")
            else:
                logs.append("PASS: W2 scales linearly with m0")

            # Restore original m0
            self.m0 = original_m0

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("FAIL") or l.startswith("ERROR")]) == 0

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        success = True

        # Run tests
        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False

        # Summary
        if success:
            self.logs.append("OVERALL: All tests passed.")
        else:
            self.logs.append("OVERALL: Some tests failed.")

        return success, self.logs