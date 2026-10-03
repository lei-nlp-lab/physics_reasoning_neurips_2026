import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Extract inputs with safe attribute access
        self.OA = getattr(solution_instance, 'OA', None)
        self.OB = getattr(solution_instance, 'OB', None)
        self.delta_t = getattr(solution_instance, 'delta_t', None)
        self.n = getattr(solution_instance, 'n', None)
        self.Pi1 = getattr(solution_instance, 'Pi1', getattr(solution_instance, 'Pi1', 2.0))  # default 2.0 as given

        # Extract results from solution_instance.results with safe fallback
        results = getattr(solution_instance, 'results', {})
        self.delta_x = results.get('delta_x', 0.0)
        self.T = results.get('T', 0.0)
        self.fraclambda_leftlambda_right = results.get('fraclambda_leftlambda_right', 0.0)
        self.fracv_leftv_right = results.get('fracv_leftv_right', 0.0)

        # For symbolic checks
        self.lambda_left = getattr(solution_instance, 'lambda_left', None)
        self.lambda_right = getattr(solution_instance, 'lambda_right', None)
        self.v_left = getattr(solution_instance, 'v_left', None)
        self.v_right = getattr(solution_instance, 'v_right', None)

        self.logs = []

    def _log(self, msg, level='INFO'):
        """Helper to log messages."""
        self.logs.append(f"{level}: {msg}")

    def test_invariants(self):
        """
        Verify fundamental physics relationships.
        """
        try:
            # 1. Check ratio of velocities equals OA/OB (if both provided)
            if self.OA is not None and self.OB is not None:
                expected_ratio = self.OA / self.OB
                if not math.isclose(self.fracv_leftv_right, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Velocity ratio mismatch: {self.fracv_leftv_right} vs {expected_ratio}", 'ERROR')
                    return False
                self._log(f"Velocity ratio check passed: {self.fracv_leftv_right} ≈ {expected_ratio}")

            # 2. Check wavelength ratio equals velocity ratio
            if not math.isclose(self.fraclambda_leftlambda_right, self.fracv_leftv_right, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Wavelength ratio mismatch: {self.fraclambda_leftlambda_right} vs {self.fracv_leftv_right}", 'ERROR')
                return False
            self._log(f"Wavelength ratio equals velocity ratio: {self.fraclambda_leftlambda_right} ≈ {self.fracv_leftv_right}")

            # 3. Symbolic check: v = f * lambda (if frequencies are available)
            f_left = getattr(self.solution, 'f_left', None)
            f_right = getattr(self.solution, 'f_right', None)
            if f_left is not None and self.v_left is not None and self.lambda_left is not None:
                if not math.isclose(self.v_left, f_left * self.lambda_left, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"v_left ≠ f_left * lambda_left: {self.v_left} vs {f_left * self.lambda_left}", 'WARNING')
            if f_right is not None and self.v_right is not None and self.lambda_right is not None:
                if not math.isclose(self.v_right, f_right * self.lambda_right, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"v_right ≠ f_right * lambda_right: {self.v_right} vs {f_right * self.lambda_right}", 'WARNING')

            # 4. Check wave speed consistency (if both velocities are given)
            if self.v_left is not None and self.v_right is not None:
                # In the same medium, speeds should be equal unless specified otherwise.
                # Here, the ratio is given as 1/2, so we don't require equality.
                pass

            self._log("All invariants passed.")
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", 'ERROR')
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate steps with final results.
        """
        try:
            # 1. Check that wavelength ratio equals velocity ratio (redundant but explicit)
            if not math.isclose(self.fraclambda_leftlambda_right, self.fracv_leftv_right, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace consistency failed: λ_left/λ_right = {self.fraclambda_leftlambda_right}, "
                          f"v_left/v_right = {self.fracv_leftv_right}", 'ERROR')
                return False
            self._log(f"Step 2 to Step 3 consistency: λ_left/λ_right = v_left/v_right = {self.fracv_leftv_right}")

            # 2. Check period formula T = 2 * delta_t / (2*n + 1) if n and delta_t are provided
            if self.n is not None and self.delta_t is not None:
                expected_T = 2 * self.delta_t / (2 * self.n + 1)
                if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Period formula mismatch: T = {self.T}, expected T = 2*Δt/(2n+1) = {expected_T}", 'ERROR')
                    return False
                self._log(f"Period formula check passed: T = {self.T} ≈ 2*Δt/(2n+1) = {expected_T}")

            # 3. Check path difference condition (if delta_x and lambda are available)
            #    Δx = (2n+1) * λ/2, where λ might be an average or specific lambda.
            #    Since lambda may differ left/right, we use an average if available.
            lambda_avg = getattr(self.solution, 'lambda_avg', None)
            if self.delta_x is not None and lambda_avg is not None and self.n is not None:
                expected_delta_x = (2 * self.n + 1) * lambda_avg / 2
                if not math.isclose(self.delta_x, expected_delta_x, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Path difference condition mismatch: Δx = {self.delta_x}, "
                              f"expected (2n+1)*λ/2 = {expected_delta_x}", 'WARNING')
            self._log("Trace consistency passed.")
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", 'ERROR')
            return False

    def test_monotonicity(self):
        """
        Perturb inputs and check expected trends.
        """
        try:
            # Monotonicity only applies if we have the necessary inputs.
            if self.n is None or self.delta_t is None:
                self._log("Skipping monotonicity: n or delta_t not provided.", 'INFO')
                return True

            original_n = self.n
            original_delta_t = self.delta_t
            original_T = self.T

            # 1. Increase n by 1, T should decrease (for fixed delta_t)
            perturbed_n = original_n + 1
            perturbed_T = 2 * original_delta_t / (2 * perturbed_n + 1)
            if not (perturbed_T < original_T):
                self._log(f"Monotonicity failed: T does not decrease when n increases. "
                          f"T(n={original_n}) = {original_T}, T(n={perturbed_n}) = {perturbed_T}", 'ERROR')
                return False
            self._log(f"Monotonicity passed: T decreases when n increases (T={original_T} -> {perturbed_T}).")

            # 2. Increase delta_t by 10%, T should increase (for fixed n)
            perturbed_delta_t = original_delta_t * 1.1
            perturbed_T = 2 * perturbed_delta_t / (2 * original_n + 1)
            if not (perturbed_T > original_T):
                self._log(f"Monotonicity failed: T does not increase when Δt increases. "
                          f"T(Δt={original_delta_t}) = {original_T}, T(Δt={perturbed_delta_t}) = {perturbed_T}", 'ERROR')
                return False
            self._log(f"Monotonicity passed: T increases when Δt increases (T={original_T} -> {perturbed_T}).")

            # 3. For wavelength ratio: if OA/OB changes, the ratio should change proportionally.
            #    Since OA and OB are fixed in the problem, we skip this unless they are parameters.
            if self.OA is not None and self.OB is not None:
                # Perturb OA by 10% and check that velocity ratio changes accordingly.
                perturbed_OA = self.OA * 1.1
                expected_ratio = perturbed_OA / self.OB
                # We cannot recompute the velocity ratio from the solution, so we only note the expected trend.
                self._log(f"If OA increases by 10%, expected v_left/v_right = {expected_ratio} (originally {self.fracv_leftv_right}).")

            self._log("All monotonicity checks passed.")
            return True
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", 'ERROR')
            return False

    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        """
        self.logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} failed.", 'ERROR')
                else:
                    self._log(f"{test_name} passed.", 'INFO')
            except Exception as e:
                success = False
                self._log(f"{test_name} raised exception: {traceback.format_exc()}", 'ERROR')

        return success, self.logs