import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Inputs: they are provided as global variables in the execution environment.
        # We try to get them from the solution_instance's namespace or as attributes.
        # If not found, we fall back to extracting from the global scope of the checker.
        # We'll use a helper to get the value.
        def get_input(name, default=0.0):
            # First try as attribute of solution_instance
            val = getattr(solution_instance, name, None)
            if val is not None:
                return float(val)
            # Then try from the results dict
            results_dict = getattr(solution_instance, 'results', {})
            if name in results_dict:
                return float(results_dict[name])
            # Finally, try from the global namespace (where the problem defines them)
            try:
                # This assumes the checker is run in the same namespace as the solution
                import __main__
                if hasattr(__main__, name):
                    return float(getattr(__main__, name))
            except:
                pass
            return default
        
        self.m = get_input('m', 0.0)
        self.P = get_input('P', 0.0)
        self.v = get_input('v', 0.0)
        self.t = get_input('t', 0.0)
        self.g = get_input('g', 9.8)
        self.theta = get_input('theta', 0.0)

        # Results (from solution_instance.results or as attributes)
        self.results = {}
        result_keys = ['f', 'x', 'E_k', 'F', 'r', 'F_n', 'W_f', 'W_P']
        for key in result_keys:
            val = getattr(solution_instance, key, 
                         getattr(solution_instance, 'results', {}).get(key, 0.0))
            # Ensure it's a float, not a sympy object
            if hasattr(val, 'evalf'):
                val = float(val.evalf())
            self.results[key] = float(val)

        # Aliases for convenience
        self.f = self.results['f']
        self.x = self.results['x']
        self.E_k = self.results['E_k']
        self.F = self.results['F']
        self.r = self.results['r']
        self.F_n = self.results['F_n']
        self.W_f = self.results['W_f']
        self.W_P = self.results['W_P']

        # Tolerance for numerical comparisons (1-2% relative)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

        # Logs for reporting
        self.logs = []

    def _log(self, message):
        """Append a message to the logs."""
        self.logs.append(message)

    def _is_close(self, a, b):
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        """Check fundamental physical laws: energy conservation and centripetal force."""
        try:
            # 1. Work-energy theorem: net work = change in kinetic energy
            # W_net = W_P - W_f (traction work minus resistance work)
            # delta_K = E_k - 0 (starts from rest)
            net_work = self.W_P - self.W_f
            delta_K = self.E_k
            if not self._is_close(net_work, delta_K):
                self._log(f"Energy conservation violated: W_P - W_f = {net_work}, ΔK = {delta_K}")
                return False
            self._log("Energy conservation (work-energy theorem) holds.")

            # 2. Centripetal force relation: F_n = m * v^2 / r and F_n = m * g * tan(theta)
            # Avoid division by zero: ensure r != 0
            if self.r == 0:
                self._log("Radius r is zero, cannot compute centripetal force from motion.")
                return False
            centripetal_from_motion = self.m * self.v**2 / self.r
            centripetal_from_banking = self.m * self.g * math.tan(self.theta)
            if not (self._is_close(self.F_n, centripetal_from_motion) and 
                    self._is_close(self.F_n, centripetal_from_banking)):
                self._log(f"Centripetal force mismatch: F_n = {self.F_n}, m*v^2/r = {centripetal_from_motion}, m*g*tanθ = {centripetal_from_banking}")
                return False
            self._log("Centripetal force relations hold.")
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results and final answers."""
        try:
            # Step 1: Traction and resistance forces at maximum speed
            # P = F * v => F = P/v, and at max speed, traction = resistance, so f = P/v
            if self.v == 0:
                self._log("Velocity v is zero, cannot compute expected forces.")
                return False
            expected_F = self.P / self.v
            if not (self._is_close(self.F, expected_F) and self._is_close(self.f, expected_F)):
                self._log(f"Step 1 force mismatch: F = {self.F}, f = {self.f}, expected {expected_F}")
                return False
            self._log("Step 1 forces (F and f) are consistent with P = F*v.")

            # Step 2: Energy and work
            # Kinetic energy
            expected_E_k = 0.5 * self.m * self.v**2
            if not self._is_close(self.E_k, expected_E_k):
                self._log(f"Step 2 kinetic energy mismatch: E_k = {self.E_k}, expected {expected_E_k}")
                return False
            # Work by traction
            expected_W_P = self.P * self.t
            if not self._is_close(self.W_P, expected_W_P):
                self._log(f"Step 2 work by traction mismatch: W_P = {self.W_P}, expected {expected_W_P}")
                return False
            # Work by resistance (depends on x, which is checked in step 3)
            expected_W_f = self.f * self.x
            if not self._is_close(self.W_f, expected_W_f):
                self._log(f"Step 2 work by resistance mismatch: W_f = {self.W_f}, expected {expected_W_f}")
                return False
            self._log("Step 2 energy and work values are consistent.")

            # Step 3: Distance traveled during acceleration
            # x = v*t - (m*v^3)/(2*P)
            if self.P == 0:
                self._log("Power P is zero, cannot compute expected distance.")
                return False
            expected_x = self.v * self.t - (self.m * self.v**3) / (2 * self.P)
            if not self._is_close(self.x, expected_x):
                self._log(f"Step 3 distance mismatch: x = {self.x}, expected {expected_x}")
                return False
            self._log("Step 3 distance formula holds.")

            # Step 4 & 5: Centripetal force
            if self.r == 0:
                self._log("Radius r is zero, cannot compute centripetal force from motion.")
                return False
            expected_F_n_motion = self.m * self.v**2 / self.r
            expected_F_n_banking = self.m * self.g * math.tan(self.theta)
            if not (self._is_close(self.F_n, expected_F_n_motion) and 
                    self._is_close(self.F_n, expected_F_n_banking)):
                self._log(f"Centripetal force mismatch: F_n = {self.F_n}, m*v^2/r = {expected_F_n_motion}, m*g*tanθ = {expected_F_n_banking}")
                return False
            self._log("Steps 4 & 5 centripetal force values are consistent.")

            # Step 6: Radius of the circular track
            if self.g * math.tan(self.theta) == 0:
                self._log("g*tan(theta) is zero, cannot compute expected radius.")
                return False
            expected_r = self.v**2 / (self.g * math.tan(self.theta))
            if not self._is_close(self.r, expected_r):
                self._log(f"Step 6 radius mismatch: r = {self.r}, expected {expected_r}")
                return False
            self._log("Step 6 radius formula holds.")

            # Final answers match target expressions
            if self._is_close(self.x, expected_x) and self._is_close(self.r, expected_r):
                self._log("Final answers match target expressions.")
                return True
            else:
                self._log("Final answers deviate from target expressions.")
                return False
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb mass m and verify expected trends in x and r."""
        try:
            # Store original mass
            original_m = self.m

            # Perturb: increase mass by 10%
            perturbed_m = original_m * 1.10

            # Recalculate x and r using the formulas (not the solution's methods)
            # x = v*t - (m*v^3)/(2*P)
            if self.P == 0:
                self._log("Power P is zero, cannot compute perturbed distance.")
                return False
            original_x = self.x
            perturbed_x = self.v * self.t - (perturbed_m * self.v**3) / (2 * self.P)

            # r = v^2/(g*tanθ) is independent of m, so should not change
            if self.g * math.tan(self.theta) == 0:
                self._log("g*tan(theta) is zero, cannot compute radius.")
                return False
            original_r = self.r
            perturbed_r = self.v**2 / (self.g * math.tan(self.theta))

            # Check trend for x: as m increases, x should decrease (since derivative w.r.t. m is negative)
            if perturbed_x < original_x:
                self._log(f"Monotonicity for x holds: increasing m from {original_m} to {perturbed_m} decreases x from {original_x} to {perturbed_x}.")
            else:
                self._log(f"Monotonicity for x violated: increasing m did not decrease x.")
                return False

            # Check that r remains unchanged (within tolerance)
            if self._is_close(original_r, perturbed_r):
                self._log(f"Radius r is independent of m: remains {original_r} after perturbation.")
            else:
                self._log(f"Radius r changed unexpectedly: from {original_r} to {perturbed_r}.")
                return False

            # Restore original mass (important!)
            self.m = original_m

            return True
        except Exception as e:
            # Restore mass on error
            self.m = original_m if 'original_m' in locals() else self.m
            self._log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs.clear()
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            self._log(f"\n--- Running {name} Test ---")
            try:
                if not test_func():
                    success = False
                    self._log(f"{name} test FAILED.")
                else:
                    self._log(f"{name} test PASSED.")
            except Exception as e:
                success = False
                self._log(f"{name} test ERROR: {e}\n{traceback.format_exc()}")

        self._log(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs