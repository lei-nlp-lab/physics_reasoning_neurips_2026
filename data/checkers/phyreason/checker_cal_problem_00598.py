import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.alpha = getattr(solution_instance, 'alpha', solution_instance.results.get('alpha', 0.0))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', 0.0))

        # Results
        self.N = getattr(solution_instance, 'N', solution_instance.results.get('N', 0.0))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', 0.0))
        self.mu = getattr(solution_instance, 'mu', solution_instance.results.get('mu', 0.0))

        # Store for restoration after perturbations
        self._original_m = self.m

    def test_invariants(self):
        """
        Verify static equilibrium and force-balance invariants.
        """
        logs = []
        success = True

        try:
            # Check normal force equilibrium: N = m*g*cos(alpha)
            expected_N = self.m * self.g * math.cos(self.alpha)
            if not math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Normal force invariant fails: N={self.N:.4f}, expected {expected_N:.4f}")
                success = False

            # Check friction force equilibrium: f = m*g*sin(alpha)
            expected_f = self.m * self.g * math.sin(self.alpha)
            if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Friction force invariant fails: f={self.f:.4f}, expected {expected_f:.4f}")
                success = False

            # Check friction coefficient: μ = tan(θ)
            expected_mu = math.tan(self.theta)
            if not math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Friction coefficient invariant fails: μ={self.mu:.4f}, expected {expected_mu:.4f}")
                success = False

        except Exception as e:
            logs.append(f"Error in invariant tests: {e}\n{traceback.format_exc()}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True

        try:
            # Check that N and f are consistent with the given formulas
            # using the extracted inputs.
            expected_N = self.m * self.g * math.cos(self.alpha)
            expected_f = self.m * self.g * math.sin(self.alpha)
            expected_mu = math.tan(self.theta)

            # Compare with stored results
            if not (math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append("Trace consistency failed: intermediate results do not match final answers.")
                success = False

        except Exception as e:
            logs.append(f"Error in trace consistency test: {e}\n{traceback.format_exc()}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """
        Perturb mass m and verify linear scaling of forces.
        """
        logs = []
        success = True

        try:
            # Perturb mass by +10%
            perturbed_m = self.m * 1.1

            # Re-calculate expected forces using formulas
            expected_N_perturbed = perturbed_m * self.g * math.cos(self.alpha)
            expected_f_perturbed = perturbed_m * self.g * math.sin(self.alpha)

            # Compute scaling factors
            # Avoid division by zero
            denom_N = self.m * self.g * math.cos(self.alpha)
            denom_f = self.m * self.g * math.sin(self.alpha)
            
            if abs(denom_N) < 1e-12:
                logs.append("N scaling skipped due to zero denominator (cos(alpha) ≈ 0).")
            else:
                N_scale = expected_N_perturbed / denom_N
                if not math.isclose(N_scale, 1.1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"N scaling monotonicity fails: scale={N_scale:.4f}, expected 1.1")
                    success = False
            
            if abs(denom_f) < 1e-12:
                logs.append("f scaling skipped due to zero denominator (sin(alpha) ≈ 0).")
            else:
                f_scale = expected_f_perturbed / denom_f
                if not math.isclose(f_scale, 1.1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"f scaling monotonicity fails: scale={f_scale:.4f}, expected 1.1")
                    success = False

        except Exception as e:
            logs.append(f"Error in monotonicity test: {e}\n{traceback.format_exc()}")
            success = False

        # Restore original mass
        self.m = self._original_m

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return a summary.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                if not success:
                    overall_success = False
                    all_logs.append(f"[FAIL] {test_name}: {logs}")
                else:
                    all_logs.append(f"[PASS] {test_name}")
            except Exception as e:
                overall_success = False
                all_logs.append(f"[ERROR] {test_name}: {e}\n{traceback.format_exc()}")

        return overall_success, all_logs