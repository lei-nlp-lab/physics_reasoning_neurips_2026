import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        solution_instance: an object containing inputs and results.
        """
        # Extract inputs with safe access and unit conversion (cm -> m)
        self.m = getattr(solution_instance, 'm', 0.4)
        self.l0 = getattr(solution_instance, 'l0', 0.06)  # 6 cm -> 0.06 m
        self.l1 = getattr(solution_instance, 'l1', 0.10)  # 10 cm -> 0.10 m
        self.l2 = getattr(solution_instance, 'l2', 0.044) # 4.4 cm -> 0.044 m
        self.g = getattr(solution_instance, 'g', 10.0)

        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        self.F_N = results.get('F_N', 0.0)
        self.F_elastic = results.get('F_elastic', 0.0)
        self.F_elasticity = results.get('F_elasticity', 0.0)
        self.F_f = results.get('F_f', 0.0)
        self.F_elasticityprime = results.get('F_elasticityprime', 0.0)
        self.mu = results.get('mu', 0.0)
        self.k = results.get('k', 0.0)

        self.logs = []

    def _log(self, message, success):
        self.logs.append((message, success))

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify physical invariants: Hooke's law and force balance in both scenarios.
        """
        try:
            # 1. Static equilibrium in Figure A: spring force equals weight
            weight = self.m * self.g
            if not self._is_close(self.F_elasticity, weight):
                self._log(f"Static equilibrium violated: F_elasticity={self.F_elasticity} vs weight={weight}", False)
                return False
            else:
                self._log("Static equilibrium in Figure A holds.", True)

            # 2. Hooke's law for spring constant k
            delta_x_stretch = self.l1 - self.l0
            F_elastic_expected = self.k * delta_x_stretch
            if not self._is_close(self.F_elastic, F_elastic_expected):
                self._log(f"Hooke's law (stretch) violated: F_elastic={self.F_elastic} vs k*Δx={F_elastic_expected}", False)
                return False
            else:
                self._log("Hooke's law (stretch) holds.", True)

            # 3. Dynamic equilibrium in Figure B: spring force equals friction force
            delta_x_compress = self.l0 - self.l2
            F_elasticprime_expected = self.k * delta_x_compress
            if not (self._is_close(self.F_elasticityprime, F_elasticprime_expected) and
                    self._is_close(self.F_f, F_elasticprime_expected)):
                self._log(f"Dynamic equilibrium violated: F_elasticity'={self.F_elasticityprime}, F_f={self.F_f} vs k*Δx'={F_elasticprime_expected}", False)
                return False
            else:
                self._log("Dynamic equilibrium in Figure B holds.", True)

            # 4. Normal force equals weight
            if not self._is_close(self.F_N, weight):
                self._log(f"Normal force mismatch: F_N={self.F_N} vs weight={weight}", False)
                return False
            else:
                self._log("Normal force equals weight.", True)

            # 5. Definition of coefficient of kinetic friction
            if self.F_N != 0:
                mu_expected = self.F_f / self.F_N
                if not self._is_close(self.mu, mu_expected):
                    self._log(f"Coefficient of friction mismatch: μ={self.mu} vs F_f/F_N={mu_expected}", False)
                    return False
                else:
                    self._log("Coefficient of friction definition holds.", True)
            else:
                self._log("Normal force is zero, cannot compute μ.", False)
                return False

            return True
        except Exception as e:
            self._log(f"Exception in test_invariants: {traceback.format_exc()}", False)
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        try:
            # Check that k computed from static equilibrium matches the stored k
            delta_x_stretch = self.l1 - self.l0
            k_expected = (self.m * self.g) / delta_x_stretch if delta_x_stretch != 0 else 0
            if not self._is_close(self.k, k_expected):
                self._log(f"Spring constant inconsistency: stored k={self.k} vs computed k={k_expected}", False)
                return False
            else:
                self._log("Spring constant consistent with static equilibrium.", True)

            # Check that F_f computed from dynamic equilibrium matches stored F_f
            delta_x_compress = self.l0 - self.l2
            F_f_expected = self.k * delta_x_compress
            if not self._is_close(self.F_f, F_f_expected):
                self._log(f"Friction force inconsistency: stored F_f={self.F_f} vs computed F_f={F_f_expected}", False)
                return False
            else:
                self._log("Friction force consistent with dynamic equilibrium.", True)

            # Check that μ computed from F_f and F_N matches stored μ
            mu_expected = self.F_f / (self.m * self.g) if (self.m * self.g) != 0 else 0
            if not self._is_close(self.mu, mu_expected):
                self._log(f"Friction coefficient inconsistency: stored μ={self.mu} vs computed μ={mu_expected}", False)
                return False
            else:
                self._log("Friction coefficient consistent with forces.", True)

            return True
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {traceback.format_exc()}", False)
            return False

    def test_monotonicity(self):
        """
        Perturb mass m and check expected trends.
        """
        original_m = self.m
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1

            # Recompute k and μ using formulas (without modifying instance)
            delta_x_stretch = self.l1 - self.l0
            perturbed_k = (perturbed_m * self.g) / delta_x_stretch if delta_x_stretch != 0 else 0
            delta_x_compress = self.l0 - self.l2
            perturbed_F_f = perturbed_k * delta_x_compress
            perturbed_mu = perturbed_F_f / (perturbed_m * self.g) if (perturbed_m * self.g) != 0 else 0

            # Check trend: k should increase with m (since k ∝ m)
            if not perturbed_k > self.k:
                self._log(f"Monotonicity violated: k does not increase with m (k={self.k}, perturbed_k={perturbed_k})", False)
                return False
            else:
                self._log(f"k increases with m (k={self.k} -> {perturbed_k}).", True)

            # μ should remain constant (since μ = (l0-l2)/(l1-l0) independent of m)
            if not self._is_close(self.mu, perturbed_mu):
                self._log(f"Monotonicity violated: μ changes with m (μ={self.mu}, perturbed_μ={perturbed_mu})", False)
                return False
            else:
                self._log(f"μ remains constant with m (μ={self.mu}).", True)

            return True
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {traceback.format_exc()}", False)
            return False
        finally:
            # Restore original mass (though we didn't modify instance, just for safety)
            self.m = original_m

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        success = True

        tests = [
            ("test_invariants", self.test_invariants),
            ("test_trace_consistency", self.test_trace_consistency),
            ("test_monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            test_success = test_func()
            if not test_success:
                success = False
            self._log(f"Test {name}: {'PASS' if test_success else 'FAIL'}", test_success)

        return success, self.logs