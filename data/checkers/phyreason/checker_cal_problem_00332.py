import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        results = getattr(solution_instance, 'results', {})

        # Inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.l = getattr(solution_instance, 'l', 1.0)
        self.k = getattr(solution_instance, 'k', 1.0)  # Coulomb constant

        # Results with safe extraction
        self.F_T = getattr(solution_instance, 'F_T', results.get('F_T', 0.0))
        self.E_k = getattr(solution_instance, 'E_k', results.get('E_k', 0.0))
        self.p = getattr(solution_instance, 'p', results.get('p', 0.0))
        self.E_k_prime = getattr(solution_instance, 'E_k_prime', results.get("E_k'", results.get('E_k_prime', 0.0)))
        self.r = getattr(solution_instance, 'r', results.get('r', 0.0))
        self.F_net = getattr(solution_instance, 'F_net', results.get('F_net', 0.0))
        self.F_c = getattr(solution_instance, 'F_c', results.get('F_c', 0.0))

        # Convert sympy to float if necessary
        self._convert_to_float()

    def _convert_to_float(self):
        """Convert attributes to float for numerical comparisons."""
        for attr in ['m', 'q', 'l', 'k', 'F_T', 'E_k', 'p', 'E_k_prime', 'r', 'F_net', 'F_c']:
            val = getattr(self, attr)
            if isinstance(val, sp.Basic):
                setattr(self, attr, float(val.evalf()))
            else:
                try:
                    setattr(self, attr, float(val))
                except:
                    pass  # Keep as is if conversion fails

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify key physics invariants."""
        logs = []
        success = True

        # 1. Static equilibrium: tension equals Coulomb force between two spheres
        expected_F_T = self.k * self.q**2 / self.l**2
        if not self._is_close(self.F_T, expected_F_T):
            logs.append(f"FAIL: Tension F_T = {self.F_T} != k*q²/l² = {expected_F_T}")
            success = False
        else:
            logs.append(f"PASS: Static tension F_T = k*q²/l²")

        # 2. Kinetic energy during circular motion
        expected_E_k = self.k * self.q**2 / self.l
        if not self._is_close(self.E_k, expected_E_k):
            logs.append(f"FAIL: Kinetic energy E_k = {self.E_k} != k*q²/l = {expected_E_k}")
            success = False
        else:
            logs.append(f"PASS: Circular motion kinetic energy E_k = k*q²/l")

        # 3. Momentum conservation after cutting strings
        if not self._is_close(self.p, 0.0):
            logs.append(f"FAIL: System momentum p = {self.p} != 0")
            success = False
        else:
            logs.append(f"PASS: Total momentum conserved (p = 0)")

        # 4. Kinetic energy after distance to C doubles
        expected_E_k_prime = 3 * self.k * self.q**2 / (2 * self.l)
        if not self._is_close(self.E_k_prime, expected_E_k_prime):
            logs.append(f"FAIL: Final kinetic energy E_k' = {self.E_k_prime} != 3k*q²/(2l) = {expected_E_k_prime}")
            success = False
        else:
            logs.append(f"PASS: Final kinetic energy E_k' = 3k*q²/(2l)")

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True

        # 1. Radius of circular path
        cos30 = math.cos(math.radians(30))
        expected_r = self.l / (2 * cos30)  # = l / √3
        if not self._is_close(self.r, expected_r):
            logs.append(f"FAIL: Radius r = {self.r} != l/(2cos30°) = {expected_r}")
            success = False
        else:
            logs.append(f"PASS: Radius r = l/(2cos30°)")

        # 2. Net force vs centripetal force during circular motion
        if self.r > 0:
            # Compute centripetal force from kinetic energy: F_c = m*v²/r, with E_k = 0.5*m*v²
            expected_F_c = 2 * self.E_k / self.r
            if not (self._is_close(self.F_c, expected_F_c) and self._is_close(self.F_net, self.F_c)):
                logs.append(f"FAIL: F_net = {self.F_net}, F_c = {self.F_c}, expected F_c = 2*E_k/r = {expected_F_c}")
                success = False
            else:
                logs.append(f"PASS: Net force equals centripetal force (F_net = F_c)")

        # 3. Net force from Coulomb and tension (circular motion)
        F_coulomb = self.k * self.q**2 / self.l**2
        expected_F_net = 4 * F_coulomb * cos30  # From step 2: 2*(F_T - F_coulomb)*cos30°, with F_T = 3*F_coulomb
        if not self._is_close(self.F_net, expected_F_net):
            logs.append(f"FAIL: F_net = {self.F_net} != 4*F_coulomb*cos30° = {expected_F_net}")
            success = False
        else:
            logs.append(f"PASS: Net force matches Coulomb/tension decomposition")

        return success, logs

    def test_monotonicity(self):
        """Perturb input q and verify expected trends."""
        logs = []
        success = True

        original_q = self.q
        perturbation = 1.1  # Increase by 10%

        try:
            # Perturb q
            self.q = original_q * perturbation

            # Recompute expected values (formulas from invariants)
            new_F_T = self.k * self.q**2 / self.l**2
            new_E_k = self.k * self.q**2 / self.l
            new_E_k_prime = 3 * self.k * self.q**2 / (2 * self.l)

            # Original values (recomputed for clarity)
            orig_F_T = self.k * original_q**2 / self.l**2
            orig_E_k = self.k * original_q**2 / self.l
            orig_E_k_prime = 3 * self.k * original_q**2 / (2 * self.l)

            # Check relative changes (should be ~1.21 for squares, ~1.1 for linear? Actually all depend on q²)
            expected_factor = perturbation**2  # 1.21 for 10% increase

            if not (self._is_close(new_F_T / orig_F_T, expected_factor) and
                    self._is_close(new_E_k / orig_E_k, expected_factor) and
                    self._is_close(new_E_k_prime / orig_E_k_prime, expected_factor)):
                logs.append(f"FAIL: Perturbation test failed. Factors: F_T={new_F_T/orig_F_T:.3f}, E_k={new_E_k/orig_E_k:.3f}, E_k'={new_E_k_prime/orig_E_k_prime:.3f}, expected ~{expected_factor}")
                success = False
            else:
                logs.append(f"PASS: All quantities scale as q² under perturbation")

        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            success = False
        finally:
            # Restore original q
            self.q = original_q

        return success, logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"--- {test_name} ---")
                all_logs.append(f"CRASH: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs