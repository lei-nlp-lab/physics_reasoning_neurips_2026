import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Do not directly access attributes; use getattr with fallbacks.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract physical parameters with safe defaults
        self.m = float(getattr(solution_instance, 'm', self.results.get('m', 1.0)))
        self.g = float(getattr(solution_instance, 'g', self.results.get('g', 9.8)))
        self.l = float(getattr(solution_instance, 'l', self.results.get('l', 1.0)))

        # Extract results with safe fallbacks
        self.W = float(getattr(solution_instance, 'W', self.results.get('W', 0.0)))
        self.F = float(getattr(solution_instance, 'F', self.results.get('F', 0.0)))
        self.v_A = float(getattr(solution_instance, 'v_A', self.results.get('v_A', 0.0)))
        self.v_B = float(getattr(solution_instance, 'v_B', self.results.get('v_B', 0.0)))

        # Compute angular velocity from v_A (if possible)
        if self.v_A != 0.0 and self.l != 0.0:
            self.omega = self.v_A / (2.0 * self.l)
        else:
            self.omega = 0.0

        # Store logs for test results
        self.logs = []

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((success, message))

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physics laws."""
        success = True

        # 1. Conservation of mechanical energy: loss in PE = gain in KE
        # Initial PE loss: mg*(2l) for A, gain: mg*l for B → net loss = m*g*l
        pe_loss = self.m * self.g * self.l
        ke_gain = 0.5 * self.m * (self.v_A**2 + self.v_B**2)
        if not self._is_close(pe_loss, ke_gain):
            self._log(f"Energy conservation failed: PE loss={pe_loss}, KE gain={ke_gain}", False)
            success = False
        else:
            self._log("Energy conservation satisfied.", True)

        # 2. Linear-angular velocity relationship: v_A = 2 * v_B (since v_A = 2lω, v_B = lω)
        # Use absolute values since direction doesn't matter for magnitude relationship
        if not self._is_close(abs(self.v_A), 2.0 * abs(self.v_B)):
            self._log(f"Velocity relation failed: |v_A|={abs(self.v_A)}, 2*|v_B|={2.0*abs(self.v_B)}", False)
            success = False
        else:
            self._log("Linear-angular velocity relation satisfied.", True)

        # 3. Work-energy theorem for ball A: mg*(2l) + W = 0.5*m*v_A^2
        work_on_A = self.m * self.g * 2.0 * self.l + self.W
        ke_A = 0.5 * self.m * self.v_A**2
        if not self._is_close(work_on_A, ke_A):
            self._log(f"Work-energy for A failed: work={work_on_A}, ΔKE={ke_A}", False)
            success = False
        else:
            self._log("Work-energy theorem for A satisfied.", True)

        # 4. Newton's second law (centripetal) for ball B: mg + F = m*v_B^2/l
        centripetal = self.m * self.v_B**2 / self.l
        net_force = self.m * self.g + self.F
        if not self._is_close(net_force, centripetal):
            self._log(f"Newton's second law for B failed: net force={net_force}, centripetal={centripetal}", False)
            success = False
        else:
            self._log("Newton's second law for B satisfied.", True)

        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True

        # Compute v_B from energy and v_A relation
        # From energy: 0.5*m*v_A^2 + 0.5*m*v_B^2 = m*g*l and v_A = 2*v_B
        # So: 0.5*m*(4*v_B^2 + v_B^2) = m*g*l → 2.5*m*v_B^2 = m*g*l → v_B^2 = (g*l)/2.5 = 0.4*g*l
        v_B_sq_expected = 0.4 * self.g * self.l
        if not self._is_close(self.v_B**2, v_B_sq_expected):
            self._log(f"v_B^2 inconsistent: expected {v_B_sq_expected}, got {self.v_B**2}", False)
            success = False
        else:
            self._log(f"v_B^2={self.v_B**2} consistent with energy and kinematics.", True)

        # Check W from work-energy theorem for A
        W_expected = 0.5 * self.m * self.v_A**2 - self.m * self.g * 2.0 * self.l
        if not self._is_close(self.W, W_expected):
            self._log(f"W inconsistent: expected {W_expected}, got {self.W}", False)
            success = False
        else:
            self._log(f"W={self.W} consistent with work-energy for A.", True)

        # Check F from centripetal equation for B
        F_expected = self.m * self.v_B**2 / self.l - self.m * self.g
        if not self._is_close(self.F, F_expected):
            self._log(f"F inconsistent: expected {F_expected}, got {self.F}", False)
            success = False
        else:
            self._log(f"F={self.F} consistent with centripetal force for B.", True)

        return success

    def test_monotonicity(self):
        """Perturb mass m and verify scaling trends."""
        original_m = self.m
        perturbation_factor = 1.1
        self.m *= perturbation_factor  # Perturb

        # Expected scaling: W ∝ m, F ∝ m
        W_expected = self.W * perturbation_factor
        F_expected = self.F * perturbation_factor

        # Recompute v_A and v_B from physics for the new mass (they should not change with m)
        # Because from energy: v_A^2 + v_B^2 = 2*g*l, independent of m
        v_B_sq = 0.4 * self.g * self.l
        v_B_new = math.sqrt(v_B_sq)
        v_A_new = 2.0 * v_B_new

        # Recompute W and F using the formulas
        W_new = 0.5 * self.m * v_A_new**2 - self.m * self.g * 2.0 * self.l
        F_new = self.m * v_B_new**2 / self.l - self.m * self.g

        # Check scaling
        success = True
        if not self._is_close(W_new, W_expected):
            self._log(f"Monotonicity (W) failed: expected {W_expected}, got {W_new}", False)
            success = False
        else:
            self._log(f"W scales linearly with m: {self.W/perturbation_factor} -> {W_new}", True)

        if not self._is_close(F_new, F_expected):
            self._log(f"Monotonicity (F) failed: expected {F_expected}, got {F_new}", False)
            success = False
        else:
            self._log(f"F scales linearly with m: {self.F/perturbation_factor} -> {F_new}", True)

        # Restore original mass
        self.m = original_m
        return success

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{name}' crashed: {traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs