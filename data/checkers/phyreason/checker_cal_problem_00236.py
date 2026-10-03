import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        self.solution = solution_instance
        self.logs = []

        # Extract inputs with safe defaults
        self.theta = math.radians(30)  # Given as 30 degrees
        self.B = getattr(solution_instance, 'B', 0.5)
        self.D = getattr(solution_instance, 'D', 0.4)
        self.m = getattr(solution_instance, 'm', 0.1)
        self.R = getattr(solution_instance, 'R', 0.25)
        self.mu = getattr(solution_instance, 'mu', math.sqrt(3)/3)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.ef = getattr(solution_instance, 'ef', 0.4)

        # Extract results from solution_instance.results or attributes
        res = getattr(solution_instance, 'results', {})
        self.F = getattr(solution_instance, 'F', res.get('F', 0.0))
        self.L = getattr(solution_instance, 'L', res.get('L', 0.0))
        self.Q = getattr(solution_instance, 'Q', res.get('Q', 0.0))
        self.N = getattr(solution_instance, 'N', res.get('N', 0.0))
        self.f = getattr(solution_instance, 'f', res.get('f', 0.0))
        self.a = getattr(solution_instance, 'a', res.get('a', 0.0))
        self.v = getattr(solution_instance, 'v', res.get('v', 0.0))
        self.I = getattr(solution_instance, 'I', res.get('I', 0.0))
        self.E = getattr(solution_instance, 'E', res.get('E', 0.0))
        self.t = getattr(solution_instance, 't', res.get('t', 0.0))
        self.F_A = getattr(solution_instance, 'F_A', res.get('F_A', 0.0))

        # Target answers from problem statement
        self.target_F = 1.5
        self.target_L = 0.5
        self.target_Q = 0.4

    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify physical invariants using KB theorems."""
        successes = []
        try:
            # 1. Normal force check (KB Theorem: Force analysis of object on an inclined plane)
            N_calc = self.m * self.g * math.cos(self.theta)
            if self._is_close(self.N, N_calc):
                self._log(f"Normal force: {self.N} ≈ {N_calc} (calculated)", True)
                successes.append(True)
            else:
                self._log(f"Normal force mismatch: {self.N} vs {N_calc}", False)
                successes.append(False)

            # 2. Friction force check (KB Theorem: Friction Force Law)
            f_calc = self.mu * self.N
            if self._is_close(self.f, f_calc):
                self._log(f"Friction force: {self.f} ≈ {f_calc} (calculated)", True)
                successes.append(True)
            else:
                self._log(f"Friction force mismatch: {self.f} vs {f_calc}", False)
                successes.append(False)

            # 3. Newton's 2nd Law during acceleration phase (0-0.4s) from Step 4-5
            # According to steps: F - f - m*g*sin(theta) = m*a (up the incline)
            a_given = 5.0  # From step 4
            F_net_calc = self.m * a_given
            F_required = F_net_calc + self.f + self.m * self.g * math.sin(self.theta)
            if self._is_close(self.F, F_required):
                self._log(f"External force F: {self.F} ≈ {F_required} (Newton's 2nd Law check)", True)
                successes.append(True)
            else:
                self._log(f"Newton's 2nd Law mismatch for F: {self.F} vs {F_required}", False)
                successes.append(False)

            # 4. Force balance during uniform motion (CORRECTED)
            # When moving uniformly at v=2.0 m/s: F - m*g*sin(theta) - f - F_A = 0
            # So F_A = F - m*g*sin(theta) - f
            v_uniform = 2.0  # From step 12
            F_gravity_parallel = self.m * self.g * math.sin(self.theta)
            F_A_required = self.F - F_gravity_parallel - self.f
            if self._is_close(self.F_A, F_A_required):
                self._log(f"Ampere force F_A: {self.F_A} ≈ F - mg sinθ - f = {F_A_required} (force balance)", True)
                successes.append(True)
            else:
                self._log(f"Force balance mismatch for F_A: {self.F_A} vs {F_A_required}", False)
                successes.append(False)

            # 5. Magnetic force relation (KB Theorem: Ampere Force Law)
            # F_A = B*I*L
            if self.I > 0 and self.L > 0:
                F_A_calc = self.B * self.I * self.L
                if self._is_close(self.F_A, F_A_calc):
                    self._log(f"Ampere force: {self.F_A} ≈ {self.B}*{self.I}*{self.L} = {F_A_calc}", True)
                    successes.append(True)
                else:
                    self._log(f"Ampere force relation mismatch: {self.F_A} vs {F_A_calc}", False)
                    successes.append(False)

            # 6. Ohm's Law and Motional EMF (Steps 10-11)
            # E = B*L*v, I = E/R
            if self.v > 0 and self.L > 0:
                E_calc = self.B * self.L * self.v
                I_calc = E_calc / self.R
                if self._is_close(self.E, E_calc) and self._is_close(self.I, I_calc):
                    self._log(f"Ohm's Law & EMF: E={self.E}≈{E_calc}, I={self.I}≈{I_calc}", True)
                    successes.append(True)
                else:
                    self._log(f"EMF/Ohm's Law mismatch: E={self.E} vs {E_calc}, I={self.I} vs {I_calc}", False)
                    successes.append(False)

            # 7. Joule heat check (Step 18)
            t_entry = self.ef / self.v if self.v > 0 else 0.0  # Time to cross magnetic field
            Q_calc = self.I**2 * self.R * (2 * t_entry)  # Entry and exit
            if self._is_close(self.Q, Q_calc):
                self._log(f"Joule heat: {self.Q} ≈ I²R(2t) = {Q_calc}", True)
                successes.append(True)
            else:
                self._log(f"Joule heat mismatch: {self.Q} vs {Q_calc}", False)
                successes.append(False)

        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            successes.append(False)

        return all(successes)

    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        successes = []
        try:
            # Step 4: acceleration a = 5 m/s² should be consistent with velocity change
            # From graph description: 0-0.4s acceleration phase
            a_calc = self.a
            if self._is_close(a_calc, 5.0):
                self._log(f"Acceleration a={a_calc} matches given 5.0 m/s²", True)
                successes.append(True)
            else:
                self._log(f"Acceleration mismatch: {a_calc} vs 5.0", False)
                successes.append(False)

            # Step 12: uniform velocity v = 2.0 m/s
            if self._is_close(self.v, 2.0):
                self._log(f"Uniform velocity v={self.v} matches given 2.0 m/s", True)
                successes.append(True)
            else:
                self._log(f"Velocity mismatch: {self.v} vs 2.0", False)
                successes.append(False)

            # Step 13: L calculated from force balance should match target
            if self._is_close(self.L, self.target_L):
                self._log(f"Length L={self.L} matches target {self.target_L}", True)
                successes.append(True)
            else:
                self._log(f"Length L mismatch: {self.L} vs {self.target_L}", False)
                successes.append(False)

            # Step 16: current I = 2.0 A
            if self._is_close(self.I, 2.0):
                self._log(f"Current I={self.I} matches expected 2.0 A", True)
                successes.append(True)
            else:
                self._log(f"Current mismatch: {self.I} vs 2.0", False)
                successes.append(False)

            # Step 17: time t = ef/v = 0.4/2.0 = 0.2 s
            t_calc = self.ef / self.v if self.v > 0 else 0.0
            if self._is_close(self.t, t_calc):
                self._log(f"Time t={self.t} ≈ ef/v = {t_calc}", True)
                successes.append(True)
            else:
                self._log(f"Time mismatch: {self.t} vs {t_calc}", False)
                successes.append(False)

            # Final answers against targets (use slightly larger tolerance for F due to rounding)
            if self._is_close(self.F, self.target_F, rel_tol=2e-2, abs_tol=2e-2):
                self._log(f"External force F={self.F} matches target {self.target_F}", True)
                successes.append(True)
            else:
                self._log(f"External force mismatch: {self.F} vs {self.target_F}", False)
                successes.append(False)

            if self._is_close(self.Q, self.target_Q):
                self._log(f"Joule heat Q={self.Q} matches target {self.target_Q}", True)
                successes.append(True)
            else:
                self._log(f"Joule heat mismatch: {self.Q} vs {self.target_Q}", False)
                successes.append(False)

        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            successes.append(False)

        return all(successes)

    def test_monotonicity(self):
        """Perturb mass m and check trends in F, L, Q."""
        successes = []
        original_m = self.m
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            self._log(f"Testing monotonicity: perturbing mass from {original_m} to {perturbed_m} kg")

            # Recalculate intermediate quantities using formulas from steps
            # 1. Normal force
            N_pert = perturbed_m * self.g * math.cos(self.theta)
            # 2. Friction force
            f_pert = self.mu * N_pert
            # 3. External force F (during acceleration, a=5 fixed)
            a_given = 5.0
            F_pert = perturbed_m * a_given + f_pert + perturbed_m * self.g * math.sin(self.theta)
            # 4. Length L from force balance during uniform motion (v=2 fixed)
            v_uniform = 2.0
            F_gravity_parallel_pert = perturbed_m * self.g * math.sin(self.theta)
            # Correct force balance: F_A = F - mg sinθ - f
            F_A_required_pert = F_pert - F_gravity_parallel_pert - f_pert
            # From F_A = B²L²v/R
            L_pert = math.sqrt(F_A_required_pert * self.R / (self.B**2 * v_uniform))
            # 5. Time to cross magnetic field
            t_pert = self.ef / v_uniform
            # 6. Current I = B*L*v/R
            I_pert = self.B * L_pert * v_uniform / self.R
            # 7. Joule heat Q = I²R(2t)
            Q_pert = I_pert**2 * self.R * (2 * t_pert)

            # Check trends: all should increase with increased mass
            F_increased = F_pert > self.F
            L_increased = L_pert > self.L
            Q_increased = Q_pert > self.Q

            if F_increased:
                self._log(f"F increases from {self.F} to {F_pert} with increased mass", True)
                successes.append(True)
            else:
                self._log(f"F did not increase: {self.F} -> {F_pert}", False)
                successes.append(False)

            if L_increased:
                self._log(f"L increases from {self.L} to {L_pert} with increased mass", True)
                successes.append(True)
            else:
                self._log(f"L did not increase: {self.L} -> {L_pert}", False)
                successes.append(False)

            if Q_increased:
                self._log(f"Q increases from {self.Q} to {Q_pert} with increased mass", True)
                successes.append(True)
            else:
                self._log(f"Q did not increase: {self.Q} -> {Q_pert}", False)
                successes.append(False)

        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            successes.append(False)
        finally:
            # Restore original mass
            self.m = original_m

        return all(successes)

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            self._log(f"--- Starting {name} Test ---")
            try:
                success = test_func()
                overall_success = overall_success and success
                self._log(f"{name} test {'passed' if success else 'failed'}", success)
            except Exception as e:
                self._log(f"{name} test crashed: {traceback.format_exc()}", False)
                overall_success = False

        self._log(f"=== Overall Test Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs