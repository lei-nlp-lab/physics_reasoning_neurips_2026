import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dictionary
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract inputs with safe defaults (values from problem context)
        self.B = getattr(solution_instance, 'B', 0.5)  # T
        self.L = getattr(solution_instance, 'L', 0.3)  # m
        self.N = getattr(solution_instance, 'N', 200)  # turns
        self.r = getattr(solution_instance, 'r', 3.0)  # Ω
        self.omega = getattr(solution_instance, 'omega', 20 * math.sqrt(2))  # rad/s
        self.R = getattr(solution_instance, 'R', 4.0)  # Ω

        # Extract results with safe defaults
        self.E_m = self.results.get('E_m', 0.0)
        self.E = self.results.get('E', 0.0)
        self.R_parallel = self.results.get('R_parallel', 0.0)
        self.U = self.results.get('U', 0.0)
        self.DeltaPhi = self.results.get('DeltaPhi', 0.0)
        self.overlineE = self.results.get('overlineE', 0.0)
        self.overlineI = self.results.get('overlineI', 0.0)
        self.q = self.results.get('q', 0.0)
        self.T = self.results.get('T', 0.0)
        self.R_ext = self.results.get('R_ext', 0.0)
        self.I = self.results.get('I', 0.0)
        self.Q = self.results.get('Q', 0.0)

        self.logs = []

    def _log(self, msg, success=True):
        """Helper to log test results."""
        self.logs.append((msg, success))
        return success

    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        try:
            # 1. Faraday's Law: E_m = N * B * L^2 * omega
            expected_E_m = self.N * self.B * (self.L ** 2) * self.omega
            if not math.isclose(self.E_m, expected_E_m, rel_tol=0.02, abs_tol=0.02):
                logs.append(("Faraday's Law (max EMF) violated", False))
            else:
                logs.append(("Faraday's Law (max EMF) satisfied", True))

            # 2. RMS relationship: E = E_m / sqrt(2)
            expected_E = self.E_m / math.sqrt(2)
            if not math.isclose(self.E, expected_E, rel_tol=0.02, abs_tol=0.02):
                logs.append(("RMS voltage relationship violated", False))
            else:
                logs.append(("RMS voltage relationship satisfied", True))

            # 3. Voltage division: U/E = R_parallel/(R_parallel + R + r)
            total_R = self.R_parallel + self.R + self.r
            expected_U = self.E * self.R_parallel / total_R
            if not math.isclose(self.U, expected_U, rel_tol=0.02, abs_tol=0.02):
                logs.append(("Voltage division rule violated", False))
            else:
                logs.append(("Voltage division rule satisfied", True))

            # 4. Ohm's Law: U = I * R_parallel
            expected_I = self.U / self.R_parallel
            if not math.isclose(self.I, expected_I, rel_tol=0.02, abs_tol=0.02):
                logs.append(("Ohm's Law (U = I*R_parallel) violated", False))
            else:
                logs.append(("Ohm's Law (U = I*R_parallel) satisfied", True))

            # 5. Charge consistency: q = N * DeltaPhi / (R_parallel + R + r)
            expected_q = self.N * self.DeltaPhi / (self.R_parallel + self.R + self.r)
            if not math.isclose(self.q, expected_q, rel_tol=0.02, abs_tol=0.02):
                logs.append(("Charge consistency (Faraday + Ohm) violated", False))
            else:
                logs.append(("Charge consistency (Faraday + Ohm) satisfied", True))

            # 6. Joule's Law: Q = I^2 * R_ext * T
            expected_Q = (self.I ** 2) * self.R_ext * self.T
            if not math.isclose(self.Q, expected_Q, rel_tol=0.02, abs_tol=0.02):
                logs.append(("Joule's Law (heat generation) violated", False))
            else:
                logs.append(("Joule's Law (heat generation) satisfied", True))

        except Exception as e:
            logs.append((f"Error in invariant tests: {e}", False))
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that R_parallel = R/2
            expected_R_parallel = self.R / 2
            if not math.isclose(self.R_parallel, expected_R_parallel, rel_tol=0.01, abs_tol=0.01):
                logs.append(("R_parallel ≠ R/2", False))
            else:
                logs.append(("R_parallel = R/2 consistent", True))

            # Check that R_ext = R_parallel + R
            expected_R_ext = self.R_parallel + self.R
            if not math.isclose(self.R_ext, expected_R_ext, rel_tol=0.01, abs_tol=0.01):
                logs.append(("R_ext ≠ R_parallel + R", False))
            else:
                logs.append(("R_ext = R_parallel + R consistent", True))

            # Check that T = 2π/ω
            expected_T = 2 * math.pi / self.omega
            if not math.isclose(self.T, expected_T, rel_tol=0.02, abs_tol=0.02):
                logs.append(("T ≠ 2π/ω", False))
            else:
                logs.append(("T = 2π/ω consistent", True))

            # Check that DeltaPhi = B * L^2 * sin(30°)
            expected_DeltaPhi = self.B * (self.L ** 2) * math.sin(math.radians(30))
            if not math.isclose(self.DeltaPhi, expected_DeltaPhi, rel_tol=0.02, abs_tol=0.02):
                logs.append(("DeltaPhi ≠ B*L²*sin30°", False))
            else:
                logs.append(("DeltaPhi = B*L²*sin30° consistent", True))

            # Check that overlineE = N * DeltaPhi / (π/(6ω)) for quarter period
            # Time for 30° rotation: Δt = (π/6) / ω
            delta_t = (math.pi / 6) / self.omega
            expected_overlineE = self.N * self.DeltaPhi / delta_t
            # Use relative tolerance for average EMF
            if not math.isclose(self.overlineE, expected_overlineE, rel_tol=0.05, abs_tol=0.05):
                logs.append(("Average EMF inconsistent with flux change", False))
            else:
                logs.append(("Average EMF consistent with flux change", True))

        except Exception as e:
            logs.append((f"Error in trace consistency: {e}", False))
        return logs

    def test_monotonicity(self):
        """Perturb inputs and check output trends."""
        logs = []
        try:
            # Store original values
            original_B = self.B
            original_omega = self.omega
            original_R = self.R
            original_E_m = self.E_m
            original_E = self.E
            original_U = self.U
            original_I = self.I
            original_q = self.q
            original_T = self.T
            original_Q = self.Q

            # Test 1: Increase B by 1%, expect E_m, E, U, I, q, Q to increase
            self.B *= 1.01
            E_m_new = self.N * self.B * (self.L ** 2) * self.omega
            E_new = E_m_new / math.sqrt(2)
            R_parallel = self.R / 2
            total_R = R_parallel + self.R + self.r
            U_new = E_new * R_parallel / total_R
            I_new = U_new / R_parallel
            DeltaPhi_new = self.B * (self.L ** 2) * math.sin(math.radians(30))
            q_new = self.N * DeltaPhi_new / (R_parallel + self.R + self.r)
            T_new = 2 * math.pi / self.omega
            R_ext_new = R_parallel + self.R
            Q_new = (I_new ** 2) * R_ext_new * T_new

            if (E_m_new > original_E_m and E_new > original_E and U_new > original_U and 
                I_new > original_I and q_new > original_q and Q_new > original_Q):
                logs.append(("Monotonicity: B↑ → all outputs↑", True))
            else:
                logs.append(("Monotonicity: B↑ failed", False))

            # Restore B
            self.B = original_B

            # Test 2: Increase ω by 1%, expect E_m, E, U, I, Q to increase, T to decrease
            self.omega *= 1.01
            E_m_new = self.N * self.B * (self.L ** 2) * self.omega
            E_new = E_m_new / math.sqrt(2)
            R_parallel = self.R / 2
            total_R = R_parallel + self.R + self.r
            U_new = E_new * R_parallel / total_R
            I_new = U_new / R_parallel
            T_new = 2 * math.pi / self.omega
            R_ext_new = R_parallel + self.R
            Q_new = (I_new ** 2) * R_ext_new * T_new

            if (E_m_new > original_E_m and E_new > original_E and U_new > original_U and 
                I_new > original_I and T_new < original_T and Q_new > original_Q):
                logs.append(("Monotonicity: ω↑ → E_m,E,U,I,Q↑, T↓", True))
            else:
                logs.append(("Monotonicity: ω↑ failed", False))

            # Restore ω
            self.omega = original_omega

            # Test 3: Increase R by 1%, expect U to change according to voltage division
            # Note: E does NOT depend on R, so E remains constant
            self.R *= 1.01
            R_parallel = self.R / 2
            total_R = R_parallel + self.R + self.r
            U_new = original_E * R_parallel / total_R  # Use original_E, not self.E
            
            # For given parameters (R=4, r=3), let's compute the actual derivative
            # U = E * (R/2) / (R/2 + R + r) = E * R / (3R + 2r)
            # dU/dR = E * (3R + 2r - 3R) / (3R + 2r)^2 = E * 2r / (3R + 2r)^2 > 0
            # So U should INCREASE when R increases
            if U_new > original_U:
                logs.append(("Monotonicity: R↑ → U↑ (for given parameters)", True))
            else:
                logs.append(("Monotonicity: R↑ → U trend unexpected", False))

            # Restore R
            self.R = original_R

        except Exception as e:
            logs.append((f"Error in monotonicity tests: {e}", False))
        return logs

    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        self.logs = []
        all_passed = True

        # Run each test suite
        for name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                results = test_func()
                for msg, passed in results:
                    self.logs.append((f"{name}: {msg}", passed))
                    if not passed:
                        all_passed = False
            except Exception as e:
                self.logs.append((f"{name}: Exception - {traceback.format_exc()}", False))
                all_passed = False

        return all_passed, self.logs