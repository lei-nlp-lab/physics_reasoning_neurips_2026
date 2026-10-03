import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        
        # Primary result: transition probability to n=2 states (only 2p_0 contributes)
        self.P = self.results.get('P', None)
        
        # Expected givens (use None defaults)
        self.E0 = self.givens.get('E0', None)
        self.Gamma = self.givens.get('Gamma', None)
        self.omega21 = self.givens.get('omega21', None)
        self.a0 = self.givens.get('a0', None)
        self.e = self.givens.get('e', None)
        self.hbar = self.givens.get('hbar', None)
        
        # Optional intermediate: matrix element <210|z|100>
        self.M210 = self.results.get('M210', None)

    def _compute_matrix_element(self):
        """Compute <210|z|100> using analytic integral.
        Returns M in units of a0. Requires self.a0.
        """
        if self.a0 is None:
            return None
        # Radial integral: ∫_0^∞ r^4 exp(-3r/(2a0)) dr = 4! / (3/(2a0))^5 = 24 * (32 a0^5/243) = 256/81 a0^5
        radial = 256.0 / 81.0 * self.a0**5
        # Angular integral: ∫_0^π cos²θ sinθ dθ = 2/3
        angular = 2.0 / 3.0
        # φ integral: 2π
        phi = 2.0 * math.pi
        # Prefactor: 1/(4√2 π a0^4)  (from ψ210* ψ100 with r cosθ included)
        prefactor = 1.0 / (4.0 * math.sqrt(2.0) * math.pi * self.a0**4)
        M = prefactor * phi * angular * radial
        return M

    def _compute_P_from_givens(self):
        """Compute P using first-order TDPT formula.
        Returns None if any required given is missing.
        """
        if any(v is None for v in [self.E0, self.Gamma, self.omega21, self.a0, self.e, self.hbar]):
            return None
        M = self._compute_matrix_element()
        if M is None:
            return None
        numerator = self.e**2 * self.E0**2 * M**2
        denominator = self.hbar**2 * (self.Gamma**2 + self.omega21**2)
        return numerator / denominator

    # ---------- Interface tests ----------
    def test_interface_P_exists(self):
        if self.P is None:
            return (False, "P not found in results['P']")
        return (True, "P key exists")

    def test_interface_P_numeric(self):
        if self.P is None:
            return (False, "SKIP: P missing")
        if not isinstance(self.P, (int, float)):
            return (False, f"P is not numeric, got {type(self.P)}")
        return (True, "P is numeric")

    # ---------- Invariant tests (physical sanity, use isclose) ----------
    def test_invariant_P_nonnegative(self):
        if self.P is None:
            return (False, "SKIP: P missing")
        if self.P < 0:
            return (False, f"P is negative: {self.P}")
        return (True, "P >= 0")

    def test_invariant_P_finite(self):
        if self.P is None:
            return (False, "SKIP: P missing")
        if math.isinf(self.P) or math.isnan(self.P):
            return (False, "P is inf or NaN")
        return (True, "P is finite")

    def test_invariant_omega21_from_energy(self):
        # Check that given omega21 matches the hydrogen energy difference
        if self.omega21 is None or self.e is None or self.a0 is None or self.hbar is None:
            return (False, "SKIP: missing omega21, e, a0, or hbar")
        # E_n = -e^2/(2 a0 n^2) for hydrogen
        # ΔE = E_2 - E_1 = e^2/(2a0)*(1 - 1/4) = 3e^2/(8a0)
        expected_omega = 3.0 * self.e**2 / (8.0 * self.a0 * self.hbar)
        if math.isclose(self.omega21, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "omega21 matches hydrogen energy difference")
        return (False, f"omega21 {self.omega21} != expected {expected_omega}")

    def test_invariant_selection_rule_only_210(self):
        # If other n=2 state probabilities are available, check they are zero (or negligibly small)
        P200 = self.results.get('P_200', None)
        P21m1 = self.results.get('P_21m1', None)
        P21p1 = self.results.get('P_21p1', None)
        all_present = all(v is not None for v in [P200, P21m1, P21p1])
        if not all_present:
            return (False, "SKIP: intermediate probabilities not provided")
        # For dipole selection rules, these should be zero
        check = math.isclose(P200, 0.0, abs_tol=1e-12) and math.isclose(P21m1, 0.0, abs_tol=1e-12) and math.isclose(P21p1, 0.0, abs_tol=1e-12)
        if check:
            return (True, "Only 210 state contributes, others have zero probability")
        return (False, "Non-zero probabilities found for forbidden transitions")

    # ---------- Modeling / trace tests (use isclose) ----------
    def test_modeling_matrix_element_value(self):
        # Check computed matrix element M210 (if available) against analytic formula
        if self.M210 is None or self.a0 is None:
            return (False, "SKIP: M210 or a0 missing")
        expected_M = (128.0 * math.sqrt(2.0) / 243.0) * self.a0
        if math.isclose(self.M210, expected_M, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "M210 matches analytic value (128√2/243)a0")
        return (False, f"M210 {self.M210} != expected {expected_M}")

    def test_modeling_P_from_TDPT(self):
        # Re‑derive P from givens using first‑order TDPT and compare with stored P
        if self.P is None:
            return (False, "SKIP: P missing")
        computed_P = self._compute_P_from_givens()
        if computed_P is None:
            return (False, "SKIP: missing givens to recompute P")
        if math.isclose(self.P, computed_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P matches re‑computed TDPT value {computed_P}")
        return (False, f"P {self.P} differs from computed {computed_P}")

    def test_trace_time_integral_amplitude(self):
        # If intermediate amplitude c_inf is present, check |c_inf|^2 = P
        c_inf = self.results.get('c_inf', None)
        if c_inf is None:
            return (False, "SKIP: c_inf not provided")
        if self.P is None:
            return (False, "SKIP: P missing")
        prob_from_amp = abs(c_inf)**2
        if math.isclose(prob_from_amp, self.P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "|c_inf|^2 matches P")
        return (False, f"|c_inf|^2 = {prob_from_amp} != P = {self.P}")

    # ---------- Monotonicity tests (perturb input, check trend) ----------
    def test_monotonicity_P_decreases_with_Gamma(self):
        # Increase Gamma, P should decrease (denominator grows)
        if self.P is None:
            return (False, "SKIP: P missing")
        base_P = self._compute_P_from_givens()
        if base_P is None:
            return (False, "SKIP: cannot compute baseline P")
        # Store original Gamma, perturb
        orig_Gamma = self.Gamma
        perturbed_Gamma = orig_Gamma * 1.1  # 10% increase
        # Temporarily change attribute for computation
        self.Gamma = perturbed_Gamma
        new_P = self._compute_P_from_givens()
        self.Gamma = orig_Gamma  # restore
        if new_P is None:
            return (False, "SKIP: failed to compute perturbed P")
        if new_P < base_P:
            return (True, f"P decreases when Gamma increases: {new_P} < {base_P}")
        return (False, f"P did not decrease: new_P={new_P} >= base_P={base_P}")

    def test_monotonicity_P_increases_with_E0(self):
        # Increase E0, P should increase (proportional to E0^2)
        if self.P is None:
            return (False, "SKIP: P missing")
        base_P = self._compute_P_from_givens()
        if base_P is None:
            return (False, "SKIP: cannot compute baseline P")
        orig_E0 = self.E0
        perturbed_E0 = orig_E0 * 1.1
        self.E0 = perturbed_E0
        new_P = self._compute_P_from_givens()
        self.E0 = orig_E0
        if new_P is None:
            return (False, "SKIP: failed to compute perturbed P")
        if new_P > base_P:
            return (True, f"P increases when E0 increases: {new_P} > {base_P}")
        return (False, f"P did not increase: new_P={new_P} <= base_P={base_P}")

    # ---------- Answer sanity (weak, no isclose, no expected values) ----------
    def test_answer_sanity_weak_probability_range(self):
        # Single weak sanity: P must be in [0,1] and not astronomically large
        if self.P is None:
            return (False, "SKIP: P missing")
        if not (0.0 <= self.P <= 1.0):
            return (False, f"P = {self.P} is outside [0,1]")
        if self.P > 1e10:   # completely unrealistic upper bound for probability
            return (False, f"P = {self.P} is astronomically large")
        return (True, "P is a valid probability (0 ≤ P ≤ 1)")

    # ---------- Runner ----------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)