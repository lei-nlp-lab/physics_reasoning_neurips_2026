import math

class PhysicsChecker:
    """
    Physics auditor for the eta^0 problem.
    Checks conservation laws, inference logic, and modeling consistency.
    """

    # Known particle properties (standard model constants)
    J_p = 0.5  # proton spin
    J_n = 0.5  # neutron spin
    P_pi = -1  # pion parity
    P_p = +1   # proton parity
    P_n = +1   # neutron parity
    I_pi = 1   # pion isospin
    I3_pi = -1 # third component
    I_p = 0.5  # proton isospin
    I3_p = +0.5
    I_n = 0.5  # neutron isospin
    I3_n = -0.5
    Q_pi = -1
    Q_p = +1
    Q_n = 0
    C_gamma = -1

    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Required result keys from pseudocode emit contract
        self.J_eta   = self.results.get('J_eta', None)
        self.P_eta   = self.results.get('P_eta', None)
        self.I_eta   = self.results.get('I_eta', None)
        self.I3_eta  = self.results.get('I3_eta', None)
        self.Q_eta   = self.results.get('Q_eta', None)
        self.C_eta   = self.results.get('C_eta', None)
        self.JPC_eta = self.results.get('JPC_eta', None)

        # Givens and other attributes (from problem statement or solution)
        self.m_eta = getattr(solution_instance, 'm_eta', self.results.get('m_eta', None))
        self.m_pi = getattr(solution_instance, 'm_pi', self.results.get('m_pi', None))
        self.m_p  = getattr(solution_instance, 'm_p',  self.results.get('m_p',  None))
        self.m_n  = getattr(solution_instance, 'm_n',  self.results.get('m_n',  None))

        # Explanation steps (for modeling checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', '')

        # Auxiliary info (optional)
        self.pseudocode = getattr(solution_instance, 'pseudocode', '')

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """All required result keys must be present and have correct types."""
        required = {
            'J_eta': (int, float),
            'P_eta': (int, float),
            'I_eta': (int, float),
            'I3_eta': (int, float),
            'Q_eta': (int, float),
            'C_eta': (int, float),
            'JPC_eta': str,
        }
        for key, expected_type in required.items():
            value = getattr(self, key, None)
            if value is None:
                return (False, f"Missing key: {key}")
            if not isinstance(value, expected_type):
                return (False, f"Wrong type for {key}: got {type(value).__name__}, expected {expected_type.__name__}")
        return (True, "All required keys present with correct types.")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------
    def test_invariant_angular_momentum(self):
        """Angular momentum conservation: |J_eta - J_n| <= J_initial <= J_eta + J_n."""
        if self.J_eta is None:
            return (False, "SKIP: J_eta not available")
        J_initial = 0.5  # J_p + L=0
        J_n = self.J_n
        diff = abs(self.J_eta - J_n)
        total = self.J_eta + J_n
        if diff > J_initial + 1e-9 or total < J_initial - 1e-9:
            return (False, f"Angular momentum violation: |{self.J_eta}-{J_n}|={diff} > {J_initial} or {self.J_eta}+{J_n}={total} < {J_initial}")
        return (True, f"Angular momentum conservation holds (J_eta={self.J_eta})")

    def test_invariant_parity(self):
        """Parity conservation: P_eta must equal -1."""
        if self.P_eta is None:
            return (False, "SKIP: P_eta not available")
        expected = self.P_pi * self.P_p  # initial P = -1 * 1 = -1
        # L=0 contributes +1
        if self.P_eta != expected:
            return (False, f"Parity violation: got P_eta={self.P_eta}, expected {expected}")
        return (True, "Parity conservation satisfied")

    def test_invariant_isospin(self):
        """Isospin and I3 conservation: I_eta=0, I3_eta=0."""
        if self.I_eta is None or self.I3_eta is None:
            return (False, "SKIP: isospin values missing")
        # I3 conservation: I3_initial = I3_pi + I3_p = -1 + 0.5 = -0.5
        # I3_final = I3_eta + I3_n = I3_eta - 0.5  => I3_eta must be 0
        # Isospin magnitude: initial I can be 0.5 or 1.5; final I = I_eta + I_n (if I_eta=0)
        if self.I3_eta != 0:
            return (False, f"I3 violation: got {self.I3_eta}, expected 0")
        if self.I_eta != 0:
            return (False, f"I violation: got {self.I_eta}, expected 0")
        # Also check that with I_eta=0, the initial I=0.5 is possible (would be ruled out if I_eta>=1)
        return (True, "Isospin conservation holds (I=0, I3=0)")

    def test_invariant_charge(self):
        """Charge conservation: Q_eta = 0."""
        if self.Q_eta is None:
            return (False, "SKIP: Q_eta not available")
        expected = self.Q_pi + self.Q_p  # -1 + 1 = 0
        if self.Q_eta != expected:
            return (False, f"Charge violation: got {self.Q_eta}, expected {expected}")
        return (True, "Charge conservation satisfied")

    def test_invariant_c_parity(self):
        """C-parity from 2gamma decay: C_eta = +1."""
        if self.C_eta is None:
            return (False, "SKIP: C_eta not available")
        # C(gamma) = -1, so two gammas give (+1)
        expected = self.C_gamma * self.C_gamma  # = +1
        if self.C_eta != expected:
            return (False, f"C-parity violation: got {self.C_eta}, expected {expected}")
        return (True, "C-parity from 2gamma decay satisfied")

    # ------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------
    def test_modeling_parity_forbids_2pi(self):
        """Explanation must mention parity suppression of eta->2pi."""
        # Look for keywords in explanation_steps or pseudocode
        text = (self.explanation_steps or "") + (self.pseudocode or "")
        if "parity" not in text.lower() or "2π" not in text:
            return (False, "Modeling missing: parity suppression of 2π decay not discussed")
        return (True, "Parity suppression of 2π decay is discussed")

    def test_modeling_c_parity_2gamma(self):
        """Explanation must connect 2gamma decay to C_eta=+1."""
        text = (self.explanation_steps or "") + (self.pseudocode or "")
        if "2γ" not in text and "two gamma" not in text.lower():
            return (False, "Modeling missing: 2γ decay not mentioned")
        if "C" not in text:
            return (False, "Modeling missing: C-parity not used for 2γ decay")
        return (True, "C-parity from 2γ decay is used")

    def test_modeling_s_wave_assumption(self):
        """Explanation must explicitly state s-wave production."""
        text = (self.explanation_steps or "") + (self.pseudocode or "")
        if "s-wave" not in text.lower() and "L=0" not in text:
            return (False, "Modeling missing: s-wave assumption not stated")
        return (True, "s-wave production assumption present")

    def test_modeling_no_eta_minus(self):
        """Explanation must use absence of eta- to infer I=0."""
        text = (self.explanation_steps or "") + (self.pseudocode or "")
        if "η⁻" not in text and "eta-" not in text:
            return (False, "Modeling missing: absence of eta- not discussed")
        return (True, "Absence of eta- used for isospin inference")

    # ------------------------------------------------------------
    # Monotonicity test (threshold momentum vs eta mass)
    # ------------------------------------------------------------
    def test_monotonicity_threshold_momentum_with_mass(self):
        """Increasing eta mass increases threshold momentum (monotonic)."""
        # Need masses: m_pi, m_p, m_n, m_eta
        if any(v is None for v in [self.m_eta, self.m_pi, self.m_p, self.m_n]):
            return (False, "SKIP: one or more masses missing")
        # Convert to GeV (or keep MeV, but constants below use GeV)
        m_pi = self.m_pi / 1000.0 if self.m_pi > 10 else self.m_pi  # assume MeV input
        m_p  = self.m_p  / 1000.0 if self.m_p  > 10 else self.m_p
        m_n  = self.m_n  / 1000.0 if self.m_n  > 10 else self.m_n
        m_eta= self.m_eta/1000.0 if self.m_eta>10 else self.m_eta

        def threshold_p(m_eta_val):
            E0 = ((m_eta_val + m_n)**2 - m_p**2 - m_pi**2) / (2 * m_p)
            return math.sqrt(max(0, E0**2 - m_pi**2))

        baseline = threshold_p(m_eta)
        perturbed = threshold_p(m_eta * 1.01)  # +1%

        if not (perturbed > baseline):
            return (False, f"Monotonic violation: perturbed P0={perturbed:.4f} <= baseline {baseline:.4f}")
        return (True, f"Threshold momentum increases with eta mass (baseline={baseline:.4f}, +1%->{perturbed:.4f})")

    # ------------------------------------------------------------
    # Final answer sanity (weak, one test only)
    # ------------------------------------------------------------
    def test_answer_sanity_final_values(self):
        """Weak sanity: final quantum numbers match target exactly."""
        if any(v is None for v in [self.J_eta, self.P_eta, self.I_eta,
                                   self.I3_eta, self.Q_eta, self.C_eta, self.JPC_eta]):
            return (False, "SKIP: missing values")
        expected = {
            'J_eta': 0,
            'P_eta': -1,
            'I_eta': 0,
            'I3_eta': 0,
            'Q_eta': 0,
            'C_eta': 1,
            'JPC_eta': '0^{-+}'
        }
        for key, val in expected.items():
            actual = getattr(self, key)
            if actual != val:
                return (False, f"{key}: got {actual}, expected {val}")
        return (True, "All final values match target answers.")

    # ------------------------------------------------------------
    # Test runner (automatically discovers all test_ methods)
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover bound methods starting with "test_"
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)