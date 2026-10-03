import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Results from the emit contract
        res = getattr(solution_instance, 'results', {})
        self.a0_prime = res.get('a0_prime', None)
        self.E_binding = res.get('E_binding', None)
        self.E1 = res.get('E1', None)
        self.E2 = res.get('E2', None)
        self.E3 = res.get('E3', None)
        self.E4 = res.get('E4', None)

        # Givens from the problem statement (extract safely)
        givens = getattr(solution_instance, 'givens', {})
        self.m_e = givens.get('m_e', getattr(solution_instance, 'm_e', None))
        self.e = givens.get('e', getattr(solution_instance, 'e', None))
        self.c = givens.get('c', getattr(solution_instance, 'c', None))
        self.hbar = givens.get('hbar', getattr(solution_instance, 'hbar', None))
        self.a0 = givens.get('a0', getattr(solution_instance, 'a0', None))
        self.delta_hf = givens.get('delta_hf', givens.get('delta__hf', None))
        self.B = givens.get('B', getattr(solution_instance, 'B', None))

        # Physical constants for derived checks
        self.eV_to_J = 1.602176634e-19

    # Helper to compute mixed energies given B (re-derivation inside checker)
    def _compute_mixed_energies(self, B_val):
        """Return (E3, E4) in eV given B in tesla.
           Returns (None, None) if required constants missing."""
        if any(x is None for x in (self.hbar, self.e, self.m_e, self.c, self.delta_hf)):
            return None, None
        hbar = self.hbar
        e = self.e
        m_e = self.m_e
        c = self.c
        delta = self.delta_hf  # in eV
        # A = delta / hbar^2  (eV/hbar^2)
        A_hbar2 = delta  # delta = A * hbar^2
        # formula: E3 = -A hbar^2/4 + sqrt((A hbar^2/2)^2 + hbar^2 (eB/(m_e c))^2)
        # we need hbar^2 (eB/(m_e c))^2 in eV^2? Actually the formula uses the same units:
        # sqrt( (Ahbar^2/2)^2 + hbar^2 (eB/(m_e c))^2 )
        # The second term is (hbar e B/(m_e c))^2. But careful: hbar * (eB/(m_e c)) has units of energy.
        # Actually (eB/(m_e c)) has units of 1/s (cyclotron frequency), times hbar gives energy.
        # So the term inside sqrt is (Ahbar^2/2)^2 + (hbar eB/(m_e c))^2.
        # All are energies squared.
        zeeman = (hbar * e * B_val) / (m_e * c)  # in Joules
        zeeman_eV = zeeman / self.eV_to_J
        term = (delta / 2.0) ** 2 + (zeeman_eV) ** 2
        sqrt_term = math.sqrt(term)
        E3 = -delta / 4.0 + sqrt_term
        E4 = -delta / 4.0 - sqrt_term
        return E3, E4

    # ------------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------------
    def test_interface_a0_prime(self):
        if self.a0_prime is None:
            return (False, "a0_prime is missing")
        if not isinstance(self.a0_prime, (int, float)):
            return (False, "a0_prime is not numeric")
        return (True, "a0_prime present and numeric")

    def test_interface_E_binding(self):
        if self.E_binding is None:
            return (False, "E_binding is missing")
        if not isinstance(self.E_binding, (int, float)):
            return (False, "E_binding is not numeric")
        return (True, "E_binding present and numeric")

    def test_interface_E1(self):
        if self.E1 is None:
            return (False, "E1 is missing")
        if not isinstance(self.E1, (int, float)):
            return (False, "E1 is not numeric")
        return (True, "E1 present and numeric")

    def test_interface_E2(self):
        if self.E2 is None:
            return (False, "E2 is missing")
        if not isinstance(self.E2, (int, float)):
            return (False, "E2 is not numeric")
        return (True, "E2 present and numeric")

    def test_interface_E3(self):
        if self.E3 is None:
            return (False, "E3 is missing")
        if not isinstance(self.E3, (int, float)):
            return (False, "E3 is not numeric")
        return (True, "E3 present and numeric")

    def test_interface_E4(self):
        if self.E4 is None:
            return (False, "E4 is missing")
        if not isinstance(self.E4, (int, float)):
            return (False, "E4 is not numeric")
        return (True, "E4 present and numeric")

    # ------------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------------
    def test_invariant_a0_prime_equals_2a0(self):
        if self.a0_prime is None or self.a0 is None:
            return (False, "SKIP: a0_prime or a0 missing")
        expected = 2.0 * self.a0
        if math.isclose(self.a0_prime, expected, rel_tol=1e-2):
            return (True, f"a0_prime ~ 2*a0: {self.a0_prime} vs {expected}")
        return (False, f"a0_prime {self.a0_prime} not close to 2*a0 ({expected})")

    def test_invariant_E_binding_computed(self):
        if self.E_binding is None:
            return (False, "SKIP: E_binding missing")
        if self.m_e is None or self.e is None or self.hbar is None:
            return (False, "SKIP: missing physical constants")
        mu = self.m_e / 2.0
        E_binding_J = mu * (self.e ** 4) / (2.0 * self.hbar ** 2)
        E_binding_eV = abs(E_binding_J) / self.eV_to_J  # positive magnitude
        if not math.isclose(self.E_binding, E_binding_eV, rel_tol=1e-2):
            return (False, f"E_binding {self.E_binding} eV not close to computed {E_binding_eV} eV")
        return (True, "E_binding matches computed ground state binding energy magnitude")

    def test_invariant_E1_E2_equal(self):
        if self.E1 is None or self.E2 is None:
            return (False, "SKIP: E1 or E2 missing")
        if math.isclose(self.E1, self.E2, rel_tol=1e-2, abs_tol=1e-5):
            return (True, f"E1 == E2: {self.E1} ~ {self.E2}")
        return (False, f"E1 {self.E1} and E2 {self.E2} differ")

    def test_invariant_E1_value_from_hyperfine(self):
        if self.E1 is None or self.delta_hf is None:
            return (False, "SKIP: E1 or delta_hf missing")
        expected = self.delta_hf / 4.0  # in eV
        if not math.isclose(self.E1, expected, rel_tol=1e-2, abs_tol=1e-5):
            return (False, f"E1 {self.E1} not close to delta_hf/4 = {expected}")
        return (True, "E1 equals delta_hf/4 (triplet ms=±1 energy)")

    def test_invariant_E4_lower_than_E3(self):
        if self.E3 is None or self.E4 is None:
            return (False, "SKIP: E3 or E4 missing")
        if math.isclose(self.E3, self.E4, rel_tol=1e-2):
            return (False, "E3 and E4 should not be equal")
        if self.E4 >= self.E3:
            return (False, f"E4 ({self.E4}) should be less than E3 ({self.E3})")
        return (True, "E4 < E3")

    def test_invariant_zero_B_limits(self):
        # At B=0, E3 should be delta_hf/4, E4 should be -3 delta_hf/4
        if any(x is None for x in (self.E3, self.E4, self.delta_hf)):
            return (False, "SKIP: E3, E4, or delta_hf missing")
        # Check B is zero (or very small)
        if self.B is not None and abs(self.B) > 1e-10:
            return (False, "SKIP: non-zero B, cannot check zero-field limit")
        expected_E3 = self.delta_hf / 4.0
        expected_E4 = -3.0 * self.delta_hf / 4.0
        if not math.isclose(self.E3, expected_E3, rel_tol=1e-2, abs_tol=1e-5):
            return (False, f"At B=0, E3 should be {expected_E3}, got {self.E3}")
        if not math.isclose(self.E4, expected_E4, rel_tol=1e-2, abs_tol=1e-5):
            return (False, f"At B=0, E4 should be {expected_E4}, got {self.E4}")
        return (True, "At B=0, E3 and E4 match unmixed triplet/singlet energies")

    # ------------------------------------------------------------------------
    # Modeling / trace tests (one relation at a time)
    # ------------------------------------------------------------------------
    def test_modeling_E3_plus_E4_equals_minus_delta_hf_over2(self):
        if self.E3 is None or self.E4 is None or self.delta_hf is None:
            return (False, "SKIP: E3, E4, or delta_hf missing")
        expected_sum = -self.delta_hf / 2.0
        actual_sum = self.E3 + self.E4
        if not math.isclose(actual_sum, expected_sum, rel_tol=1e-2, abs_tol=1e-5):
            return (False, f"E3+E4 = {actual_sum}, expected {expected_sum}")
        return (True, "Sum of mixed energies (E3+E4) equals -delta_hf/2")

    def test_modeling_E1_equal_E3_at_B0(self):
        if self.E1 is None or self.E3 is None:
            return (False, "SKIP: E1 or E3 missing")
        if self.B is not None and abs(self.B) > 1e-10:
            return (False, "SKIP: non-zero B, E3 != E1")
        if not math.isclose(self.E1, self.E3, rel_tol=1e-2, abs_tol=1e-5):
            return (False, f"At B=0, E1 and E3 should be equal, got {self.E1} vs {self.E3}")
        return (True, "E1 == E3 at B=0")

    # ------------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ------------------------------------------------------------------------
    def test_monotonicity_E3_increases_with_B_perturbation(self):
        """
        Perturb B upward by 10% and check that E3 increases (or stays the same).
        If B is missing or zero, use a small test B and check increase from B=0 to B positive.
        """
        # Need constants to recompute E3
        if self.E3 is None:
            return (False, "SKIP: E3 missing")
        if any(x is None for x in (self.hbar, self.e, self.m_e, self.c, self.delta_hf)):
            return (False, "SKIP: missing constants to recompute E3")

        # Determine baseline B. If B is None or zero, use a small positive B for comparison.
        if self.B is not None and abs(self.B) > 1e-12:
            B_base = self.B
        else:
            # Use a small field (0.1 T) as baseline if no B given
            B_base = 0.1
        B_pert = B_base * 1.1  # +10%

        E3_base, _ = self._compute_mixed_energies(B_base)
        E3_pert, _ = self._compute_mixed_energies(B_pert)
        if E3_base is None or E3_pert is None:
            return (False, "SKIP: could not compute energies")

        if E3_pert < E3_base - 1e-10:
            return (False, f"E3 decreased when B increased: {E3_base} -> {E3_pert}")
        return (True, f"E3 increased (or stayed) when B increased by 10%: {E3_base} -> {E3_pert}")

    def test_monotonicity_E4_decreases_with_B_perturbation(self):
        """Perturb B upward by 10% and check that E4 decreases (becomes more negative)."""
        if self.E4 is None:
            return (False, "SKIP: E4 missing")
        if any(x is None for x in (self.hbar, self.e, self.m_e, self.c, self.delta_hf)):
            return (False, "SKIP: missing constants to recompute E4")

        if self.B is not None and abs(self.B) > 1e-12:
            B_base = self.B
        else:
            B_base = 0.1
        B_pert = B_base * 1.1

        _, E4_base = self._compute_mixed_energies(B_base)
        _, E4_pert = self._compute_mixed_energies(B_pert)
        if E4_base is None or E4_pert is None:
            return (False, "SKIP: could not compute energies")

        if E4_pert > E4_base + 1e-10:
            return (False, f"E4 increased (less negative) when B increased: {E4_base} -> {E4_pert}")
        return (True, f"E4 decreased (more negative) when B increased by 10%: {E4_base} -> {E4_pert}")

    # ------------------------------------------------------------------------
    # Weak answer sanity (non-dominant, no math.isclose)
    # ------------------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitudes(self):
        all_finite = True
        msgs = []
        energies = [('E_binding', self.E_binding), ('E1', self.E1), ('E2', self.E2),
                    ('E3', self.E3), ('E4', self.E4)]
        for name, val in energies:
            if val is None:
                msgs.append(f"{name}: missing")
                continue
            if not math.isfinite(val):
                msgs.append(f"{name}: non-finite")
                all_finite = False
        # Broad magnitude checks
        if self.E_binding is not None:
            if not (1.0 < self.E_binding < 15.0):
                msgs.append(f"E_binding magnitude {self.E_binding} eV outside expected range (1-15)")
                all_finite = False
        if self.E1 is not None:
            if abs(self.E1) > 1e-2:  # hyperfine splittings ~1e-3 eV
                msgs.append(f"E1 magnitude {abs(self.E1)} eV seems too large")
                all_finite = False
        if all_finite:
            return (True, "All energies finite and within broad expected ranges")
        return (False, "; ".join(msgs))

    # ------------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)