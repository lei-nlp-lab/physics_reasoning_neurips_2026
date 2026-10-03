import math
import sympy
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}

        # Coefficients from the problem statement (re-derived)
        # Each element: (l, m, amplitude)
        self.coeffs = [(9, 1, 1), (8, -1, -2), (9, 3, 2)]
        self.total_norm2 = sum(amp**2 for _, _, amp in self.coeffs)  # = 9

        # Planck's constant (if provided in givens)
        self.hbar = self.givens.get('hbar', None)

        # Extract expected result quantities (safe defaults)
        self.C = self.results.get('C', None)
        self.L2_expect = self.results.get('L2_expect', None)
        self.Lz_expect = self.results.get('Lz_expect', None)
        self.Lx2_expect = self.results.get('Lx2_expect', None)
        self.L2_expect_t = self.results.get('L2_expect_t', None)
        self.Lz_expect_t = self.results.get('Lz_expect_t', None)

    # ----------------------------------------------------------------------
    # Helper to compute normalized L^2 expectation from internal coefficients
    # ----------------------------------------------------------------------
    def _compute_L2_expect(self, coeffs, hbar_val):
        """Return ⟨L²⟩ = ℏ² Σ |c_i|² l_i(l_i+1) / Σ |c_i|²"""
        norm2 = sum(amp**2 for _, _, amp in coeffs)
        weighted_sum = sum(amp**2 * l*(l+1) for l, _, amp in coeffs)
        return hbar_val**2 * weighted_sum / norm2

    def _compute_Lz_expect(self, coeffs, hbar_val):
        """Return ⟨L_z⟩ = ℏ Σ |c_i|² m_i / Σ |c_i|²"""
        norm2 = sum(amp**2 for _, _, amp in coeffs)
        weighted_sum = sum(amp**2 * m for _, m, amp in coeffs)
        return hbar_val * weighted_sum / norm2

    def _compute_Lz2_expect(self, coeffs, hbar_val):
        """Return ⟨L_z²⟩ = ℏ² Σ |c_i|² m_i² / Σ |c_i|²"""
        norm2 = sum(amp**2 for _, _, amp in coeffs)
        weighted_sum = sum(amp**2 * m**2 for _, m, amp in coeffs)
        return hbar_val**2 * weighted_sum / norm2

    def _compute_Lx2_from_ladder(self, coeffs, hbar_val):
        """
        Compute ⟨L_x²⟩ using ladder operators.
        L_x² = (1/4)(L_+² + L_+L_- + L_-L_+ + L_-²)
        Only terms that map to a state present in the expansion contribute.
        Returns numeric value.
        """
        norm2 = sum(amp**2 for _, _, amp in coeffs)
        h = hbar_val
        expectation = 0.0

        # Pre-calculate squared amplitudes (including normalization)
        for l, m, amp in coeffs:
            amp_sq = amp**2 / norm2  # normalized weight

            # Diagonal contributions from L_+L_- and L_-L_+
            # L_+L_- |l,m> = h² [l(l+1)-m(m-1)] |l,m>
            # L_-L_+ |l,m> = h² [l(l+1)-m(m+1)] |l,m>
            diag = h**2 * ( (l*(l+1) - m*(m-1)) + (l*(l+1) - m*(m+1)) ) / 4.0
            expectation += amp_sq * diag

        # Off-diagonal contributions from L_+² and L_-²
        # L_+² |l,m> = h² sqrt(l(l+1)-m(m+1)) sqrt(l(l+1)-(m+1)(m+2)) |l,m+2>
        # L_-² |l,m> = h² sqrt(l(l+1)-m(m-1)) sqrt(l(l+1)-(m-1)(m-2)) |l,m-2>
        # They contribute only if the resulting state is also in the expansion.
        for l, m, amp in coeffs:
            # Check L_+² -> (l, m+2)
            for l2, m2, amp2 in coeffs:
                if l == l2 and m+2 == m2:
                    # <ψ| L_+² |ψ> = (amp_conj) * (coeff) * matrix element
                    # Since coefficients are real, use amp * amp2
                    factor = amp * amp2 / norm2
                    mat_elem = h**2 * math.sqrt(l*(l+1) - m*(m+1)) * math.sqrt(l*(l+1) - (m+1)*(m+2))
                    expectation += 2 * factor * mat_elem / 4.0  # factor 2 from Hermitian conjugate of L_-²
                # Check L_-² -> (l, m-2)
                if l == l2 and m-2 == m2:
                    factor = amp * amp2 / norm2
                    mat_elem = h**2 * math.sqrt(l*(l+1) - m*(m-1)) * math.sqrt(l*(l+1) - (m-1)*(m-2))
                    expectation += 2 * factor * mat_elem / 4.0

        return expectation

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        expected_keys = ['C', 'L2_expect', 'Lz_expect', 'Lx2_expect',
                         'L2_expect_t', 'Lz_expect_t']
        missing = [k for k in expected_keys if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_types(self):
        # Check that 'C' is numeric (int, float, or sympy numeric)
        if self.C is None:
            return (False, "SKIP: C not available")
        if not isinstance(self.C, (int, float, sympy.Integer, sympy.Float, sympy.Rational)):
            return (False, f"C has invalid type {type(self.C)}")
        return (True, "Types acceptable")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based, use isclose)
    # ----------------------------------------------------------------------
    def test_invariant_normalization(self):
        if self.C is None:
            return (False, "SKIP: C not available")
        # Normalization: C^2 * (1^2+2^2+2^2) = 1 → C = 1/3
        expected_C = 1.0 / math.sqrt(self.total_norm2)  # = 1/3
        if math.isclose(self.C, expected_C, rel_tol=1e-2):
            return (True, f"C = {self.C} agrees with 1/3")
        else:
            return (False, f"C = {self.C}, expected {expected_C}")

    def test_invariant_L2_expectation(self):
        if self.L2_expect is None:
            return (False, "SKIP: L2_expect not available")
        if self.hbar is None:
            return (False, "SKIP: hbar not given, cannot compute")
        # Compute expected value from internal coefficients
        expected = self._compute_L2_expect(self.coeffs, self.hbar)
        if math.isclose(self.L2_expect, expected, rel_tol=1e-2):
            return (True, f"⟨L²⟩ = {self.L2_expect} agrees with {expected}")
        else:
            return (False, f"⟨L²⟩ = {self.L2_expect}, expected {expected}")

    def test_invariant_Lz_expectation(self):
        if self.Lz_expect is None:
            return (False, "SKIP: Lz_expect not available")
        if self.hbar is None:
            return (False, "SKIP: hbar not given, cannot compute")
        expected = self._compute_Lz_expect(self.coeffs, self.hbar)
        if math.isclose(self.Lz_expect, expected, rel_tol=1e-2):
            return (True, f"⟨L_z⟩ = {self.Lz_expect} agrees with {expected}")
        else:
            return (False, f"⟨L_z⟩ = {self.Lz_expect}, expected {expected}")

    def test_invariant_Lx2_expectation(self):
        if self.Lx2_expect is None:
            return (False, "SKIP: Lx2_expect not available")
        if self.hbar is None:
            return (False, "SKIP: hbar not given")
        # Expected is known analytically: (163/9) hbar²
        expected = (163.0 / 9.0) * (self.hbar**2)
        if math.isclose(self.Lx2_expect, expected, rel_tol=1e-2):
            return (True, f"⟨L_x²⟩ = {self.Lx2_expect} agrees with {expected}")
        else:
            return (False, f"⟨L_x²⟩ = {self.Lx2_expect}, expected {expected}")

    def test_invariant_L2_time_independence(self):
        if self.L2_expect is None or self.L2_expect_t is None:
            return (False, "SKIP: missing L2_expect or L2_expect_t")
        if math.isclose(self.L2_expect, self.L2_expect_t, rel_tol=1e-2):
            return (True, "⟨L²⟩_0 == ⟨L²⟩_t")
        else:
            return (False, f"⟨L²⟩_0 = {self.L2_expect}, ⟨L²⟩_t = {self.L2_expect_t}")

    def test_invariant_Lz_time_independence(self):
        if self.Lz_expect is None or self.Lz_expect_t is None:
            return (False, "SKIP: missing Lz_expect or Lz_expect_t")
        if math.isclose(self.Lz_expect, self.Lz_expect_t, rel_tol=1e-2):
            return (True, "⟨L_z⟩_0 == ⟨L_z⟩_t")
        else:
            return (False, f"⟨L_z⟩_0 = {self.Lz_expect}, ⟨L_z⟩_t = {self.Lz_expect_t}")

    # ----------------------------------------------------------------------
    # Modeling / Trace tests (focused on specific modeling details)
    # ----------------------------------------------------------------------
    def test_modeling_ladder_operator_consistency(self):
        """
        Verify that the submitted ⟨L_x²⟩ is consistent with ladder operator
        calculation from the given coefficients (l,m,amplitude). This tests the
        modeling of the ladder operator actions.
        """
        if self.Lx2_expect is None:
            return (False, "SKIP: Lx2_expect not available")
        if self.hbar is None:
            return (False, "SKIP: hbar not given")
        try:
            expected = self._compute_Lx2_from_ladder(self.coeffs, self.hbar)
        except Exception as e:
            return (False, f"Error computing ladder expectation: {e}")
        if math.isclose(self.Lx2_expect, expected, rel_tol=1e-2):
            return (True, f"⟨L_x²⟩ = {self.Lx2_expect} agrees with ladder computation {expected}")
        else:
            return (False, f"⟨L_x²⟩ = {self.Lx2_expect}, ladder computation gives {expected}")

    def test_modeling_time_evolution_correctness(self):
        """
        Verify that the time-dependent values ⟨L²⟩_t and ⟨L_z⟩_t are
        exactly the same as the static ones (modeling: phases cancel).
        """
        if any(v is None for v in [self.L2_expect, self.Lz_expect,
                                   self.L2_expect_t, self.Lz_expect_t]):
            return (False, "SKIP: missing time-dependent data")
        errors = []
        if not math.isclose(self.L2_expect, self.L2_expect_t, rel_tol=1e-2):
            errors.append("L2 mismatch")
        if not math.isclose(self.Lz_expect, self.Lz_expect_t, rel_tol=1e-2):
            errors.append("Lz mismatch")
        if errors:
            return (False, f"Time-dependent expectations differ: {', '.join(errors)}")
        return (True, "Time-dependent expectations equal static expectations, as expected")

    # ----------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_L2_increase_with_higher_l(self):
        if self.hbar is None:
            return (False, "SKIP: hbar not given")
        # Baseline coefficients
        coeffs_baseline = self.coeffs[:]
        # Baseline L2
        baseline = self._compute_L2_expect(coeffs_baseline, self.hbar)
        # Perturb: increase amplitude of third coefficient (index 2) by 10%
        perturbed_coeffs = [(l, m, amp * 1.10) if i == 2 else (l, m, amp)
                            for i, (l, m, amp) in enumerate(coeffs_baseline)]
        perturbed = self._compute_L2_expect(perturbed_coeffs, self.hbar)
        if perturbed > baseline:
            return (True, f"⟨L²⟩ increased from {baseline} to {perturbed} as expected")
        else:
            return (False, f"⟨L²⟩ did not increase: baseline {baseline}, perturbed {perturbed}")

    def test_monotonicity_Lz_increase_with_positive_m(self):
        if self.hbar is None:
            return (False, "SKIP: hbar not given")
        coeffs_baseline = self.coeffs[:]
        baseline = self._compute_Lz_expect(coeffs_baseline, self.hbar)
        # Perturb the third coefficient (l=9,m=3) by +10%
        perturbed_coeffs = [(l, m, amp * 1.10) if i == 2 else (l, m, amp)
                            for i, (l, m, amp) in enumerate(coeffs_baseline)]
        perturbed = self._compute_Lz_expect(perturbed_coeffs, self.hbar)
        if perturbed > baseline:
            return (True, f"⟨L_z⟩ increased from {baseline} to {perturbed} as expected")
        else:
            return (False, f"⟨L_z⟩ did not increase: baseline {baseline}, perturbed {perturbed}")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, no isclose)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        # Check positivity and finiteness of key results
        checks = []
        if self.C is not None:
            if not (isinstance(self.C, (int, float, sympy.Number)) and self.C > 0):
                checks.append("C not positive")
        if self.L2_expect is not None:
            try:
                val = float(self.L2_expect)
                if not (val > 0 and math.isfinite(val)):
                    checks.append("L2_expect not positive finite")
            except:
                checks.append("L2_expect cannot be converted to float")
        if self.Lz_expect is not None:
            try:
                val = float(self.Lz_expect)
                if not math.isfinite(val):
                    checks.append("Lz_expect not finite")
            except:
                checks.append("Lz_expect cannot be converted to float")
        if self.Lx2_expect is not None:
            try:
                val = float(self.Lx2_expect)
                if not (val > 0 and math.isfinite(val)):
                    checks.append("Lx2_expect not positive finite")
            except:
                checks.append("Lx2_expect cannot be converted to float")
        if self.L2_expect_t is not None:
            try:
                val = float(self.L2_expect_t)
                if not (val > 0 and math.isfinite(val)):
                    checks.append("L2_expect_t not positive finite")
            except:
                checks.append("L2_expect_t cannot be converted to float")
        if self.Lz_expect_t is not None:
            try:
                val = float(self.Lz_expect_t)
                if not math.isfinite(val):
                    checks.append("Lz_expect_t not finite")
            except:
                checks.append("Lz_expect_t cannot be converted to float")
        if checks:
            return (False, "; ".join(checks))
        return (True, "Basic sanity checks passed")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all bound methods starting with "test_"
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
                import traceback
                traceback.print_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)