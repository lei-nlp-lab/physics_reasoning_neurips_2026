import math
import traceback

class PhysicsChecker:
    """
    Validates a student solution for the helium ground state problem
    (Physics 216, problem 26.3) using theorem‑based checks and modelling quality.
    """

    def __init__(self, solution_instance):
        # Extract all relevant data safely – use None for missing values.
        self.results = solution_instance.results if hasattr(solution_instance, 'results') else {}
        self.givens = solution_instance.givens if hasattr(solution_instance, 'givens') else {}

        # Result keys – NEVER use numeric defaults
        self.psi = self.results.get('psi', None)
        self.E0 = self.results.get('E0', None)
        self.expectation = self.results.get('expectation', None)
        self.Vee = self.results.get('Vee', None)
        self.E_corr = self.results.get('E_corr', None)

        # Givens (optional) – all accessed via .get() with None default
        self.Z_given = self.givens.get('Z', None)
        self.a0_given = self.givens.get('a0', None)

        # Physics constants (not from the solution)
        self.HARTREE_EV = 27.2       # 1 Hartree = 27.2 eV
        self.RYD_EV = 13.6          # Hydrogen ground state energy
        self.EXPECTED_Z = 2

    # ------------------------------------------------------------------
    #  Interface checks: required keys exist and have correct type
    # ------------------------------------------------------------------
    def test_interface_psi_exists(self):
        """Check that psi key is present and is a string."""
        if self.psi is None:
            return (False, "Missing 'psi' in results")
        if not isinstance(self.psi, str):
            return (False, "'psi' should be a string (the wavefunction expression)")
        return (True, "'psi' present and is a string")

    def test_interface_energy_exists(self):
        """Check that E0 is present and numeric."""
        if self.E0 is None:
            return (False, "Missing 'E0' in results")
        if not isinstance(self.E0, (int, float)):
            return (False, "'E0' should be numeric (eV)")
        return (True, "'E0' present and numeric")

    def test_interface_expectation_exists(self):
        """Check that expectation is present (string or numeric)."""
        if self.expectation is None:
            return (False, "Missing 'expectation' in results")
        if not isinstance(self.expectation, (str, int, float)):
            return (False, "'expectation' should be a string or numeric")
        return (True, "'expectation' present")

    def test_interface_vee_exists(self):
        """Check that Vee is present and numeric."""
        if self.Vee is None:
            return (False, "Missing 'Vee' in results")
        if not isinstance(self.Vee, (int, float)):
            return (False, "'Vee' should be numeric (eV)")
        return (True, "'Vee' present and numeric")

    def test_interface_ecorr_exists(self):
        """Check that E_corr is present and numeric."""
        if self.E_corr is None:
            return (False, "Missing 'E_corr' in results")
        if not isinstance(self.E_corr, (int, float)):
            return (False, "'E_corr' should be numeric (eV)")
        return (True, "'E_corr' present and numeric")

    # ------------------------------------------------------------------
    #  Invariant checks (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_noninteracting_energy(self):
        """E0 should equal 2 * ( -Z^2 * 13.6 eV ) with Z=2."""
        if self.E0 is None:
            return (False, "SKIP: E0 not available")
        expected = 2 * (- (self.EXPECTED_Z**2) * self.RYD_EV)   # -108.8 eV
        if not math.isclose(self.E0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E0 = {self.E0} eV, expected ≈ {expected} eV")
        return (True, f"E0 ≈ {expected} eV (within 1% tolerance)")

    def test_invariant_expectation_dimensionless(self):
        """If expectation is numeric, its product with a0 should be ~5/4."""
        if self.expectation is None:
            return (False, "SKIP: expectation not available")
        if isinstance(self.expectation, (int, float)):
            # We need a0 to interpret; if not given, skip numeric check.
            if self.a0_given is None:
                return (False, "SKIP: a0 not given; cannot check dimensionless product")
            product = self.expectation * self.a0_given
            if not math.isclose(product, 5.0/4.0, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"⟨1/r12⟩ * a0 = {product}, expected ≈ {5.0/4.0}")
            return (True, f"⟨1/r12⟩ * a0 ≈ {5.0/4.0}")
        else:
            # string – check that it contains the symbolic form "5/(4a0)"
            expr = str(self.expectation).replace(" ", "")
            if "5/(4a0)" in expr or "5/4a0" in expr or "5/(4 a0)" in expr:
                return (True, "Expectation expression matches 5/(4a0)")
            return (False, f"Expectation = '{self.expectation}' – does not contain 5/(4a0)")

    def test_invariant_vee_from_hartree(self):
        """Vee should equal (5/4) * Hartree = 34.0 eV."""
        if self.Vee is None:
            return (False, "SKIP: Vee not available")
        expected = (5.0/4.0) * self.HARTREE_EV   # 34.0 eV
        if not math.isclose(self.Vee, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Vee = {self.Vee} eV, expected ≈ {expected} eV")
        return (True, f"Vee ≈ {expected} eV (within 1% tolerance)")

    def test_invariant_corrected_energy_adds(self):
        """E_corr should equal E0 + Vee (within tolerance)."""
        if any(x is None for x in [self.E0, self.Vee, self.E_corr]):
            return (False, "SKIP: one of E0, Vee, E_corr missing")
        expected = self.E0 + self.Vee
        if not math.isclose(self.E_corr, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_corr = {self.E_corr} eV, but E0+Vee = {expected} eV")
        return (True, "E_corr equals E0 + Vee")

    def test_invariant_spin_antisymmetry(self):
        """Wavefunction string should indicate spin singlet / antisymmetry."""
        if self.psi is None:
            return (False, "SKIP: psi not available")
        patterns = [
            r"|↑↓⟩ - |↓↑⟩",
            r"|↑↓> - |↓↑>",
            "singlet",
            "antisymmetric",
            "up down - down up",
        ]
        found = any(p in self.psi for p in patterns)
        if not found:
            return (False, "Wavefunction missing antisymmetric spin part (singlet pattern)")
        return (True, "Antisymmetric spin part present")

    def test_invariant_ground_state_1s_orbital(self):
        """Wavefunction should contain exponential factor with Z=2: e^{-2(r1+r2)/a0}."""
        if self.psi is None:
            return (False, "SKIP: psi not available")
        # Check for a clear indication of 1s orbitals and Z=2
        if r"2(r1+r2)/a0" in self.psi or r"2(r1+r2)/a0" in self.psi:
            return (True, "Spatial factor e^{-2(r1+r2)/a0} present")
        if r"e^{-2r" in self.psi and r"/a0}" in self.psi:
            return (True, "Exponential factors consistent with Z=2")
        return (False, f"Wavefunction does not clearly show hydrogenic 1s orbitals with Z=2")

    # ------------------------------------------------------------------
    #  Modelling / Trace checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_expectation_derivation(self):
        """Expectation value should be expressed as 5/(4a0), not just a raw number."""
        if self.expectation is None:
            return (False, "SKIP: expectation not available")
        expr = str(self.expectation).replace(" ", "")
        if "5/(4a0)" in expr or "5/4a0" in expr or "5/(4 a0)" in expr:
            return (True, "Expectation correctly expressed as 5/(4a0)")
        return (False, f"Expectation should be expressed as 5/(4a0), got '{self.expectation}'")

    def test_modeling_energy_conversion_consistency(self):
        """Vee should be derivable from the expectation: Vee = (5/4)*Hartree."""
        if self.Vee is None:
            return (False, "SKIP: Vee not available")
        expected = (5.0/4.0) * self.HARTREE_EV
        if math.isclose(self.Vee, expected, rel_tol=0.02):
            return (True, "Vee consistent with (5/4) Hartree")
        return (False, f"Vee = {self.Vee} eV, expected from (5/4)*27.2 eV = {expected} eV")

    # ------------------------------------------------------------------
    #  Monotonicity checks (perturb an input, compare to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_addition(self):
        """
        Perturb Vee by +1 eV; E_corr should increase by exactly +1 eV.
        (This checks the linear relationship E_corr = E0 + Vee.)
        """
        if any(x is None for x in [self.E0, self.Vee, self.E_corr]):
            return (False, "SKIP: cannot perform monotonicity check without E0, Vee, E_corr")
        delta = 1.0   # eV
        # Baseline: E_corr_student should already be E0+Vee (within tolerance)
        baseline = self.E0 + self.Vee
        if not math.isclose(self.E_corr, baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Initial E_corr does not match E0+Vee; cannot perform monotonicity")
        # Perturb: add delta to Vee
        new_Vee = self.Vee + delta
        new_E_corr = self.E0 + new_Vee
        # The change should equal delta
        change = new_E_corr - self.E_corr
        if not math.isclose(change, delta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Perturbing Vee by +1 eV changed E_corr by {change} eV, expected +1 eV")
        return (True, "Monotonicity: increasing Vee increases E_corr by the same amount")

    # ------------------------------------------------------------------
    #  Weak answer sanity (only one, no isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_energy_signs_and_range(self):
        """Energy magnitudes and signs should be plausible."""
        errors = []
        if self.E0 is not None:
            if not (self.E0 < -50 and self.E0 > -150):
                errors.append(f"E0 = {self.E0} eV, expected around -109 eV")
        if self.Vee is not None:
            if not (self.Vee > 20 and self.Vee < 50):
                errors.append(f"Vee = {self.Vee} eV, expected around 34 eV")
        if self.E_corr is not None:
            if not (self.E_corr < -50 and self.E_corr > -100):
                errors.append(f"E_corr = {self.E_corr} eV, expected around -75 eV")
        if errors:
            return (False, " | ".join(errors))
        return (True, "Energy signs and magnitudes plausible")

    # ------------------------------------------------------------------
    #  Run all tests
    # ------------------------------------------------------------------
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


# Example usage (if run as script)
if __name__ == "__main__":
    class DummySolution:
        def __init__(self):
            self.results = {
                'psi': r"ψ(r1,r2) = (8/(π a0^3)) e^{-2(r1+r2)/a0} (|↑↓⟩ - |↓↑⟩)/√2",
                'E0': -108.8,
                'expectation': "5/(4a0)",
                'Vee': 34.0,
                'E_corr': -74.8
            }
            self.givens = {'Z': 2, 'a0': 5.29e-11}
    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print("\nAll passed:", success)