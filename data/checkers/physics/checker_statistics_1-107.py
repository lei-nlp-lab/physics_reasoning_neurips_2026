import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for heat of vaporization calculation.
    Required result key: 'H_vap'
    Optional given keys: 'E_mol_eV', 'eV_to_erg', 'N_A'
    """

    def __init__(self, solution_instance):
        # Safely extract givens and results, using None for missing
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        self.H_vap = self.results.get('H_vap', None)

        # Givens (may be missing, use None)
        self.E_mol_eV = self.givens.get('E_mol_eV', None)
        self.eV_to_erg = self.givens.get('eV_to_erg', None)  # 1.6e-12 erg/eV
        self.N_A = self.givens.get('N_A', None)              # 6.022e23 mol^{-1}

    # ------------------------------------------------------------------
    # Interface checks – required keys and types
    # ------------------------------------------------------------------
    def test_interface_H_vap_present(self):
        """Check that the required key 'H_vap' exists in results."""
        if self.H_vap is None:
            return (False, "H_vap is missing from results")
        return (True, "H_vap is present")

    def test_interface_H_vap_numeric(self):
        """Check that H_vap is a finite number."""
        if self.H_vap is None:
            return (False, "SKIP: H_vap not available")
        if not isinstance(self.H_vap, (int, float, complex)):
            return (False, f"H_vap has invalid type: {type(self.H_vap)}")
        if isinstance(self.H_vap, complex):
            return (False, "H_vap is complex, expected real number")
        if not math.isfinite(self.H_vap):
            return (False, f"H_vap is not finite: {self.H_vap}")
        return (True, "H_vap is numeric and finite")

    # ------------------------------------------------------------------
    # Invariant checks – physics laws and sign
    # ------------------------------------------------------------------
    def test_invariant_H_vap_positive(self):
        """Latent heat should be positive."""
        if self.H_vap is None:
            return (False, "SKIP: H_vap not available")
        if self.H_vap <= 0:
            return (False, f"H_vap should be > 0, got {self.H_vap}")
        return (True, "H_vap > 0 (positive energy to liberate)")

    def test_invariant_order_of_magnitude(self):
        """Order of magnitude should be roughly 10^10 erg/mol."""
        if self.H_vap is None:
            return (False, "SKIP: H_vap not available")
        # Broad sanity: between 1e9 and 1e12 (covers 4.8e10)
        if not (1e9 < self.H_vap < 1e12):
            return (False, f"H_vap = {self.H_vap} not in expected range [1e9, 1e12]")
        return (True, "H_vap order of magnitude plausible")

    # ------------------------------------------------------------------
    # Modeling checks – re-derivation from given constants if available
    # ------------------------------------------------------------------
    def test_modeling_conversion_from_givens(self):
        """
        If all givens are present, re-compute H_vap and compare with the result.
        Use fuzzy matching (rel_tol=1e-2) because of potential rounding in the solver.
        """
        # Check required givens and result
        if self.H_vap is None:
            return (False, "SKIP: H_vap not available")
        if self.E_mol_eV is None or self.eV_to_erg is None or self.N_A is None:
            return (False, "SKIP: missing givens (E_mol_eV, eV_to_erg, N_A)")

        # Ensure numeric types
        try:
            E_mol_eV = float(self.E_mol_eV)
            eV_to_erg = float(self.eV_to_erg)
            N_A = float(self.N_A)
        except (TypeError, ValueError) as e:
            return (False, f"Givens not convertible to float: {e}")

        # Re-derive H_vap
        expected_H_vap = E_mol_eV * eV_to_erg * N_A

        if not math.isclose(self.H_vap, expected_H_vap, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"H_vap = {self.H_vap}, but re-derived = {expected_H_vap} "
                    f"(relative difference > 1% with rel_tol=1e-2)")
        return (True, f"H_vap consistent with given constants: {self.H_vap} ≈ {expected_H_vap}")

    def test_modeling_reverse_energy_check(self):
        """
        Cross-check: from H_vap and N_A compute per-molecule energy in erg,
        then convert back to eV using the given conversion factor, and
        compare to the expected 1/20 eV (0.05 eV) if available.
        This tests the internal consistency of the result without assuming
        the exact values of N_A or eV_to_erg (they may come from givens).
        """
        if self.H_vap is None:
            return (False, "SKIP: H_vap not available")

        # Need N_A to compute per-molecule energy.
        N_A = None
        if self.N_A is not None:
            try:
                N_A = float(self.N_A)
            except (TypeError, ValueError):
                pass
        if N_A is None:
            # Fallback to standard value (given in problem description)
            N_A = 6.022e23

        # Compute per-molecule energy in erg
        E_mol_erg = self.H_vap / N_A

        # Check positivity and reasonableness
        if not (E_mol_erg > 0 and math.isfinite(E_mol_erg)):
            return (False, f"E_mol_erg = {E_mol_erg} is not positive and finite")

        # Now convert to eV using the conversion factor
        eV_to_erg = None
        if self.eV_to_erg is not None:
            try:
                eV_to_erg = float(self.eV_to_erg)
            except (TypeError, ValueError):
                pass
        if eV_to_erg is None:
            eV_to_erg = 1.6e-12   # standard value from problem

        E_mol_eV_derived = E_mol_erg / eV_to_erg

        # The expected value from the problem is 1/20 = 0.05 eV.
        expected_E_mol_eV = 0.05

        if not math.isclose(E_mol_eV_derived, expected_E_mol_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"Derived per-molecule energy = {E_mol_eV_derived} eV, "
                    f"expected ~0.05 eV (relative difference > 1%)")
        return (True,
                f"Per-molecule energy check: {E_mol_eV_derived} eV ≈ 0.05 eV")

    # ------------------------------------------------------------------
    # Monotonicity check – explicit input perturbation
    # ------------------------------------------------------------------
    def test_monotonicity_H_vap_increases_with_E_mol_eV(self):
        """
        Verify that increasing the per‑molecule energy (E_mol_eV) leads to
        a larger heat of vaporization.  Perturb E_mol_eV by +10% and check
        that the expected output increases.
        """
        if self.H_vap is None:
            return (False, "SKIP: H_vap not available")

        # Use givens if available; otherwise fallback to standard constants.
        eV_to_erg = self.eV_to_erg if self.eV_to_erg is not None else 1.6e-12
        N_A = self.N_A if self.N_A is not None else 6.022e23

        if self.E_mol_eV is None:
            return (False, "SKIP: E_mol_eV not available (cannot perturb)")

        try:
            E_mol_eV = float(self.E_mol_eV)
            eV_to_erg = float(eV_to_erg)
            N_A = float(N_A)
        except (TypeError, ValueError) as e:
            return (False, f"SKIP: givens not numeric: {e}")

        baseline = E_mol_eV * eV_to_erg * N_A
        perturbed_E = E_mol_eV * 1.10   # 10% increase
        perturbed = perturbed_E * eV_to_erg * N_A

        if not (perturbed > baseline):
            return (False,
                    f"Perturbed output {perturbed} is not larger than baseline {baseline} "
                    f"(expected monotonic increase)")
        return (True,
                f"Increasing E_mol_eV from {E_mol_eV} to {perturbed_E} "
                f"increases expected H_vap from {baseline} to {perturbed} (monotonic)")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, must not re-derive)
    # ------------------------------------------------------------------
    def test_answer_sanity_h_vap_range(self):
        """
        Weak final‑answer sanity: H_vap must be positive and finite.
        No explicit comparison to expected value, no math.isclose usage.
        """
        if self.H_vap is None:
            return (False, "SKIP: H_vap not available")
        if not (self.H_vap > 0 and math.isfinite(self.H_vap)):
            return (False, f"H_vap = {self.H_vap} is not positive and finite")
        return (True, "H_vap is positive and finite (answer passes weak sanity)")

    # ------------------------------------------------------------------
    # Test runner – discover all test_* methods
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Execute all test methods and return (success_bool, logs_list)."""
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)