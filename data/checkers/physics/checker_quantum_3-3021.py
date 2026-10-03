import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the two spin-1/2 system with Hamiltonian
    H = A(s1z + s2z) + B s1·s2.
    Checks interface, invariants (theorem-based), modeling quality, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Extract all needed attributes safely (None if missing)
        # Givens: A, B, hbar (might be provided as attributes or in results dict)
        self.A = getattr(solution_instance, 'A', None)
        if self.A is None:
            self.A = getattr(solution_instance, 'results', {}).get('A', None)

        self.B = getattr(solution_instance, 'B', None)
        if self.B is None:
            self.B = getattr(solution_instance, 'results', {}).get('B', None)

        self.hbar = getattr(solution_instance, 'hbar', None)
        if self.hbar is None:
            self.hbar = getattr(solution_instance, 'results', {}).get('hbar', None)

        # Results: energies
        results = getattr(solution_instance, 'results', {})
        self.E1 = results.get('E1', None)
        self.E2 = results.get('E2', None)
        self.E3 = results.get('E3', None)
        self.E4 = results.get('E4', None)

        # Also store solution_instance for additional info (like explanation_steps)
        self.sol = solution_instance

    # ----------------------------------------------------------------------
    # Interface checks: required keys exist and are numeric
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four energy keys are present and are numbers."""
        keys = ['E1', 'E2', 'E3', 'E4']
        for k in keys:
            v = getattr(self, k, None)
            if v is None:
                return (False, f"Missing result key: {k}")
            if not isinstance(v, (int, float, sp.Expr, sp.Number)):
                return (False, f"Result {k} has non-numeric type: {type(v)}")
        return (True, "All required energy keys present and numeric")

    def test_interface_givens_present(self):
        """Optional: check that A and B are provided (but not required for tests)."""
        if self.A is None:
            return (False, "SKIP: A not provided, cannot run givens-dependent tests")
        if self.B is None:
            return (False, "SKIP: B not provided, cannot run givens-dependent tests")
        return (True, "Givens A and B are present")

    # ----------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose for numerical tolerance)
    # ----------------------------------------------------------------------
    def test_invariant_singlet_triplet_splitting(self):
        """
        For any B>0, singlet (E4) must be lower than the triplet energies.
        For B<0, singlet must be higher.
        """
        if self.B is None or self.E4 is None:
            return (False, "SKIP: B or E4 missing")
        if not isinstance(self.B, (int, float)):
            try:
                B_val = float(self.B)
            except:
                return (False, "SKIP: B not numeric")
        else:
            B_val = self.B

        if self.E1 is None or self.E2 is None:
            return (False, "SKIP: needed triplet energies missing")
        try:
            E4_val = float(self.E4)
            E1_val = float(self.E1)
            E2_val = float(self.E2)
        except:
            return (False, "SKIP: cannot evaluate energies numerically")

        if B_val > 0:
            if not (E4_val < E1_val and E4_val < E2_val):
                return (False, f"For B>0, singlet should be lower; E4={E4_val}, E1={E1_val}, E2={E2_val}")
        elif B_val < 0:
            if not (E4_val > E1_val and E4_val > E2_val):
                return (False, f"For B<0, singlet should be higher; E4={E4_val}, E1={E1_val}, E2={E2_val}")
        return (True, f"Singlet-triplet ordering consistent with sign of B={B_val}")

    def test_invariant_energy_order_within_triplet(self):
        """
        For A>0, hbar>0: E1 > E2 > E3 (since M decreases).
        For A<0: reverse.
        """
        if None in (self.A, self.hbar, self.E1, self.E2, self.E3):
            return (False, "SKIP: missing required for ordering test")
        try:
            A_val = float(self.A)
            hbar_val = float(self.hbar)
            E1_val = float(self.E1)
            E2_val = float(self.E2)
            E3_val = float(self.E3)
        except:
            return (False, "SKIP: cannot evaluate numeric")

        if hbar_val <= 0:
            return (False, "SKIP: hbar should be positive")

        product = A_val * hbar_val
        if product > 0:
            if not (E1_val > E2_val > E3_val):
                return (False, f"A>0: expected E1>{'>E2'>E3}; got E1={E1_val}, E2={E2_val}, E3={E3_val}")
        elif product < 0:
            if not (E1_val < E2_val < E3_val):
                return (False, f"A<0: expected E1<E2<E3; got E1={E1_val}, E2={E2_val}, E3={E3_val}")
        else: # A=0
            if not math.isclose(E1_val, E2_val, rel_tol=1e-12) or not math.isclose(E2_val, E3_val, rel_tol=1e-12):
                return (False, f"A=0: expected all triplet equal; got {E1_val}, {E2_val}, {E3_val}")
        return (True, "Triplet energy ordering consistent with sign of A*hbar")

    def test_invariant_symmetry_around_E2(self):
        """
        E1 and E3 should be symmetric about E2: (E1+E3)/2 == E2.
        """
        if None in (self.E1, self.E2, self.E3):
            return (False, "SKIP: missing E1/E2/E3")
        try:
            E1_val = float(self.E1)
            E2_val = float(self.E2)
            E3_val = float(self.E3)
        except:
            return (False, "SKIP: cannot evaluate numeric")
        avg = (E1_val + E3_val) / 2.0
        if not math.isclose(avg, E2_val, rel_tol=1e-5, abs_tol=1e-8):
            return (False, f"Symmetry broken: (E1+E3)/2 = {avg}, E2 = {E2_val}")
        return (True, "E1 and E3 symmetric about E2")

    def test_invariant_s1dot2_eigenvalue_triplet(self):
        """
        For triplet, s1·s2 eigenvalue = hbar^2/4.
        Re-derived from E2 (since M=0) or from (E1+E3)/2.
        """
        if None in (self.E2, self.hbar, self.B):
            return (False, "SKIP: missing required for s1·s2 eigenvalue check")
        try:
            B_val = float(self.B)
            hbar_val = float(self.hbar)
            E2_val = float(self.E2)
        except:
            return (False, "SKIP: cannot evaluate numeric")
        expected = (hbar_val**2) * B_val / 4.0
        if not math.isclose(E2_val, expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"Triplet s1·s2 eigenvalue inconsistent: E2={E2_val}, expected hbar^2 B/4 = {expected}")
        return (True, "Triplet s1·s2 eigenvalue matches hbar^2/4")

    def test_invariant_s1dot2_eigenvalue_singlet(self):
        """
        For singlet (S=0), s1·s2 eigenvalue = -3 hbar^2/4.
        Derive from E4 = -3 hbar^2 B/4.
        """
        if None in (self.E4, self.hbar, self.B):
            return (False, "SKIP: missing for singlet s1·s2 check")
        try:
            B_val = float(self.B)
            hbar_val = float(self.hbar)
            E4_val = float(self.E4)
        except:
            return (False, "SKIP: cannot evaluate numeric")
        expected = -3.0 * (hbar_val**2) * B_val / 4.0
        if not math.isclose(E4_val, expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"Singlet s1·s2 eigenvalue inconsistent: E4={E4_val}, expected -3 hbar^2 B/4 = {expected}")
        return (True, "Singlet s1·s2 eigenvalue matches -3 hbar^2/4")

    def test_invariant_energy_difference_triplet(self):
        """
        E1 - E2 = hbar A, and E2 - E3 = hbar A.
        """
        if None in (self.A, self.hbar, self.E1, self.E2, self.E3):
            return (False, "SKIP: missing for difference check")
        try:
            A_val = float(self.A)
            hbar_val = float(self.hbar)
            d12 = float(self.E1) - float(self.E2)
            d23 = float(self.E2) - float(self.E3)
        except:
            return (False, "SKIP: cannot evaluate numeric")
        expected = hbar_val * A_val
        if not math.isclose(d12, expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"E1 - E2 = {d12}, expected hbar A = {expected}")
        if not math.isclose(d23, expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"E2 - E3 = {d23}, expected hbar A = {expected}")
        return (True, "Energy differences within triplet consistent")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (use explanation steps if available)
    # ----------------------------------------------------------------------
    def test_modeling_sz_eigenvalue_used(self):
        """
        Check that explanation steps (if present) contain correct Sz eigenvalue usage.
        This is a light consistency check, not a full replay.
        """
        explanation = getattr(self.sol, 'explanation_steps', None)
        if explanation is None:
            return (False, "SKIP: explanation_steps not provided")
        text = str(explanation).lower()
        # Look for mention of "S_z" or "Sz eigenvalue" or "hbar M" etc.
        if "sz" in text or "s_z" in text or "m_s" in text or "hbar" in text:
            return (True, "Explanation_steps contains Sz eigenvalue references (light check)")
        return (False, "SKIP: no Sz reference found in explanation")

    def test_modeling_coupled_basis_diagonalization(self):
        """
        Check that the solution (explanation/pseudocode) acknowledges the coupled basis |S,M>.
        """
        pseudocode = getattr(self.sol, 'solving_pseudocode', None)
        if pseudocode:
            if 'coupled' in str(pseudocode).lower() or '|s,m>' in str(pseudocode).lower():
                return (True, "Pseudocode mentions coupled basis")
        explanation = getattr(self.sol, 'explanation_steps', None)
        if explanation:
            if 'χ' in str(explanation) or '|s,m>' in str(explanation).lower():
                return (True, "Explanation uses coupled basis notation")
        return (False, "SKIP: no mention of coupled basis; not required but would be good")

    # ----------------------------------------------------------------------
    # Monotonicity tests (explicitly perturb an input and compare perturbed vs baseline)
    # ----------------------------------------------------------------------
    def test_monotonicity_increase_A_increases_E1(self):
        """
        Perturb A by +10%. Using the energy formula E1 = hbar*A + hbar^2*B/4,
        the perturbed E1 should be larger than the baseline E1.
        """
        if None in (self.A, self.hbar, self.B, self.E1):
            return (False, "SKIP: missing A/hbar/B/E1 for monotonicity")
        try:
            A_val = float(self.A)
            hbar_val = float(self.hbar)
            B_val = float(self.B)
            E1_base = float(self.E1)
        except:
            return (False, "SKIP: cannot evaluate numeric")

        # Choose perturbation: increase A by 10%
        delta_A = 0.1 * A_val
        A_pert = A_val + delta_A

        # Compute perturbed E1 using the same physics formula
        E1_pert = hbar_val * A_pert + (hbar_val**2) * B_val / 4.0

        # Compare: E1_pert should be > E1_base
        if not (E1_pert > E1_base):
            return (False, f"Perturbed E1 ({E1_pert}) not > baseline E1 ({E1_base}); delta_A = {delta_A}")
        # Also check that the difference matches hbar * delta_A
        diff_expected = hbar_val * delta_A
        diff_actual = E1_pert - E1_base
        if not math.isclose(diff_actual, diff_expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"Difference {diff_actual} not close to expected {diff_expected}")
        return (True, f"A increase of {delta_A} increases E1 as expected (diff = {diff_actual})")

    def test_monotonicity_increase_A_decreases_E3(self):
        """
        Perturb A by +10%. E3 = -hbar*A + hbar^2*B/4, so it should decrease.
        """
        if None in (self.A, self.hbar, self.B, self.E3):
            return (False, "SKIP: missing A/hbar/B/E3 for monotonicity")
        try:
            A_val = float(self.A)
            hbar_val = float(self.hbar)
            B_val = float(self.B)
            E3_base = float(self.E3)
        except:
            return (False, "SKIP: cannot evaluate numeric")

        delta_A = 0.1 * A_val
        A_pert = A_val + delta_A

        E3_pert = -hbar_val * A_pert + (hbar_val**2) * B_val / 4.0

        if not (E3_pert < E3_base):
            return (False, f"Perturbed E3 ({E3_pert}) not < baseline E3 ({E3_base})")
        diff_expected = -hbar_val * delta_A
        diff_actual = E3_pert - E3_base
        if not math.isclose(diff_actual, diff_expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"Difference {diff_actual} not close to expected {diff_expected}")
        return (True, f"A increase of {delta_A} decreases E3 as expected (diff = {diff_actual})")

    def test_monotonicity_increase_B_increases_E1_and_decreases_E4(self):
        """
        Perturb B by +10%. E1 = hbar*A + hbar^2*B/4, so E1 increases.
        E4 = -3*hbar^2*B/4, so E4 decreases (becomes more negative if B>0, less positive if B<0).
        We'll check that E1 increases and E4 decreases (since derivative dE4/dB = -3hbar^2/4 <0).
        """
        if None in (self.B, self.hbar, self.A, self.E1, self.E4):
            return (False, "SKIP: missing B/hbar/A/E1/E4 for monotonicity")
        try:
            B_val = float(self.B)
            hbar_val = float(self.hbar)
            A_val = float(self.A)
            E1_base = float(self.E1)
            E4_base = float(self.E4)
        except:
            return (False, "SKIP: cannot evaluate numeric")

        delta_B = 0.1 * B_val
        B_pert = B_val + delta_B

        # Compute perturbed energies
        hbar_sq = hbar_val**2
        E1_pert = hbar_val * A_val + hbar_sq * B_pert / 4.0
        E4_pert = -3.0 * hbar_sq * B_pert / 4.0

        # Check E1 increases
        if not (E1_pert > E1_base):
            return (False, f"Perturbed E1 ({E1_pert}) not > baseline E1 ({E1_base})")
        # Check E4 decreases: E4_pert < E4_base
        if not (E4_pert < E4_base):
            return (False, f"Perturbed E4 ({E4_pert}) not < baseline E4 ({E4_base})")
        # Check differences match derivatives
        dE1_expected = hbar_sq * delta_B / 4.0
        dE1_actual = E1_pert - E1_base
        if not math.isclose(dE1_actual, dE1_expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"E1 difference {dE1_actual} not close to expected {dE1_expected}")
        dE4_expected = -3.0 * hbar_sq * delta_B / 4.0
        dE4_actual = E4_pert - E4_base
        if not math.isclose(dE4_actual, dE4_expected, rel_tol=1e-3, abs_tol=1e-8):
            return (False, f"E4 difference {dE4_actual} not close to expected {dE4_expected}")
        return (True, f"B increase of {delta_B} increases E1 and decreases E4 as expected")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (do not compare exact values)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_non_nan(self):
        """All energies must be finite real numbers."""
        for key in ['E1','E2','E3','E4']:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"Missing {key}")
            try:
                v = float(val)
                if math.isnan(v) or math.isinf(v):
                    return (False, f"{key} is NaN or Inf")
            except:
                return (False, f"{key} cannot be converted to float")
        return (True, "All energies are finite real numbers")

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)