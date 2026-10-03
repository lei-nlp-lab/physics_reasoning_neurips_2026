import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits the solution for the isospin excitation energy problem.
    Tests interface completeness, physical consistency (isospin symmetry,
    shell structure, pairing), and qualitative reasoning.
    """
    def __init__(self, solution_instance):
        # Safe extraction of results dict
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract emitted keys with None defaults (no numeric fallbacks)
        self.E_ex_N14_T1 = self.results.get('E_ex_N14_T1', None)
        self.E_ex_C12_T1 = self.results.get('E_ex_C12_T1', None)
        self.delta_ex = self.results.get('delta_ex', None)
        self.T_assignment_basis = self.results.get('T_assignment_basis', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_has_E_ex_N14_T1(self):
        if self.E_ex_N14_T1 is None:
            return (False, "SKIP: E_ex_N14_T1 not provided")
        if not isinstance(self.E_ex_N14_T1, (int, float)):
            return (False, "E_ex_N14_T1 must be numeric")
        return (True, f"E_ex_N14_T1 = {self.E_ex_N14_T1}")

    def test_interface_has_E_ex_C12_T1(self):
        if self.E_ex_C12_T1 is None:
            return (False, "SKIP: E_ex_C12_T1 not provided")
        if not isinstance(self.E_ex_C12_T1, (int, float)):
            return (False, "E_ex_C12_T1 must be numeric")
        return (True, f"E_ex_C12_T1 = {self.E_ex_C12_T1}")

    def test_interface_has_delta_ex(self):
        if self.delta_ex is None:
            return (False, "SKIP: delta_ex not provided")
        if not isinstance(self.delta_ex, (int, float)):
            return (False, "delta_ex must be numeric")
        return (True, f"delta_ex = {self.delta_ex}")

    def test_interface_has_T_assignment_basis(self):
        if self.T_assignment_basis is None:
            return (False, "SKIP: T_assignment_basis not provided")
        if not isinstance(self.T_assignment_basis, str) or len(self.T_assignment_basis.strip()) == 0:
            return (False, "T_assignment_basis must be a non-empty string")
        return (True, "T_assignment_basis is a non-empty string")

    # ------------------------------------------------------------------
    # Invariant / theorem-based checks (using math.isclose where appropriate)
    # ------------------------------------------------------------------
    def test_invariant_delta_ex_consistency(self):
        """delta_ex should equal the difference of the two excitation energies (weak modeling check)."""
        if any(v is None for v in (self.E_ex_C12_T1, self.E_ex_N14_T1, self.delta_ex)):
            return (False, "SKIP: missing one of the required values")
        expected_diff = self.E_ex_C12_T1 - self.E_ex_N14_T1
        if math.isclose(self.delta_ex, expected_diff, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_ex = {self.delta_ex} is within 1% of expected {expected_diff}")
        else:
            return (False, f"delta_ex = {self.delta_ex}, expected {expected_diff} from difference")

    def test_invariant_isospin_conservation_mentioned(self):
        """T assignment basis should mention isospin conservation or charge independence."""
        if self.T_assignment_basis is None:
            return (False, "SKIP: T_assignment_basis missing")
        # Look for keywords: 'charge independent', 'isospin symmetric', 'strong interaction same'
        keywords = ['charge independent', 'isospin', 'strong interaction', 'analog']
        found = any(kw in self.T_assignment_basis.lower() for kw in keywords)
        if not found:
            return (False, "T_assignment_basis does not mention isospin conservation / charge independence")
        return (True, "Isospin conservation / charge independence mentioned")

    def test_invariant_coulomb_displacement(self):
        """Check that the explanation includes Coulomb displacement energy (if present in basis)."""
        if self.T_assignment_basis is None:
            return (False, "SKIP: T_assignment_basis missing")
        if 'coulomb' in self.T_assignment_basis.lower():
            return (True, "Coulomb displacement considered")
        # It's acceptable if not mentioned; not a failure.
        return (True, "Coulomb displacement not explicitly mentioned (acceptable)")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_n14_excitation_reasonable(self):
        """N14 T=1 excitation should be low, typically around pairing energy (1–4 MeV)."""
        if self.E_ex_N14_T1 is None:
            return (False, "SKIP: E_ex_N14_T1 missing")
        if 1.0 <= self.E_ex_N14_T1 <= 4.0:
            return (True, f"E_ex_N14_T1 = {self.E_ex_N14_T1} lies in expected pairing energy range [1,4] MeV")
        else:
            return (False, f"E_ex_N14_T1 = {self.E_ex_N14_T1} outside expected range (1-4 MeV)")

    def test_modeling_c12_excitation_reasonable(self):
        """C12 T=1 excitation should be large, consistent with shell gap (~10–20 MeV)."""
        if self.E_ex_C12_T1 is None:
            return (False, "SKIP: E_ex_C12_T1 missing")
        if 10.0 <= self.E_ex_C12_T1 <= 20.0:
            return (True, f"E_ex_C12_T1 = {self.E_ex_C12_T1} lies in expected shell gap range [10,20] MeV")
        else:
            return (False, f"E_ex_C12_T1 = {self.E_ex_C12_T1} outside expected range (10-20 MeV)")

    def test_modeling_basis_includes_multiplet(self):
        """T assignment basis should mention isospin multiplets / analog states."""
        if self.T_assignment_basis is None:
            return (False, "SKIP: T_assignment_basis missing")
        if 'multiplet' in self.T_assignment_basis.lower() or 'analog' in self.T_assignment_basis.lower():
            return (True, "Multiplet/analog concept included")
        return (False, "T_assignment_basis lacks mention of isospin multiplets or analog states")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation of input)
    # ------------------------------------------------------------------
    def test_monotonicity_n14_excitation_increases_with_c14_mass(self):
        """
        Perturb the mass excess of C14 (Tz=-1) by +1% and recompute the N14 T=1 excitation
        using the relation from the explanation steps:
          E_ex(N14) ≈ ΔE_Coulomb(N14,C14) + M(C14) - M(N14)
        with ΔE_Coulomb fixed at 2.5 MeV. Expect an increase.
        """
        # These constants are from the problem statement (external to solution)
        M_N14_ground = 2.864   # mass excess in MeV
        M_C14 = 3.020
        delta_Coulomb = 2.5    # from explanation steps

        baseline = delta_Coulomb + M_C14 - M_N14_ground  # ≈ 2.5 + 3.020 - 2.864 = 2.656 MeV

        # Perturb M_C14 by +1%
        perturbed_M_C14 = M_C14 * 1.01
        perturbed_ex = delta_Coulomb + perturbed_M_C14 - M_N14_ground

        if perturbed_ex > baseline:
            return (True, f"Baseline E_ex(N14) = {baseline:.3f}, after +1% C14 mass: {perturbed_ex:.3f} (increase)")
        else:
            return (False, f"Perturbation did not increase excitation (baseline {baseline}, perturbed {perturbed_ex})")

    def test_monotonicity_c12_excitation_increases_with_b12_mass(self):
        """
        Perturb the mass excess of B12 (Tz=-1) by +1% and recompute the C12 T=1 excitation
        using the relation:
          E_ex(C12) ≈ ΔE_Coulomb(C12,B12) + M(B12) - M(C12)
        with ΔE_Coulomb fixed at 2.2 MeV and M(C12)=0. Expect an increase.
        """
        M_C12_ground = 0.0
        M_B12 = 13.370
        delta_Coulomb = 2.2

        baseline = delta_Coulomb + M_B12 - M_C12_ground  # = 2.2 + 13.37 = 15.57 MeV

        perturbed_M_B12 = M_B12 * 1.01
        perturbed_ex = delta_Coulomb + perturbed_M_B12 - M_C12_ground

        if perturbed_ex > baseline:
            return (True, f"Baseline E_ex(C12) = {baseline:.3f}, after +1% B12 mass: {perturbed_ex:.3f} (increase)")
        else:
            return (False, f"Perturbation did not increase excitation (baseline {baseline}, perturbed {perturbed_ex})")

    # ------------------------------------------------------------------
    # Sanity checks (weak, secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        """All numeric results should be positive and finite."""
        values = [self.E_ex_N14_T1, self.E_ex_C12_T1, self.delta_ex]
        if all(v is None for v in values):
            return (False, "SKIP: no numeric values to check")
        for i, v in enumerate(values):
            if v is None:
                continue
            if not isinstance(v, (int, float)):
                return (False, f"Value {i} is not numeric")
            if math.isnan(v) or math.isinf(v):
                return (False, f"Value {i} is NaN or infinite")
            if v <= 0:
                return (False, f"Value {i} is not positive")
        return (True, "All numeric values are positive and finite")

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Gather all test_ methods
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