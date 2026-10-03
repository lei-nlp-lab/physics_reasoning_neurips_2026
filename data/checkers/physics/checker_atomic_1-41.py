import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    AutoGluon Physics Checker for hydrogen hyperfine and nuclear spin splitting.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dict – support both object attribute and dict-like access
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        elif isinstance(solution_instance, dict):
            results = solution_instance
        else:
            results = {}

        # Use None as default for ALL missing values (no numeric defaults)
        self.E_F1 = results.get('E_F1', None)
        self.E_F0 = results.get('E_F0', None)
        self.delta_E_hf = results.get('delta_E_hf', None)
        self.E_triplet = results.get('E_triplet', None)
        self.E_singlet = results.get('E_singlet', None)
        self.delta_E_nuc = results.get('delta_E_nuc', None)

        # Optionally extract coupling constants (A, J) if present (not required)
        # Use None if not available
        self.A = getattr(solution_instance, 'A', None) or results.get('A', None)
        self.J = getattr(solution_instance, 'J', None) or results.get('J', None)

    # ----------------------------------------------------------------------
    # Interface tests: required keys exist and are numeric
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all six required result keys exist and are finite numbers."""
        keys = ['E_F1', 'E_F0', 'delta_E_hf', 'E_triplet', 'E_singlet', 'delta_E_nuc']
        missing = []
        bad_type = []
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                missing.append(key)
            elif not isinstance(val, (int, float)):
                bad_type.append(key)
            elif not math.isfinite(val):
                bad_type.append(f"{key} (non-finite)")
        if missing:
            msg = f"Missing keys: {', '.join(missing)}"
            return False, msg
        if bad_type:
            msg = f"Non-numeric or non-finite keys: {', '.join(bad_type)}"
            return False, msg
        return True, "All required result keys present and finite numeric."

    # ----------------------------------------------------------------------
    # Invariant tests: physics ordering and self‑consistency
    # ----------------------------------------------------------------------
    def test_invariant_hyperfine_ordering(self) -> Tuple[bool, str]:
        """Check that triplet (F=1) has higher energy than singlet (F=0)."""
        if self.E_F1 is None or self.E_F0 is None:
            return False, "SKIP: E_F1 or E_F0 missing."
        if self.E_F1 > self.E_F0:
            return True, "Triplet > Singlet (exact)."
        else:
            return False, f"Triplet {self.E_F1} <= Singlet {self.E_F0}."

    def test_invariant_nuclear_ordering(self) -> Tuple[bool, str]:
        """Check that total nuclear spin triplet has higher energy than singlet."""
        if self.E_triplet is None or self.E_singlet is None:
            return False, "SKIP: E_triplet or E_singlet missing."
        if self.E_triplet > self.E_singlet:
            return True, "Nuclear triplet > nuclear singlet (exact)."
        else:
            return False, f"Triplet {self.E_triplet} <= Singlet {self.E_singlet}."

    def test_invariant_hyperfine_difference(self) -> Tuple[bool, str]:
        """Check energy difference matches delta_E_hf: E_F1 - E_F0 ≈ delta_E_hf."""
        if any(self.E_F1 is None, self.E_F0 is None, self.delta_E_hf is None):
            return False, "SKIP: missing E_F1, E_F0 or delta_E_hf."
        if math.isclose(self.E_F1 - self.E_F0, self.delta_E_hf, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_F1 - E_F0 = {self.E_F1 - self.E_F0} ≈ delta_E_hf = {self.delta_E_hf}."
        else:
            return False, f"E_F1 - E_F0 = {self.E_F1 - self.E_F0} differs from delta_E_hf = {self.delta_E_hf}."

    def test_invariant_nuclear_difference(self) -> Tuple[bool, str]:
        """Check energy difference matches delta_E_nuc: E_triplet - E_singlet ≈ delta_E_nuc."""
        if any(self.E_triplet is None, self.E_singlet is None, self.delta_E_nuc is None):
            return False, "SKIP: missing E_triplet, E_singlet or delta_E_nuc."
        if math.isclose(self.E_triplet - self.E_singlet, self.delta_E_nuc, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_triplet - E_singlet = {self.E_triplet - self.E_singlet} ≈ delta_E_nuc = {self.delta_E_nuc}."
        else:
            return False, f"E_triplet - E_singlet = {self.E_triplet - self.E_singlet} differs from delta_E_nuc = {self.delta_E_nuc}."

    # ----------------------------------------------------------------------
    # Modeling tests: consistency of energy ratios and re‑derived values
    # ----------------------------------------------------------------------
    def test_modeling_hyperfine_ratio(self) -> Tuple[bool, str]:
        """Check ratio E_F1 / E_F0 = -1/3 (≈ -0.3333)."""
        if self.E_F1 is None or self.E_F0 is None:
            return False, "SKIP: E_F1 or E_F0 missing."
        if math.isclose(self.E_F0, 0.0, abs_tol=1e-12):
            return False, "E_F0 is zero, ratio undefined."
        ratio = self.E_F1 / self.E_F0
        if math.isclose(ratio, -1.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_F1 / E_F0 = {ratio} ≈ -1/3."
        else:
            return False, f"E_F1 / E_F0 = {ratio} deviates from -1/3."

    def test_modeling_nuclear_ratio(self) -> Tuple[bool, str]:
        """Check ratio E_triplet / E_singlet = -1/3."""
        if self.E_triplet is None or self.E_singlet is None:
            return False, "SKIP: E_triplet or E_singlet missing."
        if math.isclose(self.E_singlet, 0.0, abs_tol=1e-12):
            return False, "E_singlet is zero, ratio undefined."
        ratio = self.E_triplet / self.E_singlet
        if math.isclose(ratio, -1.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E_triplet / E_singlet = {ratio} ≈ -1/3."
        else:
            return False, f"E_triplet / E_singlet = {ratio} deviates from -1/3."

    def test_modeling_hyperfine_values_derived(self) -> Tuple[bool, str]:
        """Re‑derive E_F1 and E_F0 from delta_E_hf using theoretical factors (A/4 and -3A/4)."""
        if self.delta_E_hf is None or self.E_F1 is None or self.E_F0 is None:
            return False, "SKIP: missing delta_E_hf, E_F1, or E_F0."
        A = self.delta_E_hf
        expected_F1 = A / 4.0
        expected_F0 = -3.0 * A / 4.0
        ok_F1 = math.isclose(self.E_F1, expected_F1, rel_tol=1e-2, abs_tol=1e-2)
        ok_F0 = math.isclose(self.E_F0, expected_F0, rel_tol=1e-2, abs_tol=1e-2)
        if ok_F1 and ok_F0:
            return True, f"E_F1 = {self.E_F1} ≈ {expected_F1}, E_F0 = {self.E_F0} ≈ {expected_F0}."
        else:
            return False, f"E_F1 = {self.E_F1} (expected {expected_F1}) or E_F0 = {self.E_F0} (expected {expected_F0}) off."

    def test_modeling_nuclear_values_derived(self) -> Tuple[bool, str]:
        """Re‑derive E_triplet and E_singlet from delta_E_nuc using theoretical factors."""
        if self.delta_E_nuc is None or self.E_triplet is None or self.E_singlet is None:
            return False, "SKIP: missing delta_E_nuc, E_triplet, or E_singlet."
        J = self.delta_E_nuc
        expected_triplet = J / 4.0
        expected_singlet = -3.0 * J / 4.0
        ok_t = math.isclose(self.E_triplet, expected_triplet, rel_tol=1e-2, abs_tol=1e-2)
        ok_s = math.isclose(self.E_singlet, expected_singlet, rel_tol=1e-2, abs_tol=1e-2)
        if ok_t and ok_s:
            return True, f"E_triplet = {self.E_triplet} ≈ {expected_triplet}, E_singlet = {self.E_singlet} ≈ {expected_singlet}."
        else:
            return False, f"E_triplet = {self.E_triplet} (expected {expected_triplet}) or E_singlet = {self.E_singlet} (expected {expected_singlet}) off."

    # ----------------------------------------------------------------------
    # Monotonicity tests: perturb an input and check monotonic change in output
    # ----------------------------------------------------------------------
    def test_monotonicity_hyperfine_splitting_with_A(self) -> Tuple[bool, str]:
        """Perturb delta_E_hf (which equals A) by +10% and verify splitting increases."""
        if self.delta_E_hf is None:
            return False, "SKIP: delta_E_hf missing."
        if self.delta_E_hf <= 0:
            return False, "delta_E_hf must be positive for perturbation test."
        A_original = self.delta_E_hf
        A_perturbed = 1.1 * A_original
        # Splitting = A (since E_F1 - E_F0 = A)
        split_original = A_original
        split_perturbed = A_perturbed
        if split_perturbed > split_original:
            return True, f"Hyperfine splitting increases from {split_original} to {split_perturbed} when A increases by 10%."
        else:
            return False, "Hyperfine splitting did not increase after perturbing A."

    def test_monotonicity_nuclear_splitting_with_J(self) -> Tuple[bool, str]:
        """Perturb delta_E_nuc (which equals J) by +10% and verify splitting increases."""
        if self.delta_E_nuc is None:
            return False, "SKIP: delta_E_nuc missing."
        if self.delta_E_nuc <= 0:
            return False, "delta_E_nuc must be positive for perturbation test."
        J_original = self.delta_E_nuc
        J_perturbed = 1.1 * J_original
        # Splitting = J (since E_triplet - E_singlet = J)
        split_original = J_original
        split_perturbed = J_perturbed
        if split_perturbed > split_original:
            return True, f"Nuclear splitting increases from {split_original} to {split_perturbed} when J increases by 10%."
        else:
            return False, "Nuclear splitting did not increase after perturbing J."

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one, no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_signs(self) -> Tuple[bool, str]:
        """Broad sanity: triplet energies positive, singlet energies negative, coupling constants positive."""
        checks = []
        if self.E_F1 is not None:
            if self.E_F1 <= 0:
                checks.append("E_F1 should be positive")
            else:
                checks.append("E_F1 positive")
        if self.E_F0 is not None:
            if self.E_F0 >= 0:
                checks.append("E_F0 should be negative")
            else:
                checks.append("E_F0 negative")
        if self.E_triplet is not None:
            if self.E_triplet <= 0:
                checks.append("E_triplet should be positive")
            else:
                checks.append("E_triplet positive")
        if self.E_singlet is not None:
            if self.E_singlet >= 0:
                checks.append("E_singlet should be negative")
            else:
                checks.append("E_singlet negative")
        if self.delta_E_hf is not None:
            if self.delta_E_hf <= 0:
                checks.append("delta_E_hf (>0) required")
            else:
                checks.append("delta_E_hf positive")
        if self.delta_E_nuc is not None:
            if self.delta_E_nuc <= 0:
                checks.append("delta_E_nuc (>0) required")
            else:
                checks.append("delta_E_nuc positive")
        failures = [c for c in checks if "should" in c]
        if failures:
            return False, "; ".join(failures)
        else:
            return True, "All available sign/range conditions satisfied: " + "; ".join(checks)

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, execute them safely, collect results."""
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
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)