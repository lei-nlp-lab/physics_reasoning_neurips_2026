import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Physics checker for the atom-near-wall problem.
    Audits solutions for correctness, focusing on interface completeness,
    physical invariants, modeling quality, monotonicity, and final-answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens (assume direct attributes or a 'givens' dict)
        givens = getattr(solution_instance, 'givens', {})
        self.m = getattr(solution_instance, 'm', givens.get('m', None))
        self.V0 = getattr(solution_instance, 'V0', givens.get('V0', None))
        self.d = getattr(solution_instance, 'd', givens.get('d', None))
        self.hbar = getattr(solution_instance, 'hbar', givens.get('hbar', None))

        # Extract results (expected keys: delta_E, d0, V0d_min)
        self.results = getattr(solution_instance, 'results', {})
        self.delta_E = self.results.get('delta_E', None)
        self.d0 = self.results.get('d0', None)
        self.V0d_min = self.results.get('V0d_min', None)

        # Derived quantities (only if givens are all available)
        self._derived_ok = (
            self.m is not None and
            self.V0 is not None and
            self.d is not None and
            self.hbar is not None
        )
        if self._derived_ok:
            hbar2 = self.hbar ** 2
            self.kappa0 = self.m * self.V0 / hbar2
            self.hbar2 = hbar2

    # --------------------------------------------------------------------------
    # Interface checks
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present and numeric."""
        if self.delta_E is None:
            return (False, "Missing key 'delta_E' (None)")
        if self.d0 is None:
            return (False, "Missing key 'd0' (None)")
        if self.V0d_min is None:
            return (False, "Missing key 'V0d_min' (None)")
        # Optionally check types
        for val, name in [(self.delta_E, 'delta_E'), (self.d0, 'd0'), (self.V0d_min, 'V0d_min')]:
            if not isinstance(val, (int, float)):
                return (False, f"Key '{name}' is not numeric (type {type(val).__name__})")
        return (True, "All required result keys present and numeric")

    # --------------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # --------------------------------------------------------------------------
    def test_invariant_d0_definition(self) -> Tuple[bool, str]:
        """
        d0 = hbar^2 / (m * V0)  (characteristic length from far‑away condition).
        """
        if not self._derived_ok or self.d0 is None:
            return (False, "SKIP: missing givens or d0")
        expected_d0 = self.hbar2 / (self.m * self.V0)
        if math.isclose(self.d0, expected_d0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"d0 = {self.d0} matches expected {expected_d0}")
        else:
            return (False, f"d0 = {self.d0} != expected {expected_d0}")

    def test_invariant_V0d_min_condition(self) -> Tuple[bool, str]:
        """
        V0d_min = hbar^2 / (2m)   (exact existence threshold).
        """
        if self.hbar is None or self.m is None or self.V0d_min is None:
            return (False, "SKIP: missing hbar, m, or V0d_min")
        expected_threshold = self.hbar ** 2 / (2 * self.m)
        if math.isclose(self.V0d_min, expected_threshold, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V0d_min = {self.V0d_min} matches expected {expected_threshold}")
        else:
            return (False, f"V0d_min = {self.V0d_min} != expected {expected_threshold}")

    # --------------------------------------------------------------------------
    # Modeling / trace checks (quality checks based on explanation steps)
    # --------------------------------------------------------------------------
    def test_modeling_delta_E_formula(self) -> Tuple[bool, str]:
        """
        The perturbative expression for delta_E:
        delta_E = (m V0^2 / hbar^2) * exp(-2 m V0 d / hbar^2)
        """
        if not self._derived_ok or self.delta_E is None:
            return (False, "SKIP: missing givens or delta_E")
        kappa0 = self.kappa0
        d = self.d
        hbar2 = self.hbar2
        expected = (self.m * self.V0 ** 2 / hbar2) * math.exp(-2 * kappa0 * d)
        if math.isclose(self.delta_E, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_E = {self.delta_E} matches expected {expected}")
        else:
            return (False, f"delta_E = {self.delta_E} != expected {expected}")

    def test_modeling_d0_from_decay_length(self) -> Tuple[bool, str]:
        """
        d0 is exactly the characteristic decay length of the bound state in the
        far‑away limit: d0 = 1/kappa0. This is derived in the explanation steps.
        """
        if not self._derived_ok or self.d0 is None:
            return (False, "SKIP: missing givens or d0")
        expected_d0 = 1.0 / self.kappa0
        if math.isclose(self.d0, expected_d0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"d0 = {self.d0} matches 1/kappa0 = {expected_d0}")
        else:
            return (False, f"d0 = {self.d0} != 1/kappa0 = {expected_d0}")

    # --------------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input parameter)
    # --------------------------------------------------------------------------
    def test_monotonicity_delta_E_decreases_with_d(self) -> Tuple[bool, str]:
        """
        Monotonicity: perturb d → larger d ⇒ delta_E should decrease.
        We recompute delta_E using the theoretical formula to obtain baseline
        and perturbed values, then compare.
        """
        if not self._derived_ok or self.delta_E is None:
            return (False, "SKIP: missing givens or delta_E")
        kappa0 = self.kappa0
        d = self.d
        hbar2 = self.hbar2
        # baseline delta_E (using the same formula the user should have used)
        baseline = (self.m * self.V0 ** 2 / hbar2) * math.exp(-2 * kappa0 * d)
        # perturb d by +5%
        d_perturbed = d * 1.05
        perturbed = (self.m * self.V0 ** 2 / hbar2) * math.exp(-2 * kappa0 * d_perturbed)
        if perturbed < baseline - 1e-15:   # strict inequality, with some tolerance
            return (True, f"delta_E decreased from {baseline} to {perturbed} for +5% d")
        else:
            return (False, f"delta_E did not decrease: baseline={baseline}, perturbed={perturbed}")

    # --------------------------------------------------------------------------
    # Answer sanity (weak, no re‑derivation or math.isclose)
    # --------------------------------------------------------------------------
    def test_answer_sanity_positivity(self) -> Tuple[bool, str]:
        """
        Weak sanity: delta_E > 0, d0 > 0, V0d_min > 0.
        """
        failures = []
        if self.delta_E is not None and not (self.delta_E > 0):
            failures.append("delta_E should be positive")
        if self.d0 is not None and not (self.d0 > 0):
            failures.append("d0 should be positive")
        if self.V0d_min is not None and not (self.V0d_min > 0):
            failures.append("V0d_min should be positive")
        if failures:
            return (False, "; ".join(failures))
        return (True, "All values positive (delta_E, d0, V0d_min)")

    # --------------------------------------------------------------------------
    # Test runner (auto‑discovers all test_* methods)
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Iterate over all bound methods whose name begins with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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