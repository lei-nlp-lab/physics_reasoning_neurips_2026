import math
import sympy as sp
from typing import Any, List, Tuple


class PhysicsChecker:
    """
    Auditor for the problem: sphere with a spherical hollow.
    Validates electric field and potential at the hollow center using superposition.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Input givens – use None defaults (never numeric fallbacks)
        self.R1 = getattr(solution_instance, 'R1', None)
        self.R2 = getattr(solution_instance, 'R2', None)
        self.a  = getattr(solution_instance, 'a',  None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.eps0 = getattr(solution_instance, 'eps0', None)

        # Results dictionary (expected keys: E_total, V_total)
        self.results = getattr(solution_instance, 'results', None) or {}
        self.E_total = self.results.get('E_total', None)
        self.V_total = self.results.get('V_total', None)

        # Optional step-level trace
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _to_float(self, value: Any, name: str) -> float | None:
        """Try to convert value to float; return None on failure."""
        if value is None:
            return None
        try:
            if isinstance(value, sp.Basic):
                return float(sp.N(value))
            return float(value)
        except (TypeError, ValueError, sp.SympifyError):
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the results dictionary contains the mandatory keys."""
        if self.results is None:
            return (False, "No results dictionary found")
        if self.E_total is None:
            return (False, "Missing key 'E_total' in results")
        if self.V_total is None:
            return (False, "Missing key 'V_total' in results")
        return (True, "All required results keys (E_total, V_total) are present")

    def test_interface_givens_not_none(self) -> Tuple[bool, str]:
        """Ensure all input givens are provided (not None)."""
        missing = []
        for name, value in [("R1", self.R1), ("R2", self.R2),
                            ("a", self.a), ("rho", self.rho),
                            ("eps0", self.eps0)]:
            if value is None:
                missing.append(name)
        if missing:
            return (False, f"Missing required input(s): {', '.join(missing)}")
        return (True, "All input givens are present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_superposition_E(self) -> Tuple[bool, str]:
        """
        The total field at the hollow centre must equal the field from the big sphere alone
        because the field from the small sphere at its own centre is zero.
        """
        if self.E_total is None:
            return (False, "SKIP: E_total not available")
        required = [self.rho, self.eps0, self.a]
        if any(v is None for v in required):
            return (False, "SKIP: missing required givens (rho, eps0, a)")

        rho  = self._to_float(self.rho, "rho")
        eps0 = self._to_float(self.eps0, "eps0")
        a    = self._to_float(self.a, "a")
        if any(v is None for v in [rho, eps0, a]):
            return (False, "Cannot convert rho, eps0, or a to float")

        expected_E = rho * a / (3.0 * eps0)
        actual_E = self._to_float(self.E_total, "E_total")
        if actual_E is None:
            return (False, "Cannot convert E_total to float")

        if math.isclose(actual_E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (True,
                    f"E_total = {actual_E} ≈ expected E = {expected_E}")
        else:
            return (False,
                    f"E_total = {actual_E} differs from expected {expected_E}")

    def test_invariant_superposition_V(self) -> Tuple[bool, str]:
        """
        Check that V_total matches the superposition:
        V_total = V(big sphere at distance a) - V(small sphere at its centre)
               = (rho/(6*eps0))*(3*R1^2 - a^2) - (rho/(2*eps0))*R2^2
        """
        if self.V_total is None:
            return (False, "SKIP: V_total not available")
        required = [self.rho, self.eps0, self.R1, self.R2, self.a]
        if any(v is None for v in required):
            return (False, "SKIP: missing required givens")

        rho  = self._to_float(self.rho, "rho")
        eps0 = self._to_float(self.eps0, "eps0")
        R1   = self._to_float(self.R1, "R1")
        R2   = self._to_float(self.R2, "R2")
        a    = self._to_float(self.a, "a")
        if any(v is None for v in [rho, eps0, R1, R2, a]):
            return (False, "Cannot convert one or more givens to float")

        expected_V = (rho / (6.0 * eps0)) * (3.0 * R1**2 - a**2 - 3.0 * R2**2)
        actual_V = self._to_float(self.V_total, "V_total")
        if actual_V is None:
            return (False, "Cannot convert V_total to float")

        if math.isclose(actual_V, expected_V, rel_tol=1e-2, abs_tol=1e-2):
            return (True,
                    f"V_total = {actual_V} ≈ expected V = {expected_V}")
        else:
            return (False,
                    f"V_total = {actual_V} differs from expected {expected_V}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_geometric_constraint(self) -> Tuple[bool, str]:
        """
        Modeling quality check: the hollow sphere must lie entirely inside the big sphere.
        (a + R2 ≤ R1)
        """
        if any(v is None for v in [self.R1, self.R2, self.a]):
            return (False, "SKIP: missing R1, R2, or a")
        R1 = self._to_float(self.R1, "R1")
        R2 = self._to_float(self.R2, "R2")
        a  = self._to_float(self.a, "a")
        if any(v is None for v in [R1, R2, a]):
            return (False, "Cannot convert R1, R2, or a to float")
        if a + R2 <= R1:
            return (True, f"Hollow inside sphere: a+R2 = {a+R2} ≤ R1 = {R1}")
        else:
            return (False, f"Constraint violated: a+R2 = {a+R2} > R1 = {R1}")

    def test_modeling_superposition_applied(self) -> Tuple[bool, str]:
        """
        Check that the solution's reasoning (pseudocode or explanation steps)
        explicitly includes the superposition principle.
        """
        if self.pseudocode is None and self.explanation_steps is None:
            return (False, "SKIP: no pseudocode or explanation_steps available")
        texts = []
        if self.pseudocode is not None:
            if isinstance(self.pseudocode, list):
                texts.extend(self.pseudocode)
            elif isinstance(self.pseudocode, str):
                texts.append(self.pseudocode)
        if self.explanation_steps is not None:
            if isinstance(self.explanation_steps, list):
                texts.extend(self.explanation_steps)
            elif isinstance(self.explanation_steps, str):
                texts.append(self.explanation_steps)
        combined = " ".join(texts).lower()
        keywords = ["superposition", "e_total = e_big + e_small",
                    "e_total = e1 - e2", "superposition principle"]
        found = any(kw in combined for kw in keywords)
        if found:
            return (True, "Superposition principle is explicitly referenced in reasoning")
        else:
            return (False, "No explicit mention of superposition found in reasoning")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare perturbed vs baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_E_total_increases_with_a(self) -> Tuple[bool, str]:
        """
        Electric field magnitude at the hollow centre is proportional to a.
        Increase a by 10% → E_total should increase by 10%.
        """
        required = [self.rho, self.eps0, self.a]
        if any(v is None for v in required):
            return (False, "SKIP: missing rho, eps0, or a")
        rho  = self._to_float(self.rho, "rho")
        eps0 = self._to_float(self.eps0, "eps0")
        a    = self._to_float(self.a, "a")
        if any(v is None for v in [rho, eps0, a]):
            return (False, "Cannot convert rho, eps0, or a to float")
        if a <= 0:
            return (False, "SKIP: a must be positive for meaningful monotonicity test")

        # Baseline from formula
        E_baseline = rho * a / (3.0 * eps0)

        # Perturb a by +10%
        factor = 1.1
        a_pert = a * factor
        E_pert = rho * a_pert / (3.0 * eps0)

        # Relative change must match (factor-1)
        rel_change = (E_pert - E_baseline) / E_baseline
        if not math.isclose(rel_change, factor - 1, rel_tol=1e-3):
            return (False,
                    f"Relative change = {rel_change:.4f}, expected {factor-1:.4f}")
        return (True,
                f"10% increase in a gives {rel_change*100:.1f}% increase in E (expected 10%)")

    def test_monotonicity_V_total_decreases_with_a_squared(self) -> Tuple[bool, str]:
        """
        Potential at hollow centre depends on -a^2.
        Increase a by 10% → V should change by (rho/(6 eps0)) * (-(a_pert^2 - a_baseline^2)).
        Verify the numerical change matches the formula.
        """
        required = [self.rho, self.eps0, self.R1, self.R2, self.a]
        if any(v is None for v in required):
            return (False, "SKIP: missing required givens")
        rho  = self._to_float(self.rho, "rho")
        eps0 = self._to_float(self.eps0, "eps0")
        R1   = self._to_float(self.R1, "R1")
        R2   = self._to_float(self.R2, "R2")
        a    = self._to_float(self.a, "a")
        if any(v is None for v in [rho, eps0, R1, R2, a]):
            return (False, "Cannot convert givens to float")
        if a <= 0:
            return (False, "SKIP: a must be positive")

        # Baseline V (from formula, not from results)
        V_baseline = (rho / (6.0 * eps0)) * (3.0 * R1**2 - a**2 - 3.0 * R2**2)

        # Perturb a by +10%
        factor = 1.1
        a_pert = a * factor
        V_pert = (rho / (6.0 * eps0)) * (3.0 * R1**2 - a_pert**2 - 3.0 * R2**2)

        # The difference should be exactly (rho/(6 eps0)) * (a^2 - a_pert^2)
        expected_diff = (rho / (6.0 * eps0)) * (a**2 - a_pert**2)
        actual_diff = V_pert - V_baseline

        if math.isclose(actual_diff, expected_diff, rel_tol=1e-3, abs_tol=1e-8):
            return (True,
                    f"V change due to 10% a increase = {actual_diff} ≈ expected {expected_diff}")
        else:
            return (False,
                    f"V change = {actual_diff}, expected {expected_diff}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that E_total and V_total are finite numeric values
        and have plausible signs consistent with rho.
        """
        if self.rho is None:
            return (False, "SKIP: rho not available")
        rho = self._to_float(self.rho, "rho")
        if rho is None:
            return (False, "Cannot convert rho to float")

        # E_total sign should match sign of rho (since a > 0)
        if self.E_total is not None:
            E = self._to_float(self.E_total, "E_total")
            if E is None:
                return (False, "Cannot convert E_total to float")
            if not math.isfinite(E):
                return (False, f"E_total is not finite: {E}")
            if (rho > 0 and E < 0) or (rho < 0 and E > 0):
                return (False,
                        f"E_total sign inconsistent: rho={rho}, E={E}")
        else:
            return (False, "SKIP: E_total not available")

        # V_total: for positive rho it is usually positive, but we only check finiteness
        if self.V_total is not None:
            V = self._to_float(self.V_total, "V_total")
            if V is None:
                return (False, "Cannot convert V_total to float")
            if not math.isfinite(V):
                return (False, f"V_total is not finite: {V}")
        else:
            return (False, "SKIP: V_total not available")

        return (True, "E_total and V_total are finite and have plausible signs")

    # ------------------------------------------------------------------
    # Test harness
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test methods, collect results.
        Returns (all_pass, logs).
        """
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)