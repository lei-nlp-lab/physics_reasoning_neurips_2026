import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    """
    Audits the solution of a fuel cell EMF calculation.
    Checks interface completeness, thermodynamic invariants, modeling quality,
    monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract all required givens. Use None defaults.
        # Try common locations: solution_instance.givens, then parameters, then results.
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = getattr(solution_instance, 'parameters', None)
        if givens is None:
            givens = getattr(solution_instance, 'results', {})

        # Enthalpies (J/mol)
        self.H_H2 = givens.get('H_H2', None) if isinstance(givens, dict) else None
        self.H_O2 = givens.get('H_O2', None)
        self.H_H2O = givens.get('H_H2O', None)

        # Entropies (J/(mol·K))
        self.S_H2 = givens.get('S_H2', None)
        self.S_O2 = givens.get('S_O2', None)
        self.S_H2O = givens.get('S_H2O', None)

        # Temperature (K), Faraday constant, electrons transferred
        self.T = givens.get('T', None)
        self.F = givens.get('F', None)
        self.n = givens.get('n', None)

        # Also try to get them directly from the instance (fallback)
        for attr in ['H_H2', 'H_O2', 'H_H2O', 'S_H2', 'S_O2', 'S_H2O', 'T', 'F', 'n']:
            val = getattr(solution_instance, attr, None)
            if val is not None:
                setattr(self, attr, val)

        # Result: emf (V) – may be stored in results dict or as attribute
        results = getattr(solution_instance, 'results', {})
        self.E = results.get('E', None)
        if self.E is None:
            self.E = getattr(solution_instance, 'E', None)

        # Optional intermediate results for deeper checks
        self.deltaH = results.get('deltaH', None)
        self.deltaS = results.get('deltaS', None)
        self.deltaG = results.get('deltaG', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that all necessary input parameters are present and numeric."""
        missing = []
        for name, val in [('H_H2', self.H_H2), ('H_O2', self.H_O2), ('H_H2O', self.H_H2O),
                          ('S_H2', self.S_H2), ('S_O2', self.S_O2), ('S_H2O', self.S_H2O),
                          ('T', self.T), ('F', self.F), ('n', self.n)]:
            if val is None:
                missing.append(name)
        if missing:
            return False, f"Missing parameters: {', '.join(missing)}"
        # Check they are numeric (int or float)
        non_numeric = [name for name, val in
                       [('H_H2', self.H_H2), ('H_O2', self.H_O2), ('H_H2O', self.H_H2O),
                        ('S_H2', self.S_H2), ('S_O2', self.S_O2), ('S_H2O', self.S_H2O),
                        ('T', self.T), ('F', self.F), ('n', self.n)]
                       if not isinstance(val, (int, float))]
        if non_numeric:
            return False, f"Non-numeric parameters: {', '.join(non_numeric)}"
        return True, "All required givens present and numeric."

    def test_interface_result_E(self) -> Tuple[bool, str]:
        """Check that the EMF result exists and is numeric."""
        if self.E is None:
            return False, "Result E not found in solution."
        if not isinstance(self.E, (int, float)):
            return False, f"Result E is not numeric: {type(self.E)}"
        if math.isnan(self.E) or math.isinf(self.E):
            return False, f"Result E is NaN or infinite: {self.E}"
        return True, f"Result E = {self.E} is numeric and finite."

    # ------------------------------------------------------------------
    # Invariant (physics law) checks – use math.isclose
    # ------------------------------------------------------------------
    def _compute_deltaH(self) -> float:
        """Compute ΔH from givens (if available)."""
        return self.H_H2O - self.H_H2 - 0.5 * self.H_O2

    def test_invariant_deltaH(self) -> Tuple[bool, str]:
        """Check ΔH = H_H2O - H_H2 - 0.5*H_O2."""
        if None in (self.H_H2, self.H_O2, self.H_H2O):
            return False, "SKIP: missing enthalpy data."
        expected = self._compute_deltaH()
        # If the solution provided deltaH, compare; otherwise just compute and report.
        if self.deltaH is not None:
            if math.isclose(self.deltaH, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"ΔH ≈ {expected} J/mol matches solution's ΔH = {self.deltaH}"
            else:
                return False, f"ΔH mismatch: expected {expected}, got {self.deltaH}"
        else:
            # No solution deltaH – we just verify the formula self-consistency
            # (not a test of the solution, but of the given data)
            return True, f"ΔH = {expected} J/mol (computed from givens)."

    def _compute_deltaS(self) -> float:
        """Compute ΔS from givens."""
        return self.S_H2O - self.S_H2 - 0.5 * self.S_O2

    def test_invariant_deltaS(self) -> Tuple[bool, str]:
        """Check ΔS = S_H2O - S_H2 - 0.5*S_O2."""
        if None in (self.S_H2, self.S_O2, self.S_H2O):
            return False, "SKIP: missing entropy data."
        expected = self._compute_deltaS()
        if self.deltaS is not None:
            if math.isclose(self.deltaS, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"ΔS ≈ {expected} J/(mol·K) matches solution's ΔS = {self.deltaS}"
            else:
                return False, f"ΔS mismatch: expected {expected}, got {self.deltaS}"
        else:
            return True, f"ΔS = {expected} J/(mol·K) (computed from givens)."

    def test_invariant_deltaG(self) -> Tuple[bool, str]:
        """Check ΔG = ΔH - T*ΔS."""
        if None in (self.T,):
            return False, "SKIP: missing temperature."
        # Need deltaH and deltaS; compute from givens if not provided
        deltaH = self.deltaH if self.deltaH is not None else self._compute_deltaH()
        deltaS = self.deltaS if self.deltaS is not None else self._compute_deltaS()
        if None in (deltaH, deltaS):
            return False, "SKIP: missing enthalpy/entropy to compute ΔG."
        expected = deltaH - self.T * deltaS
        if self.deltaG is not None:
            if math.isclose(self.deltaG, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"ΔG ≈ {expected} J/mol matches solution's ΔG = {self.deltaG}"
            else:
                return False, f"ΔG mismatch: expected {expected}, got {self.deltaG}"
        else:
            return True, f"ΔG = {expected} J/mol (computed)."

    def test_invariant_EMF(self) -> Tuple[bool, str]:
        """Check E = -ΔG/(n*F)."""
        if None in (self.n, self.F, self.E):
            return False, "SKIP: missing n, F, or E."
        # Compute ΔG if possible
        if self.deltaG is not None:
            dG = self.deltaG
        else:
            dH = self.deltaH if self.deltaH is not None else self._compute_deltaH()
            dS = self.deltaS if self.deltaS is not None else self._compute_deltaS()
            if None in (dH, dS):
                return False, "SKIP: cannot compute ΔG."
            dG = dH - self.T * dS
        expected_E = -dG / (self.n * self.F)
        if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E ≈ {expected_E} V matches expectation from ΔG/(nF)."
        else:
            return False, f"E mismatch: expected {expected_E} V, got {self.E} V."

    # ------------------------------------------------------------------
    # Modeling checks (signs, plausibility, step consistency)
    # ------------------------------------------------------------------
    def test_modeling_deltaH_negative(self) -> Tuple[bool, str]:
        """The reaction is exothermic -> ΔH should be negative."""
        if self.deltaH is None:
            return False, "SKIP: ΔH not provided."
        if self.deltaH < 0:
            return True, f"ΔH = {self.deltaH} J/mol is negative (exothermic)."
        else:
            return False, f"ΔH = {self.deltaH} J/mol is not negative; expected exothermic reaction."

    def test_modeling_deltaS_negative(self) -> Tuple[bool, str]:
        """Gas-to-liquid reduces entropy -> ΔS negative."""
        if self.deltaS is None:
            return False, "SKIP: ΔS not provided."
        if self.deltaS < 0:
            return True, f"ΔS = {self.deltaS} J/(mol·K) is negative (entropy decreases)."
        else:
            return False, f"ΔS = {self.deltaS} J/(mol·K) is not negative; expected decrease."

    def test_modeling_deltaG_negative(self) -> Tuple[bool, str]:
        """Spontaneous reaction at constant T,P -> ΔG negative."""
        if self.deltaG is None:
            return False, "SKIP: ΔG not provided."
        if self.deltaG < 0:
            return True, f"ΔG = {self.deltaG} J/mol is negative (spontaneous)."
        else:
            return False, f"ΔG = {self.deltaG} J/mol is not negative; expected spontaneous."

    # ------------------------------------------------------------------
    # Monotonicity check (temperature dependence)
    # ------------------------------------------------------------------
    def test_monotonicity_temperature_decreases_emf(self) -> Tuple[bool, str]:
        """
        For this reaction, ΔS < 0, so increasing T makes ΔG less negative,
        hence E = -ΔG/(nF) decreases. We verify that a small relative
        increase in T leads to a negative change in E.
        """
        if None in (self.T, self.F, self.n, self.E):
            return False, "SKIP: missing T, F, n, or E."
        # Compute ΔH and ΔS from givens if needed
        deltaH = self.deltaH if self.deltaH is not None else self._compute_deltaH()
        deltaS = self.deltaS if self.deltaS is not None else self._compute_deltaS()
        if None in (deltaH, deltaS):
            return False, "SKIP: cannot compute ΔH/ΔS for monotonicity."
        # Baseline E from solution
        E_base = self.E
        # Increase T by 5%
        T_pert = self.T * 1.05
        deltaG_pert = deltaH - T_pert * deltaS
        E_pert = -deltaG_pert / (self.n * self.F)
        # Expected change: E_pert < E_base
        if E_pert < E_base:
            return True, f"Increasing T from {self.T} to {T_pert} decreases E from {E_base} to {E_pert} (correct trend)."
        else:
            return False, f"Expected E to decrease with T, but it increased or stayed same: baseline {E_base}, perturbed {E_pert}."

    # ------------------------------------------------------------------
    # Answer sanity (weak check only)
    # ------------------------------------------------------------------
    def test_answer_sanity_E_range(self) -> Tuple[bool, str]:
        """EMF of hydrogen fuel cell is usually around 1.2 V, so check plausible range 0.5–2.5 V."""
        if self.E is None:
            return False, "SKIP: E not available."
        if 0.5 <= self.E <= 2.5:
            return True, f"E = {self.E} V is within plausible range [0.5, 2.5]."
        else:
            return False, f"E = {self.E} V is outside plausible range [0.5, 2.5]."

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, collect logs."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs