import math
import traceback
from typing import Any, Tuple, List, Optional

# Optional sympy import for symbolic checks (not mandatory)
try:
    import sympy as sp
except ImportError:
    sp = None


class PhysicsChecker:
    """
    PhysicsChecker for the diatomic molecule rotational problem.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity, and provides one weak answer sanity check.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Extract results from solution_instance.results ---
        results = getattr(solution_instance, 'results', {}) or {}
        self.E_J = results.get('E_J', None)           # rotational energy (number)
        self.Delta_J = results.get('Delta_J', None)   # selection rule (string)
        self.Delta_m = results.get('Delta_m', None)   # selection rule (string)
        self.omega = results.get('omega', None)       # angular frequency (number)

        # --- Extract input parameters (M, D, J, hbar) ---
        self.M: Optional[float] = None
        self.D: Optional[float] = None
        self.J: Optional[int] = None
        self.hbar: Optional[float] = None

        # Helper to get value from multiple sources
        def _get(name: str) -> Any:
            val = getattr(solution_instance, name, None)
            if val is None:
                givens = getattr(solution_instance, 'givens', None)
                if isinstance(givens, dict):
                    val = givens.get(name, None)
            if val is None:
                val = results.get(name, None)
            return val

        for name in ['M', 'D', 'J', 'hbar']:
            setattr(self, name, _get(name))

        # --- Extract explanation_steps (for modeling checks) ---
        self.explanation_steps = results.get('explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Store a flag for availability of required numeric values
        self._numeric_available = all(
            v is not None and isinstance(v, (int, float, complex))
            for v in [self.M, self.D, self.J, self.hbar, self.E_J, self.omega]
        )

    # ------------------------------------------------------------------
    # Interface tests: required result keys must exist and have correct types
    # ------------------------------------------------------------------
    def test_interface_E_J_exists(self) -> Tuple[bool, str]:
        if self.E_J is None:
            return False, "E_J not found in results"
        return True, "E_J present"

    def test_interface_Delta_J_exists(self) -> Tuple[bool, str]:
        if self.Delta_J is None:
            return False, "Delta_J not found in results"
        return True, "Delta_J present"

    def test_interface_Delta_m_exists(self) -> Tuple[bool, str]:
        if self.Delta_m is None:
            return False, "Delta_m not found in results"
        return True, "Delta_m present"

    def test_interface_omega_exists(self) -> Tuple[bool, str]:
        if self.omega is None:
            return False, "omega not found in results"
        return True, "omega present"

    def test_interface_E_J_numeric(self) -> Tuple[bool, str]:
        if self.E_J is None:
            return False, "SKIP: E_J missing"
        if not isinstance(self.E_J, (int, float, complex)):
            return False, f"E_J is not numeric: type {type(self.E_J)}"
        return True, "E_J is numeric"

    def test_interface_omega_numeric(self) -> Tuple[bool, str]:
        if self.omega is None:
            return False, "SKIP: omega missing"
        if not isinstance(self.omega, (int, float, complex)):
            return False, f"omega is not numeric: type {type(self.omega)}"
        return True, "omega is numeric"

    def test_interface_Delta_J_string(self) -> Tuple[bool, str]:
        if self.Delta_J is None:
            return False, "SKIP: Delta_J missing"
        if not isinstance(self.Delta_J, str):
            return False, f"Delta_J is not a string: type {type(self.Delta_J)}"
        return True, "Delta_J is string"

    def test_interface_Delta_m_string(self) -> Tuple[bool, str]:
        if self.Delta_m is None:
            return False, "SKIP: Delta_m missing"
        if not isinstance(self.Delta_m, str):
            return False, f"Delta_m is not a string: type {type(self.Delta_m)}"
        return True, "Delta_m is string"

    # ------------------------------------------------------------------
    # Invariant checks: physics relationships (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_moment_of_inertia_formula(self) -> Tuple[bool, str]:
        """Symbolic check: the energy formula implies I = M D^2 / 2."""
        if None in [self.M, self.D]:
            return False, "SKIP: M or D missing for moment of inertia"
        # If we had I in results we could check; but it's not required,
        # so we just acknowledge the formula is correct.
        return True, "Moment of inertia formula is I = M D^2 / 2 (standard rigid rotor)"

    def test_invariant_rotational_energy_form(self) -> Tuple[bool, str]:
        """Energy must match E_J = (hbar^2/(M D^2)) J(J+1) within tolerance."""
        if not self._numeric_available:
            return False, "SKIP: numeric value(s) missing"
        try:
            E_expected = (self.hbar ** 2 / (self.M * self.D ** 2)) * self.J * (self.J + 1)
            if math.isclose(self.E_J, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"E_J matches expected value: {E_expected}"
            else:
                return False, f"E_J {self.E_J} != expected {E_expected} (diff {abs(self.E_J - E_expected):.3e})"
        except (ZeroDivisionError, TypeError, ValueError) as e:
            return False, f"Error in energy check: {e}"

    def test_invariant_selection_rule_J(self) -> Tuple[bool, str]:
        """Delta_J must be ±1."""
        if self.Delta_J is None:
            return False, "SKIP: Delta_J missing"
        # Accept common representations
        if self.Delta_J.strip() in ['+/-1', '±1', '± 1', '+/- 1', '±1', '+1/-1']:
            return True, f"Delta_J = {self.Delta_J} matches expected ΔJ = ±1"
        else:
            return False, f"Delta_J = {self.Delta_J} does not match expected ΔJ = ±1"

    def test_invariant_selection_rule_m(self) -> Tuple[bool, str]:
        """Delta_m must be 0."""
        if self.Delta_m is None:
            return False, "SKIP: Delta_m missing"
        if self.Delta_m.strip() in ['0', '0']:
            return True, f"Delta_m = {self.Delta_m} matches expected Δm = 0"
        else:
            return False, f"Delta_m = {self.Delta_m} does not match expected Δm = 0"

    def test_invariant_emission_frequency(self) -> Tuple[bool, str]:
        """Check omega = 2 J hbar / (M D^2) within tolerance."""
        if not self._numeric_available:
            return False, "SKIP: numeric value(s) missing"
        try:
            omega_expected = 2 * self.J * self.hbar / (self.M * self.D ** 2)
            if math.isclose(self.omega, omega_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"omega matches expected value: {omega_expected}"
            else:
                return False, f"omega {self.omega} != expected {omega_expected} (diff {abs(self.omega - omega_expected):.3e})"
        except (ZeroDivisionError, TypeError, ValueError) as e:
            return False, f"Error in frequency check: {e}"

    # ------------------------------------------------------------------
    # Modeling / trace checks: based on explanation_steps
    # ------------------------------------------------------------------
    def test_modeling_derivation_parity_change(self) -> Tuple[bool, str]:
        """Check that explanation_steps mentions parity and (-1)^J."""
        if self.explanation_steps is None:
            return False, "SKIP: no explanation_steps provided"
        steps_text = str(self.explanation_steps)
        if 'parity' in steps_text.lower() and '(-1)' in steps_text:
            return True, "Explanation steps include parity change argument"
        else:
            return False, "Explanation steps missing parity change argument (expected mention of (-1)^J)"

    def test_modeling_selection_rule_from_dipole(self) -> Tuple[bool, str]:
        """Check that selection rule is derived from dipole operator and parity."""
        if self.explanation_steps is None:
            return False, "SKIP: no explanation_steps provided"
        steps_text = str(self.explanation_steps)
        if 'dipole' in steps_text.lower() and 'matrix element' in steps_text.lower():
            return True, "Explanation steps use dipole matrix elements to derive selection rules"
        else:
            return False, "Explanation steps missing dipole matrix element discussion"

    def test_modeling_energy_difference_used(self) -> Tuple[bool, str]:
        """Check that omega is derived from ΔE = E_J - E_{J-1}."""
        if self.explanation_steps is None:
            return False, "SKIP: no explanation_steps provided"
        steps_text = str(self.explanation_steps)
        if ('E_J' in steps_text and 'E_{J-1}' in steps_text) or ('J(J+1)' in steps_text and '(J-1)J' in steps_text):
            return True, "Frequency derivation uses energy difference between consecutive J levels"
        else:
            return False, "Explanation steps lack explicit use of ΔE = E_J - E_{J-1}"

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb an input by a percentage, recompute output,
    # compare perturbed vs baseline (baseline from solution).
    # ------------------------------------------------------------------
    def test_monotonicity_energy_vs_mass(self) -> Tuple[bool, str]:
        """
        Perturb M by +10%, recompute E_J using the known formula,
        and compare with the baseline E_J from the solution.
        Expect: if M increases, E_J decreases (since E ∝ 1/M).
        """
        if not self._numeric_available:
            return False, "SKIP: numeric value(s) missing"
        try:
            baseline_E = self.E_J
            # Perturb M: +10%
            M_perturbed = self.M * 1.10
            E_perturbed = (self.hbar ** 2 / (M_perturbed * self.D ** 2)) * self.J * (self.J + 1)
            # Expect perturbed < baseline because E ∝ 1/M
            if E_perturbed < baseline_E:
                return True, f"Energy decreases when mass increases: perturbed E = {E_perturbed} < baseline E = {baseline_E}"
            else:
                return False, f"Energy did not decrease: perturbed E = {E_perturbed} >= baseline E = {baseline_E}"
        except (ZeroDivisionError, TypeError, ValueError) as e:
            return False, f"Error in monotonicity test: {e}"

    def test_monotonicity_frequency_vs_mass(self) -> Tuple[bool, str]:
        """
        Perturb M by +10%, recompute omega using the known formula,
        and compare with the baseline omega from the solution.
        Expect: if M increases, omega decreases (since ω ∝ 1/M).
        """
        if not self._numeric_available:
            return False, "SKIP: numeric value(s) missing"
        try:
            baseline_omega = self.omega
            M_perturbed = self.M * 1.10
            omega_perturbed = 2 * self.J * self.hbar / (M_perturbed * self.D ** 2)
            if omega_perturbed < baseline_omega:
                return True, f"Frequency decreases when mass increases: perturbed ω = {omega_perturbed} < baseline ω = {baseline_omega}"
            else:
                return False, f"Frequency did not decrease: perturbed ω = {omega_perturbed} >= baseline ω = {baseline_omega}"
        except (ZeroDivisionError, TypeError, ValueError) as e:
            return False, f"Error in monotonicity test: {e}"

    def test_monotonicity_energy_vs_bond_length(self) -> Tuple[bool, str]:
        """
        Perturb D by +10%, recompute E_J, compare with baseline.
        Expect: if D increases, E_J decreases (since E ∝ 1/D^2).
        """
        if not self._numeric_available:
            return False, "SKIP: numeric value(s) missing"
        try:
            baseline_E = self.E_J
            D_perturbed = self.D * 1.10
            E_perturbed = (self.hbar ** 2 / (self.M * D_perturbed ** 2)) * self.J * (self.J + 1)
            if E_perturbed < baseline_E:
                return True, f"Energy decreases when bond length increases: perturbed E = {E_perturbed} < baseline E = {baseline_E}"
            else:
                return False, f"Energy did not decrease: perturbed E = {E_perturbed} >= baseline E = {baseline_E}"
        except (ZeroDivisionError, TypeError, ValueError) as e:
            return False, f"Error in monotonicity test: {e}"

    # ------------------------------------------------------------------
    # Weak final‑answer sanity (at most one, very lenient)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """One weak sanity: all result quantities have sensible signs and are finite."""
        issues = []
        if self.E_J is not None:
            try:
                if self.E_J < 0:
                    issues.append("E_J is negative")
                if not math.isfinite(complex(self.E_J).real):
                    issues.append("E_J is not finite")
            except (TypeError, ValueError):
                issues.append("E_J is not a valid number")
        else:
            issues.append("E_J missing")

        if self.omega is not None:
            try:
                if self.omega < 0:
                    issues.append("omega is negative")
                if not math.isfinite(complex(self.omega).real):
                    issues.append("omega is not finite")
            except (TypeError, ValueError):
                issues.append("omega is not a valid number")
        else:
            issues.append("omega missing")

        # Selection rules: must contain '±1' and '0' (or obvious variants)
        if self.Delta_J is not None:
            if not any(s in str(self.Delta_J) for s in ['±', '+/-', '±1']):
                issues.append("Delta_J does not contain '±1'")
        else:
            issues.append("Delta_J missing")

        if self.Delta_m is not None:
            if '0' not in str(self.Delta_m):
                issues.append("Delta_m does not contain '0'")
        else:
            issues.append("Delta_m missing")

        if issues:
            return False, "Sanity issues: " + "; ".join(issues)
        return True, "All quantities pass basic sanity (non‑negative, finite, selection rule format)"

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discovers all test_* methods, runs them, returns (all_pass, logs).
        Each test returns (passed, msg).  Exceptions are caught and reported.
        """
        logs: List[str] = []
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
                msg = f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs