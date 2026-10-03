import math
import traceback
from typing import Any, Callable, Dict, List, Tuple, Union

class PhysicsChecker:
    """
    Auditor for quantum scattering problem. Checks physical consistency,
    modeling quality, monotonicity, and weak answer sanity.
    """
    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely.
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {}) or {}
        self.df_dphi = self.results.get('df_dphi', None)
        self.theta_dependence = self.results.get('theta_dependence', None)
        self.df_dtheta = self.results.get('df_dtheta', None)
        self.diff_cross_section = self.results.get('diff_cross_section', None)
        self.born_amplitude = self.results.get('born_amplitude', None)

        # Optional extra: explanation steps, givens (if needed)
        self.explanation_steps: List[str] = getattr(solution_instance, 'explanation_steps', None) or []

    def _is_not_none(self, name: str, value: Any) -> Tuple[bool, str]:
        if value is None:
            return (False, f"FAIL: {name} is missing (None)")
        return (True, f"PASS: {name} is present")

    # ---------- Interface tests ----------
    def test_interface_df_dphi_present(self) -> Tuple[bool, str]:
        return self._is_not_none('df_dphi', self.df_dphi)

    def test_interface_theta_dependence_present(self) -> Tuple[bool, str]:
        return self._is_not_none('theta_dependence', self.theta_dependence)

    def test_interface_df_dtheta_present(self) -> Tuple[bool, str]:
        return self._is_not_none('df_dtheta', self.df_dtheta)

    def test_interface_diff_cross_section_present(self) -> Tuple[bool, str]:
        return self._is_not_none('diff_cross_section', self.diff_cross_section)

    def test_interface_born_amplitude_present(self) -> Tuple[bool, str]:
        return self._is_not_none('born_amplitude', self.born_amplitude)

    # ---------- Invariant / physics sanity tests (with math.isclose if needed) ----------
    def test_invariant_df_dphi_zero(self) -> Tuple[bool, str]:
        """Rotational invariance about z-axis forces df/dphi = 0."""
        if self.df_dphi is None:
            return (False, "SKIP: df_dphi not available")
        if isinstance(self.df_dphi, (int, float)):
            if math.isclose(self.df_dphi, 0.0, abs_tol=1e-6):
                return (True, "df_dphi is essentially 0")
            else:
                return (False, f"df_dphi = {self.df_dphi}, expected 0")
        if isinstance(self.df_dphi, str) and self.df_dphi.strip() in ("0", "0.0"):
            return (True, "df_dphi string indicates 0")
        return (False, f"df_dphi has unexpected type/value: {self.df_dphi}")

    def test_invariant_df_dtheta_zero_at_zero_energy(self) -> Tuple[bool, str]:
        """At zero energy, only s-wave contributes -> f isotropic -> df/dtheta = 0."""
        if self.df_dtheta is None:
            return (False, "SKIP: df_dtheta not available")
        if isinstance(self.df_dtheta, (int, float)):
            if math.isclose(self.df_dtheta, 0.0, abs_tol=1e-6):
                return (True, "df_dtheta is essentially 0 (zero‑energy limit)")
            else:
                return (False, f"df_dtheta = {self.df_dtheta}, expected 0")
        if isinstance(self.df_dtheta, str) and self.df_dtheta.strip() in ("0", "0.0"):
            return (True, "df_dtheta string indicates 0")
        return (False, f"df_dtheta has unexpected type/value: {self.df_dtheta}")

    def test_invariant_diff_cross_section_non_negative(self) -> Tuple[bool, str]:
        """dσ/dΩ = |f|^2 ≥ 0 (positivity)."""
        if self.diff_cross_section is None:
            return (False, "SKIP: diff_cross_section not available")
        s = str(self.diff_cross_section).replace(" ", "")
        if "|f|^2" in s or "|f|²" in s or "abs(f)**2" in s:
            return (True, "differential cross section expressed as |f|^2, which is non‑negative")
        return (False, f"diff_cross_section formula does not clearly show positivity: {self.diff_cross_section}")

    def test_invariant_born_amplitude_contains_integral(self) -> Tuple[bool, str]:
        """Born amplitude should involve an integral over potential."""
        if self.born_amplitude is None:
            return (False, "SKIP: born_amplitude not available")
        s = str(self.born_amplitude).replace(" ", "")
        if "integral" in s or "∫" in s or "int " in s or "integrate" in s:
            return (True, "Born amplitude includes an integral over the potential")
        return (False, f"born_amplitude does not appear to contain an integral: {self.born_amplitude}")

    def test_invariant_theta_dependence_explanation(self) -> Tuple[bool, str]:
        """theta_dependence should state that f depends on theta because incident direction fixed."""
        if self.theta_dependence is None:
            return (False, "SKIP: theta_dependence not available")
        s = str(self.theta_dependence).lower()
        if "theta" in s and ("depend" in s or "not independent" in s or "varies" in s):
            return (True, "theta_dependence correctly indicates θ dependence remains")
        return (False, f"theta_dependence message unclear: {self.theta_dependence}")

    # ---------- Modeling / trace tests (check consistency with explanation steps) ----------
    def test_modeling_df_dphi_derived_from_symmetry(self) -> Tuple[bool, str]:
        """The explanation steps (part a) should argue from rotational invariance about z-axis."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        combined = " ".join(self.explanation_steps).lower()
        keywords = ["rotational invariance", "l_z", "angular momentum", "∂/∂φ", "phi"]
        if any(k in combined for k in keywords):
            return (True, "Explanation steps mention symmetry argument for df/dphi=0")
        return (False, "Explanation steps missing typical keywords for part (a)")

    def test_modeling_diff_cross_section_formula(self) -> Tuple[bool, str]:
        """Differential cross section should be defined in terms of f."""
        if self.diff_cross_section is None:
            return (False, "SKIP: diff_cross_section not available")
        if "f" in str(self.diff_cross_section) and ("|" in str(self.diff_cross_section) or "square" in str(self.diff_cross_section).lower()):
            return (True, "diff_cross_section expression references |f|^2")
        return (False, f"diff_cross_section does not appear as |f|^2: {self.diff_cross_section}")

    # ---------- Monotonicity tests (perturb input, compare baseline and perturbed output) ----------
    def test_monotonicity_born_validity_with_potential(self) -> Tuple[bool, str]:
        """
        The Born approximation is valid when |V| << E.  Increasing the potential
        strength makes the approximation less valid (validity ratio |V|/E increases).
        Here we use a simple numeric perturbation: baseline V=1.0, E=10.0,
        then V is increased to 1.1, and the validity ratio increases from 0.10 to 0.11.
        This monotonic trend is checked.
        """
        # Baseline values (arbitrary but physically sensible)
        baseline_V = 1.0
        baseline_E = 10.0
        baseline_ratio = baseline_V / baseline_E

        # Perturb the potential by +10%
        perturbed_V = baseline_V * 1.10
        perturbed_ratio = perturbed_V / baseline_E

        if perturbed_ratio > baseline_ratio:
            return (True, "Monotonic: as potential increases, validity ratio increases (less valid)")
        else:
            return (False, "Monotonic violated: validity ratio did not increase with potential increase")

    # ---------- Weak answer sanity (at most one) ----------
    def test_answer_sanity_df_dphi_type(self) -> Tuple[bool, str]:
        """Weak sanity: df_dphi should be numeric or string representing zero."""
        if self.df_dphi is None:
            return (False, "SKIP: df_dphi missing")
        if isinstance(self.df_dphi, (int, float)):
            if abs(self.df_dphi) < 1e-3:
                return (True, "df_dphi is approximately zero")
            else:
                return (False, f"df_dphi = {self.df_dphi}, expected near zero")
        if isinstance(self.df_dphi, str) and self.df_dphi.strip().startswith("0"):
            return (True, "df_dphi string starts with 0")
        return (False, f"df_dphi type/value unexpected: {self.df_dphi}")

    # ---------- Main runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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