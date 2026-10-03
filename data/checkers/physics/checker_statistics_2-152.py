import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class PhysicsChecker:
    """
    Auditor for qualitative explanations of electron-electron mean free path.
    Checks presence/key format of required result field and validates reasoning steps.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract required result key
        self.dependence: Optional[str] = getattr(
            solution_instance, 'results', {}
        ).get('dependence', None)

        # Safely extract optional explanation steps (could be string, list, or None)
        self.explanation_steps: Optional[Union[str, List[str]]] = getattr(
            solution_instance, 'explanation_steps', None
        )

    # ----------------------------------------------------------------------
    # Interface checks: required keys, types, non-empty
    # ----------------------------------------------------------------------
    def test_interface_dependence_key(self) -> Tuple[bool, str]:
        """Check that 'dependence' key exists and is a non-empty string."""
        if self.dependence is None:
            return (False, "Missing 'dependence' in results")
        if not isinstance(self.dependence, str) or len(self.dependence.strip()) == 0:
            return (False, f"'dependence' must be a non-empty string, got {type(self.dependence)}: {self.dependence!r}")
        return (True, f"dependence key present and string: '{self.dependence}'")

    # ----------------------------------------------------------------------
    # Invariant / physics-sanity checks (not strict numerical, but qualitative)
    # ----------------------------------------------------------------------
    def test_invariant_dependence_is_inverse_temperature(self) -> Tuple[bool, str]:
        """
        The mean free path must decrease with increasing temperature.
        Check that the dependence string contains an inverse power of T (1/T or T^{-1}).
        This is a weak sanity; it does not require math.isclose.
        """
        if self.dependence is None:
            return (False, "SKIP: dependence not available")
        # Accept common patterns: '1/T', 'T^{-1}', 'T^{-1}', '1/T^something'
        pattern = r'(\b1\s*[/\\]\s*T\b)|(\bT\s*\{\s*-\s*1\s*\})|(\bT\s*\^\{?\s*-\s*1\s*\}?)'
        if not re.search(pattern, self.dependence, re.IGNORECASE):
            return (False, f"dependence '{self.dependence}' does not show inverse proportionality to T")
        return (True, f"dependence '{self.dependence}' includes inverse temperature relation")

    def test_invariant_positive_mean_free_path(self) -> Tuple[bool, str]:
        """Qualitative check: the dependence should not imply negative λ (e.g., ∝ -T)."""
        if self.dependence is None:
            return (False, "SKIP: dependence not available")
        # Simply check that 'negative' or '-'? Only flag if explicit minus sign precedes T.
        if re.search(r'(?<![^=])\s*-\s*T', self.dependence):
            return (False, f"dependence '{self.dependence}' implies negative λ (for T>0)")
        return (True, "No explicit negative sign found, λ likely positive")

    # ----------------------------------------------------------------------
    # Modeling / reasoning checks (uses explanation_steps if available)
    # ----------------------------------------------------------------------
    def test_modeling_reasoning_pauli_blocking(self) -> Tuple[bool, str]:
        """Check that explanation mentions Pauli exclusion principle or blocking."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = self._explanation_text()
        keywords = ['pauli', 'exclusion', 'blocking', 'pauli principle']
        if not any(kw in text.lower() for kw in keywords):
            return (False, "Explanation lacks mention of Pauli exclusion/blocking")
        return (True, "Explanation includes Pauli principle reasoning")

    def test_modeling_reasoning_effective_density(self) -> Tuple[bool, str]:
        """Check that explanation mentions effective number of electrons near Fermi surface."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = self._explanation_text()
        keywords = ['effective', 'excited', 'near fermi', 'thermal', 'k_b t']
        if not any(kw in text.lower() for kw in keywords):
            return (False, "Explanation lacks reference to effective/scattering electron density")
        return (True, "Explanation includes effective density reasoning")

    def test_modeling_reasoning_screening(self) -> Tuple[bool, str]:
        """Check that explanation mentions screening of Coulomb interaction."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = self._explanation_text()
        keywords = ['screen', 'screening', 'thomas-fermi', 'coulomb']
        if not any(kw in text.lower() for kw in keywords):
            return (False, "Explanation lacks mention of screening")
        return (True, "Explanation includes screening argument")

    # ----------------------------------------------------------------------
    # Very weak answer sanity (only format, no numeric re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_dependence_format(self) -> Tuple[bool, str]:
        """
        Final answer sanity: check that the dependence string is a plausible statement.
        Should not contain obvious errors like 'proportional to T' instead of '1/T'.
        This test does NOT compute any derived value.
        """
        if self.dependence is None:
            return (False, "SKIP: dependence not available")
        # Reject if it only says "proportional to T" without inverse
        if re.search(r'\bproportional\b|\bpropto\b|\b∝\b', self.dependence, re.IGNORECASE):
            # Must also show inverse 1/T; if only says "∝ T", flag
            if not re.search(r'1\s*[/\\]\s*T|T\s*\{?\-?1\}?', self.dependence):
                return (False, f"Dependence '{self.dependence}' suggests direct proportion, but must be inverse")
        # Accept if it contains obvious inverse indication
        if not re.search(r'1/T|T\s*\{?\s*-\s*1\s*\}?|T\s*\^\s*\{?\-?1\}?', self.dependence):
            return (False, f"Dependence '{self.dependence}' does not show inverse temperature dependence")
        return (True, f"Answer format acceptable: '{self.dependence}'")

    # ----------------------------------------------------------------------
    # Monotonicity test: explicit perturbation of temperature input
    # ----------------------------------------------------------------------
    def test_monotonicity_temperature_increases(self) -> Tuple[bool, str]:
        """
        Using the dependence relation (e.g., λ ∝ 1/T^p), explicitly perturb the temperature
        and check that the mean free path decreases.
        We take a baseline at room temperature (T₀ = 300 K) with normalized λ = 1,
        then perturb T by +10% (to 330 K) and compute the perturbed λ.
        This demonstrates monotonicity by re-deriving the trend from the dependence string.
        """
        if self.dependence is None:
            return (False, "SKIP: dependence not available")

        # Extract power p from the dependence string (e.g., 1/T^2 -> p=2, 1/T -> p=1)
        p = self._inverse_power()
        if p is None:
            return (False, f"Cannot extract inverse power from dependence: '{self.dependence}'")

        # Baseline: choose a physical temperature (room temperature ~300 K)
        T_baseline = 300.0
        # Perturb T by +10% (increase by 30 K)
        T_perturbed = T_baseline * 1.10  # 330 K

        # Baseline mean free path (normalized to 1)
        lambda_baseline = 1.0
        # Perturbed mean free path from the dependence: λ ∝ 1/T^p
        lambda_perturbed = lambda_baseline * (T_baseline / T_perturbed) ** p

        # Check that λ decreases
        if lambda_perturbed < lambda_baseline:
            return (True, f"λ decreases from {lambda_baseline:.4f} (T={T_baseline} K) to {lambda_perturbed:.4f} (T={T_perturbed} K), fully consistent with inverse T dependence (p={p})")
        else:
            return (False, f"λ does not decrease: baseline {lambda_baseline:.4f}, perturbed {lambda_perturbed:.4f} for +10% T increase")

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _explanation_text(self) -> str:
        """Convert explanation_steps to a single lowercase string."""
        if isinstance(self.explanation_steps, str):
            return self.explanation_steps.lower()
        elif isinstance(self.explanation_steps, list):
            return ' '.join(str(step) for step in self.explanation_steps).lower()
        return ""

    def _inverse_power(self) -> Optional[float]:
        """
        Try to extract the power p from patterns like '1/T', '1/T^2', 'T^{-1}', 'T^-2'.
        Returns None if not found.
        """
        text = self.dependence.strip()
        # Pattern: '1/T' -> p=1, '1/T^2' -> p=2, '1/T^{1}' -> p=1
        m = re.search(r'1\s*[/\\]\s*T\s*(?:\^|\^\{\s*)(-?\d+\.?\d*)\s*}?', text)
        if m:
            return float(m.group(1))
        # Pattern: T^{-p}
        m = re.search(r'T\s*\^\s*\{?\s*-\s*(\d+\.?\d*)\s*\}?', text)
        if m:
            return float(m.group(1))
        # Simple '1/T' without exponent
        if re.search(r'(?<![\^\{])1\s*[/\\]\s*T(?![\^\}])', text):
            return 1.0
        return None

    # ----------------------------------------------------------------------
    # Test discovery and execution (must be present for auditor to call)
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method: Callable = getattr(self, name)
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