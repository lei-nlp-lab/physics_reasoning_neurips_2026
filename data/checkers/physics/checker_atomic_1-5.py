import math
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits solution for the electron density behavior problem.
    The problem: "As one goes away from the center of an atom, the electron density"
    Options: a) decreases like a Gaussian, b) decreases exponentially, c) oscillates with slowly decreasing amplitude.
    Target answer: c.

    Checks:
      - Interface: presence and validity of key 'behavior'.
      - Theorem‑based invariants: decay must be exponential (not Gaussian) and include oscillations.
      - Modeling: check that the behavior matches the pseudocode emit contract.
      - Monotonicity: verify asymptotic density decreases with increasing nuclear charge Z and increases with increasing principal quantum number n.
      - Weak answer sanity: behavior is a valid non‑empty option string.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safe extraction of required data
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # [emit] contract key
        self.behavior: Optional[str] = self.results.get('behavior', None)

        # Optional additional parameters (for monotonicity tests)
        self.givens: Optional[Dict[str, Any]] = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}

        # Physics constants (may be extracted from givens or explanation)
        self.Z: Optional[float] = self.givens.get('Z', None)          # nuclear charge
        self.n: Optional[float] = self.givens.get('n', None)          # principal quantum number
        self.a0: Optional[float] = self.givens.get('a0', None)        # Bohr radius

        # Explanation steps and pseudocode for trace checks
        self.explanation_steps: Optional[str] = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)

    # -------------------------------------------------------------------------
    # Helper: asymptotic density (relative, non‑normalized) at large r
    # -------------------------------------------------------------------------
    def _asymptotic_density(self, r: float, Z: float, n: float, a0: float) -> float:
        """Return ρ(r) ∝ exp(-2Zr/(na0)). Relative scale, used for monotonicity checks."""
        exponent = -2.0 * Z * r / (n * a0)
        return math.exp(exponent)

    # -------------------------------------------------------------------------
    # Interface tests
    # -------------------------------------------------------------------------
    def test_interface_behavior_exists(self) -> Tuple[bool, str]:
        if self.behavior is None:
            return False, "behavior key missing or None in results"
        if not isinstance(self.behavior, str):
            return False, f"behavior is not a string (type: {type(self.behavior).__name__})"
        return True, "behavior exists and is a string"

    def test_interface_behavior_valid_option(self) -> Tuple[bool, str]:
        valid_options = [
            "decreases like a Gaussian",
            "decreases exponentially",
            "oscillates with slowly decreasing amplitude"
        ]
        if self.behavior is None:
            return False, "SKIP: behavior not available"
        if self.behavior not in valid_options:
            return False, f"behavior '{self.behavior}' is not one of the valid options: {valid_options}"
        return True, f"behavior '{self.behavior}' is a valid option"

    # -------------------------------------------------------------------------
    # Theorem‑based invariant tests
    # -------------------------------------------------------------------------
    def test_invariant_not_gaussian(self) -> Tuple[bool, str]:
        if self.behavior is None:
            return False, "SKIP: behavior not available"
        if self.behavior == "decreases like a Gaussian":
            return False, "Gaussian decay contradicts Coulomb potential bound states; exponential decay is required"
        return True, "behavior does not claim Gaussian decay"

    def test_invariant_oscillatory(self) -> Tuple[bool, str]:
        if self.behavior is None:
            return False, "SKIP: behavior not available"
        if "oscillate" not in self.behavior.lower() and "oscillation" not in self.behavior.lower():
            return False, f"behavior '{self.behavior}' does not indicate oscillatory character, but atomic bound states oscillate due to radial nodes"
        return True, "behavior correctly indicates oscillatory character"

    # -------------------------------------------------------------------------
    # Modeling / trace test
    # -------------------------------------------------------------------------
    def test_modeling_emit_contract(self) -> Tuple[bool, str]:
        """
        Check that the behavior matches the explicit [emit] contract from the pseudocode.
        The pseudocode states: results['behavior'] = 'oscillates with slowly decreasing amplitude'.
        """
        expected_emit = "oscillates with slowly decreasing amplitude"
        if self.behavior is None:
            return False, "SKIP: behavior not available"
        if self.behavior != expected_emit:
            return False, f"behavior '{self.behavior}' does not match the pseudocode emit contract ('{expected_emit}')"
        return True, "behavior matches the pseudocode emit contract"

    def test_trace_exponential_reference(self) -> Tuple[bool, str]:
        """
        Check that the explanation_steps or pseudocode mention exponential decay or asymptotic behavior.
        This ensures the solver considered the correct physics.
        """
        if self.explanation_steps is None and self.pseudocode is None:
            return False, "SKIP: no explanation_steps or pseudocode available to check"
        text = (self.explanation_steps or "") + (self.pseudocode or "")
        keywords = ["exponential", "asymptotic", "decays"]
        found = [kw for kw in keywords if kw.lower() in text.lower()]
        if not found:
            return False, "No mention of exponential decay or asymptotic behavior in explanation/pseudocode"
        return True, f"Found reference to asymptotic/exponential decay: {found}"

    # -------------------------------------------------------------------------
    # Monotonicity tests
    # -------------------------------------------------------------------------
    def test_monotonicity_charge_increases_decay_rate(self) -> Tuple[bool, str]:
        """
        Perturb nuclear charge Z by +1% and check that the asymptotic density decreases.
        Use a large fixed radius r = 10*a0 (within asymptotic regime).
        """
        if self.Z is None or self.n is None or self.a0 is None:
            return False, "SKIP: missing Z, n, or a0 from givens"
        # Baseline density at large r
        r_large = 10.0 * self.a0
        rho_base = self._asymptotic_density(r_large, self.Z, self.n, self.a0)
        # Perturb Z by +1%
        Z_perturbed = self.Z * 1.01
        rho_pert = self._asymptotic_density(r_large, Z_perturbed, self.n, self.a0)
        if rho_pert < rho_base:
            return True, f"Z increase reduces density as expected: {rho_pert:.6g} < {rho_base:.6g}"
        else:
            return False, f"Z increase did not reduce density: baseline {rho_base:.6g}, perturbed {rho_pert:.6g}"

    def test_monotonicity_principal_number_decreases_decay_rate(self) -> Tuple[bool, str]:
        """
        Perturb principal quantum number n by +1% and check that the asymptotic density increases.
        (Larger n reduces the effective exponent → slower decay → larger density at fixed r.)
        """
        if self.Z is None or self.n is None or self.a0 is None:
            return False, "SKIP: missing Z, n, or a0 from givens"
        r_large = 10.0 * self.a0
        rho_base = self._asymptotic_density(r_large, self.Z, self.n, self.a0)
        # Perturb n by +1%
        n_perturbed = self.n * 1.01
        rho_pert = self._asymptotic_density(r_large, self.Z, n_perturbed, self.a0)
        if rho_pert > rho_base:
            return True, f"n increase increases density as expected: {rho_pert:.6g} > {rho_base:.6g}"
        else:
            return False, f"n increase did not increase density: baseline {rho_base:.6g}, perturbed {rho_pert:.6g}"

    # -------------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # -------------------------------------------------------------------------
    def test_answer_sanity_behavior_is_valid(self) -> Tuple[bool, str]:
        if self.behavior is None:
            return False, "SKIP: behavior not available"
        valid = [
            "decreases like a Gaussian",
            "decreases exponentially",
            "oscillates with slowly decreasing amplitude"
        ]
        if self.behavior not in valid:
            return False, f"behavior '{self.behavior}' is not among the expected options"
        if not self.behavior.strip():
            return False, "behavior is empty or whitespace"
        return True, "behavior is a non‑empty valid option string"

    # -------------------------------------------------------------------------
    # Run all tests
    # -------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)