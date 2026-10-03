import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Robust physics checker for hydrogen 2p lifetime and selection rules problem.
    Performs interface checks, invariant/theorem checks, modeling checks,
    monotonicity checks, and one weak answer-sanity check.
    """
    def __init__(self, solution_instance: Any):
        """
        Extract all needed data safely from solution_instance.
        Use None defaults; never numeric fallback.
        """
        # Required result keys
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Givens (expected to be direct attributes or in a givens dict)
        self.e = getattr(solution_instance, 'e', None)
        self.c = getattr(solution_instance, 'c', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.a = getattr(solution_instance, 'a', None)          # Bohr radius
        self.eV = getattr(solution_instance, 'eV', None)

        # Fallback: try givens dict if present
        givens = getattr(solution_instance, 'givens', None)
        if isinstance(givens, dict):
            if self.e is None:
                self.e = givens.get('e', None)
            if self.c is None:
                self.c = givens.get('c', None)
            if self.hbar is None:
                self.hbar = givens.get('hbar', None)
            if self.a is None:
                self.a = givens.get('a', None)
            if self.eV is None:
                self.eV = givens.get('eV', None)

        # Emitted results
        self.selection_rule = self.results.get('selection_rule', None)
        self.interpretation = self.results.get('interpretation', None)
        self.tau = self.results.get('tau', None)
        self.width_eV = self.results.get('width_eV', None)

        # Additional extracted variables that may be present
        self.r = self.results.get('r', None)
        self.omega = self.results.get('omega', None)
        self.a_acc = self.results.get('a_acc', None)
        self.P = self.results.get('P', None)
        self.Delta_E = self.results.get('Delta_E', None)

        # Explanation steps (for modeling hints, not for copying numbers)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist and have correct types."""
        if self.selection_rule is None:
            return (False, "Missing 'selection_rule' in results")
        if not isinstance(self.selection_rule, str):
            return (False, "'selection_rule' is not a string")
        if self.interpretation is None:
            return (False, "Missing 'interpretation' in results")
        if not isinstance(self.interpretation, str):
            return (False, "'interpretation' is not a string")
        if self.tau is None:
            return (False, "Missing 'tau' in results")
        if not isinstance(self.tau, (int, float)):
            return (False, "'tau' is not numeric")
        if self.width_eV is None:
            return (False, "Missing 'width_eV' in results")
        if not isinstance(self.width_eV, (int, float)):
            return (False, "'width_eV' is not numeric")
        return (True, "All required keys present and valid.")

    def test_interface_selection_rule_content(self) -> Tuple[bool, str]:
        """Check selection rule string contains essential elements."""
        if self.selection_rule is None:
            return (False, "SKIP: selection_rule missing")
        rule = self.selection_rule.lower()
        # Should mention Δl = ±1 and Δm = ±1,0 (or similar)
        if 'l' in rule and '±1' in rule and 'm' in rule:
            return (True, "Selection rule mentions Δl=±1 and Δm=±1,0")
        # Accept alternative phrasing like "Delta l = +/- 1"
        if ('delta l' in rule or 'Δl' in rule) and ('+/- 1' in rule or '±1' in rule) and ('m' in rule or 'm_l' in rule):
            return (True, "Selection rule contains required elements")
        return (False, f"Selection rule '{self.selection_rule}' does not contain expected patterns")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_energy_difference(self) -> Tuple[bool, str]:
        """Check ΔE = 3e²/(8a) using provided constants."""
        if any(x is None for x in (self.e, self.a, self.Delta_E)):
            return (False, "SKIP: Missing e, a, or Delta_E")
        expected_DE = 3.0 * self.e**2 / (8.0 * self.a)
        if math.isclose(self.Delta_E, expected_DE, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Delta_E matches expected {expected_DE:.3e}")
        return (False, f"Delta_E {self.Delta_E:.3e} != expected {expected_DE:.3e}")

    def test_invariant_lifetime_formula(self) -> Tuple[bool, str]:
        """Check tau = ΔE / P."""
        if any(x is None for x in (self.Delta_E, self.P, self.tau)):
            return (False, "SKIP: Missing Delta_E, P, or tau")
        expected_tau = self.Delta_E / self.P
        if math.isclose(self.tau, expected_tau, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"tau matches ΔE/P = {expected_tau:.3e} s")
        return (False, f"tau {self.tau:.3e} != ΔE/P {expected_tau:.3e}")

    def test_invariant_width_formula(self) -> Tuple[bool, str]:
        """Check width_eV = hbar / tau, converted to eV."""
        if any(x is None for x in (self.hbar, self.tau, self.eV, self.width_eV)):
            return (False, "SKIP: Missing hbar, tau, eV, or width_eV")
        expected_width_erg = self.hbar / self.tau
        expected_width_eV = expected_width_erg / self.eV
        if math.isclose(self.width_eV, expected_width_eV, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"width_eV matches hbar/(tau*eV) = {expected_width_eV:.3e} eV")
        return (False, f"width_eV {self.width_eV:.3e} != {expected_width_eV:.3e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (derived from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_power_formula(self) -> Tuple[bool, str]:
        """Check P = (2/3) e² / c³ * a_acc²."""
        if any(x is None for x in (self.e, self.c, self.a_acc, self.P)):
            return (False, "SKIP: Missing e, c, a_acc, or P")
        expected_P = (2.0/3.0) * self.e**2 / (self.c**3) * self.a_acc**2
        if math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P matches classical formula: {expected_P:.3e}")
        return (False, f"P {self.P:.3e} != expected {expected_P:.3e}")

    def test_modeling_acceleration(self) -> Tuple[bool, str]:
        """Check a_acc = r ω²."""
        if any(x is None for x in (self.r, self.omega, self.a_acc)):
            return (False, "SKIP: Missing r, omega, or a_acc")
        expected_acc = self.r * self.omega**2
        if math.isclose(self.a_acc, expected_acc, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"a_acc matches r ω² = {expected_acc:.3e}")
        return (False, f"a_acc {self.a_acc:.3e} != r ω² {expected_acc:.3e}")

    def test_modeling_angular_velocity(self) -> Tuple[bool, str]:
        """Check ω = v/r, with v from Bohr quantization (v = e²/(2ħ) for n=2)."""
        if any(x is None for x in (self.e, self.hbar, self.omega, self.r)):
            return (False, "SKIP: Missing e, hbar, omega, or r")
        # For n=2, v = e²/(2ħ)
        v = self.e**2 / (2.0 * self.hbar)
        expected_omega = v / self.r
        if math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ω matches v/r = {expected_omega:.3e}")
        return (False, f"ω {self.omega:.3e} != v/r {expected_omega:.3e}")

    def test_modeling_radius_n2(self) -> Tuple[bool, str]:
        """Check r = 4 a (n²a with n=2)."""
        if any(x is None for x in (self.a, self.r)):
            return (False, "SKIP: Missing a or r")
        expected_r = 4.0 * self.a
        if math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r matches n²a = {expected_r:.3e}")
        return (False, f"r {self.r:.3e} != 4a {expected_r:.3e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input and check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_lifetime_with_bohr_radius(self) -> Tuple[bool, str]:
        """
        Increase Bohr radius a by 10% and verify that the computed lifetime
        increases (since τ = ΔE/P, with ΔE ∝ 1/a and P ∝ 1/a², so τ ∝ a).
        Uses only the given constants; does not rely on stored intermediate values.
        """
        # Need at least e, hbar, c, a, and final tau to compare.
        if any(x is None for x in (self.e, self.hbar, self.c, self.a, self.tau)):
            return (False, "SKIP: Missing essential constants (e, hbar, c, a, tau)")

        # Constants for n=2
        n = 2.0
        # Baseline calculation using given a
        r = n**2 * self.a                # Bohr radius for n=2
        # Orbital velocity from Bohr: v = e^2/(n hbar)
        v = self.e**2 / (n * self.hbar)
        # Angular velocity
        omega = v / r
        # Acceleration
        a_acc = r * omega**2
        # Radiated power
        P = (2.0/3.0) * self.e**2 / (self.c**3) * a_acc**2
        # Energy difference between n=2 and n=1
        Delta_E = 3.0 * self.e**2 / (8.0 * self.a)   # using the given Bohr radius a
        # Baseline lifetime
        tau_baseline = Delta_E / P

        # Perturb a by +10%
        a_pert = self.a * 1.10
        r_pert = n**2 * a_pert
        # Note: v is independent of a (only depends on n and constants)
        omega_pert = v / r_pert
        a_acc_pert = r_pert * omega_pert**2
        P_pert = (2.0/3.0) * self.e**2 / (self.c**3) * a_acc_pert**2
        Delta_E_pert = 3.0 * self.e**2 / (8.0 * a_pert)
        tau_pert = Delta_E_pert / P_pert

        # Check that tau_pert > tau_baseline (since a increased)
        if tau_pert > tau_baseline:
            return (True, f"Perturbed lifetime {tau_pert:.3e} > baseline {tau_baseline:.3e} (expected increase)")
        else:
            return (False, f"Perturbed lifetime {tau_pert:.3e} <= baseline {tau_baseline:.3e} (expected increase)")

    # ------------------------------------------------------------------
    # Answer sanity (weak, no isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausible_ranges(self) -> Tuple[bool, str]:
        """Check that tau and width_eV are positive and in plausible physical ranges."""
        if self.tau is None or self.width_eV is None:
            return (False, "SKIP: tau or width_eV missing")
        # Check positivity
        if self.tau <= 0:
            return (False, f"tau is not positive: {self.tau}")
        if self.width_eV <= 0:
            return (False, f"width_eV is not positive: {self.width_eV}")
        # Plausible ranges: hydrogen 2p lifetime ~1e-9 to 1e-7 s, width ~1e-9 to 1e-7 eV
        if not (1e-10 <= self.tau <= 1e-6):
            return (False, f"tau {self.tau:.3e} outside typical range [1e-10, 1e-6] s")
        if not (1e-10 <= self.width_eV <= 1e-5):
            return (False, f"width_eV {self.width_eV:.3e} outside typical range [1e-10, 1e-5] eV")
        return (True, f"tau = {self.tau:.3e} s, width_eV = {self.width_eV:.3e} eV are plausible")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods, execute them in try/except,
        and return (overall_pass, list of log messages).
        """
        logs = []
        all_pass = True
        # Get all callable methods starting with 'test_'
        for attr_name in dir(self):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)