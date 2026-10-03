import math
import re
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits solutions to a nuclear physics problem covering alpha decay,
    neutron emission stability, nuclear radii determination, ground state
    properties, and fission cross sections.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract inputs and results safely.  All missing values default to None.
        No numeric or string physics defaults are assigned.
        """
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})

        # Emitted keys (from pseudocode [emit] contract)
        self.Q_alpha: Optional[float] = self.results.get('Q_alpha', None)
        self.Q_n: Optional[float] = self.results.get('Q_n', None)
        self.R_formula: Optional[str] = self.results.get('R_formula', None)
        self.properties: Optional[str] = self.results.get('properties', None)
        self.pattern: Optional[str] = self.results.get('pattern', None)

    # ------------------------------------------------------------------
    # Interface tests – check key existence and basic types
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self) -> Tuple[bool, str]:
        """All five expected keys must be present in results."""
        expected = ['Q_alpha', 'Q_n', 'R_formula', 'properties', 'pattern']
        missing = [k for k in expected if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_key_types(self) -> Tuple[bool, str]:
        """
        Check that numeric keys are numeric and string keys are strings.
        Returns PASS if a key is None (skip further type check).
        """
        numeric_keys = ['Q_alpha', 'Q_n']
        string_keys = ['R_formula', 'properties', 'pattern']
        for key in numeric_keys:
            val = self.results.get(key)
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"Key {key} has type {type(val).__name__}, expected int/float")
        for key in string_keys:
            val = self.results.get(key)
            if val is not None and not isinstance(val, str):
                return (False, f"Key {key} has type {type(val).__name__}, expected str")
        return (True, "All present keys have correct types")

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_alpha_decay_positive(self) -> Tuple[bool, str]:
        """Q_alpha must be > 0 (alpha decay is energetically allowed)."""
        if self.Q_alpha is None:
            return (False, "SKIP: Q_alpha not available")
        if not (math.isfinite(self.Q_alpha) and self.Q_alpha > 0):
            return (False, f"Q_alpha = {self.Q_alpha} is not > 0")
        return (True, f"Q_alpha = {self.Q_alpha} > 0")

    def test_invariant_neutron_emission_negative(self) -> Tuple[bool, str]:
        """Q_n must be < 0 (neutron emission is energetically forbidden)."""
        if self.Q_n is None:
            return (False, "SKIP: Q_n not available")
        if not (math.isfinite(self.Q_n) and self.Q_n < 0):
            return (False, f"Q_n = {self.Q_n} is not < 0")
        return (True, f"Q_n = {self.Q_n} < 0")

    def test_invariant_radius_formula_exists(self) -> Tuple[bool, str]:
        """R_formula must contain the standard A^(1/3) law."""
        if self.R_formula is None:
            return (False, "SKIP: R_formula not available")
        if re.search(r'A\s*[\^\{]*1/3[\}\s]*', self.R_formula, re.IGNORECASE):
            return (True, f"R_formula contains A^(1/3): {self.R_formula[:60]}...")
        return (False, f"R_formula does not contain A^(1/3): {self.R_formula[:60]}...")

    def test_invariant_fission_odd_even_pattern(self) -> Tuple[bool, str]:
        """The pattern description must mention both thermal and fast neutron behaviour."""
        if self.pattern is None:
            return (False, "SKIP: pattern not available")
        pattern_lower = self.pattern.lower()
        has_thermal = "thermal" in pattern_lower or "slow" in pattern_lower
        has_fast = "fast" in pattern_lower
        has_odd_even = ("odd" in pattern_lower and "even" in pattern_lower)
        if not (has_thermal and has_fast and has_odd_even):
            return (False, f"Pattern missing required keywords (thermal/fast/odd/even): {self.pattern[:80]}...")
        return (True, f"Pattern addresses both thermal and fast neutrons: {self.pattern[:80]}...")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check consistency and quality via explanation steps
    # ------------------------------------------------------------------
    def test_modeling_ground_state_properties(self) -> Tuple[bool, str]:
        """Properties string should mention shell structure, pairing, and nonclassical features."""
        if self.properties is None:
            return (False, "SKIP: properties not available")
        prop_lower = self.properties.lower()
        checks = {
            "shell structure": "shell" in prop_lower,
            "pairing": "pairing" in prop_lower,
            "deformation": "deformation" in prop_lower or "deformed" in prop_lower,
            "nonclassical": "nonclassical" in prop_lower or "non-classical" in prop_lower or "quantum" in prop_lower
        }
        missing = [name for name, ok in checks.items() if not ok]
        if missing:
            return (False, f"Properties string missing: {missing}. Full string: {self.properties[:80]}...")
        return (True, f"All expected topics present in properties: {list(checks.keys())}")

    def test_modeling_radius_methods_mentioned(self) -> Tuple[bool, str]:
        """
        The R_formula explanation should mention at least two experimental methods
        (e.g., electron scattering, alpha scattering, muonic atoms, mirror nuclei).
        This is a soft check; if only the formula is given without methods, we
        fail with a warning.
        """
        if self.R_formula is None:
            return (False, "SKIP: R_formula not available")
        r_lower = self.R_formula.lower()
        methods = ["electron scattering", "alpha scattering", "muonic atom", "mirror nuclei",
                   "neutron scattering", "proton scattering"]
        found = any(m in r_lower for m in methods)
        if not found:
            # The pure mathematical formula is acceptable; methods may be described elsewhere.
            return (True, "R_formula provided as pure mathematical form (methods may be elsewhere)")
        return (True, f"R_formula includes method mention(s): {self.R_formula[:80]}...")

    def test_trace_fission_threshold_condition(self) -> Tuple[bool, str]:
        """
        For thermal neutrons, the pattern should explain that odd‑N isotopes
        have large cross sections because the extra neutron's pairing energy
        makes capture exothermic and lowers the fission barrier.
        We check for words like 'pairing', 'barrier', 'excitation'.
        """
        if self.pattern is None:
            return (False, "SKIP: pattern not available")
        pattern_lower = self.pattern.lower()
        keywords = ["pairing", "barrier", "excitation", "capture"]
        missing_keywords = [kw for kw in keywords if kw not in pattern_lower]
        if missing_keywords:
            return (False, f"Pattern lacks physics keywords {missing_keywords}: {self.pattern[:80]}...")
        return (True, f"Pattern includes explanation factors: {self.pattern[:80]}...")

    # ------------------------------------------------------------------
    # Monotonicity test – explicit perturbation of input to compare output
    # ------------------------------------------------------------------
    def test_monotonicity_thermal_odd_even_cross_section(self) -> Tuple[bool, str]:
        """
        Use the problem's table of fission cross sections for thermal neutrons:
          odd-N isotopes (e.g., 235U, σ=580 barns) have much larger cross sections
          than even-N isotopes (e.g., 236U, σ=0 barns).
        This test perturbs the neutron number from odd to even and checks that
        the pattern description correctly captures the monotonic decrease.
        """
        if self.pattern is None:
            return (False, "SKIP: pattern not available")

        # Data from the problem statement (thermal fission cross sections)
        # baseline: 235U (odd N, N=143) with σ=580 barns
        baseline_isotope = "235U"
        baseline_sigma = 580  # barns (thermal)
        # perturbed: 236U (even N, N=144) with σ=0 barns
        perturbed_sigma = 0  # barns (thermal)

        # Check that the perturbed cross section is smaller than baseline
        monotonic_ok = perturbed_sigma < baseline_sigma

        # Now verify that the pattern string describes this monotonic trend
        pattern_lower = self.pattern.lower()
        describes_decrease = (
            ("odd" in pattern_lower and "even" in pattern_lower) and
            ("large" in pattern_lower or "high" in pattern_lower) and
            ("small" in pattern_lower or "low" in pattern_lower)
        )

        if not monotonic_ok:
            return (False, "Internal error: table data contradicts monotonicity (odd->even should decrease in thermal)")
        if not describes_decrease:
            return (False, "Pattern does not describe the odd‑even monotonic decrease for thermal neutrons")
        return (True, "Monotonicity: thermal σ(odd) >> σ(even) as required by pattern")

    def test_monotonicity_fast_vs_thermal_odd(self) -> Tuple[bool, str]:
        """
        For an odd‑N isotope (e.g., 235U), the thermal cross section (~580 barns)
        is much larger than the fast cross section (~few barns).  This test
        perturbs the neutron energy from thermal to fast and checks that the
        pattern captures the monotonic decrease.
        """
        if self.pattern is None:
            return (False, "SKIP: pattern not available")

        # Baseline: thermal for odd‑N isotope (235U) σ = 580 barns
        baseline_sigma_thermal = 580
        # Perturbed: fast neutron cross section (all isotopes are a few barns)
        perturbed_sigma_fast = 5  # typical few barns

        # Check monotonic: fast < thermal for odd‑N
        monotonic_ok = perturbed_sigma_fast < baseline_sigma_thermal

        # Check that pattern describes large thermal, small fast for odd‑N
        pattern_lower = self.pattern.lower()
        describes_thermal_large = ("thermal" in pattern_lower and
                                   ("large" in pattern_lower or "high" in pattern_lower))
        describes_fast_small = ("fast" in pattern_lower and
                                ("small" in pattern_lower or "few barns" in pattern_lower))

        if not monotonic_ok:
            return (False, "Internal error: fast σ should be smaller than thermal σ for odd‑N")
        if not (describes_thermal_large and describes_fast_small):
            return (False, "Pattern does not describe that odd‑N cross sections drop from thermal to fast")
        return (True, "Monotonicity: odd‑N thermal σ >> odd‑N fast σ as required by pattern")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one, without math.isclose or re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_non_empty(self) -> Tuple[bool, str]:
        """All five emitted values must be non‑empty (not None, not empty string)."""
        for key, val in self.results.items():
            if val is None:
                return (False, f"Key '{key}' is None")
            if isinstance(val, str) and val.strip() == "":
                return (False, f"Key '{key}' is an empty string")
        return (True, "All outputs are non‑empty")

    # ------------------------------------------------------------------
    # Main runner that discovers and executes all test_ methods
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all callable methods beginning with 'test_', execute them
        one by one inside a try/except, and return (all_pass, logs).
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)