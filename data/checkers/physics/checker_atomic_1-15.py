import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    PhysicsChecker for Stern–Gerlach spin-1/2 beam splitting problem.
    Tests interface presence, invariant physics, modeling consistency,
    monotonicity of probability with polarization angle, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all required result keys with safe defaults.
        No numeric fallbacks; only None for missing.
        """
        # Safely access results dictionary
        results = getattr(solution_instance, 'results', {})

        # Probabilities for each sub-case
        self.P_plus_1 = results.get('P_plus_1', None)
        self.P_plus_2 = results.get('P_plus_2', None)
        self.P_plus_3 = results.get('P_plus_3', None)
        self.P_plus_4 = results.get('P_plus_4', None)

        # Pattern descriptions
        self.pattern_1 = results.get('pattern_1', None)
        self.pattern_2 = results.get('pattern_2', None)
        self.pattern_3 = results.get('pattern_3', None)
        self.pattern_4 = results.get('pattern_4', None)

        # Distinguishing methods
        self.distinguish_method_5 = results.get('distinguish_method_5', None)
        self.distinguish_method_6 = results.get('distinguish_method_6', None)

    # ------------------------------------------------------------
    # Interface tests – each required key exists and has correct type
    # ------------------------------------------------------------
    def test_interface_P_plus_1_exists(self) -> Tuple[bool, str]:
        if self.P_plus_1 is None:
            return (False, "P_plus_1 is missing (None)")
        if not isinstance(self.P_plus_1, (int, float)):
            return (False, f"P_plus_1 has wrong type: {type(self.P_plus_1)}")
        return (True, "P_plus_1 exists and is numeric")

    def test_interface_P_plus_2_exists(self) -> Tuple[bool, str]:
        if self.P_plus_2 is None:
            return (False, "P_plus_2 is missing (None)")
        if not isinstance(self.P_plus_2, (int, float)):
            return (False, f"P_plus_2 has wrong type: {type(self.P_plus_2)}")
        return (True, "P_plus_2 exists and is numeric")

    def test_interface_P_plus_3_exists(self) -> Tuple[bool, str]:
        if self.P_plus_3 is None:
            return (False, "P_plus_3 is missing (None)")
        if not isinstance(self.P_plus_3, (int, float)):
            return (False, f"P_plus_3 has wrong type: {type(self.P_plus_3)}")
        return (True, "P_plus_3 exists and is numeric")

    def test_interface_P_plus_4_exists(self) -> Tuple[bool, str]:
        if self.P_plus_4 is None:
            return (False, "P_plus_4 is missing (None)")
        if not isinstance(self.P_plus_4, (int, float)):
            return (False, f"P_plus_4 has wrong type: {type(self.P_plus_4)}")
        return (True, "P_plus_4 exists and is numeric")

    def test_interface_pattern_1_is_string(self) -> Tuple[bool, str]:
        if self.pattern_1 is None:
            return (False, "pattern_1 is missing (None)")
        if not isinstance(self.pattern_1, str):
            return (False, f"pattern_1 must be string, got {type(self.pattern_1)}")
        return (True, "pattern_1 is string")

    def test_interface_pattern_2_is_string(self) -> Tuple[bool, str]:
        if self.pattern_2 is None:
            return (False, "pattern_2 is missing (None)")
        if not isinstance(self.pattern_2, str):
            return (False, f"pattern_2 must be string, got {type(self.pattern_2)}")
        return (True, "pattern_2 is string")

    def test_interface_pattern_3_is_string(self) -> Tuple[bool, str]:
        if self.pattern_3 is None:
            return (False, "pattern_3 is missing (None)")
        if not isinstance(self.pattern_3, str):
            return (False, f"pattern_3 must be string, got {type(self.pattern_3)}")
        return (True, "pattern_3 is string")

    def test_interface_pattern_4_is_string(self) -> Tuple[bool, str]:
        if self.pattern_4 is None:
            return (False, "pattern_4 is missing (None)")
        if not isinstance(self.pattern_4, str):
            return (False, f"pattern_4 must be string, got {type(self.pattern_4)}")
        return (True, "pattern_4 is string")

    def test_interface_distinguish_method_5_exists(self) -> Tuple[bool, str]:
        if self.distinguish_method_5 is None:
            return (False, "distinguish_method_5 is missing (None)")
        if not isinstance(self.distinguish_method_5, str):
            return (False, f"distinguish_method_5 must be string, got {type(self.distinguish_method_5)}")
        return (True, "distinguish_method_5 exists and is string")

    def test_interface_distinguish_method_6_exists(self) -> Tuple[bool, str]:
        if self.distinguish_method_6 is None:
            return (False, "distinguish_method_6 is missing (None)")
        if not isinstance(self.distinguish_method_6, str):
            return (False, f"distinguish_method_6 must be string, got {type(self.distinguish_method_6)}")
        return (True, "distinguish_method_6 exists and is string")

    # ------------------------------------------------------------
    # Invariant sanity tests – physics relationships (use isclose)
    # ------------------------------------------------------------
    def test_invariant_P_plus_range(self) -> Tuple[bool, str]:
        """All probabilities must be in [0,1]."""
        probs = [self.P_plus_1, self.P_plus_2, self.P_plus_3, self.P_plus_4]
        for i, p in enumerate(probs, 1):
            if p is None:
                return (False, f"SKIP: P_plus_{i} is None")
            if not (0.0 <= p <= 1.0):
                return (False, f"P_plus_{i} = {p} is out of [0,1]")
        return (True, "All probabilities in [0,1]")

    def test_invariant_P_plus_specific_values(self) -> Tuple[bool, str]:
        """
        Physical constraints:
        - +z polarization => P_plus_1 = 1
        - +x, +y, unpolarized => P_plus = 0.5 (all equal)
        """
        if any(p is None for p in [self.P_plus_1, self.P_plus_2, self.P_plus_3, self.P_plus_4]):
            return (False, "SKIP: one or more P_plus missing")
        # +z case
        if not math.isclose(self.P_plus_1, 1.0, rel_tol=1e-6):
            return (False, f"P_plus_1 should be 1, got {self.P_plus_1}")
        # +x, +y, unpolarized all 0.5 (with lenient tolerance for SFT noise)
        expected_half = 0.5
        if not math.isclose(self.P_plus_2, expected_half, rel_tol=1e-2):
            return (False, f"P_plus_2 should be 0.5, got {self.P_plus_2}")
        if not math.isclose(self.P_plus_3, expected_half, rel_tol=1e-2):
            return (False, f"P_plus_3 should be 0.5, got {self.P_plus_3}")
        if not math.isclose(self.P_plus_4, expected_half, rel_tol=1e-2):
            return (False, f"P_plus_4 should be 0.5, got {self.P_plus_4}")
        return (True, "Probability values satisfy physics constraints")

    def test_invariant_pattern_consistency(self) -> Tuple[bool, str]:
        """
        Check pattern description matches P_plus: single spot iff P=1 or P=0,
        two spots iff P=0.5 (within tolerance).
        """
        cases = [
            (self.pattern_1, self.P_plus_1, 1),
            (self.pattern_2, self.P_plus_2, 2),
            (self.pattern_3, self.P_plus_3, 3),
            (self.pattern_4, self.P_plus_4, 4)
        ]
        for pat, p, idx in cases:
            if pat is None or p is None:
                return (False, f"SKIP: missing data for case {idx}")
            # Determine expected pattern nature
            if math.isclose(p, 1.0, rel_tol=1e-6) or math.isclose(p, 0.0, rel_tol=1e-6):
                expected_nature = "single spot"
            elif math.isclose(p, 0.5, rel_tol=1e-2):
                expected_nature = "two spots"
            else:
                continue
            pat_lower = pat.lower()
            if expected_nature == "single spot":
                if "not split" in pat_lower or "single" in pat_lower or "not separated" in pat_lower:
                    continue
                else:
                    return (False, f"Pattern {idx} does not indicate single spot (got: {pat})")
            else:  # two spots
                if "split" in pat_lower or "two" in pat_lower or "deflected to +z and -z" in pat_lower:
                    continue
                else:
                    return (False, f"Pattern {idx} does not indicate two spots (got: {pat})")
        return (True, "Pattern descriptions consistent with probabilities")

    # ------------------------------------------------------------
    # Modeling / trace tests – based on explanation steps
    # ------------------------------------------------------------
    def test_modeling_distinguish_method_5_physics(self) -> Tuple[bool, str]:
        """
        Method 5 should involve rotating the magnetic field to measure along +y.
        Check that description mentions 'y' or 'rotate' or 'second SG' etc.
        """
        if self.distinguish_method_5 is None:
            return (False, "SKIP: distinguish_method_5 missing")
        text = self.distinguish_method_5.lower()
        keywords = ['y', 'rotate', 'turn magnetic field', 'second', 'orientation']
        found = any(kw in text for kw in keywords)
        if not found:
            return (False, f"distinguish_method_5 does not mention y-field or rotation: {self.distinguish_method_5}")
        return (True, "Method 5 mentions appropriate physics (e.g., y-field)")

    def test_modeling_distinguish_method_6_physics(self) -> Tuple[bool, str]:
        """
        Method 6 should discriminate between |+x> and unpolarized.
        Common methods: second SG with x-gradient, or use of reflector (spin rotator).
        """
        if self.distinguish_method_6 is None:
            return (False, "SKIP: distinguish_method_6 missing")
        text = self.distinguish_method_6.lower()
        keywords = ['reflector', 'second sg', 'x-gradient', 'half-wave', 'spin rotator',
                     'second', 'another sg', 'x direction']
        found = any(kw in text for kw in keywords)
        if not found:
            return (False, f"distinguish_method_6 does not describe a valid discrimination: {self.distinguish_method_6}")
        return (True, "Method 6 describes a valid further experiment")

    def test_modeling_indistinguishable_cases_logic(self) -> Tuple[bool, str]:
        """
        The fact that patterns for (ii), (iii), (iv) are all 'two spots' and that we have
        distinguish methods implies logical consistency. Check methods exist.
        """
        if not self.distinguish_method_5 or not self.distinguish_method_6:
            return (False, "Distinguish methods must be provided for indistinguishable cases")
        return (True, "Indistinguishable cases have associated distinguish methods")

    # ------------------------------------------------------------
    # Monotonicity test – perturb polarization angle from +z to +x
    # ------------------------------------------------------------
    def test_monotonicity_polarization_angle_decreases_probability(self) -> Tuple[bool, str]:
        """
        For a spin-1/2 particle polarized at an angle θ measured from +z,
        the probability for μ_z = +μ is P(θ) = cos²(θ/2).
        This function strictly decreases from θ=0 to θ=π/2.
        We test that P_plus_1 (θ=0) > P_plus_2 (θ=π/2).
        This is a monotonicity check: increasing θ reduces P.
        """
        if self.P_plus_1 is None or self.P_plus_2 is None:
            return (False, "SKIP: P_plus_1 or P_plus_2 missing")
        # Baseline: P_plus_1 for θ=0
        # Perturbation: increase θ to π/2 -> P_plus_2
        if not (self.P_plus_2 < self.P_plus_1):
            return (False, f"P_plus_2 = {P_plus_2} should be less than P_plus_1 = {self.P_plus_1} (monotonic decrease)")
        return (True, "Probability decreases as polarization angle increases from 0 to π/2")

    # ------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------
    def test_answer_sanity_broad_checks(self) -> Tuple[bool, str]:
        """
        Weak, non-close check: all patterns are non-empty strings, all probabilities are numeric.
        No derivation of expected values.
        """
        all_patterns = [self.pattern_1, self.pattern_2, self.pattern_3, self.pattern_4]
        for i, pat in enumerate(all_patterns, 1):
            if pat is None or not isinstance(pat, str) or pat.strip() == "":
                return (False, f"Pattern {i} is missing or empty")
        all_probs = [self.P_plus_1, self.P_plus_2, self.P_plus_3, self.P_plus_4]
        for i, p in enumerate(all_probs, 1):
            if p is None or not isinstance(p, (int, float)):
                return (False, f"Probability {i} is missing or non-numeric")
        # Broad range: all probabilities between 0 and 1 (already tested, but that is range, not derivation)
        if not all(0.0 <= p <= 1.0 for p in all_probs):
            return (False, "Probabilities out of [0,1]")
        return (True, "All answer fields are present and of correct type")

    # ------------------------------------------------------------
    # Test runner (discovery + execution)
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Gather all test methods
        for attr_name in dir(self):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{attr_name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)