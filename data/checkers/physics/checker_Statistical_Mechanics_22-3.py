import math
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Robust physics checker for the sound wave speed problem.
    Tests interface, invariants, modeling consistency, monotonicity,
    and a single weak answer sanity check.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all needed data safely from solution_instance.
        Use None defaults for all missing values.
        """
        # Givens (expected from problem context)
        self.R = getattr(solution_instance, 'R', None)
        self.T = getattr(solution_instance, 'T', None)
        self.gamma = getattr(solution_instance, 'gamma', None)

        # Results from the [emit] contract
        if hasattr(solution_instance, 'results'):
            res = solution_instance.results
        else:
            res = {}
        self.v2_formula = res.get('v2_formula', None)
        self.v2_isothermal = res.get('v2_isothermal', None)
        self.v2_adiabatic = res.get('v2_adiabatic', None)
        self.v_isothermal_num = res.get('v_isothermal_num', None)
        self.v_adiabatic_num = res.get('v_adiabatic_num', None)
        self.better_model = res.get('better_model', None)

        # Explanation steps (optional, for modeling quality)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------
    # Interface checks – verify that all required keys exist
    # ----------------------------------------------------------------
    def test_interface_results_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present (not None)."""
        required_keys = [
            'v2_formula', 'v2_isothermal', 'v2_adiabatic',
            'v_isothermal_num', 'v_adiabatic_num', 'better_model'
        ]
        missing = []
        for key in required_keys:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    # ----------------------------------------------------------------
    # Invariant checks – theorem-based physical sanity
    # ----------------------------------------------------------------
    def test_invariant_v2_formula_derivation(self) -> Tuple[bool, str]:
        """
        Check that v2_formula correctly states 'v^2 = dp/drho' (symbolic truth).
        This is the key invariant from the wave equation derivation.
        """
        if self.v2_formula is None:
            return (False, "SKIP: v2_formula not available")
        expected = "v^2 = dp/drho"
        if isinstance(self.v2_formula, str):
            if self.v2_formula.strip() == expected:
                return (True, "v2_formula matches expected expression")
            else:
                return (False, f"v2_formula is '{self.v2_formula}', expected '{expected}'")
        elif hasattr(self.v2_formula, 'simplify'):  # could be a sympy expression
            return (True, "v2_formula is a symbolic expression (assumed correct)")
        else:
            return (False, f"Unexpected type for v2_formula: {type(self.v2_formula)}")

    def test_invariant_isothermal_dpdrho(self) -> Tuple[bool, str]:
        """
        For isothermal ideal gas, dp/drho = RT.  Check that v2_isothermal 
        is consistent with the formula v^2 = dp/drho and the ideal gas law.
        """
        if self.v2_isothermal is None or self.R is None or self.T is None:
            return (False, "SKIP: insufficient data for isothermal dp/drho check")
        expected = self.R * self.T
        if math.isclose(self.v2_isothermal, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v2_isothermal ≈ {expected} (R*T)")
        else:
            return (False, f"v2_isothermal = {self.v2_isothermal}, expected ≈ {expected}")

    def test_invariant_adiabatic_dpdrho(self) -> Tuple[bool, str]:
        """
        For adiabatic ideal gas, dp/drho = gamma * RT.
        Check v2_adiabatic is consistent.
        """
        if self.v2_adiabatic is None or self.gamma is None or self.R is None or self.T is None:
            return (False, "SKIP: insufficient data for adiabatic dp/drho check")
        expected = self.gamma * self.R * self.T
        if math.isclose(self.v2_adiabatic, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v2_adiabatic ≈ {expected} (gamma*R*T)")
        else:
            return (False, f"v2_adiabatic = {self.v2_adiabatic}, expected ≈ {expected}")

    # ----------------------------------------------------------------
    # Modeling / trace checks – based on explanation steps or computed values
    # ----------------------------------------------------------------
    def test_modeling_numerical_isothermal(self) -> Tuple[bool, str]:
        """Verify v_isothermal_num ≈ sqrt(R*T)."""
        if self.v_isothermal_num is None or self.R is None or self.T is None:
            return (False, "SKIP: missing data for isothermal numerical speed")
        expected = math.sqrt(self.R * self.T)
        if math.isclose(self.v_isothermal_num, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_isothermal_num ≈ {expected:.1f} m/s")
        else:
            return (False, f"v_isothermal_num = {self.v_isothermal_num}, expected ≈ {expected:.1f}")

    def test_modeling_numerical_adiabatic(self) -> Tuple[bool, str]:
        """Verify v_adiabatic_num ≈ sqrt(gamma*R*T)."""
        if self.v_adiabatic_num is None or self.gamma is None or self.R is None or self.T is None:
            return (False, "SKIP: missing data for adiabatic numerical speed")
        expected = math.sqrt(self.gamma * self.R * self.T)
        if math.isclose(self.v_adiabatic_num, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v_adiabatic_num ≈ {expected:.1f} m/s")
        else:
            return (False, f"v_adiabatic_num = {self.v_adiabatic_num}, expected ≈ {expected:.1f}")

    def test_modeling_better_model_string(self) -> Tuple[bool, str]:
        """Check that better_model is the string 'adiabatic'."""
        if self.better_model is None:
            return (False, "SKIP: better_model not available")
        if not isinstance(self.better_model, str):
            return (False, f"better_model should be string, got {type(self.better_model)}")
        if self.better_model.lower().strip() == 'adiabatic':
            return (True, "better_model = 'adiabatic'")
        else:
            return (False, f"better_model = '{self.better_model}', expected 'adiabatic'")

    def test_trace_explanation_steps_contains_derivation(self) -> Tuple[bool, str]:
        """
        Check that explanation_steps (if provided) contains key phrases
        indicating the wave equation derivation. This tests modeling quality.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        # Convert to string if it's a list of strings
        if isinstance(self.explanation_steps, list):
            text = " ".join(str(step) for step in self.explanation_steps)
        else:
            text = str(self.explanation_steps)
        required_phrases = ["wave equation", "continuity", "Euler"]
        found = [phrase for phrase in required_phrases if phrase.lower() in text.lower()]
        if len(found) == len(required_phrases):
            return (True, "Explanation steps include derivation of wave equation")
        else:
            return (False, f"Missing phrases: {set(required_phrases) - set(found)}")

    # ----------------------------------------------------------------
    # Monotonicity checks – ensure speeds increase with temperature, gamma, etc.
    # ----------------------------------------------------------------
    def test_monotonicity_temperature_on_isothermal_speed(self) -> Tuple[bool, str]:
        """
        Verify that the isothermal speed increases when temperature increases
        (v ∝ sqrt(T)). We perturb T by +5% and compare the expected new speed
        to the original reported speed.
        """
        if self.v_isothermal_num is None or self.R is None or self.T is None:
            return (False, "SKIP: missing data for isothermal monotonicity test")
        if self.T <= 0:
            return (False, "SKIP: T must be positive for monotonicity test")

        # Perturb T upward by 5%
        T_perturbed = self.T * 1.05
        v_perturbed = math.sqrt(self.R * T_perturbed)   # expected speed at higher T
        v_original = self.v_isothermal_num

        if v_perturbed > v_original:
            return (True, f"Isothermal speed increases with T: {v_perturbed:.2f} > {v_original:.2f}")
        else:
            return (False, f"Isothermal speed does not increase with T: {v_perturbed:.2f} <= {v_original:.2f}")

    def test_monotonicity_gamma_on_adiabatic_speed(self) -> Tuple[bool, str]:
        """
        Verify that the adiabatic speed increases when gamma increases
        (v ∝ sqrt(gamma)). We perturb gamma by +5% and compare.
        """
        if self.v_adiabatic_num is None or self.gamma is None or self.R is None or self.T is None:
            return (False, "SKIP: missing data for gamma monotonicity test")
        if self.gamma <= 0:
            return (False, "SKIP: gamma must be positive")

        # Perturb gamma upward by 5%
        gamma_perturbed = self.gamma * 1.05
        v_perturbed = math.sqrt(gamma_perturbed * self.R * self.T)
        v_original = self.v_adiabatic_num

        if v_perturbed > v_original:
            return (True, f"Adiabatic speed increases with gamma: {v_perturbed:.2f} > {v_original:.2f}")
        else:
            return (False, f"Adiabatic speed does not increase with gamma: {v_perturbed:.2f} <= {v_original:.2f}")

    # ----------------------------------------------------------------
    # Single weak answer sanity check – combine positivity and plausible range
    # ----------------------------------------------------------------
    def test_answer_sanity_speed_properties(self) -> Tuple[bool, str]:
        """
        Weak sanity: both numerical speeds are positive and lie in a plausible
        range for sound in air (200-500 m/s). No hardcoded target values.
        """
        values = [self.v_isothermal_num, self.v_adiabatic_num]
        if any(v is None for v in values):
            return (False, "SKIP: one or both numerical speeds missing")
        if any(v <= 0 for v in values):
            negative = [v for v in values if v <= 0]
            return (False, f"Non-positive speeds found: {negative}")
        if any(v < 200 or v > 500 for v in values):
            out_of_range = [v for v in values if not (200 < v < 500)]
            return (False, f"Speeds outside plausible range (200-500 m/s): {out_of_range}")
        return (True, "Both speeds are positive and in plausible range (200-500 m/s)")

    # ----------------------------------------------------------------
    # Master runner
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute all test methods, collect logs."""
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
                passed, msg = False, f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)