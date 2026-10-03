import math
import sympy
import traceback
from typing import Any, Optional

class PhysicsChecker:
    """
    A robust physics checker for the rainbow multiple-choice problem.
    Checks interface, physical invariants, modeling consistency, monotonicity, and answer sanity.
    Safely handles missing data with None defaults.
    """

    def __init__(self, solution_instance: Optional[Any] = None) -> None:
        """
        Extract all relevant data from the solution instance using safe defaults.
        """
        if solution_instance is None:
            # Ensure all attributes are None to prevent crashes in tests
            self.correct_option = None
            self.explanation_steps = None
            return

        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', None) or {}

        # Result key from the pseudocode emit contract
        self.correct_option = results.get('correct_option', None)

        # Explanation steps (might be a string or list of strings)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            # Also check if it's stored in results
            self.explanation_steps = results.get('explanation_steps', None)

    # ---------------------------------------------------------------
    # Interface tests – mandatory key existence and type
    # ---------------------------------------------------------------
    def test_interface_correct_option_exists(self):
        """The result must contain the key 'correct_option'."""
        if self.correct_option is None:
            return (False, "Missing required key 'correct_option' in results")
        return (True, "'correct_option' is present")

    def test_interface_correct_option_type(self):
        """correct_option must be a string."""
        if self.correct_option is None:
            return (True, "SKIP: correct_option not available")
        if not isinstance(self.correct_option, str):
            return (False, f"correct_option must be a string, got {type(self.correct_option).__name__}")
        return (True, "correct_option is a string")

    def test_interface_correct_option_in_allowed(self):
        """correct_option must be one of '(a)', '(b)', '(c)'."""
        if self.correct_option is None:
            return (True, "SKIP: correct_option not available")
        allowed = {'(a)', '(b)', '(c)'}
        if self.correct_option not in allowed:
            return (False, f"correct_option '{self.correct_option}' is not one of {allowed}")
        return (True, f"correct_option is a valid option: {self.correct_option}")

    # ---------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity
    # ---------------------------------------------------------------
    def test_invariant_cloud_reflection_achromatic(self):
        """
        Cloud reflection gives white light, not a rainbow.
        If answer is (b), it violates physics.
        """
        if self.correct_option is None:
            return (True, "SKIP: correct_option not available")
        if self.correct_option == '(b)':
            return (False, "Reflection by clouds yields white light (achromatic), not a rainbow")
        return (True, "Answer is consistent with cloud reflection being achromatic")

    def test_invariant_eye_refraction_not_disperse(self):
        """
        Refraction in the human eye only focuses light, does not create colors.
        If answer is (c), it violates physics.
        """
        if self.correct_option is None:
            return (True, "SKIP: correct_option not available")
        if self.correct_option == '(c)':
            return (False, "Refraction in the human eye focuses light but does not cause dispersion/rainbow")
        return (True, "Answer is consistent with eye refraction not producing colors")

    def test_invariant_rainbow_requires_internal_reflection(self):
        """
        Rainbow formation requires internal reflection inside water droplets.
        If explanation is available, check consistency.
        """
        if self.explanation_steps is None:
            return (True, "SKIP: explanation_steps not available")
        if self.correct_option is None:
            return (True, "SKIP: correct_option not available")
        steps_lower = str(self.explanation_steps).lower()
        if 'internal reflection' not in steps_lower:
            # If explanation does not mention internal reflection but answer is (a), that's suspicious
            if self.correct_option == '(a)':
                return (True, "Explanation does not explicitly mention internal reflection, but answer is (a)")
            else:
                return (False, "Rainbow requires internal reflection; either explanation is incomplete or answer is wrong")
        return (True, "Explanation mentions internal reflection")

    # ---------------------------------------------------------------
    # Modeling tests – checks consistency with explanation steps
    # ---------------------------------------------------------------
    def test_modeling_explanation_mentions_refraction(self):
        """
        Refraction is key for rainbow formation.
        If explanation is present, it should mention refraction.
        """
        if self.explanation_steps is None:
            return (True, "SKIP: explanation_steps not available")
        if self.correct_option is None:
            return (True, "SKIP: correct_option not available")
        steps_lower = str(self.explanation_steps).lower()
        if 'refraction' not in steps_lower:
            if self.correct_option == '(a)':
                return (False, "Explanation for rainbow should mention refraction, but it does not")
            return (True, "Explanation does not mention refraction, but answer is not (a), so consistent")
        return (True, "Explanation mentions refraction")

    def test_modeling_explanation_mentions_dispersion(self):
        """
        Color separation comes from dispersion.
        If explanation is present, it should mention dispersion or 'n(lambda)'.
        """
        if self.explanation_steps is None:
            return (True, "SKIP: explanation_steps not available")
        if self.correct_option is None:
            return (True, "SKIP: correct_option not available")
        steps_lower = str(self.explanation_steps).lower()
        keywords = ['dispersion', 'n(', 'wavelength']
        if not any(k in steps_lower for k in keywords):
            if self.correct_option == '(a)':
                return (False, "Rainbow explanation should mention dispersion/wavelength dependence")
            return (True, "No dispersion mentioned, but answer is not (a)")
        return (True, "Explanation mentions dispersion")

    # ---------------------------------------------------------------
    # Monotonicity test – explicitly perturbs wavelength to check deviation ordering
    # ---------------------------------------------------------------
    def test_monotonicity_deviation_angle_decreases_with_wavelength(self):
        """
        Using Snell's law and the primary rainbow deviation formula,
        compute the deviation angle for red light (baseline) and violet light (perturbed).
        Perturbing from longer to shorter wavelength (red to violet) should decrease the deviation.
        """
        # Standard approximate refractive indices for water at visible wavelengths
        n_air = 1.0003
        n_water_red = 1.331    # ~700 nm (baseline)
        n_water_violet = 1.342 # ~400 nm (perturbed)

        # Typical incidence angle for primary rainbow (60 degrees) in radians
        theta_i_deg = 60.0
        theta_i = math.radians(theta_i_deg)

        def deviation_angle(n_water):
            sin_theta_r = (n_air / n_water) * math.sin(theta_i)
            if abs(sin_theta_r) > 1.0:
                return None
            theta_r = math.asin(sin_theta_r)
            # Deviation angle for primary rainbow: D = 4*theta_r - 2*theta_i
            D = 4 * theta_r - 2 * theta_i
            return math.degrees(D)

        D_red = deviation_angle(n_water_red)
        D_violet = deviation_angle(n_water_violet)

        if D_red is None or D_violet is None:
            return (True, "SKIP: could not compute deviation (unphysical incidence angle)")

        # Baseline: red (longer wavelength) should have larger deviation
        # Perturbation: switch to violet (shorter wavelength) -> deviation should decrease
        if D_red <= D_violet:
            return (False, f"Monotonicity violation: red deviation {D_red:.2f}° should be > violet deviation {D_violet:.2f}°")
        return (True, f"Red deviation {D_red:.2f}° > violet deviation {D_violet:.2f}°, consistent with physics")

    # ---------------------------------------------------------------
    # Answer sanity test (weak, single test allowed)
    # ---------------------------------------------------------------
    def test_answer_sanity_option_not_empty(self):
        """
        Sanity: the option should be a non-empty string.
        This is a weak check; it does not compute any expected value.
        """
        if not self.correct_option:
            return (True, "SKIP: correct_option missing, cannot check")
        if len(self.correct_option.strip()) == 0:
            return (False, "correct_option is only whitespace")
        return (True, "correct_option is a non-empty string")

    # ---------------------------------------------------------------
    # Main runner: discover and execute all test methods
    # ---------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all test_ methods, execute them safely, collect logs.
        Returns (all_pass, logs_list).
        """
        logs = []
        all_pass = True
        # Iterate over sorted attributes to have deterministic order
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