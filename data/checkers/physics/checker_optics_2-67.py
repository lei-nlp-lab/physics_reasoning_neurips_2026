import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics checker for Fraunhofer diffraction by a slit and resolving power of a telescope.
    Checks the correctness of reasoning, modeling, and results.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens, results, explanation_steps, pseudocode
        self.givens: dict = getattr(solution_instance, 'givens', {}) or {}
        self.results: dict = getattr(solution_instance, 'results', {}) or {}
        self.explanation_steps: list = getattr(solution_instance, 'explanation_steps', None) or []
        self.pseudocode: str = getattr(solution_instance, 'pseudocode', '') or ''

        # Extract required physical quantities (use None for missing)
        self.lambda_val = self.givens.get('lambda', None) or self.givens.get('wavelength', None)
        self.moon_distance = self.givens.get('moon_distance', None) or self.givens.get('moon_distance_m', None)
        self.line_separation = self.givens.get('line_separation', None) or self.givens.get('line_separation_m', None)

        # Results from pseudocode emit contract
        self.theta_min = self.results.get('theta_min', None)
        self.w = self.results.get('w', None)

        # Also try to extract from givens if w is not in results
        if self.w is None:
            self.w = self.givens.get('slit_width', None) or self.givens.get('w', None)

    # --------------------------------------------------------------------------
    # Interface tests: required keys and types
    # --------------------------------------------------------------------------
    def test_interface_theta_min_exists(self) -> Tuple[bool, str]:
        """Check that theta_min is present and is a string (symbolic expression)."""
        if self.theta_min is None:
            return (False, "theta_min missing from results")
        if not isinstance(self.theta_min, str):
            return (False, "theta_min must be a string (symbolic expression)")
        if 'lambda' not in self.theta_min.lower() or 'w' not in self.theta_min.lower():
            return (False, f"theta_min = '{self.theta_min}' does not contain lambda and w")
        return (True, f"theta_min = '{self.theta_min}' is a valid symbolic expression")

    def test_interface_w_exists(self) -> Tuple[bool, str]:
        """Check that w is present and numeric (float or int)."""
        if self.w is None:
            return (False, "w missing from results")
        if not isinstance(self.w, (int, float)):
            return (False, f"w must be numeric, got {type(self.w).__name__}")
        if math.isnan(self.w):
            return (False, "w is NaN")
        return (True, f"w = {self.w} is a valid numeric value")

    # --------------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity) – use math.isclose
    # --------------------------------------------------------------------------
    def test_invariant_fraunhofer_first_minimum(self) -> Tuple[bool, str]:
        """
        Check Fraunhofer first minimum condition: w * sin(theta) = lambda.
        Use the angular separation of lines as theta (resolution condition).
        Only if numeric lambda, w, moon_distance, line_separation are available.
        """
        if self.lambda_val is None or self.w is None or self.moon_distance is None or self.line_separation is None:
            return (False, "SKIP: missing numeric lambda, w, moon_distance, or line_separation")
        theta_lines = self.line_separation / self.moon_distance
        lhs = self.w * math.sin(theta_lines)
        rhs = self.lambda_val
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"w*sin(theta_lines) = {lhs:.6e} ≈ lambda = {rhs:.6e}")
        else:
            return (False, f"Fraunhofer condition failed: w*sin(theta) = {lhs:.6e}, expected ≈ {rhs:.6e}")

    def test_invariant_small_angle_approximation(self) -> Tuple[bool, str]:
        """
        Check that sin(theta) ≈ theta for the first minimum / resolution angle.
        """
        if self.lambda_val is None or self.w is None:
            return (False, "SKIP: missing numeric lambda or w")
        theta = self.lambda_val / self.w  # small-angle estimate
        if theta > 0.1:  # approximation valid only for small angles
            return (False, f"Angle theta = {theta} rad is too large for sin≈theta approximation")
        sin_theta = math.sin(theta)
        if math.isclose(sin_theta, theta, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"sin(theta) ≈ theta, theta = {theta}, sin = {sin_theta}")
        else:
            return (False, f"Small-angle approximation fails: theta={theta}, sin(theta)={sin_theta}")

    # --------------------------------------------------------------------------
    # Modeling quality tests (based on explanation steps and physics)
    # --------------------------------------------------------------------------
    def test_modeling_theta_lines_computation(self) -> Tuple[bool, str]:
        """
        Check that angular separation of lines is correctly computed from givens.
        Also verify consistency with resolution condition if w is available.
        """
        if self.line_separation is None or self.moon_distance is None:
            return (False, "SKIP: missing line_separation or moon_distance")
        expected_theta_lines = self.line_separation / self.moon_distance
        if self.w is not None and self.lambda_val is not None:
            theta_from_resolution = self.lambda_val / self.w
            if math.isclose(expected_theta_lines, theta_from_resolution, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"theta_lines = {expected_theta_lines:.6e} rad, consistent with lambda/w = {theta_from_resolution:.6e}")
            else:
                return (False, f"theta_lines = {expected_theta_lines:.6e} does not match lambda/w = {theta_from_resolution:.6e}")
        # If no w, return SKIP
        return (False, "SKIP: cannot verify theta_lines without w")

    def test_modeling_w_from_resolution(self) -> Tuple[bool, str]:
        """
        Check that aperture width w is correctly derived: w = lambda * moon_distance / line_separation.
        """
        if self.lambda_val is None or self.moon_distance is None or self.line_separation is None:
            return (False, "SKIP: missing lambda, moon_distance, or line_separation")
        expected_w = self.lambda_val * self.moon_distance / self.line_separation
        if self.w is None:
            return (False, "Cannot compare: w missing from results")
        if math.isclose(self.w, expected_w, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"w = {self.w} m matches expected value from resolution condition: {expected_w:.6e} m")
        else:
            return (False, f"w = {self.w} m does not match expected {expected_w:.6e} m (derived from lambda*moon_distance/line_separation)")

    def test_modeling_theta_min_expression(self) -> Tuple[bool, str]:
        """
        Check if theta_min is the correct symbolic expression (lambda/w).
        """
        if self.theta_min is None:
            return (False, "SKIP: theta_min not available")
        expr = self.theta_min.replace(' ', '').lower()
        if expr in ('lambda/w', 'λ/w', '\\lambda/w', 'lambda/w'):
            return (True, f"theta_min = '{self.theta_min}' matches expected symbolic form")
        if 'lambda' in expr and 'w' in expr and '/' in expr:
            return (True, f"theta_min = '{self.theta_min}' likely correct")
        return (False, f"theta_min = '{self.theta_min}' does not look like lambda/w")

    # --------------------------------------------------------------------------
    # Monotonicity tests (trend checks)
    # --------------------------------------------------------------------------
    def test_monotonicity_theta_increases_with_lambda(self) -> Tuple[bool, str]:
        """
        Perturb lambda slightly upward; theta_min (lambda/w) should increase.
        Only if numeric lambda and w are available.
        """
        if self.lambda_val is None or self.w is None:
            return (False, "SKIP: missing lambda or w")
        baseline_theta = self.lambda_val / self.w
        perturbed_lambda = self.lambda_val * 1.05
        perturbed_theta = perturbed_lambda / self.w
        if perturbed_theta > baseline_theta:
            return (True, f"theta increases when lambda increases: baseline={baseline_theta:.6e}, perturbed={perturbed_theta:.6e}")
        else:
            return (False, f"theta did not increase: baseline={baseline_theta}, perturbed={perturbed_theta}")

    def test_monotonicity_theta_decreases_with_w(self) -> Tuple[bool, str]:
        """
        Perturb w slightly upward; theta_min (lambda/w) should decrease.
        """
        if self.lambda_val is None or self.w is None:
            return (False, "SKIP: missing lambda or w")
        baseline_theta = self.lambda_val / self.w
        perturbed_w = self.w * 1.05
        perturbed_theta = self.lambda_val / perturbed_w
        if perturbed_theta < baseline_theta:
            return (True, f"theta decreases when w increases: baseline={baseline_theta:.6e}, perturbed={perturbed_theta:.6e}")
        else:
            return (False, f"theta did not decrease: baseline={baseline_theta}, perturbed={perturbed_theta}")

    # --------------------------------------------------------------------------
    # Weak answer sanity (single method, no hardcoded target values)
    # --------------------------------------------------------------------------
    def test_answer_sanity_w_range(self) -> Tuple[bool, str]:
        """
        Check that w is positive and of plausible order (0.1 to 1 m).
        """
        if self.w is None:
            return (False, "SKIP: w missing")
        if self.w <= 0:
            return (False, f"w must be positive, got {self.w}")
        if self.w > 5.0:
            return (False, f"w = {self.w} m is unreasonably large for aperture width")
        if self.w < 0.01:
            return (False, f"w = {self.w} m is too small (unphysical)")
        return (True, f"w = {self.w} m is in plausible range")

    # --------------------------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------------------------
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
                passed, msg = False, f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)