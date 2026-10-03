import math
import traceback

class PhysicsChecker:
    """
    Checker for a diverging lens problem: parallel rays incident at 5°,
    focal length -20 cm. Checks model correctness, interface completeness,
    physical invariants, and monotonicity of key relationships.
    """

    def __init__(self, solution_instance):
        # Extract results safely; default to None if missing
        results = getattr(solution_instance, 'results', None) or {}
        self.x_img = results.get('x_img', None)
        self.y_img = results.get('y_img', None)
        self.dist_axis = results.get('dist_axis', None)

        # Also store the instance for possible givens
        self.solution_instance = solution_instance
        # Extract givens if available (used in monotonicity tests)
        self.f = getattr(solution_instance, 'f', None) or getattr(solution_instance, 'focal_length', None) or -20.0
        self.theta_deg = getattr(solution_instance, 'theta_deg', None) or getattr(solution_instance, 'theta', None) or 5.0

    def run_all_tests(self):
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---------- Interface checks ----------
    def test_interface_required_keys_exist(self):
        """Check that all three required keys are present."""
        if self.x_img is None:
            return (False, "SKIP: x_img not available")
        if self.y_img is None:
            return (False, "SKIP: y_img not available")
        if self.dist_axis is None:
            return (False, "SKIP: dist_axis not available")
        return (True, "All required keys present")

    def test_interface_numeric_types(self):
        """Check that values are numeric (int/float)."""
        for name, val in [('x_img', self.x_img), ('y_img', self.y_img), ('dist_axis', self.dist_axis)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (type: {type(val).__name__})")
        return (True, "All values are numeric")

    # ---------- Physical invariant checks (use math.isclose) ----------
    def test_invariant_sign_x_img(self):
        """Virtual image on same side as incident light -> x_img negative."""
        if self.x_img is None:
            return (False, "SKIP: x_img not available")
        if self.x_img >= 0:
            return (False, f"x_img should be negative, got {self.x_img}")
        return (True, "x_img is negative as expected")

    def test_invariant_sign_y_img(self):
        """For positive incident angle (5°), y_img should be negative (f < 0)."""
        if self.y_img is None:
            return (False, "SKIP: y_img not available")
        if self.y_img >= 0:
            return (False, f"y_img should be negative, got {self.y_img}")
        return (True, "y_img is negative as expected")

    def test_invariant_positive_dist_axis(self):
        """dist_axis must be positive (absolute distance)."""
        if self.dist_axis is None:
            return (False, "SKIP: dist_axis not available")
        if self.dist_axis <= 0:
            return (False, f"dist_axis should be positive, got {self.dist_axis}")
        return (True, "dist_axis is positive")

    # ---------- Modeling / trace consistency checks ----------
    def test_modeling_dist_axis_equals_abs_y(self):
        """dist_axis must equal abs(y_img)."""
        if self.dist_axis is None or self.y_img is None:
            return (False, "SKIP: missing dist_axis or y_img")
        expected = abs(self.y_img)
        if math.isclose(self.dist_axis, expected, rel_tol=1e-9, abs_tol=1e-9):
            return (True, "dist_axis == abs(y_img)")
        else:
            return (False, f"dist_axis = {self.dist_axis}, expected {expected}")

    def test_modeling_focal_plane_x(self):
        """For parallel rays, image lies in focal plane -> x_img should equal f = -20 cm."""
        if self.x_img is None:
            return (False, "SKIP: x_img not available")
        expected_f = -20.0
        if math.isclose(self.x_img, expected_f, rel_tol=1e-3, abs_tol=1e-3):
            return (True, f"x_img ≈ {expected_f} cm (focal plane)")
        else:
            return (False, f"x_img = {self.x_img}, expected approx {expected_f} cm")

    def test_modeling_tan_relation(self):
        """
        Check y_img / x_img ≈ tan(5°).
        This verifies the central ray trajectory through the lens.
        """
        if self.x_img is None or self.y_img is None:
            return (False, "SKIP: missing x_img or y_img")
        if math.isclose(self.x_img, 0, abs_tol=1e-9):
            return (False, "x_img is zero, cannot compute ratio")
        ratio = self.y_img / self.x_img
        expected_tan = math.tan(math.radians(5.0))
        if math.isclose(ratio, expected_tan, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"y_img/x_img ≈ tan(5°) = {expected_tan:.4f}")
        else:
            return (False, f"y_img/x_img = {ratio:.4f}, expected ~{expected_tan:.4f}")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_angle_increase(self):
        """
        If incident angle increases, |y_img| should increase.
        Perturb theta from baseline 5° to 6° and recompute y_img.
        """
        if self.y_img is None or self.f is None:
            return (False, "SKIP: y_img or f not available")
        baseline_angle_deg = self.theta_deg
        perturbed_angle_deg = baseline_angle_deg * 1.2  # +20%
        # Recompute y_img for perturbed angle using same f
        perturbed_y = self.f * math.tan(math.radians(perturbed_angle_deg))
        # Check monotonic: |perturbed_y| > |original y_img|
        if abs(perturbed_y) > abs(self.y_img):
            return (True, f"Increasing angle from {baseline_angle_deg:.1f}° to {perturbed_angle_deg:.1f}° increases |y_img|: "
                          f"original |y| = {abs(self.y_img):.4f}, perturbed |y| = {abs(perturbed_y):.4f}")
        else:
            return (False, f"Monotonicity violation: angle increased but |y_img| did not increase. "
                          f"original |y| = {abs(self.y_img):.4f}, perturbed |y| = {abs(perturbed_y):.4f}")

    def test_monotonicity_focal_length_magnitude_increase(self):
        """
        If focal length becomes more negative (e.g., -22 cm instead of -20 cm),
        both x_img and y_img should become more negative (i.e., decrease).
        Perturb f by 10% more negative.
        """
        if self.x_img is None or self.y_img is None:
            return (False, "SKIP: x_img or y_img not available")
        baseline_f = self.f
        # Perturb f: more negative by 10% (e.g., -22 if baseline=-20)
        perturbed_f = 1.1 * baseline_f  # e.g., -22
        # Recompute x_img and y_img
        perturbed_x = perturbed_f
        perturbed_y = perturbed_f * math.tan(math.radians(self.theta_deg))
        # Check monotonic: perturbed_x < original x, perturbed_y < original y
        if (perturbed_x < self.x_img) and (perturbed_y < self.y_img):
            return (True, f"Increasing |f| from {baseline_f:.1f} to {perturbed_f:.1f} makes x and y more negative: "
                          f"x: {self.x_img:.4f} -> {perturbed_x:.4f}, y: {self.y_img:.4f} -> {perturbed_y:.4f}")
        else:
            return (False, f"Monotonicity violation: |f| increased but x or y did not become more negative. "
                          f"x: {self.x_img:.4f} -> {perturbed_x:.4f}, y: {self.y_img:.4f} -> {perturbed_y:.4f}")

    # ---------- Weak answer sanity (at most one) ----------
    def test_answer_sanity_dist_axis_range(self):
        """Weak sanity: dist_axis should be between 0.1 and 10 cm (order of magnitude)."""
        if self.dist_axis is None:
            return (False, "SKIP: dist_axis not available")
        if not (0.1 < self.dist_axis < 10.0):
            return (False, f"dist_axis = {self.dist_axis} out of plausible range [0.1, 10] cm")
        return (True, "dist_axis within plausible range")