import math

class PhysicsChecker:
    """
    Checks a solution for the Coriolis lateral displacement problem.
    """

    def __init__(self, solution_instance):
        # Extract inputs and results safely using None defaults
        self.h = getattr(solution_instance, 'h', None)
        if self.h is None and hasattr(solution_instance, 'results'):
            self.h = solution_instance.results.get('h', None)

        self.g = getattr(solution_instance, 'g', None)
        if self.g is None and hasattr(solution_instance, 'results'):
            self.g = solution_instance.results.get('g', None)

        self.omega = getattr(solution_instance, 'omega', None)
        if self.omega is None and hasattr(solution_instance, 'results'):
            self.omega = solution_instance.results.get('omega', None)

        self.lambda_deg = getattr(solution_instance, 'lambda_deg', None)
        if self.lambda_deg is None and hasattr(solution_instance, 'results'):
            self.lambda_deg = solution_instance.results.get('lambda_deg', None)

        # Expected result key 'x' (eastward displacement)
        self.x = None
        if hasattr(solution_instance, 'results'):
            self.x = solution_instance.results.get('x', None)
        if self.x is None:
            self.x = getattr(solution_instance, 'x', None)

        # Optional: explanation steps for modeling checks
        self.explanation_steps = None
        if hasattr(solution_instance, 'explanation_steps'):
            self.explanation_steps = solution_instance.explanation_steps

        # Derived fall time (can be computed from h,g)
        self.T = None
        if self.h is not None and self.g is not None and self.g != 0:
            self.T = math.sqrt(2.0 * self.h / self.g)

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

    def _is_missing(self, value, label):
        """Helper to return a skip message if value is None."""
        if value is None:
            return (False, f"SKIP: {label} not available")
        return None  # not missing

    # ------------------ Interface Tests ------------------
    def test_interface_required_givens(self):
        """Check that required input parameters are present and numeric."""
        for label, val in [("h", self.h), ("g", self.g), ("omega", self.omega), ("lambda_deg", self.lambda_deg)]:
            if val is None:
                return (False, f"Missing required given: {label}")
            if not isinstance(val, (int, float)):
                return (False, f"Given {label} is not numeric")
        return (True, "All required givens present and numeric")

    def test_interface_result_x(self):
        """Check that expected result x is present and numeric."""
        skip = self._is_missing(self.x, "x")
        if skip: return skip
        if not isinstance(self.x, (int, float)):
            return (False, "x is not numeric")
        return (True, "Result x present and numeric")

    # ------------------ Invariant Tests (use math.isclose) ------------------
    def test_invariant_lateral_displacement_formula(self):
        """Check that x matches the theoretical formula x = (1/3)*omega*cos(latitude)*g*T^3."""
        skip = self._is_missing(self.x, "x")
        if skip: return skip
        # Ensure all inputs are present
        for label, val in [("h", self.h), ("g", self.g), ("omega", self.omega), ("lambda_deg", self.lambda_deg)]:
            if val is None:
                return (False, f"Cannot compute expected x: missing {label}")
        if self.g == 0:
            return (False, "g is zero, cannot compute fall time")
        cos_lambda = math.cos(math.radians(self.lambda_deg))
        T = math.sqrt(2.0 * self.h / self.g)
        expected_x = (1.0/3.0) * self.omega * cos_lambda * self.g * (T ** 3)
        passed = math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2)
        if passed:
            return (True, f"x = {self.x} matches expected {expected_x:.6f}")
        else:
            return (False, f"x = {self.x} but expected ~{expected_x:.6f}")

    def test_invariant_coriolis_sign_northern_hemisphere(self):
        """In northern hemisphere (lambda_deg > 0), eastward displacement should be positive."""
        skip = self._is_missing(self.x, "x")
        if skip: return skip
        if self.lambda_deg is None:
            return (False, "SKIP: lambda_deg not available")
        if self.lambda_deg > 0 and self.x < 0:
            return (False, f"Expected positive x in northern hemisphere but got {self.x}")
        if self.lambda_deg < 0 and self.x > 0:
            return (False, f"Expected negative x in southern hemisphere but got {self.x}")
        return (True, "Sign of x consistent with hemisphere")

    def test_invariant_fall_time_consistency(self):
        """Check that T computed from h and g is positive and finite."""
        if self.h is None or self.g is None:
            return (False, "SKIP: h or g missing")
        if self.g <= 0:
            return (False, "g must be positive")
        T = math.sqrt(2.0 * self.h / self.g)
        if T <= 0 or math.isinf(T):
            return (False, f"Fall time T is invalid: {T}")
        return (True, f"Fall time T = {T:.4f} s is positive finite")

    # ------------------ Modeling Quality Tests (use isclose) ------------------
    def test_modeling_explanation_steps_contain_integration(self):
        """Check that explanation_steps (if provided) contain key integration words."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        text = str(self.explanation_steps).lower()
        keywords = ["integrate", "integration", "twice", "coriolis", "acceleration"]
        found = [kw for kw in keywords if kw in text]
        if len(found) < 3:
            return (False, f"Few integration/coriolis keywords found in explanation_steps: {found}")
        return (True, f"Explanation contains integration-related keywords: {found}")

    def test_modeling_consistency_between_x_and_fall_time(self):
        """
        Check that the formula x = (1/3)*omega*cos(lambda)*g*T^3 is internally consistent:
        given x, derive T_back = (3x/(omega*cos(lambda)*g))^(1/3) and compare to sqrt(2h/g).
        This tests that the solution respects the relationship between x and the fall time.
        """
        skip = self._is_missing(self.x, "x")
        if skip: return skip
        for label, val in [("h", self.h), ("g", self.g), ("omega", self.omega), ("lambda_deg", self.lambda_deg)]:
            if val is None:
                return (False, f"Cannot compute: missing {label}")
        if self.g == 0:
            return (False, "g is zero")
        cos_lambda = math.cos(math.radians(self.lambda_deg))
        # Recover T from x
        T_from_x = (3.0 * self.x / (self.omega * cos_lambda * self.g)) ** (1.0/3.0)
        # Expected fall time
        T_expected = math.sqrt(2.0 * self.h / self.g)
        if math.isclose(T_from_x, T_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T from x = {T_from_x:.4f} s matches theoretical T = {T_expected:.4f} s")
        else:
            return (False, f"T from x = {T_from_x:.4f} s but expected T = {T_expected:.4f} s")

    # ------------------ Monotonicity Tests ------------------
    def test_monotonicity_lateral_displacement_increases_with_height(self):
        """
        Perturb h by +10% and verify that computed x increases (x ∝ h^(3/2)).
        """
        skip = self._is_missing(self.x, "x")
        if skip: return skip
        if None in (self.g, self.omega, self.lambda_deg):
            return (False, "Cannot compute monotonicity: missing g, omega or lambda_deg")
        if self.g == 0:
            return (False, "g is zero")
        cos_lambda = math.cos(math.radians(self.lambda_deg))
        h_baseline = self.h
        T_base = math.sqrt(2.0 * h_baseline / self.g)
        x_base = (1.0/3.0) * self.omega * cos_lambda * self.g * (T_base ** 3)

        h_perturbed = h_baseline * 1.10
        T_pert = math.sqrt(2.0 * h_perturbed / self.g)
        x_pert = (1.0/3.0) * self.omega * cos_lambda * self.g * (T_pert ** 3)

        if not (x_pert > x_base):
            return (False, f"Monotonicity failed: x_base={x_base:.6f}, x_perturbed={x_pert:.6f} (should be larger)")
        return (True, f"Monotonicity OK: x increases with h (baseline {x_base:.6f} -> perturbed {x_pert:.6f})")

    def test_monotonicity_lateral_displacement_decreases_with_latitude(self):
        """
        Perturb latitude upward by +10 degrees and verify that x decreases (cos decreases).
        """
        skip = self._is_missing(self.x, "x")
        if skip: return skip
        if None in (self.h, self.g, self.omega, self.lambda_deg):
            return (False, "Cannot compute monotonicity: missing inputs")
        if self.g == 0:
            return (False, "g is zero")
        cos_base = math.cos(math.radians(self.lambda_deg))
        T_base = math.sqrt(2.0 * self.h / self.g)
        x_base = (1.0/3.0) * self.omega * cos_base * self.g * (T_base ** 3)

        # Perturb latitude: increase by 10 degrees, cap at 90
        lambda_pert = min(self.lambda_deg + 10, 90)
        cos_pert = math.cos(math.radians(lambda_pert))
        x_pert = (1.0/3.0) * self.omega * cos_pert * self.g * (T_base ** 3)

        if not (x_pert < x_base):
            return (False, f"Monotonicity failed: x_base={x_base:.6f}, x_perturbed={x_pert:.6f} (should be smaller)")
        return (True, f"Monotonicity OK: x decreases as latitude increases (baseline {x_base:.6f} -> perturbed {x_pert:.6f})")

    # ------------------ Answer Sanity Test (weak; no math.isclose, no derived expected values) ------------------
    def test_answer_sanity_sign_and_range(self):
        """Weak sanity: x should be positive and not larger than 0.5 m for typical Earth parameters."""
        skip = self._is_missing(self.x, "x")
        if skip: return skip
        if math.isnan(self.x) or math.isinf(self.x):
            return (False, "x is NaN or infinite")
        if self.x <= 0:
            return (False, f"x = {self.x} is not positive (should be eastward)")
        if self.x > 0.5:
            return (False, f"x = {self.x} is suspiciously large (expected ~0.017 m)")
        return (True, f"x = {self.x} m is positive and plausible (< 0.5 m)")

    # ------------------ Additional Invariant: Consistency with omega=0 limit ------------------
    def test_invariant_no_coriolis_limit(self):
        """If omega=0, the formula gives x=0. Verify property if applicable."""
        skip = self._is_missing(self.omega, "omega")
        if skip: return skip
        if self.omega == 0:
            skip2 = self._is_missing(self.x, "x")
            if skip2: return skip2
            if not math.isclose(self.x, 0, abs_tol=1e-10):
                return (False, f"With omega=0, x should be 0 but got {self.x}")
            return (True, "x=0 when omega=0, as expected")
        else:
            return (True, "SKIP: omega nonzero, property not applicable")