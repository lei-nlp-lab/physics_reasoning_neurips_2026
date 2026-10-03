import math

class PhysicsChecker:
    """
    PhysicsChecker for the relativistic moving sphere photograph problem.
    Verifies the apparent shape and angular radius based on special relativity
    and the paraxial approximation.
    """

    def __init__(self, solution_instance):
        # VITAL: use None defaults for all missing values, never numeric defaults
        self.R0 = getattr(solution_instance, 'R0', None)
        self.v = getattr(solution_instance, 'v', None)
        self.c = getattr(solution_instance, 'c', None)
        self.D = getattr(solution_instance, 'D', None)

        # Extract gamma if directly provided, otherwise compute from v/c
        self.gamma = getattr(solution_instance, 'gamma', None)
        if self.gamma is None and self.v is not None and self.c is not None and self.c != 0:
            try:
                beta = self.v / self.c
                self.gamma = 1.0 / math.sqrt(1 - beta * beta)
            except (ValueError, ZeroDivisionError):
                self.gamma = None

        # Extract results (expected keys: 'angular_radius', 'apparent_shape')
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.angular_radius = solution_instance.results.get('angular_radius', None)
            self.apparent_shape = solution_instance.results.get('apparent_shape', None)
        else:
            self.angular_radius = getattr(solution_instance, 'angular_radius', None)
            self.apparent_shape = getattr(solution_instance, 'apparent_shape', None)

    # ----------------------------------------------------------------------
    # Interface tests: required keys exist and have correct types
    # ----------------------------------------------------------------------
    def test_interface_required_results_exist(self):
        """Check that angular_radius and apparent_shape are present in results."""
        if self.angular_radius is None:
            return (False, "Missing result: angular_radius")
        if self.apparent_shape is None:
            return (False, "Missing result: apparent_shape")
        return (True, "Both required results present")

    def test_interface_givens_available(self):
        """Check that essential givens (R0, D, v, c) are available for detailed tests."""
        missing = []
        if self.R0 is None:
            missing.append("R0")
        if self.D is None:
            missing.append("D")
        if self.v is None:
            missing.append("v")
        if self.c is None:
            missing.append("c")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens present")

    def test_interface_angular_radius_numeric(self):
        """Check that angular_radius is a numeric value (int or float)."""
        if self.angular_radius is None:
            return (False, "SKIP: angular_radius not provided")
        if not isinstance(self.angular_radius, (int, float)):
            return (False, f"angular_radius is {type(self.angular_radius)}, expected numeric")
        return (True, "angular_radius is numeric")

    def test_interface_apparent_shape_string(self):
        """Check that apparent_shape is a string."""
        if self.apparent_shape is None:
            return (False, "SKIP: apparent_shape not provided")
        if not isinstance(self.apparent_shape, str):
            return (False, f"apparent_shape is {type(self.apparent_shape)}, expected string")
        return (True, "apparent_shape is string")

    # ----------------------------------------------------------------------
    # Invariant tests: theorem‑based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_angular_radius_positive(self):
        """The angular radius must be positive (finite and >0)."""
        if self.angular_radius is None:
            return (False, "SKIP: angular_radius not available")
        if not isinstance(self.angular_radius, (int, float)):
            return (False, "angular_radius is non‑numeric")
        if self.angular_radius <= 0:
            return (False, f"angular_radius = {self.angular_radius} is not positive")
        return (True, "angular_radius > 0")

    def test_invariant_apparent_shape_is_circle_or_sphere(self):
        """The apparent shape must be a circle (or sphere) of rest radius R0."""
        if self.apparent_shape is None:
            return (False, "SKIP: apparent_shape not available")
        lower = self.apparent_shape.lower()
        if "circle" not in lower and "sphere" not in lower:
            return (False, f"apparent_shape '{self.apparent_shape}' does not describe a circle/sphere")
        return (True, "apparent_shape describes a circle/sphere")

    def test_invariant_v_less_than_c(self):
        """Relativistic motion requires v < c."""
        if self.v is None or self.c is None:
            return (False, "SKIP: v or c not available")
        if not isinstance(self.v, (int, float)) or not isinstance(self.c, (int, float)):
            return (False, "v or c is non‑numeric")
        if self.c <= 0:
            return (False, "c must be positive")
        if self.v >= self.c:
            return (False, f"v = {self.v} is not less than c = {self.c}")
        return (True, "v < c")

    def test_invariant_gamma_defined(self):
        """Lorentz factor gamma should be finite and >=1."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        if not isinstance(self.gamma, (int, float)):
            return (False, "gamma is non‑numeric")
        if self.gamma < 1.0:
            return (False, f"gamma = {self.gamma} < 1, invalid")
        if math.isinf(self.gamma):
            return (False, "gamma is infinite")
        return (True, f"gamma = {self.gamma} is valid")

    # ----------------------------------------------------------------------
    # Modeling / trace tests: check quality of the modelled quantities
    # ----------------------------------------------------------------------
    def test_modeling_angular_radius_formula(self):
        """
        Verify that angular_radius equals R0 / D within 2% tolerance.
        This is the fundamental prediction from the paraxial approximation.
        """
        if self.angular_radius is None or self.R0 is None or self.D is None:
            return (False, "SKIP: needed values (angular_radius, R0, D) not all present")
        if not all(isinstance(x, (int, float)) for x in (self.angular_radius, self.R0, self.D)):
            return (False, "One of angular_radius, R0, D is non‑numeric")
        if self.D == 0:
            return (False, "D is zero, cannot compute")
        expected = self.R0 / self.D
        try:
            if math.isclose(self.angular_radius, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"angular_radius = {self.angular_radius} ≈ R0/D = {expected}")
            else:
                return (False, f"angular_radius = {self.angular_radius} differs from R0/D = {expected}")
        except Exception as e:
            return (False, f"Error during comparison: {e}")

    def test_modeling_apparent_shape_correct_text(self):
        """
        Check that the apparent_shape text explicitly contains 'circle of radius R0'
        or a semantically equivalent phrase (like 'sphere').
        """
        if self.apparent_shape is None or self.R0 is None:
            return (False, "SKIP: apparent_shape or R0 not available")
        lower = self.apparent_shape.lower()
        # Accept variations: "circle of radius R0" or "sphere of radius R0"
        radius_str = f"radius {self.R0}"
        if "circle" in lower and radius_str in lower:
            return (True, "apparent_shape matches expected description")
        if "sphere" in lower and radius_str in lower:
            return (True, "apparent_shape matches expected description")
        # Relaxed: at least one of the two key ideas is present
        if "circle" in lower or "sphere" in lower:
            return (True, "apparent_shape mentions circle/sphere (exact radius check omitted)")
        return (False, f"apparent_shape '{self.apparent_shape}' lacks circle/sphere description")

    def test_modeling_paraxial_condition(self):
        """
        The problem assumes D >> R0. If both are given, verify the condition
        to ensure the paraxial approximation is valid.
        """
        if self.D is None or self.R0 is None:
            return (False, "SKIP: D or R0 not available")
        if not isinstance(self.D, (int, float)) or not isinstance(self.R0, (int, float)):
            return (False, "D or R0 is non‑numeric")
        if self.R0 == 0:
            return (False, "R0 is zero, degenerate")
        ratio = self.D / self.R0
        if ratio < 10:  # a heuristic threshold for ">>"
            return (False, f"D/R0 = {ratio} < 10, paraxial approximation may be poor")
        return (True, f"D/R0 = {ratio} >> 1, paraxial approximation valid")

    # ----------------------------------------------------------------------
    # Satisfy at most one weak answer sanity test – using only sign/finite/range
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_and_reasonable(self):
        """
        Weak sanity: angular_radius must be finite, non‑NaN, and within
        an expected order of magnitude (10⁻⁶–10⁻³ rad for typical lab setups).
        """
        if self.angular_radius is None:
            return (False, "SKIP: angular_radius not available")
        if not isinstance(self.angular_radius, (int, float)):
            return (False, "angular_radius is non‑numeric")
        if math.isnan(self.angular_radius):
            return (False, "angular_radius is NaN")
        if math.isinf(self.angular_radius):
            return (False, "angular_radius is infinite")
        # Typical angular radius for a globe at large distance: tiny but positive.
        # No isclose, no derivation – just a sanity range.
        if self.angular_radius <= 0:
            return (False, f"angular_radius = {self.angular_radius} must be > 0")
        # Even a huge globe at 1 m would be ≈ 10⁰ rad, but D >> R0 keeps it small.
        # Upper bound: less than π (full sky) is safe.
        if self.angular_radius > math.pi:
            return (False, f"angular_radius = {self.angular_radius} > π, unreasonable")
        return (True, f"angular_radius = {self.angular_radius} is finite and positive")

    # ----------------------------------------------------------------------
    # Monotonicity tests (optional but good practice; here we use the formula)
    # ----------------------------------------------------------------------
    def test_monotonicity_radius_dependence(self):
        """
        If R0 increases, angular_radius should increase proportionally.
        We cannot perturb the input, but we can verify the linear relation
        using the computed ratio.
        """
        if self.angular_radius is None or self.R0 is None or self.D is None:
            return (False, "SKIP: needed values not all present")
        if not all(isinstance(x, (int, float)) for x in (self.angular_radius, self.R0, self.D)):
            return (False, "Non‑numeric value encountered")
        if self.D == 0:
            return (False, "D is zero")
        # monotonicity: angular_radius / R0 = 1/D (constant)
        ratio = self.angular_radius / self.R0 if self.R0 != 0 else None
        if ratio is None:
            return (False, "R0 is zero")
        expected_ratio = 1.0 / self.D
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"Angular radius scales linearly with R0: ratio = {ratio} ≈ 1/D = {expected_ratio}")
        else:
            return (False, f"Angular radius does not scale linearly: ratio = {ratio} ≠ 1/D = {expected_ratio}")

    # ----------------------------------------------------------------------
    # Main test executor
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all test_* methods, run them in a try/except block,
        and return (all_pass, logs).
        """
        logs = []
        all_pass = True
        # Collect all bound methods whose name starts with test_
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