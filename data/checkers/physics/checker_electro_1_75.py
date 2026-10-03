import math

class PhysicsChecker:
    """
    Audits a solution for the capacitance per unit length of an infinite wire
    near a grounded plane.

    Required givens:
        sigma : float – linear charge density (C/m)
        a     : float – wire radius (m)
        b     : float – distance from wire axis to plane (m), b >> a

    Required results:
        V_wire : float – potential of the wire relative to the plane (V)
        C      : float – capacitance per unit length (F/m)
    """

    def __init__(self, solution_instance):
        givens = getattr(solution_instance, 'givens', None) or {}
        results = getattr(solution_instance, 'results', None) or {}

        # Givens – safe defaults are None
        self.sigma = givens.get('sigma', None)
        self.a = givens.get('a', None)
        self.b = givens.get('b', None)

        # Results – safe defaults are None
        self.V_wire = results.get('V_wire', None)
        self.C = results.get('C', None)

    # ------------------------------------------------------------------
    # Interface checks – ensure required keys exist and are of valid type
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.V_wire is None:
            return (False, "V_wire not found in results")
        if self.C is None:
            return (False, "C not found in results")
        return (True, "All required results keys present")

    def test_interface_given_values(self):
        if self.sigma is None:
            return (False, "sigma not provided in givens")
        if self.a is None:
            return (False, "a (wire radius) not provided in givens")
        if self.b is None:
            return (False, "b (distance to plane) not provided in givens")
        if self.sigma <= 0:
            return (False, "sigma must be positive")
        if self.a <= 0:
            return (False, "a must be positive")
        if self.b <= 0:
            return (False, "b must be positive")
        if self.b <= self.a:
            return (False, "b must be > a (b >> a)")
        return (True, "Given values are positive and satisfy b > a")

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_potential_expression(self):
        """
        V_wire should equal (sigma/(2π)) * ln(2b/a) from method of images.
        """
        if any(v is None for v in [self.sigma, self.a, self.b, self.V_wire]):
            return (False, "Missing data: sigma, a, b, or V_wire")
        expected_V = (self.sigma / (2 * math.pi)) * math.log(2 * self.b / self.a)
        if math.isclose(self.V_wire, expected_V, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V_wire ≈ expected {expected_V:.4f}")
        else:
            return (False,
                    f"V_wire = {self.V_wire:.4f} is not close to "
                    f"expected {expected_V:.4f}")

    def test_invariant_capacitance_definition(self):
        """
        C must equal sigma / V_wire by definition of capacitance per unit length.
        """
        if any(v is None for v in [self.sigma, self.V_wire, self.C]):
            return (False, "Missing data: sigma, V_wire, or C")
        if math.isclose(self.V_wire, 0.0):
            return (False, "V_wire is zero, cannot compute C")
        expected_C = self.sigma / self.V_wire
        if math.isclose(self.C, expected_C, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C ≈ sigma / V_wire = {expected_C:.4e}")
        else:
            return (False,
                    f"C = {self.C:.4e} is not close to "
                    f"sigma/V_wire = {expected_C:.4e}")

    # ------------------------------------------------------------------
    # Modeling quality checks (consistency with explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_thin_wire_approximation(self):
        """
        The derivation uses b >> a.  Flag if the ratio is not large.
        """
        if self.a is None or self.b is None:
            return (False, "Missing a or b")
        ratio = self.b / self.a
        if ratio < 10:   # arbitrary threshold; strictly b >> a implies large ratio
            return (False, f"b/a = {ratio:.2f} – likely not large enough for b>>a")
        return (True, f"b/a = {ratio:.2f} – satisfies b >> a")

    def test_modeling_potential_difference_origin(self):
        """
        Verify that the potential V_wire is consistent with the expression
        σ/(2π) * ln((2b - a)/a) ≈ σ/(2π) * ln(2b/a).
        """
        if any(v is None for v in [self.sigma, self.a, self.b, self.V_wire]):
            return (False, "Missing data for this check")
        # exact distance from wire surface to image wire is 2b - a
        exact_V = (self.sigma / (2 * math.pi)) * math.log((2 * self.b - self.a) / self.a)
        approx_V = (self.sigma / (2 * math.pi)) * math.log(2 * self.b / self.a)
        if not math.isclose(self.V_wire, exact_V, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"V_wire does not match exact formula {exact_V:.4f}")
        # The approximation should be close to the exact value when b >> a
        if not math.isclose(exact_V, approx_V, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Exact V ({exact_V:.4f}) and approx V ({approx_V:.4f}) "
                           "diverge too much")
        return (True, "V_wire consistent with exact and approximate formulas")

    # ------------------------------------------------------------------
    # Monotonicity checks (trend tests via input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_capacitance_vs_distance(self):
        """
        Capacitance per unit length C must decrease when b increases.
        (Confirmed by the formula C = 2π / ln(2b/a).)
        """
        if any(v is None for v in [self.a, self.b, self.C]):
            return (False, "Missing a, b, or C for monotonicity test")
        # baseline C from result
        C0 = self.C
        # perturb b upward by 5%
        b1 = self.b * 1.05
        C1 = 2 * math.pi / math.log(2 * b1 / self.a)
        if not (C1 < C0):
            return (False, f"Monotonicity violated: C increases when b increases "
                           f"(C0={C0:.4e}, C1={C1:.4e})")
        return (True, f"C decreases as b increases (C0={C0:.4e}, C1={C1:.4e})")

    # ------------------------------------------------------------------
    # Weak answer‑sanity check (sign, finiteness only – no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        if self.C is None:
            return (False, "C not available")
        if not math.isfinite(self.C):
            return (False, "C is not finite")
        if self.C <= 0:
            return (False, "C must be positive")
        return (True, "C is positive and finite")

    # ------------------------------------------------------------------
    # Test runner – discovers all test_* methods and executes them
    # ------------------------------------------------------------------
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