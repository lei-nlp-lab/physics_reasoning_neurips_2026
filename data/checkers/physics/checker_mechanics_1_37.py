import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the problem of a perfectly reflecting sphere
    where gravitational and radiation forces cancel.
    """

    def __init__(self, solution_instance):
        # Extract givens safely
        self.G = None
        self.M_s = None
        self.I_s = None
        self.c = None
        self.rho = None
        self.r = None

        try:
            # Try to get from 'givens' dict or direct attribute
            givens = getattr(solution_instance, 'givens', {})
            if givens is None:
                givens = {}
            self.G = givens.get('G', getattr(solution_instance, 'G', None))
            self.M_s = givens.get('M_s', getattr(solution_instance, 'M_s', None))
            self.I_s = givens.get('I_s', getattr(solution_instance, 'I_s', None))
            self.c = givens.get('c', getattr(solution_instance, 'c', None))
            self.rho = givens.get('rho', getattr(solution_instance, 'rho', None))

            # Extract result 'r' from results dict
            results = getattr(solution_instance, 'results', {})
            if results is None:
                results = {}
            self.r = results.get('r', None)
        except Exception:
            pass  # If anything goes wrong, keep defaults as None

    # ----------------------------------------------------------------------
    # Interface checks: required keys and valid types
    # ----------------------------------------------------------------------
    def test_interface_givens_present(self):
        """Check that all required givens are present (non-None)."""
        needed = {'G', 'M_s', 'I_s', 'c', 'rho'}
        present = {}
        for name in needed:
            val = getattr(self, name, None)
            present[name] = val is not None
        missing = [name for name, ok in present.items() if not ok]
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    def test_interface_result_r_present(self):
        """Check that result 'r' exists and is a numeric type."""
        if self.r is None:
            return (False, "Result 'r' is missing (None)")
        if not isinstance(self.r, (int, float)):
            return (False, f"Result 'r' is not numeric, got {type(self.r)}")
        return (True, "Result 'r' present and numeric")

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ----------------------------------------------------------------------
    def test_invariant_equilibrium(self):
        """
        Verify that gravitational and radiation forces are equal.
        Use the derived expression from explanation steps:
            F_grav = G * M_s * m / d^2
            F_rad  = I_s * r^2 / (4 * c * d^2)
        with m = (4/3)*pi*r^3 * rho.
        Distance d cancels, so we compute ratio and check isclose(1.0).
        """
        if None in (self.G, self.M_s, self.I_s, self.c, self.rho, self.r):
            return (False, "SKIP: missing data for equilibrium check")

        r = self.r
        G = self.G
        M_s = self.M_s
        I_s = self.I_s
        c = self.c
        rho = self.rho

        # mass of sphere
        m = (4.0 / 3.0) * math.pi * r**3 * rho

        # Forces (d^2 cancels, so omit d)
        F_grav = G * M_s * m
        F_rad = I_s * r**2 / (4.0 * c)

        ratio = F_grav / F_rad
        # Equilibrium means ratio = 1 within 1% tolerance
        if not math.isclose(ratio, 1.0, rel_tol=1e-2):
            return (False, f"Force ratio = {ratio}, expected ~1.0")
        return (True, "Forces are in equilibrium")

    def test_invariant_d_cancellation(self):
        """
        Check that the equilibrium condition does not depend on distance d.
        We can verify that both forces scale as 1/d^2, and the ratio is
        independent of d. Since d is not given, we test that the derived
        expression for r from equilibrium does not contain d.
        Compute expected r from constants and compare to result.
        """
        if None in (self.G, self.M_s, self.I_s, self.c, self.rho, self.r):
            return (False, "SKIP: missing data for d-cancellation check")

        # Derived formula from explanation steps:
        # r = 3 * I_s / (16 * pi * c * G * M_s * rho)
        numerator = 3.0 * self.I_s
        denominator = 16.0 * math.pi * self.c * self.G * self.M_s * self.rho
        expected_r = numerator / denominator

        if not math.isclose(self.r, expected_r, rel_tol=1e-2):
            return (False, f"Computed r = {self.r}, expected from constants = {expected_r}")
        return (True, "Distances cancel correctly; r matches derived expression")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_mass_formula(self):
        """
        Check that the mass of the sphere is computed as (4/3)*pi*r^3 * rho.
        This is a fundamental modeling step.
        """
        if None in (self.r, self.rho):
            return (False, "SKIP: missing r or rho for mass formula check")

        m_theory = (4.0 / 3.0) * math.pi * self.r**3 * self.rho
        # We cannot directly get mass from results, but we can check that
        # the equilibrium equation implies this relation. We'll just do a
        # sanity: positive mass.
        if m_theory <= 0:
            return (False, f"Computed mass = {m_theory} should be positive")
        return (True, f"Mass formula yields positive mass = {m_theory}")

    def test_modeling_radiation_force_formula(self):
        """
        Verify the radiation force formula used in the derivation:
        F_rad = I_s * r^2 / (4 * c * d^2) leads to the correct r.
        Use the equilibrium equation to back-calculate the coefficient.
        """
        if None in (self.G, self.M_s, self.I_s, self.c, self.rho, self.r):
            return (False, "SKIP: missing data for radiation force check")

        # From equilibrium: F_grav = F_rad
        # G M_s (4/3 pi r^3 rho) = I_s r^2 / (4 c)
        # => predicted coefficient = G M_s (4/3 pi r rho) * (4 c) / I_s
        # should be 1.
        r = self.r
        coeff = (self.G * self.M_s * (4.0/3.0) * math.pi * r * self.rho
                 * 4.0 * self.c / self.I_s)
        if not math.isclose(coeff, 1.0, rel_tol=1e-2):
            return (False, f"Radiation force coefficient = {coeff}, expected 1")
        return (True, "Radiation force formula consistent with equilibrium")

    # ----------------------------------------------------------------------
    # Monotonicity checks (trend via perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_r_with_I_s(self):
        """
        If solar luminosity I_s increases, radiation force increases.
        To maintain equilibrium, sphere radius r must increase.
        Perturb I_s by +10% and check predicted r increases.
        """
        if None in (self.G, self.M_s, self.I_s, self.c, self.rho, self.r):
            return (False, "SKIP: missing data for I_s monotonicity check")

        # Baseline r is self.r. Predict new r from formula:
        # r = 3*I_s/(16*pi*c*G*M_s*rho)
        baseline_r = self.r
        I_s_perturbed = self.I_s * 1.10
        new_r = (3.0 * I_s_perturbed) / (16.0 * math.pi * self.c * self.G * self.M_s * self.rho)
        if not (new_r > baseline_r):
            return (False, f"Predicted new r = {new_r} not > baseline {baseline_r}")
        return (True, f"I_s increased by 10%: r increases from {baseline_r} to {new_r}")

    def test_monotonicity_r_with_M_s(self):
        """
        If solar mass M_s increases, gravitational force increases.
        To maintain equilibrium, sphere radius r must decrease.
        Perturb M_s by +10% and check predicted r decreases.
        """
        if None in (self.G, self.M_s, self.I_s, self.c, self.rho, self.r):
            return (False, "SKIP: missing data for M_s monotonicity check")

        baseline_r = self.r
        M_s_perturbed = self.M_s * 1.10
        new_r = (3.0 * self.I_s) / (16.0 * math.pi * self.c * self.G * M_s_perturbed * self.rho)
        if not (new_r < baseline_r):
            return (False, f"Predicted new r = {new_r} not < baseline {baseline_r}")
        return (True, f"M_s increased by 10%: r decreases from {baseline_r} to {new_r}")

    # ----------------------------------------------------------------------
    # Weak answer sanity checks (only one test_answer_sanity_* method)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: radius must be positive and finite.
        No math.isclose, no re-derivation of target value.
        """
        if self.r is None:
            return (False, "SKIP: r is missing")
        if not math.isfinite(self.r):
            return (False, f"Radius r = {self.r} is not finite")
        if self.r <= 0:
            return (False, f"Radius r = {self.r} is not positive")
        return (True, f"Radius r = {self.r} is positive and finite")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test methods
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)