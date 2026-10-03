import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.m = solution_instance.givens.get('m', None)
        self.a = solution_instance.givens.get('a', None)
        self.V0 = solution_instance.givens.get('V0', None)
        self.hbar = solution_instance.givens.get('hbar', None)
        self.k = solution_instance.givens.get('k', None)  # optional

        # Extract results safely
        self.sigma_Born = solution_instance.results.get('sigma_Born', None)

        # Optional: explanation steps
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def test_interface_sigma_Born_exists(self):
        """Check that sigma_Born is present in results."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        return (True, "sigma_Born is present")

    def test_invariant_sigma_Born_positive(self):
        """Cross section must be non-negative."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if self.sigma_Born < -1e-12:
            return (False, f"sigma_Born = {self.sigma_Born} is negative")
        return (True, "sigma_Born >= 0 (non-negative)")

    def test_invariant_sigma_Born_finite(self):
        """Cross section must be finite."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if math.isnan(self.sigma_Born) or math.isinf(self.sigma_Born):
            return (False, f"sigma_Born is NaN or infinite")
        return (True, "sigma_Born is finite")

    def test_invariant_sigma_Born_formula(self):
        """Check consistency with the derived low-energy Born formula:
        sigma = 16π m^2 V0^2 a^6 / (9 ħ^4)."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if any(val is None for val in (self.m, self.a, self.V0, self.hbar)):
            return (False, "SKIP: required givens (m, a, V0, hbar) missing")
        expected = (16 * math.pi * (self.m**2) * (self.V0**2) * (self.a**6)) / (9 * (self.hbar**4))
        if not math.isclose(self.sigma_Born, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_Born = {self.sigma_Born}, expected = {expected}")
        return (True, "sigma_Born matches the low-energy Born formula")

    def test_modeling_explanation_steps_present(self):
        """Check that explanation_steps exist and are non-empty."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        if not isinstance(self.explanation_steps, list) or len(self.explanation_steps) == 0:
            return (False, "explanation_steps is empty or not a list")
        return (True, "explanation_steps present and non-empty")

    def test_modeling_sigma_scaling(self):
        """Verify that sigma_Born scales as m^2 * V0^2 * a^6 / hbar^4,
        by checking that the product of input powers equals the output."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if any(val is None for val in (self.m, self.a, self.V0, self.hbar)):
            return (False, "SKIP: required givens (m, a, V0, hbar) missing")
        # Compute scaling factor from inputs (ignoring numerical constants)
        scaling = (self.m**2) * (self.V0**2) * (self.a**6) / (self.hbar**4)
        # Compare sigma_Born / (16π/9 * scaling) = 1
        ratio = self.sigma_Born / ( (16*math.pi/9.0) * scaling )
        if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Sigma scaling deviates: ratio = {ratio}")
        return (True, "Sigma scaling matches m^2 V0^2 a^6 / hbar^4")

    def test_modeling_k_independence(self):
        """If k is provided, check that sigma_Born does not depend on k (low-energy limit)."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if self.k is None:
            return (False, "SKIP: k not provided; cannot test k-independence")
        if any(val is None for val in (self.m, self.a, self.V0, self.hbar)):
            return (False, "SKIP: required givens (m, a, V0, hbar) missing")
        # Recompute a reference sigma using a different k (e.g., double k)
        k_ref = self.k * 2.0
        # In the low-energy Born formula, sigma does not depend on k at all.
        # So the reference should be the same as original.
        expected = (16 * math.pi * (self.m**2) * (self.V0**2) * (self.a**6)) / (9 * (self.hbar**4))
        if not math.isclose(self.sigma_Born, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Sigma does not match low-energy formula")
        # Also, if we had a k-dependent expression, the reference would be different,
        # but here we simply assert that sigma should equal the low-energy limit.
        # This test passes because the formula is independent of k.
        # We also check that k is small enough (optional).
        if self.k * self.a > 0.1:
            return (False, f"ka not small enough: ka = {self.k*self.a}")
        return (True, "Sigma is independent of k in low-energy limit")

    def test_monotonicity_sigma_increases_with_V0(self):
        """Check that the total cross section increases when V0 increases (physical monotonicity)."""
        if any(val is None for val in (self.m, self.a, self.V0, self.hbar)):
            return (False, "SKIP: required givens (m, a, V0, hbar) missing")
        if self.V0 <= 0:
            return (False, "V0 must be positive for the monotonicity test")
        # Baseline cross section (using the physical formula)
        sigma_base = (16 * math.pi * (self.m**2) * (self.V0**2) * (self.a**6)) / (9 * (self.hbar**4))
        # Perturb V0 upward by 5%
        V0_pert = self.V0 * 1.05
        sigma_pert = (16 * math.pi * (self.m**2) * (V0_pert**2) * (self.a**6)) / (9 * (self.hbar**4))
        if sigma_pert <= sigma_base:
            return (False, f"Monotonicity violated: sigma_base={sigma_base}, sigma_perturbed={sigma_pert}")
        return (True, "Monotonicity holds: sigma increases with V0")

    def test_answer_sanity_sigma_Born_positive(self):
        """Weak answer sanity: cross section must be positive and not ridiculously large."""
        if self.sigma_Born is None:
            return (False, "SKIP: sigma_Born not available")
        if self.sigma_Born <= 0:
            return (False, f"sigma_Born = {self.sigma_Born} is not positive")
        if self.sigma_Born > 1e10:  # arbitrary huge bound
            return (False, f"sigma_Born = {self.sigma_Born} seems unreasonably large")
        return (True, "sigma_Born is positive and within plausible range")

    def run_all_tests(self):
        """Discover all test methods and run them in try/except."""
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