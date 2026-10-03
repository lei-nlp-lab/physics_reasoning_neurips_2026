import math
import sympy as sp

class PhysicsChecker:
    """Auditor for spring pendulum solution (small oscillations)."""

    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.m = getattr(solution_instance, 'm', None)
        self.k = getattr(solution_instance, 'k', None)
        self.l = getattr(solution_instance, 'l', None)
        self.g = getattr(solution_instance, 'g', None)

        # Extract results from results dict or attribute
        results = getattr(solution_instance, 'results', {})
        self.r = results.get('r', getattr(solution_instance, 'r', None))
        self.theta = results.get('theta', getattr(solution_instance, 'theta', None))

        # Pre‑compute expected constants only if all givens present
        self.r0 = None
        self.omega_r = None
        self.omega_theta = None
        if all(v is not None for v in [self.m, self.k, self.l, self.g]):
            self.r0 = self.l + self.m * self.g / self.k
            self.omega_r = math.sqrt(self.k / self.m)
            self.omega_theta = math.sqrt(self.g / self.r0)

    # ---------- Interface tests ----------

    def test_interface_givens_present(self):
        """All required givens must be present and numeric."""
        for name, val in [('m', self.m), ('k', self.k), ('l', self.l), ('g', self.g)]:
            if val is None:
                return (False, f"Missing given: {name}")
            if not isinstance(val, (int, float)):
                return (False, f"Given {name} is not numeric: {type(val)}")
        return (True, "All givens present and numeric")

    def test_interface_result_r_exists(self):
        """Result 'r' must be present."""
        if self.r is None:
            return (False, "Result 'r' is missing")
        return (True, "Result 'r' exists")

    def test_interface_result_theta_exists(self):
        """Result 'theta' must be present."""
        if self.theta is None:
            return (False, "Result 'theta' is missing")
        return (True, "Result 'theta' exists")

    # ---------- Invariant tests (use math.isclose) ----------

    def test_invariant_results_are_strings(self):
        """Both results should be strings (LaTeX expressions)."""
        if not isinstance(self.r, str):
            return (False, f"Result 'r' is not a string: {type(self.r)}")
        if not isinstance(self.theta, str):
            return (False, f"Result 'theta' is not a string: {type(self.theta)}")
        return (True, "Both results are strings")

    def test_invariant_radial_expression_has_cos(self):
        """Radial expression must contain 'cos'."""
        if self.r is None:
            return (False, "SKIP: r not available")
        if 'cos' not in self.r:
            return (False, "Radial expression missing 'cos'")
        return (True, "Radial expression contains 'cos'")

    def test_invariant_angular_expression_has_cos(self):
        """Angular expression must contain 'cos'."""
        if self.theta is None:
            return (False, "SKIP: theta not available")
        if 'cos' not in self.theta:
            return (False, "Angular expression missing 'cos'")
        return (True, "Angular expression contains 'cos'")

    def test_invariant_radial_expression_has_equilibrium(self):
        """Radial expression should contain the equilibrium term l + mg/k."""
        if self.r is None:
            return (False, "SKIP: r not available")
        # Check for 'l' and 'mg/k' or similar patterns
        has_l = 'l' in self.r
        has_mg_over_k = ('mg' in self.r) and ('k' in self.r) and ('/k' in self.r or 'frac{mg}{k}' in self.r)
        if not has_l or not has_mg_over_k:
            return (False, "Radial expression does not contain l + mg/k pattern")
        return (True, "Radial expression contains equilibrium term")

    def test_invariant_radial_frequency_pattern(self):
        """Radial expression should contain sqrt(k/m) or sqrt{\frac{k}{m}}."""
        if self.r is None:
            return (False, "SKIP: r not available")
        patterns = ['sqrt{k}{m}', 'sqrt{\\frac{k}{m}}', 'sqrt{k/m}']
        if not any(p in self.r for p in patterns):
            return (False, "Radial expression does not contain expected square‑root frequency")
        return (True, "Radial frequency pattern found")

    def test_invariant_angular_frequency_pattern(self):
        """Angular expression should contain sqrt(kg/(kl+mg))."""
        if self.theta is None:
            return (False, "SKIP: theta not available")
        patterns = ['sqrt{kg}{kl+mg}', 'sqrt{kg}{kl + mg}', 'sqrt{\\frac{kg}{kl+mg}}']
        if not any(p in self.theta for p in patterns):
            return (False, "Angular expression does not contain expected frequency pattern")
        return (True, "Angular frequency pattern found")

    # ---------- Modeling / trace consistency ----------

    def test_modeling_radial_decoupled_from_theta(self):
        """Radial expression should not contain 'theta' or 'B'."""
        if self.r is None:
            return (False, "SKIP: r not available")
        if 'theta' in self.r or 'B' in self.r:
            return (False, "Radial expression contains angular variables")
        return (True, "Radial expression decoupled from theta")

    def test_modeling_angular_decoupled_from_radial(self):
        """Angular expression should not contain 'A' or 'r0'."""
        if self.theta is None:
            return (False, "SKIP: theta not available")
        if 'A' in self.theta or 'r0' in self.theta:
            return (False, "Angular expression contains radial variables")
        return (True, "Angular expression decoupled from radial")

    # ---------- Monotonicity tests (explicit perturbation) ----------

    def test_monotonicity_equilibrium_length_increases_with_mass(self):
        """Increasing mass m (keeping others constant) should increase r0 = l + mg/k."""
        if any(v is None for v in [self.m, self.k, self.l, self.g]):
            return (False, "SKIP: givens incomplete for monotonicity check")
        # Baseline
        r0_base = self.l + self.m * self.g / self.k
        # Perturb mass by +5%
        m_pert = self.m * 1.05
        r0_pert = self.l + m_pert * self.g / self.k
        if r0_pert <= r0_base:
            return (False, f"Equilibrium length did not increase: base={r0_base}, perturbed={r0_pert}")
        # Optionally check that the increase is approximately proportional
        expected_increase = (m_pert - self.m) * self.g / self.k
        actual_increase = r0_pert - r0_base
        if not math.isclose(actual_increase, expected_increase, rel_tol=1e-2, abs_tol=1e-6):
            return (False, f"Increase deviates: expected {expected_increase}, got {actual_increase}")
        return (True, "Equilibrium length increases with mass as expected")

    # ---------- Weak answer sanity (no math.isclose, no re‑derivation) ----------

    def test_answer_sanity_frequencies_positive(self):
        """Derived frequencies from givens must be positive finite numbers."""
        if self.omega_r is None or self.omega_theta is None:
            return (False, "SKIP: cannot compute frequencies (m, k, l, g incomplete)")
        # Just check magnitude and finiteness
        if not (math.isfinite(self.omega_r) and self.omega_r > 0):
            return (False, f"Radial frequency not positive/finite: {self.omega_r}")
        if not (math.isfinite(self.omega_theta) and self.omega_theta > 0):
            return (False, f"Angular frequency not positive/finite: {self.omega_theta}")
        return (True, "Both frequencies are positive and finite")

    # ---------- Runner ----------

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