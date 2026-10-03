import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        self.beta = None
        self.epsilon1 = None
        self.epsilon2 = None
        self.theta_I = None
        self.n1 = None
        self.n2 = None

        if hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            self.beta = getattr(givens, 'beta', None)
            self.epsilon1 = getattr(givens, 'epsilon1', None)
            self.epsilon2 = getattr(givens, 'epsilon2', None)
            self.theta_I = getattr(givens, 'theta_I', None)
            self.n1 = getattr(givens, 'n1', None)
            self.n2 = getattr(givens, 'n2', None)

        # Compute n1,n2 from epsilon if not directly given
        if self.n1 is None and self.epsilon1 is not None:
            self.n1 = math.sqrt(self.epsilon1)
        if self.n2 is None and self.epsilon2 is not None:
            self.n2 = math.sqrt(self.epsilon2)

        # Extract results safely
        self.tan_theta_R = None
        self.tan_theta_T = None
        self.R_prime = None
        self.R = None
        self.T = None

        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
            self.tan_theta_R = results.get('tan_theta_R', None)
            self.tan_theta_T = results.get('tan_theta_T', None)
            self.R_prime = results.get('R_prime', None)
            self.R = results.get('R', None)
            self.T = results.get('T', None)

    # ------------------------------------------------------------
    # Helper: compute rest-frame incidence angle
    # ------------------------------------------------------------
    def _compute_cos_theta_I_prime(self):
        """Return cos(theta'_I) from lab givens, or None if incomplete."""
        if any(x is None for x in (self.theta_I, self.beta)):
            return None
        cos_I = math.cos(self.theta_I)
        return (cos_I + self.beta) / (1.0 + self.beta * cos_I)

    def _compute_sin_theta_I_prime(self):
        """Return sin(theta'_I) from lab givens, or None if incomplete."""
        if any(x is None for x in (self.theta_I, self.beta)):
            return None
        sin_I = math.sin(self.theta_I)
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        return sin_I / (gamma * (1.0 + self.beta * math.cos(self.theta_I)))

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_givens(self):
        """All required givens must exist and be finite numbers."""
        required = ['beta', 'epsilon1', 'epsilon2', 'theta_I']
        for name in required:
            value = getattr(self, name, None)
            if value is None:
                return (False, f"Missing givens.{name}")
            if not isinstance(value, (int, float)):
                return (False, f"givens.{name} is not numeric")
            if not math.isfinite(value):
                return (False, f"givens.{name} is not finite")
        # Also need n1,n2 (either given or computed from epsilon)
        if self.n1 is None or self.n2 is None:
            return (False, "Cannot determine n1 or n2 from givens")
        return (True, "All required givens present and finite")

    def test_interface_required_results(self):
        """All required result keys must exist and be finite numbers."""
        required = ['tan_theta_R', 'tan_theta_T', 'R_prime', 'R', 'T']
        for name in required:
            value = getattr(self, name, None)
            if value is None:
                return (False, f"Missing results.{name}")
            if not isinstance(value, (int, float)):
                return (False, f"results.{name} is not numeric")
            if not math.isfinite(value):
                return (False, f"results.{name} is not finite")
        return (True, "All required results present and finite")

    # ------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_fresnel_t_equals_1_minus_R(self):
        """Energy conservation: T = 1 - R."""
        if self.R is None or self.T is None:
            return (False, "SKIP: R or T not available")
        if not math.isclose(self.T, 1.0 - self.R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T = {self.T}, 1-R = {1-self.R}")
        return (True, "T ≈ 1 - R")

    def test_invariant_R_prime_range(self):
        """Reflection coefficient in rest frame must be between 0 and 1."""
        if self.R_prime is None:
            return (False, "SKIP: R_prime not available")
        if not (0.0 <= self.R_prime <= 1.0):
            return (False, f"R_prime = {self.R_prime} outside [0,1]")
        return (True, "R_prime in [0,1]")

    def test_invariant_R_range(self):
        """Lab reflection coefficient must be between 0 and 1."""
        if self.R is None:
            return (False, "SKIP: R not available")
        if not (0.0 <= self.R <= 1.0):
            return (False, f"R = {self.R} outside [0,1]")
        return (True, "R in [0,1]")

    def test_invariant_T_range(self):
        """Lab transmission coefficient must be between 0 and 1."""
        if self.T is None:
            return (False, "SKIP: T not available")
        if not (0.0 <= self.T <= 1.0):
            return (False, f"T = {self.T} outside [0,1]")
        return (True, "T in [0,1]")

    # ------------------------------------------------------------
    # Modeling / trace tests (use math.isclose)
    # ------------------------------------------------------------
    def test_modeling_tan_theta_R_formula(self):
        """Re‑derive tan_theta_R from givens and compare to provided value."""
        if any(x is None for x in (self.beta, self.theta_I, self.tan_theta_R)):
            return (False, "SKIP: insufficient data for tan_theta_R check")
        sin_I = math.sin(self.theta_I)
        cos_I = math.cos(self.theta_I)
        numerator = (1.0 - self.beta**2) * sin_I
        denominator = (1.0 + self.beta**2) * cos_I - 2.0 * self.beta
        if abs(denominator) < 1e-12:
            return (False, "Denominator near zero, cannot compute")
        computed = numerator / denominator
        if not math.isclose(self.tan_theta_R, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tan_theta_R = {self.tan_theta_R}, expected ≈ {computed}")
        return (True, "tan_theta_R matches derived formula")

    def test_modeling_tan_theta_T_formula(self):
        """Re‑derive tan_theta_T from givens and compare to provided value."""
        if any(x is None for x in (self.beta, self.theta_I, self.n1, self.n2, self.tan_theta_T)):
            return (False, "SKIP: insufficient data for tan_theta_T check")
        sin_I = math.sin(self.theta_I)
        cos_I = math.cos(self.theta_I)
        factor1 = self.n1 * (1.0 - self.beta**2) * sin_I / (self.n2 * (1.0 - self.beta * cos_I))
        # expression inside sqrt
        inside = 1.0 - (self.n1**2 * (1.0 - self.beta**2) * sin_I**2) / (self.n2**2 * (1.0 - self.beta * cos_I)**2)
        if inside < 0:
            return (False, f"Inside sqrt negative: {inside}")
        sqrt_val = math.sqrt(inside)
        denominator = sqrt_val + self.beta
        if denominator < 1e-12:
            return (False, "Denominator near zero")
        computed = factor1 / denominator
        if not math.isclose(self.tan_theta_T, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tan_theta_T = {self.tan_theta_T}, expected ≈ {computed}")
        return (True, "tan_theta_T matches derived formula")

    def test_modeling_R_formula(self):
        """Re‑derive R from R_prime and compare to provided R."""
        if any(x is None for x in (self.beta, self.theta_I, self.R_prime, self.R)):
            return (False, "SKIP: insufficient data for R check")
        cos_I = math.cos(self.theta_I)
        factor = (1.0 - 2.0 * self.beta * cos_I + self.beta**2) / (1.0 - self.beta**2)
        computed_R = factor * self.R_prime
        if not math.isclose(self.R, computed_R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"R = {self.R}, expected ≈ {computed_R}")
        return (True, "R matches expected relation R = ((1-2βcosθ+β^2)/(1-β^2)) R_prime")

    # ------------------------------------------------------------
    # Monotonicity tests (perturb one input, recompute, compare direction)
    # ------------------------------------------------------------
    def test_monotonicity_R_increases_with_beta(self):
        """Check that R increases when beta increases slightly."""
        if any(x is None for x in (self.beta, self.theta_I, self.n1, self.n2, self.R)):
            return (False, "SKIP: insufficient data for monotonicity test")

        # compute R at current beta using the same method as in test_modeling_R_formula
        def compute_R(beta_val):
            gamma = 1.0 / math.sqrt(1.0 - beta_val**2)
            # compute theta'_I
            cos_I = math.cos(self.theta_I)
            sin_I = math.sin(self.theta_I)
            cos_I_prime = (cos_I + beta_val) / (1.0 + beta_val * cos_I)
            sin_I_prime = sin_I / (gamma * (1.0 + beta_val * cos_I))
            # compute theta'_T via Snell
            sin_T_prime = (self.n1 / self.n2) * sin_I_prime
            if sin_T_prime > 1.0:
                return None  # total internal reflection, skip
            cos_T_prime = math.sqrt(1.0 - sin_T_prime**2)
            # Fresnel r'
            r_prime = (self.n2 * cos_I_prime - self.n1 * cos_T_prime) / \
                      (self.n2 * cos_I_prime + self.n1 * cos_T_prime)
            R_prime = r_prime**2
            factor = (1.0 - 2.0 * beta_val * cos_I + beta_val**2) / (1.0 - beta_val**2)
            return factor * R_prime

        R0 = compute_R(self.beta)
        if R0 is None:
            return (False, "Cannot compute base R for given parameters")

        # perturb beta upward by 1%
        delta = 0.01 * self.beta if abs(self.beta) > 1e-6 else 0.01
        beta1 = self.beta + delta
        R1 = compute_R(beta1)
        if R1 is None:
            return (True, "SKIP: perturbed beta leads to TIR, cannot compare direction")

        if R1 >= R0:
            return (True, f"R increased when beta increased from {self.beta:.3f} to {beta1:.3f}: {R0:.4f} -> {R1:.4f}")
        else:
            return (False, f"R decreased when beta increased: {R0:.4f} -> {R1:.4f}")

    # ------------------------------------------------------------
    # Weak answer sanity (no isclose, no re‑derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Basic sanity: results are finite, not NaN, and within plausible ranges."""
        checks = []

        # tan_theta_R
        tr = self.tan_theta_R
        if tr is None:
            checks.append("tan_theta_R missing")
        else:
            if not math.isfinite(tr):
                checks.append("tan_theta_R not finite")
            if abs(tr) > 100.0:
                checks.append("tan_theta_R suspiciously large")

        # tan_theta_T
        tt = self.tan_theta_T
        if tt is None:
            checks.append("tan_theta_T missing")
        else:
            if not math.isfinite(tt):
                checks.append("tan_theta_T not finite")
            if abs(tt) > 100.0:
                checks.append("tan_theta_T suspiciously large")

        # R
        r = self.R
        if r is None:
            checks.append("R missing")
        else:
            if r < 0.0 or r > 1.0:
                checks.append(f"R = {r} outside [0,1]")

        # T
        t = self.T
        if t is None:
            checks.append("T missing")
        else:
            if t < 0.0 or t > 1.0:
                checks.append(f"T = {t} outside [0,1]")

        if checks:
            return (False, "; ".join(checks))
        return (True, "Basic sanity checks passed")

    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)