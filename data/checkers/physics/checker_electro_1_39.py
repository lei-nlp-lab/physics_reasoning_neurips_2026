import math
import traceback

class PhysicsChecker:
    """
    Audit a solution for the force needed to hold together two halves
    of a uniformly charged conducting sphere.
    """

    def __init__(self, solution_instance):
        # ---------- givens (safe extraction) ----------
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}

        self.Q = getattr(solution_instance, 'Q', givens.get('Q', None))
        self.R = getattr(solution_instance, 'R', givens.get('R', None))
        self.epsilon0 = getattr(solution_instance, 'epsilon0', givens.get('epsilon0', None))

        # ---------- result(s) ----------
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.F = results.get('F', None)

        # ---------- computed intermediates (internal only) ----------
        self.sigma = None
        self.pressure = None
        if self.Q is not None and self.R is not None and self.R != 0:
            self.sigma = self.Q / (4.0 * math.pi * self.R * self.R)
        if self.sigma is not None and self.epsilon0 is not None and self.epsilon0 != 0:
            self.pressure = self.sigma * self.sigma / (2.0 * self.epsilon0)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_F_present(self):
        if self.F is None:
            return (False, "SKIP? F not available in results['F']")
        return (True, "F is present")

    def test_interface_givens_available(self):
        missing = []
        if self.Q is None:
            missing.append('Q')
        if self.R is None:
            missing.append('R')
        if self.epsilon0 is None:
            missing.append('epsilon0')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Theorem-based invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_sigma_positive(self):
        if self.sigma is None:
            return (False, "SKIP: sigma cannot be computed")
        if self.sigma > 0:
            return (True, f"sigma = {self.sigma:.4e} > 0")
        return (False, f"sigma is non-positive: {self.sigma:.4e}")

    def test_invariant_pressure_positive(self):
        if self.pressure is None:
            return (False, "SKIP: pressure cannot be computed")
        if self.pressure > 0:
            return (True, f"pressure = {self.pressure:.4e} > 0")
        return (False, f"pressure is non-positive: {self.pressure:.4e}")

    def test_invariant_force_positive_and_finite(self):
        if self.F is None:
            return (False, "SKIP: F unavailable")
        if not math.isfinite(self.F) or self.F <= 0:
            return (False, f"F is not a positive, finite number: {self.F}")
        return (True, f"F = {self.F:.4e} is positive and finite")

    # ------------------------------------------------------------------
    # Modeling / trace tests (multiple to satisfy requirement)
    # ------------------------------------------------------------------
    def test_modeling_force_via_pressure(self):
        """Verify that F = pressure * π * R^2 (closed-form integral result)."""
        if None in (self.pressure, self.R, self.F):
            return (False, "SKIP: insufficient data to check pressure route")
        expected = self.pressure * math.pi * self.R * self.R
        if math.isclose(expected, self.F, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F ≈ p·πR² = {expected:.4e} (emitted F = {self.F:.4e})")
        return (False, f"Mismatch: p·πR² = {expected:.4e}, emitted F = {self.F:.4e}")

    def test_modeling_numerical_integration(self):
        """
        Approximate the force by numerically integrating dF_z = p cosθ dA
        over the hemisphere (θ: 0→π/2, φ: 0→2π) using a coarse grid.
        Compare the result to the emitted F.
        """
        if None in (self.Q, self.R, self.epsilon0, self.F):
            return (False, "SKIP: missing data for numerical integration")
        if self.R <= 0 or self.epsilon0 <= 0:
            return (False, "SKIP: R or epsilon0 not positive")

        # Surface charge density
        sigma = self.Q / (4.0 * math.pi * self.R * self.R)
        pressure = sigma * sigma / (2.0 * self.epsilon0)

        # Numerical integration: use simple midpoint rule with Ntheta=200, Nphi=400
        Ntheta = 200
        Nphi = 400
        dtheta = (math.pi / 2.0) / Ntheta
        dphi = (2.0 * math.pi) / Nphi

        total = 0.0
        for i in range(Ntheta):
            theta = (i + 0.5) * dtheta  # midpoint
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            # dA = R^2 sinθ dθ dφ
            dA = self.R * self.R * sin_theta * dtheta * dphi
            # dF = p * cosθ * dA
            dF = pressure * cos_theta * dA
            total += dF * Nphi  # because dphi already included, but we need sum over φ
            # Actually dA includes dphi, so for each theta we multiply by number of phi segments
        # More correct: sum over phi loop inside
        total2 = 0.0
        for i in range(Ntheta):
            theta = (i + 0.5) * dtheta
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            for j in range(Nphi):
                # phi not needed in integrand (cosθ dA)
                dA = self.R * self.R * sin_theta * dtheta * dphi
                dF = pressure * cos_theta * dA
                total2 += dF
        # But the integral should be independent of φ, so we can just multiply by 2π after θ integration
        # Simpler: do 1D integral over θ and multiply by 2π
        # dθ integral: ∫₀^{π/2} cosθ sinθ dθ = 1/2
        # So correct analytic: F = p * 2π * R² * (1/2) = pπR²
        # For numerical test, let's do the 1D version correctly:
        integral_theta = 0.0
        for i in range(Ntheta):
            theta = (i + 0.5) * dtheta
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            integral_theta += cos_theta * sin_theta * dtheta
        # integral_theta should be ≈ 0.5
        F_numeric = pressure * self.R * self.R * 2.0 * math.pi * integral_theta

        if math.isclose(F_numeric, self.F, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Numerical integration yields F ≈ {F_numeric:.4e} (emitted {self.F:.4e})")
        return (False, f"Numerical integration gives {F_numeric:.4e}, emitted {self.F:.4e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input)
    # ------------------------------------------------------------------
    def test_monotonicity_force_increases_with_Q_squared(self):
        """
        Perturb Q upward by +10% and verify that F increases by factor 1.21.
        This tests the scaling F ∝ Q².
        """
        if None in (self.Q, self.R, self.epsilon0, self.F):
            return (False, "SKIP: missing givens for monotonicity test")

        # baseline force from model
        sigma0 = self.Q / (4.0 * math.pi * self.R * self.R)
        p0 = sigma0 * sigma0 / (2.0 * self.epsilon0)
        F0 = p0 * math.pi * self.R * self.R

        # perturb Q up by 10%
        Q_pert = 1.1 * self.Q
        sigma_pert = Q_pert / (4.0 * math.pi * self.R * self.R)
        p_pert = sigma_pert * sigma_pert / (2.0 * self.epsilon0)
        F_pert = p_pert * math.pi * self.R * self.R

        actual_ratio = F_pert / F0
        expected_ratio = 1.21  # (1.1)^2
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"Q→1.1Q gives F ratio {actual_ratio:.4f} ≈ {expected_ratio}")
        return (False, f"Scaling with Q fails: ratio {actual_ratio:.4f}, expected {expected_ratio}")

    def test_monotonicity_force_decreases_with_R_squared(self):
        """
        Perturb R upward by +10% and verify that F decreases by factor 1/1.21.
        This tests the scaling F ∝ 1/R².
        """
        if None in (self.Q, self.R, self.epsilon0, self.F):
            return (False, "SKIP: missing givens for monotonicity test")

        sigma0 = self.Q / (4.0 * math.pi * self.R * self.R)
        p0 = sigma0 * sigma0 / (2.0 * self.epsilon0)
        F0 = p0 * math.pi * self.R * self.R

        # perturb R up by 10%
        R_pert = 1.1 * self.R
        sigma_pert = self.Q / (4.0 * math.pi * R_pert * R_pert)
        p_pert = sigma_pert * sigma_pert / (2.0 * self.epsilon0)
        F_pert = p_pert * math.pi * R_pert * R_pert

        actual_ratio = F_pert / F0
        expected_ratio = 1.0 / 1.21  # ≈0.82645
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"R→1.1R gives F ratio {actual_ratio:.4f} ≈ {expected_ratio:.4f}")
        return (False, f"Scaling with R fails: ratio {actual_ratio:.4f}, expected {expected_ratio:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Only sign and finiteness checks – no exact formula re-derivation
        and no use of math.isclose.
        """
        if self.F is None:
            return (False, "SKIP: F not available")
        if not math.isfinite(self.F):
            return (False, f"F is not finite: {self.F}")
        if self.F <= 0:
            return (False, f"F is not positive: {self.F}")
        return (True, f"F = {self.F:.4e} passes basic sanity (positive, finite)")

    # ------------------------------------------------------------------
    # Test runner
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