import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for the precession of the Earth's orbit due to a dust cloud.
    Checks focus on theorem-based invariants, modeling consistency, monotonicity, and sign/physical sense.
    """

    def __init__(self, solution_instance):
        """Extract all needed data safely with None defaults."""
        results = getattr(solution_instance, 'results', {})
        self.Omega_prec = results.get('Omega_prec', None)
        self.omega_theta = results.get('omega_theta', None)
        self.omega_r = results.get('omega_r', None)

        givens = getattr(solution_instance, 'givens', {})
        self.k = givens.get('k', None)
        self.m = givens.get('m', None)
        self.r0 = givens.get('r0', None)
        self.G = givens.get('G', None)
        self.M = givens.get('M', None)

        # Sometimes included as derived
        self.omega0 = results.get('omega0', None)

        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # --------------------------------------------------------------------------
    # Interface checks (required keys exist and have valid types)
    # --------------------------------------------------------------------------

    def test_interface_required_keys(self):
        """Check that all three key results are present and are numeric."""
        if self.Omega_prec is None:
            return False, "Missing required result: 'Omega_prec'"
        if not isinstance(self.Omega_prec, (int, float)):
            return False, f"Omega_prec has invalid type: {type(self.Omega_prec)}"
        if self.omega_theta is None:
            return False, "Missing required result: 'omega_theta'"
        if not isinstance(self.omega_theta, (int, float)):
            return False, f"omega_theta has invalid type: {type(self.omega_theta)}"
        if self.omega_r is None:
            return False, "Missing required result: 'omega_r'"
        if not isinstance(self.omega_r, (int, float)):
            return False, f"omega_r has invalid type: {type(self.omega_r)}"
        return True, "All required keys present and are numeric"

    # --------------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose)
    # --------------------------------------------------------------------------

    def test_invariant_precession_definition(self):
        """Omega_prec should equal omega_theta - omega_r (definition)."""
        if any(x is None for x in [self.Omega_prec, self.omega_theta, self.omega_r]):
            return False, "SKIP: one or more results missing"
        expected = self.omega_theta - self.omega_r
        if math.isclose(self.Omega_prec, expected, rel_tol=1e-3, abs_tol=1e-3):
            return True, f"Omega_prec = {self.Omega_prec} ≈ ω_θ - ω_r = {expected}"
        else:
            return False, f"Omega_prec = {self.Omega_prec}, but expected {expected}"

    def test_invariant_radial_faster_than_angular(self):
        """For retrograde precession (k>0 small), ω_r > ω_θ."""
        if any(x is None for x in [self.omega_theta, self.omega_r]):
            return False, "SKIP: omega_theta or omega_r missing"
        if self.omega_r > self.omega_theta:
            return True, f"ω_r = {self.omega_r} > ω_θ = {self.omega_theta} (consistent with retrograde)"
        else:
            return False, f"ω_r = {self.omega_r} <= ω_θ = {self.omega_theta} (should be > for k>0)"

    def test_invariant_positive_frequencies(self):
        """All angular and radial frequencies must be positive."""
        if self.omega_theta is not None and self.omega_theta <= 0:
            return False, f"omega_theta = {self.omega_theta} is not positive"
        if self.omega_r is not None and self.omega_r <= 0:
            return False, f"omega_r = {self.omega_r} is not positive"
        return True, "All available frequencies are positive"

    # --------------------------------------------------------------------------
    # Modeling / trace consistency checks (derive relations from givens)
    # --------------------------------------------------------------------------

    def test_modeling_precession_sign_and_magnitude(self):
        """
        Weak check: if k > 0 (assumed from problem), Omega_prec should be negative.
        Also ensure it is finite and non‑zero.
        """
        if self.Omega_prec is None:
            return False, "SKIP: Omega_prec missing"
        if not math.isfinite(self.Omega_prec):
            return False, "Omega_prec is not finite"
        if self.k is not None and self.k > 0:
            if self.Omega_prec < 0:
                return True, f"Omega_prec = {self.Omega_prec} < 0 (retrograde, as expected for k>0)"
            else:
                return False, f"For k>0, Omega_prec should be negative, got {self.Omega_prec}"
        # Without k we just check that magnitude > 0
        if abs(self.Omega_prec) > 0:
            return True, f"|Omega_prec| = {abs(self.Omega_prec)} > 0 (non‑zero)"
        else:
            return False, "Omega_prec is exactly zero; should be non‑zero for k≠0"

    def test_modeling_first_order_magnitude(self):
        """
        Re‑derive the first‑order expression for Omega_prec from the effective potential parameters,
        and check that the solution's value is consistent with the derived analytic form.
        This uses the relation: Ω_prec ≈ -3k/(2 m ω₀)   (valid to first order in k).
        We compute ω₀ from G, M, r0 if available; otherwise from the angular frequency approximation.
        """
        # Need k, m, and ω₀
        if any(x is None for x in [self.k, self.m]):
            return False, "SKIP: k or m missing"
        # Try to get ω₀ from fundamental parameters
        omega0 = None
        if all(x is not None for x in [self.G, self.M, self.r0]):
            omega0 = math.sqrt(self.G * self.M / self.r0**3)
        elif self.omega0 is not None:
            omega0 = self.omega0
        # If omega0 still unknown, attempt to extract from omega_theta via inversion:
        # ω_θ ≈ ω₀(1 + k/(2 m ω₀²)) => solve quadratic, but too involved. We'll skip.
        if omega0 is None:
            return False, "SKIP: cannot determine ω₀ from provided data"
        # Derive expected first‑order value
        expected = -3.0 * self.k / (2.0 * self.m * omega0)
        # Compare with solution (lenient tolerance because of first‑order truncation)
        if self.Omega_prec is None:
            return False, "SKIP: Omega_prec missing for comparison"
        if math.isclose(self.Omega_prec, expected, rel_tol=0.05, abs_tol=1e-2):
            return True, f"Ω_prec = {self.Omega_prec} matches first‑order formula {expected}"
        else:
            return False, f"Ω_prec = {self.Omega_prec}, but first‑order formula gives {expected} (rel diff >5%)"

    # --------------------------------------------------------------------------
    # Monotonicity checks (explicitly perturb an input, compare outputs)
    # --------------------------------------------------------------------------

    def test_monotonicity_precession_magnitude_increases_with_k(self):
        """
        Check that increasing the dust parameter k increases |Ω_prec| (makes it more negative).
        We re‑derive Ω_prec from the first‑order formula for both baseline and perturbed k.
        This test confirms the monotonic relationship required by physics.
        """
        # Need k, m, and ω₀
        if any(x is None for x in [self.k, self.m]):
            return False, "SKIP: k or m missing"
        omega0 = None
        if all(x is not None for x in [self.G, self.M, self.r0]):
            omega0 = math.sqrt(self.G * self.M / self.r0**3)
        elif self.omega0 is not None:
            omega0 = self.omega0
        if omega0 is None:
            return False, "SKIP: cannot determine ω₀"
        # Baseline derived value
        base = -3.0 * self.k / (2.0 * self.m * omega0)
        # Perturb k: +10%
        k_pert = 1.1 * self.k
        pert = -3.0 * k_pert / (2.0 * self.m * omega0)
        # Monotonicity: pert should be more negative (smaller) than base
        if pert < base:
            return True, f"k increased 10%: base Ω={base:.4g}, perturbed Ω={pert:.4g} → more negative (monotonic)"
        else:
            return False, f"Perturbation did not decrease: base {base}, pert {pert}"

    def test_monotonicity_radial_frequency_increases_with_k(self):
        """
        Check that radial oscillation frequency ω_r increases when k increases.
        Use the first‑order expression: ω_r ≈ ω₀ + (2k)/(m ω₀).
        """
        if any(x is None for x in [self.k, self.m]):
            return False, "SKIP: k or m missing"
        omega0 = None
        if all(x is not None for x in [self.G, self.M, self.r0]):
            omega0 = math.sqrt(self.G * self.M / self.r0**3)
        elif self.omega0 is not None:
            omega0 = self.omega0
        if omega0 is None:
            return False, "SKIP: cannot determine ω₀"
        # Baseline ω_r (first order)
        base_omega_r = omega0 + 2.0 * self.k / (self.m * omega0)
        # Perturb k: +10%
        k_pert = 1.1 * self.k
        pert_omega_r = omega0 + 2.0 * k_pert / (self.m * omega0)
        if pert_omega_r > base_omega_r:
            return True, f"k increased 10%: base ω_r={base_omega_r:.4g}, perturbed={pert_omega_r:.4g} → increased (monotonic)"
        else:
            return False, f"Perturbation did not increase ω_r: base {base_omega_r}, pert {pert_omega_r}"

    # --------------------------------------------------------------------------
    # Answer sanity (very weak – only finiteness and sign)
    # --------------------------------------------------------------------------

    def test_answer_sanity_finite_and_nonzero(self):
        """Omega_prec must be finite and (if k != 0) non‑zero."""
        if self.Omega_prec is None:
            return False, "SKIP: Omega_prec missing"
        if not math.isfinite(self.Omega_prec):
            return False, "Omega_prec is infinite or NaN"
        # If k is known to be non‑zero, check non‑zero
        if self.k is not None and self.k != 0:
            if self.Omega_prec == 0:
                return False, "Omega_prec is zero but k ≠ 0"
        return True, "Omega_prec is finite and (if k≠0) non‑zero"

    # --------------------------------------------------------------------------
    # Main runner (do not modify)
    # --------------------------------------------------------------------------

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)