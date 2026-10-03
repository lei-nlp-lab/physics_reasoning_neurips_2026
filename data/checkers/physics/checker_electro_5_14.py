import math

class PhysicsChecker:
    """Physics auditor for radial force on electron in relativistic cylindrical beam."""

    def __init__(self, solution_instance):
        # Extract givens and results safely; default None for any missing value.
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})

        self.e = givens.get('e', None)
        self.n = givens.get('n', None)
        self.v = givens.get('v', None)
        self.r = givens.get('r', None)
        self.eps0 = givens.get('eps0', None)
        self.mu0 = givens.get('mu0', None)
        self.c = givens.get('c', None)

        self.F_r = results.get('F_r', None)

    # --- Interface tests ---
    def test_interface_required_keys(self):
        """Check that the result key F_r exists and is numeric."""
        if self.F_r is None:
            return (False, "SKIP: F_r not found in results")
        if not isinstance(self.F_r, (int, float)):
            return (False, "F_r should be numeric (int or float)")
        return (True, "F_r present and numeric")

    # --- Invariant (theorem-based) tests ---
    def test_invariant_lorentz_force_relation(self):
        """Check the Lorentz force composition: F_r = -e E_r + e v B_phi
           with E_r and B_phi derived via Gauss and Ampere."""
        if any(x is None for x in [self.e, self.n, self.r, self.v, self.eps0, self.mu0]):
            return (False, "SKIP: missing givens for Lorentz check")
        E_r = -self.e * self.n * self.r / (2 * self.eps0)
        B_phi = -self.mu0 * self.e * self.n * self.v * self.r / 2
        computed_F_r = -self.e * E_r + self.e * self.v * B_phi
        if math.isclose(self.F_r, computed_F_r, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Lorentz relation holds: -e E_r + e v B_phi = {computed_F_r}")
        return (False, f"Lorentz composition fails: computed {computed_F_r}, actual {self.F_r}")

    def test_invariant_sign(self):
        """For r > 0, the net electric repulsion dominates, so F_r should be positive."""
        if self.r is None or self.F_r is None:
            return (False, "SKIP: missing r or F_r")
        if self.r > 0:
            if self.F_r > 0:
                return (True, "F_r positive for r>0 (outward, repulsive)")
            return (False, "F_r should be positive for r>0")
        return (True, "r=0, sign test skipped")

    def test_invariant_zero_at_center(self):
        """At r = 0, the radial force must be zero."""
        if self.r is None or self.F_r is None:
            return (False, "SKIP: missing r or F_r")
        if self.r == 0:
            if math.isclose(self.F_r, 0.0, abs_tol=1e-12):
                return (True, "F_r = 0 at r=0")
            return (False, f"Expected zero at r=0, got {self.F_r}")
        return (True, "r>0, zero-centre check not applicable")

    # --- Modeling / trace tests (based on explanation steps) ---
    def test_modeling_gamma_formula(self):
        """Verify that the result matches the explanation steps formula:
           F_r = (e^2 n r) / (2 ε0) * 1/γ^2, with γ = 1/√(1 - v²/c²)."""
        if any(x is None for x in [self.e, self.n, self.r, self.v, self.eps0, self.c, self.F_r]):
            return (False, "SKIP: missing data for gamma formula check")
        gamma_sq = 1 / (1 - self.v**2 / self.c**2)
        expected = (self.e**2 * self.n * self.r) / (2 * self.eps0 * gamma_sq)
        if math.isclose(self.F_r, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F_r matches the gamma-formula: {self.F_r} ~ {expected}")
        return (False, f"Gamma formula mismatch: {self.F_r} vs {expected}")

    def test_modeling_derived_fields_consistent(self):
        """Check that the underlying E_r and B_phi from the pseudocode
           have the correct signs and satisfy the Lorentz relation (as an equation)."""
        if any(x is None for x in [self.e, self.n, self.r, self.v, self.eps0, self.mu0, self.F_r]):
            return (False, "SKIP: missing data for derived fields check")
        E_r = -self.e * self.n * self.r / (2 * self.eps0)
        B_phi = -self.mu0 * self.e * self.n * self.v * self.r / 2
        lhs = self.F_r
        rhs = -self.e * E_r + self.e * self.v * B_phi
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "The derived E_r and B_phi yield the given F_r via Lorentz force.")
        return (False, "Inconsistency between derived fields and F_r")

    # --- Monotonicity (trend) tests ---
    def test_monotonicity_radial_increase(self):
        """F_r should increase when radial coordinate increases (linear scaling)."""
        if any(x is None for x in [self.e, self.n, self.r, self.v, self.eps0, self.c, self.F_r]):
            return (False, "SKIP: missing data for radial monotonicity")
        if self.r == 0:
            return (False, "SKIP: r=0, cannot perturb direction")
        # Perturb radial coordinate by +10%
        r_pert = self.r * 1.1
        gamma_sq = 1 / (1 - self.v**2 / self.c**2)
        F_r_baseline = self.F_r
        F_r_perturbed = (self.e**2 * self.n * r_pert) / (2 * self.eps0 * gamma_sq)
        # Check that F_r increases in the same direction as the perturbation
        if (r_pert > self.r and F_r_perturbed > F_r_baseline) or (r_pert < self.r and F_r_perturbed < F_r_baseline):
            return (True, f"F_r increases with r: baseline {F_r_baseline}, perturbed {F_r_perturbed}")
        return (False, f"F_r did not increase with r: baseline {F_r_baseline}, perturbed {F_r_perturbed}")

    def test_monotonicity_velocity_decrease(self):
        """F_r should decrease when velocity increases (relativistic gamma effect)."""
        if any(x is None for x in [self.e, self.n, self.r, self.v, self.eps0, self.c, self.F_r]):
            return (False, "SKIP: missing data for velocity monotonicity")
        # Perturb velocity upward by 10% (cap at 0.999c to stay physical)
        v_pert = min(self.v * 1.1, 0.999 * self.c)
        if v_pert <= self.v:
            return (False, "SKIP: velocity perturbation could not increase (maybe already near c)")
        gamma_sq_base = 1 / (1 - self.v**2 / self.c**2)
        gamma_sq_pert = 1 / (1 - v_pert**2 / self.c**2)
        F_r_baseline = self.F_r
        F_r_perturbed = (self.e**2 * self.n * self.r) / (2 * self.eps0 * gamma_sq_pert)
        # For increasing v, gamma increases, so F_r should decrease (smaller positive)
        if F_r_perturbed < F_r_baseline:
            return (True, f"F_r decreases with v: baseline {F_r_baseline}, perturbed {F_r_perturbed}")
        return (False, f"F_r did not decrease with v: baseline {F_r_baseline}, perturbed {F_r_perturbed}")

    # --- Weak answer sanity (only one such test allowed) ---
    def test_answer_sanity_weak(self):
        """Gentle sanity: finite result and sensible sign for r>0."""
        if self.F_r is None:
            return (False, "SKIP: no F_r to check")
        if not math.isfinite(self.F_r):
            return (False, "F_r is not finite (inf or NaN)")
        if self.r is not None and self.r > 0 and self.F_r <= 0:
            return (False, "F_r should be positive for positive r")
        return (True, "Sanity checks pass: finite, sign consistent")

    # --- Test runner ---
    def run_all_tests(self):
        """Discover and run all test_* methods, collect logs and overall pass/fail."""
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