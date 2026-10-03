import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs and results safely with None defaults
        # Givens: mass m, total energy E0, speed of light c (expected but optional)
        self.m = getattr(solution_instance, 'm', None)
        self.E0 = getattr(solution_instance, 'E0', None)
        self.c = getattr(solution_instance, 'c', None)

        # Results from the solver (via results dict or attributes)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.cos_half_theta = solution_instance.results.get('cos_half_theta', None)
            self.theta = solution_instance.results.get('theta', None)
            self.theta_low = solution_instance.results.get('theta_low', None)
            self.theta_high = solution_instance.results.get('theta_high', None)
        else:
            # fallback: try direct attributes
            self.cos_half_theta = getattr(solution_instance, 'cos_half_theta', None)
            self.theta = getattr(solution_instance, 'theta', None)
            self.theta_low = getattr(solution_instance, 'theta_low', None)
            self.theta_high = getattr(solution_instance, 'theta_high', None)

        # Internal consistency will check derived relations when givens exist

    # --------------------------------------------------
    # Interface tests
    # --------------------------------------------------
    def test_interface_cos_half_theta(self):
        """Check that cos_half_theta exists and is a real number."""
        if self.cos_half_theta is None:
            return (False, "cos_half_theta is missing")
        if not isinstance(self.cos_half_theta, (int, float)):
            return (False, f"cos_half_theta should be numeric, got {type(self.cos_half_theta)}")
        return (True, "cos_half_theta present and numeric")

    def test_interface_theta(self):
        """Check that theta exists and is a real number."""
        if self.theta is None:
            return (False, "theta is missing")
        if not isinstance(self.theta, (int, float)):
            return (False, f"theta should be numeric, got {type(self.theta)}")
        return (True, "theta present and numeric")

    def test_interface_theta_low(self):
        """Check that theta_low exists and is a real number."""
        if self.theta_low is None:
            return (False, "theta_low is missing")
        if not isinstance(self.theta_low, (int, float)):
            return (False, f"theta_low should be numeric, got {type(self.theta_low)}")
        return (True, "theta_low present and numeric")

    def test_interface_theta_high(self):
        """Check that theta_high exists and is a real number."""
        if self.theta_high is None:
            return (False, "theta_high is missing")
        if not isinstance(self.theta_high, (int, float)):
            return (False, f"theta_high should be numeric, got {type(self.theta_high)}")
        return (True, "theta_high present and numeric")

    def test_interface_theta_range(self):
        """Check that theta is in [0, pi] (physical range)."""
        if self.theta is None:
            return (False, "SKIP: theta not available")
        if self.theta < 0 or self.theta > math.pi:
            return (False, f"theta={self.theta} out of [0, pi]")
        return (True, "theta in physical range")

    # --------------------------------------------------
    # Invariant / modeling tests (use math.isclose)
    # --------------------------------------------------
    def test_invariant_momentum_relation(self):
        """Check p0 = 2 p1 cos(phi) using relativistic energy-momentum relations."""
        if any(x is None for x in [self.m, self.E0, self.c, self.theta]):
            return (False, "SKIP: missing givens or theta")
        # Compute E1, p0, p1 from givens
        mc2 = self.m * self.c**2
        if self.E0 < mc2:
            return (False, "E0 < mc^2, unphysical")
        E1 = (self.E0 + mc2) / 2.0
        p0c = math.sqrt(self.E0**2 - mc2**2)   # p0 * c
        p1c = math.sqrt(E1**2 - mc2**2)        # p1 * c
        phi = self.theta / 2.0
        cos_phi = math.cos(phi)
        # Predict: p0c should equal 2 * p1c * cos(phi)
        lhs = p0c
        rhs = 2.0 * p1c * cos_phi
        if p0c == 0 and p1c == 0:
            # degenerate case: both zero, relation satisfied
            return (True, "both momenta zero, relation vacuously true")
        if p1c < 0 or p0c < 0:
            return (False, "negative momentum magnitude")
        rel_diff = abs(lhs - rhs) / max(lhs, rhs) if max(lhs, rhs) > 0 else 0.0
        if rel_diff > 1e-2:
            return (False, f"p0c={lhs:.4e} != 2*p1c*cos(phi)={rhs:.4e}, rel error {rel_diff:.3e}")
        return (True, f"p0c and 2*p1c*cos(phi) agree: {lhs:.4e} vs {rhs:.4e}")

    def test_invariant_energy_conservation(self):
        """Check E_initial = E_final: E0 + mc^2 = 2E1 (via E1 from result relation)."""
        if any(x is None for x in [self.m, self.E0, self.c]):
            return (False, "SKIP: missing givens")
        mc2 = self.m * self.c**2
        E1_derived = (self.E0 + mc2) / 2.0
        # We can't directly check E1, but we can infer it from the result's cos_half_theta
        # Instead, use the derived formula to recompute E1 from cos_half_theta
        # cos_half_theta = sqrt((E0+mc2)/(E0+3mc2))
        # => (E0+3mc2)cos^2 = E0+mc2 => E0(cos^2-1) = mc2(1-3cos^2) => ...
        # But that is exactly the answer - we avoid recomputing the final formula.
        # Better: use the intermediate step: from momentum relation we already tested.
        # So we can check that cos_half_theta from the result satisfies:
        # cos_half_theta^2 * (E0 + 3mc^2) = E0 + mc^2
        cos_half = self.cos_half_theta
        if cos_half is None:
            return (False, "SKIP: cos_half_theta not available")
        if cos_half < 0 or cos_half > 1:
            return (False, f"cos_half_theta out of [0,1]: {cos_half}")
        lhs = cos_half**2 * (self.E0 + 3 * mc2)
        rhs = self.E0 + mc2
        if not math.isclose(lhs, rhs, rel_tol=1e-2):
            return (False, f"cos^2 * (E0+3mc^2) = {lhs:.4e}, expected {rhs:.4e}")
        return (True, "Energy conservation consistency verified")

    def test_invariant_cos_half_theta_one_when_high_energy(self):
        """In high-energy limit (E0 >> mc^2), cos_half_theta should approach 1."""
        if any(x is None for x in [self.m, self.E0, self.c, self.cos_half_theta]):
            return (False, "SKIP: missing data")
        mc2 = self.m * self.c**2
        if self.E0 > 100 * mc2:
            # High energy regime: check that cos_half_theta is close to 1
            if not math.isclose(self.cos_half_theta, 1.0, abs_tol=0.02):
                return (False, f"For high E0, cos_half should be ~1, got {self.cos_half_theta}")
            return (True, "High-energy cos_half ~1")
        else:
            return (False, "SKIP: not high energy regime")

    def test_invariant_cos_half_theta_sqrt_half_when_low_energy(self):
        """In low-energy limit (E0 ≈ mc^2), cos_half_theta should approach 1/sqrt(2)."""
        if any(x is None for x in [self.m, self.E0, self.c, self.cos_half_theta]):
            return (False, "SKIP: missing data")
        mc2 = self.m * self.c**2
        if math.isclose(self.E0 / mc2, 1.0, rel_tol=0.1):
            target = 1.0 / math.sqrt(2)
            if not math.isclose(self.cos_half_theta, target, rel_tol=0.02):
                return (False, f"Low-energy cos_half should be {target:.4f}, got {self.cos_half_theta}")
            return (True, "Low-energy cos_half ≈ 1/√2")
        else:
            return (False, "SKIP: not low energy regime")

    # --------------------------------------------------
    # Modeling / trace checks (using explanation steps)
    # --------------------------------------------------
    def test_trace_theta_from_cos_half(self):
        """Check that theta = 2*acos(cos_half_theta) (with numeric fallback)."""
        if self.cos_half_theta is None or self.theta is None:
            return (False, "SKIP: missing cos_half_theta or theta")
        expected_theta = 2 * math.acos(self.cos_half_theta)
        if not math.isclose(self.theta, expected_theta, rel_tol=1e-5):
            return (False, f"theta={self.theta} does not match 2*acos(cos_half)={expected_theta}")
        return (True, "theta consistent with cos_half_theta")

    def test_trace_phi_angle_reference(self):
        """Check that phi = theta/2 is used consistently (implicit from theta)."""
        if self.theta is None:
            return (False, "SKIP: theta not available")
        phi = self.theta / 2.0
        if phi < 0 or phi > math.pi / 2:
            return (False, f"phi = theta/2 = {phi} out of [0, pi/2]")
        return (True, f"phi {phi} in plausible range")

    # --------------------------------------------------
    # Monotonicity tests
    # --------------------------------------------------
    def test_monotonicity_theta_decreases_with_energy(self):
        """As E0 increases, theta should decrease (from π/2 to 0)."""
        if any(x is None for x in [self.m, self.E0, self.c, self.theta]):
            return (False, "SKIP: missing givens or theta")
        # Perturb E0 up by 10% and compute expected theta using derived formula
        E0_perturbed = self.E0 * 1.1
        # Use cos_half_theta formula derived from givens (same as physics mapping)
        mc2 = self.m * self.c**2
        # Guard against unphysical
        if E0_perturbed < mc2:
            return (False, "Perturbed E0 unphysical")
        cos_half_pert = math.sqrt((E0_perturbed + mc2) / (E0_perturbed + 3 * mc2))
        theta_pert = 2 * math.acos(cos_half_pert)
        # Baseline theta from result: use self.theta (could be from solver)
        # Instead, we can compute baseline using the same formula to avoid dependency
        # But we want to test the trend: compare baseline from result with perturbed computed
        # Since we don't have the baseline computed from formula, we use the result's theta
        if theta_pert >= self.theta - 1e-9:
            return (False, f"Theta did not decrease; baseline={self.theta}, perturbed={theta_pert}")
        return (True, f"Theta decreases: {self.theta} -> {theta_pert}")

    def test_monotonicity_cos_half_increases_with_energy(self):
        """cos_half_theta should increase with E0 (from 1/√2 to 1)."""
        if any(x is None for x in [self.m, self.E0, self.c, self.cos_half_theta]):
            return (False, "SKIP: missing data")
        E0_pert = self.E0 * 1.1
        mc2 = self.m * self.c**2
        if E0_pert < mc2:
            return (False, "Perturbed E0 unphysical")
        cos_half_pert = math.sqrt((E0_pert + mc2) / (E0_pert + 3 * mc2))
        if cos_half_pert <= self.cos_half_theta - 1e-9:
            return (False, f"cos_half did not increase; baseline={self.cos_half_theta}, perturbed={cos_half_pert}")
        return (True, f"cos_half increases: {self.cos_half_theta} -> {cos_half_pert}")

    # --------------------------------------------------
    # Weak answer sanity (only one allowed, no math.isclose, no computed expectations)
    # --------------------------------------------------
    def test_answer_sanity_limits(self):
        """Very weak check: theta_low ~ pi/2, theta_high ~ 0 (within ~0.2 rad)."""
        if self.theta_low is None or self.theta_high is None:
            return (False, "SKIP: limit values missing")
        # Not using math.isclose: simply check approximate range (broad)
        # pi/2 = 1.5708, allow 0.15 rad deviation (about 8.5°)
        low_ok = 1.42 <= self.theta_low <= 1.72
        high_ok = 0.0 <= self.theta_high <= 0.15
        if not low_ok:
            return (False, f"theta_low={self.theta_low} not near pi/2 (~1.57)")
        if not high_ok:
            return (False, f"theta_high={self.theta_high} not near 0")
        return (True, "Limits approximately correct")

    # ======================================================
    # Runner
    # ======================================================
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all bound methods whose name starts with 'test_'
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