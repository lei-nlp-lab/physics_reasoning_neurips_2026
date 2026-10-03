import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely - default to None when missing
        self.u1 = getattr(solution_instance, 'u1', None)
        self.f1 = getattr(solution_instance, 'f1', None)
        self.d = getattr(solution_instance, 'd', None)
        self.f2 = getattr(solution_instance, 'f2', None)

        # Extract results safely - default to None
        results = getattr(solution_instance, 'results', {})
        self.results = results if isinstance(results, dict) else {}

        self.v1 = self.results.get('v1', None)
        self.u2 = self.results.get('u2', None)
        self.v2 = self.results.get('v2', None)
        self.m1 = self.results.get('m1', None)
        self.m2 = self.results.get('m2', None)
        self.M = self.results.get('M', None)
        self.height_ratio = self.results.get('height_ratio', None)

    # ---- Interface tests ----
    def test_interface_required_keys(self):
        """Check that all expected result keys exist."""
        required = ['v1', 'u2', 'v2', 'm1', 'm2', 'M', 'height_ratio']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing results keys: {missing}")
        return (True, "All required keys present.")

    # ---- Invariant tests (use math.isclose) ----
    def test_invariant_lens1_equation(self):
        """Check thin lens equation for lens1: 1/v1 + 1/u1 = 1/f1."""
        if any(x is None for x in [self.v1, self.u1, self.f1]):
            return (False, "SKIP: missing v1, u1, or f1")
        left = 1/self.v1 + 1/self.u1
        right = 1/self.f1
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Lens1 equation satisfied: {left} ≈ {right}")
        return (False, f"Lens1 equation fails: left={left}, right={right}")

    def test_invariant_lens2_equation(self):
        """Check thin lens equation for lens2: 1/v2 + 1/u2 = 1/f2."""
        if any(x is None for x in [self.v2, self.u2, self.f2]):
            return (False, "SKIP: missing v2, u2, or f2")
        left = 1/self.v2 + 1/self.u2
        right = 1/self.f2
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Lens2 equation satisfied: {left} ≈ {right}")
        return (False, f"Lens2 equation fails: left={left}, right={right}")

    def test_invariant_magnification_m1(self):
        """Check m1 = -v1/u1."""
        if any(x is None for x in [self.m1, self.v1, self.u1]):
            return (False, "SKIP: missing m1, v1, or u1")
        expected = -self.v1 / self.u1
        if math.isclose(self.m1, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"m1 matches -v1/u1: {self.m1} ≈ {expected}")
        return (False, f"m1 mismatch: got {self.m1}, expected {expected}")

    def test_invariant_magnification_m2(self):
        """Check m2 = -v2/u2."""
        if any(x is None for x in [self.m2, self.v2, self.u2]):
            return (False, "SKIP: missing m2, v2, or u2")
        expected = -self.v2 / self.u2
        if math.isclose(self.m2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"m2 matches -v2/u2: {self.m2} ≈ {expected}")
        return (False, f"m2 mismatch: got {self.m2}, expected {expected}")

    def test_invariant_total_magnification(self):
        """Check M = m1 * m2."""
        if any(x is None for x in [self.M, self.m1, self.m2]):
            return (False, "SKIP: missing M, m1, or m2")
        product = self.m1 * self.m2
        if math.isclose(self.M, product, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"M equals m1*m2: {self.M} ≈ {product}")
        return (False, f"M mismatch: got {self.M}, product={product}")

    def test_invariant_height_ratio(self):
        """Check height_ratio = |M|."""
        if any(x is None for x in [self.height_ratio, self.M]):
            return (False, "SKIP: missing height_ratio or M")
        expected = abs(self.M)
        if math.isclose(self.height_ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"height_ratio matches |M|: {self.height_ratio} ≈ {expected}")
        return (False, f"height_ratio mismatch: got {self.height_ratio}, |M|={expected}")

    # ---- Modeling / trace tests (use math.isclose) ----
    def test_modeling_u2_definition(self):
        """Check u2 = d - v1."""
        if any(x is None for x in [self.u2, self.d, self.v1]):
            return (False, "SKIP: missing u2, d, or v1")
        expected = self.d - self.v1
        if math.isclose(self.u2, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"u2 equals d - v1: {self.u2} ≈ {expected}")
        return (False, f"u2 mismatch: got {self.u2}, expected d - v1 = {expected}")

    def test_modeling_sign_convention_v1(self):
        """v1 should be positive (real image from first lens)."""
        if self.v1 is None:
            return (False, "SKIP: v1 missing")
        if self.v1 > 0:
            return (True, f"v1 = {self.v1} > 0, consistent with real image")
        return (False, f"v1 = {self.v1} is not positive (real image expected)")

    def test_modeling_sign_convention_v2(self):
        """v2 should be negative (virtual image from second lens since u2 < f2)."""
        if any(x is None for x in [self.v2, self.u2, self.f2]):
            return (False, "SKIP: missing v2, u2, or f2")
        if self.u2 < self.f2:
            if self.v2 < 0:
                return (True, f"v2 = {self.v2} < 0, virtual image consistent with u2 < f2")
            else:
                return (False, f"v2 = {self.v2} should be negative because u2={self.u2} < f2={self.f2}")
        else:
            if self.v2 > 0:
                return (True, f"v2 = {self.v2} > 0, real image consistent with u2 > f2")
            else:
                return (False, f"v2 = {self.v2} should be positive because u2={self.u2} > f2={self.f2}")

    # ---- Monotonicity tests (perturb an input, recompute output, compare to baseline) ----
    def test_monotonicity_u1_v1(self):
        """Increasing u1 (while u1 > f1) decreases v1. Perturb u1 by +1% and recompute v1."""
        if any(x is None for x in [self.u1, self.f1, self.v1]):
            return (False, "SKIP: missing u1, f1, or v1")
        if self.u1 <= self.f1:
            return (True, "SKIP: u1 not > f1, monotonicity condition not applicable")
        # Perturb input: increase u1 by 1%
        u1_perturbed = self.u1 * 1.01
        # Recompute v1 using thin lens equation exactly
        v1_perturbed = 1.0 / (1.0/self.f1 - 1.0/u1_perturbed)
        # Compare perturbed output to baseline output
        if v1_perturbed < self.v1:
            return (True, f"Increasing u1 by 1% (from {self.u1:.4f} to {u1_perturbed:.4f}) decreased v1 from {self.v1:.4f} to {v1_perturbed:.4f}, verified monotonic decrease.")
        else:
            return (False, f"Increasing u1 by 1% did not decrease v1: baseline v1={self.v1:.4f}, perturbed v1={v1_perturbed:.4f}. Expected monotonic decrease.")

    def test_monotonicity_u2_v2(self):
        """Increasing u2 (while u2 < f2) makes v2 more negative. Perturb u2 by +1% and recompute v2."""
        if any(x is None for x in [self.u2, self.f2, self.v2]):
            return (False, "SKIP: missing u2, f2, or v2")
        if not (0 < self.u2 < self.f2):
            return (True, "SKIP: u2 not in (0, f2), monotonicity condition not applicable")
        # Perturb input: increase u2 by 1%
        u2_perturbed = self.u2 * 1.01
        # Recompute v2 using thin lens equation
        v2_perturbed = 1.0 / (1.0/self.f2 - 1.0/u2_perturbed)
        # Check that v2 becomes more negative (decreases)
        if v2_perturbed < self.v2:  # both negative, lower number means more negative
            return (True, f"Increasing u2 by 1% (from {self.u2:.4f} to {u2_perturbed:.4f}) made v2 more negative (from {self.v2:.4f} to {v2_perturbed:.4f}), verified monotonic decrease.")
        else:
            return (False, f"Increasing u2 by 1% did not make v2 more negative: baseline v2={self.v2:.4f}, perturbed v2={v2_perturbed:.4f}. Expected monotonic decrease.")

    def test_monotonicity_u1_magnitude_m1(self):
        """Increasing u1 decreases |m1|. Perturb u1 by +1% and recompute m1."""
        if any(x is None for x in [self.u1, self.f1, self.m1]):
            return (False, "SKIP: missing u1, f1, or m1")
        if self.u1 <= self.f1:
            return (True, "SKIP: u1 not > f1, monotonicity condition not applicable")
        # Perturb input: increase u1 by 1%
        u1_perturbed = self.u1 * 1.01
        # Recompute m1 using its formula (m1 = -v1/u1, but we can use the relation m1 = -f1/(u1-f1))
        m1_perturbed = -self.f1 / (u1_perturbed - self.f1)
        # Compare absolute magnifications
        old_abs = abs(self.m1)
        new_abs = abs(m1_perturbed)
        if new_abs < old_abs:
            return (True, f"Increasing u1 by 1% decreased |m1| from {old_abs:.4f} to {new_abs:.4f}, verified monotonic decrease.")
        else:
            return (False, f"Increasing u1 by 1% did not decrease |m1|: baseline |m1|={old_abs:.4f}, perturbed |m1|={new_abs:.4f}. Expected monotonic decrease.")

    # ---- Answer sanity (weak, combined single test) ----
    def test_answer_sanity_plausibility(self):
        """Weak sanity: check v2 negative and height ratio positive in plausible range."""
        issues = []
        if self.v2 is not None:
            if self.v2 >= 0:
                issues.append(f"v2 = {self.v2} is not negative (virtual image expected).")
            else:
                abs_v2 = abs(self.v2)
                if abs_v2 < 10 or abs_v2 > 500:
                    issues.append(f"v2 magnitude {abs_v2} cm is far outside typical range.")
        if self.height_ratio is not None:
            if not math.isfinite(self.height_ratio):
                issues.append(f"height_ratio is not finite: {self.height_ratio}.")
            elif self.height_ratio <= 0:
                issues.append(f"height_ratio must be positive, got {self.height_ratio}.")
            elif self.height_ratio > 20:
                issues.append(f"height_ratio {self.height_ratio} is unusually large ( >20).")
        if issues:
            return (False, "Issues: " + "; ".join(issues))
        if self.v2 is None and self.height_ratio is None:
            return (False, "SKIP: both v2 and height_ratio missing")
        return (True, "Plausibility checks passed: v2 negative, height_ratio positive and moderate.")

    # ---- Runner ----
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