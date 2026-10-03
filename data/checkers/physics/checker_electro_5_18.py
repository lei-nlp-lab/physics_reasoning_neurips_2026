import math
import traceback
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Physics auditor for the electromagnetic field of a charge moving with constant velocity.
    Checks interface, invariants, modeling consistency, and a weak final-answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens
        self.q = getattr(solution_instance, 'q', None)
        self.v = getattr(solution_instance, 'v', None)
        self.c = getattr(solution_instance, 'c', None)
        self.x = getattr(solution_instance, 'x', None)
        self.y = getattr(solution_instance, 'y', None)
        self.z = getattr(solution_instance, 'z', None)
        self.t = getattr(solution_instance, 't', None)

        # Safely extract emitted fields from results
        results = getattr(solution_instance, 'results', {})
        self.E_x = results.get('E_x', None)
        self.B_x = results.get('B_x', None)
        self.E_y = results.get('E_y', None)
        self.B_y = results.get('B_y', None)
        self.E_z = results.get('E_z', None)
        self.B_z = results.get('B_z', None)

        # Compute derived quantities if all required givens are present
        self.beta = None
        self.gamma = None
        self.chi = None
        if all(v is not None for v in [self.v, self.c, self.x, self.y, self.z, self.t]):
            if self.c > 0 and abs(self.v) < self.c:
                self.beta = self.v / self.c
                self.gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
                dx = self.x - self.v * self.t
                self.chi = (self.gamma**2 * dx**2 + self.y**2 + self.z**2) ** 1.5
            # else parameters inconsistent -> keep None

    def _safe_compare(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Safe comparison with None handling."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all six field keys exist and are not None."""
        required = [('E_x', self.E_x), ('B_x', self.B_x),
                    ('E_y', self.E_y), ('B_y', self.B_y),
                    ('E_z', self.E_z), ('B_z', self.B_z)]
        missing = [name for name, val in required if val is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required field keys present.")

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that essential givens (q, v, c, x, y, z, t) are provided."""
        missing = []
        for name in ['q', 'v', 'c', 'x', 'y', 'z', 't']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All givens present.")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, use isclose)
    # ----------------------------------------------------------------------
    def test_invariant_E_dot_B(self) -> Tuple[bool, str]:
        """Check that E·B ≈ 0 (perpendicular fields in vacuum)."""
        if any(v is None for v in [self.E_x, self.E_y, self.E_z, self.B_x, self.B_y, self.B_z]):
            return (False, "SKIP: field components missing")
        dot = self.E_x * self.B_x + self.E_y * self.B_y + self.E_z * self.B_z
        if math.isclose(dot, 0.0, abs_tol=1e-5):
            return (True, f"E·B = {dot} ≈ 0")
        return (False, f"E·B = {dot} not close to 0")

    def test_invariant_B_equals_beta_cross_E(self) -> Tuple[bool, str]:
        """Check B = (v/c) × E componentwise: B_y = -β E_z, B_z = β E_y, B_x = 0."""
        if any(v is None for v in [self.E_y, self.E_z, self.B_x, self.B_y, self.B_z, self.beta]):
            return (False, "SKIP: missing data for B=β×E check")
        # B_x should be 0
        if not math.isclose(self.B_x, 0.0, abs_tol=1e-5):
            return (False, f"B_x = {self.B_x} not 0")
        # B_y = -β E_z
        if not self._safe_compare(self.B_y, -self.beta * self.E_z):
            return (False, f"B_y = {self.B_y} but expected -βE_z = {-self.beta * self.E_z}")
        # B_z = β E_y
        if not self._safe_compare(self.B_z, self.beta * self.E_y):
            return (False, f"B_z = {self.B_z} but expected βE_y = {self.beta * self.E_y}")
        return (True, "B = β×E holds within tolerance.")

    def test_invariant_gamma_beta_relation(self) -> Tuple[bool, str]:
        """Check that gamma = 1/√(1-β²) is consistent with v and c."""
        if self.v is None or self.c is None:
            return (False, "SKIP: v or c missing")
        if self.c <= 0 or abs(self.v) >= self.c:
            return (False, f"Invalid v/c: |v|={self.v} >= c={self.c}")
        beta_expected = self.v / self.c
        gamma_expected = 1.0 / math.sqrt(1.0 - beta_expected**2)
        if self.gamma is None:
            return (False, "SKIP: gamma not computed (givens missing?)")
        if not math.isclose(self.gamma, gamma_expected, rel_tol=1e-5):
            return (False, f"gamma = {self.gamma} but expected {gamma_expected}")
        return (True, "γ = 1/√(1-β²) holds.")

    # ----------------------------------------------------------------------
    # Modeling/Trace checks (use isclose)
    # ----------------------------------------------------------------------
    def test_modeling_field_consistency(self) -> Tuple[bool, str]:
        """
        Derive expected field components from given parameters and compare with reported fields.
        This re-derives the formulas inside the checker (allowed by rule 9).
        """
        if any(v is None for v in [self.q, self.gamma, self.chi, self.x, self.v, self.t, self.y, self.z]):
            return (False, "SKIP: missing givens to recompute fields")
        dx = self.x - self.v * self.t
        q_gamma_over_chi = self.q * self.gamma / self.chi

        expected = {
            'E_x': q_gamma_over_chi * dx,
            'B_x': 0.0,
            'E_y': q_gamma_over_chi * self.y,
            'B_y': -q_gamma_over_chi * self.beta * self.z,
            'E_z': q_gamma_over_chi * self.z,
            'B_z': q_gamma_over_chi * self.beta * self.y,
        }

        failures = []
        for name, expected_val in expected.items():
            actual = getattr(self, name)
            if actual is None:
                failures.append(f"{name} missing")
                continue
            if not self._safe_compare(actual, expected_val):
                failures.append(f"{name}: actual {actual}, expected {expected_val}")
        if failures:
            return (False, "; ".join(failures))
        return (True, "All field components match the re-derived formulas.")

    def test_modeling_chi_consistency(self) -> Tuple[bool, str]:
        """
        Check that chi computed from coordinate formula matches chi derived from E_x and q,γ,dx.
        """
        if any(v is None for v in [self.q, self.gamma, self.chi, self.E_x, self.x, self.v, self.t]):
            return (False, "SKIP: missing data for chi consistency")
        dx = self.x - self.v * self.t
        if abs(dx) < 1e-12 and abs(self.E_x) < 1e-12:
            # E_x near zero -> avoid division by zero
            # Use E_y instead
            if self.y is not None and self.E_y is not None and abs(self.y) > 1e-12:
                chi_from_field = self.q * self.gamma * self.y / self.E_y
            elif self.z is not None and self.E_z is not None and abs(self.z) > 1e-12:
                chi_from_field = self.q * self.gamma * self.z / self.E_z
            else:
                return (False, "Cannot compute chi from fields (all denominators zero)")
        else:
            chi_from_field = self.q * self.gamma * dx / self.E_x

        if not math.isclose(chi_from_field, self.chi, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"chi from field = {chi_from_field} but expected {self.chi}")
        return (True, "chi from coordinate formula consistent with field.")

    # ----------------------------------------------------------------------
    # Monotonicity check (explicit perturbation of an input)
    # ----------------------------------------------------------------------
    def test_monotonicity_E_x_decreases_with_distance(self) -> Tuple[bool, str]:
        """
        Perturb the x-coordinate (increase dx) and verify that |E_x| decreases.
        We recompute the field using the same formulas (inside checker) for the perturbed point.
        """
        # Need base data
        if any(v is None for v in [self.q, self.gamma, self.chi, self.x, self.v, self.t, self.E_x]):
            return (False, "SKIP: missing data for monotonicity test")
        dx = self.x - self.v * self.t
        if abs(dx) < 1e-12:
            return (False, "SKIP: dx too small, cannot perturb meaningfully")
        # Ensure E_x is not zero to avoid sign flips; we care about magnitude
        if abs(self.E_x) < 1e-12:
            return (False, "SKIP: E_x too small")

        # Perturb: increase dx by 10% (but keep dy, dz same)
        delta = 0.1 * dx if dx != 0 else 1.0  # in case dx zero already skipped
        new_dx = dx + delta
        # Compute new chi and new E_x magnitude
        # y, z unchanged
        new_chi = (self.gamma**2 * new_dx**2 + self.y**2 + self.z**2) ** 1.5
        new_E_x_mag = abs(self.q * self.gamma * new_dx / new_chi)
        old_E_x_mag = abs(self.E_x)

        # Monotonicity: if we move further away (increase |dx|), magnitude should decrease.
        # But if dx is negative and we increase dx (make it less negative), |dx| decreases,
        # so magnitude should increase. So we must check sign of delta relative to dx sign.
        if dx > 0:
            # increasing positive dx -> magnitude should decrease
            expected_decrease = new_E_x_mag < old_E_x_mag
        else:
            # dx negative, increasing dx (adding positive delta) makes dx less negative,
            # so |dx| decreases -> magnitude should increase
            expected_decrease = new_E_x_mag > old_E_x_mag

        if expected_decrease:
            return (True, f"|E_x| changes monotonically: original {old_E_x_mag}, perturbed {new_E_x_mag}")
        else:
            return (False, f"|E_x| did not follow monotonic trend: original {old_E_x_mag}, perturbed {new_E_x_mag}")

    # ----------------------------------------------------------------------
    # Weak final-answer sanity (no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_and_sign(self) -> Tuple[bool, str]:
        """
        Weak check: fields are finite, non-NaN, and signs match expectations:
        E_x has same sign as (x-vt); E_y same sign as y; E_z same sign as z;
        B_x = 0; B_y opposite sign of z; B_z same sign as y.
        """
        if any(v is None for v in [self.E_x, self.B_x, self.E_y, self.B_y, self.E_z, self.B_z]):
            return (False, "SKIP: fields missing")
        # Finiteness
        for name, val in [('E_x',self.E_x),('B_x',self.B_x),('E_y',self.E_y),
                          ('B_y',self.B_y),('E_z',self.E_z),('B_z',self.B_z)]:
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")

        # Sign checks (only require correct sign, not magnitude)
        # Need givens for signs
        if all(v is not None for v in [self.x, self.v, self.t, self.y, self.z]):
            dx = self.x - self.v * self.t
            if (dx > 0 and self.E_x < 0) or (dx < 0 and self.E_x > 0):
                return (False, f"E_x sign mismatch: dx={dx}, E_x={self.E_x}")
            if (self.y > 0 and self.E_y < 0) or (self.y < 0 and self.E_y > 0):
                return (False, f"E_y sign mismatch: y={self.y}, E_y={self.E_y}")
            if (self.z > 0 and self.E_z < 0) or (self.z < 0 and self.E_z > 0):
                return (False, f"E_z sign mismatch: z={self.z}, E_z={self.E_z}")
            # B signs
            if (self.z > 0 and self.B_y > 0) or (self.z < 0 and self.B_y < 0):
                return (False, f"B_y sign mismatch: z={self.z}, B_y={self.B_y}")
            if (self.y > 0 and self.B_z < 0) or (self.y < 0 and self.B_z > 0):
                return (False, f"B_z sign mismatch: y={self.y}, B_z={self.B_z}")
        # else: coordinate signs not available -> skip sign checks (still finite OK)
        # Note: we do NOT check B_x magnitude with isclose.
        return (True, "Fields are finite and signs consistent.")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)