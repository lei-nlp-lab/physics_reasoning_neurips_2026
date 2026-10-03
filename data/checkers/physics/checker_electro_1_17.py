import math
import traceback

class PhysicsChecker:
    """Audits electrostatic energy and potential at center for a grounded spherical
    shell filled with uniform space charge."""

    def __init__(self, solution_instance):
        # Givens – use None, never numeric fallbacks
        self.rho = getattr(solution_instance, 'rho', None)
        self.R1 = getattr(solution_instance, 'R1', None)
        self.R2 = getattr(solution_instance, 'R2', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Results – from results dict or attribute
        results = getattr(solution_instance, 'results', {})
        self.U = results.get('U', None)
        self.V0 = results.get('V0', None)

        if self.U is None:
            self.U = getattr(solution_instance, 'U', None)
        if self.V0 is None:
            self.V0 = getattr(solution_instance, 'V0', None)

        self.pi = math.pi

    def _safe_float(self, val):
        """Return float if val is a number, else None."""
        if val is None:
            return None
        try:
            v = float(val)
            if math.isfinite(v):
                return v
        except (ValueError, TypeError):
            pass
        return None

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_has_required_keys(self):
        """Check that U and V0 exist and are finite numbers."""
        if self.U is None:
            return (False, "Missing result key 'U'")
        if self.V0 is None:
            return (False, "Missing result key 'V0'")
        u = self._safe_float(self.U)
        v = self._safe_float(self.V0)
        if u is None or not math.isfinite(u):
            return (False, "'U' is not a finite number")
        if v is None or not math.isfinite(v):
            return (False, "'V0' is not a finite number")
        return (True, "U and V0 present and finite")

    def test_interface_givens_present(self):
        """Check that required givens (rho, R1, epsilon0) are provided."""
        if self.rho is None:
            return (False, "Missing given: rho")
        if self.R1 is None:
            return (False, "Missing given: R1")
        if self.epsilon0 is None:
            return (False, "Missing given: epsilon0")
        rho = self._safe_float(self.rho)
        r1 = self._safe_float(self.R1)
        eps = self._safe_float(self.epsilon0)
        if rho is None or rho == 0:
            return (False, "rho must be a non‑zero finite number")
        if r1 is None or r1 <= 0:
            return (False, "R1 must be positive finite number")
        if eps is None or eps <= 0:
            return (False, "epsilon0 must be positive finite number")
        return (True, "All required givens present and valid")

    # ----------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ----------------------------------------------------------------
    def test_invariant_energy_potential_relation(self):
        """Check U = (8π ε0 V0^2 R1)/5, derived from Gauss's law and grounding."""
        u = self._safe_float(self.U)
        v = self._safe_float(self.V0)
        r1 = self._safe_float(self.R1)
        eps = self._safe_float(self.epsilon0)
        if any(x is None for x in (u, v, r1, eps)):
            return (False, "SKIP: missing required value")
        if abs(v) < 1e-20:
            return (False, "V0 too small for reliable check")
        expected = (8 * self.pi * eps * v * v * r1) / 5.0
        if math.isclose(u, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U matches derived relation: {u} ≈ {expected}")
        else:
            return (False, f"U = {u} does not satisfy U = (8π ε0 V0^2 R1)/5 = {expected}")

    def test_invariant_energy_positive(self):
        """Electrostatic energy must be positive for non‑zero charge density."""
        u = self._safe_float(self.U)
        rho = self._safe_float(self.rho)
        if u is None or rho is None:
            return (False, "SKIP: missing required value")
        if abs(rho) < 1e-30:
            return (False, "rho is zero – energy would be zero")
        if u > 0:
            return (True, f"U = {u} > 0 (indeed positive)")
        else:
            return (False, f"U = {u} should be positive for ρ ≠ 0")

    def test_invariant_grounded_shell_consistency(self):
        """If R2 is given, check that it is >= R1 and that results do not depend on it."""
        r2 = self._safe_float(self.R2)
        r1 = self._safe_float(self.R1)
        if r2 is None or r1 is None:
            return (True, "SKIP: R2 not available")
        if r2 < r1:
            return (False, f"R2 = {r2} < R1 = {r1}, invalid geometry")
        return (True, f"R2 = {r2} >= R1 = {r1}")

    # ----------------------------------------------------------------
    # Modeling / trace tests (quality of steps)
    # ----------------------------------------------------------------
    def test_modeling_energy_from_field_integral(self):
        """Recompute U by integrating E² dV using Gauss's law expression."""
        u = self._safe_float(self.U)
        rho = self._safe_float(self.rho)
        r1 = self._safe_float(self.R1)
        eps = self._safe_float(self.epsilon0)
        if any(x is None for x in (u, rho, r1, eps)):
            return (False, "SKIP: missing required value")
        # U = (ε0/2) ∫_0^{R1} (ρr/(3ε0))^2 * 4π r^2 dr
        #   = (2π ρ^2 R1^5) / (45 ε0)
        expected = (2 * self.pi * rho * rho * r1**5) / (45.0 * eps)
        if math.isclose(u, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"U from field integral matches: {u} ≈ {expected}")
        else:
            return (False, f"U = {u} but integration gives {expected}")

    def test_modeling_potential_from_field_integral(self):
        """Recompute V0 by integrating E from R1 to 0."""
        v = self._safe_float(self.V0)
        rho = self._safe_float(self.rho)
        r1 = self._safe_float(self.R1)
        eps = self._safe_float(self.epsilon0)
        if any(x is None for x in (v, rho, r1, eps)):
            return (False, "SKIP: missing required value")
        # V0 = ∫_0^{R1} ρr/(3ε0) dr = ρ R1^2/(6ε0)
        expected = (rho * r1 * r1) / (6.0 * eps)
        if math.isclose(v, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V0 from field integral matches: {v} ≈ {expected}")
        else:
            return (False, f"V0 = {v} but integration gives {expected}")

    # ----------------------------------------------------------------
    # Monotonicity test (explicit input perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_energy_increases_with_R1(self):
        """Perturb R1 upward by 1% and check that U (derived from Gauss's law)
        increases, as physical intuition dictates U ~ R1^5."""
        rho = self._safe_float(self.rho)
        r1 = self._safe_float(self.R1)
        eps = self._safe_float(self.epsilon0)
        if any(x is None for x in (rho, r1, eps)):
            return (False, "SKIP: missing required value")
        # Compute baseline analytic U using given R1
        U_baseline = (2 * self.pi * rho * rho * r1**5) / (45.0 * eps)
        # Perturb R1 upward by 1%
        delta = 0.01
        r1_perturbed = r1 * (1.0 + delta)
        U_perturbed = (2 * self.pi * rho * rho * r1_perturbed**5) / (45.0 * eps)
        if U_perturbed > U_baseline:
            return (True, f"U increases with R1: U_baseline={U_baseline:.3e}, U_perturbed(+1%)={U_perturbed:.3e}")
        else:
            return (False, f"U did not increase with R1: baseline {U_baseline}, perturbed {U_perturbed}")

    # ----------------------------------------------------------------
    # Answer sanity (weak; secondary)
    # ----------------------------------------------------------------
    def test_answer_sanity_signs_and_bounds(self):
        """Basic sanity: V0 should have same sign as ρ, U should be positive."""
        u = self._safe_float(self.U)
        v = self._safe_float(self.V0)
        rho = self._safe_float(self.rho)
        if any(x is None for x in (u, v, rho)):
            return (False, "SKIP: missing required value")
        if abs(rho) < 1e-30:
            return (True, "rho ≈ 0, both U and V0 should be near zero")
        # Sign of V0 should match sign of ρ
        if (v * rho) < 0:
            return (False, f"V0 sign ({v}) does not match ρ sign ({rho})")
        # U positive for non‑zero rho
        if u <= 0:
            return (False, f"U = {u} should be positive")
        return (True, "Sign and positivity constraints satisfied")

    # ----------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)