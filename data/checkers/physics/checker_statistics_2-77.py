import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits neutrino temperature calculations for lab, closure, and standard big bang.
    """
    def __init__(self, solution_instance):
        # Extract results safely (use None if missing)
        self.T_lab = None
        self.T_max = None
        self.T_pred = None
        # Try getting from solution_instance attributes
        self.T_lab = getattr(solution_instance, 'T_nu_lab', None)
        self.T_max = getattr(solution_instance, 'T_nu_max', None)
        self.T_pred = getattr(solution_instance, 'T_nu_pred', None)

        # If not found, try from .results dict (common pattern)
        if self.T_lab is None and hasattr(solution_instance, 'results'):
            self.T_lab = solution_instance.results.get('T_nu_lab', None)
            self.T_max = solution_instance.results.get('T_nu_max', None)
            self.T_pred = solution_instance.results.get('T_nu_pred', None)

        # Extract given inputs if present (e.g., from .given or direct attributes)
        self.T_gamma = getattr(solution_instance, 'T_gamma', None)
        self.rho_ratio = getattr(solution_instance, 'rho_ratio', None)
        # Also try from .given dict
        if self.T_gamma is None and hasattr(solution_instance, 'given'):
            self.T_gamma = solution_instance.given.get('T_gamma', None)
            self.rho_ratio = solution_instance.given.get('rho_ratio', None)

        # Physical constants used in derivations
        self.factor_nu = 21/8  # (7/8)*(6/2) for three neutrino flavors

    def _is_numeric(self, x):
        return x is not None and isinstance(x, (int, float))

    # ---------- Interface checks ----------
    def test_interface_required_keys(self):
        """All three result keys must exist and be numeric."""
        if not self._is_numeric(self.T_lab):
            return (False, "T_nu_lab missing or non‑numeric")
        if not self._is_numeric(self.T_max):
            return (False, "T_nu_max missing or non‑numeric")
        if not self._is_numeric(self.T_pred):
            return (False, "T_nu_pred missing or non‑numeric")
        return (True, "All required result keys present")

    def test_interface_positive_temperatures(self):
        """All temperatures must be positive (finite)."""
        if any(v is None or v <= 0 for v in (self.T_lab, self.T_max, self.T_pred)):
            return (False, "One or more temperatures ≤ 0 or missing")
        return (True, "All temperatures positive")

    # ---------- Invariant checks (use isclose) ----------
    def test_invariant_standard_model(self):
        """Standard big‑bang prediction: T_nu = (4/11)^(1/3) T_gamma."""
        if self.T_pred is None or self.T_gamma is None:
            return (False, "SKIP: T_pred or T_gamma missing")
        expected = (4/11)**(1/3) * self.T_gamma
        if math.isclose(self.T_pred, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Standard model relation holds: {self.T_pred:.3f} ≈ {expected:.3f} K")
        else:
            return (False, f"Standard model relation fails: expected {expected:.3f}, got {self.T_pred:.3f}")

    def test_invariant_closure_limit(self):
        """Closure limit: T_nu^4 ≤ T_gamma^4 / (factor_nu * rho_ratio)."""
        if self.T_max is None or self.T_gamma is None or self.rho_ratio is None:
            return (False, "SKIP: T_max, T_gamma, or rho_ratio missing")
        factor = self.factor_nu * self.rho_ratio
        bound = self.T_gamma / (factor**0.25)   # correct bound
        # The solution's T_max should be ≤ bound (within numerical tolerance)
        # Use rel_tol=0.1 because it's an inequality, but we also check approximate equality
        # to avoid failing if the solution is slightly below bound
        if self.T_max <= bound * 1.01:
            return (True, f"Closure limit satisfied: T_max={self.T_max:.3f} ≤ {bound:.3f} K")
        else:
            return (False, f"Closure limit violated: T_max={self.T_max:.3f} > bound={bound:.3f} K")

    def test_invariant_lab_limit_magnitude(self):
        """Lab limit should be of order 10^6 K (between 10^5 and 10^7)."""
        if self.T_lab is None:
            return (False, "SKIP: T_lab missing")
        if 1e5 <= self.T_lab <= 1e7:
            return (True, f"Lab limit magnitude plausible: {self.T_lab:.2e} K")
        else:
            return (False, f"Lab limit magnitude improbable: {self.T_lab:.2e} K (expected ~1e6)")

    def test_invariant_temperature_ordering(self):
        """Physical hierarchy: lab > closure > predicted."""
        if any(v is None for v in (self.T_lab, self.T_max, self.T_pred)):
            return (False, "SKIP: missing temperature(s)")
        if self.T_lab > self.T_max and self.T_max > self.T_pred:
            return (True, "Correct ordering: lab limit > closure limit > predicted")
        else:
            return (False, "Ordering violated: expected T_lab > T_max > T_pred")

    # ---------- Modeling checks (relation quality) ----------
    def test_modeling_lab_limit_from_non_detection(self):
        """Lab limit is an upper bound; it must be larger than the predicted temperature."""
        if self.T_lab is None or self.T_pred is None:
            return (False, "SKIP: T_lab or T_pred missing")
        if self.T_lab > self.T_pred:
            return (True, "Lab limit consistent as upper bound (> predicted)")
        else:
            return (False, "Lab limit should be > predicted temperature")

    def test_modeling_closure_using_energy_density(self):
        """Check that the closure limit is derived from energy density comparison."""
        # Not a numerical check; just require that T_max is not smaller than predicted (should be larger)
        if self.T_max is None or self.T_pred is None:
            return (False, "SKIP: T_max or T_pred missing")
        if self.T_max > self.T_pred:
            return (True, "Closure limit larger than predicted (expected)")
        else:
            return (False, "Closure limit should exceed predicted temperature")

    # ---------- Monotonicity check (with perturbation, if possible) ----------
    def test_monotonicity_closure_with_T_gamma(self):
        """
        If T_gamma were increased by 10%, the closure bound would scale linearly.
        Check that the solution's T_max is still ≤ new bound.
        """
        if self.T_max is None or self.T_gamma is None or self.rho_ratio is None:
            return (False, "SKIP: required inputs missing for monotonicity")
        T_gamma_perturbed = self.T_gamma * 1.1
        bound_perturbed = T_gamma_perturbed / ( (self.factor_nu * self.rho_ratio)**0.25 )
        if self.T_max <= bound_perturbed * 1.01:
            return (True, f"Monotonicity holds: T_max ≤ new bound {bound_perturbed:.3f}")
        else:
            return (False, f"Monotonicity violation: T_max > new bound {bound_perturbed:.3f}")

    # ---------- Weak answer sanity (only one allowed) ----------
    def test_answer_sanity_general(self):
        """Weak checks: all temperatures finite, positive, and within realistic order-of-magnitude ranges."""
        if any(v is None for v in (self.T_lab, self.T_max, self.T_pred)):
            return (False, "SKIP: missing temperature(s)")
        if not (math.isfinite(self.T_lab) and math.isfinite(self.T_max) and math.isfinite(self.T_pred)):
            return (False, "Non‑finite temperature encountered")
        # Broad order-of-magnitude ranges
        ok = True
        msg = ""
        if not (1e4 < self.T_lab < 1e8):
            msg += f"T_lab={self.T_lab:e} out of plausible range (1e4-1e8); "
            ok = False
        if not (1 < self.T_max < 200):
            msg += f"T_max={self.T_max:e} out of plausible range (1-200); "
            ok = False
        if not (0.5 < self.T_pred < 5):
            msg += f"T_pred={self.T_pred:e} out of plausible range (0.5-5); "
            ok = False
        return (ok, msg if not ok else "All temperatures within plausible orders of magnitude")

    # ---------- Run all tests ----------
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