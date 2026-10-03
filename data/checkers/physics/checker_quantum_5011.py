import math
import traceback

class PhysicsChecker:
    """
    Auditor for a charged harmonic oscillator ground-state energy shift
    under a constant electric field. Checks physical consistency and 
    modeling quality.
    """
    def __init__(self, solution_instance):
        # Helper to extract from attribute or results dict
        self._sol = solution_instance
        # Givens (expected: m, q, omega, E, hbar)
        self.m = self._get_value('m')
        self.q = self._get_value('q')
        self.omega = self._get_value('omega')
        self.E = self._get_value('E')
        self.hbar = self._get_value('hbar')   # not strictly required for the local checks
        # Results (expected from pseudocode [emit] contract)
        self.x0 = self._get_value('x0')
        self.delta_E0 = self._get_value('delta_E0')
        self.shift_magnitude = self._get_value('shift_magnitude')

    def _get_value(self, key):
        """Try attribute first, then results dict, else None."""
        try:
            val = getattr(self._sol, key, None)
            if val is not None:
                return val
        except Exception:
            pass
        try:
            if hasattr(self._sol, 'results') and isinstance(self._sol.results, dict):
                return self._sol.results.get(key, None)
        except Exception:
            pass
        return None

    # ---------- Interface checks ----------
    def test_interface_results_exist(self):
        """Check that all emitted result keys are present and numeric."""
        failures = []
        for key, val in [('x0', self.x0),
                         ('delta_E0', self.delta_E0),
                         ('shift_magnitude', self.shift_magnitude)]:
            if val is None:
                failures.append(f"{key} is None")
            else:
                if not isinstance(val, (int, float)):
                    failures.append(f"{key} is not numeric (type={type(val).__name__})")
        if failures:
            return (False, "; ".join(failures))
        return (True, "All required result keys present and numeric")

    def test_interface_givens_available(self):
        """Check that at least a minimal set of givens (m, q, omega, E) are available for consistency tests."""
        missing = []
        for key, val in [('m', self.m), ('q', self.q), ('omega', self.omega), ('E', self.E)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}. Checks that depend on these will be skipped.")
        return (True, "All key givens (m, q, omega, E) present")

    # ---------- Invariant checks (theorem‑based, use isclose) ----------
    def test_invariant_delta_E0_sign(self):
        """Delta_E0 must be negative (bound state lowered)."""
        if self.delta_E0 is None:
            return (False, "SKIP: delta_E0 not available")
        if self.delta_E0 >= 0:
            return (False, f"delta_E0 should be negative, got {self.delta_E0}")
        return (True, f"delta_E0 is negative ({self.delta_E0})")

    def test_invariant_shift_magnitude_sign(self):
        """Shift magnitude must be positive (reported as absolute value)."""
        if self.shift_magnitude is None:
            return (False, "SKIP: shift_magnitude not available")
        if self.shift_magnitude <= 0:
            return (False, f"shift_magnitude should be positive, got {self.shift_magnitude}")
        return (True, f"shift_magnitude is positive ({self.shift_magnitude})")

    def test_invariant_delta_E0_magnitude_consistency(self):
        """Check that shift_magnitude = -delta_E0 (within tolerance)."""
        if self.delta_E0 is None or self.shift_magnitude is None:
            return (False, "SKIP: either delta_E0 or shift_magnitude missing")
        if not math.isclose(self.shift_magnitude, -self.delta_E0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"shift_magnitude ({self.shift_magnitude}) not close to -delta_E0 ({-self.delta_E0})")
        return (True, "shift_magnitude = -delta_E0 holds")

    def test_invariant_energy_finite(self):
        """Delta_E0 must be finite (not NaN, not infinite)."""
        if self.delta_E0 is None:
            return (False, "SKIP: delta_E0 not available")
        if math.isnan(self.delta_E0) or math.isinf(self.delta_E0):
            return (False, f"delta_E0 is NaN or infinite ({self.delta_E0})")
        return (True, "delta_E0 is finite")

    # ---------- Modeling/trace checks (using physics relations, isclose allowed) ----------
    def test_modeling_x0_from_force_balance(self):
        """
        Check that x0 agrees with the force‑balance relation: x0 = qE / (m ω^2).
        """
        if any(v is None for v in [self.m, self.q, self.omega, self.E, self.x0]):
            return (False, "SKIP: missing givens or x0 for this check")
        x0_pred = self.q * self.E / (self.m * self.omega**2)
        if not math.isclose(self.x0, x0_pred, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"x0 ({self.x0}) not close to predicted ({x0_pred})")
        return (True, f"x0 consistent with force balance: {self.x0} ~ {x0_pred}")

    def test_modeling_delta_E0_from_x0(self):
        """
        Check that delta_E0 = - 1/2 m ω^2 x0^2.
        """
        if any(v is None for v in [self.m, self.omega, self.x0, self.delta_E0]):
            return (False, "SKIP: missing data for this check")
        delta_pred = -0.5 * self.m * self.omega**2 * self.x0**2
        if not math.isclose(self.delta_E0, delta_pred, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"delta_E0 ({self.delta_E0}) not close to -½ m ω² x0² ({delta_pred})")
        return (True, "delta_E0 consistent with completing‑square relation")

    # ---------- Monotonicity test (perturb input, recompute output) ----------
    def test_monotonicity_shift_magnitude_increases_with_E(self):
        """
        Perturb electric field E by +10% and check that the magnitude of the shift increases.
        Baseline: magnitude = |delta_E0|. Under larger E, the magnitude should become larger.
        We recompute the predicted delta_E0' from the same physics relation.
        """
        if any(v is None for v in [self.m, self.q, self.omega, self.E, self.delta_E0]):
            return (False, "SKIP: missing givens or delta_E0 for this check")
        # Baseline magnitude
        mag0 = -self.delta_E0  # positive because delta_E0 negative
        # Perturb electric field upward by 10%
        E_pert = self.E * 1.1
        # Predicted delta_E0' from physics (derived inside checker)
        delta_E0_pert = - (self.q**2 * E_pert**2) / (2 * self.m * self.omega**2)
        mag_pert = -delta_E0_pert
        # Check that perturbed magnitude is larger than baseline
        if mag_pert <= mag0:
            return (False, f"Predicted magnitude after E increase ({mag_pert}) not larger than baseline ({mag0})")
        return (True, f"Shift magnitude increases with E: baseline {mag0}, perturbed {mag_pert} (> baseline)")

    # ---------- Answer sanity (very weak) ----------
    def test_answer_sanity_weak(self):
        """
        Weak sanity: shift magnitude must be positive, finite, and not unreasonably large.
        """
        if self.shift_magnitude is None:
            return (False, "SKIP: shift_magnitude not available")
        if not isinstance(self.shift_magnitude, (int, float)):
            return (False, "shift_magnitude not numeric")
        if math.isnan(self.shift_magnitude) or math.isinf(self.shift_magnitude):
            return (False, "shift_magnitude is NaN or infinite")
        if self.shift_magnitude <= 0:
            return (False, "shift_magnitude not positive")
        if self.shift_magnitude > 1e100:
            return (False, "shift_magnitude unreasonably large (>1e100)")
        return (True, "shift_magnitude passes weak sanity (positive, finite, modest)")

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)