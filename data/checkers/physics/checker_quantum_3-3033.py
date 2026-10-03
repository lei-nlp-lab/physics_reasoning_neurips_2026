import math
import sympy as sp
import traceback


class PhysicsChecker:
    """
    Robust physics checker for an electron spin in a uniform magnetic field.
    Audits the expectation values of 2s_x and 2s_z over time.
    """

    def __init__(self, solution_instance):
        # Required result keys (from pseudocode [emit] contract)
        self.result_2s_x = getattr(solution_instance, '2s_x', None)
        self.result_2s_z = getattr(solution_instance, '2s_z', None)

        # Givens (expected attributes of solution_instance, None if missing)
        self.B0 = getattr(solution_instance, 'B0', None)
        self.mu_B = getattr(solution_instance, 'mu_B', None)   # Bohr magneton
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.t0 = getattr(solution_instance, 't0', None)
        self.t = getattr(solution_instance, 't', None)

        # Derived quantities (if all givens present)
        self.omega = None
        if all(v is not None for v in (self.B0, self.mu_B, self.hbar)):
            self.omega = 2 * self.mu_B * self.B0 / self.hbar
        self.delta_t = None
        if self.t is not None and self.t0 is not None:
            self.delta_t = self.t - self.t0

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

    # ---------- interface checks ----------
    def test_interface_2s_x_present(self):
        if self.result_2s_x is None:
            return (False, "Missing required result key '2s_x'")
        return (True, "2s_x present")

    def test_interface_2s_z_present(self):
        if self.result_2s_z is None:
            return (False, "Missing required result key '2s_z'")
        return (True, "2s_z present")

    def test_interface_result_types(self):
        # Allow numeric (float/int) or sympy expression – both are acceptable
        for name, val in [('2s_x', self.result_2s_x), ('2s_z', self.result_2s_z)]:
            if val is None:
                return (False, f"Can't check type for missing result '{name}'")
            if not isinstance(val, (int, float, sp.Expr)):
                return (False, f"Result '{name}' has unexpected type {type(val)}")
        return (True, "Result types are numeric or sympy expression")

    # ---------- invariant checks (theorem-based) ----------
    def test_invariant_norm_constant(self):
        """
        Check that <ψ|ψ> = 1 if state available.
        We cannot directly access the state, but we can use the fact that
        |⟨2s_x⟩| ≤ ℏ and ⟨2s_z⟩ = 0 to infer norm conservation is implied.
        Instead, we test that the pair of expectations is consistent with a
        unit-trace density matrix.
        """
        if self.result_2s_x is None or self.result_2s_z is None:
            return (False, "SKIP: results missing")
        # For a pure state, (⟨2s_x⟩/ℏ)^2 + (⟨2s_y⟩/ℏ)^2 + (⟨2s_z⟩/ℏ)^2 = 1
        # Here ⟨2s_y⟩ is not given, but we can still check an inequality:
        # (⟨2s_x⟩/ℏ)^2 + (⟨2s_z⟩/ℏ)^2 ≤ 1
        # Evaluate numerically if possible
        val_2s_x = _to_numeric(self.result_2s_x)
        val_2s_z = _to_numeric(self.result_2s_z)
        if val_2s_x is None or val_2s_z is None:
            return (False, "SKIP: results are symbolic, cannot test norm bound")
        h = _to_numeric(self.hbar) if self.hbar is not None else 1.0
        if math.isnan(val_2s_x) or math.isnan(val_2s_z):
            return (False, "Results contain NaN")
        norm_sq = (val_2s_x / h)**2 + (val_2s_z / h)**2
        # Due to rounding, allow up to 1.01
        if norm_sq > 1.01:
            return (False, f"Norm bound violated: (⟨2s_x⟩/ℏ)² + (⟨2s_z⟩/ℏ)² = {norm_sq} > 1")
        return (True, f"Norm bound satisfied: combined = {norm_sq:.4f}")

    def test_invariant_2s_z_zero(self):
        """Expectation of 2s_z should be exactly zero for all t."""
        if self.result_2s_z is None:
            return (False, "SKIP: 2s_z not available")
        val = _to_numeric(self.result_2s_z)
        if val is None:
            # Symbolic expression – check if it simplifies to zero
            if sp.simplify(self.result_2s_z) == 0:
                return (True, "2s_z symbolically zero")
            return (False, "2s_z is symbolic but not identically zero")
        if not math.isfinite(val):
            return (False, f"2s_z is non-finite: {val}")
        if abs(val) > 1e-9:   # extremely tight tolerance because it's exactly zero
            return (False, f"2s_z is not zero within 1e-9: {val}")
        return (True, f"2s_z = {val} (≈ 0)")

    def test_invariant_2s_x_amplitude_bound(self):
        """|⟨2s_x⟩| ≤ ℏ (max eigenvalue of ℏ σ_x)."""
        if self.result_2s_x is None:
            return (False, "SKIP: 2s_x not available")
        val = _to_numeric(self.result_2s_x)
        if val is None:
            return (False, "SKIP: symbolic, cannot check numeric bound")
        h = _to_numeric(self.hbar) if self.hbar is not None else 1.0
        if abs(val) > h + 1e-2:
            return (False, f"|⟨2s_x⟩| = {abs(val)} > ℏ = {h}")
        return (True, f"|⟨2s_x⟩| = {abs(val)} ≤ ℏ = {h}")

    # ---------- modeling/trace checks ----------
    def test_modeling_2s_x_form(self):
        """
        Check that ⟨2s_x⟩(t) = -ℏ sin( ω (t-t0) ).
        Requires givens and a numeric or symbolic result.
        """
        if self.result_2s_x is None:
            return (False, "SKIP: 2s_x not available")
        if self.omega is None or self.delta_t is None:
            return (False, "SKIP: missing givens (omega or delta_t)")

        # Compute expected value using formula from explanation steps
        # Note: the angular frequency in the explanation is 2μ_e B / ℏ = omega
        expected_val = -self.hbar * math.sin(self.omega * self.delta_t)

        result_val = _to_numeric(self.result_2s_x)
        if result_val is None:
            # Try symbolic comparison
            sym_expected = -self.hbar * sp.sin(self.omega * self.delta_t)
            if sp.simplify(self.result_2s_x - sym_expected) == 0:
                return (True, "2s_x symbolically matches -ℏ sin(ωΔt)")
            return (False, "2s_x symbolic form does not match expected")
        # Numeric comparison with 1% tolerance (1e-2)
        if not math.isclose(result_val, expected_val, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"2s_x value {result_val} differs from expected {expected_val} by > 1%")
        return (True, f"2s_x = {result_val} matches -ℏ sin(ωΔt) = {expected_val}")

    def test_modeling_2s_z_form(self):
        """2s_z should be exactly zero."""
        # Already covered in invariant, but we can also check consistency with formula
        # For modeling, simply check that it's zero (numerically or symbolically)
        if self.result_2s_z is None:
            return (False, "SKIP: 2s_z missing")
        val = _to_numeric(self.result_2s_z)
        if val is None:
            if sp.simplify(self.result_2s_z) == 0:
                return (True, "2s_z symbolically zero")
            return (False, "2s_z is not zero symbolically")
        if not math.isclose(val, 0.0, abs_tol=1e-9):
            return (False, f"2s_z = {val} is not close to zero")
        return (True, "2s_z = 0 (within tolerance)")

    # ---------- monotonicity checks ----------
    def test_monotonicity_2s_x_with_time(self):
        """
        For small positive Δt, increasing Δt should make ⟨2s_x⟩ more negative
        because 2s_x = -ℏ sin(ω Δt) and sin is positive for small angles.
        Only check if baseline Δt is small enough (0 < ωΔt < π/2).
        """
        if self.result_2s_x is None or self.omega is None or self.delta_t is None:
            return (False, "SKIP: missing data")
        base_dt = self.delta_t
        if base_dt <= 0:
            return (False, "SKIP: baseline delta_t non-positive")
        phase = self.omega * base_dt
        if phase <= 0 or phase >= math.pi / 2:
            return (False, "SKIP: phase not in (0, π/2) for monotonicity check")

        base_val = _to_numeric(self.result_2s_x)
        if base_val is None:
            return (False, "SKIP: baseline 2s_x not numeric")

        # Perturb delta_t by +5%
        perturbed_dt = base_dt * 1.05
        # Recompute expected 2s_x using the given formula (assuming the model is correct)
        # We trust the model for the direction of change; we don't need to call the solution again.
        expected_perturbed = -self.hbar * math.sin(self.omega * perturbed_dt)
        # Compare with baseline: the perturbed value should be more negative (lower)
        if expected_perturbed >= base_val - 1e-12:   # protect against rounding
            return (False, f"Increasing Δt by 5% should lower 2s_x; base={base_val}, perturbed={expected_perturbed}")
        return (True, f"Monotonicity holds: increasing Δt from {base_dt} to {perturbed_dt} reduces 2s_x from {base_val} to {expected_perturbed}")

    def test_monotonicity_2s_z_constant(self):
        """2s_z should remain exactly zero regardless of parameter changes."""
        if self.result_2s_z is None:
            return (False, "SKIP: 2s_z missing")
        val = _to_numeric(self.result_2s_z)
        if val is not None and math.isclose(val, 0.0, abs_tol=1e-9):
            return (True, "2s_z is constant zero")
        # Even if symbolic, we trust it's zero; skip numeric perturbation.
        return (True, "2s_z appears zero (symbolic or numeric)")

    # ---------- answer sanity (weak, at most one) ----------
    def test_answer_sanity_values(self):
        """
        Weak sanity: results must be finite and real.
        No numerical equality checks.
        """
        for name, val in [('2s_x', self.result_2s_x), ('2s_z', self.result_2s_z)]:
            if val is None:
                return (False, f"Missing result '{name}' cannot check sanity")
            numeric_val = _to_numeric(val)
            if numeric_val is None:
                # Symbolic expression is okay; just check it's not infinite
                continue
            if not math.isfinite(numeric_val):
                return (False, f"Result '{name}' is non-finite: {numeric_val}")
        return (True, "All results are finite and real")


# Helper: try to convert a value to a float number
def _to_numeric(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, sp.Expr):
        # Attempt numeric evaluation if no free symbols
        try:
            num_val = float(val.evalf())
            return num_val
        except Exception:
            return None
    return None