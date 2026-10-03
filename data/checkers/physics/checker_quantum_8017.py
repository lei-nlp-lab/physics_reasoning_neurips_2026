import math
import traceback


class PhysicsChecker:
    """
    Robust physics checker for the variational ground state problem of
    H = -hbar^2/(2m) d^2/dx^2 + b x^4 with trial function phi(x) = A exp(-lambda^2 x^2).

    Performs:
        - interface checks (correct output keys)
        - theorem-based invariant checks (positivity, normalization)
        - modeling checks (energy formula consistency, derivative condition)
        - monotonicity checks (energy increases away from optimum)
        - one weak final-answer sanity check
    """

    def __init__(self, solution_instance):
        # ---------- extract givens ----------
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.hbar = givens.get('hbar', None)
        self.m = givens.get('m', None)
        self.b = givens.get('b', None)

        # ---------- extract results ----------
        results = getattr(solution_instance, 'results', {}) or {}
        self.lam_opt = results.get('lam_opt', None)
        self.E_min = results.get('E_min', None)

        # ---------- extract explanation steps (optional) ----------
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # ---------- derived quantities (computed from givens, not hardcoded) ----------
        self._derived_E_formula = None
        self._derived_lam_formula = None
        if all(x is not None for x in (self.hbar, self.m, self.b)):
            # lam_opt = ( (3*b*m)/(4*hbar^2) )^(1/6)
            self._derived_lam_formula = (
                (3 * self.b * self.m) / (4 * self.hbar**2)
            ) ** (1/6)
            # E_min = (3/4)*(3/4)^(1/3) * (b*hbar^4/m^2)^(1/3)
            self._derived_E_formula = (
                (3/4) * (3/4) ** (1/3) * (self.b * self.hbar**4 / self.m**2) ** (1/3)
            )

        # Energy as a function of lambda: E(λ) = 0.5*(ħ²λ²/m + 3b/(8λ⁴))
        self._E_of_lambda = None
        if all(x is not None for x in (self.hbar, self.m, self.b)):
            def E_lambda(lam):
                return 0.5 * (self.hbar**2 * lam**2 / self.m + 3 * self.b / (8 * lam**4))
            self._E_of_lambda = E_lambda

    # ===================== Interface Tests =====================
    def test_interface_lam_opt_exists(self):
        """Check that 'lam_opt' key exists in results (not None)."""
        if self.lam_opt is None:
            return False, "lam_opt is missing (None)"
        return True, "lam_opt exists"

    def test_interface_E_min_exists(self):
        """Check that 'E_min' key exists in results (not None)."""
        if self.E_min is None:
            return False, "E_min is missing (None)"
        return True, "E_min exists"

    def test_interface_givens_present(self):
        """Check that all required givens (hbar, m, b) are provided."""
        missing = []
        if self.hbar is None:
            missing.append("hbar")
        if self.m is None:
            missing.append("m")
        if self.b is None:
            missing.append("b")
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All givens present"

    # ===================== Invariant Tests (theorem-based) =====================
    def test_invariant_energy_positive(self):
        """E_min should be positive (b>0, hbar>0, m>0)."""
        if self.E_min is None:
            return False, "SKIP: E_min not available"
        if not math.isfinite(self.E_min):
            return False, "E_min is not finite"
        if self.E_min <= 0:
            return False, f"E_min must be positive, got {self.E_min}"
        return True, f"E_min = {self.E_min} > 0"

    def test_invariant_lam_positive(self):
        """lam_opt should be positive (width of Gaussian)."""
        if self.lam_opt is None:
            return False, "SKIP: lam_opt not available"
        if not math.isfinite(self.lam_opt):
            return False, "lam_opt is not finite"
        if self.lam_opt <= 0:
            return False, f"lam_opt must be positive, got {self.lam_opt}"
        return True, f"lam_opt = {self.lam_opt} > 0"

    def test_invariant_normalization(self):
        """
        Check that normalization condition holds for the provided lam_opt.
        Normalization: ∫|φ|^2 dx = 1.
        With φ = A exp(-λ² x²), A = (2λ²/π)^(1/4).
        Then ∫|φ|^2 dx = A² ∫ exp(-2λ² x²) dx = 1.
        Compute A from lam_opt and verify integral equals 1 (within tolerance).
        """
        if self.lam_opt is None:
            return False, "SKIP: lam_opt not available"
        lam = self.lam_opt
        # A² = sqrt(2λ²/π) = λ * sqrt(2/π)
        A_sq = math.sqrt(2 * lam**2 / math.pi)  # = lam * sqrt(2/π)
        # integral ∫ e^{-2λ² x²} dx = sqrt(π/(2λ²))
        integral = math.sqrt(math.pi / (2 * lam**2))
        norm = A_sq * integral
        if not math.isclose(norm, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Normalization integral = {norm}, expected ~1.0"
        return True, "Normalization condition satisfied"

    # ===================== Modeling / Trace Tests =====================
    def test_modeling_energy_formula(self):
        """
        Check that the computed E_min agrees with the formula:
        E_min = (3/4)*(3/4)^(1/3) * (b hbar^4/m^2)^(1/3)
        using the derived formula from givens (not copied from explanation).
        """
        if self._derived_E_formula is None:
            return False, "SKIP: insufficient givens to compute derived E_min formula"
        if self.E_min is None:
            return False, "SKIP: E_min not available"
        if not math.isclose(self.E_min, self._derived_E_formula, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"E_min mismatch: provided={self.E_min}, "
                f"derived={self._derived_E_formula}"
            )
        return True, "E_min matches derived formula"

    def test_modeling_lam_opt_formula(self):
        """
        Check that lam_opt agrees with the formula:
        lam_opt = ( (3*b*m)/(4*hbar^2) )^(1/6)
        """
        if self._derived_lam_formula is None:
            return False, "SKIP: insufficient givens to compute derived lam_opt formula"
        if self.lam_opt is None:
            return False, "SKIP: lam_opt not available"
        if not math.isclose(self.lam_opt, self._derived_lam_formula, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"lam_opt mismatch: provided={self.lam_opt}, "
                f"derived={self._derived_lam_formula}"
            )
        return True, "lam_opt matches derived formula"

    def test_modeling_derivative_at_optimum(self):
        """
        Check that dE/dλ at λ = lam_opt is zero (within tolerance).
        Analytic derivative: dE/dλ = ħ²λ/m - 3b/(4λ⁵).
        Use the provided lam_opt to evaluate.
        """
        if any(x is None for x in (self.hbar, self.m, self.b, self.lam_opt)):
            return False, "SKIP: missing hbar, m, b, or lam_opt"
        lam = self.lam_opt
        deriv = self.hbar**2 * lam / self.m - (3 * self.b) / (4 * lam**5)
        if not math.isclose(deriv, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Derivative at lam_opt = {deriv}, expected ~0"
        return True, "Derivative condition satisfied"

    def test_modeling_energy_formula_consistency(self):
        """
        Using the local energy formula E(λ) = 0.5*(ħ²λ²/m + 3b/(8λ⁴)),
        compute E at lam_opt and compare to the provided E_min.
        This is a consistency check, not a copy from explanation.
        """
        if self._E_of_lambda is None:
            return False, "SKIP: cannot compute E(λ) – givens missing"
        if self.lam_opt is None or self.E_min is None:
            return False, "SKIP: lam_opt or E_min missing"
        E_computed = self._E_of_lambda(self.lam_opt)
        if not math.isclose(self.E_min, E_computed, rel_tol=1e-2, abs_tol=1e-2):
            return False, (
                f"E_min from formula = {E_computed}, "
                f"provided E_min = {self.E_min}"
            )
        return True, "E_min consistent with local E(λ) formula"

    # ===================== Monotonicity Tests =====================
    def test_monotonicity_energy_increase_larger_lam(self):
        """
        If λ is increased slightly (e.g., +10%) from the optimal lam_opt,
        the energy E(λ) should be larger than E_min.
        Uses the local energy formula derived inside the checker.
        """
        if self._E_of_lambda is None:
            return False, "SKIP: cannot compute E(λ) – givens missing"
        if self.lam_opt is None or self.E_min is None:
            return False, "SKIP: lam_opt or E_min missing"
        lam_increased = self.lam_opt * 1.10
        E_higher = self._E_of_lambda(lam_increased)
        # Allow small tolerance for numerical rounding
        if not math.isclose(E_higher, self.E_min, rel_tol=1e-2, abs_tol=1e-2) and E_higher <= self.E_min:
            return False, f"Energy at +10% λ = {E_higher} should be > E_min = {self.E_min}"
        return True, "Energy increases for λ > λ_opt"

    def test_monotonicity_energy_increase_smaller_lam(self):
        """
        If λ is decreased slightly (e.g., -10%) from the optimal lam_opt,
        the energy E(λ) should be larger than E_min.
        """
        if self._E_of_lambda is None:
            return False, "SKIP: cannot compute E(λ) – givens missing"
        if self.lam_opt is None or self.E_min is None:
            return False, "SKIP: lam_opt or E_min missing"
        lam_decreased = self.lam_opt * 0.90
        E_lower = self._E_of_lambda(lam_decreased)
        if not math.isclose(E_lower, self.E_min, rel_tol=1e-2, abs_tol=1e-2) and E_lower <= self.E_min:
            return False, f"Energy at -10% λ = {E_lower} should be > E_min = {self.E_min}"
        return True, "Energy increases for λ < λ_opt"

    # ===================== Weak Answer Sanity Test (only one) =====================
    def test_answer_sanity_both_finite(self):
        """Weak sanity: both lam_opt and E_min must be finite and not NaN."""
        checks = []
        if self.lam_opt is None:
            checks.append("lam_opt missing")
        else:
            if not math.isfinite(self.lam_opt):
                checks.append("lam_opt not finite")
        if self.E_min is None:
            checks.append("E_min missing")
        else:
            if not math.isfinite(self.E_min):
                checks.append("E_min not finite")
        if checks:
            return False, "; ".join(checks)
        return True, "Both lam_opt and E_min are finite"

    # ===================== Test Runner =====================
    def run_all_tests(self):
        """Discover and run all test_* methods, return (success, logs)."""
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
        return all_pass, logs