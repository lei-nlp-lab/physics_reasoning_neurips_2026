import math
import traceback

class PhysicsChecker:
    """
    Audits a Carnot engine work ratio calculation.
    Tests interface, invariant, modeling, monotonicity, and a weak final answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.W_mono = results.get('W_mono', None)
        self.W_diat = results.get('W_diat', None)
        self.ratio = results.get('ratio', None)

        # Givens (may be missing)
        self.gamma_mono = getattr(solution_instance, 'gamma_mono', None)
        self.gamma_diat = getattr(solution_instance, 'gamma_diat', None)
        self.T_h = getattr(solution_instance, 'T_h', None)
        self.T_c = getattr(solution_instance, 'T_c', None)
        self.V_A = getattr(solution_instance, 'V_A', None)
        self.V_C = getattr(solution_instance, 'V_C', None)
        self.r_T = getattr(solution_instance, 'r_T', None)
        self.r_VC = getattr(solution_instance, 'r_VC', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_results_keys(self):
        """All three result keys exist and are numeric."""
        missing = [k for k in ('W_mono', 'W_diat', 'ratio') if getattr(self, k, None) is None]
        invalid = [k for k in ('W_mono', 'W_diat', 'ratio') 
                   if getattr(self, k, None) is not None and not isinstance(getattr(self,k), (int, float))]
        if missing:
            return (False, f"Missing keys: {missing}")
        if invalid:
            return (False, f"Non‑numeric keys: {invalid}")
        return (True, "All required result keys present and numeric.")

    # ------------------------------------------------------------------
    # Invariant checks (physics theorems)
    # ------------------------------------------------------------------
    def test_invariant_work_positive(self):
        if self.W_mono is None:
            return (False, "SKIP: W_mono not available")
        if self.W_mono > 0:
            return (True, f"W_mono = {self.W_mono} > 0")
        return (False, f"W_mono = {self.W_mono} <= 0")

    def test_invariant_work_diat_positive(self):
        if self.W_diat is None:
            return (False, "SKIP: W_diat not available")
        if self.W_diat > 0:
            return (True, f"W_diat = {self.W_diat} > 0")
        return (False, f"W_diat = {self.W_diat} <= 0")

    def test_invariant_ratio_positive(self):
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if self.ratio > 0:
            return (True, f"ratio = {self.ratio} > 0")
        return (False, f"ratio = {self.ratio} <= 0")

    def test_invariant_ratio_below_one(self):
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if self.ratio < 1.0:
            return (True, f"ratio = {self.ratio} < 1")
        return (False, f"ratio = {self.ratio} >= 1 (diatomic gas should give less net work)")

    def test_invariant_work_diat_less_than_mono(self):
        if self.W_mono is None or self.W_diat is None:
            return (False, "SKIP: one of the works missing")
        if self.W_diat < self.W_mono:
            return (True, f"W_diat ({self.W_diat}) < W_mono ({self.W_mono})")
        return (False, f"W_diat ({self.W_diat}) >= W_mono ({self.W_mono})")

    # ------------------------------------------------------------------
    # Modeling / trace quality checks
    # ------------------------------------------------------------------
    def test_modeling_ratio_derived(self):
        """Re‑derive the ratio from givens and compare with output (lenient tolerance)."""
        # Ensure necessary givens are available
        if self.gamma_mono is None or self.gamma_diat is None:
            return (False, "SKIP: gamma values missing")
        # Determine r_T
        r_T = self.r_T
        if r_T is None:
            if self.T_h is not None and self.T_c is not None and self.T_h != 0:
                r_T = self.T_c / self.T_h
            else:
                return (False, "SKIP: r_T (or T_h,T_c) missing")
        # Determine r_VC
        r_VC = self.r_VC
        if r_VC is None:
            if self.V_A is not None and self.V_C is not None and self.V_A != 0:
                r_VC = self.V_C / self.V_A
            else:
                return (False, "SKIP: r_VC (or V_A,V_C) missing")

        try:
            ln_rV_mono = math.log(r_VC) + (1.0 / (self.gamma_mono - 1)) * math.log(r_T)
            ln_rV_diat = math.log(r_VC) + (1.0 / (self.gamma_diat - 1)) * math.log(r_T)
            expected_ratio = ln_rV_diat / ln_rV_mono
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Derivation error: {e}")

        if self.ratio is None:
            return (False, "SKIP: output ratio missing")
        if math.isclose(self.ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derived {expected_ratio:.5f} matches output {self.ratio:.5f}")
        return (False, f"Output {self.ratio:.5f} differs from derived {expected_ratio:.5f}")

    def test_modeling_ratio_consistent_with_works(self):
        """Check that the output ratio equals W_diat/W_mono (internal consistency)."""
        if any(x is None for x in (self.W_mono, self.W_diat, self.ratio)):
            return (False, "SKIP: one of the work values or ratio missing")
        # Allow absolute tolerance for very small numbers
        if math.isclose(self.ratio, self.W_diat / self.W_mono, rel_tol=1e-9, abs_tol=1e-12):
            return (True, f"ratio {self.ratio} equals W_diat/W_mono ({self.W_diat}/{self.W_mono})")
        else:
            return (False, f"ratio {self.ratio} does not match W_diat/W_mono = {self.W_diat}/{self.W_mono}")

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_ratio_decreases_with_gamma_diat(self):
        """
        When gamma_diat increases (while gamma_mono fixed), the work ratio W_diat/W_mono 
        should decrease because ln_rV_diat decreases.
        Perturb gamma_diat upward by a small amount and check the new ratio < baseline.
        """
        # Need givens to compute baseline ratio
        if self.gamma_mono is None or self.gamma_diat is None:
            return (False, "SKIP: gamma values missing")
        r_T = self.r_T
        if r_T is None:
            if self.T_h is not None and self.T_c is not None and self.T_h != 0:
                r_T = self.T_c / self.T_h
            else:
                return (False, "SKIP: r_T missing")
        r_VC = self.r_VC
        if r_VC is None:
            if self.V_A is not None and self.V_C is not None and self.V_A != 0:
                r_VC = self.V_C / self.V_A
            else:
                return (False, "SKIP: r_VC missing")

        try:
            ln_rV_mono = math.log(r_VC) + (1.0 / (self.gamma_mono - 1)) * math.log(r_T)
            # Baseline with original gamma_diat
            ln_rV_diat_base = math.log(r_VC) + (1.0 / (self.gamma_diat - 1)) * math.log(r_T)
            base_ratio = ln_rV_diat_base / ln_rV_mono

            # Perturb gamma_diat upward by 10% (at least 0.1)
            delta = max(0.1, 0.1 * self.gamma_diat)
            gamma_diat_pert = self.gamma_diat + delta
            ln_rV_diat_pert = math.log(r_VC) + (1.0 / (gamma_diat_pert - 1)) * math.log(r_T)
            perturbed_ratio = ln_rV_diat_pert / ln_rV_mono

            if perturbed_ratio < base_ratio:
                return (True, f"Perturbed gamma_diat from {self.gamma_diat} to {gamma_diat_pert:.3f}: "
                              f"ratio {base_ratio:.5f} -> {perturbed_ratio:.5f} (decrease)")
            else:
                return (False, f"Perturbed gamma_diat from {self.gamma_diat} to {gamma_diat_pert:.3f}: "
                              f"ratio {base_ratio:.5f} -> {perturbed_ratio:.5f} (did not decrease)")
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Computation error: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such test)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """Only sign and broad range; no re‑derivation."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if self.ratio <= 0:
            return (False, f"Ratio {self.ratio} is not positive")
        if self.ratio >= 1:
            return (False, f"Ratio {self.ratio} is not below 1")
        return (True, f"Ratio {self.ratio} is positive and below 1")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
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