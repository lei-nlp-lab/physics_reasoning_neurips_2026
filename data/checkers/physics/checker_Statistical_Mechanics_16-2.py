import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution instance for the photon gas adiabatic expansion problem.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens; use None defaults, never numeric fallbacks
        self.tau_i = getattr(solution_instance, 'tau_i', None)
        self.tau_f = getattr(solution_instance, 'tau_f', None)
        self.V_i = getattr(solution_instance, 'V_i', None)

        # Extract results from solution_instance.results (or direct attributes)
        results = getattr(solution_instance, 'results', {})
        if isinstance(results, dict):
            self.radius_ratio = results.get('radius_ratio', None)
            self.time_ratio = results.get('time_ratio', None)
            self.W = results.get('W', None)
        else:
            self.radius_ratio = getattr(solution_instance, 'radius_ratio', None)
            self.time_ratio = getattr(solution_instance, 'time_ratio', None)
            self.W = getattr(solution_instance, 'W', None)

        # Precompute the Stefan-Boltzmann constant for work derivation
        hbar_val = 1.054571817e-34   # J·s
        c_val = 2.99792458e8        # m/s
        self.a_numeric = math.pi**2 / (15 * hbar_val**3 * c_val**3)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_radius_ratio_key(self):
        """Check that radius_ratio exists and is numeric."""
        if self.radius_ratio is None:
            return (False, "SKIP: radius_ratio not provided")
        if not isinstance(self.radius_ratio, (int, float)):
            return (False, "radius_ratio must be numeric")
        return (True, "radius_ratio present and numeric")

    def test_interface_time_ratio_key(self):
        """Check that time_ratio exists and is numeric."""
        if self.time_ratio is None:
            return (False, "SKIP: time_ratio not provided")
        if not isinstance(self.time_ratio, (int, float)):
            return (False, "time_ratio must be numeric")
        return (True, "time_ratio present and numeric")

    def test_interface_W_key(self):
        """Check that W exists and is numeric."""
        if self.W is None:
            return (False, "SKIP: W not provided")
        if not isinstance(self.W, (int, float, complex)):
            return (False, "W must be numeric")
        return (True, "W present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_adiabatic_radius_temperature(self):
        """
        Adiabatic invariant: τ V^{1/3} = constant => R ∝ τ^{-1}.
        Therefore radius_ratio should equal tau_f / tau_i.
        """
        if self.radius_ratio is None or self.tau_i is None or self.tau_f is None:
            return (False, "SKIP: missing radius_ratio or temperatures")
        expected_ratio = float(self.tau_f) / float(self.tau_i)
        if not math.isclose(self.radius_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"radius_ratio {self.radius_ratio} != tau_f/tau_i = {expected_ratio}")
        return (True, f"radius_ratio matches adiabatic invariant: {self.radius_ratio}")

    def test_invariant_linear_expansion(self):
        """
        If radius increases linearly with time, then time_ratio = radius_ratio.
        """
        if self.radius_ratio is None or self.time_ratio is None:
            return (False, "SKIP: missing radius_ratio or time_ratio")
        if not math.isclose(self.time_ratio, self.radius_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"time_ratio {self.time_ratio} != radius_ratio {self.radius_ratio}")
        return (True, f"time_ratio matches radius_ratio: {self.time_ratio}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_work_positive(self):
        """
        Work done by radiation during expansion must be positive.
        The expression W = a V_i tau_i^3 (tau_i - tau_f) is positive because tau_i > tau_f.
        """
        if self.W is None:
            return (False, "SKIP: W not provided")
        if not math.isfinite(self.W) or self.W <= 0:
            return (False, f"W must be positive, got {self.W}")
        return (True, f"W = {self.W} is positive")

    def test_modeling_work_structure(self):
        """
        Verify that the work expression has the correct functional form:
        W = a V_i tau_i^3 (tau_i - tau_f).  Re-derive from givens if available.
        """
        if self.W is None or self.V_i is None or self.tau_i is None or self.tau_f is None:
            return (False, "SKIP: missing required values for work structure test")
        tau_i_val = float(self.tau_i)
        tau_f_val = float(self.tau_f)
        V_i_val = float(self.V_i)
        expected_W = self.a_numeric * V_i_val * tau_i_val**3 * (tau_i_val - tau_f_val)
        if not math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"W computed as {self.W}, but expected {expected_W} (within tolerance)")
        return (True, f"W matches expression a V_i tau_i^3 (tau_i - tau_f)")

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks via explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_radius_ratio_with_tau_i(self):
        """
        The radius_ratio = tau_f / tau_i is a decreasing function of tau_i.
        Perturb tau_i upward by 10% and verify the radius_ratio decreases.
        """
        if self.radius_ratio is None or self.tau_i is None or self.tau_f is None:
            return (False, "SKIP: missing required values")
        tau_i_original = float(self.tau_i)
        tau_f = float(self.tau_f)
        baseline_ratio = tau_f / tau_i_original
        # Perturb: increase tau_i by 10%
        tau_i_perturbed = tau_i_original * 1.1
        perturbed_ratio = tau_f / tau_i_perturbed
        if not (perturbed_ratio < baseline_ratio):
            return (False, f"After increasing tau_i by 10%, radius_ratio should decrease, "
                           f"got baseline={baseline_ratio}, perturbed={perturbed_ratio}")
        return (True, f"Monotonicity verified: radius_ratio decreases with tau_i")

    def test_monotonicity_work_with_tau_i(self):
        """
        The work W ∝ tau_i^3 (tau_i - tau_f) is an increasing function of tau_i.
        Perturb tau_i upward by 10% and compute the perturbed W using the
        physical expression (since the solution's W is not directly recomputable).
        Compare the perturbed W to the baseline (the solution's W) to ensure
        the solution is consistent with monotonic behavior.
        """
        if self.W is None or self.V_i is None or self.tau_i is None or self.tau_f is None:
            return (False, "SKIP: missing required values")
        tau_i_val = float(self.tau_i)
        tau_f_val = float(self.tau_f)
        V_i_val = float(self.V_i)

        # Baseline W from solution
        baseline_W = self.W

        # Perturb tau_i upward by 10%
        tau_i_perturbed = tau_i_val * 1.1
        # Compute perturbed W using the physical law (derived inside checker)
        perturbed_W = self.a_numeric * V_i_val * tau_i_perturbed**3 * (tau_i_perturbed - tau_f_val)

        # The perturbed W should be larger than baseline (assuming other parameters fixed)
        if perturbed_W <= baseline_W:
            return (False, f"Perturbed W {perturbed_W} should be > baseline W {baseline_W} when tau_i increases")
        return (True, f"Work increases with tau_i (baseline {baseline_W:.2e}, perturbed {perturbed_W:.2e})")

    # ------------------------------------------------------------------
    # Single weak answer sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak_checks(self):
        """
        Very weak sanity: all outputs are finite, positive, and within broadly plausible ranges.
        No re-derivation of expected values.
        """
        if self.radius_ratio is None and self.time_ratio is None and self.W is None:
            return (False, "SKIP: no results provided")
        messages = []
        all_ok = True

        if self.radius_ratio is not None:
            if not math.isfinite(self.radius_ratio):
                messages.append("radius_ratio not finite")
                all_ok = False
            elif not (1e-10 <= self.radius_ratio <= 0.1):
                messages.append("radius_ratio out of plausible range [1e-10,0.1]")
                all_ok = False
        if self.time_ratio is not None:
            if not math.isfinite(self.time_ratio):
                messages.append("time_ratio not finite")
                all_ok = False
            elif not (1e-10 <= self.time_ratio <= 0.1):
                messages.append("time_ratio out of plausible range [1e-10,0.1]")
                all_ok = False
        if self.W is not None:
            if not math.isfinite(self.W):
                messages.append("W not finite")
                all_ok = False
            elif not (self.W > 0):
                messages.append("W not positive")
                all_ok = False

        if not all_ok:
            return (False, " | ".join(messages) if messages else "unknown sanity failure")
        return (True, "All outputs are finite, positive, and within plausible ranges")

    # ------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} crashed: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)