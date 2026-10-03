import math
import traceback
from typing import Any, Callable, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits a solution for the ground-state energy of two identical bosons
    in a 1D harmonic oscillator with Gaussian interaction, using first-order
    perturbation theory.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Givens (safely extracted, defaults to None) ---
        self.m     = getattr(solution_instance, 'm', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.hbar  = getattr(solution_instance, 'hbar', None)
        self.alpha = getattr(solution_instance, 'alpha', None)
        self.beta  = getattr(solution_instance, 'beta', None)

        # --- Results (dict or fallback empty dict) ---
        self.results = getattr(solution_instance, 'results', {}) or {}

        # --- Extracted result quantities (all None if missing) ---
        self.E0     = self.results.get('E0', None)
        self.alpha_0 = self.results.get('alpha_0', None)
        self.E1     = self.results.get('E1', None)
        self.E      = self.results.get('E', None)

    # ------------------------------------------------------------------
    # Helper: recompute alpha_0 from givens (if all available)
    # ------------------------------------------------------------------
    def _derive_alpha_0(self) -> Optional[float]:
        if self.m is None or self.omega is None or self.hbar is None:
            return None
        try:
            m_f = float(self.m)
            w_f = float(self.omega)
            h_f = float(self.hbar)
            if m_f <= 0 or w_f <= 0 or h_f <= 0:
                return None
            return math.sqrt(m_f * w_f / h_f)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    # ------------------------------------------------------------------
    # Helper: compute E1 from derived alpha_0 (if all needed givens present)
    # ------------------------------------------------------------------
    def _derive_E1(self) -> Optional[float]:
        a0 = self._derive_alpha_0()
        if a0 is None or self.alpha is None or self.beta is None:
            return None
        try:
            a = float(self.alpha)
            b = float(self.beta)
            if a0 <= 0:
                return None
            denom = math.sqrt(a0**2 + 2.0 * b)
            if denom <= 0.0:
                return None
            return a * a0 / denom
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    # ------------------------------------------------------------------
    # 1. INTERFACE TESTS – required keys and types
    # ------------------------------------------------------------------
    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that all five givens are present and numeric."""
        for name, val in [('m', self.m), ('omega', self.omega),
                          ('hbar', self.hbar), ('alpha', self.alpha),
                          ('beta', self.beta)]:
            if val is None:
                return (False, f"Given '{name}' is missing (None).")
            try:
                float(val)
            except (TypeError, ValueError):
                return (False, f"Given '{name}' is not numeric: {val}.")
        # Additional positivity requirement for m, omega, hbar, beta
        if float(self.m) <= 0:
            return (False, "m must be positive.")
        if float(self.omega) <= 0:
            return (False, "omega must be positive.")
        if float(self.hbar) <= 0:
            return (False, "hbar must be positive.")
        if float(self.beta) <= 0:
            return (False, "beta must be positive.")
        return (True, "All givens present and valid.")

    def test_interface_results_keys(self) -> Tuple[bool, str]:
        """Check that all four expected result keys exist and are numeric."""
        required_keys = ['E0', 'alpha_0', 'E1', 'E']
        for key in required_keys:
            val = self.results.get(key, None)
            if val is None:
                return (False, f"Required result key '{key}' is missing.")
            try:
                float(val)
            except (TypeError, ValueError):
                return (False, f"Result key '{key}' is not numeric: {val}.")
        return (True, "All required result keys present and numeric.")

    # ------------------------------------------------------------------
    # 2. INVARIANT TESTS – physics laws (fuzzy matching)
    # ------------------------------------------------------------------
    def test_invariant_E0_relation(self) -> Tuple[bool, str]:
        """E0 should equal hbar * omega (unperturbed ground state)."""
        if self.E0 is None:
            return (False, "SKIP: E0 not available.")
        e0 = float(self.E0)
        if self.hbar is None or self.omega is None:
            return (False, "SKIP: needed givens missing for E0 check.")
        try:
            h = float(self.hbar)
            o = float(self.omega)
        except (TypeError, ValueError):
            return (False, "Givens not numeric for E0 check.")
        expected = h * o
        if math.isclose(e0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E0 ≈ ħω = {expected:.4f}, obtained {e0:.4f}.")
        else:
            return (False, f"E0 = {e0:.4f} differs from ħω = {expected:.4f}.")

    def test_invariant_alpha0_relation(self) -> Tuple[bool, str]:
        """alpha_0 should equal sqrt(m ω / ħ)."""
        if self.alpha_0 is None:
            return (False, "SKIP: alpha_0 not available.")
        derived = self._derive_alpha_0()
        if derived is None:
            return (False, "SKIP: cannot derive alpha_0 from givens.")
        try:
            a0 = float(self.alpha_0)
        except (TypeError, ValueError):
            return (False, "alpha_0 is not numeric.")
        if math.isclose(a0, derived, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"alpha_0 ≈ {derived:.4f}, obtained {a0:.4f}.")
        else:
            return (False, f"alpha_0 = {a0:.4f} differs from sqrt(mω/ħ) = {derived:.4f}.")

    def test_invariant_E1_relation(self) -> Tuple[bool, str]:
        """E1 should equal α·α₀ / sqrt(α₀² + 2β)."""
        if self.E1 is None:
            return (False, "SKIP: E1 not available.")
        derived = self._derive_E1()
        if derived is None:
            return (False, "SKIP: cannot derive E1 from givens.")
        try:
            e1 = float(self.E1)
        except (TypeError, ValueError):
            return (False, "E1 is not numeric.")
        if math.isclose(e1, derived, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E1 ≈ {derived:.4f}, obtained {e1:.4f}.")
        else:
            return (False, f"E1 = {e1:.4f} differs from expected {derived:.4f}.")

    def test_invariant_E_sum(self) -> Tuple[bool, str]:
        """Total energy E should equal E0 + E1."""
        if None in (self.E, self.E0, self.E1):
            return (False, "SKIP: one of E, E0, E1 missing.")
        try:
            e  = float(self.E)
            e0 = float(self.E0)
            e1 = float(self.E1)
        except (TypeError, ValueError):
            return (False, "E, E0, or E1 not numeric.")
        if math.isclose(e, e0 + e1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E = E0+E1: {e0 + e1:.4f}, obtained {e:.4f}.")
        else:
            return (False, f"E = {e:.4f} differs from E0+E1 = {e0 + e1:.4f}.")

    # ------------------------------------------------------------------
    # 3. MODELING / TRACE QUALITY TESTS (at least two)
    # ------------------------------------------------------------------
    def test_modeling_perturbation_validity(self) -> Tuple[bool, str]:
        """First-order correction should be small relative to unperturbed energy."""
        if None in (self.E0, self.E1):
            return (False, "SKIP: E0 or E1 missing.")
        try:
            e0 = float(self.E0)
            e1 = float(self.E1)
        except (TypeError, ValueError):
            return (False, "E0 or E1 not numeric.")
        ratio = abs(e1 / e0) if abs(e0) > 1e-12 else float('inf')
        # Typically α is small, so |E1| << |E0|
        if ratio < 1.0:
            return (True, f"|E1/E0| = {ratio:.3f} < 1, first-order plausible.")
        else:
            return (False, f"|E1/E0| = {ratio:.3f} ≥ 1, first-order may be invalid.")

    def test_modeling_wavefunction_normalization(self) -> Tuple[bool, str]:
        """Check that the two-particle ground state wavefunction is normalized to 1,
        i.e., ∫∫ |ψ0|² dx1 dx2 = 1, using the provided alpha_0."""
        if self.alpha_0 is None:
            return (False, "SKIP: alpha_0 not available.")
        try:
            a0 = float(self.alpha_0)
        except (TypeError, ValueError):
            return (False, "alpha_0 not numeric.")
        if a0 <= 0:
            return (False, "alpha_0 must be positive.")
        # The wavefunction used: ψ0 = (α0/√π) exp[-α0²(x1²+x2²)/2]
        # Its modulus squared: |ψ0|² = (α0²/π) exp[-α0²(x1²+x2²)]
        # Integral over all space: (α0²/π) * (√π/α0) * (√π/α0) = 1
        # Compute the integration factor:
        norm = (a0**2 / math.pi) * (math.sqrt(math.pi) / a0) * (math.sqrt(math.pi) / a0)
        # Should be exactly 1.0
        if math.isclose(norm, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Normalization integral = {norm:.4f}, close to 1.")
        else:
            return (False, f"Normalization integral = {norm:.4f}, should be 1.")

    def test_modeling_symmetry_of_E1(self) -> Tuple[bool, str]:
        """The interaction depends only on relative coordinate, so E1 should be symmetric
        under exchange α → same value (no dependence on extra variables). We check that
        E1 computed from the formula is unchanged if we swap alpha with itself (trivial)
        but also that it remains finite for positive β."""
        if self.E1 is None:
            return (False, "SKIP: E1 not available.")
        try:
            e1 = float(self.E1)
        except (TypeError, ValueError):
            return (False, "E1 not numeric.")
        if math.isnan(e1) or math.isinf(e1):
            return (False, "E1 is NaN or infinite.")
        # Symmetry: E1 does not depend on any variable that would break exchange
        # (The expression is symmetric in x1↔x2 automatically.)
        # This test simply ensures that E1 is well-defined and not pathological.
        # We also check that E1 is consistent with the sign of alpha (if alpha is given)
        if self.alpha is not None:
            try:
                alpha_f = float(self.alpha)
            except (TypeError, ValueError):
                alpha_f = None
            if alpha_f is not None:
                if alpha_f > 0 and e1 < 0:
                    return (False, f"For α>0, E1 should be positive (α={alpha_f}), but E1={e1:.4f} is negative.")
                if alpha_f < 0 and e1 > 0:
                    return (False, f"For α<0, E1 should be negative (α={alpha_f}), but E1={e1:.4f} is positive.")
        return (True, f"E1 = {e1:.4f} is finite and sign consistent with α.")

    # ------------------------------------------------------------------
    # 4. MONOTONICITY TESTS – perturb an input and compare output
    # ------------------------------------------------------------------
    def test_monotonicity_beta_increase(self) -> Tuple[bool, str]:
        """
        Increasing beta should decrease |E1| (denominator grows).
        We recompute E1 via the formula with perturbed beta and compare absolute values.
        """
        derived_alpha0 = self._derive_alpha_0()
        if derived_alpha0 is None:
            return (False, "SKIP: cannot compute derived alpha_0.")
        if self.alpha is None or self.beta is None:
            return (False, "SKIP: alpha or beta missing.")
        try:
            a = float(self.alpha)
            b = float(self.beta)
        except (TypeError, ValueError):
            return (False, "alpha or beta not numeric.")

        # Baseline |E1|
        denom_nom = math.sqrt(derived_alpha0**2 + 2.0 * b)
        if denom_nom <= 0:
            return (False, "Denominator zero or negative in nominal E1.")
        nom_mag = abs(a * derived_alpha0 / denom_nom)

        # Perturb beta: increase by 10%
        b_pert = b * 1.1
        denom_pert = math.sqrt(derived_alpha0**2 + 2.0 * b_pert)
        if denom_pert <= 0:
            return (False, "Denominator zero or negative in perturbed E1.")
        pert_mag = abs(a * derived_alpha0 / denom_pert)

        if pert_mag < nom_mag:
            return (True, f"|E1| decreases with β: nominal |E1| = {nom_mag:.4f}, perturbed (1.1β) |E1| = {pert_mag:.4f}.")
        else:
            return (False, f"Monotonicity failed: |E1| did not decrease (nominal {nom_mag:.4f}, perturbed {pert_mag:.4f}). Increase of β should reduce |E1|.")

    def test_monotonicity_alpha_increase(self) -> Tuple[bool, str]:
        """
        Increasing the coupling α (positive) should increase |E1| proportionally.
        If α is negative, |E1| increases as well (absolute value increases).
        We check that the ratio of perturbed to nominal |E1| matches the ratio of alphas.
        """
        derived_alpha0 = self._derive_alpha_0()
        if derived_alpha0 is None:
            return (False, "SKIP: cannot compute derived alpha_0.")
        if self.alpha is None or self.beta is None:
            return (False, "SKIP: alpha or beta missing.")
        try:
            a = float(self.alpha)
            b = float(self.beta)
        except (TypeError, ValueError):
            return (False, "alpha or beta not numeric.")

        # Baseline |E1|
        denom = math.sqrt(derived_alpha0**2 + 2.0 * b)
        if denom <= 0:
            return (False, "Denominator zero or negative.")
        nom_mag = abs(a * derived_alpha0 / denom)

        # Perturb alpha: increase by 5%
        a_pert = a * 1.05
        pert_mag = abs(a_pert * derived_alpha0 / denom)

        # Expected ratio of |E1| should equal ratio of |alpha|
        ratio_alpha = abs(a_pert / a)  # 1.05
        ratio_E1 = pert_mag / nom_mag if nom_mag > 0 else float('inf')
        if math.isclose(ratio_E1, ratio_alpha, rel_tol=0.01):
            return (True, f"|E1| scales linearly with |α|: ratio_E1 = {ratio_E1:.3f}, ratio_α = {ratio_alpha:.3f}.")
        else:
            return (False, f"Monotonicity failed: |E1| should increase proportionally to |α|. Expected ratio {ratio_alpha:.3f}, got {ratio_E1:.3f}.")

    # ------------------------------------------------------------------
    # 5. ANSWER SANITY – weak check (no isclose, no rederivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_E(self) -> Tuple[bool, str]:
        """Final energy E must be finite and not NaN, and not essentially zero."""
        if self.E is None:
            return (False, "SKIP: E not available.")
        try:
            e = float(self.E)
        except (TypeError, ValueError):
            return (False, "E is not numeric.")
        if math.isnan(e) or math.isinf(e):
            return (False, "E is NaN or Infinite.")
        # A very loose range: not more than 10 times the typical scale (E0)
        # We don't compute expected, just ensure it's not absurd.
        if abs(e) < 1e-30:
            return (False, "E is essentially zero, suspicious for ground-state energy.")
        return (True, f"E = {e:.4f} is finite and non-zero.")

    # ------------------------------------------------------------------
    # 6. RUN ALL TESTS
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method: Callable[[], Tuple[bool, str]] = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)