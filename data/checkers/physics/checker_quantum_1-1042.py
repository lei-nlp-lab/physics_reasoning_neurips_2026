import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Tuple, Union

class PhysicsChecker:
    """
    Auditor for the quantum free‑particle probability current problem.
    """

    def __init__(self, solution_instance: Any) -> None:
        self.sol = solution_instance
        self.results: Dict = getattr(self.sol, 'results', {})
        self.givens: Dict = getattr(self.sol, 'givens', {})

        # Primary output
        self.j: Union[float, None] = self.results.get('j', None)

        # Physical parameters – first from givens, then from instance attributes
        self.m: Union[float, None] = self._safe_get('m')
        self.a: Union[float, None] = self._safe_get('a')
        self.L: Union[float, None] = self._safe_get('L')
        self.t: Union[float, None] = self._safe_get('t')
        self.hbar: Union[float, None] = self._safe_get('hbar')

    # ------------------------------------------------------------------
    # Helper: fetch a value from givens or from the solution instance itself
    # ------------------------------------------------------------------
    def _safe_get(self, key: str) -> Union[float, None]:
        val = self.givens.get(key, None)
        if val is not None:
            return val
        return getattr(self.sol, key, None)

    # ------------------------------------------------------------------
    # Helper: compute probability current for given parameters (re‑derived formula)
    # ------------------------------------------------------------------
    def _compute_j(self, m: float, a: float, L: float, t: float, hbar: float) -> float:
        denom = 1.0 + (hbar * t / (m * a * a)) ** 2
        prefactor = (hbar * hbar * L * t) / (math.sqrt(math.pi) * (a ** 5) * (m * m))
        factor = denom ** (-1.5)
        exponent = - (L * L) / (a * a) / denom
        return prefactor * factor * math.exp(exponent)

    # ==================================================================
    # INTERFACE TESTS
    # ==================================================================

    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Ensure all essential keys exist in results/givens."""
        missing = []
        if self.j is None:
            missing.append("results['j']")
        if self.m is None:
            missing.append("givens['m'] or instance.m")
        if self.a is None:
            missing.append("givens['a'] or instance.a")
        if self.L is None:
            missing.append("givens['L'] or instance.L")
        if self.t is None:
            missing.append("givens['t'] or instance.t")
        if self.hbar is None:
            missing.append("givens['hbar'] or instance.hbar")
        if missing:
            return (False, f"Missing: {', '.join(missing)}")
        return (True, "All required keys present.")

    def test_interface_givens_numeric(self) -> Tuple[bool, str]:
        """Check all givens are finite, positive numbers."""
        non_numeric = []
        for name, val in [('m', self.m), ('a', self.a), ('L', self.L),
                          ('t', self.t), ('hbar', self.hbar)]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                non_numeric.append(f"{name} is {type(val).__name__}")
            elif not math.isfinite(val):
                non_numeric.append(f"{name} is not finite")
            elif val <= 0:
                non_numeric.append(f"{name} ≤ 0 (got {val})")
        if non_numeric:
            return (False, "; ".join(non_numeric))
        return (True, "All givens are numeric, finite, and positive.")

    # ==================================================================
    # INVARIANT TESTS
    # ==================================================================

    def test_invariant_j_positive(self) -> Tuple[bool, str]:
        """Probability current must be positive for t>0, L>0."""
        if self.j is None:
            return (False, "SKIP: j not available")
        if self.j <= 0:
            return (False, f"j = {self.j} is not positive")
        return (True, f"j = {self.j} > 0")

    def test_invariant_j_finite(self) -> Tuple[bool, str]:
        """j must be a finite number."""
        if self.j is None:
            return (False, "SKIP: j not available")
        if not math.isfinite(self.j):
            return (False, f"j is not finite (got {self.j})")
        return (True, "j is finite.")

    # ==================================================================
    # MODELING / TRACE TESTS  (at least two)
    # ==================================================================

    def test_modeling_sympy_derivation(self) -> Tuple[bool, str]:
        """
        Use symbolic sympy to verify that the simplified expression for j
        matches the definition j = (hbar/m) Im(ψ* dψ/dx) from the given
        Gaussian wavepacket.  This re‑derives the formula inside the checker.
        """
        if any(v is None for v in [self.m, self.a, self.L, self.t, self.hbar]):
            return (False, "SKIP: insufficient parameters for symbolic check")

        # Define symbolic variables
        m_s, a_s, L_s, t_s, hbar_s = sp.symbols('m a L t hbar', positive=True)
        x = sp.symbols('x')
        gamma = 1 + sp.I * hbar_s * t_s / (m_s * a_s**2)
        psi = (1 / (sp.pi**(sp.Rational(1,4)) * sp.sqrt(a_s) * sp.sqrt(gamma))) * \
              sp.exp(-x**2 / (2 * a_s**2 * gamma))
        dpsi_dx = sp.diff(psi, x)
        j_expr = (hbar_s / m_s) * sp.im(sp.conjugate(psi) * dpsi_dx)
        j_expr_L = sp.simplify(j_expr.subs(x, L_s))

        denom_s = 1 + (hbar_s * t_s / (m_s * a_s**2))**2
        j_expected = (hbar_s**2 * L_s * t_s) / (sp.sqrt(sp.pi) * a_s**5 * m_s**2) * \
                     denom_s**(-sp.Rational(3,2)) * \
                     sp.exp(-L_s**2 / (a_s**2 * denom_s))

        difference = sp.simplify(j_expr_L - j_expected)
        if difference == 0:
            return (True, "Symbolic derivation matches the simplified formula.")
        else:
            subs_dict = {m_s: self.m, a_s: self.a, L_s: self.L,
                         t_s: self.t, hbar_s: self.hbar}
            diff_num = float(complex(difference.subs(subs_dict).evalf()))
            if abs(diff_num) < 1e-12:
                return (True, "Numerical check of symbolic difference passes.")
            else:
                return (False, f"Symbolic difference = {difference}; "
                               f"numerical = {diff_num:.2e}")

    def test_modeling_j_L_ratio(self) -> Tuple[bool, str]:
        """
        Verify the functional dependence on L by comparing the ratio j(2L)/j(L)
        to the analytic expression: ratio = 2 * exp( -3L^2/(a^2 * denom) ).
        """
        if any(v is None for v in [self.m, self.a, self.L, self.t, self.hbar, self.j]):
            return (False, "SKIP: insufficient data for L ratio test")

        # Compute j(L) using internal formula (should match provided j)
        baseline_j = self._compute_j(self.m, self.a, self.L, self.t, self.hbar)
        if not math.isclose(baseline_j, self.j, rel_tol=1e-9, abs_tol=1e-15):
            return (False, f"Provided j ({self.j}) does not match internally computed "
                           f"j ({baseline_j:.6e}).")

        # Compute j(2L)
        L2 = 2.0 * self.L
        j2L = self._compute_j(self.m, self.a, L2, self.t, self.hbar)
        computed_ratio = j2L / self.j

        # Expected analytic ratio
        denom = 1.0 + (self.hbar * self.t / (self.m * self.a * self.a))**2
        expected_ratio = 2.0 * math.exp(-3.0 * (self.L**2) / (self.a**2 * denom))

        if math.isclose(computed_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"j(2L)/j(L) = {computed_ratio:.6e} matches expected "
                          f"{expected_ratio:.6e}.")
        else:
            return (False, f"j(2L)/j(L) = {computed_ratio:.6e} does not match expected "
                           f"{expected_ratio:.6e}.")

    def test_modeling_j_small_t_linear(self) -> Tuple[bool, str]:
        """
        For very small t (t << m a^2 / hbar), j should be approximately linear in t.
        Check that j(2t) / j(t) ≈ 2.
        """
        if any(v is None for v in [self.m, self.a, self.L, self.t, self.hbar, self.j]):
            return (False, "SKIP: insufficient data for small t test")

        # Define characteristic time
        t_char = self.m * self.a * self.a / self.hbar
        if self.t >= 0.3 * t_char:  # not small enough
            return (True, "SKIP: t is not sufficiently small (t/t_char = "
                          f"{self.t/t_char:.3f})")

        # Baseline j at given t
        baseline_j = self._compute_j(self.m, self.a, self.L, self.t, self.hbar)
        if not math.isclose(baseline_j, self.j, rel_tol=1e-9, abs_tol=1e-15):
            return (False, f"Provided j ({self.j}) does not match internally computed "
                           f"j ({baseline_j:.6e}).")

        # Double t
        t2 = 2.0 * self.t
        j2t = self._compute_j(self.m, self.a, self.L, t2, self.hbar)
        ratio = j2t / self.j
        if math.isclose(ratio, 2.0, rel_tol=0.05):  # 5% tolerance for linearity
            return (True, f"For small t, j(2t)/j(t) = {ratio:.6f} ≈ 2 (linear regime).")
        else:
            return (False, f"For small t, j(2t)/j(t) = {ratio:.6f} should be ~2.")

    # ==================================================================
    # ANSWER SANITY (weak, at most one – only sign/range/order-of-magnitude)
    # ==================================================================

    def test_answer_sanity_magnitude(self) -> Tuple[bool, str]:
        """
        Weak sanity check: j must be positive and not astronomically large.
        No re-derivation of formula used.
        """
        if self.j is None:
            return (False, "SKIP: j not available")
        # Simple range check (order-of-magnitude)
        if self.j <= 0:
            return (False, f"j = {self.j} is not positive")
        if self.j > 1e100:
            return (False, f"j = {self.j} is unreasonably large (> 1e100)")
        return (True, f"j = {self.j} is positive and within plausible magnitude.")

    # ==================================================================
    # MONOTONICITY TESTS (each perturbs an input and compares)
    # ==================================================================

    def test_monotonicity_j_decreases_with_L(self) -> Tuple[bool, str]:
        """
        Perturb L by +5% and verify that j decreases.
        (The analytic formula is strictly decreasing in L for L>0.)
        """
        if any(v is None for v in [self.m, self.a, self.L, self.t, self.hbar, self.j]):
            return (False, "SKIP: insufficient data for monotonicity check")

        baseline_j = self._compute_j(self.m, self.a, self.L, self.t, self.hbar)
        if not math.isclose(baseline_j, self.j, rel_tol=1e-9, abs_tol=1e-15):
            return (False, f"Provided j ({self.j}) does not match internally computed "
                           f"j ({baseline_j:.6e}). Cannot trust monotonicity test.")

        L_pert = self.L * 1.05
        perturbed_j = self._compute_j(self.m, self.a, L_pert, self.t, self.hbar)
        if perturbed_j >= baseline_j:
            return (False, f"Perturbed L = {L_pert:.4e} gave j_pert = {perturbed_j:.6e}, "
                           f"but j should decrease (baseline = {baseline_j:.6e}).")
        return (True, f"j decreases when L increases by 5% (baseline {baseline_j:.6e} → "
                      f"{perturbed_j:.6e}).")

    def test_monotonicity_j_increases_with_t_small_t(self) -> Tuple[bool, str]:
        """
        For small t (t << m a² / ħ), j is increasing in t.
        Perturb t by +5% and verify that j increases.
        """
        if any(v is None for v in [self.m, self.a, self.L, self.t, self.hbar, self.j]):
            return (False, "SKIP: insufficient data for monotonicity check")

        t_char = self.m * self.a * self.a / self.hbar
        if self.t >= 0.5 * t_char:
            return (True, "SKIP: t is not small enough for the increasing‑with‑t regime.")

        baseline_j = self._compute_j(self.m, self.a, self.L, self.t, self.hbar)
        if not math.isclose(baseline_j, self.j, rel_tol=1e-9, abs_tol=1e-15):
            return (False, f"Provided j ({self.j}) does not match internally computed "
                           f"j ({baseline_j:.6e}). Cannot trust monotonicity test.")

        t_pert = self.t * 1.05
        perturbed_j = self._compute_j(self.m, self.a, self.L, t_pert, self.hbar)
        if perturbed_j <= baseline_j:
            return (False, f"For small t, j should increase with t, but perturbed t = {t_pert:.4e} "
                           f"gave j_pert = {perturbed_j:.6e} ≤ baseline = {baseline_j:.6e}.")
        return (True, f"For small t, j increases when t increases by 5% "
                      f"(baseline {baseline_j:.6e} → {perturbed_j:.6e}).")

    # ==================================================================
    # RUNNER
    # ==================================================================

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
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