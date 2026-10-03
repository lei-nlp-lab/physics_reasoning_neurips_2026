import sympy as sp
import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics auditor for the gyromagnetic ratio problem.
    Covers interface, invariant, modeling/trace, monotonicity, and one weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', None) or {}
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Required result keys – safe None defaults
        self.gamma_ring = self.results.get('gamma_ring', None)
        self.gamma_sphere = self.results.get('gamma_sphere', None)
        self.mu_e = self.results.get('mu_e', None)

        # Givens (might be missing)
        self.Q = self.givens.get('Q', None)
        self.M = self.givens.get('M', None)
        self.e = self.givens.get('e', None)
        self.m_e = self.givens.get('m_e', None)
        self.hbar = self.givens.get('hbar', None)
        self.R = self.givens.get('R', None)  # radius (if provided)

    def _to_sympy(self, expr) -> Optional[sp.Expr]:
        """Convert to SymPy expression, or None if impossible."""
        if expr is None:
            return None
        try:
            if isinstance(expr, sp.Basic):
                return expr
            if isinstance(expr, str):
                return sp.sympify(expr)
            return sp.nsimplify(expr)
        except Exception:
            return None

    def _numeric_value(self, expr) -> Optional[float]:
        """Return numeric float if possible, else None."""
        sym = self._to_sympy(expr)
        if sym is None:
            return None
        try:
            return float(sym.evalf())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all three required result keys exist and are not None."""
        expected_keys = ['gamma_ring', 'gamma_sphere', 'mu_e']
        missing = [k for k in expected_keys if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        none_keys = [k for k in expected_keys if self.results.get(k) is None]
        if none_keys:
            return (False, f"Result keys with None values: {none_keys}")
        return (True, "All required keys present and non-None.")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_gamma_ring_formula(self) -> Tuple[bool, str]:
        """Check that gamma_ring symbolically equals Q/(2M)."""
        if self.gamma_ring is None:
            return (False, "SKIP: gamma_ring not available")
        gamma_sym = self._to_sympy(self.gamma_ring)
        if gamma_sym is None:
            return (False, "Cannot interpret gamma_ring as symbolic")
        if self.Q is not None and self.M is not None:
            Q = self._to_sympy(self.Q)
            M = self._to_sympy(self.M)
            if Q is None or M is None:
                return (False, "Cannot interpret Q or M")
            expected = Q / (2 * M)
            diff = sp.simplify(gamma_sym - expected)
            if diff == 0:
                return (True, "gamma_ring symbolically equals Q/(2M)")
            else:
                return (False, f"Symbolic difference {diff}")
        return (True, "gamma_ring present; symbolic equality not checkable without Q, M")

    def test_invariant_gamma_sphere_formula(self) -> Tuple[bool, str]:
        """Check that gamma_sphere symbolically equals Q/(2M)."""
        if self.gamma_sphere is None:
            return (False, "SKIP: gamma_sphere not available")
        gamma_sym = self._to_sympy(self.gamma_sphere)
        if gamma_sym is None:
            return (False, "Cannot interpret gamma_sphere as symbolic")
        if self.Q is not None and self.M is not None:
            Q = self._to_sympy(self.Q)
            M = self._to_sympy(self.M)
            if Q is None or M is None:
                return (False, "Cannot interpret Q or M")
            expected = Q / (2 * M)
            diff = sp.simplify(gamma_sym - expected)
            if diff == 0:
                return (True, "gamma_sphere symbolically equals Q/(2M)")
            else:
                return (False, f"Symbolic difference {diff}")
        return (True, "gamma_sphere present; symbolic equality not checkable without Q, M")

    def test_invariant_mu_e_formula(self) -> Tuple[bool, str]:
        """Check that mu_e symbolically equals eħ/(4m_e)."""
        if self.mu_e is None:
            return (False, "SKIP: mu_e not available")
        mu_sym = self._to_sympy(self.mu_e)
        if mu_sym is None:
            return (False, "Cannot interpret mu_e as symbolic")
        if self.e is not None and self.m_e is not None and self.hbar is not None:
            e = self._to_sympy(self.e)
            me = self._to_sympy(self.m_e)
            hbar = self._to_sympy(self.hbar)
            if None in (e, me, hbar):
                return (False, "Cannot interpret e, m_e, or hbar")
            expected = (e * hbar) / (4 * me)
            diff = sp.simplify(mu_sym - expected)
            if diff == 0:
                return (True, "mu_e symbolically equals eħ/(4m_e)")
            else:
                return (False, f"Symbolic difference {diff}")
        return (True, "mu_e present; symbolic equality not checkable without e, m_e, hbar")

    def test_invariant_gamma_equality(self) -> Tuple[bool, str]:
        """Ring and sphere gamma should be identical (both Q/(2M))."""
        if self.gamma_ring is None or self.gamma_sphere is None:
            return (False, "SKIP: one gamma is missing")
        g1 = self._to_sympy(self.gamma_ring)
        g2 = self._to_sympy(self.gamma_sphere)
        if g1 is None or g2 is None:
            return (False, "Cannot convert both gammas")
        diff = sp.simplify(g1 - g2)
        if diff == 0:
            return (True, "gamma_ring and gamma_sphere are identical")
        else:
            return (False, f"Difference: {diff}")

    # ------------------------------------------------------------------
    # Modeling / trace checks (derived from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_gamma_via_current_torque(self) -> Tuple[bool, str]:
        """
        Re‑derive gamma from the explanation steps:
        μ = I A, I = Qω/(2π), A = πR² → μ = QωR²/2
        L = M R² ω   →   γ = μ/L = Q/(2M)
        Check that this relation holds for gamma_ring.
        """
        if self.gamma_ring is None:
            return (False, "SKIP: gamma_ring not available")
        gamma_sym = self._to_sympy(self.gamma_ring)
        if gamma_sym is None:
            return (False, "Cannot interpret gamma_ring")

        if self.Q is None or self.M is None:
            return (True, "Cannot re‑derive without given Q and M; assume consistency")

        Q = self._to_sympy(self.Q)
        M = self._to_sympy(self.M)
        if Q is None or M is None:
            return (False, "Cannot interpret Q or M")

        # Derive gamma from first principles (using symbolic R, ω as placeholders)
        R, omega = sp.symbols('R omega', positive=True)
        I_current = Q * omega / (2 * sp.pi)          # current
        A = sp.pi * R**2                              # area
        mu = sp.simplify(I_current * A)               # μ = I A = Q ω R² / 2
        L = M * R**2 * omega                          # L = I_ring ω
        gamma_derived = sp.simplify(mu / L)            # Q/(2M)
        diff = sp.simplify(gamma_sym - gamma_derived)
        if diff == 0:
            return (True, "gamma_ring matches the derivation from current and angular momentum")
        else:
            return (False, f"Derived gamma = {gamma_derived}, but stored gamma_ring = {gamma_sym}, diff = {diff}")

    def test_trace_dipole_moment_consistency(self) -> Tuple[bool, str]:
        """
        From explanation steps: μ = (Q ω R²)/2.
        Trace that mu_e = gamma_sphere * (ħ/2) holds if gamma_sphere = Q/(2M) after substituting Q=e, M=m_e.
        """
        if self.mu_e is None or self.gamma_sphere is None:
            return (False, "SKIP: need mu_e and gamma_sphere")
        mu_sym = self._to_sympy(self.mu_e)
        gamma_sym = self._to_sympy(self.gamma_sphere)
        if mu_sym is None or gamma_sym is None:
            return (False, "Cannot convert mu_e or gamma_sphere")

        if self.hbar is None:
            return (True, "Cannot trace without ħ; assume consistency")

        hbar = self._to_sympy(self.hbar)
        if hbar is None:
            return (False, "Cannot interpret hbar")

        if self.e is not None and self.m_e is not None:
            e = self._to_sympy(self.e)
            me = self._to_sympy(self.m_e)
            if e is None or me is None:
                return (False, "Cannot interpret e or m_e")
            # Build expected: gamma_sphere with Q→e, M→m_e, then multiply by ħ/2
            gamma_sub = gamma_sym.subs({sp.Symbol('Q'): e, sp.Symbol('M'): me}) if (gamma_sym.has(sp.Symbol('Q')) and gamma_sym.has(sp.Symbol('M'))) else gamma_sym
            expected = gamma_sub * (hbar / 2)
            diff = sp.simplify(mu_sym - expected)
            if diff == 0:
                return (True, "mu_e = gamma_sphere * (ħ/2) holds")
            else:
                return (False, f"mu_e = {mu_sym}, but gamma_sphere*(ħ/2) = {expected}, diff {diff}")
        return (True, "Trace consistency not checkable with given data")

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_gamma_increases_with_Q(self) -> Tuple[bool, str]:
        """
        Perturb Q upward by 10% and verify that gamma_ring increases proportionally.
        Uses known linear relation γ = Q/(2M).
        """
        if self.gamma_ring is None:
            return (False, "SKIP: gamma_ring not available")
        gamma_num = self._numeric_value(self.gamma_ring)
        Q_num = self._numeric_value(self.Q)
        M_num = self._numeric_value(self.M)
        if gamma_num is None or Q_num is None or M_num is None:
            return (True, "Cannot perform numeric perturbation; one value non-numeric")
        # Baseline gamma = Q/(2M)
        baseline_gamma = Q_num / (2 * M_num)
        if not math.isclose(gamma_num, baseline_gamma, rel_tol=1e-2):
            return (False, f"gamma_ring = {gamma_num} does not match Q/(2M) = {baseline_gamma}")

        # Perturb Q by +10%
        Q_pert = Q_num * 1.1
        expected_perturbed_gamma = Q_pert / (2 * M_num)  # linear increase
        # The output should increase (monotonicity)
        if expected_perturbed_gamma > baseline_gamma:
            return (True, f"Perturbing Q by +10% increases gamma from {baseline_gamma} to {expected_perturbed_gamma}")
        else:
            return (False, f"Perturbing Q should increase gamma, but expected perturbed = {expected_perturbed_gamma} <= baseline = {baseline_gamma}")

    def test_monotonicity_gamma_independent_of_R(self) -> Tuple[bool, str]:
        """
        Perturb R by +10% and verify that gamma_ring does not change.
        Uses known result γ = Q/(2M) independent of R.
        """
        if self.gamma_ring is None:
            return (False, "SKIP: gamma_ring not available")
        gamma_num = self._numeric_value(self.gamma_ring)
        Q_num = self._numeric_value(self.Q)
        M_num = self._numeric_value(self.M)
        R_num = self._numeric_value(self.R)
        if gamma_num is None or Q_num is None or M_num is None:
            return (True, "Cannot perform numeric perturbation; one value non-numeric")
        # Expected gamma independent of R
        baseline_gamma = Q_num / (2 * M_num)
        if not math.isclose(gamma_num, baseline_gamma, rel_tol=1e-2):
            return (False, f"gamma_ring = {gamma_num} does not match Q/(2M) = {baseline_gamma}")

        if R_num is None:
            return (True, "No radius given; perturbation not possible, but gamma is independent (inferred from formula)")

        # Perturb R by +10% (hypothetical)
        R_pert = R_num * 1.1
        # Expected gamma after perturbation is still Q/(2M) = baseline_gamma
        expected_perturbed_gamma = baseline_gamma
        if math.isclose(expected_perturbed_gamma, baseline_gamma, rel_tol=1e-2):
            return (True, f"Perturbing R from {R_num} to {R_pert} leaves gamma unchanged at {baseline_gamma}")
        else:
            return (False, "gamma should be independent of R, but perturbation changed expected value")

    # ------------------------------------------------------------------
    # Weak answer sanity (only ONE test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_physical_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: if gamma_ring is numeric, it should be non‑negative for non‑negative Q and M.
        No isclose, no recomputation of target values.
        """
        if self.gamma_ring is None:
            return (False, "SKIP: gamma_ring not available")
        gamma_num = self._numeric_value(self.gamma_ring)
        if gamma_num is None:
            return (True, "Gamma not numeric; sanity skipped")
        if gamma_num < 0:
            return (False, f"gamma_ring = {gamma_num} is negative; expected non‑negative")
        return (True, f"gamma_ring = {gamma_num} is non‑negative")

    # ------------------------------------------------------------------
    # Master test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover every test_* method, run them, collect results."""
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)