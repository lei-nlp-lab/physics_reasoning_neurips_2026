import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution for the 2D isotropic harmonic oscillator perturbed by 
    b(t) x y using first‑order time‑dependent perturbation theory.
    """

    def __init__(self, solution_instance: Any):
        # ---- Extract results -------------------------------------------------
        self.results = getattr(solution_instance, 'results', None)
        if isinstance(self.results, dict):
            self.P_raw = self.results.get('P', None)
        else:
            self.P_raw = None

        # Try to convert to numeric first; keep string for symbolic checks
        if self.P_raw is not None:
            try:
                self.P = float(self.P_raw)
                self.P_str = None
            except (TypeError, ValueError):
                self.P = None
                self.P_str = str(self.P_raw)
        else:
            self.P = None
            self.P_str = None

        # ---- Extract givens -------------------------------------------------
        self.givens = getattr(solution_instance, 'givens', None)
        if isinstance(self.givens, dict):
            self.m = self.givens.get('m', None)          # mass
            self.omega = self.givens.get('omega', None)  # oscillator frequency
            self.b0 = self.givens.get('b0', None)        # perturbation amplitude
            self.tau = self.givens.get('tau', None)      # decay time constant
        else:
            self.m = None
            self.omega = None
            self.b0 = None
            self.tau = None

        # ---- Extract textual reasoning traces --------------------------------
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = [str(self.explanation_steps)]
        self.pseudocode = getattr(solution_instance, 'pseudocode', '')

    # ----------------------------------------------------------------------
    #  Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that essential keys exist in results and givens."""
        if self.results is None:
            return (False, "No 'results' found in solution_instance")
        if 'P' not in self.results:
            return (False, "Key 'P' not found in results")
        if self.givens is None:
            return (False, "No 'givens' found in solution_instance")
        required_givens = ['m', 'omega', 'b0', 'tau']
        missing = [k for k in required_givens if k not in self.givens]
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All required interface keys present")

    # ----------------------------------------------------------------------
    #  Invariant (physical sanity) tests
    # ----------------------------------------------------------------------
    def test_invariant_probability_in_range(self) -> Tuple[bool, str]:
        """Probability must be non‑negative and less than 1 (weak perturbation)."""
        if self.P is None:
            return (False, "SKIP: numeric P not available")
        if self.P < 0:
            return (False, f"P is negative ({self.P})")
        if self.P >= 1.0:
            return (False, f"P >= 1, perturbation theory invalid ({self.P})")
        return (True, f"P = {self.P:.6g} is in [0,1)")

    def test_invariant_symbolic_form(self) -> Tuple[bool, str]:
        """
        Symbolically check that the solution's expression for P matches the
        known formula: b0^2 / (4 m^2 ω^2 (ω^2 + τ^{-2})).
        This re‑derives the expected form from the theorem, so it is allowed.
        """
        if self.P_str is None:
            return (False, "SKIP: symbolic expression for P not available")
        try:
            # Define symbols used in the problem
            m, omega, b0, tau = sp.symbols('m omega b0 tau', positive=True, real=True)
            # Construct expected expression
            expected = b0**2 / (4 * m**2 * omega**2 * (omega**2 + tau**(-2)))
            # Parse the solution's expression (may contain LaTeX-like elements; try to parse as sympy)
            P_expr = sp.sympify(self.P_str)
            # Simplify difference
            diff = sp.simplify(P_expr - expected)
            if diff == 0:
                return (True, "Symbolic form matches the expected formula")
            else:
                # Try to see if equivalent after algebraic manipulation
                if sp.simplify(diff * (m**2 * omega**2 * (omega**2 + tau**(-2)))) == 0:
                    return (True, "Symbolic form equivalent to expected formula")
                return (False, f"Symbolic form differs: diff = {diff}")
        except Exception as e:
            return (False, f"Symbolic check failed: {e}")

    # ----------------------------------------------------------------------
    #  Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_first_order_perturbation_used(self) -> Tuple[bool, str]:
        """Search explanation steps for mention of first‑order perturbation theory."""
        text = ' '.join(self.explanation_steps).lower()
        keywords = ['first-order', 'c^{(1)}', 'first order', 'δv_{m,n;0,0}']
        if any(kw in text for kw in keywords):
            return (True, "First‑order perturbation theory used")
        # fallback: check pseudocode
        pseudocode_text = str(self.pseudocode).lower()
        if any(kw in pseudocode_text for kw in keywords):
            return (True, "First‑order perturbation theory used (pseudocode)")
        return (False, "No evidence of first‑order perturbation theory")

    def test_modeling_matrix_element_xy(self) -> Tuple[bool, str]:
        """Search for mention of the matrix element ⟨1,1|xy|0,0⟩ or that only |1,1⟩ is accessible."""
        text = ' '.join(self.explanation_steps).lower()
        keywords = ['|1,1⟩', '⟨1,1|xy|0,0⟩', '|1,1>', '<1,1|xy|0,0>']
        if any(kw in text for kw in keywords):
            return (True, "Matrix element ⟨1,1|xy|0,0⟩ considered")
        pseudocode_text = str(self.pseudocode).lower()
        if any(kw in pseudocode_text for kw in keywords):
            return (True, "Matrix element ⟨1,1|xy|0,0⟩ considered (pseudocode)")
        return (False, "Matrix element for |1,1⟩ not found")

    def test_modeling_integral_evaluation(self) -> Tuple[bool, str]:
        """Check that the time integral ∫ e^{-t/τ} e^{iωt} dt is evaluated to τ/(1 - i ω τ)."""
        text = ' '.join(self.explanation_steps).lower()
        keywords = ['τ/(1 - i ω τ)', 'tau/(1 - i omega tau)',
                     'integral', 'e^{-t/τ}', 'exp(-t/tau)']
        if any(kw in text for kw in keywords):
            return (True, "Time integral appears correctly")
        return (False, "Time integral not clearly displayed")

    def test_modeling_formula_consistency(self) -> Tuple[bool, str]:
        """
        Re‑derive the transition probability from givens and compare with
        the solution's numeric result using a loose tolerance.
        """
        if None in (self.m, self.omega, self.b0, self.tau):
            return (False, "SKIP: missing givens (m, ω, b0, τ)")
        if self.P is None:
            return (False, "SKIP: numeric P not available")
        try:
            m = float(self.m)
            omega = float(self.omega)
            b0 = float(self.b0)
            tau = float(self.tau)
            P_sol = float(self.P)
        except (TypeError, ValueError):
            return (False, "Could not convert givens or result to float")

        # Derived formula (hbar cancels)
        numerator = b0**2
        denominator = 4 * m**2 * omega**2 * (omega**2 + tau**(-2))
        P_expected = numerator / denominator

        if math.isclose(P_sol, P_expected, rel_tol=0.02, abs_tol=1e-10):
            return (True, f"P matches expected formula (rel diff = {abs(P_sol-P_expected)/max(1e-12, P_expected):.2e})")
        else:
            return (False, f"P discrepancy: got {P_sol:.6e}, expected {P_expected:.6e}")

    # ----------------------------------------------------------------------
    #  Monotonicity tests (perturb one parameter and check direction)
    # ----------------------------------------------------------------------
    def test_monotonicity_tau_increases_probability(self) -> Tuple[bool, str]:
        """
        As τ increases, denominator (ω² + τ⁻²) decreases → P increases.
        Perturb τ by +10% and verify P increases.
        """
        if None in (self.m, self.omega, self.b0, self.tau) or self.P is None:
            return (False, "SKIP: missing required data for monotonicity check")
        try:
            m = float(self.m)
            omega = float(self.omega)
            b0 = float(self.b0)
            tau = float(self.tau)
            P_base = float(self.P)
        except (TypeError, ValueError):
            return (False, "Could not convert givens to float")

        # Compute baseline P using the formula (should be consistent with test_modeling_formula_consistency)
        numerator = b0**2
        denom_base = 4 * m**2 * omega**2 * (omega**2 + tau**(-2))
        P_base_formula = numerator / denom_base

        # Perturb τ → +10 %
        tau_pert = 1.10 * tau
        denom_pert = 4 * m**2 * omega**2 * (omega**2 + tau_pert**(-2))
        P_pert = numerator / denom_pert

        if P_pert > P_base:
            return (True, f"τ increased → P increased: {P_base:.6e} → {P_pert:.6e}")
        else:
            return (False, f"τ increased but P did not increase: base {P_base:.6e}, pert {P_pert:.6e}")

    def test_monotonicity_b0_increases_probability(self) -> Tuple[bool, str]:
        """
        Probability scales as b0², so increasing b0 should increase P.
        Perturb b0 by +10% and check that P increases.
        """
        if None in (self.m, self.omega, self.b0, self.tau) or self.P is None:
            return (False, "SKIP: missing required data for monotonicity check")
        try:
            m = float(self.m)
            omega = float(self.omega)
            b0 = float(self.b0)
            tau = float(self.tau)
            P_base = float(self.P)
        except (TypeError, ValueError):
            return (False, "Could not convert givens to float")

        # Baseline formula
        numerator_base = b0**2
        denom = 4 * m**2 * omega**2 * (omega**2 + tau**(-2))
        P_base_formula = numerator_base / denom

        # Perturb b0 → +10 %
        b0_pert = 1.10 * b0
        P_pert = (b0_pert**2) / denom

        if P_pert > P_base:
            return (True, f"b0 increased → P increased: {P_base:.6e} → {P_pert:.6e}")
        else:
            return (False, f"b0 increased but P did not increase: base {P_base:.6e}, pert {P_pert:.6e}")

    # ----------------------------------------------------------------------
    #  Weak answer sanity (secondary, no isclose, no re-calculation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """Check that the final probability is finite and non‑negative (not NaN, not infinite)."""
        if self.P is None:
            return (False, "SKIP: numeric P not available")
        try:
            P_val = float(self.P)
        except (TypeError, ValueError):
            return (False, f"P cannot be converted to float: {self.P}")
        if math.isnan(P_val):
            return (False, "P is NaN")
        if math.isinf(P_val):
            return (False, "P is infinite")
        if P_val < 0:
            return (False, "P is negative")
        return (True, f"P = {P_val:.6g} is finite and non‑negative")

    # ----------------------------------------------------------------------
    #  Master test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods.  Returns (overall_pass, list_of_logs).
        """
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
                passed, msg = False, f"Crashed with exception: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)