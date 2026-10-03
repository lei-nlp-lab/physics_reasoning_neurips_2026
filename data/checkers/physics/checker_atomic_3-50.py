import sympy as sp
import math
import traceback
from typing import Any, Dict, List, Tuple, Callable

class PhysicsChecker:
    """
    Auditor for the pion decay problem.
    Checks interface completeness, theorem-based invariants,
    modeling consistency with explanation steps, monotonicity trends,
    and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # ---------- results (from pseudocode emit contract) ----------
        results = getattr(solution_instance, 'results', {})
        self.p_mu_max = results.get('p_mu_max', None)
        self.p_mu_min = results.get('p_mu_min', None)
        self.h_mu = results.get('h_mu', None)
        self.h_e = results.get('h_e', None)
        self.L_mu_conserved = results.get('L_mu_conserved', None)
        self.L_e_conserved = results.get('L_e_conserved', None)

        # ---------- givens (from problem context) ----------
        givens = getattr(solution_instance, 'givens', {})
        self.m_pi = givens.get('m_pi', None)
        self.m_mu = givens.get('m_mu', None)
        self.m_e = givens.get('m_e', None)
        self.p = givens.get('p', None)

        # Helicity inputs for invariant checks
        self.nu1_helicity = givens.get('nu1_helicity', None)
        self.nu2_helicity = givens.get('nu2_helicity', None)
        self.nu3_bar_helicity = givens.get('nu3_bar_helicity', None)

    # -----------------------------------------------------------------
    # Helper: safely parse a formula string (replace ^ by **)
    # -----------------------------------------------------------------
    @staticmethod
    def _parse_expr(expr_str: str) -> sp.Expr:
        """Convert a physics formula string to a sympy expression."""
        if expr_str is None:
            return None
        # Replace ^ with ** (exponentiation)
        cleaned = expr_str.replace('^', '**')
        try:
            return sp.sympify(cleaned)
        except Exception:
            return None

    # =================================================================
    # Interface tests
    # =================================================================
    def test_interface_keys_exist(self) -> Tuple[bool, str]:
        """All required result keys must exist (not None)."""
        missing = []
        for key in ['p_mu_max', 'p_mu_min', 'h_mu', 'h_e', 'L_mu_conserved', 'L_e_conserved']:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present.")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check types of result values."""
        errors = []
        # strings
        for key in ['p_mu_max', 'p_mu_min', 'h_mu', 'h_e']:
            val = getattr(self, key)
            if not isinstance(val, str):
                errors.append(f"'{key}' should be str, got {type(val).__name__}")
        # bools
        for key in ['L_mu_conserved', 'L_e_conserved']:
            val = getattr(self, key)
            if not isinstance(val, bool):
                errors.append(f"'{key}' should be bool, got {type(val).__name__}")
        if errors:
            return (False, " | ".join(errors))
        return (True, "All types correct.")

    # =================================================================
    # Theorem-based invariant tests
    # =================================================================
    def test_invariant_two_body_rest_frame_momentum(self) -> Tuple[bool, str]:
        """Check that p_mu_star = (m_pi^2 - m_mu^2) / (2*m_pi) is correctly used."""
        if self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: m_pi or m_mu not provided.")
        if self.p_mu_max is None or self.p_mu_min is None or self.p is None:
            return (False, "SKIP: p_mu_max, p_mu_min, or p not available.")
        m_pi, m_mu, p = sp.symbols('m_pi m_mu p', positive=True)
        p_star_exact = (m_pi**2 - m_mu**2) / (2*m_pi)
        p_max_expr = self._parse_expr(self.p_mu_max)
        p_min_expr = self._parse_expr(self.p_mu_min)
        if p_max_expr is None or p_min_expr is None:
            return (False, "Cannot parse p_mu_max or p_mu_min expressions.")
        # Reconstruct E_mu_star from sum formula
        derived_E_mu_star = sp.simplify((p_max_expr + p_min_expr) * m_pi / (2 * p))
        expected_E_mu_star = sp.sqrt(p_star_exact**2 + m_mu**2)
        # Pick a numeric test with p >> m_pi,m_mu
        m_pi_val = 139.57   # MeV
        m_mu_val = 105.66   # MeV
        p_val = 10000.0     # 10 GeV
        subs = {m_pi: m_pi_val, m_mu: m_mu_val, p: p_val}
        try:
            derived_num = float(derived_E_mu_star.subs(subs).evalf())
            expected_num = float(expected_E_mu_star.subs(subs).evalf())
            if math.isclose(derived_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "p_mu_star relation holds numerically.")
            else:
                return (False, f"p_mu_star mismatch: derived {derived_num:.4e}, expected {expected_num:.4e}")
        except Exception:
            # Fall back to symbolic simplification (may be approximate)
            diff = sp.simplify(derived_E_mu_star**2 - expected_E_mu_star**2)
            if diff == 0:
                return (True, "p_mu_star relation holds symbolically.")
            try:
                diff_series = sp.series(diff, p, sp.oo, 1).removeO()
                if diff_series == 0:
                    return (True, "p_mu_star relation holds to leading order.")
            except Exception:
                pass
            return (False, f"Cannot verify invariant: symbolic difference {diff}")

    def test_invariant_helicity_muon_vs_neutrino(self) -> Tuple[bool, str]:
        """Check that muon helicity is consistent with angular momentum conservation."""
        if self.h_mu is None or self.nu1_helicity is None:
            return (False, "SKIP: h_mu or nu1_helicity not provided.")
        if self.nu1_helicity != -1:
            return (False, "nu1_helicity is not -1; cannot perform invariant check.")
        expected_h_mu = "negative"
        if self.h_mu.lower() != expected_h_mu:
            return (False, f"h_mu = '{self.h_mu}', expected '{expected_h_mu}'.")
        return (True, "Helicity of muon is consistent with angular momentum conservation.")

    def test_invariant_helicity_positron_vs_neutrinos(self) -> Tuple[bool, str]:
        """Check that positron helicity is consistent with V‑A and given neutrino helicities."""
        if self.h_e is None or self.nu2_helicity is None or self.nu3_bar_helicity is None:
            return (False, "SKIP: h_e, nu2_helicity, or nu3_bar_helicity missing.")
        expected_h_e = "positive"
        if self.h_e.lower() != expected_h_e:
            return (False, f"h_e = '{self.h_e}', expected '{expected_h_e}'.")
        if self.nu2_helicity != -1:
            return (False, "nu2_helicity should be -1 (left‑handed).")
        if self.nu3_bar_helicity != 1:
            return (False, "nu3_bar_helicity should be +1 (right‑handed).")
        return (True, "Positron helicity consistent with V‑A and neutrino helicities.")

    def test_invariant_lepton_number_conservation(self) -> Tuple[bool, str]:
        """Check that lepton number conservation flags are set True."""
        if self.L_mu_conserved is None or self.L_e_conserved is None:
            return (False, "SKIP: lepton conservation flags not provided.")
        if not isinstance(self.L_mu_conserved, bool) or not isinstance(self.L_e_conserved, bool):
            return (False, "Lepton conservation flags must be bool.")
        if self.L_mu_conserved and self.L_e_conserved:
            return (True, "Lepton numbers separately conserved (both True).")
        return (False, "Lepton number conservation flags not both True.")

    # =================================================================
    # Modeling / trace tests (check consistency with explanation steps)
    # =================================================================
    def test_modeling_muon_momentum_max_form(self) -> Tuple[bool, str]:
        """Check that p_mu_max has the correct leading‑order form."""
        if self.p_mu_max is None or self.m_pi is None or self.m_mu is None or self.p is None:
            return (False, "SKIP: required symbols missing.")
        p_max_expr = self._parse_expr(self.p_mu_max)
        if p_max_expr is None:
            return (False, "Cannot parse p_mu_max expression.")
        p, m_pi, m_mu = sp.symbols('p m_pi m_mu', positive=True)
        expected = p + (m_pi**2 - m_mu**2) / (4*p)
        try:
            p_max_series = sp.series(p_max_expr, p, sp.oo, 2)
            expected_series = sp.series(expected, p, sp.oo, 2)
            diff = sp.simplify(p_max_series - expected_series)
            if diff == 0:
                return (True, "p_mu_max has correct leading-order form.")
            if diff.as_ordered_terms() and diff.coeff(1/p, 0) == 0:
                return (True, "p_mu_max series matches to leading order.")
            return (False, f"p_mu_max series mismatch: difference {diff}")
        except Exception as e:
            return (False, f"Series comparison failed: {e}")

    def test_modeling_muon_momentum_min_form(self) -> Tuple[bool, str]:
        """Check that p_mu_min has the correct leading‑order form."""
        if self.p_mu_min is None or self.m_pi is None or self.m_mu is None or self.p is None:
            return (False, "SKIP: required symbols missing.")
        p_min_expr = self._parse_expr(self.p_mu_min)
        if p_min_expr is None:
            return (False, "Cannot parse p_mu_min expression.")
        p, m_pi, m_mu = sp.symbols('p m_pi m_mu', positive=True)
        expected = (m_mu**2 / m_pi**2) * p - (m_pi**2 - m_mu**2) / (4*p)
        try:
            p_min_series = sp.series(p_min_expr, p, sp.oo, 2)
            expected_series = sp.series(expected, p, sp.oo, 2)
            diff = sp.simplify(p_min_series - expected_series)
            if diff == 0:
                return (True, "p_mu_min has correct leading-order form.")
            if diff.as_ordered_terms() and diff.coeff(1/p, 0) == 0:
                return (True, "p_mu_min series matches to leading order.")
            return (False, f"p_mu_min series mismatch: difference {diff}")
        except Exception as e:
            return (False, f"Series comparison failed: {e}")

    # =================================================================
    # Monotonicity tests (explicit perturbation)
    # =================================================================
    def test_monotonicity_p_mu_max_increases_with_p(self) -> Tuple[bool, str]:
        """Perturb p by +10% and check that p_mu_max increases."""
        if self.p_mu_max is None or self.m_pi is None or self.m_mu is None or self.p is None:
            return (False, "SKIP: required symbols missing.")
        expr = self._parse_expr(self.p_mu_max)
        if expr is None:
            return (False, "Cannot parse p_mu_max expression.")
        # Choose a numeric baseline: p=10000, m_pi=139.57, m_mu=105.66
        p, m_pi, m_mu = sp.symbols('p m_pi m_mu', positive=True)
        subs = {m_pi: 139.57, m_mu: 105.66, p: 10000.0}
        try:
            base_val = float(expr.subs(subs).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate baseline: {e}")
        # Perturb p by +10%
        subs_pert = subs.copy()
        subs_pert[p] = 11000.0
        try:
            perturbed_val = float(expr.subs(subs_pert).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate perturbed: {e}")
        if perturbed_val > base_val:
            return (True, f"p_mu_max increased from {base_val:.4e} to {perturbed_val:.4e} when p increased by 10%.")
        else:
            return (False, f"p_mu_max did not increase: baseline {base_val:.4e}, perturbed {perturbed_val:.4e}")

    def test_monotonicity_p_mu_min_increases_with_p(self) -> Tuple[bool, str]:
        """Perturb p by +10% and check that p_mu_min increases."""
        if self.p_mu_min is None or self.m_pi is None or self.m_mu is None or self.p is None:
            return (False, "SKIP: required symbols missing.")
        expr = self._parse_expr(self.p_mu_min)
        if expr is None:
            return (False, "Cannot parse p_mu_min expression.")
        p, m_pi, m_mu = sp.symbols('p m_pi m_mu', positive=True)
        subs = {m_pi: 139.57, m_mu: 105.66, p: 10000.0}
        try:
            base_val = float(expr.subs(subs).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate baseline: {e}")
        subs_pert = subs.copy()
        subs_pert[p] = 11000.0
        try:
            perturbed_val = float(expr.subs(subs_pert).evalf())
        except Exception as e:
            return (False, f"Cannot evaluate perturbed: {e}")
        if perturbed_val > base_val:
            return (True, f"p_mu_min increased from {base_val:.4e} to {perturbed_val:.4e} when p increased by 10%.")
        else:
            return (False, f"p_mu_min did not increase: baseline {base_val:.4e}, perturbed {perturbed_val:.4e}")

    # =================================================================
    # Weak answer sanity (only one, no re-derivation)
    # =================================================================
    def test_answer_sanity_output_sanity(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that all result strings are non-empty and
        helicity strings are valid (negative/positive).
        """
        # Check strings
        for name, val in [("p_mu_max", self.p_mu_max), ("p_mu_min", self.p_mu_min),
                          ("h_mu", self.h_mu), ("h_e", self.h_e)]:
            if val is None:
                return (False, f"SKIP: {name} is None.")
            if not isinstance(val, str) or len(val.strip()) == 0:
                return (False, f"{name} is not a non-empty string.")
        # Check helicity values
        allowed_helicities = {"negative", "positive"}
        if self.h_mu.lower() not in allowed_helicities:
            return (False, f"h_mu = '{self.h_mu}' is not a valid helicity string.")
        if self.h_e.lower() not in allowed_helicities:
            return (False, f"h_e = '{self.h_e}' is not a valid helicity string.")
        # Check bools exist
        if not isinstance(self.L_mu_conserved, bool):
            return (False, "L_mu_conserved is not bool.")
        if not isinstance(self.L_e_conserved, bool):
            return (False, "L_e_conserved is not bool.")
        return (True, "All output strings and bools are plausible.")

    # =================================================================
    # Runner
    # =================================================================
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods. Returns (all_pass, logs)."""
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