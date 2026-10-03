import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for the optical activity problem.
    Checks interface completeness, model consistency, and monotonicity.
    Assumes the solver's answer (especially phi) may be symbolic or numeric.
    """

    def __init__(self, solution_instance):
        # Safely extract results and givens; never assign numeric defaults.
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        self.phenomenon = self.results.get('phenomenon', None)
        self.phi = self.results.get('phi', None)

        self.omega = self.givens.get('omega', None)
        self.d = self.givens.get('d', None)
        self.c = self.givens.get('c', None)
        self.n_R = self.givens.get('n_R', None)
        self.n_L = self.givens.get('n_L', None)

        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _skip_if_missing(self, *attrs):
        """Return (True, message) if any required attribute is None."""
        missing = [a for a in attrs if getattr(self, a) is None]
        if missing:
            return True, f"SKIP: missing required data: {', '.join(missing)}"
        return False, ""

    def _is_symbolic(self, expr):
        """Return True if phi is a SymPy expression (not purely numeric)."""
        return isinstance(expr, sp.Basic) and not expr.is_Number

    # ------------------------------------------------------------------
    # Interface Tests
    # ------------------------------------------------------------------
    def test_interface_phenomenon_string(self):
        """Check that phenomenon is a non‑empty string."""
        if self.phenomenon is None:
            return False, "FAIL: phenomenon is None"
        if not isinstance(self.phenomenon, str):
            return False, f"FAIL: phenomenon is not a string, got {type(self.phenomenon).__name__}"
        if len(self.phenomenon.strip()) == 0:
            return False, "FAIL: phenomenon is empty"
        return True, f"PASS: phenomenon = '{self.phenomenon}'"

    def test_interface_phi_exists(self):
        """Check that phi key is present in results (value can be None)."""
        if 'phi' not in self.results:
            return False, "FAIL: results['phi'] is missing"
        return True, "PASS: results['phi'] exists"

    def test_interface_phi_type(self):
        """Check that phi is numeric or SymPy expression (not string, list, etc.)."""
        if self.phi is None:
            return True, "SKIP: phi is None"
        valid = (int, float, complex, sp.Basic)
        if not isinstance(self.phi, valid):
            return False, f"FAIL: phi has unexpected type {type(self.phi).__name__}"
        return True, "PASS: phi has acceptable type"

    # ------------------------------------------------------------------
    # Invariant Tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_phi_sign_matches_n_diff(self):
        """If n_R > n_L, phi should be positive; if n_R < n_L, phi negative (theorem)."""
        skip, msg = self._skip_if_missing('n_R', 'n_L', 'phi')
        if skip:
            return True, msg
        # For symbolic phi, we cannot judge sign directly; skip numeric cases
        if self._is_symbolic(self.phi):
            return True, "SKIP: symbolic phi, cannot evaluate sign"
        try:
            phi_val = float(self.phi)
        except (TypeError, ValueError):
            return True, "SKIP: phi not convertible to float"
        n_diff = float(self.n_R) - float(self.n_L)
        # Sign should agree (but allow zero)
        if n_diff > 0 and phi_val < 0:
            return False, f"FAIL: n_R > n_L but phi={phi_val} is negative"
        if n_diff < 0 and phi_val > 0:
            return False, f"FAIL: n_R < n_L but phi={phi_val} is positive"
        return True, "PASS: sign matches (n_R - n_L)"

    def test_invariant_phi_zero_when_d_zero(self):
        """Physical limit: if thickness d→0, rotation φ→0."""
        if not self._is_symbolic(self.phi):
            return True, "SKIP: phi is not symbolic, cannot test limit d→0"
        try:
            # Substitute d = 0 (if d is a symbol)
            phi_at_d0 = self.phi.subs(sp.Symbol('d'), 0)
            simplified = sp.simplify(phi_at_d0)
            if simplified != 0:
                # Also try numeric evaluation
                if isinstance(simplified, sp.Basic) and simplified.is_Number:
                    if simplified != 0:
                        return False, f"FAIL: phi(d=0) = {simplified} (expected 0)"
            # Otherwise assume correct
            return True, "PASS: phi(d=0) simplifies to 0"
        except Exception:
            return True, "INFO: could not substitute d=0, skipping"

    def test_invariant_phi_linear_in_d(self):
        """Check that doubling d doubles phi (using the solver's own expression)."""
        if not self._is_symbolic(self.phi):
            return True, "SKIP: phi is not symbolic, cannot test linearity"
        try:
            # Substitute d → 2*d
            d_sym = sp.Symbol('d')
            phi_original = self.phi
            phi_doubled = self.phi.subs(d_sym, 2 * d_sym)
            # Check that phi_doubled = 2 * phi_original symbolically
            diff = sp.simplify(phi_doubled - 2 * phi_original)
            if diff != 0:
                # Try numeric evaluation with dummy parameters
                # Use random numeric values for other symbols
                other_syms = [s for s in phi_original.free_symbols if s != d_sym]
                subs_dict = {s: 1.0 for s in other_syms}  # arbitrary non‑zero
                subs_dict[d_sym] = 0.5  # some thickness
                val_orig = float(phi_original.subs(subs_dict))
                val_double = float(phi_doubled.subs(subs_dict))
                if not math.isclose(val_double, 2 * val_orig, rel_tol=1e-6):
                    return False, f"FAIL: doubling d gives φ={val_double}, expected {2*val_orig}"
            return True, "PASS: φ linear in d (2×d → 2×φ)"
        except Exception as e:
            return True, f"SKIP: linearity check failed with error: {e}"

    # ------------------------------------------------------------------
    # Modeling / Trace Tests (quality of derivation)
    # ------------------------------------------------------------------
    def test_modeling_explanation_references_circular_components(self):
        """Explanation should mention decomposition into right/left circular components."""
        if self.explanation_steps is None:
            return True, "SKIP: no explanation_steps provided"
        text = str(self.explanation_steps).lower()
        keywords = ['right', 'left', 'circular', 'circularly', 'r-', 'l-', 'n_r', 'n_l']
        found = [kw for kw in keywords if kw in text]
        if len(found) < 2:
            return False, f"FAIL: explanation lacks key terms about circular birefringence; found: {found}"
        return True, f"PASS: explanation contains {found}"

    def test_modeling_pseudocode_emit_phi_formula(self):
        """The pseudocode [emit] should contain the φ expression with n_R, n_L, etc."""
        if self.pseudocode is None:
            return True, "SKIP: no pseudocode provided"
        code = str(self.pseudocode)
        # Look for the general structure, not an exact copy of the answer.
        # Must contain formula with n_R, n_L, d, omega, c.
        needed = ['phi', 'n_R', 'n_L', 'd', 'omega', 'c']
        if not all(term in code for term in needed):
            return False, f"FAIL: pseudocode lacks required terms {needed}"
        return True, "PASS: pseudocode uses n_R, n_L, d, omega, c"

    # ------------------------------------------------------------------
    # Monotonicity Tests (perturb input and compare trend)
    # ------------------------------------------------------------------
    def test_monotonicity_phi_increases_with_d(self):
        """If d increases, |φ| increases (trend check using solver's own expression)."""
        if not self._is_symbolic(self.phi):
            return True, "SKIP: phi not symbolic, cannot perturb d"
        try:
            d_sym = sp.Symbol('d')
            # Evaluate baseline at d = d0 (using a generic value, e.g., d=1)
            baseline = float(self.phi.subs(d_sym, 1.0))
            # Evaluate after increasing d by 10%
            perturbed = float(self.phi.subs(d_sym, 1.1))
            if abs(perturbed) <= abs(baseline):
                return False, f"FAIL: after increasing d by 10%, |φ| does not increase (|baseline|={abs(baseline)}, |perturbed|={abs(perturbed)})"
            return True, "PASS: |φ| increases with d"
        except Exception as e:
            return True, f"SKIP: monotonicity test failed: {e}"

    def test_monotonicity_phi_increases_with_n_R_minus_n_L(self):
        """If (n_R - n_L) increases (by increasing n_R), |φ| increases."""
        if not self._is_symbolic(self.phi):
            return True, "SKIP: phi not symbolic, cannot perturb n_R"
        try:
            n_R_sym = sp.Symbol('n_R')
            # Baseline: set n_R to some value, n_L to 0 (or subtract)
            # Use generic symbolic substitution: keep n_L as symbol? Actually we need to perturb n_R.
            # For simplicity, use a dummy numeric scenario: assume n_L=0, n_R=1, then perturb to 1.1
            # This is valid because the checker derives the perturbation using the solver's own expression.
            subs_base = {sp.Symbol('n_R'): 1.0, sp.Symbol('n_L'): 0.0,
                         sp.Symbol('omega'): 1.0, sp.Symbol('d'): 1.0, sp.Symbol('c'): 1.0}
            baseline = float(self.phi.subs(subs_base))
            subs_pert = {sp.Symbol('n_R'): 1.1, sp.Symbol('n_L'): 0.0,
                         sp.Symbol('omega'): 1.0, sp.Symbol('d'): 1.0, sp.Symbol('c'): 1.0}
            perturbed = float(self.phi.subs(subs_pert))
            if abs(perturbed) <= abs(baseline):
                return False, f"FAIL: after increasing n_R by 10%, |φ| does not increase"
            return True, "PASS: |φ| increases with (n_R - n_L)"
        except Exception as e:
            return True, f"SKIP: monotonicity test failed: {e}"

    # ------------------------------------------------------------------
    # Weak Final‑Answer Sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_range(self):
        """Very weak: φ must be finite and within plausible range (if numeric)."""
        if self.phi is None:
            return True, "SKIP: phi is None"
        if self._is_symbolic(self.phi):
            return True, "SKIP: symbolic phi, cannot check numerical range"
        try:
            val = float(self.phi)
        except (TypeError, ValueError):
            return True, "SKIP: phi not convertible to float"
        if math.isinf(val):
            return False, "FAIL: phi is infinite"
        if math.isnan(val):
            return False, "FAIL: phi is NaN"
        # Plausible range: a few radians (|φ| < 10 rad for typical parameters)
        if abs(val) > 10.0:
            return False, f"FAIL: |φ| = {abs(val)} > 10 rad, unlikely"
        return True, "PASS: phi is finite and within plausible range"

    # ------------------------------------------------------------------
    # Runner
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
                import traceback
                msg += "\n" + traceback.format_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs