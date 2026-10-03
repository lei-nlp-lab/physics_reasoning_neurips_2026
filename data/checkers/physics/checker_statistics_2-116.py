import sympy as sp
import re
import math

class PhysicsChecker:
    """
    Audit checker for the canonical ensemble derivation:
    C_v = (1/(kT^2)) ⟨(E - ⟨E⟩)^2⟩.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely
        results = getattr(solution_instance, 'results', {})
        self.C_v = results.get('C_v', None) if isinstance(results, dict) else None
        # Explanation steps for context (not used for heavy hardcoding)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def run_all_tests(self):
        """Discover and run all test_* methods, return (all_pass, logs)."""
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

    # ---------- interface ----------
    def test_interface_C_v_key(self):
        """Result key 'C_v' must exist and be non-None."""
        if self.C_v is None:
            return (False, "Missing or None value for key 'C_v' in results.")
        return (True, "'C_v' key present and not None.")

    # ---------- invariant (theorem-based sanity) ----------
    def test_invariant_factor_one_over_kT2(self):
        """
        Fluctuation-dissipation theorem requires the factor 1/(kT^2).
        Check that the expression contains this factor.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v not available.")
        expr = str(self.C_v)
        pattern = r'1/([kK]\s*[Tt]\s*\^?\s*2|[kK][Tt]2|\{kT\^2\}|kT\^2)'
        if re.search(pattern, expr):
            return (True, "Expression contains the factor 1/(kT^2).")
        else:
            return (False, "Missing the 1/(kT^2) factor.")

    def test_invariant_variance_term(self):
        """
        The result must contain the variance term ⟨(E-⟨E⟩)^2⟩ or equivalent.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v not available.")
        expr = str(self.C_v)
        expr_clean = re.sub(r'\s+', '', expr)
        # Look for variance patterns
        has_variance = bool(re.search(r'\(E-.*?\)\^2|⟨.*?E.*?⟩\^2|⟨E\^2⟩-⟨E⟩\^2', expr_clean))
        if has_variance:
            return (True, "Expression includes a variance term ⟨(E-⟨E⟩)^2⟩.")
        else:
            return (False, "Expression does not contain a recognisable variance term.")

    # ---------- modeling / trace (at least two) ----------
    def test_modeling_chain_rule_factor(self):
        """
        The derivation uses the chain rule: C_v = (d⟨E⟩/dβ)*(dβ/dT).
        Check that the factor 1/(kT^2) appears explicitly (from dβ/dT).
        """
        if self.C_v is None:
            return (False, "SKIP: C_v not available.")
        expr = str(self.C_v)
        pattern = r'1/([kK]\s*[Tt]\s*\^?\s*2|[kK][Tt]2|\{kT\^2\}|kT\^2)'
        if re.search(pattern, expr):
            return (True, "Expression contains the chain‑rule factor 1/(kT^2).")
        else:
            return (False, "Missing the 1/(kT^2) factor from the chain rule.")

    def test_modeling_variance_structure(self):
        """
        The variance should be written as (E - ⟨E⟩)^2 inside expectation brackets.
        Check that the expression contains (E - \langle E \rangle)^2 or a LaTeX equivalent.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v not available.")
        expr = str(self.C_v)
        expr_clean = re.sub(r'\s+', '', expr)
        # Pattern for (E - \langle E \rangle)^2
        pattern = r'\(E\s*-\s*\\langle\s*E\s*\\rangle\s*\)\s*\^2'
        if re.search(pattern, expr_clean):
            return (True, "Variance written as (E - ⟨E⟩)^2.")
        else:
            # Also accept \langle (E - \langle E \rangle)^2 \rangle
            pattern2 = r'\\langle\s*\(E\s*-\s*\\langle\s*E\s*\\rangle\s*\)\s*\^2\s*\\rangle'
            if re.search(pattern2, expr_clean):
                return (True, "Variance written as \\langle (E - \\langle E \\rangle)^2 \\rangle.")
            else:
                return (False, "Variance not in the expected (E-⟨E⟩)^2 form.")

    def test_modeling_no_extraneous_variables(self):
        """
        The final expression should depend only on k, T, and the variance.
        Check that it does not contain 'N' or 'V' (particle number/volume) explicitly.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v not available.")
        expr = str(self.C_v)
        # Exclude cases where N or V appear as part of other symbols like "N_A" or "V_avg" – but keep simple.
        if re.search(r'\bN\b', expr) or re.search(r'\bV\b', expr):
            return (False, "Expression contains extraneous variables 'N' or 'V'.")
        return (True, "No extraneous variables N or V found.")

    # ---------- monotonicity ----------
    def test_monotonicity_variance_linearity(self):
        """
        C_v is proportional to the variance ⟨(E-⟨E⟩)^2⟩.
        Test linearity by substituting two variance values and checking scaling.
        If parsing fails, skip gracefully.
        """
        if self.C_v is None:
            return (False, "SKIP: C_v not available.")
        expr_str = str(self.C_v)
        # Try to replace the variance term with 'var'
        patterns = [
            r'\\langle\s*\(E\s*-\s*\\langle\s*E\s*\\rangle\s*\)\s*\^2\s*\\rangle',
            r'\\langle\s*E\^2\s*\\rangle\s*-\s*\\langle\s*E\s*\\rangle\s*\^2',
            r'\((E-\d?)\)\^2',
        ]
        replaced = False
        for pat in patterns:
            match = re.search(pat, expr_str)
            if match:
                new_expr_str = expr_str[:match.start()] + 'var' + expr_str[match.end():]
                replaced = True
                break
        if not replaced:
            return (False, "SKIP: Cannot isolate variance term for linearity test.")

        T, k = sp.symbols('T k')
        try:
            expr_sym = sp.sympify(new_expr_str, locals={'T': T, 'k': k})
        except Exception:
            return (False, "SKIP: Failed to parse expression after variance replacement.")

        subst = {T: 300, k: 1.380649e-23}
        try:
            val1 = expr_sym.subs({**subst, sp.Symbol('var'): 1}).evalf()
            val2 = expr_sym.subs({**subst, sp.Symbol('var'): 2}).evalf()
        except Exception as e:
            return (False, f"SKIP: Cannot evaluate numerically: {e}")

        if abs(val2 - 2*val1) < 1e-6 * (abs(val1) + abs(val2) + 1):
            return (True, f"C_v scales linearly with variance (ratio={float(val2/val1):.4f}).")
        else:
            return (False, f"Linearity test failed: ratio={float(val2/val1):.4f} (expected ~2).")

    # ---------- weak answer sanity (at most one) ----------
    def test_answer_sanity_format(self):
        """
        Weak check: the result is a non‑empty string with alphanumeric content.
        """
        if self.C_v is None:
            return (False, "C_v missing; cannot check format.")
        try:
            text = str(self.C_v)
        except Exception:
            return (False, "Cannot convert C_v to string.")
        if len(text.strip()) == 0:
            return (False, "C_v is empty string.")
        if re.search(r'[a-zA-Z0-9]', text):
            return (True, "C_v is a non‑empty string with alphanumeric content.")
        else:
            return (False, "C_v appears to contain no alphanumeric characters.")