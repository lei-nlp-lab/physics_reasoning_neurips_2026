import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the radiation pressure problem.
    Expected result keys: p, p_form.
    """
    def __init__(self, solution_instance):
        # Safe extraction with None defaults
        self.results = getattr(solution_instance, 'results', {}) if hasattr(solution_instance, 'results') else {}
        self.p_str = self.results.get('p', None)
        self.p_form_str = self.results.get('p_form', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.givens = getattr(solution_instance, 'givens', None)

    # ----------------------------------------------------------------------
    # Interface checks – each only checks existence and type of one key
    # ----------------------------------------------------------------------
    def test_interface_p_key(self):
        if self.p_str is None:
            return (False, "SKIP: key 'p' not present in results")
        if not isinstance(self.p_str, str):
            return (False, f"key 'p' exists but is not a string (type: {type(self.p_str).__name__})")
        return (True, "key 'p' exists and is a string")

    def test_interface_p_form_key(self):
        if self.p_form_str is None:
            return (False, "SKIP: key 'p_form' not present in results")
        if not isinstance(self.p_form_str, str):
            return (False, f"key 'p_form' exists but is not a string (type: {type(self.p_form_str).__name__})")
        return (True, "key 'p_form' exists and is a string")

    # ----------------------------------------------------------------------
    # Invariant checks – theorem-based physical sanity (symbolic where applicable)
    # ----------------------------------------------------------------------
    def test_invariant_pressure_energy_relation(self):
        """
        (a) proves p = u/3 for isotropic radiation on a perfectly reflecting wall.
        Check that the emitted p formula satisfies this relation.
        """
        if self.p_str is None:
            return (False, "SKIP: cannot test invariant without p")
        try:
            u = sp.Symbol('u')
            # Parse the formula; assume it contains 'u' and represents p = expression
            # Accept forms like "p = u/3", "u/3", "p == u/3"
            expr = sp.sympify(self.p_str.replace("p", "").replace("=", "").replace("==", "").strip())
            target = u / 3
            # Check equivalence: expr - u/3 should simplify to 0
            diff = sp.simplify(expr - target)
            if diff == 0:
                return (True, "p formula matches expected p = u/3")
            else:
                return (False, f"p formula does not reduce to u/3; difference = {diff}")
        except Exception as e:
            return (False, f"Error parsing p string: {e}")

    def test_invariant_p_form_steffan_boltzmann_law(self):
        """
        (c) yields p ∝ T⁴. Check that the p_form string contains T^4 or T**4.
        Use symbolic parsing to confirm proportionality.
        """
        if self.p_form_str is None:
            return (False, "SKIP: cannot test without p_form")
        try:
            T = sp.Symbol('T')
            expr = sp.sympify(self.p_form_str.replace("p", "").replace("=", "").replace("==", "").strip())
            # Check that expr / T^4 is independent of T (i.e., only a constant)
            ratio = sp.simplify(expr / (T**4))
            if ratio.free_symbols == set():
                return (True, "p_form is proportional to T^4 (ratio is constant)")
            else:
                return (False, f"p_form is not purely proportional to T^4; extra symbols: {ratio.free_symbols}")
        except Exception as e:
            return (False, f"Error parsing p_form string: {e}")

    def test_invariant_thermodynamic_derivative_sign(self):
        """
        From dp/dT = 4p/T, given p>0, T>0, derivative should be positive
        for any positive constant. Check that the p_form expression has
        a positive derivative with respect to T.
        """
        if self.p_form_str is None:
            return (False, "SKIP: cannot test derivative without p_form")
        try:
            T = sp.Symbol('T', positive=True)
            expr = sp.sympify(self.p_form_str.replace("p", "").replace("=", "").replace("==", "").strip())
            d_expr = sp.diff(expr, T)
            # For T>0, derivative should be positive. Use numeric substitution.
            # Assume any unknown constants (like a) are positive.
            subs = {}
            for s in expr.free_symbols:
                if s != T:
                    subs[s] = 1.0  # arbitrary positive value
            val = float(d_expr.subs({T: 300.0, **subs}))
            if val > 0:
                return (True, f"Derivative positive at T=300: {val}")
            else:
                return (False, f"Derivative non-positive at T=300: {val}")
        except Exception as e:
            return (False, f"Error computing derivative: {e}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks – consistency with explanation steps and reasoning
    # ----------------------------------------------------------------------
    def test_modeling_uses_maxwell_relation(self):
        """
        Check that the explanation steps (if available) mention the Maxwell relation
        derived from Helmholtz free energy.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        steps_text = " ".join(self.explanation_steps).lower()
        keywords = ["maxwell", "free energy", "helmholtz", "dF", "-S dT - p dV"]
        found = [kw for kw in keywords if kw in steps_text]
        if len(found) >= 2:
            return (True, f"Explanation mentions Maxwell relation: {found}")
        else:
            return (False, f"Insufficient mention of Maxwell relation; found keywords: {found}")

    def test_modeling_includes_energy_density_reasoning(self):
        """
        (b) requires arguing that u depends only on T, using second law argument.
        Check that explanation steps discuss this.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        steps_text = " ".join(self.explanation_steps).lower()
        required_phrases = ["energy density", "temperature", "second law", "cavity"]
        missing = [p for p in required_phrases if p not in steps_text]
        if not missing:
            return (True, "Explanation covers cavity independence argument")
        else:
            return (False, f"Missing key phrases in explanation: {missing}")

    def test_trace_emitted_p(self):
        """
        Pseudocode contract expects results['p'] = 'p = u/3'.
        Check that the string matches exactly (case/whitespace insensitive).
        """
        if self.p_str is None:
            return (False, "SKIP: p key missing")
        expected = "p = u/3"
        # Normalize whitespace
        norm_actual = self.p_str.replace(" ", "").lower()
        norm_expected = expected.replace(" ", "").lower()
        if norm_actual == norm_expected:
            return (True, "Emitted p matches expected contract")
        else:
            return (False, f"Emitted p '{self.p_str}' does not match expected '{expected}'")

    def test_trace_emitted_p_form(self):
        """
        Pseudocode contract expects results['p_form'] = 'p = (1/3) a T^4'.
        Check that it contains the essential structure.
        """
        if self.p_form_str is None:
            return (False, "SKIP: p_form key missing")
        # Accept any form that has a/3 * T^4 or 1/3 a T^4 etc.
        # Simplified check: contains "T^4" or "T**4" and a constant factor
        text = self.p_form_str.replace(" ", "")
        has_T4 = "T^4" in text or "T**4" in text
        has_factor = "1/3" in text or "a/3" in text or "*a" in text
        if has_T4 and has_factor:
            return (True, "p_form contains T^4 and a factor like 1/3")
        else:
            return (False, f"p_form '{self.p_form_str}' does not match expected pattern")

    # ----------------------------------------------------------------------
    # Monotonicity check – explicit perturbation of temperature
    # ----------------------------------------------------------------------
    def test_monotonicity_p_increases_with_temperature(self):
        """
        Perturb temperature by +10% and verify that pressure increases.
        """
        if self.p_form_str is None:
            return (False, "SKIP: cannot test monotonicity without p_form")
        try:
            T = sp.Symbol('T')
            expr = sp.sympify(self.p_form_str.replace("p", "").replace("=", "").replace("==", "").strip())
            # Assign positive numeric values to any free symbols (except T)
            subs = {}
            for s in expr.free_symbols:
                if s != T:
                    subs[s] = 1.0  # arbitrary positive constant
            # Baseline: T = 300 K (or any positive number)
            T0 = 300.0
            p0 = float(expr.subs({T: T0, **subs}))
            # Perturbed: increase T by 10%
            T1 = T0 * 1.1
            p1 = float(expr.subs({T: T1, **subs}))
            if p1 > p0:
                return (True, f"Pressure increases with +10% temperature: p0={p0:.4f}, p1={p1:.4f}")
            else:
                return (False, f"Pressure does not increase: p0={p0:.4f}, p1={p1:.4f}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity – only structure check
    # ----------------------------------------------------------------------
    def test_answer_sanity_structure(self):
        """
        Final answer should contain a power of 4 in temperature.
        Do not re-derive; just check presence of T^4 or T**4.
        """
        if self.p_form_str is None:
            return (False, "SKIP: p_form not available")
        text = self.p_form_str.lower()
        if "t^4" in text or "t**4" in text:
            return (True, "Final answer contains T^4")
        else:
            return (False, "Final answer does not show T^4 dependence")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
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