import math
import sympy as sp

class PhysicsChecker:
    """
    Audits the Stark effect for the ground state of hydrogen.
    Covers:
      - interface: required result keys
      - invariant: parity vanishing first-order shift, sign of second-order shift
      - modeling/trace: selection rules, sum over all states including continuum
      - monotonicity: perturbing field strength increases magnitude of second-order shift
    """
    
    def __init__(self, solution_instance):
        # Extract results safely, defaulting to None.
        results = getattr(solution_instance, 'results', {})
        self.first_order_shift = results.get('first_order_shift', None)
        self.second_order_shift = results.get('second_order_shift', None)
        
        # Extract givens for possible epsilon (field strength)
        givens = getattr(solution_instance, 'givens', None)
        self.epsilon = None
        if givens is not None and isinstance(givens, dict):
            self.epsilon = givens.get('epsilon', None)
        
        # Extract explanation steps and pseudocode for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        
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
    
    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """First_order_shift and second_order_shift must be present."""
        if self.first_order_shift is None:
            return (False, "Missing 'first_order_shift' in results")
        if self.second_order_shift is None:
            return (False, "Missing 'second_order_shift' in results")
        return (True, "Both required keys present")
    
    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_parity_first_order_zero(self):
        """
        First-order Stark shift must vanish due to parity.
        (Ground state even, operator z odd => expectation value zero.)
        """
        shift = self.first_order_shift
        if shift is None:
            return (False, "SKIP: first_order_shift is None")
        try:
            if isinstance(shift, sp.Basic):
                float_val = float(shift.evalf())
            else:
                float_val = float(shift)
            if not math.isclose(float_val, 0.0, rel_tol=1e-9, abs_tol=1e-12):
                return (False, f"first_order_shift = {float_val}, should be 0")
            return (True, f"first_order_shift = {float_val} ~ 0")
        except Exception as e:
            return (False, f"Cannot evaluate first_order_shift: {e}")
    
    def test_invariant_second_order_sign(self):
        """
        Second-order Stark shift must be negative (energy lowering).
        For ground state: E1 < En, denominator (E1 - En) negative,
        numerator positive, so total shift is negative.
        """
        shift = self.second_order_shift
        if shift is None:
            return (False, "SKIP: second_order_shift is None")
        try:
            if isinstance(shift, sp.Basic):
                # Check that the expression is not identically zero and its structure implies negativity
                if shift.is_zero:
                    return (False, "second_order_shift is identically zero, should be negative")
                # Perform substitution with positive dummy values for e and epsilon
                sub_dict = {}
                for s in shift.free_symbols:
                    if s.name == 'e':
                        sub_dict[s] = 1.0
                    elif s.name.lower() in ('epsilon', 'varepsilon', 'eps'):
                        sub_dict[s] = 1.0
                numeric_val = float(shift.subs(sub_dict).evalf()) if sub_dict else None
                if numeric_val is None:
                    # Fallback: check leading sign from string representation
                    s_str = str(shift)
                    if s_str.startswith('-') or s_str.startswith(' -'):
                        numeric_val = -1.0
                    else:
                        return (False, f"Cannot determine sign of symbolic expression: {shift}")
                if numeric_val > 0:
                    return (False, f"second_order_shift evaluates positive: {numeric_val}")
                return (True, f"second_order_shift is negative (evaluated as {numeric_val})")
            else:
                float_val = float(shift)
                if float_val > 0:
                    return (False, f"second_order_shift = {float_val}, should be <= 0")
                return (True, f"second_order_shift = {float_val} is non-positive")
        except Exception as e:
            return (False, f"Cannot evaluate second_order_shift sign: {e}")
    
    # ------------------------------------------------------------------
    # Modeling / trace tests (at least two)
    # ------------------------------------------------------------------
    def test_modeling_selection_rules_mentioned(self):
        """
        Explanation steps must mention dipole selection rules (Δl=±1, Δm=0)
        for the matrix elements ⟨nlm|z|100⟩.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        # Treat explanation_steps as a string or list of strings
        if isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        elif isinstance(self.explanation_steps, list):
            text = " ".join(self.explanation_steps)
        else:
            return (False, "SKIP: explanation_steps has unexpected type")
        
        keywords = ['Δl', 'l=1', 'l=0', 'dipole', 'selection', 'parity']
        found_any = any(kw in text for kw in keywords)
        if not found_any:
            return (False, "Explanation steps do not mention selection rules (dipole or parity)")
        return (True, "Selection rules mentioned in explanation")
    
    def test_modeling_sum_over_all_states(self):
        """
        The second-order shift expression (in explanation_steps or pseudocode)
        should include a sum over all excited states, including the continuum.
        """
        # Check both explanation_steps and pseudocode
        texts = []
        if self.explanation_steps:
            if isinstance(self.explanation_steps, str):
                texts.append(self.explanation_steps)
            elif isinstance(self.explanation_steps, list):
                texts.append(" ".join(self.explanation_steps))
        if self.pseudocode:
            if isinstance(self.pseudocode, str):
                texts.append(self.pseudocode)
            elif isinstance(self.pseudocode, (list, dict)):
                texts.append(str(self.pseudocode))
        if not texts:
            return (False, "SKIP: no explanation_steps or pseudocode available")
        
        full_text = " ".join(texts)
        # Look for phrases indicating sum over all states (including continuum)
        keywords = ['sum over', 'all states', 'continuum', 'n≠1', 'n != 1', 'n\\neq 1', 'excited states']
        found_any = any(kw in full_text for kw in keywords)
        if not found_any:
            return (False, "Explanation/pseudocode does not mention sum over all excited states/continuum")
        return (True, "Sum over all excited states (including continuum) mentioned")
    
    # ------------------------------------------------------------------
    # Monotonicity test (perturb field strength)
    # ------------------------------------------------------------------
    def test_monotonicity_second_order_field_strength(self):
        """
        Verify that increasing the electric field strength ε makes the second-order
        shift more negative (i.e., the magnitude increases). 
        Requires epsilon in givens and a symbolic or analytic expression.
        """
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available in givens")
        if self.second_order_shift is None:
            return (False, "SKIP: second_order_shift is None")
        
        try:
            shift_expr = self.second_order_shift
            if isinstance(shift_expr, sp.Basic):
                # Find the epsilon symbol
                eps_sym = None
                for s in shift_expr.free_symbols:
                    if s.name.lower() in ('epsilon', 'varepsilon', 'eps'):
                        eps_sym = s
                        break
                if eps_sym is None:
                    return (False, "SKIP: cannot identify epsilon symbol in expression")
                # Baseline value at original epsilon
                orig_val = float(shift_expr.subs({eps_sym: self.epsilon}).evalf())
                # Perturb epsilon by +10%
                pert_epsilon = self.epsilon * 1.1
                pert_val = float(shift_expr.subs({eps_sym: pert_epsilon}).evalf())
            else:
                # Numeric value; cannot perturb because scaling unknown
                return (False, "SKIP: second_order_shift is numeric, cannot vary epsilon")
            
            # Check monotonicity: baseline is negative; perturbed should be more negative
            if orig_val > 0:
                return (False, "Baseline shift is positive, cannot test monotonicity")
            if pert_val >= orig_val:
                return (False, f"Perturbed shift ({pert_val}) is not more negative than baseline ({orig_val})")
            return (True, f"Shift becomes more negative: {orig_val} -> {pert_val}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")
    
    # ------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_first_order_finite(self):
        """Weak sanity: first_order_shift should be finite and not NaN."""
        shift = self.first_order_shift
        if shift is None:
            return (False, "SKIP: first_order_shift is None")
        try:
            if isinstance(shift, sp.Basic):
                return (True, "first_order_shift is symbolic (finite by definition)")
            val = float(shift)
            if math.isnan(val) or math.isinf(val):
                return (False, f"first_order_shift is {val}")
            return (True, f"first_order_shift is finite")
        except Exception as e:
            return (False, f"Cannot check finiteness: {e}")