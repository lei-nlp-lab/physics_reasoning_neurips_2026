import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics checker for angular momentum coupling problem:
    Expressing states |j,m> with m = l-1/2 in terms of uncoupled basis |l,m_l; s,m_s>.
    """

    def __init__(self, solution_instance):
        # Extract inputs safely
        givens = getattr(solution_instance, 'givens', {})
        self.l = givens.get('l', None)
        if self.l is None:
            self.l = getattr(solution_instance, 'l', None)
        if self.l is None:
            self.l = None

        results = getattr(solution_instance, 'results', {})
        # Expect state_1 and state_2 as lists of two coefficients: [c_{|l,l-1;1/2,1/2>}, c_{|l,l;1/2,-1/2>}]
        self.state_1 = results.get('state_1', None)
        self.state_2 = results.get('state_2', None)
        # If they are strings, attempt simple parse (only for numeric l)
        if isinstance(self.state_1, str):
            self.state_1 = self._parse_string_coeffs(self.state_1)
        if isinstance(self.state_2, str):
            self.state_2 = self._parse_string_coeffs(self.state_2)

        # Symbolic for easier checking
        self.l_sym = sp.Symbol('l', positive=True, integer=True)
        self.expected_state1 = [
            sp.sqrt(2*self.l_sym) / sp.sqrt(2*self.l_sym + 1),
            1 / sp.sqrt(2*self.l_sym + 1)
        ]
        self.expected_state2 = [
            -1 / sp.sqrt(2*self.l_sym + 1),
            sp.sqrt(2*self.l_sym) / sp.sqrt(2*self.l_sym + 1)
        ]

    def _parse_string_coeffs(self, s):
        """Attempt to parse a string like 'sqrt(2l/(2l+1))' into a list of two coefficients.
        If l is numeric, evaluate numerically. Otherwise returns None."""
        # Very basic: look for sqrt patterns and split by plus/minus
        # We assume the string is in the form "coeff1 * |...> + coeff2 * |...>"
        # For simplicity, we only handle numeric l or fallback to symbolic.
        # This is a minimal parser; in production, one would use sympy.parsing.
        return None  # Let tests fail gracefully

    def _evaluate_sympy_expr(self, expr):
        """If self.l is numeric, evaluate sympy expression numerically, else return symbolic."""
        if self.l is not None:
            try:
                return float(expr.subs(self.l_sym, self.l))
            except Exception:
                return None
        return expr

    def _check_coeff_list(self, coeffs, expected_sympy):
        """Check if coeffs (list of two) matches expected_sympy (list of sympy expressions)."""
        if coeffs is None or len(coeffs) != 2:
            return False, "Coefficient list missing or not length 2"
        # Convert to sympy numbers or expressions
        try:
            actual = [sp.nsimplify(c) for c in coeffs]
        except Exception:
            actual = coeffs
        # Compute difference vector
        diff = [sp.simplify(actual[i] - expected_sympy[i]) for i in range(2)]
        # Check if both differences are zero (symbolic) or numerically close
        zero1 = sp.simplify(diff[0])
        zero2 = sp.simplify(diff[1])
        if zero1 == 0 and zero2 == 0:
            return True, "Coefficients match exactly (symbolic)"
        # Try numeric evaluation
        if self.l is not None:
            try:
                val1 = float(zero1.subs(self.l_sym, self.l))
                val2 = float(zero2.subs(self.l_sym, self.l))
                if math.isclose(val1, 0, abs_tol=1e-2) and math.isclose(val2, 0, abs_tol=1e-2):
                    return True, f"Coefficients match within tolerance: {val1}, {val2}"
                else:
                    return False, f"Coefficient differences: {val1}, {val2}"
            except Exception as e:
                return False, f"Cannot evaluate numeric: {e}"
        # If l unknown, check symbolic zero
        if zero1 == 0 and zero2 == 0:
            return True, "Coefficients symbolically zero"
        return False, f"Coefficients do not match: {diff}"

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

    # Interface tests
    def test_interface_required_keys(self):
        if self.state_1 is None:
            return (False, "Missing required result: state_1")
        if self.state_2 is None:
            return (False, "Missing required result: state_2")
        if not isinstance(self.state_1, (list, tuple)) or len(self.state_1) != 2:
            return (False, "state_1 must be a list/tuple of two coefficients")
        if not isinstance(self.state_2, (list, tuple)) or len(self.state_2) != 2:
            return (False, "state_2 must be a list/tuple of two coefficients")
        return (True, "All required keys present and correctly structured")

    # Invariant tests (theorem-based physical sanity)
    def test_invariant_normalization_state1(self):
        if self.state_1 is None:
            return (False, "SKIP: state_1 not available")
        try:
            coeffs = [float(c) if not isinstance(c, sp.Expr) else c for c in self.state_1]
            if self.l is not None and any(isinstance(c, sp.Expr) for c in coeffs):
                coeffs = [float(c.subs(self.l_sym, self.l)) for c in coeffs]
            norm_sq = sum(c**2 for c in coeffs)
            if isinstance(norm_sq, sp.Expr):
                norm_sq_simpl = sp.simplify(norm_sq)
                if norm_sq_simpl == 1:
                    return (True, "Normalization symbolically equals 1")
                elif self.l is not None:
                    norm_val = float(norm_sq_simpl.subs(self.l_sym, self.l))
                    if math.isclose(norm_val, 1, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Normalization close to 1: {norm_val}")
                    else:
                        return (False, f"Normalization not 1: {norm_val}")
                else:
                    return (False, "Cannot evaluate normalization without numeric l")
            else:
                if math.isclose(norm_sq, 1, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Normalization close to 1: {norm_sq}")
                else:
                    return (False, f"Normalization not 1: {norm_sq}")
        except Exception as e:
            return (False, f"Error in normalization test: {e}")

    def test_invariant_normalization_state2(self):
        if self.state_2 is None:
            return (False, "SKIP: state_2 not available")
        try:
            coeffs = [float(c) if not isinstance(c, sp.Expr) else c for c in self.state_2]
            if self.l is not None and any(isinstance(c, sp.Expr) for c in coeffs):
                coeffs = [float(c.subs(self.l_sym, self.l)) for c in coeffs]
            norm_sq = sum(c**2 for c in coeffs)
            if isinstance(norm_sq, sp.Expr):
                norm_sq_simpl = sp.simplify(norm_sq)
                if norm_sq_simpl == 1:
                    return (True, "Normalization symbolically equals 1")
                elif self.l is not None:
                    norm_val = float(norm_sq_simpl.subs(self.l_sym, self.l))
                    if math.isclose(norm_val, 1, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Normalization close to 1: {norm_val}")
                    else:
                        return (False, f"Normalization not 1: {norm_val}")
                else:
                    return (False, "Cannot evaluate normalization without numeric l")
            else:
                if math.isclose(norm_sq, 1, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Normalization close to 1: {norm_sq}")
                else:
                    return (False, f"Normalization not 1: {norm_sq}")
        except Exception as e:
            return (False, f"Error in normalization test: {e}")

    def test_invariant_orthogonality(self):
        if self.state_1 is None or self.state_2 is None:
            return (False, "SKIP: state_1 or state_2 not available")
        try:
            dot = 0
            for c1, c2 in zip(self.state_1, self.state_2):
                c1_num = float(c1) if not isinstance(c1, sp.Expr) else float(c1.subs(self.l_sym, self.l)) if self.l is not None else c1
                c2_num = float(c2) if not isinstance(c2, sp.Expr) else float(c2.subs(self.l_sym, self.l)) if self.l is not None else c2
                dot += c1_num * c2_num
            if isinstance(dot, sp.Expr):
                dot_simpl = sp.simplify(dot)
                if dot_simpl == 0:
                    return (True, "Orthogonality symbolically satisfied")
                elif self.l is not None:
                    dot_val = float(dot_simpl.subs(self.l_sym, self.l))
                    if math.isclose(dot_val, 0, abs_tol=1e-2):
                        return (True, f"Orthogonality close to 0: {dot_val}")
                    else:
                        return (False, f"Orthogonality not zero: {dot_val}")
                else:
                    return (False, "Cannot evaluate orthogonality without numeric l")
            else:
                if math.isclose(dot, 0, abs_tol=1e-2):
                    return (True, f"Orthogonality close to 0: {dot}")
                else:
                    return (False, f"Orthogonality not zero: {dot}")
        except Exception as e:
            return (False, f"Error in orthogonality test: {e}")

    # Modeling tests (based on explanation steps)
    def test_modeling_state1_coefficient_ratio(self):
        if self.state_1 is None or self.l is None:
            return (False, "SKIP: state_1 or l not available")
        if len(self.state_1) != 2:
            return (False, "state_1 coefficients not length 2")
        try:
            coeff1 = float(self.state_1[0])
            coeff2 = float(self.state_1[1])
            if coeff2 == 0:
                return (False, "Denominator coefficient zero")
            ratio = coeff1 / coeff2
            expected_ratio = math.sqrt(2 * self.l)
            if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
                return (True, f"Ratio {ratio} matches expected {expected_ratio}")
            else:
                return (False, f"Ratio {ratio} does not match expected {expected_ratio}")
        except Exception as e:
            return (False, f"Error in ratio test: {e}")

    def test_modeling_state2_coefficient_ratio(self):
        if self.state_2 is None or self.l is None:
            return (False, "SKIP: state_2 or l not available")
        if len(self.state_2) != 2:
            return (False, "state_2 coefficients not length 2")
        try:
            # Expected ratio: coeff for |l,l;...> / coeff for |l,l-1;...> = (sqrt(2l)/sqrt(2l+1)) / (-1/sqrt(2l+1)) = -sqrt(2l)
            coeff1 = float(self.state_2[0])  # coefficient for |l,l-1;...>
            coeff2 = float(self.state_2[1])  # coefficient for |l,l;...>
            if coeff1 == 0:
                return (False, "Denominator coefficient zero")
            ratio = coeff2 / coeff1
            expected_ratio = -math.sqrt(2 * self.l)
            if math.isclose(ratio, expected_ratio, rel_tol=1e-2):
                return (True, f"Ratio {ratio} matches expected {expected_ratio}")
            else:
                return (False, f"Ratio {ratio} does not match expected {expected_ratio}")
        except Exception as e:
            return (False, f"Error in ratio test: {e}")

    # Monotonicity tests (explicit perturbation of l)
    def test_monotonicity_state1_first_coeff_increases(self):
        """Perturb l by +1 and check that the first coefficient of state_1 increases."""
        if self.l is None:
            return (False, "SKIP: l not available")
        try:
            # Get baseline coefficient (first coeff of state_1)
            baseline = float(self.state_1[0])
            # Perturb l by +1 (use l+1)
            l_pert = self.l + 1
            # Compute expected coefficient at l_pert from symbolic formula
            # Use symbolic expression for first coefficient: sqrt(2l)/sqrt(2l+1)
            expr = sp.sqrt(2*self.l_sym) / sp.sqrt(2*self.l_sym + 1)
            expected_perturbed = float(expr.subs(self.l_sym, l_pert))
            if expected_perturbed > baseline:
                return (True, f"Perturbed coefficient {expected_perturbed} > baseline {baseline}")
            else:
                return (False, f"Perturbed coefficient {expected_perturbed} <= baseline {baseline}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    def test_monotonicity_state2_second_coeff_increases(self):
        """Perturb l by +1 and check that the second coefficient of state_2 increases."""
        if self.l is None:
            return (False, "SKIP: l not available")
        try:
            # Baseline second coefficient of state_2
            baseline = float(self.state_2[1])
            l_pert = self.l + 1
            # Symbolic expression for second coefficient of state_2: sqrt(2l)/sqrt(2l+1) (same formula)
            expr = sp.sqrt(2*self.l_sym) / sp.sqrt(2*self.l_sym + 1)
            expected_perturbed = float(expr.subs(self.l_sym, l_pert))
            if expected_perturbed > baseline:
                return (True, f"Perturbed coefficient {expected_perturbed} > baseline {baseline}")
            else:
                return (False, f"Perturbed coefficient {expected_perturbed} <= baseline {baseline}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    # Weak answer sanity (only one method, combine sign and NaN checks)
    def test_answer_sanity(self):
        """Weak sanity: check signs and finite values for both states."""
        if self.state_1 is None or self.state_2 is None or self.l is None:
            return (False, "SKIP: state data or l not available")
        try:
            c1_s1 = float(self.state_1[0])
            c2_s1 = float(self.state_1[1])
            c1_s2 = float(self.state_2[0])
            c2_s2 = float(self.state_2[1])
            # Check finite
            for val, name in [(c1_s1, "state_1[0]"), (c2_s1, "state_1[1]"), (c1_s2, "state_2[0]"), (c2_s2, "state_2[1]")]:
                if math.isnan(val):
                    return (False, f"{name} is NaN")
                if math.isinf(val):
                    return (False, f"{name} is infinite")
            # Check sign patterns
            # state_1: both positive
            if not (c1_s1 > 0 and c2_s1 > 0):
                return (False, f"state_1 signs wrong: {c1_s1}, {c2_s1} (both should be positive)")
            # state_2: first negative, second positive
            if not (c1_s2 < 0 and c2_s2 > 0):
                return (False, f"state_2 signs wrong: {c1_s2}, {c2_s2} (first negative, second positive)")
            return (True, "All coefficients finite, signs correct")
        except Exception as e:
            return (False, f"Error in answer sanity: {e}")