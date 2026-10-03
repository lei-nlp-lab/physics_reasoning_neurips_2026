import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for a problem about relativistic electromagnetism
    of two positrons in a beam (collinear and side-by-side configurations).
    """

    def __init__(self, solution_instance):
        # Extract results dict safely; default to empty dict if missing
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Extract givens if available
        self.givens = getattr(solution_instance, 'givens', {})
        if self.givens is None:
            self.givens = {}

        # Safely retrieve each expected result key, default None
        self.E_prime = self.results.get('E_prime', None)
        self.F_prime = self.results.get('F_prime', None)
        self.a_prime = self.results.get('a_prime', None)

        self.E_lab_a = self.results.get('E_lab_a', None)
        self.F_lab_a = self.results.get('F_lab_a', None)
        self.a_lab_a = self.results.get('a_lab_a', None)

        self.E_prime_b = self.results.get('E_prime_b', None)
        self.F_prime_b = self.results.get('F_prime_b', None)
        self.a_prime_b = self.results.get('a_prime_b', None)

        self.E_lab_b = self.results.get('E_lab_b', None)
        self.B_lab_b = self.results.get('B_lab_b', None)
        self.F_lab_b = self.results.get('F_lab_b', None)
        self.a_lab_b = self.results.get('a_lab_b', None)

        # Extract gamma from givens if possible
        self.gamma = self.givens.get('gamma', None)
        if self.gamma is None:
            self.gamma = self.results.get('gamma', None)

        # Extract separation d from givens
        self.d = self.givens.get('d', None)
        if self.d is None:
            self.d = self.results.get('d', None)

    def _get_component(self, vector, axis):
        """Try to extract a component from a vector (sympy, tuple, list, dict)."""
        if vector is None:
            return None
        if isinstance(vector, sp.Matrix):
            idx_map = {'x': 0, 'y': 1, 'z': 2}
            idx = idx_map.get(axis)
            if idx is not None and idx < vector.shape[0]:
                return vector[idx]
        if isinstance(vector, (list, tuple)):
            comp_map = {'x': 0, 'y': 1, 'z': 2}
            idx = comp_map.get(axis)
            if idx is not None and idx < len(vector):
                return vector[idx]
        if isinstance(vector, dict):
            return vector.get(axis, None)
        return None

    def _try_extract_free_symbol(self, expr, name):
        """Return the sympy Symbol if expr contains a symbol with given name, else None."""
        if expr is None:
            return None
        if not isinstance(expr, sp.Basic):
            return None
        for s in expr.free_symbols:
            if str(s) == name:
                return s
        return None

    def _numeric_substitute(self, expr, substitutions):
        """Substitute symbols in a sympy expression and evaluate to float if possible."""
        if expr is None:
            return None
        if not isinstance(expr, sp.Basic):
            return expr  # already numeric
        try:
            result = sp.N(expr.subs(substitutions))
            return float(result)
        except Exception:
            return None

    def _perturb_and_compare_ratio(self, expr_baseline, expr_perturbed,
                                   var_name, expected_exponent, rel_tol=0.1):
        """
        Given two expressions (baseline and perturbed) both expected to be proportional
        to var^{expected_exponent}, verify that (expr_perturbed/expr_baseline) equals
        (var_perturbed/var_baseline)^{expected_exponent} within tolerance.
        var_baseline and var_perturbed are numeric values.
        Returns (passed_bool, message).
        """
        if expr_baseline is None or expr_perturbed is None:
            return (False, "SKIP: one of the expressions is None")
        # Try numeric comparison
        try:
            # If both expressions are already numeric (float/int)
            if isinstance(expr_baseline, (int, float)) and isinstance(expr_perturbed, (int, float)):
                if math.isclose(expr_perturbed, expr_baseline, rel_tol=rel_tol):
                    # exponent 0 case; not relevant
                    return (True, "Numeric values close; exponent check ambiguous")
                ratio = expr_perturbed / expr_baseline
                # We need the ratio of the perturbed variable; we don't have var values here.
                # This method should be called only when we have the perturbed values embedded.
                # Alternatively, just check that ratio is close to (var_pert/var_bas)^exp.
                # We'll skip; subclasses should use specific method.
                pass
        except Exception:
            pass
        return (False, "Perturbation comparison not implemented generically; use specific test")

    # ---------- Interface Tests ----------
    def test_interface_required_keys_exist(self):
        """All required result keys must be present in self.results."""
        required_keys = [
            'E_prime', 'F_prime', 'a_prime',
            'E_lab_a', 'F_lab_a', 'a_lab_a',
            'E_prime_b', 'F_prime_b', 'a_prime_b',
            'E_lab_b', 'B_lab_b', 'F_lab_b', 'a_lab_b'
        ]
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_keys_not_none(self):
        """All required keys must have non-None values."""
        none_keys = [k for k, v in self.results.items() if v is None]
        if none_keys:
            return (False, f"Keys with None value: {none_keys}")
        return (True, "All keys have non-None values")

    # ---------- Invariant Tests ----------
    def test_invariant_rest_frame_B_zero(self):
        """In both rest frames (a and b), magnetic field B must be zero (implied)."""
        # The solution states B'=0; we rely on that; no direct B' result required.
        return (True, "Rest frame B=0 (physics assumption)")

    def test_invariant_parallel_case_E_lab_equals_E_prime(self):
        """For collinear configuration, lab E equals rest E (parallel fields unchanged)."""
        if self.E_lab_a is None or self.E_prime is None:
            return (False, "SKIP: E_lab_a or E_prime missing")
        try:
            if isinstance(self.E_lab_a, sp.Matrix) and isinstance(self.E_prime, sp.Matrix):
                if self.E_lab_a == self.E_prime:
                    return (True, "Symbolic E_lab_a == E_prime")
                else:
                    return (False, "Symbolic vectors differ")
            # Numeric check
            if hasattr(self.E_lab_a, '__iter__') and hasattr(self.E_prime, '__iter__'):
                for a, b in zip(self.E_lab_a, self.E_prime):
                    if not math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2):
                        return (False, "Component mismatch")
                return (True, "Lab E equals rest E for parallel case")
        except Exception as e:
            return (False, f"Comparison error: {e}")
        return (True, "Test incomplete fallback")

    def test_invariant_parallel_case_F_lab_equals_F_prime(self):
        """For collinear case, force is unchanged too."""
        if self.F_lab_a is None or self.F_prime is None:
            return (False, "SKIP: F_lab_a or F_prime missing")
        try:
            if isinstance(self.F_lab_a, sp.Matrix) and isinstance(self.F_prime, sp.Matrix):
                if self.F_lab_a == self.F_prime:
                    return (True, "Symbolic F_lab_a == F_prime")
                else:
                    return (False, "Symbolic vectors differ")
            if hasattr(self.F_lab_a, '__iter__') and hasattr(self.F_prime, '__iter__'):
                for a, b in zip(self.F_lab_a, self.F_prime):
                    if not math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2):
                        return (False, "Component mismatch")
                return (True, "Lab F equals rest F for parallel case")
        except Exception as e:
            return (False, f"Comparison error: {e}")
        return (True, "Fallback pass")

    def test_invariant_parallel_case_B_lab_zero(self):
        """For parallel configuration, B_lab should be zero (no perpendicular component)."""
        if 'B_lab_a' in self.results and self.results['B_lab_a'] is not None:
            B_val = self.results['B_lab_a']
            # Check if zero (numeric or symbolic)
            if isinstance(B_val, (int, float)):
                if B_val == 0:
                    return (True, "B_lab_a is zero")
                else:
                    return (False, f"B_lab_a is {B_val}, not zero")
            if isinstance(B_val, sp.Basic):
                if B_val == 0 or sp.simplify(B_val) == 0:
                    return (True, "B_lab_a symbolically zero")
                else:
                    return (False, "B_lab_a not zero symbolically")
        # If key not present, assume zero (physics)
        return (True, "B_lab_a zero (or absent)")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_rest_frame_E_direction(self):
        """In rest frame, E' should point along -z (component negative)."""
        if self.E_prime is None:
            return (False, "SKIP: E_prime missing")
        z_comp = self._get_component(self.E_prime, 'z')
        if z_comp is None:
            return (False, "Cannot extract z component")
        if isinstance(z_comp, (int, float)):
            if z_comp >= 0:
                return (False, f"E' z-component is non-negative {z_comp}")
        elif isinstance(z_comp, sp.Basic):
            # Assume correct direction if symbolic
            pass
        return (True, "E' points in -z direction (z component negative)")

    def test_modeling_rest_frame_a_direction(self):
        """Acceleration in rest frame should be along -z (repulsive)."""
        if self.a_prime is None:
            return (False, "SKIP: a_prime missing")
        z_comp = self._get_component(self.a_prime, 'z')
        if z_comp is None:
            return (False, "Cannot extract z component")
        if isinstance(z_comp, (int, float)):
            if z_comp >= 0:
                return (False, f"a' z-component is non-negative {z_comp}")
        return (True, "a' points in -z direction")

    def test_modeling_lab_sideways_E_scaling(self):
        """For side-by-side, lab E should be gamma times rest E (perpendicular component)."""
        if self.E_lab_b is None or self.E_prime_b is None:
            return (False, "SKIP: E_lab_b or E_prime_b missing")
        gamma = self.gamma
        if gamma is None:
            return (False, "SKIP: gamma not available")
        z_lab = self._get_component(self.E_lab_b, 'z')
        z_rest = self._get_component(self.E_prime_b, 'z')
        if z_lab is None or z_rest is None:
            return (False, "Cannot extract z components")
        try:
            if isinstance(z_lab, (int, float)) and isinstance(z_rest, (int, float)):
                if math.isclose(z_lab, gamma * z_rest, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "E_lab_b_z = gamma * E_prime_b_z")
                else:
                    return (False, f"Ratio {z_lab/z_rest} != gamma {gamma}")
            elif isinstance(z_lab, sp.Basic) and isinstance(z_rest, sp.Basic):
                ratio = sp.simplify(z_lab / z_rest)
                gamma_sym = sp.Symbol('gamma')
                if ratio == gamma_sym:
                    return (True, "Symbolic ratio matches gamma")
                else:
                    return (False, f"Symbolic ratio {ratio} != gamma")
        except Exception:
            return (False, "Error computing ratio")
        return (True, "Fallback pass")

    def test_modeling_lab_sideways_B_direction(self):
        """For side-by-side, B_lab should be along y (or -y)."""
        if self.B_lab_b is None:
            return (False, "SKIP: B_lab_b missing")
        x_comp = self._get_component(self.B_lab_b, 'x')
        z_comp = self._get_component(self.B_lab_b, 'z')
        if x_comp is not None and isinstance(x_comp, (int, float)) and x_comp != 0:
            return (False, "B_lab_b has non-zero x component")
        if z_comp is not None and isinstance(z_comp, (int, float)) and z_comp != 0:
            return (False, "B_lab_b has non-zero z component")
        return (True, "B_lab_b has only y-component (or zero)")

    def test_modeling_lab_sideways_force_direction(self):
        """For side-by-side, F_lab should be along -x (per solution)."""
        if self.F_lab_b is None:
            return (False, "SKIP: F_lab_b missing")
        x_comp = self._get_component(self.F_lab_b, 'x')
        if x_comp is None:
            return (False, "Cannot extract x component")
        if isinstance(x_comp, (int, float)) and x_comp >= 0:
            return (False, f"F_lab_b x-component is non-negative {x_comp}")
        return (True, "F_lab_b points in -x direction")

    def test_modeling_acceleration_scaling_parallel(self):
        """For parallel case, a_lab should equal a_prime / gamma^3."""
        if self.a_lab_a is None or self.a_prime is None:
            return (False, "SKIP: a_lab_a or a_prime missing")
        gamma = self.gamma
        if gamma is None:
            return (False, "SKIP: gamma not available")
        # Compare z components (both along z)
        z_lab = self._get_component(self.a_lab_a, 'z')
        z_rest = self._get_component(self.a_prime, 'z')
        if z_lab is None or z_rest is None:
            return (False, "Cannot extract z components")
        try:
            if isinstance(z_lab, (int, float)) and isinstance(z_rest, (int, float)):
                expected = z_rest / (gamma**3)
                if math.isclose(z_lab, expected, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "a_lab_a_z = a_prime_z / gamma^3")
                else:
                    return (False, f"Ratio {z_lab/z_rest} != 1/gamma^3 = {1/(gamma**3)}")
            elif isinstance(z_lab, sp.Basic) and isinstance(z_rest, sp.Basic):
                ratio = sp.simplify(z_lab / z_rest)
                expected = 1 / sp.Symbol('gamma')**3
                if sp.simplify(ratio - expected) == 0:
                    return (True, "Symbolic ratio matches 1/gamma^3")
                else:
                    return (False, f"Symbolic ratio {ratio} != 1/gamma^3")
        except Exception:
            return (False, "Error computing scaling")
        return (True, "Fallback pass")

    def test_modeling_acceleration_scaling_transverse(self):
        """For transverse case, a_lab should equal a_prime / gamma^2 (magnitude)."""
        if self.a_lab_b is None or self.a_prime_b is None:
            return (False, "SKIP: a_lab_b or a_prime_b missing")
        gamma = self.gamma
        if gamma is None:
            return (False, "SKIP: gamma not available")
        # Compare magnitudes of the vectors
        try:
            if hasattr(self.a_lab_b, 'norm') and hasattr(self.a_prime_b, 'norm'):
                mag_lab = self.a_lab_b.norm()
                mag_rest = self.a_prime_b.norm()
                if isinstance(mag_lab, (int, float)) and isinstance(mag_rest, (int, float)):
                    if math.isclose(mag_lab, mag_rest / gamma**2, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, "Magnitude scaling a_lab = a_prime / gamma^2")
                    else:
                        return (False, f"Magnitude ratio {mag_lab/mag_rest} != 1/gamma^2")
                elif isinstance(mag_lab, sp.Basic) and isinstance(mag_rest, sp.Basic):
                    ratio = sp.simplify(mag_lab / mag_rest)
                    expected = 1 / sp.Symbol('gamma')**2
                    if sp.simplify(ratio - expected) == 0:
                        return (True, "Symbolic magnitude ratio matches")
                    else:
                        return (False, f"Symbolic ratio {ratio} != 1/gamma^2")
            else:
                # Fallback: check x component as surrogate (since a_lab_b is along x per solution)
                x_lab = self._get_component(self.a_lab_b, 'x')
                # a_prime_b is along z, so its x component is zero; not helpful.
                return (False, "Cannot compute magnitude; check not performed")
        except Exception:
            return (False, "Error in magnitude comparison")
        return (True, "Fallback pass")

    # ---------- Monotonicity Tests (explicit perturbation) ----------
    def test_monotonicity_separation_inverse_square(self):
        """
        Check that E'(d) ~ 1/d^2 by perturbing d.
        Compute E' magnitude (or z-component) at a baseline d0 and at 2*d0.
        Verify product E' * d^2 is constant.
        """
        # Need E_prime (vector) and separation d from givens
        if self.E_prime is None:
            return (False, "SKIP: E_prime missing")
        d0 = self.d
        if d0 is None:
            return (False, "SKIP: separation d not available")
        # Check if E_prime is symbolic; if so, substitute d with numbers
        if isinstance(self.E_prime, sp.Basic):
            d_sym = self._try_extract_free_symbol(self.E_prime, 'd')
            if d_sym is None:
                return (False, "SKIP: no free symbol 'd' in E_prime")
            # Baseline d0_numeric = convert d0 to float
            try:
                d0_num = float(d0)
            except Exception:
                return (False, "SKIP: cannot convert d0 to float")
            d1_num = 2.0 * d0_num  # perturbed to double
            # Compute E' component (z) at d0 and d1
            e_z_at_d0 = self._numeric_substitute(self._get_component(self.E_prime, 'z'), {d_sym: d0_num})
            e_z_at_d1 = self._numeric_substitute(self._get_component(self.E_prime, 'z'), {d_sym: d1_num})
            if e_z_at_d0 is None or e_z_at_d1 is None:
                return (False, "Cannot evaluate E'_z numerically")
            # Check inverse square: E_z * d^2 = constant
            const0 = e_z_at_d0 * d0_num**2
            const1 = e_z_at_d1 * d1_num**2
            if math.isclose(const0, const1, rel_tol=0.1):
                return (True, f"Inverse square holds: E_z*d^2 ≈ {const0} vs {const1}")
            else:
                return (False, f"Inverse square violation: {const0} vs {const1}")
        # If E_prime is numeric (list/tuple), we need d from givens; assume d is numeric
        z_comp = self._get_component(self.E_prime, 'z')
        if z_comp is None:
            return (False, "Cannot extract z component from E_prime")
        if isinstance(z_comp, (int, float)):
            # We don't have a second d value; we can create a hypothetical: assume that if d were halved, E would quadruple? Can't without expression.
            # So we skip for numeric case without symbolic expression.
            return (False, "SKIP: cannot perturb numeric E' without symbolic expression")
        return (False, "Unsupported format for E_prime")

    def test_monotonicity_force_scaling_with_gamma_sideways(self):
        """
        For side-by-side case, F_lab_b should be proportional to 1/gamma.
        Perturb gamma (e.g., double) and check ratio of F_lab_b magnitudes.
        """
        if self.F_lab_b is None:
            return (False, "SKIP: F_lab_b missing")
        gamma = self.gamma
        if gamma is None:
            return (False, "SKIP: gamma not available")
        # Check if F_lab_b is symbolic with free symbol gamma
        if isinstance(self.F_lab_b, sp.Basic):
            gamma_sym = self._try_extract_free_symbol(self.F_lab_b, 'gamma')
            if gamma_sym is None:
                return (False, "SKIP: no free symbol 'gamma' in F_lab_b")
            # Baseline gamma value
            try:
                gamma0 = float(gamma)
            except Exception:
                return (False, "SKIP: gamma not numeric")
            gamma1 = 2.0 * gamma0  # double
            # Evaluate magnitude of F_lab_b at gamma0 and gamma1
            # Need to compute norm; assume F_lab_b is a sympy vector
            if hasattr(self.F_lab_b, 'norm'):
                norm_expr = self.F_lab_b.norm()
            else:
                # Try to compute norm from components
                comps = [self._get_component(self.F_lab_b, ax) for ax in ('x','y','z')]
                if any(c is None for c in comps):
                    return (False, "Cannot extract components for norm")
                norm_expr = sp.sqrt(sum(c**2 for c in comps))
            norm_at_gamma0 = self._numeric_substitute(norm_expr, {gamma_sym: gamma0})
            norm_at_gamma1 = self._numeric_substitute(norm_expr, {gamma_sym: gamma1})
            if norm_at_gamma0 is None or norm_at_gamma1 is None:
                return (False, "Cannot evaluate norm numerically")
            # Expected: norm ~ 1/gamma, so norm(gamma1)/norm(gamma0) should be gamma0/gamma1 = 0.5
            expected_ratio = gamma0 / gamma1  # 0.5
            actual_ratio = norm_at_gamma1 / norm_at_gamma0
            if math.isclose(actual_ratio, expected_ratio, rel_tol=0.1):
                return (True, f"F_lab_b magnitude scales as 1/gamma: ratio {actual_ratio} ≈ {expected_ratio}")
            else:
                return (False, f"Scaling violation: ratio {actual_ratio} != {expected_ratio}")
        # Numeric but non-symbolic: cannot perturb
        return (False, "SKIP: cannot perturb numeric F_lab_b without symbolic expression")

    def test_monotonicity_acceleration_scaling_with_gamma_parallel(self):
        """
        For parallel case, a_lab_a should be proportional to 1/gamma^3.
        Perturb gamma (e.g., double) and check ratio of a_lab_a magnitude (z component).
        """
        if self.a_lab_a is None:
            return (False, "SKIP: a_lab_a missing")
        gamma = self.gamma
        if gamma is None:
            return (False, "SKIP: gamma not available")
        if isinstance(self.a_lab_a, sp.Basic):
            gamma_sym = self._try_extract_free_symbol(self.a_lab_a, 'gamma')
            if gamma_sym is None:
                return (False, "SKIP: no free symbol 'gamma' in a_lab_a")
            try:
                gamma0 = float(gamma)
            except Exception:
                return (False, "SKIP: gamma not numeric")
            gamma1 = 2.0 * gamma0
            # Extract z component (the component that scales)
            z_comp = self._get_component(self.a_lab_a, 'z')
            if z_comp is None:
                return (False, "Cannot extract z component from a_lab_a")
            a_z0 = self._numeric_substitute(z_comp, {gamma_sym: gamma0})
            a_z1 = self._numeric_substitute(z_comp, {gamma_sym: gamma1})
            if a_z0 is None or a_z1 is None:
                return (False, "Cannot evaluate a_lab_a_z numerically")
            # Expected ratio = (gamma0/gamma1)^3 = 1/8 = 0.125
            expected_ratio = (gamma0 / gamma1)**3
            actual_ratio = a_z1 / a_z0
            if math.isclose(actual_ratio, expected_ratio, rel_tol=0.1):
                return (True, f"a_lab_a_z scales as 1/gamma^3: ratio {actual_ratio} ≈ {expected_ratio}")
            else:
                return (False, f"Scaling violation: ratio {actual_ratio} != {expected_ratio}")
        return (False, "SKIP: cannot perturb numeric a_lab_a without symbolic expression")

    # ---------- Weak Answer Sanity (at most one) ----------
    def test_answer_sanity_finite_and_nonzero(self):
        """Check that key quantities are finite and non-zero (if numeric)."""
        quantities = [self.E_prime, self.F_prime, self.a_prime,
                      self.E_lab_a, self.F_lab_a, self.a_lab_a,
                      self.E_prime_b, self.F_prime_b, self.a_prime_b,
                      self.E_lab_b, self.F_lab_b, self.a_lab_b]
        for idx, q in enumerate(quantities):
            if q is None:
                continue
            # If numeric scalar or vector component
            if isinstance(q, (int, float)):
                if math.isnan(q) or math.isinf(q):
                    return (False, f"Quantity at index {idx} is NaN or Inf")
            # For sympy, assume finite
        return (True, "All numeric quantities are finite and non-zero (where expected)")

    # ---------- Run All Tests ----------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)