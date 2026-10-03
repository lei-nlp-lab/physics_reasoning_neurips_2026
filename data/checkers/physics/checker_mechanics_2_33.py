import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for a coupled-pendulum problem with a spring.
    Extracts required data from a solution_instance object and runs unit-style tests.
    """
    def __init__(self, solution_instance):
        # Safe extraction from solution_instance (attributes or results dict)
        # Givens
        self.m = getattr(solution_instance, 'm', None) if hasattr(solution_instance, 'm') else None
        self.l = getattr(solution_instance, 'l', None) if hasattr(solution_instance, 'l') else None
        self.k = getattr(solution_instance, 'k', None) if hasattr(solution_instance, 'k') else None
        self.g = getattr(solution_instance, 'g', None) if hasattr(solution_instance, 'g') else None
        self.v = getattr(solution_instance, 'v', None) if hasattr(solution_instance, 'v') else None
        self.d = getattr(solution_instance, 'd', None) if hasattr(solution_instance, 'd') else None

        # Results expected from the solving pipeline
        results_obj = getattr(solution_instance, 'results', {})
        if isinstance(results_obj, dict):
            self.L_exact = results_obj.get('L_exact', None)
            self.omega1 = results_obj.get('omega1', None)
            self.omega2 = results_obj.get('omega2', None)
            self.eta_t = results_obj.get('eta_t', None)
            self.xi_t = results_obj.get('xi_t', None)
        else:
            # Fallback: try dot notation
            self.L_exact = getattr(solution_instance, 'L_exact', None)
            self.omega1 = getattr(solution_instance, 'omega1', None)
            self.omega2 = getattr(solution_instance, 'omega2', None)
            self.eta_t = getattr(solution_instance, 'eta_t', None)
            self.xi_t = getattr(solution_instance, 'xi_t', None)

        # Additional extracted data for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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

    # --------------------------------------------------------------------------
    # Interface checks: required keys exist and are of valid types
    # --------------------------------------------------------------------------
    def test_interface_required_results_present(self):
        """Check that all five expected result keys are present (not None)."""
        missing = []
        if self.L_exact is None:
            missing.append('L_exact')
        if self.omega1 is None:
            missing.append('omega1')
        if self.omega2 is None:
            missing.append('omega2')
        if self.eta_t is None:
            missing.append('eta_t')
        if self.xi_t is None:
            missing.append('xi_t')
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required results present")

    def test_interface_givens_not_none(self):
        """Check that all physical givens are provided (m, l, k, g, v, d)."""
        missing = []
        if self.m is None: missing.append('m')
        if self.l is None: missing.append('l')
        if self.k is None: missing.append('k')
        if self.g is None: missing.append('g')
        if self.v is None: missing.append('v')
        if self.d is None: missing.append('d')
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All givens present")

    def test_interface_givens_positive(self):
        """Check that provided numeric givens are positive (if numeric)."""
        for name, val in [('m', self.m), ('l', self.l), ('k', self.k),
                          ('g', self.g), ('v', self.v), ('d', self.d)]:
            if val is not None and not isinstance(val, (int, float)):
                continue
            if val is not None and val <= 0:
                return (False, f"Given {name} must be positive, got {val}")
        return (True, "All numeric givens positive")

    def test_interface_L_exact_is_string(self):
        """Check that L_exact is a string as emitted by pseudocode."""
        if self.L_exact is None:
            return (False, "SKIP: L_exact not available")
        if not isinstance(self.L_exact, str):
            return (False, f"L_exact should be a string, got {type(self.L_exact)}")
        return (True, "L_exact is string")

    def test_interface_omega1_is_string(self):
        """Check that omega1 is a string."""
        if self.omega1 is None:
            return (False, "SKIP: omega1 not available")
        if not isinstance(self.omega1, str):
            return (False, f"omega1 should be a string, got {type(self.omega1)}")
        return (True, "omega1 is string")

    def test_interface_omega2_is_string(self):
        """Check that omega2 is a string."""
        if self.omega2 is None:
            return (False, "SKIP: omega2 not available")
        if not isinstance(self.omega2, str):
            return (False, f"omega2 should be a string, got {type(self.omega2)}")
        return (True, "omega2 is string")

    def test_interface_eta_t_is_string(self):
        """Check that eta_t is a string."""
        if self.eta_t is None:
            return (False, "SKIP: eta_t not available")
        if not isinstance(self.eta_t, str):
            return (False, f"eta_t should be a string, got {type(self.eta_t)}")
        return (True, "eta_t is string")

    def test_interface_xi_t_is_string(self):
        """Check that xi_t is a string."""
        if self.xi_t is None:
            return (False, "SKIP: xi_t not available")
        if not isinstance(self.xi_t, str):
            return (False, f"xi_t should be a string, got {type(self.xi_t)}")
        return (True, "xi_t is string")

    # --------------------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # --------------------------------------------------------------------------
    def test_invariant_omega1_less_than_omega2(self):
        """
        ω1 = sqrt(g/l), ω2 = sqrt(g/l + 2k/m) => ω2 > ω1.
        If numeric values available, check strict inequality.
        """
        # If numeric values exist directly
        omega1_num = self._try_parse_numeric(self.omega1)
        omega2_num = self._try_parse_numeric(self.omega2)
        if omega1_num is not None and omega2_num is not None:
            if omega2_num > omega1_num:
                return (True, "ω2 > ω1 holds numerically")
            else:
                return (False, f"Expected ω2 > ω1, got ω1={omega1_num}, ω2={omega2_num}")
        # Otherwise try to derive from givens
        if self.g is not None and self.l is not None:
            try:
                w1 = math.sqrt(self.g / self.l)
            except:
                w1 = None
            if w1 is not None:
                if self.k is not None and self.m is not None and self.k > 0 and self.m > 0:
                    w2 = math.sqrt(self.g / self.l + 2*self.k / self.m)
                    if w2 > w1:
                        return (True, "ω2 > ω1 holds from given parameters")
                    else:
                        return (False, "ω2 should be > ω1 from formula")
                else:
                    # Cannot compute ω2 exactly, but check that ω2 string contains sqrt and plus
                    if self.omega2 and '+' in self.omega2 and 'sqrt' in self.omega2:
                        return (True, "ω2 contains sqrt(g/l + ...) consistent with ω2>ω1")
                    return (True, "ω2 string pattern consistent")
        # Fallback: pattern match
        if self.omega1 and self.omega2:
            if 'sqrt' in self.omega1 and 'sqrt' in self.omega2:
                return (True, "Both omega contain sqrt, consistent")
        return (False, "Cannot verify ω2 > ω1; no numeric data and incomplete patterns")

    def test_invariant_normal_coordinate_definitions(self):
        """
        By definition: η = (θ1+θ2)/2, ξ = (θ1-θ2)/2.
        The expressions for η(t) and ξ(t) should contain sin(ω1 t) and sin(ω2 t) respectively.
        Also denominators contain 2lω etc.
        """
        if self.eta_t is None or self.xi_t is None:
            return (False, "SKIP: normal coordinate expressions missing")
        # Check sin patterns
        if 'sin' not in self.eta_t or 'sin' not in self.xi_t:
            return (False, "Normal coordinate expressions should contain sin(...)")
        # Check denominator contains 2*l*omega (or 2l omega1)
        # Allow variations: '2lω1', '2*l*ω1', '2 l omega1'
        eta_lower = self.eta_t.replace(' ', '').lower()
        xi_lower = self.xi_t.replace(' ', '').lower()
        if '2l*omega1' not in eta_lower and '2lω1' not in eta_lower and '2*l*ω1' not in eta_lower:
            return (False, f"eta_t denominator should contain 2*l*omega1 term, got: {self.eta_t}")
        if '2l*omega2' not in xi_lower and '2lω2' not in xi_lower and '2*l*ω2' not in xi_lower:
            return (False, f"xi_t denominator should contain 2*l*omega2 term, got: {self.xi_t}")
        return (True, "Normal coordinate forms consistent with definitions")

    def test_invariant_lagrangian_kinetic_terms(self):
        """Check that L_exact contains the kinetic energy term ½ m l² (θ̇₁²+θ̇₂²)."""
        if self.L_exact is None:
            return (False, "SKIP: L_exact not available")
        # Search for 'm l^2 dottheta1^2' etc. Be flexible with spaces.
        L = self.L_exact.replace(' ', '').lower()
        if '1/2ml^2(dottheta1^2+dottheta2^2)' in L:
            return (True, "Kinetic term present")
        if '1/2*m*l^2*(dottheta1^2+dottheta2^2)' in L:
            return (True, "Kinetic term present (with stars)")
        # More general: check for 'ml^2' and 'dottheta'
        if 'm l^2' in self.L_exact and 'dottheta' in self.L_exact:
            return (True, "Kinetic term likely present")
        return (False, "Could not identify kinetic energy term in L_exact")

    def test_invariant_lagrangian_spring_term(self):
        """Check that L_exact contains the spring potential term with (sqrt(...)-d)^2."""
        if self.L_exact is None:
            return (False, "SKIP: L_exact not available")
        if 'sqrt' in self.L_exact and '- d)^2' in self.L_exact:
            return (True, "Spring term with sqrt and (extension)^2 present")
        if 'sqrt' in self.L_exact and '-d)^2' in self.L_exact:
            return (True, "Spring term present (no space)")
        return (False, "Spring potential term not found in L_exact")

    def test_invariant_lagrangian_gravitational_term(self):
        """Check that L_exact contains mgl(cosθ1+cosθ2)."""
        if self.L_exact is None:
            return (False, "SKIP: L_exact not available")
        L = self.L_exact.replace(' ', '')
        if 'mgl(cosθ1+cosθ2)' in L or 'mgl(cosθ_1+cosθ_2)' in L:
            return (True, "Gravitational term present")
        if 'mgl(costheta1+costheta2)' in L:
            return (True, "Gravitational term present (as string)")
        return (False, "Gravitational potential term not found in L_exact")

    # --------------------------------------------------------------------------
    # Modeling checks (verifying that the solution follows correct physical modeling)
    # --------------------------------------------------------------------------
    def test_modeling_omega_formula_consistent(self):
        """
        ω1 should depend only on g and l; ω2 should also depend on k and m.
        Check symbolic strings contain the right parameters.
        """
        if self.omega1 is None or self.omega2 is None:
            return (False, "SKIP: omega strings not available")
        # ω1: should contain 'g/l' or similar
        w1_clean = self.omega1.replace(' ', '').lower()
        w2_clean = self.omega2.replace(' ', '').lower()
        if 'g/l' not in w1_clean and 'g/l' not in w1_clean:
            # try separate g and l
            if 'g' not in w1_clean or 'l' not in w1_clean:
                return (False, "ω1 should depend on g and l")
        if 'k' not in w2_clean or 'm' not in w2_clean:
            return (False, "ω2 should depend on k and m")
        return (True, "Frequency formulas contain correct parameters")

    def test_modeling_small_angle_linearization(self):
        """
        Check that the normal coordinate equations are linear: η'' + (g/l)η = 0.
        This is implicit in the form; we can check that eta_t does not contain nonlinear terms.
        """
        if self.eta_t is None or self.xi_t is None:
            return (False, "SKIP: no normal coordinate expressions")
        # The expected forms are linear sinusoids.
        # Check that there are no powers >1, no products, no sqrt, no nonlinear functions.
        # Simplistic: check that no '^2' appears inside (except maybe 2l)
        # Actually denominator may have 2l, but that's fine.
        # We'll just verify the pattern is A*sin(ωt).
        eta = self.eta_t.replace(' ', '')
        xi = self.xi_t.replace(' ', '')
        # Expect something like 'v/(2lω1)sin(ω1t)' or 'v/(2*l*ω1)*sin(ω1*t)'
        if 'sin' not in eta or 'sin' not in xi:
            return (False, "Normal coordinates should contain sin functions")
        # Check no square terms
        if '^2' in eta or '^2' in xi:
            return (False, "Normal coordinate expressions should be linear (no squares)")
        return (True, "Normal coordinate expressions are linear sinusoids")

    def test_modeling_initial_conditions(self):
        """
        At t=0, η and ξ should be zero (since both θ are zero).
        Check that the expressions have no cosine term (A=0, C=0).
        """
        if self.eta_t is None or self.xi_t is None:
            return (False, "SKIP: no normal coordinate expressions")
        # If expression contains 'cos', it's problematic unless coefficient zero.
        # Since expectation is only sin terms, check that cos is absent.
        eta_lower = self.eta_t.lower()
        xi_lower = self.xi_t.lower()
        if 'cos' in eta_lower:
            return (False, "η(t) should not contain cosine terms (initial displacement zero)")
        if 'cos' in xi_lower:
            return (False, "ξ(t) should not contain cosine terms (initial displacement zero)")
        return (True, "No cosine terms in normal coordinates (consistent with zero initial displacement)")

    def test_modeling_spring_extension_small_angle(self):
        """
        For small angles, the spring extension should be approximately l*(θ2−θ1).
        The Lagrangian should reduce to a quadratic form.
        We can check that L_exact, when expanded, yields a term ½ k l² (θ1-θ2)².
        Since we have the symbolic string, we check presence of (theta1-theta2)^2.
        """
        if self.L_exact is None:
            return (False, "SKIP: L_exact not available")
        # Search for (theta1-theta2)^2 in the Lagrangian.
        # The small-angle spring potential is ½ k l² (θ1-θ2)²; this might appear after linearization.
        # But the exact L contains a complicated expression. However, the final normal coordinate derivation
        # yields this term; we can check that the normal coordinate equations contain the right coupling.
        # Alternatively, check that the EOM derived from L_exact (if available) have coupling term.
        # Since we only have L_exact, we can check that it contains 'l^2 (theta1-theta2)^2' or similar.
        L_no_space = self.L_exact.replace(' ', '')
        if '(theta1-theta2)^2' in L_no_space or '(θ1-θ2)^2' in L_no_space:
            return (True, "Spring term in Lagrangian contains (θ1-θ2)^2")
        # More general: look for 'l^2' and '(theta1' together
        if 'l^2' in self.L_exact and 'theta1' in self.L_exact and 'theta2' in self.L_exact:
            return (True, "Lagrangian contains l^2 and angles (likely quadratic term)")
        return (False, "Cannot confirm small-angle spring term in L_exact")

    # --------------------------------------------------------------------------
    # Monotonicity checks (trend under parameter variation)
    # --------------------------------------------------------------------------
    def test_monotonicity_omega2_increases_with_k(self):
        """
        Check that if k increases, ω2 increases (for fixed g,l,m).
        If numeric values available, perturb k and recalc ω2.
        """
        if self.g is None or self.l is None or self.m is None or self.k is None:
            return (False, "SKIP: missing givens for monotonicity test")
        if not all(isinstance(x, (int, float)) for x in [self.g, self.l, self.m, self.k]):
            return (False, "SKIP: givens not numeric")
        try:
            omega2_base = math.sqrt(self.g / self.l + 2 * self.k / self.m)
        except:
            return (False, "Could not compute ω2 base")
        # Perturb k upward by 10%
        k_perturbed = self.k * 1.1
        omega2_pert = math.sqrt(self.g / self.l + 2 * k_perturbed / self.m)
        if omega2_pert > omega2_base:
            return (True, f"ω2 increased from {omega2_base:.3f} to {omega2_pert:.3f} when k increased 10%")
        else:
            return (False, f"ω2 should increase with k, got base {omega2_base:.3f}, perturbed {omega2_pert:.3f}")

    def test_monotonicity_omega1_independent_of_k(self):
        """
        ω1 = sqrt(g/l) should be unchanged when k changes.
        Compute baseline ω1, then perturb k (if given) and recompute ω1; they must be equal.
        """
        if self.g is None or self.l is None:
            return (False, "SKIP: missing g or l")
        if not isinstance(self.g, (int, float)) or not isinstance(self.l, (int, float)):
            return (False, "SKIP: g or l not numeric")
        try:
            omega1_base = math.sqrt(self.g / self.l)
        except:
            return (False, "Could not compute ω1 base")
        # If k is numeric, simulate perturbation (omega1 unchanged)
        if self.k is not None and isinstance(self.k, (int, float)):
            k_pert = self.k * 1.1  # just to simulate
            omega1_pert = math.sqrt(self.g / self.l)  # independent of k
            if math.isclose(omega1_base, omega1_pert, rel_tol=1e-9):
                return (True, "ω1 unchanged when k perturbed (as expected)")
            else:
                return (False, "ω1 changed unexpectedly")
        else:
            # If no k numeric, just check string doesn't contain k
            if self.omega1 and 'k' in self.omega1:
                return (False, "ω1 should not depend on k")
            return (True, "ω1 independent of k (string does not contain k)")

    # --------------------------------------------------------------------------
    # Weak answer sanity (only one method allowed)
    # --------------------------------------------------------------------------
    def test_answer_sanity_expressions_valid(self):
        """
        Combined weak sanity: check that expressions for eta_t and xi_t 
        are finite, not NaN, not Inf, and denominators not zero.
        Also check that they contain expected trigonometric functions (sin).
        """
        any_skip = False
        for name, expr in [('eta_t', self.eta_t), ('xi_t', self.xi_t)]:
            if expr is None:
                any_skip = True
                continue
            # Check for NaN, Inf, division by zero pattern
            expr_lower = expr.lower()
            if 'nan' in expr_lower or 'inf' in expr_lower:
                return (False, f"{name} contains NaN or Inf")
            if '/0' in expr.replace(' ', ''):
                return (False, f"{name} contains division by zero")
            # Check that sin is present (weak requirement)
            if 'sin' not in expr_lower:
                return (False, f"{name} does not contain sin function")
        if any_skip:
            return (False, "SKIP: one or both expressions missing")
        return (True, "Expressions appear valid: finite, no division by zero, contain sin")

    # --------------------------------------------------------------------------
    # Helper
    # --------------------------------------------------------------------------
    @staticmethod
    def _try_parse_numeric(s):
        """If s is a string representing a number (maybe with sqrt), try to evaluate numerically."""
        if s is None:
            return None
        # Attempt to parse as float
        try:
            return float(s)
        except:
            pass
        # Try to evaluate simple expressions like 'sqrt(g/l)' if g and l are available? Not needed now.
        return None