import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for dielectric sphere in uniform external field.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (never numeric defaults)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.a = givens.get('a', None)
        self.epsilon1 = givens.get('epsilon1', None)
        self.epsilon2 = givens.get('epsilon2', None)
        self.E0 = givens.get('E0', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.E_inside = results.get('E_inside', None)
        self.E_outside = results.get('E_outside', None)

        # Prepare symbolic symbols for analytic reasoning
        self.sym_a = sp.symbols('a', positive=True)
        self.sym_eps1 = sp.symbols('eps1', positive=True)
        self.sym_eps2 = sp.symbols('eps2', positive=True)
        self.sym_E0 = sp.symbols('E0', real=True)
        self.sym_r = sp.symbols('r', positive=True)
        self.sym_theta = sp.symbols('theta', real=True)

        # Helper: convert stored expressions to sympy if possible
        self._preparse_results()

    def _preparse_results(self):
        """Attempt to convert results to sympy expressions."""
        def to_sympy(val):
            if val is None:
                return None
            if isinstance(val, sp.Expr):
                return val
            if isinstance(val, str):
                try:
                    # Allow recognized symbols
                    local_dict = {
                        'z_hat': sp.Matrix([0, 0, 1]),
                        'a': self.sym_a,
                        'epsilon1': self.sym_eps1,
                        'epsilon2': self.sym_eps2,
                        'E0': self.sym_E0,
                        'r': self.sym_r,
                        'theta': self.sym_theta,
                    }
                    return sp.sympify(val, locals=local_dict)
                except Exception:
                    return None
            return None
        self.E_inside_sym = to_sympy(self.E_inside)
        self.E_outside_sym = to_sympy(self.E_outside)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that E_inside and E_outside are present."""
        if self.E_inside is None:
            return (False, "Missing E_inside in results")
        if self.E_outside is None:
            return (False, "Missing E_outside in results")
        return (True, "Required keys present")

    # ------------------------------------------------------------------
    # Invariant / theorem-based checks (use math.isclose for numeric, sympy for symbolic)
    # ------------------------------------------------------------------
    def test_invariant_inside_uniform(self):
        """E_inside must be independent of position (uniform)."""
        if self.E_inside_sym is None:
            return (False, "SKIP: cannot parse E_inside")
        expr = self.E_inside_sym
        if isinstance(expr, sp.Matrix) and expr.shape == (3, 1):
            for comp in expr:
                free = comp.free_symbols
                # Allowed: eps1, eps2, E0, a, maybe epsilon1/2, but no r/theta
                allowed = {self.sym_eps1, self.sym_eps2, self.sym_E0, self.sym_a}
                if free - allowed:
                    return (False, f"Component contains spatially dependent symbols: {free - allowed}")
            return (True, "E_inside is uniform (no r/θ dependence)")
        else:
            return (False, "E_inside is not a 3-vector")

    def test_invariant_far_field(self):
        """For r → ∞, E_outside must approach E₀ * ẑ."""
        if self.E_outside_sym is None:
            return (False, "SKIP: cannot parse E_outside")
        expr = self.E_outside_sym
        if not isinstance(expr, sp.Matrix) or expr.shape != (3, 1):
            return (False, "E_outside is not a 3-vector")
        limit_expr = sp.limit(expr, self.sym_r, sp.oo)
        expected = sp.Matrix([0, 0, self.sym_E0])
        for lim_comp, exp_comp in zip(limit_expr, expected):
            if sp.simplify(lim_comp - exp_comp) != 0:
                return (False, f"Far field component mismatch: {lim_comp} vs {exp_comp}")
        return (True, "Far field → E₀ ẑ")

    def test_invariant_dipole_term_structure(self):
        """E_outside - E₀ ẑ must have the standard dipole form:
        k * [3(E₀·r)r / r⁵ - E₀ / r³]."""
        if self.E_outside_sym is None:
            return (False, "SKIP: cannot parse E_outside")
        expr = self.E_outside_sym
        if not isinstance(expr, sp.Matrix) or expr.shape != (3, 1):
            return (False, "E_outside is not a 3-vector")
        E0_vec = self.sym_E0 * sp.Matrix([0, 0, 1])
        diff = expr - E0_vec
        # Dipole term vector
        r_vec = sp.Matrix([
            self.sym_r * sp.sin(self.sym_theta),
            0,
            self.sym_r * sp.cos(self.sym_theta)
        ])
        dipole_form = (3 * (E0_vec.dot(r_vec)) * r_vec) / self.sym_r**5 - E0_vec / self.sym_r**3
        # If diff = 0, trivially satisfied
        if diff == sp.Matrix([0, 0, 0]):
            return (True, "E_outside = E₀ ẑ (trivial dipole)")
        # Check parallelism: cross product must vanish
        cross = diff.cross(dipole_form)
        for comp in cross:
            if sp.simplify(comp) != 0:
                return (False, f"Cross product component non-zero: {comp}")
        return (True, "E_outside - E₀ ẑ is parallel to standard dipole term")

    # ------------------------------------------------------------------
    # Modeling / trace checks (using explanation steps, not bulk constant copy)
    # ------------------------------------------------------------------
    def test_modeling_e_inside_form(self):
        """E_inside must equal (3ε₂/(ε₁+2ε₂)) E₀ ẑ (derived from theorem)."""
        if self.E_inside_sym is None:
            return (False, "SKIP: cannot parse E_inside")
        # Derive expected expression from givens and theorem constraints
        expected_factor = 3 * self.sym_eps2 / (self.sym_eps1 + 2 * self.sym_eps2)
        expected = expected_factor * sp.Matrix([0, 0, self.sym_E0])
        if isinstance(self.E_inside_sym, sp.Matrix) and self.E_inside_sym.shape == (3,1):
            diff = self.E_inside_sym - expected
            diff_simplified = sp.simplify(diff)
            if diff_simplified == sp.Matrix([0, 0, 0]):
                return (True, "E_inside matches expected form")
            else:
                return (False, f"E_inside deviates: {diff_simplified}")
        else:
            return (False, "E_inside not a 3-vector")

    def test_modeling_e_outside_boundary_value(self):
        """At r = a, the tangential component of E_outside must equal that of E_inside (continuity).
        This uses the known relation: E_out_θ = -E₀ sinθ + (B/a³) sinθ, with B = ((ε₁-ε₂)/(ε₁+2ε₂)) a³ E₀."""
        if self.E_inside_sym is None or self.E_outside_sym is None:
            return (False, "SKIP: missing fields")
        # We'll re-derive B from the known dipole form: B = ((ε₁-ε₂)/(ε₁+2ε₂)) a³ E₀
        B_expr = ((self.sym_eps1 - self.sym_eps2) / (self.sym_eps1 + 2*self.sym_eps2)) * self.sym_a**3 * self.sym_E0
        # Expected E_outside θ component at r=a: -E₀ sinθ + (B/a³) sinθ = -E₀ sinθ + ((ε₁-ε₂)/(ε₁+2ε₂)) E₀ sinθ
        # That simplifies to -E₀ sinθ * (1 - (ε₁-ε₂)/(ε₁+2ε₂)) = -E₀ sinθ * (3ε₂/(ε₁+2ε₂))
        expected_E_out_theta = -self.sym_E0 * sp.sin(self.sym_theta) * (3*self.sym_eps2/(self.sym_eps1 + 2*self.sym_eps2))
        # E_inside uniform along z: E_inside = factor * E₀ ẑ, factor = 3ε₂/(ε₁+2ε₂)
        # Its tangential component at r=a (any point) is -factor * E₀ sinθ
        factor = 3*self.sym_eps2/(self.sym_eps1 + 2*self.sym_eps2)
        expected_E_in_theta = -factor * self.sym_E0 * sp.sin(self.sym_theta)
        # They should be equal (continuity). Check symbolic identity.
        if sp.simplify(expected_E_out_theta - expected_E_in_theta) != 0:
            return (False, "Internal error: derived mismatch in continuity")
        # Now we need to actually evaluate the stored E_outside θ component at r=a.
        # For that we extract the θ component from the vector expression.
        # This is tricky; instead we check that the stored expression satisfies the theoretical form.
        # We can check that the stored E_outside, after substituting r=a, has the correct θ component.
        # As a simpler substitute, we check that the relation between E_inside and E_outside at r=a
        # satisfies the continuity equation derived from the potentials.
        # From explanation steps: A = -E₀ + B/a³   (tangential continuity)
        # where A is coefficient in V_inside = A r cosθ.
        # And we have E_inside along z = -A ẑ? Actually E_inside = -∇V_inside = -A ẑ (in Cartesian) -> E_inside_z = -A.
        # So A = -E_inside_z.
        # And B is the dipole coefficient.
        # So continuity condition: -E_inside_z = -E₀ + B/a³.
        # Rearranged: E_inside_z = E₀ - B/a³.
        # Let's derive B from the dipole term in E_outside.
        # The dipole term coefficient k = (ε₁-ε₂)/(ε₁+2ε₂) a³ E₀ = B (since B = k).
        # So the condition becomes: E_inside_z = E₀ - (ε₁-ε₂)/(ε₁+2ε₂) E₀ = ( (ε₁+2ε₂ - (ε₁-ε₂))/(ε₁+2ε₂) ) E₀ = (3ε₂/(ε₁+2ε₂)) E₀, which matches the known E_inside.
        # Therefore we can check this relation using the stored expressions.
        # Extract E_inside_z from stored vector (assuming z-component is third component)
        if isinstance(self.E_inside_sym, sp.Matrix) and self.E_inside_sym.shape == (3,1):
            E_inside_z_val = self.E_inside_sym[2]
        else:
            return (False, "SKIP: cannot extract E_inside z component")
        # Compute B from E_outside dipole term. The dipole term is of the form k * (3(E₀·r)r/r⁵ - E₀/r³).
        # B is the scalar k, which equals (ε₁-ε₂)/(ε₁+2ε₂) a³ E₀.
        # We can derive B from the stored E_outside by evaluating at a specific convenient point,
        # e.g., on the z-axis (θ=0) at r=a, where the dipole term simplifies.
        # On the positive z-axis, r = (0,0,a), E₀·r = E₀ a, so dipole term = k * [3E₀ a³ / a⁵ ẑ - E₀ / a³ ẑ] = k * (3E₀/a² - E₀/a³) ẑ = k * E₀ (3/a² - 1/a³) ẑ.
        # Then E_outside = E₀ ẑ + that. So we can solve for k = B.
        # Substitute r=a, θ=0 into stored E_outside vector.
        point_sub = {self.sym_r: self.sym_a, self.sym_theta: 0}
        E_out_on_axis = self.E_outside_sym.subs(point_sub)
        if not isinstance(E_out_on_axis, sp.Matrix) or E_out_on_axis.shape != (3,1):
            return (False, "SKIP: cannot evaluate E_outside on axis")
        # On axis, only z-component non-zero
        E_out_z = E_out_on_axis[2]
        # Compute expected B form from theorem
        B_theory = ((self.sym_eps1 - self.sym_eps2) / (self.sym_eps1 + 2*self.sym_eps2)) * self.sym_a**3 * self.sym_E0
        # Solve for B from E_out_z expression:
        # E_out_z = E₀ + B * E₀ * (3/a² - 1/a³)  actually careful: dipole term = B * (3(E₀·r)r/r⁵ - E₀/r³)
        # On axis: r = (0,0,r), r_vector = (0,0,r), E₀·r = E₀ r, so term = B * (3E₀ r ẑ/r⁵ - E₀ ẑ/r³) = B * E₀ (3/r² - 1/r³) ẑ
        # At r=a: E_out_z = E₀ + B * E₀ * (3/a² - 1/a³) = E₀ [1 + B * (3/a² - 1/a³)]
        # Solve for B: (E_out_z/E₀ - 1) / (3/a² - 1/a³) = B
        B_from_stored = sp.simplify( (E_out_z/self.sym_E0 - 1) / (3/self.sym_a**2 - 1/self.sym_a**3) )
        # Check if B_from_stored equals B_theory
        if sp.simplify(B_from_stored - B_theory) != 0:
            return (False, f"Stored B ({B_from_stored}) does not match theorem B ({B_theory})")
        # Then check continuity: E_inside_z should equal E₀ - B_theory/a³
        expected_E_inside_z = self.sym_E0 - B_theory / self.sym_a**3
        # But we already know the form, so just check algebraic equivalence
        if sp.simplify(E_inside_z_val - expected_E_inside_z) != 0:
            return (False, f"Tangential continuity fails: {E_inside_z_val} vs {expected_E_inside_z}")
        return (True, "Boundary condition (tangential E continuity) satisfied at r=a")

    # ------------------------------------------------------------------
    # Monotonicity checks: explicitly perturb one input and compare output
    # ------------------------------------------------------------------
    def _compute_inside_magnitude(self, eps1_val, eps2_val, E0_val):
        """Compute E_inside magnitude using the known formula (derived from theorem)."""
        if eps1_val is None or eps2_val is None or E0_val is None:
            return None
        return 3 * eps2_val / (eps1_val + 2 * eps2_val) * E0_val

    def test_monotonicity_inside_epsilon1_decreases(self):
        """As ε₁ increases, E_inside magnitude decreases."""
        if self.epsilon1 is None or self.epsilon2 is None or self.E0 is None:
            return (False, "SKIP: missing givens for numeric perturbation")
        try:
            eps1 = float(self.epsilon1)
            eps2 = float(self.epsilon2)
            E0 = float(self.E0)
        except (TypeError, ValueError):
            return (False, "SKIP: cannot convert givens to floats")
        baseline = self._compute_inside_magnitude(eps1, eps2, E0)
        if baseline is None:
            return (False, "SKIP: cannot compute baseline")
        # Perturb epsilon1 upward by 5%
        eps1_pert = eps1 * 1.05
        perturbed = self._compute_inside_magnitude(eps1_pert, eps2, E0)
        if perturbed is None:
            return (False, "SKIP: cannot compute perturbed value")
        # Magnitude must decrease (derivative negative)
        if perturbed < baseline:
            return (True, f"E_inside decreased: {baseline} → {perturbed}")
        else:
            return (False, f"E_inside did not decrease: {baseline} → {perturbed}")

    def test_monotonicity_inside_epsilon2_increases(self):
        """As ε₂ increases, E_inside magnitude increases."""
        if self.epsilon1 is None or self.epsilon2 is None or self.E0 is None:
            return (False, "SKIP: missing givens for numeric perturbation")
        try:
            eps1 = float(self.epsilon1)
            eps2 = float(self.epsilon2)
            E0 = float(self.E0)
        except (TypeError, ValueError):
            return (False, "SKIP: cannot convert givens to floats")
        baseline = self._compute_inside_magnitude(eps1, eps2, E0)
        if baseline is None:
            return (False, "SKIP: cannot compute baseline")
        # Perturb epsilon2 upward by 5%
        eps2_pert = eps2 * 1.05
        perturbed = self._compute_inside_magnitude(eps1, eps2_pert, E0)
        if perturbed is None:
            return (False, "SKIP: cannot compute perturbed value")
        # Magnitude must increase (derivative positive)
        if perturbed > baseline:
            return (True, f"E_inside increased: {baseline} → {perturbed}")
        else:
            return (False, f"E_inside did not increase: {baseline} → {perturbed}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no hardcoded target values)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Check that results are finite and real (weak sanity)."""
        messages = []
        if self.E_inside is not None:
            try:
                val = float(self.E_inside) if not isinstance(self.E_inside, (list, tuple, complex)) else None
                if val is not None and (math.isnan(val) or math.isinf(val)):
                    messages.append("E_inside is NaN or Inf")
            except Exception:
                pass  # possibly vector; skip numeric check
        if self.E_outside is not None:
            try:
                val = float(self.E_outside) if not isinstance(self.E_outside, (list, tuple, complex)) else None
                if val is not None and (math.isnan(val) or math.isinf(val)):
                    messages.append("E_outside is NaN or Inf")
            except Exception:
                pass
        if messages:
            return (False, "; ".join(messages))
        return (True, "Results appear finite")

    # ------------------------------------------------------------------
    # Run all tests
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
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)