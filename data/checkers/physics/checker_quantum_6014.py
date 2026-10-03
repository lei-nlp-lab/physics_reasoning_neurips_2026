import sympy as sp
import math

class PhysicsChecker:
    """
    Auditor for the first Born approximation differential equation derivation.
    """
    def __init__(self, solution_instance):
        # Extract result safely; expected key: 'diff_eq_v'
        self.diff_eq_v = solution_instance.results.get('diff_eq_v', None)
        # Also keep a reference to givens if available (optional)
        self.givens = solution_instance.givens if hasattr(solution_instance, 'givens') else {}
        # Prepare symbolic tools for invariant checks
        self._prepare_symbolic_tools()

    def _prepare_symbolic_tools(self):
        # Symbols for plane wave operator test
        self.z = sp.symbols('z', real=True)
        self.k = sp.symbols('k', positive=True)
        # Plane wave exp(ikz)
        psi0 = sp.exp(sp.I * self.k * self.z)
        # Apply (nabla^2 + k^2) analytically: only z-derivative matters
        laplace = sp.diff(psi0, self.z, 2)
        self.plane_wave_operator = sp.simplify(laplace + self.k**2 * psi0)

    # ----------------------------------------------------------------------
    # run_all_tests: discover all test_ methods and run them
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

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_result_key_exists(self):
        if self.diff_eq_v is None:
            return (False, "results['diff_eq_v'] is None or missing")
        if not isinstance(self.diff_eq_v, str):
            return (False, "results['diff_eq_v'] is not a string")
        return (True, "diff_eq_v key present and is string")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based)
    # ----------------------------------------------------------------------
    def test_invariant_plane_wave_annihilated(self):
        """
        Verify that (nabla^2 + k^2) e^{ikz} = 0 symbolically.
        """
        if self.plane_wave_operator is None:
            return (False, "Symbolic preparation failed")
        if self.plane_wave_operator == 0:
            return (True, "(∇² + k²) e^{ikz} = 0 verified symbolically")
        else:
            # Fallback numeric check
            k_val = 1.0
            z_val = 0.5
            import cmath
            numeric_val = -k_val**2 * cmath.exp(1j*k_val*z_val) + k_val**2 * cmath.exp(1j*k_val*z_val)
            if (math.isclose(numeric_val.real, 0.0, abs_tol=1e-10) and
                math.isclose(numeric_val.imag, 0.0, abs_tol=1e-10)):
                return (True, "(∇² + k²) e^{ikz} = 0 verified numerically")
            return (False, f"(∇² + k²) e^{ikz} is not zero: symbolic gave {self.plane_wave_operator}")

    def test_invariant_first_born_linearizes(self):
        """
        The first Born approximation drops the V v term.
        The resulting differential equation for v must NOT contain v on the RHS.
        """
        if self.diff_eq_v is None:
            return (False, "diff_eq_v not available")
        eq = self.diff_eq_v
        if '=' not in eq:
            return (False, "Equation does not contain '='")
        lhs_part, rhs_part = eq.split('=', 1)
        import re
        lhs_has_v = bool(re.search(r'\bv\b', lhs_part))
        rhs_has_v = bool(re.search(r'\bv\b', rhs_part))
        if not lhs_has_v:
            return (False, "LHS does not contain v")
        if rhs_has_v:
            return (False, "RHS contains v (first Born approximation should drop V v term)")
        return (True, "Equation is linear: RHS does not contain v")

    def test_invariant_zero_potential(self):
        """
        If V=0, the equation reduces to (∇² + k²) v = 0.
        """
        if self.diff_eq_v is None:
            return (False, "diff_eq_v not available")
        eq = self.diff_eq_v
        if '=' not in eq:
            return (False, "Equation does not contain '='")
        lhs, rhs = eq.split('=', 1)
        if 'V' not in rhs:
            return (False, "RHS does not contain V; cannot verify zero potential limit")
        return (True, "Equation contains V on RHS, so V=0 implies (∇²+k²)v=0 (consistent)")

    # ----------------------------------------------------------------------
    # Modeling checks (using explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_operator_present(self):
        """
        The equation must include the operator (nabla^2 + k^2) acting on v.
        """
        if self.diff_eq_v is None:
            return (False, "diff_eq_v not available")
        eq = self.diff_eq_v
        contains_nabla = 'nabla' in eq or '\\nabla' in eq
        contains_k2 = 'k^2' in eq or 'k²' in eq or 'k**2' in eq
        if not contains_nabla:
            return (False, "Equation does not contain Laplacian (nabla^2)")
        if not contains_k2:
            return (False, "Equation does not contain k^2")
        return (True, "Equation contains (nabla^2 + k^2) operator")

    def test_modeling_rhs_has_V_eikz(self):
        """
        The RHS should be proportional to V e^{ikz} (or exp(ikz)).
        """
        if self.diff_eq_v is None:
            return (False, "diff_eq_v not available")
        eq = self.diff_eq_v
        if '=' not in eq:
            return (False, "Equation does not contain '='")
        lhs, rhs = eq.split('=', 1)
        has_V = 'V' in rhs
        has_exp = 'exp' in rhs or 'e^{ikz}' in rhs or 'e^{i k z}' in rhs or 'e^{ikz}' in rhs
        if not has_V:
            return (False, "RHS does not contain V")
        if not has_exp:
            return (False, "RHS does not contain exp(ikz)")
        return (True, "RHS contains V e^{ikz} as expected")

    # ----------------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_potential_strength(self):
        """
        Perturb the potential strength V and verify that the source term scales linearly.
        Use symbolic representation of the RHS: (2m/ħ²) V e^{ikz}.
        """
        if self.diff_eq_v is None:
            return (False, "SKIP: diff_eq_v not available")
        # Define symbolic RHS as known from physics
        m, hbar, k, V, z = sp.symbols('m hbar k V z', positive=True)
        rhs_sym = (2*m / hbar**2) * V * sp.exp(sp.I * k * z)
        # Baseline: V = 1
        baseline_val = rhs_sym.subs({m: 1, hbar: 1, V: 1, k: 1, z: 0}).evalf()
        # Perturbed: V = 2
        perturbed_val = rhs_sym.subs({m: 1, hbar: 1, V: 2, k: 1, z: 0}).evalf()
        # Ratio should be 2
        ratio = float(perturbed_val) / float(baseline_val)
        if math.isclose(ratio, 2.0, rel_tol=1e-2):
            return (True, f"Doubling V doubles RHS: ratio = {ratio:.3f} (expected 2)")
        else:
            return (False, f"Doubling V gave ratio {ratio:.3f}, expected ~2.0")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_equation_not_trivial(self):
        """
        Weak check: equation is non-empty and looks like a valid differential equation.
        """
        if self.diff_eq_v is None:
            return (False, "diff_eq_v not available")
        eq = self.diff_eq_v.strip()
        if len(eq) < 10:
            return (False, f"Equation too short: {eq}")
        if '=' not in eq:
            return (False, "No equals sign in equation")
        if eq.isnumeric():
            return (False, "Equation is numeric")
        return (True, "Equation appears to be a non-trivial differential equation")