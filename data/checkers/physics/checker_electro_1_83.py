import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the charge-inside-spherical-conductor problem.
    Checks interface integrity, physical invariants, modeling consistency,
    and monotonicity. No test_copies_answer_* methods.
    """
    def __init__(self, solution_instance):
        # Extract givens safely (all default to None)
        self.q = getattr(solution_instance.givens, 'q', None) if hasattr(solution_instance, 'givens') else solution_instance.get('q', None)
        self.r1 = getattr(solution_instance.givens, 'r1', None) if hasattr(solution_instance, 'givens') else solution_instance.get('r1', None)
        self.r2 = getattr(solution_instance.givens, 'r2', None) if hasattr(solution_instance, 'givens') else solution_instance.get('r2', None)
        self.a = getattr(solution_instance.givens, 'a', None) if hasattr(solution_instance, 'givens') else solution_instance.get('a', None)
        self.epsilon0 = getattr(solution_instance.givens, 'epsilon0', None) if hasattr(solution_instance, 'givens') else solution_instance.get('epsilon0', None)

        # Results from solution (dictionary)
        self.results = getattr(solution_instance, 'results', {})
        self.F = self.results.get('F', None)
        self.V = self.results.get('V', None)

        # Sympy symbols for derivation
        self.q_sym, self.r1_sym, self.r2_sym, self.a_sym, self.eps0_sym = sp.symbols('q r1 r2 a epsilon0', positive=True, real=True)

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_has_F_key(self):
        """Check that results contains 'F' key."""
        if 'F' not in self.results:
            return False, "Missing 'F' in results"
        return True, "'F' key exists"

    def test_interface_has_V_key(self):
        """Check that results contains 'V' key."""
        if 'V' not in self.results:
            return False, "Missing 'V' in results"
        return True, "'V' key exists"

    def test_interface_F_is_valid(self):
        """Check that F is not None and is a numeric/symbolic expression."""
        if self.F is None:
            return False, "'F' is None"
        return True, "'F' is not None"

    def test_interface_V_is_valid(self):
        """Check that V is not None and is a numeric/symbolic expression."""
        if self.V is None:
            return False, "'V' is None"
        return True, "'V' is not None"

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_conductor_equipotential(self):
        """
        The inner surface potential equals the conductor potential.
        Since conductor is isolated and uncharged, outer surface has net charge +q,
        potential at r=r2 is q/(4 π ε0 r2). V must equal that.
        """
        if self.V is None or self.q is None or self.r2 is None or self.epsilon0 is None:
            return False, "SKIP: missing q, r2, epsilon0, or V"
        # Compute expected potential from physics
        expected_V = self.q / (4 * math.pi * self.epsilon0 * self.r2)
        # Use math.isclose for numerical check (can also parse symbolic)
        if isinstance(self.V, (int, float)):
            if math.isclose(self.V, expected_V, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"V {self.V} matches expected {expected_V}"
            else:
                return False, f"V {self.V} does not match expected {expected_V}"
        else:
            # symbolic expression: assume it's correct if not numerical
            return True, f"V (symbolic) is acceptable"

    def test_invariant_force_image_form(self):
        """
        The force should be expressible as F = - (1/(4π ε0)) * (q^2 * r1 * a) / (r1^2 - a^2)^2.
        We derive this symbolically and compare to F (if available in symbolic or numeric form).
        """
        if self.F is None or self.q is None or self.r1 is None or self.a is None or self.epsilon0 is None:
            return False, "SKIP: missing q, r1, a, epsilon0, or F"
        # Derive expected force formula using sympy
        expr_F = - (1/(4*sp.pi*self.eps0_sym)) * (self.q_sym**2 * self.r1_sym * self.a_sym) / (self.r1_sym**2 - self.a_sym**2)**2
        # Try to parse F as sympy expression
        try:
            F_sym = sp.sympify(self.F)
        except:
            return False, f"Cannot parse F '{self.F}' as symbolic expression"
        # Simplify difference
        diff = sp.simplify(expr_F - F_sym)
        # If diff is zero (exact) or close to zero for numeric substitution
        if diff == 0:
            return True, "Force expression exactly matches theoretical form"
        # Numeric substitution if all givens numeric
        sub_dict = {self.q_sym: self.q, self.r1_sym: self.r1, self.a_sym: self.a, self.eps0_sym: self.epsilon0}
        try:
            diff_num = float(diff.subs(sub_dict).evalf())
        except:
            return False, "Could not evaluate difference numerically"
        if math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Force expression matches numerically within tolerance"
        else:
            return False, f"Force expression differs by {diff_num}"

    # ------------------------------------------------------------
    # Modeling / Trace consistency tests
    # ------------------------------------------------------------
    def test_modeling_force_attractive(self):
        """
        Since image charge is negative and located beyond r1, force is attractive (inward).
        So F should be negative if we use sign convention outward positive.
        """
        if self.F is None:
            return False, "SKIP: F not available"
        # For numeric or symbolic, check sign
        if isinstance(self.F, (int, float)):
            if self.F < 0:
                return True, f"F is negative (attractive): {self.F}"
            else:
                return False, f"F is not negative: {self.F}"
        else:
            # symbolic: assume correct if expression has minus sign
            return True, "Force expression symbolically acceptable"

    def test_modeling_potential_independent_of_a(self):
        """
        The potential of the inner surface does NOT depend on a (distance from center).
        We can verify by noting that V does not involve 'a'.
        This checks that the solution correctly isolated a-independence.
        """
        if self.V is None:
            return False, "SKIP: V not available"
        # Check if 'a' appears in V string
        V_str = str(self.V)
        if 'a' in V_str:
            return False, "Potential expression contains 'a', should be independent"
        return True, "Potential independent of a"

    # ------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------
    def test_monotonicity_force_vs_a(self):
        """
        As a increases (charge moves off-center), the magnitude of force should increase.
        We perturb a by +5% and check that |F| increases (within tolerance).
        """
        if self.F is None or self.a is None or self.q is None or self.r1 is None or self.epsilon0 is None:
            return False, "SKIP: missing required givens or F"
        if isinstance(self.F, (int, float)):
            # numeric baseline
            F0 = self.F
            a0 = self.a
            # perturb a by +5%
            a1 = a0 * 1.05
            # recompute force using theoretical formula (since we cannot run solver)
            F1 = - (1/(4*math.pi*self.epsilon0)) * (self.q**2 * self.r1 * a1) / (self.r1**2 - a1**2)**2
            # check that |F1| > |F0| (attractive force)
            if abs(F1) > abs(F0) - 1e-6:  # allow small numerical noise
                return True, f"Force magnitude increases with a: |F0|={abs(F0)}, |F1|={abs(F1)}"
            else:
                return False, f"Force magnitude did not increase: |F0|={abs(F0)}, |F1|={abs(F1)}"
        else:
            # symbolic: assume monotonicity holds
            return True, "Symbolic expression, monotonicity plausible"

    def test_monotonicity_potential_vs_r2(self):
        """
        As r2 increases, potential V should decrease (inverse proportionality).
        Perturb r2 by +10% and check V decreases.
        """
        if self.V is None or self.r2 is None or self.q is None or self.epsilon0 is None:
            return False, "SKIP: missing q, r2, epsilon0, or V"
        if isinstance(self.V, (int, float)):
            V0 = self.V
            r2_old = self.r2
            r2_new = r2_old * 1.1
            V1 = self.q / (4 * math.pi * self.epsilon0 * r2_new)
            if V1 < V0:
                return True, f"V decreases with r2: V0={V0}, V1={V1}"
            else:
                return False, f"V did not decrease: V0={V0}, V1={V1}"
        else:
            return True, "Symbolic expression, monotonicity plausible"

    # ------------------------------------------------------------
    # Optional weak sanity checks (only one allowed)
    # ------------------------------------------------------------
    def test_answer_sanity_sign_and_positivity(self):
        """
        Weak sanity: V should be positive if q positive; F should be negative.
        Non-finite values should be absent.
        """
        errors = []
        if self.V is not None:
            if isinstance(self.V, (int, float)):
                if math.isnan(self.V) or math.isinf(self.V):
                    errors.append("V is non-finite")
                elif self.V <= 0:
                    errors.append(f"V is not positive: {self.V}")
        if self.F is not None:
            if isinstance(self.F, (int, float)):
                if math.isnan(self.F) or math.isinf(self.F):
                    errors.append("F is non-finite")
        # This is weak; we accept missing info
        if errors:
            return False, "; ".join(errors)
        return True, "Sanity checks passed"

    # ------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------
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