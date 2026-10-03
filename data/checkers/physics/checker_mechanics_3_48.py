import sympy as sp
import math
import random
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class PhysicsChecker:
    """
    PhysicsChecker for relativistic Hamiltonian derivation problem.
    Expected solution_instance.results['H'] is a string or sympy expression:
    H = sqrt((p - q*A/c)**2 * c**2 + m0**2 * c**4) + q*Phi
    """

    def __init__(self, solution_instance):
        # Safely extract results dict
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.H = None
        else:
            self.H = self.results.get('H', None)

        # Define symbols for later use
        self.p, self.A, self.q, self.c, self.m0, self.Phi = sp.symbols('p A q c m0 Phi', real=True, positive=True)
        # Correct reference expression (used for symbolic comparison)
        self.H_correct = sp.sqrt((self.p - self.q * self.A / self.c)**2 * self.c**2 + self.m0**2 * self.c**4) + self.q * self.Phi

        # Parse H if it's a string
        self.H_expr = None
        if self.H is not None:
            if isinstance(self.H, str):
                try:
                    # Parse string expression with implicit multiplication handling
                    transformations = (standard_transformations + (implicit_multiplication_application,))
                    self.H_expr = parse_expr(self.H, local_dict={'p': self.p, 'A': self.A, 'q': self.q, 'c': self.c, 'm0': self.m0, 'Phi': self.Phi}, transformations=transformations)
                except Exception:
                    self.H_expr = None  # Parsing failed, will be caught by interface tests
            elif isinstance(self.H, sp.Expr):
                self.H_expr = self.H
            else:
                self.H_expr = None

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

    # ─── Interface Tests ─────────────────────────────────────────────────
    def test_interface_H_present(self):
        if self.results is None:
            return (False, "Results dict is missing")
        if self.H is None:
            return (False, "H key not found in results")
        return (True, "H key is present")

    def test_interface_H_is_expression(self):
        if self.H_expr is None:
            return (False, "H could not be parsed as a sympy expression (string or sympy.Expr expected)")
        return (True, "H is a valid sympy expression")

    # ─── Invariant Tests (theorem-based, using math.isclose) ────────────
    def test_invariant_structure_sqrt_plus_qPhi(self):
        if self.H_expr is None:
            return (False, "H expression not available")
        # Check overall structure: H = sqrt(...) + q*Phi
        if not isinstance(self.H_expr, sp.Add):
            return (False, "H is not of the form sqrt(...) + q*Phi (not an Add)")
        terms = self.H_expr.args
        sqrt_term = None
        phi_term = None
        for term in terms:
            if term.has(sp.sqrt):
                sqrt_term = term
            elif term.has(self.Phi):
                phi_term = term
        if sqrt_term is None or phi_term is None:
            return (False, "H must contain a sqrt term and a Phi term")
        # Check that phi_term equals q*Phi (possibly after simplification)
        if sp.simplify(phi_term - self.q * self.Phi) != 0:
            return (False, "The Phi term is not exactly q*Phi")
        return (True, "H structure is sqrt(...) + q*Phi")

    def test_invariant_inside_sqrt(self):
        if self.H_expr is None:
            return (False, "H expression not available")
        # Extract the sqrt term and check inside
        terms = self.H_expr.args if isinstance(self.H_expr, sp.Add) else [self.H_expr]
        sqrt_term = None
        for term in terms:
            if term.has(sp.sqrt):
                sqrt_term = term
        if sqrt_term is None:
            return (False, "No sqrt term found")
        # Inside should be (p - q*A/c)**2 * c**2 + m0**2 * c**4
        inside = sqrt_term.args[0]
        expected_inside = (self.p - self.q * self.A / self.c)**2 * self.c**2 + self.m0**2 * self.c**4
        diff = sp.simplify(inside - expected_inside)
        if diff != 0:
            return (False, f"Inside sqrt does not match expected form. Difference: {diff}")
        return (True, "Inside sqrt has correct symbolic form")

    # ─── Modeling / Trace Tests (quality checks derived from explanation steps) ─
    def test_modeling_substitution_consistency(self):
        # Re-derive H from the canonical momentum and energy-momentum relation
        # Check that the provided H is equivalent to the correct symbolic form.
        if self.H_expr is None:
            return (False, "H expression not available")
        diff = sp.simplify(self.H_expr - self.H_correct)
        if diff != 0:
            return (False, f"Provided H is not equivalent to the correct Hamiltonian. Difference: {diff}")
        return (True, "Provided H matches the correct symbolic expression")

    def test_modeling_no_vector_potential_limit(self):
        # In the limit A -> 0, the Hamiltonian should reduce to free particle:
        # H = sqrt(p^2 c^2 + m0^2 c^4) + q*Phi
        if self.H_expr is None:
            return (False, "H expression not available")
        # Substitute A = 0
        subs_A0 = {self.A: 0}
        try:
            H_A0 = self.H_expr.subs(subs_A0)
        except Exception as e:
            return (False, f"Substitution failed: {e}")
        expected_A0 = sp.sqrt(self.p**2 * self.c**2 + self.m0**2 * self.c**4) + self.q * self.Phi
        if sp.simplify(H_A0 - expected_A0) != 0:
            return (False, "When A=0, H does not reduce to the free particle Hamiltonian")
        return (True, "In limit A->0, H correctly reduces to free particle expression")

    # ─── Monotonicity Tests (explicit perturbation of one input) ────────
    def test_monotonicity_H_increases_with_A(self):
        # For fixed p, q, c, m0, Phi, increasing A should increase H (because |p - qA/c| decreases if p>0? Actually careful: 
        # The term inside sqrt is c^2 * (p - qA/c)^2 = (p*c - qA)^2. The function f(A) = (p*c - qA)^2 is convex and has minimum at A = p*c/q.
        # For A smaller than that, increasing A reduces the square; but since we consider a perturbation from a baseline, monotonicity is not trivial.
        # Instead, we choose a baseline where p >> (qA/c) so that (p - qA/c) > 0, then increasing A reduces the term and H decreases.
        # Let's test a specific baseline: p=5, A=1, q=1, c=3 -> p - qA/c = 5-0.333=4.667 > 0. Increase A to 2 -> term = 5-0.667=4.333 < 4.667, so sqrt decreases. So H should decrease.
        if self.H_expr is None:
            return (False, "H expression not available")
        # Baseline parameters
        p_baseline = 5.0
        A_baseline = 1.0
        q_val = 1.0
        c_val = 3.0
        m0_val = 1.0
        Phi_val = 0.5
        subs_base = {self.p: p_baseline, self.A: A_baseline, self.q: q_val, self.c: c_val, self.m0: m0_val, self.Phi: Phi_val}
        try:
            H_base = self.H_expr.subs(subs_base).evalf()
        except Exception as e:
            return (False, f"Baseline evaluation failed: {e}")
        # Perturb A: increase by 50% (so that (p - qA/c) stays positive)
        A_perturbed = 1.5  # from 1 to 1.5
        subs_pert = {self.p: p_baseline, self.A: A_perturbed, self.q: q_val, self.c: c_val, self.m0: m0_val, self.Phi: Phi_val}
        try:
            H_pert = self.H_expr.subs(subs_pert).evalf()
        except Exception as e:
            return (False, f"Perturbed evaluation failed: {e}")
        # Expect H to decrease because (p - qA/c) decreases -> sqrt decreases
        if H_pert >= H_base:
            # (since with these numbers, the decrease is about 0.02)
            return (False, f"H did not decrease when A increased: base={H_base}, perturbed={H_pert}")
        return (True, "H decreases monotonically with A for chosen baseline")

    def test_monotonicity_H_increases_with_Phi(self):
        # H = sqrt(...) + q*Phi; increasing Phi should increase H linearly.
        if self.H_expr is None:
            return (False, "H expression not available")
        # Baseline
        subs_base = {self.p: 3.0, self.A: 1.0, self.q: 1.0, self.c: 3.0, self.m0: 2.0, self.Phi: 0.5}
        try:
            H_base = self.H_expr.subs(subs_base).evalf()
        except Exception as e:
            return (False, f"Baseline evaluation failed: {e}")
        # Increase Phi to 1.5
        subs_pert = subs_base.copy()
        subs_pert[self.Phi] = 1.5
        try:
            H_pert = self.H_expr.subs(subs_pert).evalf()
        except Exception as e:
            return (False, f"Perturbed evaluation failed: {e}")
        if H_pert <= H_base:
            return (False, f"H did not increase when Phi increased: base={H_base}, perturbed={H_pert}")
        return (True, "H increases with Phi")

    # ─── Weak Answer Sanity (only one, no re-derivation) ────────────────
    def test_answer_sanity_weak(self):
        # Weak sanity: H must be real and positive for typical positive parameters
        if self.H_expr is None:
            return (False, "H expression not available")
        try:
            # Use a typical set of positive numbers
            subs_test = {self.p: 2.0, self.A: 1.0, self.q: 1.0, self.c: 3.0, self.m0: 1.0, self.Phi: 0.5}
            val = self.H_expr.subs(subs_test).evalf()
            if val.is_real() is False:
                return (False, f"H evaluated to a complex number: {val}")
            if val < 0:
                return (False, f"H is negative: {val}")
        except Exception as e:
            return (False, f"Evaluation failed: {e}")
        return (True, "Weak sanity: H is real and positive")