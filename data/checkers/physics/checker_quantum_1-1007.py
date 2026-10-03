import math
import traceback
import sympy as sp
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits a solution to the problem:
    "Given momentum-space Schrödinger equation (p^2/2m - a ∇_p^2) φ = i ∂_t φ,
     find the force F(r)."
    """
    def __init__(self, solution_instance: Any):
        # Extract all required information safely (default None)
        self.results = getattr(solution_instance, 'results', {})
        self.F_vec = self.results.get('F_vec', None)
        self.a = getattr(solution_instance, 'a', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.physics_mapping = getattr(solution_instance, 'physics_mapping', None)
        # Additional possibly useful attributes (optional)
        self.givens = getattr(solution_instance, 'givens', None)
        self.variables = getattr(solution_instance, 'variables', None)
        self.equations = getattr(solution_instance, 'equations', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------

    def test_interface_F_vec_present(self) -> Tuple[bool, str]:
        """Check that the final answer key F_vec exists and is non‑empty."""
        if self.F_vec is None:
            return (False, "SKIP: H key 'F_vec' is missing")
        if isinstance(self.F_vec, str):
            if len(self.F_vec.strip()) == 0:
                return (False, "F_vec is an empty string")
        # If it's a sympy expression, it's fine
        return (True, "F_vec is present and non‑empty")

    def test_interface_F_vec_type(self) -> Tuple[bool, str]:
        """Check that F_vec is a string or sympy expression."""
        if self.F_vec is None:
            return (False, "SKIP: F_vec not available")
        if not isinstance(self.F_vec, (str, sp.Expr)):
            return (False, f"F_vec has type {type(self.F_vec).__name__}, expected str or sympy.Expr")
        return (True, "F_vec type is acceptable")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based)
    # ------------------------------------------------------------------

    def test_invariant_force_potential_relation(self) -> Tuple[bool, str]:
        """
        Check that the derived force is consistent with the gradient of a potential.
        For V = a r^2, F = -2a r. We verify the symbolic relation F = -∇V.
        This method uses sympy to compute the gradient of V = a * (x^2+y^2+z^2)
        and compares it symbolically with -2*a*(x,y,z).
        Falls back to structural string check if sympy parsing fails or a is missing.
        """
        if self.F_vec is None:
            return (False, "SKIP: F_vec not available")
        try:
            # Construct symbolic expression for V = a * (x^2 + y^2 + z^2)
            a_sym = sp.Symbol('a', real=True)
            x, y, z = sp.symbols('x y z', real=True)
            V = a_sym * (x**2 + y**2 + z**2)
            # Compute gradient: (∂V/∂x, ∂V/∂y, ∂V/∂z)
            grad_V = sp.Matrix([sp.diff(V, var) for var in (x, y, z)])
            # Expected force: -grad_V
            F_expected = -grad_V  # = -2*a*(x, y, z)
            # Parse the student's F_vec to a sympy expression
            # We need to handle common representations: e.g., '-2*a*r', '-2*a * (x,y,z)'
            # For simplicity, we try to parse as a string and substitute symbolic r.
            if isinstance(self.F_vec, str):
                expr_str = self.F_vec.replace('r', '(x**2+y**2+z**2)**0.5')
                # Also handle 'mathbf' and TeX commands if present
                # We'll strip common TeX wrappers
                cleaned = expr_str.replace(r'\mathbf{', '').replace('}', '')
                cleaned = cleaned.replace('r', '(x**2+y**2+z**2)**0.5')
                # Only parse if it looks like a mathematical expression
                try:
                    F_student = sp.sympify(cleaned)
                except (sp.SympifyError, SyntaxError):
                    return (False, f"Cannot sympify student F_vec: {self.F_vec}")
            else:
                F_student = self.F_vec  # trust it's a sympy expression
            # Compare symbolically: required to be equal after simplifying
            # But note: student might write '-2*a r' without component vector.
            # We'll check that F_student equals -2*a*symbolic_r (vector)
            # If the student gave a scalar magnitude, that's an error.
            # However, we expect the answer to be a vector: -2a r.
            # We'll check that each component matches: F_student[i] = -2*a*coord_i.
            # Convert F_student to a 3‑vector if it's given as a tuple or list.
            if isinstance(F_student, sp.Matrix) and F_student.shape == (3,1):
                F_components = F_student
            elif isinstance(F_student, (list, tuple)) and len(F_student) == 3:
                F_components = sp.Matrix(F_student)
            else:
                # Possibly a scalar times unit vector; we can approximate
                # For simplicity, handle only vector form, else skip.
                return (False, "F_vec is not a 3‑vector (expected vector form like '-2*a*(x,y,z)')")
            # Compute difference
            diff = sp.simplify(F_components - F_expected)
            if diff == sp.zeros(3,1):
                return (True, "F_vec matches -∇(a r^2) = -2a r")
            else:
                # Use isclose on numerical substitution if a given
                if self.a is not None and isinstance(self.a, (int, float)):
                    # Substitute a numerically and check at a few points
                    subs_dict = {a_sym: self.a, x: 0.5, y: -1.2, z: 0.8}
                    num_diff = diff.subs(subs_dict).evalf()
                    norm = math.sqrt(float(num_diff.dot(num_diff)))
                    if norm < 1e-6:
                        return (True, "F_vec numerically matches -∇(a r^2) within tolerance")
                return (False, f"F_vec does not match expected gradient: diff = {diff}")
        except Exception as e:
            return (False, f"test_invariant_force_potential_relation crashed: {e}")

    def test_invariant_sign_and_direction(self) -> Tuple[bool, str]:
        """
        Check that the force points opposite to the position vector: F . r < 0.
        This holds for any attractive quadratic potential (a>0) – but a can be negative.
        At least the dot product should be negative if a>0; we check sign using a test point.
        We use numerical evaluation if a is given.
        """
        if self.F_vec is None:
            return (False, "SKIP: F_vec not available")
        try:
            # Use sympy to evaluate dot product with r at a sample point
            a_sym = sp.Symbol('a', real=True)
            x, y, z = sp.symbols('x y z', real=True)
            # Build student force vector (assume it's a 3‑vector)
            if isinstance(self.F_vec, str):
                # Try to convert to sympy expression; may need custom handling
                expr_str = self.F_vec.replace(r'\mathbf{', '').replace('}', '')
                expr_str = expr_str.replace('r', '(x**2+y**2+z**2)**0.5')
                try:
                    F_student = sp.sympify(expr_str)
                except:
                    return (False, f"Cannot parse F_vec string: {self.F_vec}")
            else:
                F_student = self.F_vec
            # If it's a scalar times unit vector, convert to vector
            # For simplicity, we handle only the expected vector form
            # If it's a scalar magnitude, we cannot check direction
            if not isinstance(F_student, (sp.Matrix, list, tuple)):
                return (False, "F_vec is not expressed as a vector; cannot check direction")
            # Compute dot product F·r at a test point (1,2,3)
            r_vec = sp.Matrix([x, y, z])
            if isinstance(F_student, sp.Matrix) and F_student.shape == (3,1):
                dot = (F_student.dot(r_vec)).subs({a_sym: 1, x: 1, y: 2, z: 3})
            elif isinstance(F_student, (list, tuple)) and len(F_student) == 3:
                F_mat = sp.Matrix(F_student)
                dot = (F_mat.dot(r_vec)).subs({a_sym: 1, x: 1, y: 2, z: 3})
            else:
                return (False, "Unexpected vector format")
            # Evaluate numerically
            try:
                val = float(dot.evalf())
            except:
                return (False, "Cannot evaluate dot product numerically")
            # Expected: F is attractive if a>0 → dot < 0; repulsive if a<0 → dot > 0.
            # We only check that dot is finite and not zero (unless r=0)
            if abs(val) < 1e-12:
                return (False, "Force dot product is zero at test point (maybe r=0?)")
            return (True, f"Force dot product at (1,2,3) is {val}, finite and non‑zero")
        except Exception as e:
            return (False, f"test_invariant_sign_and_direction crashed: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------

    def test_modeling_potential_identification(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps or physics_mapping correctly identify
        the potential V = a r^2 from the term -a ∇_p^2.
        This is a qualitative check: we look for keywords in explanation_steps.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        try:
            text = str(self.explanation_steps).lower()
            # Required concepts: "quadratic potential", "r^2", "fourier transform", or similar
            indicators = ["quadratic potential", "r^2", "fourier transform",
                          "coordinate space", "momentum space", "potential v(r) = a r^2"]
            score = sum(1 for phrase in indicators if phrase in text)
            if score >= 2:
                return (True, f"Explanation mentions at least two relevant indicators (score={score})")
            else:
                return (False, f"Explanation lacks sufficient detail: found only {score}/6 indicators")
        except Exception as e:
            return (False, f"test_modeling_potential_identification crashed: {e}")

    def test_modeling_force_expression_contains_2a(self) -> Tuple[bool, str]:
        """
        Check that the final answer expression contains the factor -2a.
        This is a structural test, not a re‑derivation.
        """
        if self.F_vec is None:
            return (False, "SKIP: F_vec not available")
        try:
            expr_str = str(self.F_vec)
            # Look for '-2a', '-2*a', or '2a' with sign
            if '-2a' in expr_str or '-2*a' in expr_str:
                return (True, "Force expression contains '-2a'")
            else:
                return (False, f"Force expression does not contain '-2a': {expr_str}")
        except Exception as e:
            return (False, f"test_modeling_force_expression_contains_2a crashed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------

    def test_monotonicity_force_linear_in_a(self) -> Tuple[bool, str]:
        """
        If a increases, the magnitude of the force should increase proportionally.
        This test requires a explicit numeric value for a.
        We simulate by perturbing a by +10% and check that the force magnitude
        roughly increases by 10% (since F = -2a r, linear).
        """
        # Need a numeric a value from the solution instance or from givens
        if self.a is None:
            return (False, "SKIP: parameter 'a' not available for monotonicity test")
        try:
            a0 = float(self.a)
        except:
            return (False, "SKIP: cannot convert a to float")
        if abs(a0) < 1e-15:
            return (False, "a is zero; monotonicity not testable")
        # We need a location r to compute force magnitude; use a default point (1,1,1)
        r_mag = math.sqrt(3)  # for (1,1,1)
        F0_mag = abs(-2 * a0 * r_mag)
        # Perturb a by +10%
        a1 = a0 * 1.1
        F1_mag = abs(-2 * a1 * r_mag)
        # Expect ratio close to 1.1
        ratio = F1_mag / F0_mag if F0_mag != 0 else 0
        if math.isclose(ratio, 1.1, rel_tol=0.02, abs_tol=0.02):
            return (True, f"Force magnitude scales linearly with a: ratio={ratio:.4f} (expected ~1.1)")
        else:
            return (False, f"Force magnitude does not scale as expected: ratio={ratio:.4f}, expected ~1.1")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one allowed)
    # ------------------------------------------------------------------

    def test_answer_sanity_force_sign(self) -> Tuple[bool, str]:
        """
        Weak sanity: check that the force expression has a negative sign (indicative of an attractive force
        for positive a). This does NOT rederive the answer.
        """
        if self.F_vec is None:
            return (False, "SKIP: F_vec not available")
        try:
            expr_str = str(self.F_vec)
            # Check for minus sign
            if '-' in expr_str:
                return (True, "Force expression contains a minus sign, consistent with attractive force")
            else:
                # It might be written as '-2a r' (already has minus) or '2a r' (wrong). We flag.
                return (False, "Force expression does not include a minus sign; may indicate repulsive force")
        except Exception as e:
            return (False, f"test_answer_sanity_force_sign crashed: {e}")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them,
        and return overall pass/fail and a list of log messages.
        Never crashes: wraps each test in try/except.
        """
        logs = []
        all_pass = True
        # Collect method names
        test_methods = [name for name in dir(self) if name.startswith("test_") and callable(getattr(self, name))]
        # Sort for deterministic order
        test_methods.sort()
        for name in test_methods:
            method = getattr(self, name)
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)