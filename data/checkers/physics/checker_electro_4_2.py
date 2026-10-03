import sympy as sp
import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Auditor for the relation between the speed of light c, vacuum permittivity ε0,
    and vacuum permeability μ0.

    Expected solution key:
        - results['c_relation']: a string representing the correct relation,
          e.g., 'c = sqrt(1/(epsilon0 * mu0))' or 'c = 1/sqrt(epsilon0 * mu0)'.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # c_relation: string from emit contract (default None)
        self.c_relation = results.get('c_relation', None)

        # Optional additional data for richer checks (default None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Pre-compile sympy symbols and expected expression for invariant check
        self.c, self.epsilon0, self.mu0 = sp.symbols('c epsilon0 mu0', real=True, positive=True)
        self.expected_rhs = 1 / sp.sqrt(self.epsilon0 * self.mu0)   # c = 1/sqrt(ε0 μ0)
        # Incorrect options (a) and (b)
        self.wrong_a = sp.sqrt(self.epsilon0 / self.mu0)
        self.wrong_b = sp.sqrt(self.mu0 / self.epsilon0)

    # ------------------------------------------------------------------
    #  Helper: parse the RHS of user's relation as a sympy expression
    # ------------------------------------------------------------------
    def _parse_rhs(self) -> Optional[sp.Expr]:
        """Parse the RHS of the user's c_relation string into a sympy expression."""
        if self.c_relation is None or not isinstance(self.c_relation, str):
            return None
        expr_str = self.c_relation.strip()
        if '=' not in expr_str:
            return None
        rhs_str = expr_str.split('=', 1)[1].strip()
        try:
            # Map common spellings
            subs_dict = {
                'epsilon0': self.epsilon0,
                'eps0': self.epsilon0,
                'mu0': self.mu0,
                'mu_0': self.mu0,
                'epsilon_0': self.epsilon0,
            }
            rhs = sp.sympify(rhs_str, locals=subs_dict)
            return rhs
        except Exception:
            return None

    # ------------------------------------------------------------------
    #  Interface Checks
    # ------------------------------------------------------------------
    def test_interface_c_relation_key_exists(self) -> Tuple[bool, str]:
        """Check that results['c_relation'] exists."""
        if self.c_relation is None:
            return (False, "Missing required key 'c_relation' in results")
        return (True, "Key 'c_relation' is present")

    def test_interface_c_relation_is_string(self) -> Tuple[bool, str]:
        """Check that c_relation is a non‑empty string."""
        if not isinstance(self.c_relation, str) or len(self.c_relation.strip()) == 0:
            return (False, f"c_relation is not a non‑empty string (got type {type(self.c_relation)})")
        return (True, "c_relation is a non‑empty string")

    # ------------------------------------------------------------------
    #  Invariant (theorem‑based) Checks
    # ------------------------------------------------------------------
    def test_invariant_maxwell_speed_relation(self) -> Tuple[bool, str]:
        """
        Verify that the derived expression is mathematically equivalent to
        c = 1/sqrt(ε0 μ0) (the Maxwell wave speed relation).
        """
        rhs = self._parse_rhs()
        if rhs is None:
            return (False, "SKIP: cannot parse RHS of c_relation")

        # Compare with expected form
        try:
            diff = sp.simplify(rhs - self.expected_rhs)
        except Exception as e:
            return (False, f"Symbolic simplification failed: {e}")

        if diff == 0:
            return (True, "The relation is exactly c = 1/sqrt(ε0 μ0)")

        # Also try alternative representation: c = sqrt(1/(ε0 μ0))
        alt_expected = sp.sqrt(1 / (self.epsilon0 * self.mu0))
        diff2 = sp.simplify(rhs - alt_expected)
        if diff2 == 0:
            return (True, "The relation is equivalent to c = sqrt(1/(ε0 μ0))")

        return (False, f"Expression is not equivalent to Maxwell's relation. Simplified difference: {diff}")

    def test_invariant_not_sqrt_epsilon_over_mu(self) -> Tuple[bool, str]:
        """
        Verify that the expression is NOT sqrt(ε0/μ0) (option a).
        This is a theorem-based invariant from Maxwell's equations.
        """
        rhs = self._parse_rhs()
        if rhs is None:
            return (False, "SKIP: cannot parse RHS of c_relation")

        try:
            diff = sp.simplify(rhs - self.wrong_a)
        except Exception as e:
            return (False, f"Symbolic comparison failed: {e}")

        if diff == 0:
            return (False, "Expression matches the incorrect option (a): sqrt(ε0/μ0)")
        return (True, "Expression does not equal sqrt(ε0/μ0)")

    def test_invariant_not_sqrt_mu_over_epsilon(self) -> Tuple[bool, str]:
        """
        Verify that the expression is NOT sqrt(μ0/ε0) (option b).
        This is a theorem-based invariant from Maxwell's equations.
        """
        rhs = self._parse_rhs()
        if rhs is None:
            return (False, "SKIP: cannot parse RHS of c_relation")

        try:
            diff = sp.simplify(rhs - self.wrong_b)
        except Exception as e:
            return (False, f"Symbolic comparison failed: {e}")

        if diff == 0:
            return (False, "Expression matches the incorrect option (b): sqrt(μ0/ε0)")
        return (True, "Expression does not equal sqrt(μ0/ε0)")

    # ------------------------------------------------------------------
    #  Modelling / Trace Quality Checks
    # ------------------------------------------------------------------
    def test_modeling_proper_format(self) -> Tuple[bool, str]:
        """
        Check that the string contains both 'epsilon0' (or 'eps0') and 'mu0'
        in the right context (simple sanity, not substituting the actual physics).
        """
        if self.c_relation is None or not isinstance(self.c_relation, str):
            return (False, "SKIP: c_relation missing or invalid type")
        expr = self.c_relation.lower()
        if 'epsilon0' not in expr and 'eps0' not in expr:
            return (False, "Expression does not contain 'epsilon0' or 'eps0'")
        if 'mu0' not in expr:
            return (False, "Expression does not contain 'mu0'")
        return (True, "Expression contains the necessary physical constants")

    def test_modeling_denominator_contains_product(self) -> Tuple[bool, str]:
        """
        Verify that the expression has epsilon0 and mu0 in the denominator (i.e., the
        product epsilon0*mu0 appears under a reciprocal). This is derived from the
        expected form c = 1/sqrt(epsilon0*mu0) or c = sqrt(1/(epsilon0*mu0)).
        We check that the normalized form contains a fraction with epsilon0*mu0 in denominator.
        """
        rhs = self._parse_rhs()
        if rhs is None:
            return (False, "SKIP: cannot parse RHS of c_relation")

        # Express the RHS as 1 / something
        try:
            # Simplify to a fraction
            frac = sp.together(rhs)
        except Exception:
            return (False, "Could not combine expression into a single fraction")

        # Check that denominator is symbolic and contains both epsilon0 and mu0
        denom = sp.denom(frac)
        if denom == 1:
            # Maybe the user wrote sqrt(1/...)? Let's check the numerator
            # For sqrt(1/(e0 m0)), sp.together gives sqrt(1/(e0*m0)) which cannot be expressed as a simple fraction.
            # Alternative approach: check that the expression contains a sqrt with reciprocal inside.
            # We'll check if the expression's atoms contain a reciprocal or power of -1.
            pass  # Continue with more direct check

        # Check for presence of epsilon0 and mu0 in the denominator
        # Use .has to see if they appear inside a reciprocal or as denominator
        # We can check if the expression, rewritten as 1/x, yields product
        # Simple: check that the free symbols are exactly {epsilon0, mu0}
        free_syms = rhs.free_symbols
        if free_syms != {self.epsilon0, self.mu0}:
            return (False, f"Expression contains unexpected symbols: {free_syms - {self.epsilon0, self.mu0}}")
        # Check that it is not simply epsilon0*mu0 (no reciprocal)
        # Use analytic: the expression must be decreasing in both (already in monotonicity)
        # But we can check that the product free appears in a denominator (under a power -1/2)
        # Simplest: check that the exponent of epsilon0 in the expression (when multiplied by mu0) is negative
        # We'll compute the partial derivative sign (strictly negative) - that's monotonicity.
        # Instead, check the form: after clearing radicals, the RHS should be 1/sqrt(product).
        # We can test that rhs * sqrt(epsilon0 * mu0) == 1.
        try:
            product = sp.sqrt(self.epsilon0 * self.mu0)
            simplified_product = sp.simplify(rhs * product)
        except Exception:
            return (False, "Could not compute product with sqrt")

        if simplified_product == 1:
            return (True, "Expression is of the form 1/sqrt(ε0 μ0)")
        else:
            return (False, f"Expression does not satisfy c * sqrt(ε0 μ0) = 1 (got {simplified_product})")

    def test_modeling_no_extraneous_operations(self) -> Tuple[bool, str]:
        """
        Check that the expression uses only multiplication, division, and square root.
        No addition, subtraction, or powers other than 1/2.
        """
        rhs = self._parse_rhs()
        if rhs is None:
            return (False, "SKIP: cannot parse RHS of c_relation")

        # Walk the expression tree to detect disallowed operations
        disallowed_types = (sp.Add, sp.Pow)  # Pow is allowed only for sqrt (exp=1/2)
        allowed = True
        reason = ""
        try:
            for sub in sp.preorder_traversal(rhs):
                if isinstance(sub, sp.Add):
                    allowed = False
                    reason = "Expression contains addition"
                    break
                if isinstance(sub, sp.Pow):
                    # Allow only if exponent is Rational(1,2) or -Rational(1,2)
                    if sub.args[1] not in (sp.Rational(1,2), sp.Rational(-1,2)):
                        allowed = False
                        reason = f"Expression contains power with exponent {sub.args[1]} (only sqrt allowed)"
                        break
        except Exception as e:
            return (False, f"Error traversing expression: {e}")

        if allowed:
            return (True, "Expression uses only multiplication, division, and square root")
        else:
            return (False, reason)

    # ------------------------------------------------------------------
    #  Monotonicity Checks (explicit perturbation of inputs)
    # ------------------------------------------------------------------
    def test_monotonicity_c_decreases_with_epsilon0(self) -> Tuple[bool, str]:
        """
        Perturb ε0 by +10% and verify that the computed c (from the user's expression)
        decreases. The correct relation has c ∝ 1/sqrt(ε0), so increasing ε0 must decrease c.
        """
        rhs = self._parse_rhs()
        if rhs is None:
            return (False, "SKIP: cannot parse RHS of c_relation")

        # Baseline: ε0=1, μ0=1
        try:
            baseline = float(rhs.subs({self.epsilon0: 1.0, self.mu0: 1.0}).evalf())
        except Exception as e:
            return (False, f"Could not evaluate baseline: {e}")

        # Perturb: ε0 increased by 10%
        try:
            perturbed = float(rhs.subs({self.epsilon0: 1.1, self.mu0: 1.0}).evalf())
        except Exception as e:
            return (False, f"Could not evaluate perturbed: {e}")

        if perturbed < baseline:
            return (True, f"c decreases when ε0 is increased by 10%: {baseline:.6f} -> {perturbed:.6f}")
        else:
            return (False, f"c does not decrease when ε0 is increased (baseline={baseline:.6f}, perturbed={perturbed:.6f}). "
                          "Correct relation requires c to decrease with increasing ε0.")

    def test_monotonicity_c_decreases_with_mu0(self) -> Tuple[bool, str]:
        """
        Perturb μ0 by +10% and verify that the computed c (from the user's expression)
        decreases. The correct relation has c ∝ 1/sqrt(μ0), so increasing μ0 must decrease c.
        """
        rhs = self._parse_rhs()
        if rhs is None:
            return (False, "SKIP: cannot parse RHS of c_relation")

        # Baseline: ε0=1, μ0=1
        try:
            baseline = float(rhs.subs({self.epsilon0: 1.0, self.mu0: 1.0}).evalf())
        except Exception as e:
            return (False, f"Could not evaluate baseline: {e}")

        # Perturb: μ0 increased by 10%
        try:
            perturbed = float(rhs.subs({self.epsilon0: 1.0, self.mu0: 1.1}).evalf())
        except Exception as e:
            return (False, f"Could not evaluate perturbed: {e}")

        if perturbed < baseline:
            return (True, f"c decreases when μ0 is increased by 10%: {baseline:.6f} -> {perturbed:.6f}")
        else:
            return (False, f"c does not decrease when μ0 is increased (baseline={baseline:.6f}, perturbed={perturbed:.6f}). "
                          "Correct relation requires c to decrease with increasing μ0.")

    # ------------------------------------------------------------------
    #  Weak Answer Sanity (only one, non‑dominant)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        """
        Weak sanity: ensure the relation is not trivial or empty.
        Does NOT re‑derive the answer; only checks that the expression
        contains meaningful operators and is not a constant.
        """
        if self.c_relation is None or not isinstance(self.c_relation, str):
            return (False, "SKIP: c_relation missing or invalid type")
        # Must contain a square root and a division (from the known correct forms)
        if 'sqrt' not in self.c_relation.lower():
            return (False, "Expression does not contain 'sqrt' (expected)")
        if '/' not in self.c_relation:
            return (False, "Expression does not contain division (expected)")
        return (True, "Expression passes basic structural check (contains sqrt and division)")

    # ------------------------------------------------------------------
    #  Run all tests (discovery pattern)
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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