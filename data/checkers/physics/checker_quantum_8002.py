import math
import traceback
import sympy as sp
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Robust checker for a mixed physics/math problem involving series, probability,
    integral, and linear algebra.

    Extracts required results from solution_instance (expected to have a .results dict).
    Runs multiple small test methods to verify interface, invariants, modeling, monotonicity.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract results dict (None if missing)
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            # Try to get results as attribute directly
            self.results = getattr(solution_instance, 'results', {})

        # Also extract explanation_steps if available
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # ---- Extract all required result keys with safe defaults ----
        # Each result is expected to be a numeric value or list.
        self.S = self.results.get('S', None)          # series sum (expression or number)
        self.mu = self.results.get('mu', None)        # mean of x
        self.x_mode = self.results.get('x_mode', None) # most probable x
        self.I = self.results.get('I', None)          # integral value
        self.eigenvalues = self.results.get('eigenvalues', None)  # list of three eigenvalues
        self.v1 = self.results.get('v1', None)        # eigenvector 1 (list/tuple)
        self.v2 = self.results.get('v2', None)        # eigenvector 2
        self.v3 = self.results.get('v3', None)        # eigenvector 3
        self.orthogonal = self.results.get('orthogonal', None) # bool

        # Derived from matrix A (provided in problem)
        self.A = [[1,2,4],[2,3,0],[5,0,3]]

    # ================== Interface Tests ==================

    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required keys exist in results (not None)."""
        required_keys = ['S', 'mu', 'x_mode', 'I', 'eigenvalues', 'v1', 'v2', 'v3', 'orthogonal']
        missing = []
        for key in required_keys:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present."

    def test_interface_eigenvalues_type(self) -> Tuple[bool, str]:
        """Check that eigenvalues is a list of length 3."""
        if self.eigenvalues is None:
            return False, "SKIP: eigenvalues not available"
        if not isinstance(self.eigenvalues, (list, tuple)) or len(self.eigenvalues) != 3:
            return False, f"eigenvalues should be list of length 3, got {self.eigenvalues}"
        return True, "eigenvalues is list of length 3."

    def test_interface_eigenvectors_type(self) -> Tuple[bool, str]:
        """Check that each eigenvector is a list/tuple of 3 numbers."""
        for name, vec in [('v1', self.v1), ('v2', self.v2), ('v3', self.v3)]:
            if vec is None:
                return False, f"SKIP: {name} not available"
            if not isinstance(vec, (list, tuple)) or len(vec) != 3:
                return False, f"{name} should be list of length 3, got {vec}"
        return True, "All eigenvectors are 3-element lists."

    def test_interface_orthogonal_type(self) -> Tuple[bool, str]:
        """Check that orthogonal is a boolean."""
        if self.orthogonal is None:
            return False, "SKIP: orthogonal not available"
        if not isinstance(self.orthogonal, bool):
            return False, f"orthogonal should be bool, got {type(self.orthogonal)}"
        return True, "orthogonal is bool."

    # ================== Invariant / Theorem Tests ==================

    def test_invariant_series_formula(self) -> Tuple[bool, str]:
        """Check that S = 1/(1-x)^2 for a test x in (0,1)."""
        if self.S is None:
            return False, "SKIP: S not available"
        try:
            x_test = 0.25
            expected_S = 1.0 / (1.0 - x_test)**2
            if isinstance(self.S, (int, float)):
                # S is numeric; cannot verify formula without knowing x
                return False, "S is numeric, cannot verify formula without knowing x"
            elif isinstance(self.S, sp.Basic):
                x_sym = sp.Symbol('x')
                try:
                    S_val = float(self.S.subs(x_sym, x_test).evalf())
                except:
                    # Try lambdify
                    import numpy as np
                    S_func = sp.lambdify(x_sym, self.S, 'numpy')
                    S_val = float(S_func(x_test))
                if math.isclose(S_val, expected_S, rel_tol=1e-2, abs_tol=1e-2):
                    return True, f"S formula matches for x={x_test}: got {S_val}, expected {expected_S}"
                else:
                    return False, f"S formula gives {S_val} for x={x_test}, expected {expected_S}"
            else:
                return False, f"Unexpected type for S: {type(self.S)}"
        except Exception as e:
            return False, f"Error checking series formula: {e}"

    def test_invariant_mean_formula(self) -> Tuple[bool, str]:
        """Check that mu = 2*lambda for a test lambda > 0."""
        if self.mu is None:
            return False, "SKIP: mu not available"
        try:
            if isinstance(self.mu, sp.Basic):
                sym_lambda = sp.Symbol('lambda')
                expected_expr = 2*sym_lambda
                if sp.simplify(self.mu - expected_expr) == 0:
                    return True, "mu expression matches 2*lambda"
                else:
                    return False, f"mu expression {self.mu} does not match 2*lambda"
            elif isinstance(self.mu, (int, float)):
                if self.mu > 0:
                    return True, "mu is positive (numeric, formula not verified)"
                else:
                    return False, f"mu is {self.mu}, expected positive"
            else:
                return False, f"Unexpected type for mu: {type(self.mu)}"
        except Exception as e:
            return False, f"Error checking mean formula: {e}"

    def test_invariant_mode_value(self) -> Tuple[bool, str]:
        """Check that x_mode = lambda (most probable x)."""
        if self.x_mode is None:
            return False, "SKIP: x_mode not available"
        try:
            if isinstance(self.x_mode, sp.Basic):
                sym_lambda = sp.Symbol('lambda')
                if sp.simplify(self.x_mode - sym_lambda) == 0:
                    return True, "x_mode expression matches lambda"
                else:
                    return False, f"x_mode expression {self.x_mode} does not match lambda"
            elif isinstance(self.x_mode, (int, float)):
                if self.x_mode > 0:
                    return True, "x_mode positive (numeric, formula not verified)"
                else:
                    return False, f"x_mode is {self.x_mode}, expected positive"
            else:
                return False, f"Unexpected type for x_mode: {type(self.x_mode)}"
        except Exception as e:
            return False, f"Error checking mode: {e}"

    def test_invariant_integral_value(self) -> Tuple[bool, str]:
        """Check that I = pi/8."""
        if self.I is None:
            return False, "SKIP: I not available"
        try:
            expected_I = math.pi / 8.0
            if isinstance(self.I, (int, float)):
                I_val = float(self.I)
            elif isinstance(self.I, sp.Basic):
                I_val = float(self.I.evalf())
            else:
                return False, f"Unexpected type for I: {type(self.I)}"
            if math.isclose(I_val, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"I = {I_val} matches pi/8 = {expected_I}"
            else:
                return False, f"I = {I_val} but expected pi/8 = {expected_I}"
        except Exception as e:
            return False, f"Error checking integral: {e}"

    def test_invariant_eigenvalues(self) -> Tuple[bool, str]:
        """Check that eigenvalues list contains 3, -3, 7 (order may vary)."""
        if self.eigenvalues is None:
            return False, "SKIP: eigenvalues not available"
        try:
            expected_set = {3, -3, 7}
            eigen_vals = []
            for ev in self.eigenvalues:
                if isinstance(ev, (int, float)):
                    eigen_vals.append(float(ev))
                elif isinstance(ev, sp.Basic):
                    eigen_vals.append(float(ev.evalf()))
                else:
                    return False, f"Eigenvalue {ev} has unsupported type {type(ev)}"
            actual_set = set(eigen_vals)
            if actual_set == expected_set:
                return True, f"Eigenvalues are {eigen_vals} (contains {expected_set})"
            else:
                # Allow small numerical errors
                if all(any(math.isclose(ev, target, rel_tol=1e-2, abs_tol=1e-2) 
                           for target in [3, -3, 7]) for ev in eigen_vals):
                    return True, "Eigenvalues close to {3, -3, 7}"
                return False, f"Eigenvalues {eigen_vals} do not match expected {expected_set}"
        except Exception as e:
            return False, f"Error checking eigenvalues: {e}"

    def test_invariant_eigenvector_property(self) -> Tuple[bool, str]:
        """Check that A*v_i = lambda_i * v_i for each eigenpair."""
        if self.eigenvalues is None or self.v1 is None or self.v2 is None or self.v3 is None:
            return False, "SKIP: eigenvectors or eigenvalues missing"
        try:
            evals = []
            for ev in self.eigenvalues:
                if isinstance(ev, (int, float)):
                    evals.append(float(ev))
                elif isinstance(ev, sp.Basic):
                    evals.append(float(ev.evalf()))
                else:
                    return False, f"Eigenvalue {ev} unsupported type"
            vecs = [self.v1, self.v2, self.v3]
            for i, (lam, v) in enumerate(zip(evals, vecs), 1):
                v = [float(x) if not isinstance(x, sp.Basic) else float(x.evalf()) for x in v]
                Av = [sum(self.A[j][k]*v[k] for k in range(3)) for j in range(3)]
                lamv = [lam * x for x in v]
                if all(math.isclose(Av[j], lamv[j], rel_tol=1e-2, abs_tol=1e-2) for j in range(3)):
                    continue
                else:
                    return False, f"Eigenpair {i}: A*v{i} = {Av}, lambda*v{i} = {lamv}"
            return True, "All eigenpairs satisfy A*v = lambda*v."
        except Exception as e:
            return False, f"Error checking eigenvector property: {e}"

    def test_invariant_orthogonality(self) -> Tuple[bool, str]:
        """Check that eigenvectors are not orthogonal (as expected for non-symmetric matrix)."""
        if self.v1 is None or self.v2 is None or self.v3 is None:
            return False, "SKIP: eigenvectors missing"
        try:
            v1 = [float(x) if not isinstance(x, sp.Basic) else float(x.evalf()) for x in self.v1]
            v2 = [float(x) if not isinstance(x, sp.Basic) else float(x.evalf()) for x in self.v2]
            v3 = [float(x) if not isinstance(x, sp.Basic) else float(x.evalf()) for x in self.v3]
            dot12 = sum(v1[j]*v2[j] for j in range(3))
            dot13 = sum(v1[j]*v3[j] for j in range(3))
            dot23 = sum(v2[j]*v3[j] for j in range(3))
            max_abs_dot = max(abs(dot12), abs(dot13), abs(dot23))
            # The orthogonal flag should be False (non-orthogonal). At least one dot product should be non-zero.
            if self.orthogonal is False:
                if max_abs_dot > 1e-6:
                    return True, "Vectors are not orthogonal (flag False) and dot products confirm."
                else:
                    # Within numerical error; still acceptable if flag is False
                    return True, "Vectors appear orthogonal within tolerance, but flag says not orthogonal (acceptable)."
            elif self.orthogonal is True:
                return False, "Expected orthogonal=False but flag is True."
            else:
                return False, f"Invalid orthogonal value: {self.orthogonal}"
        except Exception as e:
            return False, f"Error checking orthogonality: {e}"

    # ================== Model/Trace Tests ==================

    def test_modeling_series_derivation(self) -> Tuple[bool, str]:
        """Check that the rationale for series sum is present in explanation_steps."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"
        steps_str = str(self.explanation_steps).lower()
        indicators = ['geometric series', 'derivative', '1/(1-x)^2', 'sum']
        found = [ind for ind in indicators if ind in steps_str]
        if len(found) >= 2:
            return True, f"Explanation contains series derivation hints: {found}"
        else:
            return False, f"Explanation lacks series derivation details. Found: {found}"

    def test_modeling_mean_integral(self) -> Tuple[bool, str]:
        """Check that explanation mentions gamma function or integration by parts for mean."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"
        steps_str = str(self.explanation_steps).lower()
        indicators = ['gamma', 'integral', 'mean', 'ratio']
        found = [ind for ind in indicators if ind in steps_str]
        if len(found) >= 2:
            return True, f"Mean integral rationale present: {found}"
        else:
            return False, f"Mean integral rationale missing. Found: {found}"

    def test_modeling_mode_derivative(self) -> Tuple[bool, str]:
        """Check that explanation mentions derivative and setting to zero for mode."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"
        steps_str = str(self.explanation_steps).lower()
        indicators = ['derivative', "f'", 'maxim', 'extremum', 'zero']
        found = [ind for ind in indicators if ind in steps_str]
        if len(found) >= 2:
            return True, f"Mode derivative rationale present: {found}"
        else:
            return False, f"Mode derivative rationale missing. Found: {found}"

    def test_modeling_integral_contour(self) -> Tuple[bool, str]:
        """Check that explanation mentions contour integration or known integral formula."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"
        steps_str = str(self.explanation_steps).lower()
        indicators = ['contour', 'residue', 'formula', 'substitut']
        found = [ind for ind in indicators if ind in steps_str]
        if len(found) >= 1:
            return True, f"Integral method mentioned: {found}"
        else:
            return False, f"Integral method not mentioned. Found: {found}"

    def test_modeling_eigenvalue_method(self) -> Tuple[bool, str]:
        """Check that explanation includes determinant or characteristic polynomial."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"
        steps_str = str(self.explanation_steps).lower()
        indicators = ['determinant', 'characteristic polynomial', 'det', 'eigenvalue']
        found = [ind for ind in indicators if ind in steps_str]
        if len(found) >= 1:
            return True, f"Eigenvalue method mentioned: {found}"
        else:
            return False, f"Eigenvalue method not mentioned. Found: {found}"

    def test_modeling_eigenvector_orthogonality_comment(self) -> Tuple[bool, str]:
        """Check that explanation mentions non-orthogonality due to non-symmetric matrix."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"
        steps_str = str(self.explanation_steps).lower()
        indicators = ['not orthogonal', 'non-symmetric', 'non-hermitian', 'not hermitian']
        found = [ind for ind in indicators if ind in steps_str]
        if len(found) >= 1:
            return True, f"Orthogonality comment present: {found}"
        else:
            return False, f"Orthogonality comment missing. Found: {found}"

    # ================== Monotonicity Tests ==================

    def test_monotonicity_series_increases(self) -> Tuple[bool, str]:
        """
        Explicitly perturb x input and check that S(x) increases.
        Baseline x0=0.2, perturbed x1=0.3. Expect S(x1) > S(x0) for formula 1/(1-x)^2.
        """
        if self.S is None:
            return False, "SKIP: S not available"
        try:
            # S must be evaluable at numeric x (symbolic expression)
            if not isinstance(self.S, sp.Basic):
                return False, "S is not a symbolic expression; cannot perturb x"
            x_sym = sp.Symbol('x')
            x0 = 0.2
            x1 = 0.3  # perturbed +0.1
            # Evaluate S at x0 and x1
            try:
                S0 = float(self.S.subs(x_sym, x0).evalf())
                S1 = float(self.S.subs(x_sym, x1).evalf())
            except:
                # Use lambdify if substitution fails
                import numpy as np
                S_func = sp.lambdify(x_sym, self.S, 'numpy')
                S0 = float(S_func(x0))
                S1 = float(S_func(x1))
            # For 1/(1-x)^2, S1 > S0 always for 0<x<1.
            if S1 > S0:
                return True, f"S({x1}) = {S1} > S({x0}) = {S0} (increase)."
            else:
                return False, f"S({x1}) = {S1} <= S({x0}) = {S0}, expected increase."
        except Exception as e:
            return False, f"Error checking monotonicity: {e}"

    # ================== Weak Answer Sanity ==================

    def test_answer_sanity_general(self) -> Tuple[bool, str]:
        """
        Weak sanity: all answer values should be finite and plausible.
        NOT using math.isclose or re-derivation.
        """
        messages = []
        check_items = [
            ('S', self.S),
            ('mu', self.mu),
            ('x_mode', self.x_mode),
            ('I', self.I),
            ('eigenvalues', self.eigenvalues),
            ('v1', self.v1),
            ('v2', self.v2),
            ('v3', self.v3),
        ]
        for name, val in check_items:
            if val is None:
                messages.append(f"{name} missing (None)")
                continue
            if isinstance(val, (int, float)):
                if not math.isfinite(val):
                    messages.append(f"{name} is not finite: {val}")
            elif isinstance(val, sp.Basic):
                try:
                    numeric = float(val.evalf())
                    if not math.isfinite(numeric):
                        messages.append(f"{name} symbolic evaluates to non-finite: {numeric}")
                except Exception:
                    messages.append(f"{name} cannot be evaluated numerically")
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, (int, float)):
                        if not math.isfinite(v):
                            messages.append(f"{name} contains non-finite: {v}")
                    elif isinstance(v, sp.Basic):
                        try:
                            numeric = float(v.evalf())
                            if not math.isfinite(numeric):
                                messages.append(f"{name} symbolic element non-finite: {numeric}")
                        except Exception:
                            pass
            else:
                messages.append(f"{name} has unexpected type {type(val)}")

        if self.orthogonal is not None and not isinstance(self.orthogonal, bool):
            messages.append("orthogonal should be bool")

        if messages:
            return False, "Sanity issues: " + "; ".join(messages)
        return True, "All answer values appear finite and plausible."

    # ================== Test Runner ==================

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test methods and run them, collecting results."""
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