import sympy as sp
import math
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics checker for the rotating planet oblateness problem.
    Checks interface requirements, physical invariants, model consistency,
    monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens and results; use None if missing.
        # Givens
        self.omega = getattr(solution_instance, 'omega', None)
        self.R_E = getattr(solution_instance, 'R_E', None)
        self.M_e = getattr(solution_instance, 'M_e', None)
        self.G = getattr(solution_instance, 'G', None)
        self.g = getattr(solution_instance, 'g', None)

        # Results (expected from [emit] contract)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.epsilon = solution_instance.results.get('epsilon', None)
            self.epsilon_num = solution_instance.results.get('epsilon_num', None)
        else:
            self.epsilon = getattr(solution_instance, 'epsilon', None)
            self.epsilon_num = getattr(solution_instance, 'epsilon_num', None)

        # Derived parameter (lambda) if available
        self.lambda_ = getattr(solution_instance, 'lambda', None)
        # If not stored, we can compute it from givens later if available.

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

    # --- Interface tests ---

    def test_interface_required_keys_exist(self) -> Tuple[bool, str]:
        """Check that epsilon and epsilon_num are present."""
        if self.epsilon is None:
            return (False, "Missing required result: epsilon")
        if self.epsilon_num is None:
            return (False, "Missing required result: epsilon_num")
        return (True, "epsilon and epsilon_num are present")

    def test_interface_epsilon_type(self) -> Tuple[bool, str]:
        """epsilon should be a string, sympy expression, or numeric compatible with sympy."""
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        # Accept string, sympy expression, or number
        if isinstance(self.epsilon, (str, sp.Basic, int, float)):
            return (True, "epsilon type is acceptable")
        else:
            return (False, f"epsilon has unexpected type {type(self.epsilon)}")

    def test_interface_epsilon_num_type(self) -> Tuple[bool, str]:
        """epsilon_num should be a number (int or float)."""
        if self.epsilon_num is None:
            return (False, "SKIP: epsilon_num not available")
        if isinstance(self.epsilon_num, (int, float)):
            return (True, "epsilon_num is numeric")
        else:
            return (False, f"epsilon_num type {type(self.epsilon_num)} not numeric")

    def test_interface_givens_for_earth(self) -> Tuple[bool, str]:
        """If numerical Earth estimate is present, basic givens should be available."""
        # Not mandatory, but if we have epsilon_num we should have some givens.
        if self.epsilon_num is None:
            return (True, "SKIP: no epsilon_num, no requirement on givens")
        needed = [self.omega, self.R_E, self.M_e, self.G]
        missing = [name for name, val in zip(['omega','R_E','M_e','G'], needed) if val is None]
        if missing:
            return (False, f"Numerical estimate present but missing givens: {missing}")
        return (True, "All necessary givens present for Earth value")

    # --- Invariant tests ---

    def test_invariant_equipotential_condition(self) -> Tuple[bool, str]:
        """
        From the equipotential condition: epsilon = (5/6) * (omega^2 * R_E^3) / (G * M_e).
        If lambda is provided, check epsilon = 5/6 * lambda.
        """
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        # If lambda is directly given, use it.
        if self.lambda_ is not None:
            # Try symbolic simplification
            try:
                expected = sp.Rational(5, 6) * self.lambda_
                diff = sp.simplify(self.epsilon - expected)
                if diff == 0:
                    return (True, "epsilon = 5/6 * lambda symbolically")
                else:
                    return (False, f"Symbolic mismatch: epsilon - (5/6)*lambda = {diff}")
            except (TypeError, sp.SympifyError):
                # Fall back to numeric check
                try:
                    eps = float(self.epsilon)
                    lam = float(self.lambda_)
                    expected_num = 5.0/6.0 * lam
                    if math.isclose(eps, expected_num, rel_tol=1e-3):
                        return (True, "epsilon numerically matches 5/6*lambda")
                    else:
                        return (False, f"epsilon={eps}, expected={expected_num}")
                except (TypeError, ValueError):
                    return (False, "Cannot evaluate epsilon and lambda as numbers")
        else:
            # Try to compute lambda from givens
            if self.omega is not None and self.R_E is not None and self.M_e is not None and self.G is not None:
                try:
                    omega = float(self.omega)
                    R_E = float(self.R_E)
                    M_e = float(self.M_e)
                    G = float(self.G)
                    lam = omega**2 * R_E**3 / (G * M_e)
                    expected_num = 5.0/6.0 * lam
                    # If epsilon is a sympy expression, evaluate numerically
                    if isinstance(self.epsilon, sp.Basic):
                        eps = float(self.epsilon.subs({sp.Symbol('lambda'): lam}))
                    else:
                        eps = float(self.epsilon)
                    if math.isclose(eps, expected_num, rel_tol=1e-2, abs_tol=1e-3):
                        return (True, "epsilon matches (5/6)*lambda computed from givens")
                    else:
                        return (False, f"epsilon={eps}, expected={expected_num}")
                except Exception as e:
                    return (False, f"Error during numeric check: {e}")
            else:
                return (False, "SKIP: insufficient data to check equipotential condition")

    def test_invariant_epsilon_positive(self) -> Tuple[bool, str]:
        """Epsilon must be positive."""
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        try:
            eps = float(self.epsilon) if not isinstance(self.epsilon, sp.Basic) else float(self.epsilon.evalf())
            if eps > 0:
                return (True, f"epsilon = {eps} > 0")
            else:
                return (False, f"epsilon = {eps} is not positive")
        except:
            return (False, "Cannot evaluate epsilon as a number")

    def test_invariant_epsilon_small(self) -> Tuple[bool, str]:
        """Epsilon should be much less than 1 (order 1e-3 for Earth)."""
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        try:
            eps = float(self.epsilon) if not isinstance(self.epsilon, sp.Basic) else float(self.epsilon.evalf())
            if eps < 0.1:
                return (True, f"epsilon = {eps} < 0.1, consistent with small distortion")
            else:
                return (False, f"epsilon = {eps} is not small")
        except:
            return (False, "Cannot evaluate epsilon as a number")

    # --- Modeling consistency tests ---

    def test_modeling_earth_numerical_estimate(self) -> Tuple[bool, str]:
        """
        Check that epsilon_num is approximately 2.9e-3 (weak tolerance).
        Recompute from givens if available.
        """
        if self.epsilon_num is None:
            return (False, "SKIP: epsilon_num not available")
        # Use Earth standard values if givens are missing
        if self.omega is not None and self.R_E is not None and self.g is not None:
            try:
                omega = float(self.omega)
                R_E = float(self.R_E)
                g = float(self.g)
            except:
                return (False, "Cannot convert givens to floats")
            lambda_earth = omega**2 * R_E / g
            expected_eps = 5.0/6.0 * lambda_earth
        else:
            # Fall back to expected approximate value 2.9e-3
            expected_eps = 2.9e-3
        try:
            eps_num = float(self.epsilon_num)
        except:
            return (False, "epsilon_num cannot be converted to float")
        if math.isclose(eps_num, expected_eps, rel_tol=0.1):
            return (True, f"epsilon_num = {eps_num} ~ {expected_eps}")
        else:
            return (False, f"epsilon_num = {eps_num} deviates from expected {expected_eps}")

    def test_modeling_derived_expression(self) -> Tuple[bool, str]:
        """
        Check that the derived epsilon expression is of the form (5/6)*lambda.
        Only if epsilon is available as sympy expression or string.
        """
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        if isinstance(self.epsilon, (sp.Basic, str)):
            try:
                # Try to parse string
                if isinstance(self.epsilon, str):
                    eps = sp.sympify(self.epsilon)
                else:
                    eps = self.epsilon
                # Check if it simplifies to (5/6)*lambda
                lam = sp.symbols('lambda')
                diff = sp.simplify(eps - sp.Rational(5,6)*lam)
                if diff == 0:
                    return (True, "epsilon expression is exactly (5/6)*lambda")
                else:
                    return (False, f"epsilon expression differs from (5/6)*lambda by {diff}")
            except:
                return (False, "Cannot parse epsilon expression")
        else:
            return (False, "epsilon not in symbolic or string form")

    # --- Monotonicity test ---

    def test_monotonicity_epsilon_increases_with_omega(self) -> Tuple[bool, str]:
        """
        Perturb omega by +10% and check that epsilon (or epsilon_num) increases.
        Requires access to omega and epsilon_num or a way to compute epsilon.
        """
        if self.epsilon_num is None:
            return (False, "SKIP: epsilon_num not available")
        if self.omega is None:
            return (False, "SKIP: omega not available")
        try:
            omega0 = float(self.omega)
            eps0 = float(self.epsilon_num)
            # Simulate a 10% increase in omega
            omega1 = omega0 * 1.1
            # Compute new lambda (need g or R_E and g)
            # If g available, use lambda = omega^2 R_E/g
            if self.g is not None and self.R_E is not None:
                g = float(self.g)
                R_E = float(self.R_E)
                lambda0 = omega0**2 * R_E / g
                lambda1 = omega1**2 * R_E / g
                eps1 = 5.0/6.0 * lambda1
            else:
                # Fallback: approximate, assume epsilon ~ omega^2
                eps1 = eps0 * (omega1/omega0)**2
            if eps1 > eps0 + 1e-10:
                return (True, f"Increasing omega from {omega0} to {omega1} raises epsilon from {eps0} to {eps1}")
            else:
                return (False, f"Monotonicity fails: eps not clearly increasing")
        except Exception as e:
            return (False, f"Error during monotonicity test: {e}")

    # --- Weak answer sanity ---

    def test_answer_sanity_epsilon_range(self) -> Tuple[bool, str]:
        """
        Final answer: epsilon should be between 0 and 1, and numerical estimate
        should be around 1e-3 to 1e-2.
        """
        if self.epsilon_num is None:
            return (False, "SKIP: epsilon_num not available")
        try:
            eps = float(self.epsilon_num)
            if 0 <= eps <= 1:
                if 1e-4 <= eps <= 0.1:
                    return (True, f"epsilon_num = {eps} in plausible range")
                else:
                    return (False, f"epsilon_num = {eps} outside typical 1e-4 to 0.1")
            else:
                return (False, f"epsilon_num = {eps} outside [0,1]")
        except:
            return (False, "Cannot convert epsilon_num to float")

    # ========================================
    # Additional test for explanation steps consistency
    # ========================================
    def test_invariant_centrifugal_gravity_balance(self) -> Tuple[bool, str]:
        """
        From explanation: surface equilibrium condition leads to
        R_E^3 * omega^2 * sinθ - R_E^2 * b * G * M_e * sinθ ≈ 0,
        where b = 6*epsilon*R_E^2/(5*R^4) and R ≈ R_E.
        This simplifies to epsilon = (5*R_E*omega^2)/(6*g).
        Check if epsilon satisfies this relation (if g available).
        """
        if self.epsilon is None:
            return (False, "SKIP: epsilon not available")
        if self.R_E is None or self.omega is None or self.g is None:
            return (False, "SKIP: missing R_E, omega, or g")
        try:
            R_E = float(self.R_E)
            omega = float(self.omega)
            g = float(self.g)
            eps = float(self.epsilon) if not isinstance(self.epsilon, sp.Basic) else float(self.epsilon.evalf())
            expected = (5 * R_E * omega**2) / (6 * g)
            if math.isclose(eps, expected, rel_tol=1e-2, abs_tol=1e-3):
                return (True, f"epsilon ≈ (5R_E ω^2)/(6g) = {expected}")
            else:
                return (False, f"epsilon = {eps}, expected {expected}")
        except Exception as e:
            return (False, f"Error: {e}")