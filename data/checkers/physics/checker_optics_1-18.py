import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for the rotating mercury mirror telescope problem.

    Checks:
    - Interface: presence and type of required output keys.
    - Invariant: parabolic surface relation and focal length formula.
    - Modeling: surface equation structure.
    - Monotonicity: focal length vs. angular velocity trend.
    - Answer sanity: positivity and rough magnitude of omega.
    """

    def __init__(self, solution_instance):
        # Extract results safely; all defaults are None.
        # The solution_instance is expected to have a 'results' dict.
        results = getattr(solution_instance, 'results', {})
        self.surface_eq = results.get('surface_eq', None)
        self.omega = results.get('omega', None)

        # Known constants (from problem statement)
        self.g = 9.8  # m/s^2
        self.f_target = 0.1  # m

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_surface_eq_type(self):
        """surface_eq must be a non-empty string."""
        if self.surface_eq is None:
            return (False, "surface_eq is missing (None)")
        if not isinstance(self.surface_eq, str):
            return (False, f"surface_eq is not a string, got {type(self.surface_eq)}")
        if len(self.surface_eq.strip()) == 0:
            return (False, "surface_eq is an empty string")
        return (True, "surface_eq is a non-empty string")

    def test_interface_omega_type(self):
        """omega must be a positive number (int or float)."""
        if self.omega is None:
            return (False, "omega is missing (None)")
        if not isinstance(self.omega, (int, float)):
            return (False, f"omega is not a number, got {type(self.omega)}")
        if self.omega <= 0:
            return (False, f"omega is not positive: {self.omega}")
        return (True, f"omega is a positive number: {self.omega}")

    # ----------------------------------------------------------------------
    # Invariant / physics sanity checks (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_parabolic_form(self):
        """
        For a parabola z = a r^2, the focal length is f = 1/(4a).
        The relation a = ω^2/(2g) must hold.
        Verify that if we extract 'a' from the surface equation and compare
        with ω^2/(2g), they match within tolerance.
        """
        if self.surface_eq is None or self.omega is None:
            return (False, "SKIP: missing surface_eq or omega")

        # Parse the surface equation to extract 'a'.
        # Expected form: 'z = (omega^2/(2g)) r^2' or similar.
        # We'll do a simple symbolic check: assume the equation is of the
        # form z = a*r^2. Use sympy to extract a.
        try:
            # Build symbolic expression
            r = sp.symbols('r')
            # We'll attempt to parse the string as an equation 'z = expr'
            eq_str = self.surface_eq.strip()
            if '=' not in eq_str:
                return (False, "surface_eq does not contain '='")
            # Take right-hand side
            rhs_str = eq_str.split('=')[1].strip()
            # Convert to sympy expression
            rhs_expr = sp.sympify(rhs_str)
            # Check that it is of the form a*r^2
            if rhs_expr.is_Mul or rhs_expr.is_Pow:
                # Try to get coefficient of r^2
                coeff = sp.expand(rhs_expr / r**2)
                # coeff should be a constant expression (might contain omega and g)
                # Compare with ω^2/(2g)
                omega_sym = sp.symbols('omega')
                g_sym = sp.symbols('g')
                expected_a = omega_sym**2 / (2 * g_sym)
                # Substitute numeric values
                coeff_num = float(coeff.subs({sp.symbols('omega'): self.omega, sp.symbols('g'): self.g}))
                expected_a_num = self.omega**2 / (2 * self.g)
                if math.isclose(coeff_num, expected_a_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Extracted a = {coeff_num} matches ω^2/(2g) = {expected_a_num}")
                else:
                    return (False, f"Extracted a = {coeff_num} does not match ω^2/(2g) = {expected_a_num}")
            else:
                return (False, "Surface equation is not of the form z = a * r^2")
        except Exception as e:
            return (False, f"Failed to parse surface_eq: {e}")

    def test_invariant_focal_length_formula(self):
        """
        For a parabola, f = 1/(4a). Derive a from omega and g,
        compute f = g/(2ω^2), and compare with the target f_target.
        """
        if self.omega is None:
            return (False, "SKIP: omega missing")
        # Derived focal length
        f_derived = self.g / (2 * self.omega**2)
        if math.isclose(f_derived, self.f_target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derived focal length {f_derived} m matches target {self.f_target} m")
        else:
            return (False, f"Derived focal length {f_derived} m does not match target {self.f_target} m")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_surface_eq_contains_omega_sq(self):
        """Surface equation should contain 'omega^2' (or 'ω^2') as part of the expression."""
        if self.surface_eq is None:
            return (False, "SKIP: surface_eq missing")
        # Check for 'omega^2' or its Unicode variant
        if 'omega^2' in self.surface_eq or 'ω^2' in self.surface_eq:
            return (True, "Surface equation contains omega^2")
        else:
            return (False, "Surface equation does not contain omega^2 or ω^2")

    def test_modeling_surface_eq_contains_r_sq(self):
        """Surface equation should contain 'r^2'."""
        if self.surface_eq is None:
            return (False, "SKIP: surface_eq missing")
        if 'r^2' in self.surface_eq:
            return (True, "Surface equation contains r^2")
        else:
            return (False, "Surface equation does not contain r^2")

    # ----------------------------------------------------------------------
    # Monotonicity checks
    # ----------------------------------------------------------------------
    def test_monotonicity_focal_length_decreases_with_omega(self):
        """
        As omega increases, focal length f = g/(2ω^2) should decrease.
        Perturb omega by +10% and verify that f_perturbed < f_original.
        """
        if self.omega is None:
            return (False, "SKIP: omega missing")
        omega_orig = self.omega
        f_orig = self.g / (2 * omega_orig**2)

        # Increase omega by 10%
        omega_pert = omega_orig * 1.1
        f_pert = self.g / (2 * omega_pert**2)

        if f_pert < f_orig:
            return (True, f"f decreased from {f_orig} to {f_pert} when omega increased by 10%")
        else:
            return (False, f"Monotonicity fails: f did not decrease (orig {f_orig}, perturb {f_pert})")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one such method, no numeric re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_omega_sign_and_magnitude(self):
        """
        Check that omega is positive and roughly in the ballpark of 7 rad/s.
        Accept a range 5-10 rad/s based on g=9.8 and f=0.1 m.
        """
        if self.omega is None:
            return (False, "SKIP: omega missing")
        if self.omega <= 0:
            return (False, f"omega is not positive: {self.omega}")
        # Rough magnitude: for g=9.8, f=0.1 => ω ~ sqrt(9.8/(2*0.1)) = sqrt(49) = 7
        if not (5 < self.omega < 10):
            return (False, f"omega = {self.omega} is outside expected range [5, 10] rad/s")
        return (True, f"omega = {self.omega} rad/s is positive and within expected range")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all methods starting with "test_"
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