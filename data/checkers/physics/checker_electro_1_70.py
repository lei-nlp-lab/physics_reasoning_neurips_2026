import math
import sympy as sp
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits a solution instance for the conducting hemispherical shell problem.
    
    Expected solution_instance attributes (accessed via getattr with None default):
        - R: radius of sphere (float)
        - phi0: potential of upper half (float)
        - results: dict containing keys 'B_0', 'B_1', 'B_2', 'B_3', 'phi'
    """

    def __init__(self, solution_instance: Any):
        # Retrieve numerical inputs safely; do NOT assign numeric defaults.
        self.R = getattr(solution_instance, 'R', None)
        self.phi0 = getattr(solution_instance, 'phi0', None)

        # Retrieve results dict; if missing, use empty dict.
        results = getattr(solution_instance, 'results', None) or {}

        # Extract coefficients and phi expression.
        self.B_0 = results.get('B_0', None)
        self.B_1 = results.get('B_1', None)
        self.B_2 = results.get('B_2', None)
        self.B_3 = results.get('B_3', None)
        self.phi = results.get('phi', None)  # should be a string or sympy expression

        # Also extract explanation steps and pseudocode if present (not used heavily).
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Precompute some symbolic components if phi is a string
        self.phi_sym = None
        if isinstance(self.phi, str):
            try:
                # Parse the target expression, but keep unknowns (r, theta, R, phi0)
                self.phi_sym = sp.sympify(self.phi)
            except Exception:
                pass

        # Define the spherical coordinates symbols (used in tests)
        self.r, self.theta = sp.symbols('r theta', positive=True, real=True)
        self.mu = sp.cos(self.theta)      # cos(theta)

    # --------------------------------------------------------------------------
    # Interface Tests
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist and are not None."""
        required = ['B_0', 'B_1', 'B_2', 'B_3', 'phi']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return False, f"Missing result(s): {missing}"
        return True, "All required keys present and non-None."

    # --------------------------------------------------------------------------
    # Invariant Tests (physical laws)
    # --------------------------------------------------------------------------
    def test_invariant_laplace_equation(self) -> Tuple[bool, str]:
        """Numerically check that φ satisfies Laplace's equation at a test point."""
        if self.phi_sym is None or self.R is None:
            return False, "SKIP: phi expression or R missing."

        # Choose a test point (r0, theta0) well above the sphere
        r0 = 2.0 * self.R if self.R else 2.0
        theta0 = math.pi / 4.0
        # Compute φ and its derivatives using symbolic then numeric evaluation
        phi_expr = self.phi_sym
        # Substitute known R if self.R is a number; otherwise keep symbolic
        subs = {self.r: r0, self.theta: theta0}
        if self.R is not None:
            R_sym = sp.Symbol('R')
            # But if phi_sym already contains R, we need to substitute
            phi_num = phi_expr.subs({sp.Symbol('R'): self.R}).evalf(subs=subs)
        else:
            phi_num = phi_expr.evalf(subs=subs)

        # We cannot compute Laplacian numerically without explicit R substitution.
        # Instead, we do a finite-difference check at a single point.
        # Approximate ∂²φ/∂r² + (2/r)∂φ/∂r + (1/(r² sinθ))∂(sinθ ∂φ/∂θ)/∂θ = 0
        # Use small steps
        dr = 0.01 * r0
        dtheta = 0.01

        def phi_val(rr, tt):
            s = {self.r: rr, self.theta: tt}
            if self.R is not None:
                val = phi_expr.subs({sp.Symbol('R'): self.R}).evalf(subs=s)
            else:
                val = phi_expr.evalf(subs=s)
            return float(val)

        # central differences
        phip = phi_val(r0 + dr, theta0)
        phim = phi_val(r0 - dr, theta0)
        d2r = (phip - 2*phi_num + phim) / (dr**2)
        drr = (phip - phim) / (2*dr)

        phip_th = phi_val(r0, theta0 + dtheta)
        phim_th = phi_val(r0, theta0 - dtheta)
        d2th_num = (phip_th - 2*phi_num + phim_th) / (dtheta**2)
        dth_num = (phip_th - phim_th) / (2*dtheta)

        sinth = math.sin(theta0)
        # Laplacian in spherical coordinates (no phi dependence)
        laplacian = d2r + (2/r0)*drr + (1/(r0**2 * sinth)) * (math.cos(theta0)*dth_num + sinth * d2th_num)

        if math.isclose(float(laplacian), 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Laplacian ~ {laplacian:.5e} near zero."
        else:
            return False, f"Laplacian = {laplacian:.5e}, not sufficiently zero."

    def test_invariant_boundary_at_r_R(self) -> Tuple[bool, str]:
        """Check that φ(R,θ) approximates the step function (up to l=3)."""
        if self.phi_sym is None or self.R is None:
            return False, "SKIP: phi expression or R missing."

        # Evaluate phi at r=R for two angles: θ=0 (North pole) and θ=π (South pole)
        subs_north = {self.r: self.R, self.theta: 0}
        subs_south = {self.r: self.R, self.theta: math.pi}

        phi_north = float(self.phi_sym.subs({sp.Symbol('R'): self.R}).evalf(subs=subs_north))
        phi_south = float(self.phi_sym.subs({sp.Symbol('R'): self.R}).evalf(subs=subs_south))

        # Expected: phi0 at North, 0 at South, but truncation gives error
        expected_north = self.phi0 if self.phi0 is not None else 1.0
        expected_south = 0.0

        errors = []
        if self.phi0 is not None:
            # Allow 10% relative error due to truncation
            if not math.isclose(phi_north, self.phi0, rel_tol=1e-1):
                errors.append(f"North pole φ = {phi_north}, expected ~{self.phi0}")
            if not math.isclose(phi_south, 0.0, abs_tol=0.1*expected_north if expected_north else 0.1):
                errors.append(f"South pole φ = {phi_south}, expected ~0")
        else:
            # Without phi0, just check signs
            if phi_north <= 0:
                errors.append("Expected positive φ at north pole")
            if abs(phi_south) > abs(phi_north):
                errors.append("Expected |φ| at south pole to be smaller than at north pole")

        if errors:
            return False, "; ".join(errors)
        return True, "Approximate boundary conditions satisfied."

    # --------------------------------------------------------------------------
    # Modeling Quality Tests (Legendre coefficients)
    # --------------------------------------------------------------------------
    def _safe_B_l(self, l: int) -> Tuple[bool, str]:
        """Check coefficient B_l against expected value using fuzzy match."""
        coeff = getattr(self, f'B_{l}', None)
        if coeff is None:
            return False, f"B_{l} is missing."
        if self.R is None or self.phi0 is None:
            return False, "SKIP: R or phi0 missing, cannot compute expected value."
        # Compute expected from formula: B_l = R^(l+1) * (2l+1)/2 * phi0 * I_l
        # I_l = int_0^1 P_l(x) dx
        # I_0 = 1, I_1 = 1/2, I_2 = 0, I_3 = -1/8
        I_values = {0: 1, 1: 0.5, 2: 0, 3: -1/8}
        I = I_values.get(l, None)
        if I is None:
            return False, f"Only l=0..3 supported, got l={l}."
        expected_B = (self.R**(l+1)) * ((2*l+1)/2) * self.phi0 * I
        if math.isclose(float(coeff), float(expected_B), rel_tol=1e-2, abs_tol=1e-2):
            return True, f"B_{l} = {coeff} matches expected {expected_B:.5f}"
        else:
            return False, f"B_{l} = {coeff}, expected {expected_B:.5f}"

    def test_modeling_coefficient_B_0(self):
        return self._safe_B_l(0)

    def test_modeling_coefficient_B_1(self):
        return self._safe_B_l(1)

    def test_modeling_coefficient_B_2(self):
        return self._safe_B_l(2)

    def test_modeling_coefficient_B_3(self):
        return self._safe_B_l(3)

    def test_modeling_phi_structure(self) -> Tuple[bool, str]:
        """Check that phi expression has correct terms: powers 1/r, 1/r^2, 1/r^4 and no 1/r^3 term."""
        if self.phi_sym is None:
            return False, "SKIP: phi expression not available."
        # Expand phi expression in terms of r
        expr = sp.expand(self.phi_sym)
        # Look for terms with r^(-1), r^(-2), r^(-4) and check absence of r^(-3)
        powers = []
        for term in sp.Add.make_args(expr):
            # Get r exponent
            if term.has(self.r):
                try:
                    _, exp = term.as_independent(self.r, as_Add=False)[1].as_base_exp()
                    powers.append(exp)
                except:
                    powers.append(None)
        # Expect -1, -2, -4 (the exponents are negative because r in denominator)
        expected_exponents = {-1, -2, -4}
        unique = set(powers) - {None}
        missing = expected_exponents - unique
        extra = unique - expected_exponents
        if missing:
            return False, f"Missing term(s) with r^{list(missing)}."
        if extra:
            return False, f"Unexpected term(s) with r^{list(extra)}."
        return True, "Phi expression has correct r-dependence."

    # --------------------------------------------------------------------------
    # Monotonicity Tests (explicit perturbation)
    # --------------------------------------------------------------------------
    def test_monotonicity_potential_decays_with_increasing_r(self) -> Tuple[bool, str]:
        """
        Perturb r (radial coordinate) by +5% and check that |φ| decreases.
        Baseline at r0 = 2*R, perturb to r0*1.05.
        """
        if self.phi_sym is None or self.R is None:
            return False, "SKIP: phi expression or R missing."
        theta_test = math.pi / 4.0  # fixed angle
        r_base = 2.0 * self.R
        r_perturbed = r_base * 1.05  # 5% increase

        subs_base = {self.r: r_base, self.theta: theta_test}
        subs_pert = {self.r: r_perturbed, self.theta: theta_test}

        try:
            phi_base = float(self.phi_sym.subs({sp.Symbol('R'): self.R}).evalf(subs=subs_base))
            phi_pert = float(self.phi_sym.subs({sp.Symbol('R'): self.R}).evalf(subs=subs_pert))
        except Exception:
            return False, "Error evaluating phi."

        # Expect magnitude to decrease as r increases
        if abs(phi_pert) >= abs(phi_base):
            return False, f"|phi| does not decrease: baseline={abs(phi_base):.6f}, perturbed={abs(phi_pert):.6f}."
        return True, f"|phi| decreased from {abs(phi_base):.6f} to {abs(phi_pert):.6f} upon +5% r increase."

    def test_monotonicity_potential_decreases_with_theta(self) -> Tuple[bool, str]:
        """
        Perturb θ by +10% (from π/6 to a larger angle) and check that φ decreases
        (since upper half is φ0 and lower half is 0, φ should generally decrease
        as we move from north pole toward equator).
        """
        if self.phi_sym is None or self.R is None:
            return False, "SKIP: phi expression or R missing."
        r_fixed = 2.0 * self.R
        theta_base = math.pi / 6.0   # 30 degrees
        theta_pert = theta_base * 1.10  # +10% (still < π/2)

        subs_base = {self.r: r_fixed, self.theta: theta_base}
        subs_pert = {self.r: r_fixed, self.theta: theta_pert}

        try:
            phi_base = float(self.phi_sym.subs({sp.Symbol('R'): self.R}).evalf(subs=subs_base))
            phi_pert = float(self.phi_sym.subs({sp.Symbol('R'): self.R}).evalf(subs=subs_pert))
        except Exception:
            return False, "Error evaluating phi."

        # Expect φ to be positive and decreasing (since upper half)
        if phi_base <= 0 or phi_pert <= 0:
            return False, "Phi is not positive in this region (expected positive near north pole)."
        if phi_pert >= phi_base:
            return False, f"phi did not decrease: baseline={phi_base:.6f}, perturbed={phi_pert:.6f}."
        return True, f"phi decreased from {phi_base:.6f} to {phi_pert:.6f} upon +10% θ increase."

    def test_monotonicity_phi0_scales_linear(self) -> Tuple[bool, str]:
        """
        Perturb phi0 by factor 2 and check that φ scales linearly.
        Baseline at phi0, perturbed at 2*phi0.
        """
        if self.phi0 is None or self.phi_sym is None or self.R is None:
            return False, "SKIP: phi0, phi expression, or R missing."

        r_test = self.R * 2
        theta_test = math.pi / 4
        subs = {self.r: r_test, self.theta: theta_test}

        try:
            # Use symbolic substitution to evaluate with different phi0
            # The expression likely contains phi0 as a symbol
            sym_phi0 = sp.Symbol('phi0')
            phi_base = float(self.phi_sym.subs({sp.Symbol('R'): self.R, sym_phi0: self.phi0}).evalf(subs=subs))
            phi_double = float(self.phi_sym.subs({sp.Symbol('R'): self.R, sym_phi0: 2*self.phi0}).evalf(subs=subs))
        except Exception:
            return False, "Error evaluating phi with perturbation."

        # The ratio should be 2
        if math.isclose(phi_double / phi_base, 2.0, rel_tol=1e-2):
            return True, "phi scales linearly with phi0."
        else:
            return False, f"Scaling factor {phi_double/phi_base} not close to 2."

    # --------------------------------------------------------------------------
    # Weak Answer Sanity (only one such method allowed)
    # --------------------------------------------------------------------------
    def test_answer_sanity_finite_and_sign(self) -> Tuple[bool, str]:
        """Weak check: phi should be finite and have same sign as phi0 (for reasonable region)."""
        if self.phi_sym is None or self.R is None:
            return False, "SKIP: phi expression or R missing."

        # Test a few points: near axis, far away.
        test_points = [
            (self.R * 1.2, 0.0),
            (self.R * 2.0, math.pi/2),
            (self.R * 5.0, math.pi)
        ]
        errors = []
        for r_val, th_val in test_points:
            subs = {self.r: r_val, self.theta: th_val}
            try:
                val = float(self.phi_sym.subs({sp.Symbol('R'): self.R}).evalf(subs=subs))
            except Exception:
                errors.append(f"Failed to evaluate at (r={r_val}, θ={th_val})")
                continue
            if not math.isfinite(val):
                errors.append(f"Non-finite value at (r={r_val}, θ={th_val}): {val}")
            # If phi0 is known, check sign consistency
            if self.phi0 is not None and self.phi0 != 0:
                # At north pole (θ=0), sign should match phi0 (plus small positive)
                if th_val == 0 and val * self.phi0 < 0:
                    errors.append(f"Sign mismatch at north pole: phi={val}, phi0={self.phi0}")
        if errors:
            return False, "; ".join(errors)
        return True, "Basic sanity checks passed."

    # --------------------------------------------------------------------------
    # Test Runner
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collecting results."""
        logs = []
        all_pass = True
        test_methods = [getattr(self, name) for name in dir(self)
                        if name.startswith('test_') and callable(getattr(self, name))]
        for method in test_methods:
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {method.__name__}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs