import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the dielectric sphere in a uniform electric field problem.
    Audits correctness of computed electric fields inside and outside the sphere.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.R = givens.get('R', None)
        self.epsilon = givens.get('epsilon', None)          # dielectric constant of sphere
        self.epsilon0 = givens.get('epsilon0', None)        # permittivity of vacuum
        self.E0 = givens.get('E0', None)                    # magnitude of external field

        # Extract results safely (default None)
        results = getattr(solution_instance, 'results', {}) or {}
        self.E_in = results.get('E_in', None)
        self.E_out = results.get('E_out', None)

        # Optional: explanation steps (not heavily used)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Define symbolic variables for later evaluation (if needed)
        self.r_var = sp.Symbol('r', positive=True, real=True)
        self.theta_var = sp.Symbol('theta', real=True)

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_has_E_in(self):
        """Check that results contain E_in and it is not None."""
        if self.E_in is None:
            return (False, "E_in is None")
        return (True, "E_in is present")

    def test_interface_has_E_out(self):
        """Check that results contain E_out and it is not None."""
        if self.E_out is None:
            return (False, "E_out is None")
        return (True, "E_out is present")

    def test_interface_givens_exist(self):
        """Check that all required givens are provided and not None."""
        missing = []
        if self.R is None:
            missing.append("R")
        if self.epsilon is None:
            missing.append("epsilon")
        if self.epsilon0 is None:
            missing.append("epsilon0")
        if self.E0 is None:
            missing.append("E0")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens present")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_E_in_uniform(self):
        """
        Inside the sphere (r < R), the electric field must be uniform
        (same magnitude and direction at all points).
        We evaluate E_in at two different interior points and compare.
        """
        if self.E_in is None or self.R is None or self.epsilon is None \
           or self.epsilon0 is None or self.E0 is None:
            return (False, "SKIP: insufficient data for this test")

        try:
            # Evaluate at two points inside
            r1 = 0.2 * self.R
            r2 = 0.5 * self.R
            theta1 = 0.0
            theta2 = math.pi / 4

            # Helper to compute magnitude squared at a point
            def mag2_at(r, theta):
                if isinstance(self.E_in, (list, tuple)):
                    if len(self.E_in) < 3:
                        return None
                    subs = {self.r_var: r, self.theta_var: theta}
                    comps = [float(c.subs(subs).evalf()) for c in self.E_in]
                else:
                    # Assume it's a SymPy vector object with components
                    try:
                        vx = self.E_in.dot(sp.Matrix([1, 0, 0]))
                        vy = self.E_in.dot(sp.Matrix([0, 1, 0]))
                        vz = self.E_in.dot(sp.Matrix([0, 0, 1]))
                        subs = {self.r_var: r, self.theta_var: theta}
                        comps = [float(c.subs(subs).evalf()) for c in (vx, vy, vz)]
                    except Exception:
                        return None
                return sum(v*v for v in comps)

            mag1 = mag2_at(r1, theta1)
            mag2 = mag2_at(r2, theta2)
            if mag1 is None or mag2 is None:
                return (False, "Could not numerically evaluate E_in at sample points")

            # Fields should be equal (uniform)
            if not math.isclose(mag1, mag2, rel_tol=1e-2):
                return (False, f"E_in magnitude not uniform: {mag1} vs {mag2}")
            return (True, "E_in is uniform inside the sphere")
        except Exception as e:
            return (False, f"Error in test_invariant_E_in_uniform: {e}")

    def test_invariant_E_out_at_infinity(self):
        """
        Far away from the sphere (r >> R), the field should approach
        the external uniform field E0 * e_z.
        """
        if self.E_out is None or self.R is None or self.E0 is None:
            return (False, "SKIP: insufficient data for this test")

        try:
            # Evaluate at large distance, e.g., r = 100 * R, on the z-axis (theta=0)
            r_large = 100.0 * self.R
            theta = 0.0

            # Extract z-component
            if isinstance(self.E_out, (list, tuple)):
                if len(self.E_out) < 3:
                    return (False, "E_out does not have three components")
                Ez_expr = self.E_out[2]
            else:
                try:
                    Ez_expr = self.E_out.dot(sp.Matrix([0, 0, 1]))
                except AttributeError:
                    return (False, "Cannot extract z-component from E_out")

            subs_dict = {self.r_var: r_large, self.theta_var: theta}
            Ez_val = float(Ez_expr.subs(subs_dict).evalf())

            # Expected: |E_out| ≈ E0 (with small dipole contribution)
            expected = float(self.E0)
            if not math.isclose(Ez_val, expected, rel_tol=1e-2):
                return (False, f"E_out at large r is {Ez_val}, expected {expected}")
            return (True, "E_out approaches E0 at infinity")
        except Exception as e:
            return (False, f"Error in test_invariant_E_out_at_infinity: {e}")

    # ----------------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # ----------------------------------------------------------------------
    def test_monotonicity_E_in_decreases_with_epsilon(self):
        """
        The internal field E_in decreases as the dielectric constant epsilon increases.
        Perturb epsilon by +10% and verify the magnitude decreases.
        """
        if self.E_in is None or self.epsilon is None or self.epsilon0 is None or self.E0 is None:
            return (False, "SKIP: insufficient data for this test")

        try:
            # Use analytic formula for |E_in| (re-derived inside checker)
            def E_in_mag(eps):
                return abs(3 * self.epsilon0 / (eps + 2 * self.epsilon0)) * abs(self.E0)

            baseline_eps = float(self.epsilon)
            # Perturb epsilon upward by 10%
            pert_eps = baseline_eps * 1.1

            base_mag = E_in_mag(baseline_eps)
            pert_mag = E_in_mag(pert_eps)

            if pert_mag >= base_mag - 1e-12:  # allow tiny tolerance for numerical noise
                return (False, f"E_in magnitude did not decrease when epsilon increased: "
                               f"base={base_mag}, perturbed={pert_mag}")
            return (True, "E_in decreases monotonically with increasing epsilon")
        except Exception as e:
            return (False, f"Error in test_monotonicity_E_in_decreases_with_epsilon: {e}")

    def test_monotonicity_E_out_dipole_decays_with_r(self):
        """
        Outside the sphere, the dipole part of the field decays as 1/r^3.
        Perturb radial distance r by +10% and verify that the dipole
        contribution |E_out - E0| decreases.
        """
        if self.E_out is None or self.R is None or self.E0 is None:
            return (False, "SKIP: insufficient data for this test")

        try:
            # Baseline: r = 2*R
            r_baseline = 2.0 * self.R
            # Perturbed: r = 2.2*R (10% increase)
            r_pert = 2.2 * self.R
            theta = 0.0  # on the axis for simplicity

            # Extract z-component of E_out
            if isinstance(self.E_out, (list, tuple)):
                if len(self.E_out) < 3:
                    return (False, "E_out does not have three components")
                Ez_expr = self.E_out[2]
            else:
                try:
                    Ez_expr = self.E_out.dot(sp.Matrix([0, 0, 1]))
                except AttributeError:
                    return (False, "Cannot extract z-component from E_out")

            # Helper: compute |E_out - E0| along z-axis (since E0 is only z)
            def dipole_magnitude(r):
                subs = {self.r_var: r, self.theta_var: theta}
                Ez = float(Ez_expr.subs(subs).evalf())
                return abs(Ez - float(self.E0))

            base_dip = dipole_magnitude(r_baseline)
            pert_dip = dipole_magnitude(r_pert)

            # Expect dipole to decrease when r increases
            if pert_dip >= base_dip - 1e-12:
                return (False, f"Dipole contribution did not decrease when r increased: "
                               f"baseline={base_dip}, perturbed={pert_dip}")
            return (True, "E_out dipole term decays with increasing r")
        except Exception as e:
            return (False, f"Error in test_monotonicity_E_out_dipole_decays_with_r: {e}")

    # ----------------------------------------------------------------------
    # Modeling quality checks (based on physics constraints)
    # ----------------------------------------------------------------------
    def test_modeling_E_in_form(self):
        """
        The internal field should be of the form C * e_z with
        C = 3ε0/(ε+2ε0) E0 (derived from boundary conditions).
        We compare the symbolic expression of E_in to this expected form.
        """
        if self.E_in is None or self.epsilon is None or self.epsilon0 is None or self.E0 is None:
            return (False, "SKIP: insufficient data for this test")

        try:
            # Define expected magnitude expression
            expected_mag = 3 * self.epsilon0 / (self.epsilon + 2 * self.epsilon0) * self.E0

            # Extract z-component (assume E_in is [0, 0, Ez] or similar)
            if isinstance(self.E_in, (list, tuple)):
                if len(self.E_in) < 3:
                    return (False, "E_in does not have three components")
                Ex_in, Ey_in, Ez_in = self.E_in[0], self.E_in[1], self.E_in[2]
            else:
                try:
                    Ex_in = self.E_in.dot(sp.Matrix([1, 0, 0]))
                    Ey_in = self.E_in.dot(sp.Matrix([0, 1, 0]))
                    Ez_in = self.E_in.dot(sp.Matrix([0, 0, 1]))
                except AttributeError:
                    return (False, "Cannot extract components from E_in")

            # Check x and y components are zero
            if not sp.simplify(Ex_in) == 0 or not sp.simplify(Ey_in) == 0:
                return (False, "E_in has non-zero x or y components")
            # Check z component matches expected
            if not sp.simplify(Ez_in - expected_mag) == 0:
                return (False, f"E_in z-component does not match expected: {Ez_in} vs {expected_mag}")
            return (True, "E_in has correct form: (0,0, C) with C = 3ε0/(ε+2ε0) E0")
        except Exception as e:
            return (False, f"Error in test_modeling_E_in_form: {e}")

    def test_modeling_E_out_dipole_structure(self):
        """
        The outside field should be E0 e_z plus a dipole term proportional to
        (ε-ε0)/(ε+2ε0) R^3 [3(E0·r)r/r^5 - E0/r^3].
        We check the form on the z-axis (theta=0) and equator (theta=π/2).
        """
        if self.E_out is None or self.R is None or self.epsilon is None \
           or self.epsilon0 is None or self.E0 is None:
            return (False, "SKIP: insufficient data for this test")

        try:
            # Extract z-component
            if isinstance(self.E_out, (list, tuple)):
                if len(self.E_out) < 3:
                    return (False, "E_out does not have three components")
                Ez_out = self.E_out[2]
            else:
                try:
                    Ez_out = self.E_out.dot(sp.Matrix([0, 0, 1]))
                except AttributeError:
                    return (False, "Cannot extract z-component from E_out")

            r = self.r_var
            theta = self.theta_var

            # Expected expressions:
            # On z-axis (theta=0): dipole term = 2*A*R^3 / r^3, A = (ε-ε0)/(ε+2ε0) E0
            A = (self.epsilon - self.epsilon0) / (self.epsilon + 2 * self.epsilon0) * self.E0
            expected_Ez_axis = self.E0 + 2 * A * self.R**3 / r**3

            subs_axis = {theta: 0}
            Ez_axis_expr = sp.simplify(Ez_out.subs(subs_axis))
            diff_axis = sp.simplify(Ez_axis_expr - expected_Ez_axis)
            if diff_axis != 0:
                return (False, f"E_out on axis does not match expected form: {diff_axis}")

            # On equator (theta=π/2): dipole term = -A*R^3 / r^3
            expected_Ez_equator = self.E0 - A * self.R**3 / r**3
            subs_equator = {theta: sp.pi/2}
            Ez_equator_expr = sp.simplify(Ez_out.subs(subs_equator))
            diff_equator = sp.simplify(Ez_equator_expr - expected_Ez_equator)
            if diff_equator != 0:
                return (False, f"E_out on equator does not match expected form: {diff_equator}")

            return (True, "E_out has correct dipole structure on axis and equator")
        except Exception as e:
            return (False, f"Error in test_modeling_E_out_dipole_structure: {e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, must be minimal)
    # ----------------------------------------------------------------------
    def test_answer_sanity_sign(self):
        """
        Basic sanity: E_in magnitude should be non-negative and finite.
        Also E_out should be finite for any r>0.
        """
        if self.E_in is None or self.E_out is None:
            return (False, "SKIP: E_in or E_out missing")
        try:
            # Evaluate E_in at an interior point (r=0.5R, theta=0)
            if isinstance(self.E_in, (list, tuple)):
                if len(self.E_in) < 3:
                    return (False, "E_in not a 3-component vector")
                subs = {self.r_var: 0.5 * self.R, self.theta_var: 0}
                comps = [float(c.subs(subs).evalf()) for c in self.E_in]
            else:
                # Assume vector with .norm()
                try:
                    vx = self.E_in.dot(sp.Matrix([1, 0, 0]))
                    vy = self.E_in.dot(sp.Matrix([0, 1, 0]))
                    vz = self.E_in.dot(sp.Matrix([0, 0, 1]))
                    subs = {self.r_var: 0.5 * self.R, self.theta_var: 0}
                    comps = [float(c.subs(subs).evalf()) for c in (vx, vy, vz)]
                except AttributeError:
                    return (False, "Cannot extract components from E_in")

            for val in comps:
                if math.isnan(val) or math.isinf(val):
                    return (False, f"E_in contains NaN or Inf: {comps}")
                if val < 0:
                    return (False, f"E_in component negative: {val}")
            # Also check E_out at some point
            if isinstance(self.E_out, (list, tuple)):
                if len(self.E_out) < 3:
                    return (False, "E_out not a 3-component vector")
                subs_out = {self.r_var: 2 * self.R, self.theta_var: 0}
                comps_out = [float(c.subs(subs_out).evalf()) for c in self.E_out]
                for val in comps_out:
                    if math.isnan(val) or math.isinf(val):
                        return (False, f"E_out contains NaN or Inf: {comps_out}")
            return (True, "E_in and E_out appear finite and non-negative")
        except Exception as e:
            return (False, f"Error in test_answer_sanity_sign: {e}")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """
        Execute all test methods and return (overall_pass, logs).
        """
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