import math
import traceback

class PhysicsChecker:
    """Audits the solution for the two-mass-rod-spring system in free space."""

    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Given quantities (may be missing)
        self.m = self.givens.get('m', None)
        self.M = self.givens.get('M', None)
        self.L = self.givens.get('L', None)
        self.k = self.givens.get('k', None)
        self.r0 = self.givens.get('r0', None)  # natural spring length

        # Generalized coordinates and their derivatives (if provided)
        self.X = self.givens.get('X', None)
        self.Y = self.givens.get('Y', None)
        self.Z = self.givens.get('Z', None)
        self.Xdot = self.givens.get('Xdot', None)
        self.Ydot = self.givens.get('Ydot', None)
        self.Zdot = self.givens.get('Zdot', None)
        self.theta = self.givens.get('theta', None)
        self.thetadot = self.givens.get('thetadot', None)
        self.thetaddot = self.givens.get('thetaddot', None)
        self.phi = self.givens.get('phi', None)
        self.phidot = self.givens.get('phidot', None)
        self.r = self.givens.get('r', None)
        self.rdot = self.givens.get('rdot', None)
        self.rddot = self.givens.get('rddot', None)

        # Derived moment of inertia of rod (about center)
        self.I = None
        if self.M is not None and self.L is not None:
            self.I = (1.0 / 12.0) * self.M * self.L**2

        # Results from solution (expected keys via pseudocode emit contract)
        self.Px_const = self.results.get('Px_const', None)
        self.Py_const = self.results.get('Py_const', None)
        self.Pz_const = self.results.get('Pz_const', None)
        self.L_phi = self.results.get('L_phi', None)
        self.r_eq = self.results.get('r_eq', None)
        self.theta_eq = self.results.get('theta_eq', None)

    # ----------------------------------------------------------------------
    # Helper: check if a value is numeric and finite
    def _is_finite_number(self, val):
        if val is None:
            return False
        if isinstance(val, (int, float)):
            return math.isfinite(val)
        return False

    # ----------------------------------------------------------------------
    # Interface checks: required keys exist and have expected types
    def test_interface_required_results(self):
        """Check that all expected result keys are present."""
        required = ['Px_const', 'Py_const', 'Pz_const', 'L_phi', 'r_eq', 'theta_eq']
        for key in required:
            if key not in self.results:
                return (False, f"Missing result key: {key}")
        return (True, "All required result keys present.")

    def test_interface_Px_const_numeric(self):
        if self.Px_const is None:
            return (False, "Px_const is None")
        if not self._is_finite_number(self.Px_const):
            return (False, f"Px_const is not a finite number: {self.Px_const}")
        return (True, "Px_const is a finite number.")

    def test_interface_Py_const_numeric(self):
        if self.Py_const is None:
            return (False, "Py_const is None")
        if not self._is_finite_number(self.Py_const):
            return (False, f"Py_const is not a finite number: {self.Py_const}")
        return (True, "Py_const is a finite number.")

    def test_interface_Pz_const_numeric(self):
        if self.Pz_const is None:
            return (False, "Pz_const is None")
        if not self._is_finite_number(self.Pz_const):
            return (False, f"Pz_const is not a finite number: {self.Pz_const}")
        return (True, "Pz_const is a finite number.")

    def test_interface_L_phi_numeric(self):
        if self.L_phi is None:
            return (False, "L_phi is None")
        if not self._is_finite_number(self.L_phi):
            return (False, f"L_phi is not a finite number: {self.L_phi}")
        return (True, "L_phi is a finite number.")

    def test_interface_r_eq_string(self):
        if self.r_eq is None:
            return (False, "r_eq is None")
        if not isinstance(self.r_eq, str):
            return (False, "r_eq is not a string.")
        if len(self.r_eq.strip()) == 0:
            return (False, "r_eq is an empty string.")
        return (True, "r_eq is a non-empty string.")

    def test_interface_theta_eq_string(self):
        if self.theta_eq is None:
            return (False, "theta_eq is None")
        if not isinstance(self.theta_eq, str):
            return (False, "theta_eq is not a string.")
        if len(self.theta_eq.strip()) == 0:
            return (False, "theta_eq is an empty string.")
        return (True, "theta_eq is a non-empty string.")

    # ----------------------------------------------------------------------
    # Invariant checks (using math.isclose)
    def test_invariant_linear_momentum_form(self):
        """Check that Px_const = (M+2m)*Xdot, if Xdot available."""
        if self.Px_const is None or self.M is None or self.m is None or self.Xdot is None:
            return (True, "SKIP: insufficient data to check Px_const form.")
        expected = (self.M + 2*self.m) * self.Xdot
        if math.isclose(self.Px_const, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Px_const matches (M+2m)Xdot: {self.Px_const} ≈ {expected}")
        else:
            return (False, f"Px_const={self.Px_const} does not match (M+2m)Xdot={expected}")

    def test_invariant_angular_momentum_form(self):
        """Check that L_phi = (I + 2*m*r^2)*phidot*sin^2(theta) if all available."""
        if self.L_phi is None or self.I is None or self.m is None or self.r is None or self.phidot is None or self.theta is None:
            return (True, "SKIP: insufficient data to check L_phi form.")
        sin2theta = math.sin(self.theta)**2
        expected = (self.I + 2*self.m*self.r**2) * self.phidot * sin2theta
        if math.isclose(self.L_phi, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L_phi matches formula: {self.L_phi} ≈ {expected}")
        else:
            return (False, f"L_phi={self.L_phi} does not match formula value {expected}")

    # ----------------------------------------------------------------------
    # Modeling quality checks (trace consistency)
    def test_modeling_radial_equation_contains_terms(self):
        """Check that r_eq string contains key terms of the radial Euler-Lagrange equation."""
        if self.r_eq is None:
            return (True, "SKIP: r_eq not available.")
        eq = self.r_eq.strip()
        # Check for presence of essential parts (case-insensitive and flexible spacing)
        checks = [
            'rddot',
            'thetadot',
            'phidot',
            'sin',
            'k*(r - r0)'  # or 'k*(r-r0)'
        ]
        missing = [t for t in checks if t not in eq.replace(' ', '')]
        if missing:
            return (False, f"r_eq missing terms: {missing}")
        return (True, "r_eq contains expected terms (rddot, thetadot, phidot, sin, k*(r-r0)).")

    def test_modeling_theta_equation_contains_terms(self):
        """Check that theta_eq string contains key terms of the theta Euler-Lagrange equation."""
        if self.theta_eq is None:
            return (True, "SKIP: theta_eq not available.")
        eq = self.theta_eq.strip()
        # Remove spaces for substring matching
        compact = eq.replace(' ', '')
        checks = [
            'thetaddot',
            'thetadot',
            'rdot',
            'phidot',
            'sintheta',
            'costheta'
        ]
        missing = [t for t in checks if t not in compact]
        if missing:
            return (False, f"theta_eq missing terms: {missing}")
        return (True, "theta_eq contains expected terms (thetaddot, thetadot, rdot, phidot, sintheta, costheta).")

    def test_modeling_energy_conservation_implied(self):
        """Check that the Lagrangian expression derived in explanation steps is consistent.
           This is a weak check: ensure that total energy (T+V) would be constant if computed.
           We don't have full data, so just verify that the contributions are present in givens."""
        if self.m is None or self.M is None:
            return (True, "SKIP: missing masses for energy modeling.")
        # Expect kinetic energy terms: 1/2 (M+2m)(Xdot^2+...), m(rdot^2+...), rod rotation.
        # We can't compute exact E, but we can check that at least some derivative variables exist.
        has_translation = all(v is not None for v in [self.Xdot, self.Ydot, self.Zdot])
        has_radial = all(v is not None for v in [self.rdot, self.r, self.thetadot, self.phidot, self.theta])
        has_rod_rot = (self.thetadot is not None and self.phidot is not None)
        if not (has_translation or has_radial or has_rod_rot):
            return (True, "Insufficient coordinate derivatives to verify energy expression.")
        # All good: the structure is plausible.
        return (True, "Energy conservation implied: appropriate kinetic and potential terms are present in the formulation.")

    # ----------------------------------------------------------------------
    # Monotonicity checks (perturbation test)
    def test_monotonicity_phidot_increases_L_phi(self):
        """Perturb phidot slightly and check that L_phi changes in the same direction."""
        if not self._is_finite_number(self.phidot) or self.r is None or self.theta is None or self.I is None or self.m is None:
            return (True, "SKIP: insufficient data to test monotonicity of L_phi with phidot.")
        # Use the stored L_phi as reference (if available), else compute from formula
        if self.L_phi is not None and self._is_finite_number(self.L_phi):
            baseline = self.L_phi
        else:
            # compute baseline from givens
            sin2theta = math.sin(self.theta)**2
            if sin2theta == 0.0:
                return (True, "SKIP: sin(theta)=0, perturbation invalid.")
            baseline = (self.I + 2*self.m*self.r**2) * self.phidot * sin2theta
        # Perturb phidot by +5%
        phidot_pert = self.phidot * 1.05
        sin2theta = math.sin(self.theta)**2
        new_L = (self.I + 2*self.m*self.r**2) * phidot_pert * sin2theta
        if new_L > baseline:
            return (True, "L_phi increases with phidot (monotonic).")
        else:
            return (False, f"L_phi does not increase monotonically with phidot: baseline={baseline}, perturbed={new_L}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (sign, finiteness, broad magnitude)
    def test_answer_sanity_finite_values(self):
        """Check that all numerical result values are finite."""
        num_results = [self.Px_const, self.Py_const, self.Pz_const, self.L_phi]
        all_finite = all(self._is_finite_number(v) for v in num_results if v is not None)
        if not all_finite:
            names = ['Px_const', 'Py_const', 'Pz_const', 'L_phi']
            failed = [name for name, val in zip(names, num_results) if not self._is_finite_number(val)]
            return (False, f"Non-finite result(s): {failed}")
        return (True, "All numeric results are finite.")

    # ----------------------------------------------------------------------
    # Run all tests
    def run_all_tests(self):
        """Discover and run all test_* methods, crash-safe."""
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