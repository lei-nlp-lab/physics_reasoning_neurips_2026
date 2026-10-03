import math

class PhysicsChecker:
    """
    Auditor for the two-masses-spring-charges-in-E-field problem.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (use None for missing)
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.K = getattr(solution_instance, 'K', None)
        self.E0 = getattr(solution_instance, 'E0', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        # ke might be directly given or computed from epsilon0
        self.ke = getattr(solution_instance, 'ke', None)
        if self.ke is None and self.epsilon0 is not None:
            self.ke = 1.0 / (4.0 * math.pi * self.epsilon0)
        # mu is m/2 if m available
        self.mu = None if self.m is None else self.m / 2.0

        # Extract results safely (use None for missing)
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.d = results.get('d', None)
        self.a_cm = results.get('a_cm', None)
        self.omega = results.get('omega', None)
        self.x_sum = results.get('x_sum', None)
        self.y_sum = results.get('y_sum', None)
        self.F = results.get('F', None)
        self.H = results.get('H', None)
        self.t_integral = results.get('t_integral', None)
        self.theta_solution = results.get('theta_solution', None)

        # Constants for later checks
        self.pi = math.pi
        self._tol = 1e-2  # 1% relative tolerance for physics invariants

    # ------------------------------------------------------------------
    # Interface tests: required keys exist and are not None
    # ------------------------------------------------------------------
    def test_interface_required_results_keys(self):
        """All expected result keys must be present (not None)."""
        keys = ['d', 'a_cm', 'omega', 'x_sum', 'y_sum', 'F', 'H', 't_integral', 'theta_solution']
        missing = []
        for k in keys:
            if getattr(self, k) is None:
                missing.append(k)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_givens_available(self):
        """Essential givens (m, q, K, E0, epsilon0) must exist."""
        missing = []
        if self.m is None:
            missing.append('m')
        if self.q is None:
            missing.append('q')
        if self.K is None:
            missing.append('K')
        if self.E0 is None:
            missing.append('E0')
        if self.epsilon0 is None:
            missing.append('epsilon0')
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All essential givens present")

    # ------------------------------------------------------------------
    # Invariant tests: theorem‐based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_center_of_mass_acceleration(self):
        """CM acceleration from Newton's second law: a_cm = qE0 / m."""
        if any(v is None for v in [self.m, self.q, self.E0, self.a_cm]):
            return (False, "SKIP: missing required givens or a_cm")
        expected = self.q * self.E0 / self.m
        if math.isclose(self.a_cm, expected, rel_tol=self._tol, abs_tol=self._tol):
            return (True, f"a_cm = {self.a_cm} ≈ {expected}")
        else:
            return (False, f"a_cm mismatch: got {self.a_cm}, expected ~{expected}")

    def test_invariant_equilibrium_distance(self):
        """Balance of spring and Coulomb forces: d^3 = ke*q^2 / K."""
        if any(v is None for v in [self.d, self.ke, self.q, self.K]):
            return (False, "SKIP: missing d, ke, q, or K")
        d_cubed = self.d ** 3
        expected = self.ke * self.q**2 / self.K
        if math.isclose(d_cubed, expected, rel_tol=self._tol, abs_tol=self._tol):
            return (True, f"d^3 = {d_cubed} ≈ {expected}")
        else:
            return (False, f"d^3 mismatch: got {d_cubed}, expected ~{expected}")

    def test_invariant_small_oscillation_frequency(self):
        """Linearized frequency: omega = sqrt(6K / m)."""
        if any(v is None for v in [self.omega, self.K, self.m]):
            return (False, "SKIP: missing omega, K, or m")
        expected = math.sqrt(6.0 * self.K / self.m)
        if math.isclose(self.omega, expected, rel_tol=self._tol, abs_tol=self._tol):
            return (True, f"omega = {self.omega} ≈ {expected}")
        else:
            return (False, f"omega mismatch: got {self.omega}, expected ~{expected}")

    # ------------------------------------------------------------------
    # Modeling/trace checks (structural consistency with pseudocode)
    # ------------------------------------------------------------------
    def test_modeling_center_of_mass_x_string(self):
        """x_sum must contain at least 'qE0/m', '^2', 'C1', 'C2'."""
        s = self.x_sum
        if s is None:
            return (False, "SKIP: x_sum is None")
        ok = ('qE0/m' in s) and ('t^2' in s) and ('C1' in s) and ('C2' in s)
        if ok:
            return (True, "x_sum string structure plausible")
        else:
            return (False, f"x_sum '{s}' missing expected tokens")

    def test_modeling_center_of_mass_y_string(self):
        """y_sum must contain 'D1', 'D2', 't'."""
        s = self.y_sum
        if s is None:
            return (False, "SKIP: y_sum is None")
        ok = ('D1' in s) and ('D2' in s) and ('t' in s)
        if ok:
            return (True, "y_sum string structure plausible")
        else:
            return (False, f"y_sum '{s}' missing expected tokens")

    def test_modeling_radial_integral_string(self):
        """t_integral must contain 'integral dr', 'F', '(2K/m)*r^2', 'q^2/(piepsilon0 m r)', 'H^2/r^2'."""
        s = self.t_integral
        if s is None:
            return (False, "SKIP: t_integral is None")
        tokens = ['integral dr', 'F', '(2K/m)*r^2', 'q^2/(piepsilon0 m r)', 'H^2/r^2']
        # replace the tokens with flexible whitespace? Use simple in.
        ok = all(tok in s for tok in tokens)
        if ok:
            return (True, "t_integral string contains expected terms")
        else:
            missing = [tok for tok in tokens if tok not in s]
            return (False, f"t_integral missing tokens: {missing}")

    def test_modeling_angular_solution_string(self):
        """theta_solution must contain 'Ht + V', 'integral r^2 dtheta'."""
        s = self.theta_solution
        if s is None:
            return (False, "SKIP: theta_solution is None")
        tokens = ['Ht + V', 'integral r^2 dtheta']
        ok = ('Ht + V' in s) and ('integral r^2 dtheta' in s)
        if ok:
            return (True, "theta_solution structure plausible")
        else:
            return (False, f"theta_solution '{s}' missing expected tokens")

    # ------------------------------------------------------------------
    # Monotonicity: perturb an input and compare the resulting d
    # ------------------------------------------------------------------
    def test_monotonicity_d_charge_increase(self):
        """If q is increased, equilibrium distance d must increase."""
        if any(v is None for v in [self.d, self.q, self.K, self.ke]):
            return (False, "SKIP: missing required values for monotonicity")
        # Increase q by 10%
        q_pert = self.q * 1.1
        d_pert = (self.ke * q_pert**2 / self.K) ** (1.0/3.0)
        # d_pert must be greater than original d (within numerical tolerance)
        if d_pert > self.d - 1e-12:
            return (True, f"d_pert={d_pert:.6f} > d={self.d:.6f}, as expected")
        else:
            return (False, f"d_pert={d_pert:.6f} is not > d={self.d:.6f}")

    def test_monotonicity_d_spring_increase(self):
        """If K is increased, equilibrium distance d must decrease."""
        if any(v is None for v in [self.d, self.q, self.K, self.ke]):
            return (False, "SKIP: missing required values for monotonicity")
        # Increase K by 10%
        K_pert = self.K * 1.1
        d_pert = (self.ke * self.q**2 / K_pert) ** (1.0/3.0)
        # d_pert must be smaller than original d
        if d_pert < self.d + 1e-12:
            return (True, f"d_pert={d_pert:.6f} < d={self.d:.6f}, as expected")
        else:
            return (False, f"d_pert={d_pert:.6f} is not < d={self.d:.6f}")

    # ------------------------------------------------------------------
    # Answer sanity: weak checks (sign, finiteness, non-NaN)
    # ------------------------------------------------------------------
    def test_answer_sanity_d_positive(self):
        """d must be positive."""
        if self.d is None:
            return (False, "SKIP: d not available")
        if self.d > 0:
            return (True, "d > 0")
        else:
            return (False, f"d = {self.d} is not positive")

    # ------------------------------------------------------------------
    # Runner: discover and run all test_* methods
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all bound methods starting with 'test_'
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