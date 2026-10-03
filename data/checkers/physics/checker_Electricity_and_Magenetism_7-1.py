import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs and results safely with None defaults
        self.m1 = getattr(solution_instance, 'm1', None)
        self.m2 = getattr(solution_instance, 'm2', None)
        self.r = getattr(solution_instance, 'r', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.theta = getattr(solution_instance, 'theta', None)
        
        results = getattr(solution_instance, 'results', {})
        # Result keys as specified in pseudocode [emit] contract
        self.F = results.get('F', None)
        self.tau = results.get('tau', None)
        self.theta_eq = results.get('theta_eq', None)

    # ------------------- Interface Tests -------------------
    def test_interface_required_keys(self):
        missing = []
        for key in ['F', 'tau', 'theta_eq']:
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_values_not_none(self):
        if self.F is None or self.tau is None or self.theta_eq is None:
            return (False, "One or more result values are None")
        return (True, "All result values are non-None")

    # ------------------- Invariant Tests (physics sanity) -------------------
    def test_invariant_force_magnitude(self):
        if any(v is None for v in [self.m1, self.m2, self.r, self.mu0, self.F]):
            return (False, "SKIP: missing givens or force result")
        # Compute expected magnitude: F_mag = 3*mu0*m1*m2/(2*pi*r^4)
        expected_mag = (3 * self.mu0 * self.m1 * self.m2) / (2 * math.pi * self.r**4)
        # Extract magnitude from force vector (assumed to be a tuple/list/array)
        if isinstance(self.F, (tuple, list)):
            actual_mag = math.sqrt(sum(v**2 for v in self.F))
        else:
            return (False, "Unexpected type for force; expected vector")
        if math.isclose(actual_mag, expected_mag, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Force magnitude matches: {actual_mag} ~ {expected_mag}")
        return (False, f"Force magnitude mismatch: {actual_mag} != {expected_mag} (tol 1%)")

    def test_invariant_force_sign_attractive(self):
        if self.F is None:
            return (False, "SKIP: force result missing")
        # For aligned dipoles, force should be negative (attractive) in z-direction
        if isinstance(self.F, (tuple, list)):
            # Assume z-component is last (or only component)
            z_comp = self.F[-1]  # typical: (0,0,Fz) or (Fz,) etc.
            if z_comp < 0:
                return (True, "Force is attractive (negative z-component)")
            return (False, "Force is not attractive: z-component >= 0")
        return (False, "Unexpected force type")

    def test_invariant_torque_formula(self):
        if any(v is None for v in [self.m1, self.m2, self.r, self.mu0, self.theta, self.tau]):
            return (False, "SKIP: missing givens or torque result")
        # Expected torque magnitude: tau_mag = mu0*m1*m2*|sinθ|/(2*pi*r^3)
        expected_mag = (self.mu0 * self.m1 * self.m2 * abs(math.sin(self.theta))) / (2 * math.pi * self.r**3)
        if isinstance(self.tau, (tuple, list)):
            actual_mag = math.sqrt(sum(v**2 for v in self.tau))
        else:
            return (False, "Unexpected type for torque; expected vector")
        if math.isclose(actual_mag, expected_mag, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Torque magnitude matches: {actual_mag} ~ {expected_mag}")
        return (False, f"Torque magnitude mismatch: {actual_mag} != {expected_mag}")

    def test_invariant_torque_direction_orthogonal(self):
        if any(v is None for v in [self.tau, self.theta]):
            return (False, "SKIP: missing data")
        # B is along zhat (if m1 along z)
        # m2 = m2(x comp, 0, z comp) with x comp = m2*sinθ, z comp = m2*cosθ
        # torque should be along -yhat
        if isinstance(self.tau, (tuple, list)):
            # expect y-component negative, x and z zero within tolerance
            tolerance = 1e-2
            if abs(self.tau[0]) < tolerance and abs(self.tau[2]) < tolerance and self.tau[1] < -tolerance:
                return (True, "Torque direction consistent: along -y")
            return (False, f"Torque direction inconsistent: {self.tau}")
        return (False, "Unexpected torque type")

    def test_invariant_theta_eq_zero(self):
        if self.theta_eq is None:
            return (False, "SKIP: theta_eq missing")
        if abs(self.theta_eq - 0) < 1e-6:
            return (True, "Equilibrium orientation is zero (aligned)")
        return (False, f"Expected theta_eq=0, got {self.theta_eq}")

    # ------------------- Modeling / Trace Tests -------------------
    def test_modeling_force_direction_aligned(self):
        if self.F is None:
            return (False, "SKIP: force missing")
        # For aligned dipoles, force should be along -z (0,0,negative)
        if isinstance(self.F, (tuple, list)):
            # Check that only z-component is non-zero
            non_zero = [i for i, v in enumerate(self.F) if abs(v) > 1e-6]
            if len(non_zero) == 1 and non_zero[-1] == len(self.F)-1:
                return (True, "Force is along z-axis")
            return (False, "Force has components off z-axis")
        return (False, "Unexpected force type")

    def test_modeling_torque_from_cross_product(self):
        # Explanation step: tau = m2 x B
        if any(v is None for v in [self.theta, self.m1, self.m2, self.r, self.mu0, self.tau]):
            return (False, "SKIP: missing data")
        # Recompute expected torque vector
        B_z = (self.mu0 * self.m1) / (2 * math.pi * self.r**3)  # B along z
        m2_x = self.m2 * math.sin(self.theta)
        m2_z = self.m2 * math.cos(self.theta)
        # m2 = (m2_x, 0, m2_z), B = (0,0,B_z)
        # tau = m2 x B = (m2_y*B_z - m2_z*0, m2_z*0 - m2_x*B_z, m2_x*0 - m2_y*0) = (0, -m2_x*B_z, 0)
        expected_tau = (0.0, - m2_x * B_z, 0.0)
        if isinstance(self.tau, (tuple, list)):
            if all(math.isclose(a, e, rel_tol=1e-2, abs_tol=1e-2) for a, e in zip(self.tau, expected_tau)):
                return (True, "Torque matches cross product")
            return (False, f"Torque does not match cross product: {self.tau} vs {expected_tau}")
        return (False, "Unexpected torque type")

    def test_modeling_energy_minimum(self):
        # Potential energy U = -mu0*m1*m2*cosθ/(2*pi*r^3) stable at θ=0
        if any(v is None for v in [self.m1, self.m2, self.r, self.mu0, self.theta_eq]):
            return (False, "SKIP: missing")
        if abs(self.theta_eq - 0) < 1e-6:
            return (True, "Equilibrium at θ=0 corresponds to energy minimum")
        return (False, "theta_eq not zero")

    # ------------------- Monotonicity Tests -------------------
    def test_monotonicity_force_decreases_with_r(self):
        if any(v is None for v in [self.m1, self.m2, self.r, self.mu0, self.F]):
            return (False, "SKIP: missing data")
        # Original magnitude
        if isinstance(self.F, (tuple, list)):
            orig_mag = math.sqrt(sum(v**2 for v in self.F))
        else:
            return (False, "Force not vector")
        # Perturb r by +10%
        r_pert = self.r * 1.1
        if r_pert <= 0:
            return (False, "Invalid perturbation")
        expected_new_mag = (3 * self.mu0 * self.m1 * self.m2) / (2 * math.pi * r_pert**4)
        if expected_new_mag < orig_mag:
            return (True, f"Force magnitude decreases when r increases: {expected_new_mag} < {orig_mag}")
        return (False, "Monotonicity violated: force does not decrease with r")

    def test_monotonicity_torque_increases_with_sin_theta(self):
        if any(v is None for v in [self.theta, self.m1, self.m2, self.r, self.mu0, self.tau]):
            return (False, "SKIP: missing data")
        # Original torque magnitude
        if isinstance(self.tau, (tuple, list)):
            orig_mag = math.sqrt(sum(v**2 for v in self.tau))
        else:
            return (False, "Torque not vector")
        # Perturb theta by +0.1 rad (if not near saturation)
        theta_pert = self.theta + 0.1
        if theta_pert > math.pi:
            return (False, "SKIP: theta near pi, cannot perturb upward")
        expected_new_mag = (self.mu0 * self.m1 * self.m2 * abs(math.sin(theta_pert))) / (2 * math.pi * self.r**3)
        if expected_new_mag > orig_mag:
            return (True, f"Torque magnitude increases with sinθ: {expected_new_mag} > {orig_mag}")
        return (False, "Monotonicity violated: torque does not increase with sinθ")

    def test_monotonicity_force_increases_with_m1_m2(self):
        if any(v is None for v in [self.m1, self.m2, self.r, self.mu0, self.F]):
            return (False, "SKIP: missing")
        if isinstance(self.F, (tuple, list)):
            orig_mag = math.sqrt(sum(v**2 for v in self.F))
        else:
            return (False, "Force not vector")
        # Double m1
        new_m1 = 2 * self.m1
        expected_new_mag = (3 * self.mu0 * new_m1 * self.m2) / (2 * math.pi * self.r**4)
        if expected_new_mag > orig_mag:
            return (True, f"Force magnitude increases with m1: {expected_new_mag} > {orig_mag}")
        return (False, "Monotonicity violated: force does not increase with m1")

    # ------------------- Single Weak Answer Sanity Test -------------------
    def test_answer_sanity_checks(self):
        # Combined weak sanity: finiteness and range checks, no re-derivation.
        problems = []
        # Check F
        if self.F is not None:
            if isinstance(self.F, (tuple, list)):
                if any(math.isinf(v) or math.isnan(v) for v in self.F):
                    problems.append("Force contains inf or NaN")
        else:
            problems.append("Force is None")
        # Check tau
        if self.tau is not None:
            if isinstance(self.tau, (tuple, list)):
                if any(math.isinf(v) or math.isnan(v) for v in self.tau):
                    problems.append("Torque contains inf or NaN")
        else:
            problems.append("Torque is None")
        # Check theta_eq
        if self.theta_eq is not None:
            if self.theta_eq < 0 or self.theta_eq > math.pi:
                problems.append(f"theta_eq {self.theta_eq} out of range [0, pi]")
        else:
            problems.append("theta_eq is None")
        if problems:
            return (False, "; ".join(problems))
        return (True, "All answer sanity checks passed (finite, in range)")

    # ------------------- Main Runner -------------------
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)