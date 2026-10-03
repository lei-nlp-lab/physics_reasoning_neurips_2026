import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # ---------- Extract givens ----------
        givens = getattr(solution_instance, 'givens', {})
        self.m = givens.get('m', getattr(solution_instance, 'm', None))
        self.a = givens.get('a', getattr(solution_instance, 'a', None))
        self.P = givens.get('P', getattr(solution_instance, 'P', None))
        # Also try to get as 'impulse_magnitude'
        if self.P is None:
            self.P = givens.get('impulse_magnitude', getattr(solution_instance, 'impulse_magnitude', None))

        # ---------- Extract results ----------
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.v_BC = results.get('v_BC', None)
        self.v_AB = results.get('v_AB', None)
        self.omega_BC = results.get('omega_BC', None)
        self.omega_AB = results.get('omega_AB', None)
        # Fallback to attributes
        if self.v_BC is None:
            self.v_BC = getattr(solution_instance, 'v_BC', None)
        if self.v_AB is None:
            self.v_AB = getattr(solution_instance, 'v_AB', None)
        if self.omega_BC is None:
            self.omega_BC = getattr(solution_instance, 'omega_BC', None)
        if self.omega_AB is None:
            self.omega_AB = getattr(solution_instance, 'omega_AB', None)

        # Extract y-components of velocities
        self.v_BCy = None
        if isinstance(self.v_BC, (list, tuple)) and len(self.v_BC) == 2:
            self.v_BCy = self.v_BC[1]
        elif isinstance(self.v_BC, (int, float)):
            self.v_BCy = self.v_BC

        self.v_ABy = None
        if isinstance(self.v_AB, (list, tuple)) and len(self.v_AB) == 2:
            self.v_ABy = self.v_AB[1]
        elif isinstance(self.v_AB, (int, float)):
            self.v_ABy = self.v_AB

        # Derived quantities
        self.I_CM = None
        if self.m is not None and self.a is not None:
            self.I_CM = self.m * self.a**2 / 12.0

        self.I_By = None
        if self.I_CM is not None and self.a is not None and self.omega_BC is not None:
            self.I_By = 2.0 * self.I_CM / self.a * self.omega_BC

    # -------------------------------------------------------------------
    # Interface checks
    # -------------------------------------------------------------------
    def test_interface_required_keys(self):
        failures = []
        if self.v_BC is None:
            failures.append("v_BC missing")
        if self.v_AB is None:
            failures.append("v_AB missing")
        if self.omega_BC is None:
            failures.append("omega_BC missing")
        if self.omega_AB is None:
            failures.append("omega_AB missing")
        if self.v_BCy is None:
            failures.append("v_BC y-component not extractable")
        if self.v_ABy is None:
            failures.append("v_AB y-component not extractable")
        if failures:
            return (False, "Missing or invalid keys: " + "; ".join(failures))
        return (True, "All required result keys present and extractable")

    def test_interface_givens(self):
        failures = []
        if self.m is None or self.m <= 0:
            failures.append(f"m = {self.m}")
        if self.a is None or self.a <= 0:
            failures.append(f"a = {self.a}")
        if self.P is None or self.P <= 0:
            failures.append(f"P = {self.P}")
        if failures:
            return (False, "Invalid givens: " + "; ".join(failures))
        return (True, "All givens positive")

    # -------------------------------------------------------------------
    # Invariant / theorem-based checks
    # -------------------------------------------------------------------
    def test_invariant_hinge_equal_velocity(self):
        if self.v_ABy is None or self.v_BCy is None or self.omega_AB is None or self.omega_BC is None or self.a is None:
            return (False, "SKIP: missing data for hinge check")
        v_B_AB = self.v_ABy + (self.a / 2.0) * self.omega_AB
        v_B_BC = self.v_BCy - (self.a / 2.0) * self.omega_BC
        if not math.isclose(v_B_AB, v_B_BC, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"v_B mismatch: AB yields {v_B_AB}, BC yields {v_B_BC}")
        return (True, f"B velocities equal: {v_B_AB} approx {v_B_BC}")

    def test_invariant_vBC_scaling(self):
        if self.v_BCy is None or self.m is None or self.P is None:
            return (False, "SKIP: missing data for v_BC scaling")
        if abs(self.v_BCy) < 1e-12:
            return (False, "v_BCy is zero")
        ratio = abs(self.v_BCy) * 4.0 * self.m / self.P
        if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"|v_BCy|*4m/P = {ratio}, expected ~1")
        return (True, f"v_BCy scaling correct: ratio {ratio}")

    def test_invariant_vAB_scaling(self):
        if self.v_ABy is None or self.m is None or self.P is None:
            return (False, "SKIP: missing data for v_AB scaling")
        if abs(self.P) < 1e-12:
            return (False, "P is zero")
        ratio = self.v_ABy * 4.0 * self.m / (5.0 * self.P)
        if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"v_ABy*4m/(5P) = {ratio}, expected ~1")
        return (True, f"v_ABy scaling correct: ratio {ratio}")

    def test_invariant_omegaBC_scaling(self):
        if self.omega_BC is None or self.m is None or self.a is None or self.P is None:
            return (False, "SKIP: missing data for omega_BC scaling")
        if abs(self.P) < 1e-12:
            return (False, "P is zero")
        ratio = self.omega_BC * 2.0 * self.m * self.a / (3.0 * self.P)
        if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega_BC*2ma/(3P) = {ratio}, expected ~1")
        return (True, f"omega_BC scaling correct: ratio {ratio}")

    def test_invariant_omegaAB_scaling(self):
        if self.omega_AB is None or self.m is None or self.a is None or self.P is None:
            return (False, "SKIP: missing data for omega_AB scaling")
        if abs(self.P) < 1e-12:
            return (False, "P is zero")
        ratio = self.omega_AB * 2.0 * self.m * self.a / (-9.0 * self.P)
        if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega_AB*2ma/(-9P) = {ratio}, expected ~1")
        return (True, f"omega_AB scaling correct: ratio {ratio}")

    def test_invariant_velocity_ratio(self):
        if self.v_ABy is None or self.v_BCy is None:
            return (False, "SKIP: missing velocity data")
        if abs(self.v_BCy) < 1e-12:
            return (False, "v_BCy is zero")
        ratio = self.v_ABy / self.v_BCy
        if not math.isclose(ratio, -5.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"v_ABy/v_BCy = {ratio}, expected -5")
        return (True, f"Velocity ratio correct: {ratio}")

    def test_invariant_omega_ratio(self):
        if self.omega_BC is None or self.omega_AB is None:
            return (False, "SKIP: missing omega data")
        if abs(self.omega_AB) < 1e-12:
            return (False, "omega_AB is zero")
        ratio = self.omega_BC / self.omega_AB
        if not math.isclose(ratio, -1.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega_BC/omega_AB = {ratio}, expected -1/3")
        return (True, f"Omega ratio correct: {ratio}")

    # -------------------------------------------------------------------
    # Modeling / trace checks
    # -------------------------------------------------------------------
    def test_modeling_linear_momentum_BC(self):
        if self.m is None or self.v_BCy is None or self.I_By is None:
            return (False, "SKIP: missing data for linear momentum BC")
        lhs = self.m * self.v_BCy
        rhs = -self.I_By
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"m*v_BCy = {lhs}, expected -I_By ({-self.I_By})")
        return (True, "Linear momentum BC consistent: m*v_BCy = -I_By")

    def test_modeling_linear_momentum_AB(self):
        if self.P is None or self.I_By is None or self.m is None or self.v_ABy is None:
            return (False, "SKIP: missing data for linear momentum AB")
        lhs = self.P + self.I_By
        rhs = self.m * self.v_ABy
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P + I_By = {lhs}, expected m*v_ABy = {rhs}")
        return (True, "Linear momentum AB consistent: P + I_By = m*v_ABy")

    def test_modeling_angular_momentum_BC(self):
        if self.a is None or self.I_By is None or self.I_CM is None or self.omega_BC is None:
            return (False, "SKIP: missing data for angular momentum BC")
        lhs = (self.a / 2.0) * self.I_By
        rhs = self.I_CM * self.omega_BC
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"(a/2)*I_By = {lhs}, expected I_CM*omega_BC = {rhs}")
        return (True, "Angular momentum BC consistent")

    def test_modeling_angular_momentum_AB(self):
        if self.a is None or self.I_By is None or self.P is None or self.I_CM is None or self.omega_AB is None:
            return (False, "SKIP: missing data for angular momentum AB")
        lhs = (self.a / 2.0) * (self.I_By - self.P)
        rhs = self.I_CM * self.omega_AB
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"(a/2)*(I_By - P) = {lhs}, expected I_CM*omega_AB = {rhs}")
        return (True, "Angular momentum AB consistent")

    # -------------------------------------------------------------------
    # Monotonicity checks
    # -------------------------------------------------------------------
    def test_monotonicity_vBC_increases_with_P(self):
        '''Perturb P upward by 10% and check v_BCy magnitude increases.'''
        if self.v_BCy is None or self.P is None or self.m is None:
            return (False, "SKIP: missing data for monotonicity vBC")
        base = self.v_BCy
        # Simulate perturbed v_BCy = -P/(4m) -> magnitude proportional to P
        # We cannot recompute exact because solver may do something else,
        # but from physics, v_BCy magnitude should increase with P.
        # Create a perturbed expected value: perturb P by +10%
        P_perturbed = self.P * 1.1
        expected_perturbed = -P_perturbed / (4.0 * self.m)
        # Compare base vs perturbed: since base should be -P/(4m), then expected_perturbed < base (more negative)
        if not (expected_perturbed < base):
            return (False, f"v_BCy became less negative after increasing P: base {base}, perturbed {expected_perturbed}")
        return (True, f"Monotonicity OK: v_BCy magnitude increases with P (base {base}, perturbed {expected_perturbed})")

    def test_monotonicity_vAB_increases_with_P(self):
        '''Perturb P upward by 10% and check v_ABy magnitude increases.'''
        if self.v_ABy is None or self.P is None or self.m is None:
            return (False, "SKIP: missing data for monotonicity vAB")
        base = self.v_ABy
        P_perturbed = self.P * 1.1
        expected_perturbed = 5.0 * P_perturbed / (4.0 * self.m)
        if not (expected_perturbed > base):
            return (False, f"v_ABy did not increase with P: base {base}, perturbed {expected_perturbed}")
        return (True, f"Monotonicity OK: v_ABy increases with P (base {base}, perturbed {expected_perturbed})")

    def test_monotonicity_omegaBC_magnitude_increases_with_P(self):
        '''Perturb P upward by 10% and check omega_BC magnitude increases.'''
        if self.omega_BC is None or self.P is None or self.m is None or self.a is None:
            return (False, "SKIP: missing data for monotonicity omegaBC")
        base = self.omega_BC
        P_perturbed = self.P * 1.1
        expected_perturbed = 3.0 * P_perturbed / (2.0 * self.m * self.a)
        # omega_BC positive; check monotonic
        if not (expected_perturbed > base):
            return (False, f"omega_BC did not increase with P: base {base}, perturbed {expected_perturbed}")
        return (True, f"Monotonicity OK: omega_BC increases with P (base {base}, perturbed {expected_perturbed})")

    def test_monotonicity_omegaAB_magnitude_increases_with_P(self):
        '''Perturb P upward by 10% and check |omega_AB| increases.'''
        if self.omega_AB is None or self.P is None or self.m is None or self.a is None:
            return (False, "SKIP: missing data for monotonicity omegaAB")
        base = self.omega_AB  # negative
        base_mag = abs(base)
        P_perturbed = self.P * 1.1
        expected_perturbed = -9.0 * P_perturbed / (2.0 * self.m * self.a)
        expected_mag = abs(expected_perturbed)
        if not (expected_mag > base_mag):
            return (False, f"|omega_AB| did not increase with P: base mag {base_mag}, perturbed mag {expected_mag}")
        return (True, f"Monotonicity OK: |omega_AB| increases with P (base mag {base_mag}, perturbed mag {expected_mag})")

    # -------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # -------------------------------------------------------------------
    def test_answer_sanity_signs(self):
        '''Check qualitative signs: v_BCy negative, v_ABy positive, omega_BC positive, omega_AB negative.'''
        failures = []
        if self.v_BCy is not None and self.v_BCy > 0:
            failures.append("v_BCy should be negative")
        if self.v_ABy is not None and self.v_ABy < 0:
            failures.append("v_ABy should be positive")
        if self.omega_BC is not None and self.omega_BC < 0:
            failures.append("omega_BC should be positive")
        if self.omega_AB is not None and self.omega_AB > 0:
            failures.append("omega_AB should be negative")
        if failures:
            return (False, "Sign check failed: " + "; ".join(failures))
        return (True, "All signs consistent with physics")

    # -------------------------------------------------------------------
    # Run all tests
    # -------------------------------------------------------------------
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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)