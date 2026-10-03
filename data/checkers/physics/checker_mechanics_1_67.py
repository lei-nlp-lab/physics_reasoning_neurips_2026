import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.solution_instance = solution_instance
        self.m = getattr(solution_instance, 'm', None)
        self.k = getattr(solution_instance, 'k', None)
        self.r = getattr(solution_instance, 'r', None)

        # Extract results safely (default None)
        results = getattr(solution_instance, 'results', {})
        self.E = results.get('E', None)
        self.L = results.get('L', None)
        self.omega = results.get('omega', None)
        self.omega_r = results.get('omega_r', None)

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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        required = ["E", "L", "omega", "omega_r"]
        results = getattr(self.solution_instance, 'results', {})
        missing = [key for key in required if key not in results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_mass_positive(self):
        if self.m is None:
            return (False, "SKIP: mass not available")
        if self.m <= 0:
            return (False, f"Mass must be positive, got {self.m}")
        return (True, f"Mass = {self.m} > 0")

    def test_interface_k_positive(self):
        if self.k is None:
            return (False, "SKIP: k not available")
        if self.k <= 0:
            return (False, f"k must be positive, got {self.k}")
        return (True, f"k = {self.k} > 0")

    def test_interface_r_positive(self):
        if self.r is None:
            return (False, "SKIP: r not available")
        if self.r <= 0:
            return (False, f"r must be positive, got {self.r}")
        return (True, f"r = {self.r} > 0")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose with 1e-2 tolerance)
    # ------------------------------------------------------------------
    def test_invariant_energy(self):
        if any(x is None for x in [self.E, self.k, self.r]):
            return (False, "SKIP: missing E, k, or r")
        expected = 1.5 * self.k * self.r
        if math.isclose(self.E, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E = {self.E} close to expected {expected}")
        return (False, f"E = {self.E}, expected ~{expected}")

    def test_invariant_angular_momentum(self):
        if any(x is None for x in [self.L, self.m, self.k, self.r]):
            return (False, "SKIP: missing L, m, k, or r")
        expected = math.sqrt(self.m * self.k * self.r**3)
        if math.isclose(self.L, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L = {self.L} close to expected {expected}")
        return (False, f"L = {self.L}, expected ~{expected}")

    def test_invariant_angular_frequency(self):
        if any(x is None for x in [self.omega, self.k, self.m, self.r]):
            return (False, "SKIP: missing omega, k, m, or r")
        expected = math.sqrt(self.k / (self.m * self.r))
        if math.isclose(self.omega, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"omega = {self.omega} close to expected {expected}")
        return (False, f"omega = {self.omega}, expected ~{expected}")

    def test_invariant_radial_frequency_ratio(self):
        if any(x is None for x in [self.omega_r, self.omega]):
            return (False, "SKIP: missing omega_r or omega")
        expected = math.sqrt(3) * self.omega
        if math.isclose(self.omega_r, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"omega_r = {self.omega_r} close to expected sqrt(3)*omega = {expected}")
        return (False, f"omega_r = {self.omega_r}, expected ~{expected}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (check consistency of the physics model)
    # ------------------------------------------------------------------
    def test_modeling_effective_potential_consistency(self):
        # E should equal U_eff = L²/(2 m r²) + k r
        if any(x is None for x in [self.E, self.L, self.m, self.k, self.r]):
            return (False, "SKIP: missing E, L, m, k, or r")
        U_eff = self.L**2 / (2 * self.m * self.r**2) + self.k * self.r
        if math.isclose(self.E, U_eff, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E = {self.E} equals U_eff = {U_eff}")
        return (False, f"E = {self.E}, but U_eff = {U_eff}")

    def test_modeling_circular_condition_derivative(self):
        # dU_eff/dr = -L²/(m r³) + k should be ~0
        if any(x is None for x in [self.L, self.m, self.k, self.r]):
            return (False, "SKIP: missing L, m, k, or r")
        deriv = -self.L**2 / (self.m * self.r**3) + self.k
        if math.isclose(deriv, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dU_eff/dr = {deriv} is close to zero")
        return (False, f"dU_eff/dr = {deriv}, expected ~0")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb one input and check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_omega_with_mass(self):
        # omega ∝ 1/√m : increasing m decreases omega
        if any(x is None for x in [self.omega, self.m, self.k, self.r]):
            return (False, "SKIP: missing omega, m, k, or r")
        m_pert = self.m * 1.01
        omega_expected = math.sqrt(self.k / (m_pert * self.r))
        # ratio = sqrt(m/m_pert) < 1
        if omega_expected < self.omega:
            return (True, f"omega decreases when m increases: {omega_expected} < {self.omega}")
        return (False, f"omega did not decrease; perturbed omega = {omega_expected} >= {self.omega}")

    def test_monotonicity_L_with_mass(self):
        # L ∝ √m : increasing m increases L
        if any(x is None for x in [self.L, self.m, self.k, self.r]):
            return (False, "SKIP: missing L, m, k, or r")
        m_pert = self.m * 1.01
        L_expected = math.sqrt(m_pert * self.k * self.r**3)
        if L_expected > self.L:
            return (True, f"L increases when m increases: {L_expected} > {self.L}")
        return (False, f"L did not increase; perturbed L = {L_expected} <= {self.L}")

    def test_monotonicity_omega_with_k(self):
        # omega ∝ √k : increasing k increases omega
        if any(x is None for x in [self.omega, self.m, self.k, self.r]):
            return (False, "SKIP: missing omega, m, k, or r")
        k_pert = self.k * 1.01
        omega_expected = math.sqrt(k_pert / (self.m * self.r))
        if omega_expected > self.omega:
            return (True, f"omega increases when k increases: {omega_expected} > {self.omega}")
        return (False, f"omega did not increase; perturbed omega = {omega_expected} <= {self.omega}")

    def test_monotonicity_L_with_k(self):
        # L ∝ √k : increasing k increases L
        if any(x is None for x in [self.L, self.m, self.k, self.r]):
            return (False, "SKIP: missing L, m, k, or r")
        k_pert = self.k * 1.01
        L_expected = math.sqrt(self.m * k_pert * self.r**3)
        if L_expected > self.L:
            return (True, f"L increases when k increases: {L_expected} > {self.L}")
        return (False, f"L did not increase; perturbed L = {L_expected} <= {self.L}")

    def test_monotonicity_omega_with_r(self):
        # omega ∝ 1/√r : increasing r decreases omega
        if any(x is None for x in [self.omega, self.m, self.k, self.r]):
            return (False, "SKIP: missing omega, m, k, or r")
        r_pert = self.r * 1.01
        omega_expected = math.sqrt(self.k / (self.m * r_pert))
        if omega_expected < self.omega:
            return (True, f"omega decreases when r increases: {omega_expected} < {self.omega}")
        return (False, f"omega did not decrease; perturbed omega = {omega_expected} >= {self.omega}")

    def test_monotonicity_L_with_r(self):
        # L ∝ r^{3/2} : increasing r increases L
        if any(x is None for x in [self.L, self.m, self.k, self.r]):
            return (False, "SKIP: missing L, m, k, or r")
        r_pert = self.r * 1.01
        L_expected = math.sqrt(self.m * self.k * r_pert**3)
        if L_expected > self.L:
            return (True, f"L increases when r increases: {L_expected} > {self.L}")
        return (False, f"L did not increase; perturbed L = {L_expected} <= {self.L}")

    def test_monotonicity_omega_r_with_omega(self):
        # omega_r = sqrt(3)*omega, so if omega increases, omega_r increases
        if any(x is None for x in [self.omega_r, self.omega]):
            return (False, "SKIP: missing omega_r or omega")
        # Perturb omega by 5% (simulate)
        omega_pert = self.omega * 1.05
        expected_omega_r = math.sqrt(3) * omega_pert
        if expected_omega_r > self.omega_r:
            return (True, f"omega_r increases with omega: {expected_omega_r} > {self.omega_r}")
        return (False, f"omega_r did not increase with omega; perturbed omega_r = {expected_omega_r} <= {self.omega_r}")

    # ------------------------------------------------------------------
    # Weak answer sanity (no re-derivation, simple checks)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_values(self):
        # All results should be finite and positive (or sensible)
        results = getattr(self.solution_instance, 'results', {})
        for key in ['E', 'L', 'omega', 'omega_r']:
            val = results.get(key, None)
            if val is None:
                return (False, f"Missing result '{key}'")
            if not isinstance(val, (int, float)):
                return (False, f"Result '{key}' is not numeric")
            if math.isnan(val) or math.isinf(val):
                return (False, f"Result '{key}' is NaN or Inf")
            if val < 0:
                return (False, f"Result '{key}' is negative ({val})")
        return (True, "All results finite and non-negative")