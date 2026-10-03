import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results (expected keys from [emit] contract)
        self.results = getattr(solution_instance, 'results', {})
        self.E = self.results.get('E', None)
        self.B_required = self.results.get('B_required', None)

        # Extract givens (safe defaults: None)
        self.m = getattr(solution_instance, 'm', None)          # in MeV/c^2
        self.tau = getattr(solution_instance, 'tau', None)      # seconds
        self.R = getattr(solution_instance, 'R', None)          # meters
        self.c = getattr(solution_instance, 'c', None)          # m/s
        self.q = getattr(solution_instance, 'q', None)          # Coulombs
        self.B_earth = getattr(solution_instance, 'B_earth', None)  # Tesla

        # Derive useful constants if givens are available
        if all(x is not None for x in [self.R, self.c, self.tau]):
            self.K = 2 * math.pi * self.R / (self.c * self.tau)   # dimensionless constant
        else:
            self.K = None

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_results(self):
        """Check that results['E'] and results['B_required'] exist and are numeric."""
        if self.E is None:
            return False, "Missing key 'E' in results."
        if self.B_required is None:
            return False, "Missing key 'B_required' in results."
        if not isinstance(self.E, (int, float)):
            return False, f"'E' is not a number, got {type(self.E)}"
        if not isinstance(self.B_required, (int, float)):
            return False, f"'B_required' is not a number, got {type(self.B_required)}"
        return True, "Both required result keys present and numeric."

    def test_interface_givens_present(self):
        """Check that all required physical givens are present and positive."""
        givens = {'m': self.m, 'tau': self.tau, 'R': self.R,
                  'c': self.c, 'q': self.q, 'B_earth': self.B_earth}
        missing = [name for name, val in givens.items() if val is None]
        if missing:
            return False, f"Missing givens: {missing}"
        for name, val in givens.items():
            if val <= 0:
                return False, f"Given '{name}' is not positive (value {val})"
        return True, "All required givens present and positive."

    # ----------------------------------------------------------------------
    # Theorem-based invariant checks
    # ----------------------------------------------------------------------
    def test_invariant_time_dilation_constraint(self):
        """Check gamma * beta * c * tau ≈ 2πR."""
        if any(x is None for x in [self.E, self.m, self.c, self.tau, self.R]):
            return False, "SKIP: Missing E, m, c, tau, or R"

        gamma = self.E / (self.m * self.c**2)
        if gamma < 1:
            return False, f"gamma = {gamma} < 1, impossible"
        beta = math.sqrt(1 - 1/gamma**2)
        lhs = gamma * beta * self.c * self.tau
        rhs = 2 * math.pi * self.R
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"gamma*beta*c*tau ≈ {lhs:.2e}, 2πR = {rhs:.2e}"
        else:
            return False, f"gamma*beta*c*tau = {lhs:.2e}, 2πR = {rhs:.2e}, difference {abs(lhs-rhs):.2e}"

    def test_invariant_energy_momentum_relation(self):
        """Check E² ≈ (pc)² + (mc²)², where pc = 2πR m c / τ."""
        if any(x is None for x in [self.E, self.m, self.c, self.tau, self.R]):
            return False, "SKIP: Missing E, m, c, tau, or R"

        pc = (2 * math.pi * self.R * self.m * self.c) / self.tau
        mc2 = self.m * self.c**2
        E_sq = self.E**2
        expected_sq = pc**2 + mc2**2
        if math.isclose(E_sq, expected_sq, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E² ≈ {E_sq:.2e}, (pc)²+(mc²)² ≈ {expected_sq:.2e}"
        else:
            return False, f"E² = {E_sq:.2e}, (pc)²+(mc²)² = {expected_sq:.2e}, difference {abs(E_sq-expected_sq):.2e}"

    def test_invariant_lorentz_force_balance(self):
        """Check that q * B_required * R ≈ γ m v = m * (2πR/τ)."""
        if any(x is None for x in [self.B_required, self.m, self.tau, self.q, self.R]):
            return False, "SKIP: Missing B_required, m, tau, q, or R"

        gamma_m_v = self.m * (2 * math.pi * self.R / self.tau)   # since gamma*v = 2πR/τ
        lhs = self.q * self.B_required * self.R
        if math.isclose(lhs, gamma_m_v, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"q B R ≈ {lhs:.2e}, γ m v = {gamma_m_v:.2e}"
        else:
            return False, f"q B R = {lhs:.2e}, γ m v = {gamma_m_v:.2e}, difference {abs(lhs-gamma_m_v):.2e}"

    def test_invariant_earth_field_insufficient(self):
        """Check that B_required > B_earth (Earth's field too weak)."""
        if any(x is None for x in [self.B_required, self.B_earth]):
            return False, "SKIP: Missing B_required or B_earth"
        if self.B_required > self.B_earth:
            return True, f"B_required = {self.B_required:.3e} T > B_earth = {self.B_earth:.3e} T"
        else:
            return False, f"B_required = {self.B_required:.3e} T NOT > B_earth = {self.B_earth:.3e} T"

    # ----------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_energy_formula(self):
        """Check that E matches the formula E = m c² sqrt(1 + (2πR/(τ c))²)."""
        if any(x is None for x in [self.E, self.m, self.c, self.tau, self.R]):
            return False, "SKIP: Missing E, m, c, tau, or R"

        gamma_expected = math.sqrt(1 + self.K**2)
        E_expected = gamma_expected * self.m * self.c**2
        if math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E ≈ {E_expected:.3e} (expected)"
        else:
            return False, f"E = {self.E:.3e}, expected {E_expected:.3e}, difference {abs(self.E-E_expected):.3e}"

    def test_modeling_required_B_formula(self):
        """Check B_required matches formula B = 2π m / (τ q)."""
        if any(x is None for x in [self.B_required, self.m, self.tau, self.q]):
            return False, "SKIP: Missing B_required, m, tau, or q"

        B_formula = (2 * math.pi * self.m) / (self.tau * self.q)
        if math.isclose(self.B_required, B_formula, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"B_required ≈ {B_formula:.3e} (expected)"
        else:
            return False, f"B_required = {self.B_required:.3e}, expected {B_formula:.3e}, difference {abs(self.B_required-B_formula):.3e}"

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_energy_increases_with_R(self):
        """Check that the derived energy increases when R is increased (by 10%)."""
        if any(x is None for x in [self.E, self.m, self.c, self.tau, self.R]):
            return False, "SKIP: Missing E, m, c, tau, or R"

        # Compute energy for base R
        gamma_base = math.sqrt(1 + (2*math.pi*self.R/(self.c*self.tau))**2)
        E_base = gamma_base * self.m * self.c**2

        # Compute energy for 1.1 * R
        R_big = 1.1 * self.R
        gamma_big = math.sqrt(1 + (2*math.pi*R_big/(self.c*self.tau))**2)
        E_big = gamma_big * self.m * self.c**2

        if E_big > E_base:
            return True, f"Energy increases with R: E({R_big:.2e}) = {E_big:.3e} > E({self.R:.2e}) = {E_base:.3e}"
        else:
            return False, f"Energy did not increase: E({R_big:.2e}) = {E_big:.3e} <= E({self.R:.2e}) = {E_base:.3e}"

    def test_monotonicity_B_decreases_with_tau(self):
        """Check that the required B decreases when tau is increased (by 10%)."""
        if any(x is None for x in [self.B_required, self.m, self.tau, self.q]):
            return False, "SKIP: Missing B_required, m, tau, or q"

        B_base = (2*math.pi*self.m)/(self.tau*self.q)
        tau_big = 1.1 * self.tau
        B_big = (2*math.pi*self.m)/(tau_big*self.q)
        if B_big < B_base:
            return True, f"B decreases with tau: B({tau_big:.2e}) = {B_big:.3e} < B({self.tau:.2e}) = {B_base:.3e}"
        else:
            return False, f"B did not decrease: B({tau_big:.2e}) = {B_big:.3e} >= B({self.tau:.2e}) = {B_base:.3e}"

    # ----------------------------------------------------------------------
    # Weak answer sanity (exactly one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Weak sanity: E and B_required are positive, finite, and in plausible orders of magnitude."""
        checks = []
        if self.E is None or self.B_required is None:
            return False, "SKIP: Missing E or B_required"

        # Energy
        if not math.isfinite(self.E):
            return False, "E is not finite"
        if self.E <= 0:
            return False, "E is not positive"
        if self.E < 100:   # mc^2 = 100 MeV, so E must be at least that
            return False, f"E = {self.E:.2f} MeV < 100 MeV, impossible for a muon"
        if self.E > 1e12:  # arbitrary very high
            checks.append("E is suspiciously large (>1e12 MeV)")

        # Magnetic field
        if not math.isfinite(self.B_required):
            return False, "B_required is not finite"
        if self.B_required <= 0:
            return False, "B_required is not positive"
        if self.B_required > 10:   # expected ~0.0035 T
            checks.append("B_required suspiciously large (>10 T)")

        if checks:
            return True, "Answer sanity passed with minor notes: " + "; ".join(checks)
        return True, "Answer sanity passed: both values positive, finite, and plausible."

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all bound methods starting with "test_"
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