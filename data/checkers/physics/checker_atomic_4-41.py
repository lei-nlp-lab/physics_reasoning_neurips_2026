import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the J/psi decay problem.
    Checks interface, invariants, modeling traces, monotonicity, and sanity.
    """

    def __init__(self, solution_instance):
        # Extract given quantities (use None for missing)
        self.m_J = getattr(solution_instance, 'm_J', None)
        self.p_J = getattr(solution_instance, 'p_J', None)
        self.Gamma = getattr(solution_instance, 'Gamma', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.c = getattr(solution_instance, 'c', None)
        self.m_e = getattr(solution_instance, 'm_e', None)  # no numeric default

        # Extract results (expected keys: L, E_e, theta_deg)
        results = getattr(solution_instance, 'results', {})
        self.L = results.get('L', None)
        self.E_e = results.get('E_e', None)
        self.theta_deg = results.get('theta_deg', None)

    # ------------------------------------------------------------
    #  Interface tests
    # ------------------------------------------------------------
    def test_interface_has_L(self):
        if self.L is None:
            return (False, "SKIP: L not available")
        if not isinstance(self.L, (int, float)):
            return (False, "L is not numeric")
        return (True, "L exists and is numeric")

    def test_interface_has_E_e(self):
        if self.E_e is None:
            return (False, "SKIP: E_e not available")
        if not isinstance(self.E_e, (int, float)):
            return (False, "E_e is not numeric")
        return (True, "E_e exists and is numeric")

    def test_interface_has_theta_deg(self):
        if self.theta_deg is None:
            return (False, "SKIP: theta_deg not available")
        if not isinstance(self.theta_deg, (int, float)):
            return (False, "theta_deg is not numeric")
        return (True, "theta_deg exists and is numeric")

    # ------------------------------------------------------------
    #  Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_proper_lifetime(self):
        if self.Gamma is None or self.hbar is None:
            return (False, "SKIP: Gamma or hbar missing")
        tau = self.hbar / self.Gamma
        if tau <= 0:
            return (False, f"Proper lifetime must be positive, got {tau}")
        return (True, f"Proper lifetime = {tau} s (positive)")

    def test_invariant_L_via_lifetime(self):
        if any(x is None for x in [self.m_J, self.p_J, self.Gamma, self.hbar, self.c, self.L]):
            return (False, "SKIP: missing data for L invariant")
        # L = (p_J / m_J) * (hbar / Gamma) * c
        L_expected = (self.p_J / self.m_J) * (self.hbar / self.Gamma) * self.c
        if not math.isclose(self.L, L_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"L {self.L} differs from expected {L_expected}")
        return (True, f"L matches derived value {L_expected}")

    def test_invariant_E_e_half_E_J(self):
        if any(x is None for x in [self.m_J, self.p_J, self.E_e]):
            return (False, "SKIP: missing data for E_e invariant")
        E_J = math.sqrt(self.p_J**2 + self.m_J**2)
        E_e_expected = E_J / 2.0
        if not math.isclose(self.E_e, E_e_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_e {self.E_e} differs from expected {E_e_expected}")
        return (True, f"E_e matches E_J/2 = {E_e_expected}")

    def test_invariant_theta_formula(self):
        if any(x is None for x in [self.m_J, self.p_J, self.theta_deg]):
            return (False, "SKIP: missing data for theta invariant")
        theta_expected = math.atan(self.m_J / self.p_J) * 180.0 / math.pi
        if not math.isclose(self.theta_deg, theta_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"theta_deg {self.theta_deg} differs from expected {theta_expected}")
        return (True, f"theta_deg matches arctan(m_J/p_J) = {theta_expected}")

    # ------------------------------------------------------------
    #  Modeling / trace consistency tests
    # ------------------------------------------------------------
    def test_modeling_momentum_conservation(self):
        if any(x is None for x in [self.p_J, self.E_e, self.theta_deg]):
            return (False, "SKIP: missing data for momentum conservation")
        theta_rad = self.theta_deg * math.pi / 180.0
        # p_e ≈ E_e (since m_e negligible)
        p_e = self.E_e
        p_J_derived = 2 * p_e * math.cos(theta_rad)
        if not math.isclose(self.p_J, p_J_derived, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"p_J = {self.p_J} does not match 2*p_e*cos(theta) ≈ {p_J_derived}")
        return (True, f"Momentum conservation holds: p_J ≈ 2p_e cosθ = {p_J_derived}")

    def test_modeling_energy_conservation(self):
        if any(x is None for x in [self.m_J, self.p_J, self.E_e]):
            return (False, "SKIP: missing data for energy conservation")
        E_J = math.sqrt(self.p_J**2 + self.m_J**2)
        if not math.isclose(E_J, 2 * self.E_e, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_J = {E_J} does not match 2 E_e = {2*self.E_e}")
        return (True, f"Energy conservation holds: E_J ≈ 2E_e = {2*self.E_e}")

    # ------------------------------------------------------------
    #  Monotonicity tests (directional trend checks)
    # ------------------------------------------------------------
    def test_monotonicity_L_increases_with_p_J(self):
        if any(x is None for x in [self.m_J, self.p_J, self.Gamma, self.hbar, self.c]):
            return (False, "SKIP: missing data for L monotonicity")
        # Compute baseline from givens (formula-independent)
        baseline = (self.p_J / self.m_J) * (self.hbar / self.Gamma) * self.c
        # Perturb p_J by +1%
        p_J_new = self.p_J * 1.01
        perturbed = (p_J_new / self.m_J) * (self.hbar / self.Gamma) * self.c
        if perturbed <= baseline:
            return (False, f"L did not increase when p_J +1%: perturbed={perturbed} <= baseline={baseline}")
        return (True, f"L increases with p_J (baseline={baseline}, perturbed={perturbed})")

    def test_monotonicity_E_e_increases_with_p_J(self):
        if any(x is None for x in [self.m_J, self.p_J]):
            return (False, "SKIP: missing data for E_e monotonicity")
        E_J = math.sqrt(self.p_J**2 + self.m_J**2)
        baseline = E_J / 2.0
        p_J_new = self.p_J * 1.01
        E_J_new = math.sqrt(p_J_new**2 + self.m_J**2)
        perturbed = E_J_new / 2.0
        if perturbed <= baseline:
            return (False, f"E_e did not increase when p_J +1%: perturbed={perturbed} <= baseline={baseline}")
        return (True, f"E_e increases with p_J (baseline={baseline}, perturbed={perturbed})")

    def test_monotonicity_theta_decreases_with_p_J(self):
        if any(x is None for x in [self.m_J, self.p_J]):
            return (False, "SKIP: missing data for theta monotonicity")
        baseline = math.atan(self.m_J / self.p_J) * 180.0 / math.pi
        p_J_new = self.p_J * 1.01
        perturbed = math.atan(self.m_J / p_J_new) * 180.0 / math.pi
        if perturbed >= baseline:
            return (False, f"theta did not decrease when p_J +1%: perturbed={perturbed} >= baseline={baseline}")
        return (True, f"theta decreases with p_J (baseline={baseline}, perturbed={perturbed})")

    # ------------------------------------------------------------
    #  Answer sanity (weak, broad checks only)
    # ------------------------------------------------------------
    def test_answer_sanity(self):
        failures = []
        # L sanity
        if self.L is not None:
            if not (1e-12 < self.L < 1e-8):
                failures.append(f"L = {self.L} not in expected range (1e-12, 1e-8) m")
        else:
            failures.append("L missing")
        # E_e sanity
        if self.E_e is not None:
            if self.m_J is not None and self.p_J is not None:
                E_J = math.sqrt(self.p_J**2 + self.m_J**2)
                # Allow wide window around E_J/2
                lower, upper = E_J/2 - 10, E_J/2 + 10
                if not (lower < self.E_e < upper):
                    failures.append(f"E_e = {self.E_e} not in ({lower}, {upper})")
            else:
                failures.append("Cannot compute E_J for E_e sanity")
        else:
            failures.append("E_e missing")
        # theta_deg sanity
        if self.theta_deg is not None:
            if not (0 < self.theta_deg < 10):
                failures.append(f"theta_deg = {self.theta_deg} not in (0, 10) degrees")
        else:
            failures.append("theta_deg missing")
        if failures:
            return (False, " | ".join(failures))
        return (True, "All answer sanity checks passed")

    # ------------------------------------------------------------
    #  Run all tests
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)