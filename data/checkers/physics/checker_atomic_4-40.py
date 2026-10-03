import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all required givens and results
        # Use None default for missing values
        self.m_p = getattr(solution_instance, 'm_p', None) or solution_instance.results.get('m_p', None)
        self.m_pi = getattr(solution_instance, 'm_pi', None) or solution_instance.results.get('m_pi', None)
        self.E_gamma_b = getattr(solution_instance, 'E_gamma_b', None) or solution_instance.results.get('E_gamma_b', None)
        self.E_gamma_a = getattr(solution_instance, 'E_gamma_a', None) or solution_instance.results.get('E_gamma_a', None)
        self.Ep_min = getattr(solution_instance, 'Ep_min', None) or solution_instance.results.get('Ep_min', None)

        # Additional derived quantities for internal use (if needed)
        if None not in (self.m_p, self.m_pi):
            self.s_th = (self.m_p + self.m_pi)**2
        else:
            self.s_th = None

    # ----------------------------------------------------------------------
    # Interface checks: ensure required keys exist and have valid values
    # ----------------------------------------------------------------------
    def test_interface_has_m_p(self):
        if self.m_p is None:
            return (False, "SKIP: m_p not available")
        return (True, f"m_p = {self.m_p}")

    def test_interface_has_m_pi(self):
        if self.m_pi is None:
            return (False, "SKIP: m_pi not available")
        return (True, f"m_pi = {self.m_pi}")

    def test_interface_has_E_gamma_b(self):
        if self.E_gamma_b is None:
            return (False, "SKIP: E_gamma_b not available")
        return (True, f"E_gamma_b = {self.E_gamma_b}")

    def test_interface_has_E_gamma_a(self):
        if self.E_gamma_a is None:
            return (False, "SKIP: E_gamma_a not available")
        return (True, f"E_gamma_a = {self.E_gamma_a}")

    def test_interface_has_Ep_min(self):
        if self.Ep_min is None:
            return (False, "SKIP: Ep_min not available")
        return (True, f"Ep_min = {self.Ep_min}")

    def test_interface_values_positive(self):
        """Check that all numerical values are positive and finite."""
        values = {
            'm_p': self.m_p,
            'm_pi': self.m_pi,
            'E_gamma_b': self.E_gamma_b,
            'E_gamma_a': self.E_gamma_a,
            'Ep_min': self.Ep_min
        }
        for name, val in values.items():
            if val is not None:
                if not math.isfinite(val):
                    return (False, f"{name} is not finite: {val}")
                if val <= 0:
                    return (False, f"{name} is not positive: {val}")
        return (True, "All present values are positive and finite")

    # ----------------------------------------------------------------------
    # Invariant checks: theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_threshold_condition_part_a(self):
        """Check that for part (a) the invariant mass squared equals (m_p+m_pi)^2."""
        if None in (self.m_p, self.m_pi, self.E_gamma_a):
            return (False, "SKIP: missing m_p, m_pi, or E_gamma_a")
        # s = (E_gamma + m_p)^2 - p_gamma^2, with p_gamma = E_gamma (photon)
        s = (self.E_gamma_a + self.m_p)**2 - self.E_gamma_a**2
        expected = (self.m_p + self.m_pi)**2
        if math.isclose(s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Part (a) invariant s = {s} ≈ expected {expected}")
        else:
            return (False, f"Part (a) invariant s = {s} differs from expected {expected}")

    def test_invariant_threshold_condition_part_b(self):
        """Check that for part (b) the invariant mass squared equals (m_p+m_pi)^2."""
        if None in (self.m_p, self.m_pi, self.E_gamma_b, self.Ep_min):
            return (False, "SKIP: missing one or more required quantities for part (b)")
        # Head-on collision: proton along +x, photon along -x.
        # Photon 4-momentum: (E_gamma_b, -E_gamma_b) in 1+1D.
        # Proton 4-momentum: (Ep_min, pp), with pp = sqrt(Ep_min^2 - m_p^2)
        pp = math.sqrt(self.Ep_min**2 - self.m_p**2)
        # s = (E_gamma_b + Ep_min)^2 - (-E_gamma_b - pp)^2 = (E_gamma_b + Ep_min)^2 - (E_gamma_b + pp)^2
        s = (self.E_gamma_b + self.Ep_min)**2 - (self.E_gamma_b + pp)**2
        expected = (self.m_p + self.m_pi)**2
        if math.isclose(s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Part (b) invariant s = {s} ≈ expected {expected}")
        else:
            return (False, f"Part (b) invariant s = {s} differs from expected {expected}")

    # ----------------------------------------------------------------------
    # Modeling checks: validate that derived formulas match between results
    # and the physics relations from explanation steps.
    # ----------------------------------------------------------------------
    def test_modeling_derived_formula_part_a(self):
        """Check that E_gamma_a matches the analytic formula from threshold condition."""
        if None in (self.m_p, self.m_pi, self.E_gamma_a):
            return (False, "SKIP: missing m_p, m_pi, or E_gamma_a")
        # Formula: (m_pi^2 + 2*m_p*m_pi) / (2*m_p)
        expected = (self.m_pi**2 + 2*self.m_p*self.m_pi) / (2*self.m_p)
        if math.isclose(self.E_gamma_a, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_gamma_a = {self.E_gamma_a} ≈ formula value {expected}")
        else:
            return (False, f"E_gamma_a = {self.E_gamma_a} but formula gives {expected}")

    def test_modeling_derived_formula_part_b(self):
        """Check that Ep_min matches the analytic formula from the head-on condition."""
        if None in (self.m_p, self.m_pi, self.E_gamma_b, self.Ep_min):
            return (False, "SKIP: missing required quantities for part (b)")
        # Derive expected Ep_min from the formula:
        # C = (m_pi^2 + 2*m_p*m_pi) / (2*E_gamma_b)
        C = (self.m_pi**2 + 2*self.m_p*self.m_pi) / (2*self.E_gamma_b)
        expected = (C**2 + self.m_p**2) / (2*C)
        if math.isclose(self.Ep_min, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Ep_min = {self.Ep_min} ≈ formula value {expected}")
        else:
            return (False, f"Ep_min = {self.Ep_min} but formula gives {expected}")

    # ----------------------------------------------------------------------
    # Monotonicity checks: trend/physical behavior via perturbation
    # ----------------------------------------------------------------------
    def test_monotonicity_Ep_vs_Egamma_b(self):
        """If E_gamma_b increases, the required Ep_min should decrease (easier reaction)."""
        if None in (self.m_p, self.m_pi, self.E_gamma_b, self.Ep_min):
            return (False, "SKIP: missing required quantities for monotonicity")
        # Perturb E_gamma_b upward by 10%
        perturbed = self.E_gamma_b * 1.10
        # Compute new Ep_min from formula
        C_new = (self.m_pi**2 + 2*self.m_p*self.m_pi) / (2*perturbed)
        Ep_new = (C_new**2 + self.m_p**2) / (2*C_new)
        if Ep_new < self.Ep_min:
            return (True, f"Increasing E_gamma_b from {self.E_gamma_b} to {perturbed} reduces Ep_min from {self.Ep_min} to {Ep_new}")
        else:
            return (False, f"Expected decrease in Ep_min but got increase: Ep_old={self.Ep_min}, Ep_new={Ep_new}")

    # ----------------------------------------------------------------------
    # Weak answer sanity checks (only one allowed, no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_orders_of_magnitude(self):
        """Check that the answers are in the right ballpark (~145 MeV and ~6.8e13 MeV)."""
        messages = []
        if self.E_gamma_a is not None:
            if self.E_gamma_a <= 0:
                messages.append("E_gamma_a is not positive")
            elif self.E_gamma_a < 100 or self.E_gamma_a > 200:
                messages.append(f"E_gamma_a = {self.E_gamma_a} seems far from expected ~145 MeV")
        if self.Ep_min is not None:
            if self.Ep_min <= 0:
                messages.append("Ep_min is not positive")
            elif self.Ep_min < 1e13 or self.Ep_min > 1e15:
                messages.append(f"Ep_min = {self.Ep_min} seems far from expected ~6.8e13 MeV")
            elif self.Ep_min < self.m_p:
                messages.append("Ep_min is less than proton mass, impossible for physics")
        if not messages:
            return (True, "Answer sanity OK: both values appear in plausible ranges")
        else:
            return (False, "; ".join(messages))

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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