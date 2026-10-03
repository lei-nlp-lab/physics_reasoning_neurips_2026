import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens: m_p, m_pi
        self.m_p = getattr(solution_instance, 'm_p', None)
        if self.m_p is None:
            self.m_p = getattr(solution_instance, 'givens', {}).get('m_p', None)
        if self.m_p is None:
            self.m_p = getattr(solution_instance, 'results', {}).get('m_p', None)
        
        self.m_pi = getattr(solution_instance, 'm_pi', None)
        if self.m_pi is None:
            self.m_pi = getattr(solution_instance, 'givens', {}).get('m_pi', None)
        if self.m_pi is None:
            self.m_pi = getattr(solution_instance, 'results', {}).get('m_pi', None)
        
        # Extract threshold kinetic energy K
        self.K = None
        results = getattr(solution_instance, 'results', {})
        if isinstance(results, dict):
            self.K = results.get('K', None)
        if self.K is None:
            self.K = getattr(solution_instance, 'K', None)
        
        # Optional: store E1, s if present
        self.E1 = results.get('E1', None) if isinstance(results, dict) else None
        self.s = results.get('s', None) if isinstance(results, dict) else None

    # ---- Interface tests ----
    def test_interface_required_keys(self):
        if None in (self.m_p, self.m_pi, self.K):
            missing = []
            if self.m_p is None: missing.append('m_p')
            if self.m_pi is None: missing.append('m_pi')
            if self.K is None: missing.append('K')
            return (False, f"Missing required keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_types(self):
        for name, val in [('m_p', self.m_p), ('m_pi', self.m_pi), ('K', self.K)]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (type: {type(val).__name__})")
        return (True, "All values are numeric")

    # ---- Invariant tests (theorem-based physical sanity) ----
    def test_invariant_threshold_s(self):
        if None in (self.m_p, self.m_pi, self.K):
            return (False, "SKIP: missing required data")
        s_from_K = 2 * self.m_p * self.K + 4 * self.m_p**2
        s_th = (2 * self.m_p + self.m_pi)**2
        if math.isclose(s_from_K, s_th, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s from K ({s_from_K:.1f} MeV²) ≈ s_th ({s_th:.1f} MeV²)")
        else:
            return (False, f"s from K ({s_from_K:.1f} MeV²) != s_th ({s_th:.1f} MeV²)")

    def test_invariant_energy_positivity(self):
        if self.K is None:
            return (False, "SKIP: K not available")
        if self.K <= 0:
            return (False, f"K must be positive, got {self.K}")
        return (True, f"K = {self.K} > 0")

    def test_invariant_relativistic_relation(self):
        if self.E1 is None or self.s is None or self.m_p is None:
            return (False, "SKIP: E1 or s not available")
        p1_sq = (self.E1 + self.m_p)**2 - self.s
        expected_p1_sq = self.E1**2 - self.m_p**2
        if math.isclose(p1_sq, expected_p1_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Relativistic relation holds")
        else:
            return (False, f"p1² mismatch: {p1_sq:.1f} vs {expected_p1_sq:.1f}")

    # ---- Modeling/trace tests (based on explanation steps) ----
    def test_modeling_threshold_formula(self):
        if None in (self.m_p, self.m_pi, self.K):
            return (False, "SKIP: missing data")
        expected_K = 2 * self.m_pi + self.m_pi**2 / (2 * self.m_p)
        if math.isclose(self.K, expected_K, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"K = {self.K} ≈ expected {expected_K:.3f} MeV")
        else:
            return (False, f"K = {self.K} != expected {expected_K:.3f} MeV")

    def test_modeling_threshold_condition(self):
        if None in (self.m_p, self.m_pi, self.K):
            return (False, "SKIP: missing data")
        s_from_K = 2 * self.m_p * self.K + 4 * self.m_p**2
        s_th = (2 * self.m_p + self.m_pi)**2
        if s_from_K >= s_th - 1e-6:
            return (True, f"s from K ({s_from_K:.1f}) >= s_th ({s_th:.1f})")
        else:
            return (False, f"s from K ({s_from_K:.1f}) < s_th ({s_th:.1f})")

    # ---- Monotonicity test ----
    def test_monotonicity_increasing_pion_mass_increases_K(self):
        """
        Perturb the pion mass upward and check that threshold kinetic energy increases.
        Uses the derived formula K = 2*m_pi + m_pi^2/(2*m_p).
        """
        if None in (self.m_p, self.m_pi):
            return (False, "SKIP: m_p or m_pi missing")
        # Baseline threshold kinetic energy from given masses
        K_base = 2 * self.m_pi + self.m_pi**2 / (2 * self.m_p)
        # Perturb: increase m_pi by 1%
        m_pi_pert = self.m_pi * 1.01
        K_pert = 2 * m_pi_pert + m_pi_pert**2 / (2 * self.m_p)
        if K_pert > K_base:
            return (True, f"Increasing m_pi by 1% increases K from {K_base:.3f} to {K_pert:.3f} MeV")
        else:
            return (False, f"Monotonicity violated: K decreased from {K_base:.3f} to {K_pert:.3f} MeV")

    # ---- Weak answer sanity test (only one allowed) ----
    def test_answer_sanity_proton_threshold(self):
        if self.K is None:
            return (False, "SKIP: K not available")
        if 200 < self.K < 400:
            return (True, f"K = {self.K} MeV is within plausible range (200-400 MeV)")
        else:
            return (False, f"K = {self.K} MeV outside plausible range (200-400 MeV)")

    # ---- Runner ----
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)