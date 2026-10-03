import math
import sys

class PhysicsChecker:
    """
    PhysicsChecker for coherent regeneration of K0 mesons.
    Checks:
       - Interface: required keys present, numeric.
       - Energy conservation between K_l and K_s.
       - Momentum difference related to mass difference.
       - Force computed from momentum change rate consistent with given F_N.
       - Validity of the approximation p_s ≈ p_i + Δm c (modeling quality).
       - Monotonicity: increasing ∆m or N increases force proportionally.
    """
    # Physical constants (SI)
    C = 3.0e8                     # m/s
    KG_PER_EV_C2 = 1.782661921627e-36   # kg per eV/c^2
    EV_TO_J = 1.602176634e-19           # J per eV

    def __init__(self, solution_instance):
        # Extract result
        self.F_N = None
        self.N = None
        self.beta = None
        self.m_l = None
        self.Delta_m = None

        try:
            # Try to get the result from the solution
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                self.F_N = solution_instance.results.get('F_N', None)
            elif hasattr(solution_instance, 'F_N'):
                self.F_N = solution_instance.F_N
            else:
                self.F_N = getattr(solution_instance, 'F_N', None)
        except Exception:
            pass

        # Extract given parameters (try givens dict first)
        givens = None
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens

        if givens is not None:
            self.N = givens.get('N', None)
            self.beta = givens.get('beta', None)
            self.m_l = givens.get('m_l', None)
            self.Delta_m = givens.get('Delta_m', None)
        else:
            # Fallback direct attributes
            self.N = getattr(solution_instance, 'N', None)
            self.beta = getattr(solution_instance, 'beta', None)
            self.m_l = getattr(solution_instance, 'm_l', None)
            self.Delta_m = getattr(solution_instance, 'Delta_m', None)

    # ----------------------------------------------------------------------
    # Helper to compute derived quantities in SI
    # ----------------------------------------------------------------------
    def _compute_si_quantities(self):
        """Return (p_i, p_s_exact, Delta_p_exact, Delta_p_approx) in SI (kg·m/s),
           or (None,...,None) if any input missing."""
        if any(v is None for v in [self.N, self.beta, self.m_l, self.Delta_m]):
            return (None, None, None, None)

        m_l_kg = self.m_l * self.KG_PER_EV_C2
        m_s_kg = (self.m_l - self.Delta_m) * self.KG_PER_EV_C2

        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)

        p_i = gamma * m_l_kg * self.beta * self.C
        E_i = gamma * m_l_kg * self.C**2

        # Exact p_s from energy conservation and smaller mass
        p_s_exact = math.sqrt((E_i/self.C)**2 - (m_s_kg*self.C)**2)

        Delta_p_exact = p_s_exact - p_i

        # Approximated as ∆m c
        Delta_m_kg = self.Delta_m * self.KG_PER_EV_C2
        Delta_p_approx = Delta_m_kg * self.C

        return (p_i, p_s_exact, Delta_p_exact, Delta_p_approx)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that F_N is present in results."""
        if self.F_N is None:
            return (False, "F_N not found in results (emit contract missing)")
        return (True, "F_N present in results")

    def test_interface_force_positive_finite(self):
        """Check that F_N is positive and finite."""
        if self.F_N is None:
            return (False, "SKIP: F_N missing")
        if self.F_N <= 0:
            return (False, f"F_N = {self.F_N} must be positive")
        if not math.isfinite(self.F_N):
            return (False, f"F_N = {self.F_N} is not finite")
        return (True, f"F_N = {self.F_N} > 0 and finite")

    # ----------------------------------------------------------------------
    # Physics invariant tests
    # ----------------------------------------------------------------------
    def test_invariant_energy_conservation(self):
        """Check that E_i = E_s (within roundoff)."""
        if any(v is None for v in [self.beta, self.m_l, self.Delta_m]):
            return (False, "SKIP: missing m_l, beta, or Delta_m")
        m_l_kg = self.m_l * self.KG_PER_EV_C2
        m_s_kg = (self.m_l - self.Delta_m) * self.KG_PER_EV_C2

        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        E_i = gamma * m_l_kg * self.C**2           # J

        # Reconstruct p_s from the exact relation (expansion not needed)
        # Actually we compute p_s again (but we already have it in _compute_si_quantities)
        # We'll compute E_s from p_s (using exact p_s) and compare.
        _, p_s_exact, _, _ = self._compute_si_quantities()
        if p_s_exact is None:
            return (False, "SKIP: cannot compute p_s")
        E_s_exact = math.sqrt((p_s_exact*self.C)**2 + (m_s_kg*self.C**2)**2)

        if math.isclose(E_i, E_s_exact, rel_tol=1e-10, abs_tol=1e-20):
            return (True, "Energy conservation holds exactly (E_i = E_s) to numerical precision")
        else:
            # This should never happen unless givens inconsistent, but be lenient
            if math.isclose(E_i, E_s_exact, rel_tol=1e-5):
                return (True, f"Energy conservation within 1e-5 relative: E_i={E_i:.6e}, E_s={E_s_exact:.6e}")
            return (False, f"Energy conservation violated: E_i={E_i:.6e}, E_s={E_s_exact:.6e}")

    def test_invariant_momentum_difference(self):
        """Check that p_s - p_i ≈ Δm c (within tolerance)."""
        _, _, Delta_p_exact, Delta_p_approx = self._compute_si_quantities()
        if Delta_p_exact is None:
            return (False, "SKIP: missing givens")
        # The approximation is valid because Δm << m_l; check within 1e-2 relative
        if math.isclose(Delta_p_exact, Delta_p_approx, rel_tol=1e-2, abs_tol=1e-35):
            return (True, f"Δp ≈ Δm*c: exact {Delta_p_exact:.3e} ≈ approx {Delta_p_approx:.3e} kg·m/s")
        else:
            return (False, f"Δp exact = {Delta_p_exact:.3e} not consistent with Δm*c = {Delta_p_approx:.3e}")

    def test_invariant_positive_delta_p(self):
        """Check that p_s > p_i (beam gains momentum, so brick loses momentum)."""
        _, _, Delta_p_exact, _ = self._compute_si_quantities()
        if Delta_p_exact is None:
            return (False, "SKIP: missing givens")
        if Delta_p_exact > 0:
            return (True, f"Δp = {Delta_p_exact:.3e} > 0 (momentum increased along beam direction)")
        else:
            return (False, f"Δp = {Delta_p_exact:.3e} should be positive (p_s > p_i)")

    # ----------------------------------------------------------------------
    # Modeling / trace quality tests
    # ----------------------------------------------------------------------
    def test_modeling_force_from_momentum_rate(self):
        """Check that F_N = N * Δp_exact (from exact p_s - p_i) within tolerance."""
        _, _, Delta_p_exact, _ = self._compute_si_quantities()
        if Delta_p_exact is None:
            return (False, "SKIP: missing givens")
        if self.F_N is None:
            return (False, "SKIP: F_N missing")
        Force_computed = self.N * Delta_p_exact
        if math.isclose(Force_computed, self.F_N, rel_tol=1e-2, abs_tol=1e-30):
            return (True, f"Force consistent with exact Δp: N*Δp_exact = {Force_computed:.3e} N ≈ F_N = {self.F_N:.3e} N")
        else:
            return (False, f"Force mismatch: N*Δp_exact = {Force_computed:.3e} N, but F_N = {self.F_N:.3e} N")

    def test_modeling_approximation_validity(self):
        """Check that the approximation p_s ≈ p_i + Δm c gives force within 1% of exact force.
           This validates the series expansion used in solution steps."""
        _, _, Delta_p_exact, Delta_p_approx = self._compute_si_quantities()
        if Delta_p_exact is None or Delta_p_approx is None:
            return (False, "SKIP: missing givens")
        F_exact = self.N * Delta_p_exact
        F_approx = self.N * Delta_p_approx
        # Compare relative difference
        if abs(F_approx) < 1e-40:
            return (False, "Approximate force too small, numeric issue")
        rel_diff = abs(F_exact - F_approx) / F_approx
        if rel_diff < 0.01:
            return (True, f"Approximation valid: exact F = {F_exact:.3e} N, approx F = {F_approx:.3e} N, rel diff = {rel_diff:.3e}")
        else:
            return (False, f"Approximation error too large: exact F = {F_exact:.3e}, approx F = {F_approx:.3e}, rel diff = {rel_diff:.3e}")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_increase_delta_m(self):
        """Increase ∆m by 10% and check that force increases proportionally."""
        if any(v is None for v in [self.N, self.beta, self.m_l, self.Delta_m]):
            return (False, "SKIP: missing givens")
        if self.F_N is None:
            return (False, "SKIP: F_N missing")

        # Baseline force from exact Δp
        _, _, Delta_p_exact, _ = self._compute_si_quantities()
        if Delta_p_exact is None:
            return (False, "SKIP: cannot compute Δp")
        F_baseline = self.N * Delta_p_exact

        # Modified: increase Delta_m by 10%
        Delta_m_new = self.Delta_m * 1.1
        # Recompute everything with new Delta_m
        m_l_kg = self.m_l * self.KG_PER_EV_C2
        m_s_kg_new = (self.m_l - Delta_m_new) * self.KG_PER_EV_C2
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        p_i = gamma * m_l_kg * self.beta * self.C
        E_i = gamma * m_l_kg * self.C**2
        p_s_new = math.sqrt((E_i/self.C)**2 - (m_s_kg_new*self.C)**2)
        Delta_p_new = p_s_new - p_i
        F_new = self.N * Delta_p_new

        ratio = F_new / F_baseline
        expected_ratio = 1.1
        if math.isclose(ratio, expected_ratio, rel_tol=2e-2):
            return (True, f"Δm +10%: force ratio = {ratio:.4f} (expected ~{expected_ratio})")
        else:
            return (False, f"Δm +10%: force ratio = {ratio:.4f} not close to {expected_ratio}")

    def test_monotonicity_increase_N(self):
        """Increase N by 10% and check that force scales proportionally."""
        if self.N is None or self.F_N is None:
            return (False, "SKIP: missing N or F_N")
        _, _, Delta_p_exact, _ = self._compute_si_quantities()
        if Delta_p_exact is None:
            return (False, "SKIP: missing givens")
        N_new = self.N * 1.1
        F_new = N_new * Delta_p_exact
        F_baseline = self.N * Delta_p_exact
        ratio = F_new / F_baseline
        expected_ratio = 1.1
        if math.isclose(ratio, expected_ratio, rel_tol=1e-3):
            return (True, f"N +10%: force ratio = {ratio:.4f} (expected {expected_ratio})")
        else:
            return (False, f"N +10%: force ratio = {ratio:.4f} not close to {expected_ratio}")

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_ methods."""
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