import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the Princeton synchrotron problem.
    Checks invariant relations, modeling consistency, monotonicity,
    and answer sanity based on the provided solution_instance.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Givens from solution_instance attributes or results dictionary
        self.E_p = getattr(solution_instance, 'E_p', None) or self.results.get('E_p', None)
        self.m_p = getattr(solution_instance, 'm_p', None) or self.results.get('m_p', None)
        self.m_N = getattr(solution_instance, 'm_N', None) or self.results.get('m_N', None)
        self.q_p = getattr(solution_instance, 'q_p', None) or self.results.get('q_p', None)
        self.q_N = getattr(solution_instance, 'q_N', None) or self.results.get('q_N', None)

        # Emitted result
        self.K_N = self.results.get('K_N', None)

        # Infer missing givens from known problem context (only if one is present and the other missing)
        if self.m_p is not None and self.m_N is None:
            self.m_N = 14.0 * self.m_p
        if self.q_p is not None and self.q_N is None:
            self.q_N = 6.0 * self.q_p

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---------- Interface checks ----------
    def test_interface_required_result_present(self):
        """Check that the emitted kinetic energy K_N is present."""
        if self.K_N is None:
            return (False, "K_N not found in results")
        if not isinstance(self.K_N, (int, float)):
            return (False, f"K_N has invalid type: {type(self.K_N)}")
        return (True, "K_N is present and numeric")

    def test_interface_required_givens_present(self):
        """Check that minimum required givens (E_p, m_p) are present."""
        missing = []
        if self.E_p is None:
            missing.append("E_p")
        if self.m_p is None:
            missing.append("m_p")
        if not missing:
            return (True, "All required givens present")
        else:
            return (False, f"Missing: {', '.join(missing)}")

    # ---------- Invariant checks (use math.isclose, rel_tol=1e-2) ----------
    def test_invariant_proton_energy_momentum(self):
        """Check that E_p^2 = p_p^2 + m_p^2 holds for the proton."""
        if None in (self.E_p, self.m_p):
            return (False, "SKIP: E_p or m_p missing")
        p_p = math.sqrt(self.E_p**2 - self.m_p**2)
        lhs = self.E_p**2
        rhs = p_p**2 + self.m_p**2
        if math.isclose(lhs, rhs, rel_tol=1e-2):
            return (True, "Proton energy-momentum relation holds")
        else:
            return (False, f"E_p^2 = {lhs}, p_p^2 + m_p^2 = {rhs}")

    def test_invariant_momentum_scaling(self):
        """Check that p_N = 6 * p_p (charge scaling of momentum)."""
        if None in (self.E_p, self.m_p, self.m_N, self.K_N):
            return (False, "SKIP: missing givens or K_N")
        # Compute p_p from proton
        p_p = math.sqrt(self.E_p**2 - self.m_p**2)
        # Compute p_N from nitrogen kinetic energy
        E_N = self.K_N + self.m_N
        if E_N <= self.m_N:
            return (False, "E_N <= m_N, cannot compute p_N")
        p_N = math.sqrt(E_N**2 - self.m_N**2)
        expected_p_N = 6.0 * p_p
        if math.isclose(p_N, expected_p_N, rel_tol=1e-2):
            return (True, "p_N ~ 6 * p_p holds within 1%")
        else:
            return (False, f"p_N = {p_N}, expected 6*p_p = {expected_p_N}")

    def test_invariant_nitrogen_energy_momentum(self):
        """Check that E_N^2 = p_N^2 + m_N^2 holds for nitrogen."""
        if None in (self.m_N, self.K_N):
            return (False, "SKIP: m_N or K_N missing")
        E_N = self.K_N + self.m_N
        if E_N <= self.m_N:
            return (False, "E_N <= m_N, cannot compute p_N")
        p_N = math.sqrt(E_N**2 - self.m_N**2)
        lhs = E_N**2
        rhs = p_N**2 + self.m_N**2
        if math.isclose(lhs, rhs, rel_tol=1e-2):
            return (True, "Nitrogen energy-momentum relation holds")
        else:
            return (False, f"E_N^2 = {lhs}, p_N^2 + m_N^2 = {rhs}")

    # ---------- Modeling / trace consistency checks (at least two) ----------
    def test_modeling_kinetic_energy_formula(self):
        """Recompute K_N from givens and compare with emitted value."""
        if None in (self.E_p, self.m_p, self.m_N, self.K_N):
            return (False, "SKIP: missing givens or K_N")
        # Recompute using the prescribed steps
        p_p = math.sqrt(self.E_p**2 - self.m_p**2)
        p_N = 6.0 * p_p
        E_N = math.sqrt(p_N**2 + self.m_N**2)
        K_N_calc = E_N - self.m_N
        if math.isclose(K_N_calc, self.K_N, rel_tol=1e-2):
            return (True, "Recomputed K_N matches emitted K_N within 1%")
        else:
            return (False, f"Recomputed K_N = {K_N_calc}, emitted K_N = {self.K_N}")

    def test_modeling_mass_consistency(self):
        """Check that the nitrogen mass m_N is 14 times the proton mass (if both available)."""
        if self.m_p is None or self.m_N is None:
            return (False, "SKIP: m_p or m_N missing")
        expected_m_N = 14.0 * self.m_p
        if math.isclose(self.m_N, expected_m_N, rel_tol=1e-2):
            return (True, f"m_N = {self.m_N} ≈ 14 * m_p = {expected_m_N}")
        else:
            return (False, f"m_N = {self.m_N}, expected 14*m_p = {expected_m_N}")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_charge_increase_increases_kinetic_energy(self):
        """
        Show that increasing the charge ratio (q_N/q_p) increases the kinetic energy.
        Perturb the charge ratio from 6 to 6.6 and verify K_N increases.
        """
        if None in (self.E_p, self.m_p, self.m_N, self.K_N):
            return (False, "SKIP: missing givens or K_N")
        # Baseline p_p
        p_p = math.sqrt(self.E_p**2 - self.m_p**2)
        # Perturbed charge ratio: 6 -> 6.6 (10% increase)
        perturbed_ratio = 6.0 * 1.1
        p_N_pert = perturbed_ratio * p_p
        E_N_pert = math.sqrt(p_N_pert**2 + self.m_N**2)
        K_N_pert = E_N_pert - self.m_N
        if K_N_pert > self.K_N:
            return (True, f"Increasing charge ratio from 6 to {perturbed_ratio} increases K_N "
                          f"(baseline {self.K_N:.4f} -> perturbed {K_N_pert:.4f})")
        else:
            return (False, f"Expected increase, got perturbed K_N = {K_N_pert} <= baseline {self.K_N}")

    # ---------- Answer sanity (weak) ----------
    def test_answer_sanity_positive_finite(self):
        """Check that K_N is positive and finite."""
        if self.K_N is None:
            return (False, "SKIP: K_N not available")
        if not math.isfinite(self.K_N):
            return (False, f"K_N is not finite: {self.K_N}")
        if self.K_N <= 0:
            return (False, f"K_N is not positive: {self.K_N}")
        return (True, "K_N is positive and finite")