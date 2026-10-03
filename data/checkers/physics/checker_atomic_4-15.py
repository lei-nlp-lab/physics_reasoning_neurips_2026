import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens with safe defaults (None if missing)
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}

        self.m_p = givens.get('m_p', None)
        self.m_pi = givens.get('m_pi', None)

        # Fallback: if not in givens, try solution_instance attribute
        if self.m_p is None:
            self.m_p = getattr(solution_instance, 'm_p', None)
        if self.m_pi is None:
            self.m_pi = getattr(solution_instance, 'm_pi', None)
        # Final fallback: from results (unlikely, but safe)
        if self.m_p is None:
            self.m_p = results.get('m_p', None)
        if self.m_pi is None:
            self.m_pi = results.get('m_pi', None)

        # Computed results (expected keys: E1, E2, E3)
        self.E1 = results.get('E1', None)
        self.E2 = results.get('E2', None)
        self.E3 = results.get('E3', None)

    # ===== Helper methods =====
    def _threshold_energy(self, m_p, m_pi, n_pions):
        """Compute threshold energy E = ( (2*m_p + n_pions*m_pi)^2 ) / (2*m_p) - m_p."""
        if m_p is None or m_pi is None:
            return None
        M = 2 * m_p + n_pions * m_pi
        E = M**2 / (2 * m_p) - m_p
        return E

    # ===== Interface tests =====
    def test_interface_required_keys_exist(self):
        if None in (self.E1, self.E2, self.E3):
            missing = [k for k, v in [('E1', self.E1), ('E2', self.E2), ('E3', self.E3)] if v is None]
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required keys (E1, E2, E3) are present")

    def test_interface_keys_positive(self):
        for name, val in [('E1', self.E1), ('E2', self.E2), ('E3', self.E3)]:
            if val is None:
                return (False, f"{name} missing")
            if val <= 0:
                return (False, f"{name} = {val} not positive")
        return (True, "All energies are positive")

    def test_interface_givens_present(self):
        if self.m_p is None:
            return (False, "m_p missing from givens")
        if self.m_pi is None:
            return (False, "m_pi missing from givens")
        return (True, "Both masses available")

    def test_interface_givens_positive(self):
        if self.m_p is not None and self.m_p <= 0:
            return (False, f"m_p = {self.m_p} not positive")
        if self.m_pi is not None and self.m_pi <= 0:
            return (False, f"m_pi = {self.m_pi} not positive")
        if self.m_p is None or self.m_pi is None:
            return (False, "SKIP: some masses missing")
        return (True, "All masses positive")

    # ===== Theorem-based invariant tests =====
    def test_invariant_threshold_Q1(self):
        if None in (self.m_p, self.m_pi, self.E1):
            return (False, "SKIP: insufficient data for Q1")
        M = 2 * self.m_p + 4 * self.m_pi
        expected_s = M**2
        s = 2 * self.m_p * (self.E1 + self.m_p)
        if math.isclose(s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q1 s={s:.4f} ≈ {expected_s:.4f}")
        return (False, f"Q1 s={s:.4f} != M^2={expected_s:.4f}")

    def test_invariant_threshold_Q2(self):
        if None in (self.m_p, self.m_pi, self.E2):
            return (False, "SKIP: insufficient data for Q2")
        M = 2 * self.m_p + 4 * self.m_pi
        expected_s = M**2
        s = 2 * self.m_p * (self.E2 + self.m_p)
        if math.isclose(s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q2 s={s:.4f} ≈ {expected_s:.4f}")
        return (False, f"Q2 s={s:.4f} != M^2={expected_s:.4f}")

    def test_invariant_threshold_Q3(self):
        if None in (self.m_p, self.m_pi, self.E3):
            return (False, "SKIP: insufficient data for Q3")
        M = 2 * self.m_p + 8 * self.m_pi
        expected_s = M**2
        s = 2 * self.m_p * (self.E3 + self.m_p)
        if math.isclose(s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q3 s={s:.4f} ≈ {expected_s:.4f}")
        return (False, f"Q3 s={s:.4f} != M^2={expected_s:.4f}")

    # ===== Modeling / Trace tests =====
    def test_modeling_Q1_Q2_identical(self):
        if None in (self.E1, self.E2):
            return (False, "SKIP: missing E1/E2")
        if math.isclose(self.E1, self.E2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E1 ≈ E2: {self.E1:.4f} ≈ {self.E2:.4f}")
        return (False, f"E1={self.E1:.4f} != E2={self.E2:.4f}")

    def test_modeling_formula_consistency(self):
        """Verify that computed energies match the derived formula E = M^2/(2m_p) - m_p."""
        if None in (self.m_p, self.m_pi):
            return (False, "SKIP: masses missing")
        failures = []
        # Q1 & Q2: 4 pions
        M12 = 2*self.m_p + 4*self.m_pi
        expected_E12 = M12**2 / (2*self.m_p) - self.m_p
        if self.E1 is not None and not math.isclose(self.E1, expected_E12, rel_tol=1e-2, abs_tol=1e-2):
            failures.append(f"E1={self.E1:.4f} != expected {expected_E12:.4f}")
        if self.E2 is not None and not math.isclose(self.E2, expected_E12, rel_tol=1e-2, abs_tol=1e-2):
            failures.append(f"E2={self.E2:.4f} != expected {expected_E12:.4f}")
        # Q3: 8 pions
        M3 = 2*self.m_p + 8*self.m_pi
        expected_E3 = M3**2 / (2*self.m_p) - self.m_p
        if self.E3 is not None and not math.isclose(self.E3, expected_E3, rel_tol=1e-2, abs_tol=1e-2):
            failures.append(f"E3={self.E3:.4f} != expected {expected_E3:.4f}")
        if failures:
            return (False, " | ".join(failures))
        return (True, "All energies match derived formula")

    # ===== Monotonicity tests (must explicitly perturb) =====
    def test_monotonicity_energy_with_pion_count(self):
        """Perturb the number of pions by +1 and verify threshold energy increases."""
        if None in (self.m_p, self.m_pi):
            return (False, "SKIP: masses missing")
        # Baseline: 4 pions total
        base_E = self._threshold_energy(self.m_p, self.m_pi, 4)
        # Perturbed: 5 pions total (+1)
        perturbed_E = self._threshold_energy(self.m_p, self.m_pi, 5)
        if base_E is None or perturbed_E is None:
            return (False, "Failed to compute energies")
        if perturbed_E > base_E:
            return (True, f"E increased: {perturbed_E:.4f} > {base_E:.4f} (n_pions: 4 -> 5)")
        return (False, f"E did not increase: base={base_E:.4f}, perturbed={perturbed_E:.4f}")

    def test_monotonicity_energy_with_pion_mass(self):
        """Perturb pion mass by +10% and verify threshold energy increases (for fixed 4 pions)."""
        if None in (self.m_p, self.m_pi):
            return (False, "SKIP: masses missing")
        # Baseline: using given m_pi
        base_E = self._threshold_energy(self.m_p, self.m_pi, 4)
        # Perturb: increase m_pi by 10%
        perturbed_m_pi = self.m_pi * 1.10
        perturbed_E = self._threshold_energy(self.m_p, perturbed_m_pi, 4)
        if base_E is None or perturbed_E is None:
            return (False, "Failed to compute energies")
        if perturbed_E > base_E:
            return (True, f"E increased: {perturbed_E:.4f} > {base_E:.4f} (m_pi: {self.m_pi:.4f} -> {perturbed_m_pi:.4f})")
        return (False, f"E did not increase: base={base_E:.4f}, perturbed={perturbed_E:.4f}")

    # ===== Weak answer sanity (only one allowed) =====
    def test_answer_sanity_bounds(self):
        """Check all energies are > m_p and < 10 GeV (broad range)."""
        if None in (self.m_p, self.E1, self.E2, self.E3):
            return (False, "SKIP: insufficient data")
        for name, val in [('E1', self.E1), ('E2', self.E2), ('E3', self.E3)]:
            if val <= self.m_p:
                return (False, f"{name} = {val:.4f} ≤ m_p = {self.m_p:.4f}")
            if val > 10:
                return (False, f"{name} = {val:.4f} > 10 GeV (unreasonable)")
        return (True, "All energies within plausible range (m_p < E < 10 GeV)")

    # ===== Main runner =====
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