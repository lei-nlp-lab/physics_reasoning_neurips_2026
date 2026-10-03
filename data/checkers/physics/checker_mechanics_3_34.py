import math
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits the solution for the K meson decay problem.
    Checks invariance of energy/momentum conservation, internal consistency,
    monotonicity under parameter perturbations, and weak final answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens (rest energies in MeV, c=1)
        self.mK = getattr(solution_instance, 'mK', None)
        if self.mK is None:
            givens = getattr(solution_instance, 'givens', {})
            self.mK = givens.get('mK', None)

        self.mmu = getattr(solution_instance, 'mmu', None)
        if self.mmu is None:
            givens = getattr(solution_instance, 'givens', {})
            self.mmu = givens.get('mmu', None)

        # Safely extract results
        results = getattr(solution_instance, 'results', {})
        self.p = results.get('p', None)
        self.K_nu = results.get('K_nu', None)
        self.K_mu = results.get('K_mu', None)

        # Explanation steps are only used to decide which relations to check
        self.explanation = getattr(solution_instance, 'explanation_steps', [])

    # ----- INTERFACE TESTS -----
    def test_interface_required_keys_exist(self) -> Tuple[bool, str]:
        """Check that all three result keys are present."""
        if self.p is None:
            return (False, "Missing key: p")
        if self.K_nu is None:
            return (False, "Missing key: K_nu")
        if self.K_mu is None:
            return (False, "Missing key: K_mu")
        return (True, "All required keys present (p, K_nu, K_mu)")

    def test_interface_values_positive(self) -> Tuple[bool, str]:
        """All result values should be positive (momenta and kinetic energies)."""
        if self.p is not None and self.p <= 0:
            return (False, f"p should be positive, got {self.p}")
        if self.K_nu is not None and self.K_nu <= 0:
            return (False, f"K_nu should be positive, got {self.K_nu}")
        if self.K_mu is not None and self.K_mu <= 0:
            return (False, f"K_mu should be positive, got {self.K_mu}")
        return (True, "All positive values")

    # ----- INVARIANT TESTS (use math.isclose) -----
    def test_invariant_energy_conservation(self) -> Tuple[bool, str]:
        """Check mK = sqrt(p^2 + mmu^2) + p."""
        if None in (self.mK, self.mmu, self.p):
            return (False, "SKIP: missing givens or p for energy conservation")
        try:
            lhs = self.mK
            rhs = math.sqrt(self.p**2 + self.mmu**2) + self.p
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Energy conserved: mK={lhs} ≈ sqrt(p^2+mmu^2)+p={rhs}")
            else:
                return (False, f"Energy NOT conserved: mK={lhs}, RHS={rhs}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_invariant_momentum_equality(self) -> Tuple[bool, str]:
        """p (muon momentum magnitude) should equal K_nu (massless neutrino kinetic energy = momentum)."""
        if self.p is None or self.K_nu is None:
            return (False, "SKIP: missing p or K_nu")
        try:
            if math.isclose(self.p, self.K_nu, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"p ≈ K_nu: {self.p} ≈ {self.K_nu}")
            else:
                return (False, f"p != K_nu: p={self.p}, K_nu={self.K_nu}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_invariant_muon_kinetic_energy(self) -> Tuple[bool, str]:
        """Check K_mu = sqrt(p^2 + mmu^2) - mmu."""
        if None in (self.p, self.mmu, self.K_mu):
            return (False, "SKIP: missing p, mmu, or K_mu")
        try:
            expected_K_mu = math.sqrt(self.p**2 + self.mmu**2) - self.mmu
            if math.isclose(expected_K_mu, self.K_mu, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"K_mu consistent: {self.K_mu} ≈ {expected_K_mu}")
            else:
                return (False, f"K_mu mismatch: got {self.K_mu}, expected {expected_K_mu}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_invariant_total_kinetic_energy(self) -> Tuple[bool, str]:
        """Total kinetic energy = K_mu + K_nu should equal mK - mmu (energy conservation)."""
        if None in (self.mK, self.mmu, self.K_mu, self.K_nu):
            return (False, "SKIP: missing required quantities")
        try:
            lhs = self.K_mu + self.K_nu
            rhs = self.mK - self.mmu
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Total KE = {lhs} ≈ mK - mmu = {rhs}")
            else:
                return (False, f"Total KE = {lhs}, expected {rhs}")
        except Exception as e:
            return (False, f"Exception: {e}")

    # ----- MODELING / TRACE TESTS -----
    def test_modeling_p_formula(self) -> Tuple[bool, str]:
        """
        Check that p was derived correctly from the analytic formula (with c=1):
        p = (mK^2 - mmu^2) / (2*mK)
        """
        if None in (self.mK, self.mmu, self.p):
            return (False, "SKIP: missing givens or p")
        try:
            expected_p = (self.mK**2 - self.mmu**2) / (2 * self.mK)
            if math.isclose(self.p, expected_p, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"p ≈ (mK^2 - mmu^2)/(2*mK) = {expected_p}")
            else:
                return (False, f"p={self.p}, expected {expected_p}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_modeling_muon_total_energy(self) -> Tuple[bool, str]:
        """
        Check that muon total energy (E_mu = sqrt(p^2+mmu^2)) is consistent with
        the derived analytic expression (mK^2+mmu^2)/(2*mK) from explanation.
        """
        if None in (self.mK, self.mmu, self.p):
            return (False, "SKIP: missing data")
        try:
            E_mu_from_p = math.sqrt(self.p**2 + self.mmu**2)
            E_mu_expected = (self.mK**2 + self.mmu**2) / (2 * self.mK)
            if math.isclose(E_mu_from_p, E_mu_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E_mu consistent: {E_mu_from_p} ≈ {E_mu_expected}")
            else:
                return (False, f"E_mu mismatch: computed {E_mu_from_p}, expected {E_mu_expected}")
        except Exception as e:
            return (False, f"Exception: {e}")

    # ----- MONOTONICITY TESTS -----
    def test_monotonicity_mK_increases_p_and_K_nu(self) -> Tuple[bool, str]:
        """
        Perturb mK upward by 5% (if given) and check that the corresponding
        p and K_nu computed from the analytic formulas increase.
        Uses the same formulas that the solution should have used.
        """
        if None in (self.mK, self.mmu, self.p, self.K_nu):
            return (False, "SKIP: missing givens or p/K_nu for monotonicity check")
        try:
            # Perturb mK by +5%
            delta = 0.05 * self.mK
            mK_pert = self.mK + delta
            # Recompute expected p and K_nu with perturbed mass
            p_pert = (mK_pert**2 - self.mmu**2) / (2 * mK_pert)
            K_nu_pert = p_pert   # because K_nu = p
            # Check monotonic increase
            if p_pert > self.p and K_nu_pert > self.K_nu:
                return (True, f"p and K_nu increase when mK increases: p {self.p} -> {p_pert}, K_nu {self.K_nu} -> {K_nu_pert}")
            else:
                return (False, f"Expected increase but got p {self.p} -> {p_pert}, K_nu {self.K_nu} -> {K_nu_pert}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_monotonicity_mK_increases_K_mu(self) -> Tuple[bool, str]:
        """
        Perturb mK upward by 5% and check that K_mu (= sqrt(p^2+mmu^2)-mmu) increases.
        """
        if None in (self.mK, self.mmu, self.p, self.K_mu):
            return (False, "SKIP: missing data for K_mu monotonicity")
        try:
            delta = 0.05 * self.mK
            mK_pert = self.mK + delta
            # Compute new p from analytic formula
            p_pert = (mK_pert**2 - self.mmu**2) / (2 * mK_pert)
            K_mu_pert = math.sqrt(p_pert**2 + self.mmu**2) - self.mmu
            if K_mu_pert > self.K_mu:
                return (True, f"K_mu increases when mK increases: {self.K_mu} -> {K_mu_pert}")
            else:
                return (False, f"Expected increase but got {self.K_mu} -> {K_mu_pert}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_monotonicity_mmu_increases_p_and_K_nu_decrease(self) -> Tuple[bool, str]:
        """
        Perturb mmu upward by 5% and check that p and K_nu decrease.
        """
        if None in (self.mK, self.mmu, self.p, self.K_nu):
            return (False, "SKIP: missing data for mmu monotonicity")
        try:
            delta = 0.05 * self.mmu
            mmu_pert = self.mmu + delta
            # Compute new p with perturbed mmu
            p_pert = (self.mK**2 - mmu_pert**2) / (2 * self.mK)
            K_nu_pert = p_pert
            if p_pert < self.p and K_nu_pert < self.K_nu:
                return (True, f"p and K_nu decrease when mmu increases: p {self.p} -> {p_pert}, K_nu {self.K_nu} -> {K_nu_pert}")
            else:
                return (False, f"Expected decrease but got p {self.p} -> {p_pert}, K_nu {self.K_nu} -> {K_nu_pert}")
        except Exception as e:
            return (False, f"Exception: {e}")

    def test_monotonicity_mmu_increases_K_mu_decrease(self) -> Tuple[bool, str]:
        """
        Perturb mmu upward by 5% and check that K_mu decreases.
        """
        if None in (self.mK, self.mmu, self.p, self.K_mu):
            return (False, "SKIP: missing data for K_mu monotonicity")
        try:
            delta = 0.05 * self.mmu
            mmu_pert = self.mmu + delta
            # Compute new p with perturbed mmu
            p_pert = (self.mK**2 - mmu_pert**2) / (2 * self.mK)
            # Compute new K_mu with perturbed mmu
            K_mu_pert = math.sqrt(p_pert**2 + mmu_pert**2) - mmu_pert
            if K_mu_pert < self.K_mu:
                return (True, f"K_mu decreases when mmu increases: {self.K_mu} -> {K_mu_pert}")
            else:
                return (False, f"Expected decrease but got {self.K_mu} -> {K_mu_pert}")
        except Exception as e:
            return (False, f"Exception: {e}")

    # ----- WEAK ANSWER SANITY (no math.isclose, no re-derivation) -----
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity: kinetic energies must be positive and less than mK.
        """
        if self.K_nu is None or self.K_mu is None or self.mK is None:
            return (False, "SKIP: missing results")
        msgs = []
        ok = True
        if not (0 < self.K_nu < self.mK):
            ok = False
            msgs.append(f"K_nu={self.K_nu} not in (0, mK={self.mK})")
        if not (0 < self.K_mu < self.mK):
            ok = False
            msgs.append(f"K_mu={self.K_mu} not in (0, mK={self.mK})")
        if ok:
            return (True, "Weak sanity passed: both KE positive and < mK")
        else:
            return (False, "; ".join(msgs))

    # ----- RUN ALL TESTS -----
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Gather all bound methods starting with test_
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
        return all_pass, logs