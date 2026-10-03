import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the 27Si positron decay problem.
    Audits solution correctness using Coulomb energy difference and Q-value.
    """

    def __init__(self, solution_instance):
        # Extract data safely; if not present, use None (no numeric defaults)
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})
        constants = getattr(solution_instance, 'constants', {})

        # Key physical inputs
        self.Q = givens.get('Q', None)               # maximum positron total energy (MeV)
        self.A = givens.get('A', None)               # mass number
        self.Zi = givens.get('Z_initial', None)      # initial charge
        self.Zf = givens.get('Z_final', None)        # final charge

        # Constants from solution (must be provided by solution)
        self.me_c2 = constants.get('me_c2', None)        # MeV
        self.e2_4pie0 = constants.get('e2_4pie0', None)  # MeV·fm

        # Result to be tested
        self.r0 = results.get('r0', None)

        # Explanation steps (not used for numeric checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def run_all_tests(self):
        """Discover and run all test_* methods, collect results."""
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

    # -----------------------------------------------------------------
    # Interface tests: ensure required keys exist
    # -----------------------------------------------------------------
    def test_interface_required_givens(self):
        if self.Q is None:
            return (False, "Missing givens['Q']")
        if self.A is None:
            return (False, "Missing givens['A']")
        if self.Zi is None or self.Zf is None:
            return (False, "Missing initial/final charge numbers")
        if self.me_c2 is None:
            return (False, "Missing constants['me_c2']")
        if self.e2_4pie0 is None:
            return (False, "Missing constants['e2_4pie0']")
        return (True, "All required givens present")

    def test_interface_result_r0(self):
        if self.r0 is None:
            return (False, "Missing results['r0']")
        if not isinstance(self.r0, (int, float)):
            return (False, f"r0 is not numeric: {type(self.r0)}")
        return (True, "r0 present and numeric")

    # -----------------------------------------------------------------
    # Invariant tests: theorem‑based physics checks (use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_coulomb_energy_difference(self):
        """
        Check that r0 is consistent with the Coulomb energy difference formula.
        Expected ΔE_C = Q + 2*m_e c^2 (explanation steps).
        """
        if any(v is None for v in (self.r0, self.Q, self.me_c2, self.e2_4pie0, self.A, self.Zi, self.Zf)):
            return (False, "SKIP: missing givens or result for invariant check")

        expected_delta_EC = self.Q + 2 * self.me_c2   # 3.48 + 1.022 = 4.502 MeV
        # Radius from r0
        R_from_r0 = self.r0 * (self.A ** (1.0/3.0))
        # Coulomb energy difference from formula
        delta_EC_from_r0 = (3.0/5.0) * self.e2_4pie0 * (self.Zi**2 - self.Zf**2) / R_from_r0

        if math.isclose(delta_EC_from_r0, expected_delta_EC, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ΔE_C from r0 = {delta_EC_from_r0:.4f} MeV ≈ expected {expected_delta_EC:.4f} MeV")
        else:
            return (False, f"ΔE_C mismatch: got {delta_EC_from_r0:.4f}, expected {expected_delta_EC:.4f}")

    def test_invariant_radius_scaling(self):
        """
        Check that the effective radius R (from Coulomb formula) matches r0 * A^(1/3).
        Use the same expected ΔE_C to compute a reference R, then compare to derived R.
        """
        if any(v is None for v in (self.r0, self.Q, self.me_c2, self.e2_4pie0, self.A, self.Zi, self.Zf)):
            return (False, "SKIP: missing data for radius scaling check")

        expected_delta_EC = self.Q + 2 * self.me_c2
        # R from Coulomb formula: R = (3/5)*e^2/(4πϵ0)*(Z_i^2 - Z_f^2) / ΔE_C
        R_coulomb = (3.0/5.0) * self.e2_4pie0 * (self.Zi**2 - self.Zf**2) / expected_delta_EC
        # R from r0
        R_from_r0 = self.r0 * (self.A ** (1.0/3.0))

        if math.isclose(R_from_r0, R_coulomb, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R from r0 = {R_from_r0:.3f} fm ≈ Coulomb R = {R_coulomb:.3f} fm")
        else:
            return (False, f"R mismatch: r0 gives {R_from_r0:.3f}, Coulomb gives {R_coulomb:.3f}")

    # -----------------------------------------------------------------
    # Modeling / trace tests: consistency with the derived relation
    # -----------------------------------------------------------------
    def test_modeling_delta_EC_from_Q(self):
        """
        Verify that ΔE_C = Q + 2*m_e c^2 (the relation used in the explanation).
        This tests whether the solver correctly interpreted Q.
        """
        if self.Q is None or self.me_c2 is None or self.r0 is None:
            return (False, "SKIP: missing Q, m_e c^2, or r0 for modeling check")

        expected = self.Q + 2 * self.me_c2
        R = self.r0 * (self.A ** (1.0/3.0))
        derived_delta = (3.0/5.0) * self.e2_4pie0 * (self.Zi**2 - self.Zf**2) / R
        if math.isclose(derived_delta, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derived ΔE_C = {derived_delta:.4f} matches Q+2m_e c^2 = {expected:.4f}")
        else:
            return (False, f"Derived ΔE_C = {derived_delta:.4f} does not match Q+2m_e c^2 = {expected:.4f}")

    def test_modeling_coulomb_energy_positive(self):
        """
        Check that the Coulomb energy difference is positive (repulsive).
        """
        if self.r0 is None:
            return (False, "SKIP: no r0")
        R = self.r0 * (self.A ** (1.0/3.0))
        delta = (3.0/5.0) * self.e2_4pie0 * (self.Zi**2 - self.Zf**2) / R
        if delta > 0:
            return (True, f"ΔE_C = {delta:.4f} MeV > 0")
        else:
            return (False, f"Expected ΔE_C > 0, got {delta}")

    # -----------------------------------------------------------------
    # Monotonicity tests: perturb an input and compare relative change
    # -----------------------------------------------------------------
    def test_monotonicity_r0_increases_with_e2_4pie0(self):
        """
        If e^2/(4πϵ0) increases, r0 should increase linearly (direct proportionality).
        """
        if any(v is None for v in (self.r0, self.Q, self.me_c2, self.e2_4pie0, self.A, self.Zi, self.Zf)):
            return (False, "SKIP: missing data for monotonicity check")

        perturbation_factor = 1.1  # +10%
        new_e2 = self.e2_4pie0 * perturbation_factor

        # Recompute r0 with perturbed constant (same relation)
        expected_delta = self.Q + 2 * self.me_c2
        R_new = (3.0/5.0) * new_e2 * (self.Zi**2 - self.Zf**2) / expected_delta
        r0_new = R_new / (self.A ** (1.0/3.0))

        # The original solution r0 should be proportionally smaller
        ratio_new_old = r0_new / self.r0
        expected_ratio = perturbation_factor
        if math.isclose(ratio_new_old, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r0 scales linearly: perturbed/old = {ratio_new_old:.3f}, expected {expected_ratio:.3f}")
        else:
            return (False, f"r0 scaling mismatch: perturbed/old = {ratio_new_old:.3f}, expected {expected_ratio:.3f}")

    def test_monotonicity_r0_decreases_with_Q(self):
        """
        If Q increases (larger measured kinetic energy), ΔE_C increases,
        so R decreases, and r0 decreases (inverse proportionality).
        """
        if any(v is None for v in (self.r0, self.Q, self.me_c2, self.e2_4pie0, self.A, self.Zi, self.Zf)):
            return (False, "SKIP: missing data for monotonicity check")

        perturbation_factor = 1.1  # +10%
        new_Q = self.Q * perturbation_factor

        new_delta = new_Q + 2 * self.me_c2
        R_new = (3.0/5.0) * self.e2_4pie0 * (self.Zi**2 - self.Zf**2) / new_delta
        r0_new = R_new / (self.A ** (1.0/3.0))

        # r0 should decrease when Q increases
        if r0_new < self.r0:
            # Verify relative change: r0_new/r0_old ≈ (Q_old+2m)/(Q_new+2m)
            expected_ratio = (self.Q + 2*self.me_c2) / (new_Q + 2*self.me_c2)
            actual_ratio = r0_new / self.r0
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"r0 decreases as Q increases: ratio = {actual_ratio:.4f}, expected {expected_ratio:.4f}")
            else:
                return (False, f"r0 ratio mismatch: {actual_ratio:.4f} vs expected {expected_ratio:.4f}")
        else:
            return (False, f"Expected r0 to decrease when Q increases, but r0 increased or stayed same")

    # -----------------------------------------------------------------
    # Weak final answer sanity (at most one, no numeric derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_r0_physical_range(self):
        """
        Check that r0 is positive, finite, and within a plausible range (0.5 to 3 fm).
        """
        if self.r0 is None:
            return (False, "SKIP: no r0 available")
        if self.r0 <= 0 or math.isnan(self.r0) or math.isinf(self.r0):
            return (False, f"r0 = {self.r0} is not physically plausible")
        if 0.5 < self.r0 < 3.0:
            return (True, f"r0 = {self.r0:.3f} fm is within typical range (0.5–3 fm)")
        else:
            return (False, f"r0 = {self.r0:.3f} fm is outside expected range (0.5–3 fm)")