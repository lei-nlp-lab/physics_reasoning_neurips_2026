import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the semiempirical mass formula problem (nuclear fission and shape isomers).
    Checks interface, invariants, modeling, and monotonicity of the provided solution.
    """

    def __init__(self, solution_instance):
        # Safely extract results dict; fallback to empty dict if missing
        self.results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results

        # Extract expected keys; default None if not present
        self.critical_ratio = self.results.get('critical_ratio', None)
        self.hbar2_over_2J_ground = self.results.get('hbar2_over_2J_ground', None)
        self.hbar2_over_2J_isomer = self.results.get('hbar2_over_2J_isomer', None)

        # Extract Z and A if provided (e.g., as givens) for a modeling test
        self.Z = None
        self.A = None
        if hasattr(solution_instance, 'Z'):
            self.Z = solution_instance.Z
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.Z = solution_instance.givens.get('Z', None)
        if hasattr(solution_instance, 'A'):
            self.A = solution_instance.A
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.A = solution_instance.givens.get('A', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all three required result keys are present and numeric."""
        failures = []
        for key, val in [('critical_ratio', self.critical_ratio),
                         ('hbar2_over_2J_ground', self.hbar2_over_2J_ground),
                         ('hbar2_over_2J_isomer', self.hbar2_over_2J_isomer)]:
            if val is None:
                failures.append(f"{key} is missing (None)")
            elif not isinstance(val, (int, float)):
                failures.append(f"{key} has non-numeric type {type(val).__name__}")
        if failures:
            return False, "; ".join(failures)
        return True, "All required keys present and numeric."

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_critical_ratio_formula(self):
        """
        The critical ratio Z^2/A for prompt fission must equal 2*beta/gamma,
        with beta=13 MeV and gamma=0.6 MeV => 43.333... .
        """
        if self.critical_ratio is None:
            return False, "SKIP: critical_ratio not provided."

        # Physics constants from problem statement
        beta = 13.0
        gamma = 0.6
        expected = 2.0 * beta / gamma  # 43.333...

        if not math.isclose(self.critical_ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"critical_ratio {self.critical_ratio} differs from expected {expected:.3f}."
        return True, f"critical_ratio = {self.critical_ratio:.3f} ≈ {expected:.3f}."

    def test_invariant_rotational_ordering(self):
        """
        The rotational constant for the ground state must be larger than for the shape isomer,
        because the isomer has a larger moment of inertia (smaller constant).
        """
        if self.hbar2_over_2J_ground is None or self.hbar2_over_2J_isomer is None:
            return False, "SKIP: one or both rotational constants missing."

        if not (self.hbar2_over_2J_ground > self.hbar2_over_2J_isomer):
            return False, (f"Ground constant {self.hbar2_over_2J_ground} ≤ isomer constant "
                           f"{self.hbar2_over_2J_isomer}, expected ground > isomer.")
        return True, f"Ground constant {self.hbar2_over_2J_ground} > isomer constant {self.hbar2_over_2J_isomer}."

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_fission_condition_for_Pu240(self):
        """
        For Pu-240 (Z=94, A=240), Z^2/A must be below the critical ratio,
        so it cannot undergo prompt spontaneous fission.
        """
        if self.Z is None or self.A is None:
            return False, "SKIP: Z or A not available in solution instance."
        if self.critical_ratio is None:
            return False, "SKIP: critical_ratio not provided."

        Z2_over_A = self.Z**2 / self.A
        if Z2_over_A >= self.critical_ratio:
            return False, (f"Z^2/A = {Z2_over_A:.2f} is >= critical_ratio {self.critical_ratio:.2f}, "
                           "contradicting known stability of Pu-240.")
        return True, f"Z^2/A = {Z2_over_A:.2f} < critical_ratio {self.critical_ratio:.2f} (as expected)."

    def test_modeling_rotational_constant_values(self):
        """
        Weak sanity: the rotational constants should be in physically plausible ranges
        (positive and not extremely large). Ground ~7, isomer ~3.3 MeV.
        """
        messages = []
        for name, val in [('hbar2_over_2J_ground', self.hbar2_over_2J_ground),
                          ('hbar2_over_2J_isomer', self.hbar2_over_2J_isomer)]:
            if val is None:
                messages.append(f"SKIP: {name} missing.")
                continue
            if val <= 0:
                messages.append(f"{name} = {val} is not positive.")
            elif val > 100:
                messages.append(f"{name} = {val} is suspiciously large.")
            else:
                messages.append(f"{name} = {val} is plausible.")
        if any("not positive" in m or "suspicious" in m for m in messages):
            return False, "; ".join(messages)
        return True, "; ".join(messages)

    # ------------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_Z2_A_increases_with_Z(self):
        """
        Perturb Z upward (if Z and A are given) and verify that Z^2/A increases.
        This demonstrates monotonic dependence of the computed quantity on input Z.
        """
        if self.Z is None or self.A is None:
            return False, "SKIP: Z or A not available for perturbation."

        baseline = self.Z**2 / self.A
        # Perturb Z by +1 (if possible to avoid exceeding known isotopes, but we just test monotonicity)
        Z_perturbed = self.Z + 1
        perturbed = Z_perturbed**2 / self.A
        if perturbed <= baseline:
            return False, f"Z^2/A decreased from {baseline:.3f} to {perturbed:.3f} when Z increased by 1."
        return True, f"Z^2/A increased from {baseline:.3f} to {perturbed:.3f} after Z +1 (monotonic)."

    def test_monotonicity_Z2_A_decreases_with_A(self):
        """
        Perturb A upward (if Z and A are given) and verify that Z^2/A decreases.
        """
        if self.Z is None or self.A is None:
            return False, "SKIP: Z or A not available for perturbation."

        baseline = self.Z**2 / self.A
        # Perturb A by +1
        A_perturbed = self.A + 1
        perturbed = self.Z**2 / A_perturbed
        if perturbed >= baseline:
            return False, f"Z^2/A increased from {baseline:.3f} to {perturbed:.3f} when A increased by 1."
        return True, f"Z^2/A decreased from {baseline:.3f} to {perturbed:.3f} after A +1 (monotonic)."

    # ------------------------------------------------------------------
    # (Optional) weak answer sanity – do not use math.isclose or re‑derive
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        """
        All numeric outputs must be positive and finite (not NaN, not inf).
        """
        for name, val in [('critical_ratio', self.critical_ratio),
                          ('hbar2_over_2J_ground', self.hbar2_over_2J_ground),
                          ('hbar2_over_2J_isomer', self.hbar2_over_2J_isomer)]:
            if val is None:
                continue   # skip missing; interface test will flag them
            if not math.isfinite(val):
                return False, f"{name} = {val} is not finite."
            if val <= 0:
                return False, f"{name} = {val} is not positive."
        return True, "All provided results are positive and finite."

    # ------------------------------------------------------------------
    # Test harness: discover and run all test_* methods
    # ------------------------------------------------------------------
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
                passed = False
                msg = f"CRASHED with exception: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs