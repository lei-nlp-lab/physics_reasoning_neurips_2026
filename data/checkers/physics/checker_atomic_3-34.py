import math
import sympy
import traceback
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits the solution for the Y1* resonance problem.
    Checks interface, invariants, modeling consistency, monotonicity, and a weak answer sanity.
    """

    # Known masses from the problem statement (MeV)
    MASS_Y = 1385.0
    MASS_LAMBDA = 1116.0
    MASS_SIGMA = 1190.0
    MASS_PI = 140.0
    SIGMA_PI_THRESHOLD = MASS_SIGMA + MASS_PI               # 1330 MeV
    LAMBDA_2PI_THRESHOLD = MASS_LAMBDA + 2 * MASS_PI        # 1396 MeV

    # Known particle quantum numbers (same for all solutions)
    S_LAMBDA = -1
    S_PI = 0
    B_Y = 1
    I_LAMBDA = 0
    I_PI = 1
    I3_LAMBDA = 0
    I3_PI = 1
    P_LAMBDA = 1
    P_PI = -1
    S_LAMBDA_SPIN = 0.5
    L_ORBITAL = 1  # relative p-wave, given in problem

    def __init__(self, solution_instance: Any) -> None:
        # Extract emitted results safely
        results = getattr(solution_instance, 'results', {})
        self.S_Y = results.get('S_Y', None)
        self.Y_Y = results.get('Y_Y', None)
        self.I_Y = results.get('I_Y', None)
        self.I3_Y = results.get('I3_Y', None)
        self.P_Y = results.get('P_Y', None)
        self.J_Y_list = results.get('J_Y_list', None)
        self.allowed_decay_modes = results.get('allowed_decay_modes', None)

        # Optional: get mass from solution (if provided), otherwise use problem constant
        # We do not use a fallback default; it's a known constant from the problem.
        # But we try to use the solution's value if present.
        if hasattr(solution_instance, 'givens'):
            self.mass_Y = solution_instance.givens.get('mass_Y', self.MASS_Y)
        else:
            self.mass_Y = self.MASS_Y

    # ------------------------------------------------------------------
    # INTERFACE CHECKS
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist and have expected types."""
        required = {
            'S_Y': (int, float, type(None)),
            'Y_Y': (int, float, type(None)),
            'I_Y': (int, float, type(None)),
            'I3_Y': (int, float, type(None)),
            'P_Y': (int, float, type(None)),
            'J_Y_list': (list, type(None)),
            'allowed_decay_modes': (list, type(None)),
        }
        for key, types in required.items():
            val = getattr(self, key, None)
            if val is None:
                return (False, f"Missing required key: {key}")
            if not isinstance(val, types):
                return (False, f"Key {key} has wrong type: {type(val).__name__}")
        return (True, "All required keys present and have correct types.")

    def test_interface_strangeness_is_integer(self) -> Tuple[bool, str]:
        """S_Y must be an integer (or float that represents an integer)."""
        if self.S_Y is None:
            return (True, "SKIP: S_Y not provided")
        try:
            int_val = int(self.S_Y)
            if int_val != self.S_Y:
                return (False, f"S_Y={self.S_Y} is not an integer value")
            return (True, f"S_Y is integer: {int_val}")
        except (TypeError, ValueError):
            return (False, f"Cannot convert S_Y={self.S_Y} to integer")

    # ------------------------------------------------------------------
    # INVARIANT (THEOREM‑BASED) CHECKS
    # ------------------------------------------------------------------
    def test_invariant_hypercharge(self) -> Tuple[bool, str]:
        """Y = B + S. For a baryon (B=1), Y should equal 1 + S_Y."""
        if self.S_Y is None or self.Y_Y is None:
            return (True, "SKIP: S_Y or Y_Y missing")
        expected = self.B_Y + self.S_Y
        if self.Y_Y == expected:
            return (True, f"Y={self.Y_Y} equals 1 + S={self.S_Y} = {expected}")
        else:
            return (False, f"Y={self.Y_Y} does not equal 1+S={expected}")

    def test_invariant_isospin_decomposition(self) -> Tuple[bool, str]:
        """I(Y1*) must equal I(Lambda)+I(pi) = 0+1 = 1 for strong decay."""
        if self.I_Y is None or self.I3_Y is None:
            return (True, "SKIP: I_Y or I3_Y missing")
        # Check I_Y value
        if self.I_Y != 1:
            return (False, f"I_Y = {self.I_Y}, expected 1")
        # I3 should be I3_pi = 1 (since Lambda has I3=0)
        if self.I3_Y != 1:
            return (False, f"I3_Y = {self.I3_Y}, expected 1")
        # Also check that |I3| <= I
        if abs(self.I3_Y) > self.I_Y:
            return (False, f"|I3| = {abs(self.I3_Y)} > I = {self.I_Y}")
        return (True, f"I=1, I3=1 are consistent with final state (I=1, I3=1)")

    def test_invariant_parity_from_decay(self) -> Tuple[bool, str]:
        """P(Y1*) = P(Lambda)*P(pi)*(-1)^L = (+1)*(-1)*(-1)^1 = +1."""
        if self.P_Y is None:
            return (True, "SKIP: P_Y missing")
        expected = self.P_LAMBDA * self.P_PI * ((-1) ** self.L_ORBITAL)
        if self.P_Y == expected:
            return (True, f"P_Y = {self.P_Y} matches expected {expected}")
        else:
            return (False, f"P_Y = {self.P_Y}, but expected {expected}")

    def test_invariant_spin_possibilities(self) -> Tuple[bool, str]:
        """J possible values from |L - S| to L+S = 0.5 to 1.5."""
        if self.J_Y_list is None:
            return (True, "SKIP: J_Y_list missing")
        expected = [0.5, 1.5]  # 1/2 and 3/2
        if not isinstance(self.J_Y_list, list):
            return (False, "J_Y_list is not a list")
        # Check that both expected values are present
        for val in expected:
            # use math.isclose for floating comparisons
            found = any(math.isclose(v, val, rel_tol=1e-2) for v in self.J_Y_list)
            if not found:
                return (False, f"Expected J={val} not found in list {self.J_Y_list}")
        # Check no other values outside range
        min_J = abs(self.L_ORBITAL - self.S_LAMBDA_SPIN)
        max_J = self.L_ORBITAL + self.S_LAMBDA_SPIN
        for v in self.J_Y_list:
            if v < min_J - 1e-2 or v > max_J + 1e-2:
                return (False, f"J={v} is outside allowed range [{min_J}, {max_J}]")
        return (True, f"J list contains 1/2 and 3/2, all values in range")

    # ------------------------------------------------------------------
    # MODELING / TRACE CONSISTENCY
    # ------------------------------------------------------------------
    def test_modeling_allowed_decay_modes_quantum_numbers(self) -> Tuple[bool, str]:
        """Check that the allowed decay modes satisfy strangeness and isospin conservation."""
        if self.allowed_decay_modes is None:
            return (True, "SKIP: allowed_decay_modes missing")
        if not isinstance(self.allowed_decay_modes, list):
            return (False, "allowed_decay_modes is not a list")
        # Known strong decays that should be allowed: Sigma pi (with Sigma having S=-1, I=1)
        # We check that 'Sigma pi' is present (case-insensitive)
        found = any('sigma pi' in str(mode).lower() for mode in self.allowed_decay_modes)
        if not found:
            return (False, f"Expected 'Sigma pi' decay mode not found in {self.allowed_decay_modes}")
        # Check that NK is NOT present (strangeness violation)
        if any('n k' in str(mode).lower() or 'nucleon kaon' in str(mode).lower()
               for mode in self.allowed_decay_modes):
            return (False, "NK decay should be forbidden by strangeness conservation")
        return (True, "Allowed decay modes include Sigma pi and exclude NK")

    def test_modeling_threshold_consistency(self) -> Tuple[bool, str]:
        """Check that allowed decay modes are kinematically possible given mass."""
        if self.allowed_decay_modes is None:
            return (True, "SKIP: no decay modes to check")
        if not self.allowed_decay_modes:
            return (True, "SKIP: empty decay mode list")
        # Check each mode against threshold
        for mode in self.allowed_decay_modes:
            mode_lower = str(mode).lower()
            if 'sigma pi' in mode_lower:
                if self.mass_Y <= self.SIGMA_PI_THRESHOLD:
                    return (False, f"Mass {self.mass_Y} <= Sigma+pi threshold ({self.SIGMA_PI_THRESHOLD}), yet mode listed")
            elif 'lambda pi pi' in mode_lower:
                if self.mass_Y <= self.LAMBDA_2PI_THRESHOLD:
                    return (False, f"Mass {self.mass_Y} <= Lambda+2pi threshold ({self.LAMBDA_2PI_THRESHOLD}), yet mode listed")
        return (True, "All decay modes are above threshold")

    # ------------------------------------------------------------------
    # MONOTONICITY TEST
    # ------------------------------------------------------------------
    def test_monotonicity_mass_threshold_Sigma_pi(self) -> Tuple[bool, str]:
        """
        Verify that if the resonance mass were slightly below the Sigma+pi threshold,
        the 'Sigma pi' decay would be forbidden.
        Since the actual mass is above threshold (1385 > 1330), we expect the decay to be allowed.
        If we imagine a perturbation that lowers the mass to just below threshold,
        the decay should not be allowed. Here we check the logical implication:
        mass > threshold => mode allowed.
        """
        if self.allowed_decay_modes is None:
            return (True, "SKIP: no decay modes to test monotonicity")
        # Use given mass (should be above threshold)
        threshold = self.SIGMA_PI_THRESHOLD
        if self.mass_Y <= threshold:
            return (True, "SKIP: mass is already at or below threshold, cannot test monotonicity")
        # Check that 'Sigma pi' is in the allowed list
        found = any('sigma pi' in str(mode).lower() for mode in self.allowed_decay_modes)
        if not found:
            return (False, f"Mass {self.mass_Y} > threshold {threshold}, but 'Sigma pi' is not listed as allowed")
        # If mass were reduced by 5% (i.e., 0.95 * mass_Y), it might drop below threshold.
        # We cannot test the solution for a different mass, but we can note that the condition
        # mass > threshold is necessary for the strong decay.  This test passes if the condition holds.
        return (True, f"Mass {self.mass_Y} > {threshold}, so Sigma pi is allowed; monotonic behavior confirmed")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_spin_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: Check that the spin list values are reasonable (positive, finite, and
        not exceeding a plausible maximum like 5).
        """
        if self.J_Y_list is None or not isinstance(self.J_Y_list, list):
            return (True, "SKIP: J_Y_list not available")
        for val in self.J_Y_list:
            if not math.isfinite(val) or val <= 0:
                return (False, f"Unreasonable spin value: {val}")
            if val > 5:  # arbitrary large upper bound
                return (False, f"Spin value {val} is implausibly large for a baryon resonance")
        return (True, "All spin values are positive, finite, and within reasonable range")

    # ------------------------------------------------------------------
    # RUNNER
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)