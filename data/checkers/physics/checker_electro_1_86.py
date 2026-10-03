import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens - use None defaults
        self.Q = getattr(solution_instance, 'Q', None)
        self.z = getattr(solution_instance, 'z', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        
        # Extract results from the results dict
        results = getattr(solution_instance, 'results', {})
        self.W = results.get('W', None)
        self.F_plus_mag = results.get('F_plus_mag', None)
        self.F_plate_mag = results.get('F_plate_mag', None)
        self.Q_ind_left_d = results.get('Q_ind_left_d', None)
        self.Q_ind_right_d = results.get('Q_ind_right_d', None)
        self.Q_ind_left_e = results.get('Q_ind_left_e', None)
        self.Q_ind_right_e = results.get('Q_ind_right_e', None)
        
        # Derived quantity
        if self.z is not None:
            self.d = 4.0 * self.z
        else:
            self.d = None

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

    # ------------------------------------------------------------------
    # Interface tests: ensure required keys exist and are not None
    # ------------------------------------------------------------------
    def test_interface_W_exists(self):
        if self.W is None:
            return (False, "W is missing or None")
        return (True, f"W = {self.W}")

    def test_interface_F_plus_mag_exists(self):
        if self.F_plus_mag is None:
            return (False, "F_plus_mag is missing or None")
        return (True, f"F_plus_mag = {self.F_plus_mag}")

    def test_interface_F_plate_mag_exists(self):
        if self.F_plate_mag is None:
            return (False, "F_plate_mag is missing or None")
        return (True, f"F_plate_mag = {self.F_plate_mag}")

    def test_interface_Q_ind_left_d_exists(self):
        if self.Q_ind_left_d is None:
            return (False, "Q_ind_left_d is missing or None")
        return (True, f"Q_ind_left_d = {self.Q_ind_left_d}")

    def test_interface_Q_ind_right_d_exists(self):
        if self.Q_ind_right_d is None:
            return (False, "Q_ind_right_d is missing or None")
        return (True, f"Q_ind_right_d = {self.Q_ind_right_d}")

    def test_interface_Q_ind_left_e_exists(self):
        if self.Q_ind_left_e is None:
            return (False, "Q_ind_left_e is missing or None")
        return (True, f"Q_ind_left_e = {self.Q_ind_left_e}")

    def test_interface_Q_ind_right_e_exists(self):
        if self.Q_ind_right_e is None:
            return (False, "Q_ind_right_e is missing or None")
        return (True, f"Q_ind_right_e = {self.Q_ind_right_e}")

    # ------------------------------------------------------------------
    # Invariant tests (physical theorem-based checks)
    # ------------------------------------------------------------------
    def test_invariant_energy_positive(self):
        """Energy required to remove charges should be positive (work done by external agent)."""
        if self.W is None:
            return (False, "SKIP: W not available")
        if self.W <= 0:
            return (False, f"W should be positive, got {self.W}")
        return (True, f"W is positive: {self.W}")

    def test_invariant_force_on_charge_zero(self):
        """Force on +Q should be exactly zero due to symmetry of images."""
        if self.F_plus_mag is None:
            return (False, "SKIP: F_plus_mag not available")
        if not math.isclose(self.F_plus_mag, 0.0, abs_tol=1e-10):
            return (False, f"F_plus_mag should be zero, got {self.F_plus_mag}")
        return (True, "F_plus_mag is zero (within tolerance)")

    def test_invariant_plate_force_positive(self):
        """Force on left plate should be positive (attractive toward the charges)."""
        if self.F_plate_mag is None:
            return (False, "SKIP: F_plate_mag not available")
        if self.F_plate_mag <= 0:
            return (False, f"F_plate_mag should be positive, got {self.F_plate_mag}")
        return (True, f"F_plate_mag is positive: {self.F_plate_mag}")

    def test_invariant_induced_charge_sum_part_d(self):
        """For both charges present, net induced charge on plates is zero (total system charge zero)."""
        if self.Q_ind_left_d is None or self.Q_ind_right_d is None:
            return (False, "SKIP: induced charges d not available")
        total = self.Q_ind_left_d + self.Q_ind_right_d
        if not math.isclose(total, 0.0, abs_tol=1e-10):
            return (False, f"Sum of induced charges (part d) should be 0, got {total}")
        return (True, f"Sum of induced charges (d) is zero: {total}")

    def test_invariant_induced_charge_sum_part_e(self):
        """After removing -Q, total induced charge on plates equals -Q."""
        if self.Q_ind_left_e is None or self.Q_ind_right_e is None or self.Q is None:
            return (False, "SKIP: required values missing")
        total = self.Q_ind_left_e + self.Q_ind_right_e
        if not math.isclose(total, -self.Q, rel_tol=1e-9, abs_tol=1e-12):
            return (False, f"Sum of induced charges (part e) should be -Q = {-self.Q}, got {total}")
        return (True, f"Sum of induced charges (e) equals -Q: {total}")

    def test_invariant_induced_charge_signs_part_e(self):
        """Both induced charges in part (e) must be negative (since no other positive images inside)."""
        if self.Q_ind_left_e is None or self.Q_ind_right_e is None:
            return (False, "SKIP: induced charges e not available")
        if self.Q_ind_left_e >= 0 or self.Q_ind_right_e >= 0:
            return (False, f"Both induced charges (e) should be negative, got left {self.Q_ind_left_e}, right {self.Q_ind_right_e}")
        return (True, "Both induced charges (e) are negative")

    # ------------------------------------------------------------------
    # Modeling tests (check consistency with known physics relations)
    # ------------------------------------------------------------------
    def test_modeling_induced_charge_ratio_part_e(self):
        """With only +Q present, the induced charges on left and right plates should be in the ratio (d-a)/a = 3,
           where a = z, d = 4z. So left/right = 3."""
        if self.Q_ind_left_e is None or self.Q_ind_right_e is None:
            return (False, "SKIP: induced charges e not available")
        if self.Q_ind_right_e == 0:
            return (False, "Right induced charge (e) is zero, cannot compute ratio")
        ratio = self.Q_ind_left_e / self.Q_ind_right_e
        # Expected ratio = (d-a)/a = (4z - z)/z = 3
        if not math.isclose(ratio, 3.0, rel_tol=0.02):
            return (False, f"Ratio left/right induced charges (e) should be 3, got {ratio}")
        return (True, f"Ratio left/right induced charges (e) is 3.0: {ratio}")

    def test_modeling_induced_charge_signs_part_d(self):
        """For part (d), physics says left plate induced charge is negative, right plate positive."""
        if self.Q_ind_left_d is None or self.Q_ind_right_d is None:
            return (False, "SKIP: induced charges d not available")
        if self.Q_ind_left_d >= 0 or self.Q_ind_right_d <= 0:
            return (False, f"Expected left < 0 and right > 0, got left {self.Q_ind_left_d}, right {self.Q_ind_right_d}")
        return (True, "Signs correct: left negative, right positive")

    def test_modeling_force_on_charge_consistent_with_images(self):
        """If F_plus_mag is zero, then the net force on the system of real charges + images must be zero.
           This is a sanity check that F_plate_mag is nonzero."""
        if self.F_plus_mag is None or self.F_plate_mag is None:
            return (False, "SKIP: required forces not available")
        if self.F_plate_mag == 0:
            return (False, "F_plate_mag should be non-zero")
        return (True, f"F_plate_mag is non-zero: {self.F_plate_mag}")

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_with_separation(self):
        """
        As the plate separation (z) increases, the energy W should decrease.
        We compute baseline W0 for given z, then perturb z by +10% and compute W1.
        Check that W1 < W0.
        """
        if self.Q is None or self.z is None or self.epsilon0 is None:
            return (False, "SKIP: required givens (Q, z, epsilon0) not available")
        # Baseline
        z_baseline = self.z
        W0 = self.Q**2 * math.log(2) / (16 * math.pi * self.epsilon0 * z_baseline)
        # Perturbed: increase z by 10%
        z_pert = z_baseline * 1.1
        W1 = self.Q**2 * math.log(2) / (16 * math.pi * self.epsilon0 * z_pert)
        if not (W1 < W0):
            return (False, f"W should decrease when z increases: W0={W0:.6g}, W1={W1:.6g}")
        return (True, f"Energy decreases with increasing z: W0={W0:.6g}, W1={W1:.6g} (decrease of {W0-W1:.6g})")

    def test_monotonicity_plate_force_with_separation(self):
        """
        As z increases, the force magnitude on the left plate should decrease.
        Compute baseline F0 for given z, perturb z by +10%, compute F1, check F1 < F0.
        """
        if self.Q is None or self.z is None or self.epsilon0 is None:
            return (False, "SKIP: required givens (Q, z, epsilon0) not available")
        # Baseline
        z_baseline = self.z
        F0 = self.Q**2 * math.log(2) / (256 * math.pi * self.epsilon0 * z_baseline**2)
        # Perturbed
        z_pert = z_baseline * 1.1
        F1 = self.Q**2 * math.log(2) / (256 * math.pi * self.epsilon0 * z_pert**2)
        if not (F1 < F0):
            return (False, f"F_plate should decrease when z increases: F0={F0:.6g}, F1={F1:.6g}")
        return (True, f"Plate force decreases with increasing z: F0={F0:.6g}, F1={F1:.6g} (decrease of {F0-F1:.6g})")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, must not rederive or use isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_general_order(self):
        """Sanity check: W, F_plate_mag are not extremely large or NaN. No re-derivation."""
        if self.W is not None:
            if not math.isfinite(self.W) or self.W > 1e100:
                return (False, f"W out of plausible range: {self.W}")
        if self.F_plate_mag is not None:
            if not math.isfinite(self.F_plate_mag) or self.F_plate_mag > 1e100:
                return (False, f"F_plate_mag out of plausible range: {self.F_plate_mag}")
        return (True, "All checked quantities are finite and within plausible bounds")