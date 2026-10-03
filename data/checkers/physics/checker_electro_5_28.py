import math
import sys

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely, default to None
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}

        # Inputs
        self.E_eV = self.givens.get('E', None)          # electron energy in eV
        self.R_m = self.givens.get('R', None)           # orbit radius in meters

        # Outputs (from solver)
        self.B = self.results.get('B', None)            # magnetic field in Gauss
        self.deltaE_over_E = self.results.get('deltaE_over_E', None)

        # Constants (Gaussian units)
        self.c = 2.998e10      # cm/s
        self.e = 4.803e-10     # esu
        self.m = 9.109e-28     # g
        self.eV_to_erg = 1.602e-12

    # ---------- interface tests ----------
    def test_interface_required_keys(self):
        """Check that both required results keys exist and are numeric."""
        if self.B is None:
            return (False, "Results key 'B' missing")
        if self.deltaE_over_E is None:
            return (False, "Results key 'deltaE_over_E' missing")
        if not isinstance(self.B, (int, float)):
            return (False, f"B is not numeric, got {type(self.B)}")
        if not isinstance(self.deltaE_over_E, (int, float)):
            return (False, f"deltaE_over_E is not numeric, got {type(self.deltaE_over_E)}")
        return (True, "Both required keys present and numeric")

    def test_interface_givens_present(self):
        """Check that crucial givens are provided (not strictly required but helpful)."""
        if self.E_eV is None:
            return (False, "Given 'E' (eV) missing")
        if self.R_m is None:
            return (False, "Given 'R' (m) missing")
        return (True, "Givens E and R present")

    # ---------- invariant tests (theorem-based physical checks) ----------
    def test_invariant_B_from_formula(self):
        """Check that B matches the relativistic formula B ≈ E/(eR) in Gaussian units."""
        if any(x is None for x in (self.B, self.E_eV, self.R_m)):
            return (False, "SKIP: missing data for B invariant")
        # Convert to Gaussian: E in erg, R in cm
        E_erg = self.E_eV * self.eV_to_erg
        R_cm = self.R_m * 100.0
        expected_B = E_erg / (self.e * R_cm)
        if not math.isfinite(expected_B):
            return (False, f"Computed expected B is not finite: {expected_B}")
        if math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"B {self.B:.6e} matches expected {expected_B:.6e} within 1%")
        return (False, f"B {self.B:.6e} deviates from expected {expected_B:.6e} more than 1%")

    def test_invariant_deltaE_over_E_from_formula(self):
        """Check that ΔE/E matches the synchrotron radiation formula."""
        if any(x is None for x in (self.deltaE_over_E, self.E_eV, self.R_m)):
            return (False, "SKIP: missing data for ΔE/E invariant")
        E_erg = self.E_eV * self.eV_to_erg
        R_cm = self.R_m * 100.0
        gamma = E_erg / (self.m * self.c**2)
        if gamma <= 1:
            return (False, f"Gamma = {gamma} is not >> 1, relativistic approximation invalid")
        expected_ratio = (4.0 * math.pi / 3.0) * (self.e**2 * gamma**3) / (R_cm * self.m * self.c**2)
        if not math.isfinite(expected_ratio):
            return (False, f"Computed expected ΔE/E is not finite: {expected_ratio}")
        if math.isclose(self.deltaE_over_E, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ΔE/E {self.deltaE_over_E:.6e} matches expected {expected_ratio:.6e} within 1%")
        return (False, f"ΔE/E {self.deltaE_over_E:.6e} deviates from expected {expected_ratio:.6e} more than 1%")

    # ---------- modeling/trace tests ----------
    def test_modeling_gamma_large(self):
        """Check that γ computed from E is much larger than 1 (E >> mc²)."""
        if self.E_eV is None:
            return (False, "SKIP: E not available")
        E_erg = self.E_eV * self.eV_to_erg
        gamma = E_erg / (self.m * self.c**2)
        if gamma >= 10:
            return (True, f"gamma = {gamma:.1f} >> 1, consistent with relativistic regime")
        return (False, f"gamma = {gamma:.1f} is not >> 1, relativistic approximation questionable")

    def test_modeling_v_approx_c(self):
        """Since E >> mc², v should be very close to c (β≈1)."""
        if self.E_eV is None:
            return (False, "SKIP: E not available")
        E_erg = self.E_eV * self.eV_to_erg
        gamma = E_erg / (self.m * self.c**2)
        if gamma <= 1:
            return (False, "gamma <= 1, relativistic approximation invalid")
        beta = math.sqrt(1 - 1/gamma**2)
        if beta > 0.99:
            return (True, f"v/c = {beta:.4f} ~ 1, consistent with ultrarelativistic limit")
        return (False, f"v/c = {beta:.4f} is not close enough to 1 for E >> mc²")

    # ---------- monotonicity tests (explicit perturbation of inputs) ----------
    def test_monotonicity_physics_B_increases_with_E(self):
        """
        Check the physical relationship: B = E/(eR) is monotonic increasing with E.
        We perturb E by +1% and -1%, compute B using the same formula, and verify
        the monotonic trend.
        """
        if any(x is None for x in (self.E_eV, self.R_m)):
            return (False, "SKIP: missing givens for monotonicity B vs E")
        R_cm = self.R_m * 100.0
        # Baseline E and corresponding B
        E_erg_base = self.E_eV * self.eV_to_erg
        B_base = E_erg_base / (self.e * R_cm)
        # Perturb E up by 1%
        E_erg_up = E_erg_base * 1.01
        B_up = E_erg_up / (self.e * R_cm)
        # Perturb E down by 1%
        E_erg_down = E_erg_base * 0.99
        B_down = E_erg_down / (self.e * R_cm)
        if B_up > B_base and B_down < B_base:
            return (True, f"Physics monotonic: B increases with E (base {B_base:.6e}, +1% {B_up:.6e}, -1% {B_down:.6e})")
        return (False, f"Physics not monotonic: base {B_base:.6e}, up {B_up:.6e}, down {B_down:.6e}")

    def test_monotonicity_physics_B_decreases_with_R(self):
        """
        Check the physical relationship: B = E/(eR) is monotonic decreasing with R.
        Perturb R by +1% and -1%.
        """
        if any(x is None for x in (self.E_eV, self.R_m)):
            return (False, "SKIP: missing givens for monotonicity B vs R")
        E_erg = self.E_eV * self.eV_to_erg
        R_cm_base = self.R_m * 100.0
        B_base = E_erg / (self.e * R_cm_base)
        R_cm_up = R_cm_base * 1.01
        B_up = E_erg / (self.e * R_cm_up)
        R_cm_down = R_cm_base * 0.99
        B_down = E_erg / (self.e * R_cm_down)
        if B_up < B_base and B_down > B_base:
            return (True, f"Physics monotonic: B decreases with R (base {B_base:.6e}, +1% {B_up:.6e}, -1% {B_down:.6e})")
        return (False, f"Physics not monotonic: base {B_base:.6e}, up {B_up:.6e}, down {B_down:.6e}")

    def test_monotonicity_physics_deltaE_over_E_increases_with_E(self):
        """
        Check the physical relationship: ΔE/E ∝ E³, so it should increase with E.
        Perturb E by +1% and -1%.
        """
        if any(x is None for x in (self.E_eV, self.R_m)):
            return (False, "SKIP: missing givens for monotonicity ΔE/E vs E")
        R_cm = self.R_m * 100.0
        m = self.m
        c = self.c
        e = self.e
        E_erg_base = self.E_eV * self.eV_to_erg
        gamma_base = E_erg_base / (m * c**2)
        ratio_base = (4*math.pi/3) * (e**2 * gamma_base**3) / (R_cm * m * c**2)
        # perturb E up by 1%
        E_erg_up = E_erg_base * 1.01
        gamma_up = E_erg_up / (m * c**2)
        ratio_up = (4*math.pi/3) * (e**2 * gamma_up**3) / (R_cm * m * c**2)
        # perturb E down by 1%
        E_erg_down = E_erg_base * 0.99
        gamma_down = E_erg_down / (m * c**2)
        ratio_down = (4*math.pi/3) * (e**2 * gamma_down**3) / (R_cm * m * c**2)
        if ratio_up > ratio_base and ratio_down < ratio_base:
            return (True, f"Physics monotonic: ΔE/E increases with E (base {ratio_base:.6e}, +1% {ratio_up:.6e}, -1% {ratio_down:.6e})")
        return (False, f"Physics not monotonic: base {ratio_base:.6e}, up {ratio_up:.6e}, down {ratio_down:.6e}")

    # ---------- answer sanity checks (only one weak method) ----------
    def test_answer_sanity_sign_and_range(self):
        """Check that B and ΔE/E are positive and within broad plausible ranges."""
        issues = []
        if self.B is not None:
            if self.B <= 0 or not math.isfinite(self.B):
                issues.append("B is not positive and finite")
            elif not (1000 <= self.B <= 10000):
                issues.append(f"B = {self.B:.2f} Gs is outside plausible range [1000, 10000]")
        else:
            issues.append("B not available")
        if self.deltaE_over_E is not None:
            if self.deltaE_over_E <= 0 or not math.isfinite(self.deltaE_over_E):
                issues.append("ΔE/E is not positive and finite")
            elif not (1e-5 <= self.deltaE_over_E <= 1e-2):
                issues.append(f"ΔE/E = {self.deltaE_over_E:.3e} outside plausible range [1e-5, 1e-2]")
        else:
            issues.append("ΔE/E not available")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Both outputs positive and within plausible ranges")

    # ---------- runner ----------
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