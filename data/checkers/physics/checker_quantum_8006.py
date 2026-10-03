import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Auditor for a solution to the Stern-Gerlach / atomic physics problem.
    Checks interface completeness, physical invariants, modeling quality,
    monotonicity, and provides a single weak sanity check on final answers.
    """
    # Known physical constants (cgs units)
    SIGMA = 5.67e-5  # erg cm^-2 s^-1 K^-4
    C_CGS = 3e10     # cm/s
    T_ROOM = 300.0   # K

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract results
        results = getattr(solution_instance, 'results', None) or {}
        # Required keys from pseudocode [emit] contract
        self.J = results.get('J', None)          # total angular momentum (integer)
        self.mu = results.get('mu', None)        # magnetic moment
        self.u = results.get('u_ergcm3', None)   # energy density in erg/cm^3
        self.I_ratio = results.get('I_ratio', None)  # intensity ratio
        # Optionally store explanation steps (may be list or string)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_J_exists(self) -> Tuple[bool, str]:
        if self.J is None:
            return (False, "Missing key 'J' in results")
        if not isinstance(self.J, (int, float)):
            return (False, f"J must be numeric, got {type(self.J)}")
        return (True, f"J = {self.J}")

    def test_interface_mu_exists(self) -> Tuple[bool, str]:
        if self.mu is None:
            return (False, "Missing key 'mu' in results")
        if not isinstance(self.mu, (int, float)):
            return (False, f"mu must be numeric, got {type(self.mu)}")
        return (True, f"mu = {self.mu}")

    def test_interface_u_ergcm3_exists(self) -> Tuple[bool, str]:
        if self.u is None:
            return (False, "Missing key 'u_ergcm3' in results")
        if not isinstance(self.u, (int, float)):
            return (False, f"u_ergcm3 must be numeric, got {type(self.u)}")
        return (True, f"u = {self.u} erg/cm^3")

    def test_interface_I_ratio_exists(self) -> Tuple[bool, str]:
        if self.I_ratio is None:
            return (False, "Missing key 'I_ratio' in results")
        if not isinstance(self.I_ratio, (int, float)):
            return (False, f"I_ratio must be numeric, got {type(self.I_ratio)}")
        return (True, f"I_ratio = {self.I_ratio}")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_J_from_lines(self) -> Tuple[bool, str]:
        """Stern‑Gerlach: 2J+1 = 5 because five lines observed."""
        if self.J is None:
            return (False, "SKIP: J not available")
        expected_lines = 5
        computed_lines = 2 * self.J + 1
        if not math.isclose(computed_lines, expected_lines, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"2J+1 = {computed_lines}, expected {expected_lines}")
        return (True, f"2J+1 = {computed_lines} ≈ {expected_lines}")

    def test_invariant_mu_zero_for_J0(self) -> Tuple[bool, str]:
        """Magnetic moment vanishes when J=0 (^3P_0 state)."""
        if self.J is None or self.mu is None:
            return (False, "SKIP: J or mu not available")
        if self.J == 0:
            if not math.isclose(self.mu, 0.0, abs_tol=1e-6):
                return (False, f"mu should be 0 when J=0, got {self.mu}")
            return (True, "mu ≈ 0 for J=0")
        else:
            # This test is only for J=0 case; skip if J not 0
            return (True, "SKIP: J != 0, mu not required to be zero")

    def test_invariant_energy_density_formula(self) -> Tuple[bool, str]:
        """Blackbody energy density: u = (4σ/c) T^4."""
        if self.u is None:
            return (False, "SKIP: u not available")
        # Recompute from constants
        T = self.T_ROOM
        sigma = self.SIGMA
        c = self.C_CGS
        u_expected = (4.0 * sigma / c) * (T ** 4)
        if not math.isclose(self.u, u_expected, rel_tol=2e-1, abs_tol=1e-4):
            return (False, f"u = {self.u}, expected ≈ {u_expected:.4e} erg/cm^3")
        return (True, f"u = {self.u:.4e} ≈ {u_expected:.4e}")

    def test_invariant_intensity_ratio_from_weights(self) -> Tuple[bool, str]:
        """Intensity ratio I(2P1/2)/I(2P3/2) = (2J1+1)/(2J2+1) = 2/4 = 0.5."""
        if self.I_ratio is None:
            return (False, "SKIP: I_ratio not available")
        # J values for the two upper states
        J1 = 0.5   # 2P1/2
        J2 = 1.5   # 2P3/2
        weight1 = 2*J1 + 1   # 2
        weight2 = 2*J2 + 1   # 4
        expected_ratio = weight1 / weight2  # 0.5
        if not math.isclose(self.I_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"I_ratio = {self.I_ratio}, expected {expected_ratio}")
        return (True, f"I_ratio = {self.I_ratio} ≈ {expected_ratio}")

    # ------------------------------------------------------------------
    # Modeling / explanation quality checks
    # ------------------------------------------------------------------
    def test_modeling_explanation_includes_SternGerlach(self) -> Tuple[bool, str]:
        """Check that explanation_steps mentions Stern‑Gerlach for part (a)."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        steps_str = str(self.explanation_steps).lower()
        keywords = ["stern", "gerlach", "split", "beam"]
        found = sum(1 for kw in keywords if kw in steps_str)
        if found < 2:
            return (False, "Explanation lacks clear Stern‑Gerlach reasoning")
        return (True, "Explanation includes Stern‑Gerlach context")

    def test_modeling_explanation_includes_filled_shell(self) -> Tuple[bool, str]:
        """Check that explanation mentions filled valence shell for noble gases."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        steps_str = str(self.explanation_steps).lower()
        if "full" in steps_str or "filled" in steps_str or "complete shell" in steps_str:
            return (True, "Explanation addresses noble gas inertness via filled shell")
        return (False, "Noble gas reasoning missing filled-shell argument")

    def test_modeling_explanation_includes_helium_spin(self) -> Tuple[bool, str]:
        """Check that explanation mentions spin states (S=0,1) for helium."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps provided")
        steps_str = str(self.explanation_steps).lower()
        if "spin" in steps_str and ("singlet" in steps_str or "triplet" in steps_str):
            return (True, "Explanation discusses spin states in helium")
        return (False, "Helium term scheme explanation missing spin discussion")

    # ------------------------------------------------------------------
    # Monotonicity checks (each must perturb an input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_blackbody_energy_increases_with_T(self) -> Tuple[bool, str]:
        """Perturb T by +1% and +5% and verify u increases."""
        if self.u is None:
            return (False, "SKIP: u not available")
        # Baseline at T=300K
        T0 = self.T_ROOM
        u0 = self.u
        sigma = self.SIGMA
        c = self.C_CGS
        # Perturbations
        for factor, label in [(1.01, "+1%"), (1.05, "+5%")]:
            T_new = T0 * factor
            u_new = (4.0 * sigma / c) * (T_new ** 4)
            # Check monotonicity: predicted u_new > u0
            if u_new <= u0:
                return (False, f"u at {label} T ({T_new}) should be > u at {T0}, but predicted {u_new:.4e}")
        # Also check that the given u0 is consistent with the expected monotonic relation
        # Not required for monotonicity, but we can compare the ratio
        # (the given u0 should be close to the computed u0 from constants)
        u_computed_baseline = (4.0 * sigma / c) * (T0 ** 4)
        if not math.isclose(u0, u_computed_baseline, rel_tol=0.2):
            return (False, f"Given u0 = {u0} is not close to computed baseline {u_computed_baseline:.4e}")
        return (True, "Energy density increases with T as expected")

    # ------------------------------------------------------------------
    # Single weak answer sanity check (no re-derivation, no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Very broad sanity check covering sign, finiteness, and plausible orders.
        Does not compare against expected values.
        """
        issues = []
        # J: small positive integer
        if self.J is not None:
            if not isinstance(self.J, (int, float)):
                issues.append("J not numeric")
            elif self.J <= 0:
                issues.append("J non-positive")
            elif self.J > 10:
                issues.append("J > 10 (suspiciously large)")
        else:
            issues.append("J missing")

        # mu: can be zero; check it's a number
        if self.mu is not None:
            if not isinstance(self.mu, (int, float)):
                issues.append("mu not numeric")
            elif math.isnan(self.mu) or math.isinf(self.mu):
                issues.append("mu is NaN or infinite")
        else:
            issues.append("mu missing")

        # u: positive and on order of 10^-5 erg/cm^3 (weak range)
        if self.u is not None:
            if not isinstance(self.u, (int, float)):
                issues.append("u not numeric")
            elif self.u <= 0:
                issues.append("u non-positive")
            elif math.isnan(self.u) or math.isinf(self.u):
                issues.append("u is NaN or infinite")
            elif not (1e-7 <= self.u <= 1e-3):
                issues.append(f"u = {self.u} far from expected order (10^-5)")
        else:
            issues.append("u missing")

        # I_ratio: between 0 and 1
        if self.I_ratio is not None:
            if not isinstance(self.I_ratio, (int, float)):
                issues.append("I_ratio not numeric")
            elif self.I_ratio <= 0 or self.I_ratio >= 1:
                issues.append(f"I_ratio = {self.I_ratio} outside (0,1)")
        else:
            issues.append("I_ratio missing")

        if issues:
            return (False, "Weak sanity issues: " + "; ".join(issues))
        return (True, "All final answers pass basic sanity (positive, finite, plausible order)")

    # ------------------------------------------------------------------
    # Run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)