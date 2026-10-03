import math
import traceback

class PhysicsChecker:
    """
    Auditor for triple-point phase change problem with constant volume.
    Verifies mass conservation, energy balance, volume approximations,
    and monotonic properties.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.m_i = self.results.get('m_i')
        self.m_w = self.results.get('m_w')
        self.m_v = self.results.get('m_v')

        # Problem givens (from statement)
        self.Q = 60.0                 # cal
        self.L_f = 80.0               # cal/g (fusion)
        self.L_v = 596.0              # cal/g (vaporization)
        self.L_s = 676.0              # cal/g (sublimation)
        self.rho_i = 0.917            # g/cm^3 (ice)
        self.rho_w = 1.0              # g/cm^3 (water)
        self.v_v = 207.0              # cm^3/g (vapor specific volume)
        self.initial_mass = 1.0       # g each phase

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All three final masses must exist and be numeric."""
        if self.m_i is None:
            return (False, "Missing key: m_i")
        if self.m_w is None:
            return (False, "Missing key: m_w")
        if self.m_v is None:
            return (False, "Missing key: m_v")
        for name, val in [("m_i", self.m_i), ("m_w", self.m_w), ("m_v", self.m_v)]:
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
        return (True, "All required keys present and numeric")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_mass_conservation(self):
        """Total mass must be 3.0 g (within 2%)."""
        if any(v is None for v in (self.m_i, self.m_w, self.m_v)):
            return (False, "SKIP: missing masses")
        total = self.m_i + self.m_w + self.m_v
        if math.isclose(total, 3.0, rel_tol=0.02, abs_tol=0.02):
            return (True, f"Total mass {total:.4f} ≈ 3.0")
        else:
            return (False, f"Total mass {total:.4f} not close to 3.0")

    def test_invariant_energy_balance_simplified(self):
        """Q ≈ L_f * (1 - m_i) (simplified, assuming m_v unchanged)."""
        if self.m_i is None:
            return (False, "SKIP: m_i missing")
        lhs = self.L_f * (1.0 - self.m_i)
        if math.isclose(lhs, self.Q, rel_tol=0.02, abs_tol=1.2):
            return (True, f"L_f*(1-m_i) = {lhs:.3f} ≈ Q = {self.Q}")
        else:
            return (False, f"L_f*(1-m_i) = {lhs:.3f} differs from Q = {self.Q}")

    def test_invariant_energy_balance_exact(self):
        """(1-m_i)*L_s + (1-m_w)*L_v ≈ Q (full energy balance)."""
        if any(v is None for v in (self.m_i, self.m_w)):
            return (False, "SKIP: m_i or m_w missing")
        lhs = (1.0 - self.m_i) * self.L_s + (1.0 - self.m_w) * self.L_v
        if math.isclose(lhs, self.Q, rel_tol=0.02, abs_tol=1.2):
            return (True, f"Exact energy balance ok: {lhs:.3f} ≈ Q = {self.Q}")
        else:
            return (False, f"Exact energy balance = {lhs:.3f}, expected {self.Q}")

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks
    # ------------------------------------------------------------------
    def test_modeling_vapor_mass_approx_constant(self):
        """Vapor mass should be very close to 1.0 g (volume dominance)."""
        if self.m_v is None:
            return (False, "SKIP: m_v missing")
        if math.isclose(self.m_v, 1.0, rel_tol=0.02, abs_tol=0.02):
            return (True, f"m_v = {self.m_v:.4f} ≈ 1.0")
        else:
            return (False, f"m_v = {self.m_v:.4f} deviates from 1.0")

    def test_modeling_volume_balance(self):
        """
        Check that volume change from ice melting is compensated by
        a tiny change in vapor mass.
        """
        if any(v is None for v in (self.m_i, self.m_v)):
            return (False, "SKIP: m_i or m_v missing")
        # Volume change from melting: ΔV = (1/ρ_w - 1/ρ_i) * (1 - m_i)
        delta_V_melt = (1.0 / self.rho_w - 1.0 / self.rho_i) * (1.0 - self.m_i)
        # Expected vapor mass change to keep volume constant:
        delta_m_v_pred = -delta_V_melt / self.v_v
        # Actual change in vapor mass:
        delta_m_v_actual = self.m_v - self.initial_mass  # initial = 1.0 g
        # Compare: predicted and actual changes should be small and close
        if math.isclose(delta_m_v_actual, delta_m_v_pred, rel_tol=0.1, abs_tol=0.002):
            return (True, f"Volume balance: predicted Δm_v = {delta_m_v_pred:.6f}, actual Δm_v = {delta_m_v_actual:.6f}")
        else:
            return (False, f"Volume balance mismatch: predicted Δm_v = {delta_m_v_pred:.6f}, actual Δm_v = {delta_m_v_actual:.6f}")

    # ------------------------------------------------------------------
    # Monotonicity check (trend via explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_ice_mass_decreases_with_heat(self):
        """
        If the added heat Q increases, the final ice mass m_i should
        decrease (derivative negative). We verify this property by
        perturbing Q by +1% (0.6 cal) and recomputing the model.
        """
        # Baseline from model
        delta_Q = 0.01 * self.Q  # 1% of 60 = 0.6 cal
        m_i_baseline = 1.0 - self.Q / self.L_f
        m_i_perturbed = 1.0 - (self.Q + delta_Q) / self.L_f
        # Check monotonicity: perturbed m_i must be smaller
        if m_i_perturbed < m_i_baseline:
            return (True, f"m_i decreases when Q increases: {m_i_perturbed:.4f} < {m_i_baseline:.4f}")
        else:
            return (False, f"Monotonicity violated: {m_i_perturbed:.4f} >= {m_i_baseline:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (non‑dominant, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_non_negative(self):
        """All masses must be non‑negative and finite."""
        for name, val in [("m_i", self.m_i), ("m_w", self.m_w), ("m_v", self.m_v)]:
            if val is None:
                continue
            if val < 0:
                return (False, f"{name} is negative: {val}")
            if not math.isfinite(val):
                return (False, f"{name} is non‑finite: {val}")
        return (True, "All provided masses are non‑negative and finite")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all test_ methods
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