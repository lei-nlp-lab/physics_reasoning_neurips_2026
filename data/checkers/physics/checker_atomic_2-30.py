import math
import traceback
from typing import Tuple

class PhysicsChecker:
    """
    Auditor for a nuclear fission problem covering:
    - Binding energy per nucleon curve vs A
    - Energy release Q
    - Fission barrier condition
    """

    def __init__(self, solution_instance):
        # Extract results safely, default to None for missing keys.
        res = getattr(solution_instance, 'results', {})
        self.A_opt = res.get('A_opt', None)
        self.epsilon_max = res.get('epsilon_max', None)
        self.Q = res.get('Q', None)
        self.x_iron = res.get('x_iron', None)
        self.x_lead = res.get('x_lead', None)

        # Liquid-drop constants (from pseudocode, used for verification)
        self.a_V = 15.75   # MeV
        self.a_S = 17.8    # MeV
        self.a_C = 0.711   # MeV
        self.a_sym = 23.7  # MeV

    # ------------------------------------------------------------------
    # interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        required = ['A_opt', 'epsilon_max', 'Q', 'x_iron', 'x_lead']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required keys present")

    def test_interface_key_types_and_signs(self) -> Tuple[bool, str]:
        values = {
            'A_opt': self.A_opt,
            'epsilon_max': self.epsilon_max,
            'Q': self.Q,
            'x_iron': self.x_iron,
            'x_lead': self.x_lead,
        }
        for name, val in values.items():
            if val is None:
                return (False, f"SKIP: {name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
            if val < 0:
                return (False, f"{name} is negative")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
        return (True, "All key types and signs valid")

    # ------------------------------------------------------------------
    # invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_A_opt_peak_near_56(self) -> Tuple[bool, str]:
        if self.A_opt is None:
            return (False, "SKIP: A_opt not available")
        # Expect A_opt ~ 56 (iron group)
        if not math.isclose(self.A_opt, 56, abs_tol=5):
            return (False, f"A_opt={self.A_opt} not near 56")
        return (True, f"A_opt={self.A_opt} is plausible near iron peak")

    def test_invariant_epsilon_max_positive_and_reasonable(self) -> Tuple[bool, str]:
        if self.epsilon_max is None:
            return (False, "SKIP: epsilon_max not available")
        # Typical maximum ~8.8 MeV, allow broad range 7-9.5 MeV
        if not (7.0 <= self.epsilon_max <= 9.5):
            return (False, f"epsilon_max={self.epsilon_max} out of plausible range [7,9.5] MeV")
        return (True, f"epsilon_max={self.epsilon_max} MeV is plausible")

    def test_invariant_Q_positive(self) -> Tuple[bool, str]:
        if self.Q is None:
            return (False, "SKIP: Q not available")
        if self.Q <= 0:
            return (False, f"Q={self.Q} is not positive")
        return (True, f"Q={self.Q} > 0")

    def test_invariant_fission_parameter_iron_below_one(self) -> Tuple[bool, str]:
        if self.x_iron is None:
            return (False, "SKIP: x_iron not available")
        if self.x_iron >= 1.0:
            return (False, f"x_iron={self.x_iron} >= 1, should be <1 for stable fission barrier")
        return (True, f"x_iron={self.x_iron} < 1, indicates barrier exists")

    def test_invariant_fission_parameter_lead_below_one(self) -> Tuple[bool, str]:
        if self.x_lead is None:
            return (False, "SKIP: x_lead not available")
        if self.x_lead >= 1.0:
            return (False, f"x_lead={self.x_lead} >= 1, should be <1 for stable fission barrier")
        return (True, f"x_lead={self.x_lead} < 1, indicates barrier exists")

    # ------------------------------------------------------------------
    # modeling / trace tests (use isclose, rederive expected values if possible)
    # ------------------------------------------------------------------
    def test_modeling_binding_energy_at_iron(self) -> Tuple[bool, str]:
        """Re‑derive epsilon for A=56, Z~26 and compare to epsilon_max."""
        if self.epsilon_max is None:
            return (False, "SKIP: epsilon_max not available")
        # compute liquid‑drop epsilon for iron using standard constants
        A, Z = 56, 26
        B = (self.a_V*A - self.a_S*A**(2/3) - self.a_C*Z**2/A**(1/3)
             - self.a_sym*(A-2*Z)**2/A)
        eps_computed = B / A
        if not math.isclose(eps_computed, self.epsilon_max, rel_tol=1e-2, abs_tol=2.0):
            return (False, f"epsilon_max={self.epsilon_max} not consistent with computed iron binding {eps_computed:.2f} MeV")
        return (True, f"epsilon_max={self.epsilon_max} matches computed iron binding ~{eps_computed:.2f} MeV")

    def test_modeling_Q_symmetric_fission(self) -> Tuple[bool, str]:
        """Check Q ~200 MeV using symmetric fission approximation."""
        if self.Q is None or self.A_opt is None:
            return (False, "SKIP: Q or A_opt not available for Q check")
        # Use 235U symmetric fission as typical example (A=235, Z=92)
        A_parent, Z_parent = 235, 92
        A_frag = A_parent / 2
        Z_frag = Z_parent / 2
        # binding energy for parent
        B_parent = (self.a_V*A_parent - self.a_S*A_parent**(2/3) - self.a_C*Z_parent**2/A_parent**(1/3)
                    - self.a_sym*(A_parent-2*Z_parent)**2/A_parent)
        eps_parent = B_parent / A_parent
        # binding per fragment (using same formula)
        B_frag = (self.a_V*A_frag - self.a_S*A_frag**(2/3) - self.a_C*Z_frag**2/A_frag**(1/3)
                  - self.a_sym*(A_frag-2*Z_frag)**2/A_frag)
        eps_frag = B_frag / A_frag
        Q_computed = 2 * A_frag * eps_frag - A_parent * eps_parent
        if not math.isclose(Q_computed, self.Q, rel_tol=5e-2, abs_tol=50):
            return (False, f"Q={self.Q} not consistent with computed ~{Q_computed:.0f} MeV")
        return (True, f"Q={self.Q} is consistent with symmetric fission of 235U (~{Q_computed:.0f} MeV)")

    def test_modeling_fission_parameter_formula(self) -> Tuple[bool, str]:
        """Check that x_iron and x_lead roughly match Z^2/A / (2*a_S/a_C)."""
        if self.x_iron is None or self.x_lead is None:
            return (False, "SKIP: x_iron or x_lead not available")
        # compute expected x for iron (A=56, Z=26) and lead (A=208, Z=82)
        crit = 2 * self.a_S / self.a_C  # ~50.1
        iron_Z2_A = 26**2 / 56.0
        lead_Z2_A = 82**2 / 208.0
        expected_x_iron = iron_Z2_A / crit
        expected_x_lead = lead_Z2_A / crit
        if not math.isclose(self.x_iron, expected_x_iron, rel_tol=2e-2, abs_tol=0.05):
            return (False, f"x_iron={self.x_iron} != expected {expected_x_iron:.3f}")
        if not math.isclose(self.x_lead, expected_x_lead, rel_tol=2e-2, abs_tol=0.05):
            return (False, f"x_lead={self.x_lead} != expected {expected_x_lead:.3f}")
        return (True, "x_iron and x_lead match Z^2/A / (2a_S/a_C)")

    # ------------------------------------------------------------------
    # monotonicity tests  (must perturb an input and compare to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_fission_parameter_decreases_with_increasing_A(self) -> Tuple[bool, str]:
        """
        For a fixed Z (82, lead), increasing A should decrease the fission parameter x.
        Baseline: use x_lead from solution (A=208, Z=82).
        Perturb: increase A to 218 (by 10), recompute x with same Z.
        Check that perturbed x < baseline.
        """
        if self.x_lead is None:
            return (False, "SKIP: x_lead not available (needed as baseline)")
        crit = 2 * self.a_S / self.a_C  # ~50.1
        # Baseline x_lead corresponds to A=208, Z=82.
        # Compute perturbed x for A=218, Z=82.
        A_pert = 218
        Z = 82
        x_pert = (Z**2 / A_pert) / crit
        if x_pert >= self.x_lead:
            return (False, f"Perturbed x={x_pert:.4f} (A=218) is not less than baseline x_lead={self.x_lead:.4f}")
        return (True, f"Perturbed x={x_pert:.4f} < baseline x_lead={self.x_lead:.4f}, monotonicity verified")

    # ------------------------------------------------------------------
    # weak answer sanity (only one allowed, no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """Check that all numerical answer fields are finite and non‑negative."""
        fields = {'A_opt': self.A_opt, 'epsilon_max': self.epsilon_max,
                  'Q': self.Q, 'x_iron': self.x_iron, 'x_lead': self.x_lead}
        for name, val in fields.items():
            if val is None:
                continue  # already handled by other tests
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
            if val < 0:
                return (False, f"{name} is negative")
        return (True, "All listed answer fields are finite and non‑negative")

    # ------------------------------------------------------------------
    # main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
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