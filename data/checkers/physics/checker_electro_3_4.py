import math
import sys
from typing import Any, Optional, Tuple, List

class PhysicsChecker:
    """
    Checker for the capacitor-wire melting problem.
    Extracts problem data from solution_instance and performs physical audits.
    """

    def __init__(self, solution_instance: Any):
        """Extract all relevant data with safe None defaults."""
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Outputs from solution
        self.E_cap = self.results.get('E_cap', None)
        self.Q_heat = self.results.get('Q_heat', None)
        self.Q_melt = self.results.get('Q_melt', None)
        self.E_needed = self.results.get('E_needed', None)
        self.final_answer = getattr(solution_instance, 'final_answer', None)
        if self.final_answer is None:
            self.final_answer = self.results.get('final_answer', None)

        # Input parameters (givens) – try results, then instance attributes
        # Expected from problem statement
        self.C_each = self._safe_get('C_each', 1e-6)          # 1 µF
        self.N = self._safe_get('N', 4)                       # number of capacitors
        self.V = self._safe_get('V', 200.0)                   # charging voltage
        self.L_wire = self._safe_get('L_wire', 5e-3)          # 5 mm
        self.R_per_m = self._safe_get('R_per_m', 4.0)         # 4 Ω/m
        self.mass_per_m = self._safe_get('mass_per_m', 0.045e-3)  # 0.045 g/m → kg/m

        # Material properties (may be given or assumed)
        self.c = self._safe_get('c', 385.0)                   # J/(kg·K)
        self.T0 = self._safe_get('T0', 20.0)                  # initial temp °C
        self.Tm = self._safe_get('Tm', 1085.0)                # melting point °C
        self.Lf = self._safe_get('Lf', 205e3)                 # latent heat J/kg

        # Derived quantities (computed inside checker for verification)
        self.C_total = None
        if self.C_each is not None and self.N is not None:
            self.C_total = self.N * self.C_each

        self.m_wire = None
        if self.L_wire is not None and self.mass_per_m is not None:
            self.m_wire = self.L_wire * self.mass_per_m

        # Explanation steps may contain alternate values – we use as reference only
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    def _safe_get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Try to get value from results or instance attributes."""
        if key in self.results:
            return self.results[key]
        return getattr(self, key, default)

    # ------------------------------------------------------------------
    # Interface tests – check that required result keys exist
    # ------------------------------------------------------------------
    def test_interface_E_cap_present(self) -> Tuple[bool, str]:
        if self.E_cap is None:
            return (False, "Missing E_cap in results")
        return (True, "E_cap is present")

    def test_interface_Q_heat_present(self) -> Tuple[bool, str]:
        if self.Q_heat is None:
            return (False, "Missing Q_heat in results")
        return (True, "Q_heat is present")

    def test_interface_Q_melt_present(self) -> Tuple[bool, str]:
        if self.Q_melt is None:
            return (False, "Missing Q_melt in results")
        return (True, "Q_melt is present")

    def test_interface_E_needed_present(self) -> Tuple[bool, str]:
        if self.E_needed is None:
            return (False, "Missing E_needed in results")
        return (True, "E_needed is present")

    def test_interface_C_total_valid(self) -> Tuple[bool, str]:
        if self.C_total is None:
            return (False, "Cannot compute C_total – missing C_each or N")
        expected = 4e-6  # from problem: 4 × 1 µF
        if not math.isclose(self.C_total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_total = {self.C_total}, expected approx {expected}")
        return (True, f"C_total = {self.C_total} is correct")

    def test_interface_wire_mass_valid(self) -> Tuple[bool, str]:
        if self.m_wire is None:
            return (False, "Cannot compute wire mass – missing L_wire or mass_per_m")
        expected = 2.25e-7  # 0.005 m * 0.045e-3 kg/m
        if not math.isclose(self.m_wire, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Wire mass = {self.m_wire}, expected approx {expected}")
        return (True, f"Wire mass = {self.m_wire} is correct")

    # ------------------------------------------------------------------
    # Invariant tests – physical laws that must hold
    # ------------------------------------------------------------------
    def test_invariant_energy_non_negative(self) -> Tuple[bool, str]:
        """All energies and heats must be non-negative."""
        for name, val in [("E_cap", self.E_cap), ("Q_heat", self.Q_heat),
                          ("Q_melt", self.Q_melt), ("E_needed", self.E_needed)]:
            if val is not None and val < 0:
                return (False, f"{name} is negative: {val}")
        return (True, "All energies are non-negative")

    def test_invariant_Q_heat_positive(self) -> Tuple[bool, str]:
        if self.Q_heat is not None and self.Q_heat <= 0:
            return (False, f"Q_heat must be positive, got {self.Q_heat}")
        return (True, "Q_heat is positive (or not available)")

    def test_invariant_Q_melt_positive(self) -> Tuple[bool, str]:
        if self.Q_melt is not None and self.Q_melt <= 0:
            return (False, f"Q_melt must be positive, got {self.Q_melt}")
        return (True, "Q_melt is positive (or not available)")

    def test_invariant_E_cap_formula(self) -> Tuple[bool, str]:
        """Stored energy must be 0.5*C_total*V^2."""
        if self.E_cap is None or self.C_total is None or self.V is None:
            return (False, "SKIP: missing data for E_cap formula check")
        expected = 0.5 * self.C_total * self.V**2
        if not math.isclose(self.E_cap, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_cap = {self.E_cap}, expected approx {expected}")
        return (True, f"E_cap = {self.E_cap} matches formula")

    def test_invariant_E_needed_equals_sum(self) -> Tuple[bool, str]:
        """E_needed should be approximately Q_heat + Q_melt."""
        if any(x is None for x in [self.E_needed, self.Q_heat, self.Q_melt]):
            return (False, "SKIP: missing data for E_needed sum check")
        expected_sum = self.Q_heat + self.Q_melt
        if not math.isclose(self.E_needed, expected_sum, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_needed = {self.E_needed}, expected sum approx {expected_sum}")
        return (True, "E_needed equals Q_heat + Q_melt within tolerance")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check consistency with problem parameters
    # ------------------------------------------------------------------
    def test_modeling_Q_heat_from_formula(self) -> Tuple[bool, str]:
        """Check Q_heat derived from m, c, dT."""
        if any(x is None for x in [self.Q_heat, self.m_wire, self.c, self.Tm, self.T0]):
            return (False, "SKIP: missing data to recompute Q_heat")
        dT = self.Tm - self.T0
        expected = self.m_wire * self.c * dT
        if not math.isclose(self.Q_heat, expected, rel_tol=1e-1, abs_tol=1e-2):
            # large tolerance because constants may vary (e.g., T0=20 vs 25, Tm=1085 vs 1356)
            return (False, f"Q_heat = {self.Q_heat}, derived expected approx {expected} (with dT={dT})")
        return (True, f"Q_heat = {self.Q_heat} consistent with formula")

    def test_modeling_Q_melt_from_formula(self) -> Tuple[bool, str]:
        """Check Q_melt = m * Lf."""
        if any(x is None for x in [self.Q_melt, self.m_wire, self.Lf]):
            return (False, "SKIP: missing data to recompute Q_melt")
        expected = self.m_wire * self.Lf
        if not math.isclose(self.Q_melt, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q_melt = {self.Q_melt}, expected approx {expected}")
        return (True, f"Q_melt = {self.Q_melt} consistent with formula")

    def test_modeling_energy_conservation_judgment(self) -> Tuple[bool, str]:
        """Based on energies, the wire should not melt (E_cap < E_needed)."""
        if self.E_cap is None or self.E_needed is None:
            return (False, "SKIP: need E_cap and E_needed to judge melting")
        # The proper physical conclusion: if E_cap < E_needed, wire does not melt.
        if self.E_cap >= self.E_needed:
            return (False, f"E_cap ({self.E_cap}) >= E_needed ({self.E_needed}), which would imply melting, but physics says not enough energy")
        return (True, f"E_cap ({self.E_cap}) < E_needed ({self.E_needed}) → wire does not melt")

    def test_modeling_answer_consistency(self) -> Tuple[bool, str]:
        """Check that final answer (if present) says wire does not melt."""
        if self.final_answer is None:
            return (False, "No final answer provided")
        # Accept any string that indicates no melting.
        answer_str = str(self.final_answer).lower()
        if "not melt" in answer_str or "will not melt" in answer_str:
            return (True, "Final answer correctly states wire does not melt")
        return (False, f"Final answer does not indicate no melting: '{self.final_answer}'")

    # ------------------------------------------------------------------
    # Sanity test (weak) – only basic checks, no re-derivation
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """Simple sanity: energies should be in reasonable range (0.01-0.2 J)."""
        checks = []
        for name, val in [("E_cap", self.E_cap), ("Q_heat", self.Q_heat),
                          ("Q_melt", self.Q_melt), ("E_needed", self.E_needed)]:
            if val is not None and not (1e-4 < val < 1.0):
                checks.append(f"{name} = {val} outside plausible range")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All energies within plausible range (0.0001–1 J)")

    # ------------------------------------------------------------------
    # Monotonicity tests – trend behaviour
    # ------------------------------------------------------------------
    def test_monotonicity_increase_voltage_increases_Ecap(self) -> Tuple[bool, str]:
        """Perturb voltage +10%, stored energy should increase."""
        if self.E_cap is None or self.V is None or self.C_total is None:
            return (False, "SKIP: missing data for monotonicity test")
        # Use the relation E = 0.5*C*V^2 – perturbation analysis
        V_perturbed = self.V * 1.10
        E_perturbed = 0.5 * self.C_total * V_perturbed**2
        if E_perturbed <= self.E_cap:
            return (False, "Increasing voltage did not increase stored energy (should be monotonic)")
        return (True, f"Increasing V to {V_perturbed} increases E_cap to {E_perturbed:.4f}")

    def test_monotonicity_increase_wire_mass_increases_Qheat_and_Qmelt(self) -> Tuple[bool, str]:
        """Perturb L_wire +10%, Q_heat and Q_melt should increase."""
        if any(x is None for x in [self.Q_heat, self.Q_melt, self.L_wire, self.mass_per_m,
                                   self.c, self.Tm, self.T0, self.Lf]):
            return (False, "SKIP: missing data for monotonicity test")
        L_perturbed = self.L_wire * 1.10
        m_perturbed = L_perturbed * self.mass_per_m
        dT = self.Tm - self.T0
        Qheat_pert = m_perturbed * self.c * dT
        Qmelt_pert = m_perturbed * self.Lf
        if Qheat_pert <= self.Q_heat or Qmelt_pert <= self.Q_melt:
            return (False, "Increasing wire length did not increase required heats")
        return (True, "Increasing wire length increases Q_heat and Q_melt as expected")

    # ------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)