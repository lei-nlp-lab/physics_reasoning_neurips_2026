import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for solid state physics problem: specific heat and mean squared displacement.
    """

    # Problem constants (given data)
    TEMPS = [1000, 500, 40, 20]        # K
    CVALS = [20, 20, 8, 1]            # arbitrary units

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract results dict
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        
        # Required outputs from solver
        self.material: Optional[str] = self.results.get('material', None)
        self.scaling_high: Optional[str] = self.results.get('scaling_high', None)
        self.scaling_low: Optional[str] = self.results.get('scaling_low', None)
        self.avg_U2_zero: Any = self.results.get('avg_U2_zero', None)

    # ---------- Interface checks (keys exist and have valid types) ----------

    def test_interface_material_exists(self) -> Tuple[bool, str]:
        """Check that 'material' key exists and is a string."""
        if self.material is None:
            return False, "material key missing"
        if not isinstance(self.material, str):
            return False, f"material is not a string, got {type(self.material)}"
        return True, "material key exists and is string"

    def test_interface_scaling_high_exists(self) -> Tuple[bool, str]:
        """Check that 'scaling_high' key exists and is a string."""
        if self.scaling_high is None:
            return False, "scaling_high key missing"
        if not isinstance(self.scaling_high, str):
            return False, f"scaling_high is not a string, got {type(self.scaling_high)}"
        return True, "scaling_high key exists and is string"

    def test_interface_scaling_low_exists(self) -> Tuple[bool, str]:
        """Check that 'scaling_low' key exists and is a string."""
        if self.scaling_low is None:
            return False, "scaling_low key missing"
        if not isinstance(self.scaling_low, str):
            return False, f"scaling_low is not a string, got {type(self.scaling_low)}"
        return True, "scaling_low key exists and is string"

    def test_interface_avg_U2_zero_exists(self) -> Tuple[bool, str]:
        """Check that 'avg_U2_zero' key exists."""
        if self.avg_U2_zero is None:
            return False, "avg_U2_zero key missing"
        return True, "avg_U2_zero key exists"

    # ---------- Invariant checks (theorem-based physical sanity) ----------

    def test_invariant_Cv_T3_scaling(self) -> Tuple[bool, str]:
        """Low-temperature C_v data should follow Debye T^3 law."""
        if len(self.CVALS) < 4:
            return False, "Insufficient C_v data"
        # T=20K, Cv=1; T=40K, Cv=8
        c20 = 1.0
        c40 = 8.0
        ratio = c40 / c20
        expected_ratio = (40.0 / 20.0) ** 3
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"C_v ratio {ratio} matches T^3 ratio {expected_ratio}"
        return False, f"C_v ratio {ratio} does not match T^3 ratio {expected_ratio}"

    def test_invariant_Cv_constant_high_T(self) -> Tuple[bool, str]:
        """High-temperature C_v should be constant (Dulong-Petit)."""
        c_high = [20.0, 20.0]  # from 500K and 1000K
        if not math.isclose(c_high[0], c_high[1], rel_tol=1e-2, abs_tol=1e-2):
            return False, f"C_v at high T not constant: {c_high} are not equal"
        return True, "C_v at high T is constant (500K and 1000K both 20)"

    def test_invariant_zero_point_expression(self) -> Tuple[bool, str]:
        """
        Re-derive zero-point ⟨U²⟩ from integral at T=0:
        I(0) = ∫_0^{ℏω_D} ε dε = (ℏω_D)^2/2. Multiply by constant C.
        Final expression: (9ℏ)/(4M ω_D).
        Check that solver's avg_U2_zero matches this derived form.
        """
        if self.avg_U2_zero is None:
            return False, "avg_U2_zero is None"

        ℏ, M, ω_D = sp.symbols('ℏ M ω_D', positive=True)
        # Derive zero-point value from theorem
        C = 9 * ℏ**2 / (2 * M * (ℏ * ω_D)**3)
        zero_integral = (ℏ * ω_D)**2 / 2
        derived_zero = sp.simplify(C * zero_integral)  # = 9*ℏ/(4*M*ω_D)

        # Parse solver's expression
        solver_str = self.avg_U2_zero
        if isinstance(solver_str, str):
            try:
                solver_expr = sp.sympify(solver_str)
            except (sp.SympifyError, TypeError):
                return False, f"Cannot sympify avg_U2_zero string: '{solver_str}'"
        else:
            solver_expr = solver_str

        diff = sp.simplify(solver_expr - derived_zero)
        if diff == 0:
            return True, "Zero-point expression matches derived value 9ℏ/(4M ω_D)"
        else:
            return False, f"Zero-point expression difference: {diff}"


    # ---------- Modeling checks (quality of derivation / expression) ----------

    def test_modeling_high_T_scaling(self) -> Tuple[bool, str]:
        """
        High-T scaling of ⟨U²⟩ should be linear in T.
        Uses symbolic re-derivation of the integral with high-T expansion.
        Then checks that scaling_high mentions temperature T.
        """
        T, kB, ℏ, ω_D, ε = sp.symbols('T kB ℏ ω_D ε', positive=True)
        n = 1 / (sp.exp(ε/(kB*T)) - 1)
        # High T expansion: n ≈ kB*T/ε
        n_approx = kB*T/ε
        I_approx = sp.integrate(ε * (1 + 2*n_approx), (ε, 0, ℏ*ω_D))
        coeff_T = sp.simplify(sp.expand(I_approx).coeff(T, 1))
        if coeff_T == 0:
            return False, f"High T expansion gives no linear term in T; I_approx = {I_approx}"
        if self.scaling_high is None:
            return False, "scaling_high is missing"
        if 'T' not in self.scaling_high:
            return False, f"scaling_high '{self.scaling_high}' does not mention T"
        return True, f"High-T scaling linear in T confirmed (coefficient {coeff_T}) and scaling_high mentions T"

    def test_modeling_low_T_scaling(self) -> Tuple[bool, str]:
        """
        Low-T scaling of ⟨U²⟩ should be constant + T².
        Uses symbolic re-derivation of the integral with low-T behavior.
        """
        T, kB, ℏ, ω_D, ε = sp.symbols('T kB ℏ ω_D ε', positive=True)
        n = 1 / (sp.exp(ε/(kB*T)) - 1)
        # Low T: use standard integral ∫ x/(e^x-1) dx from 0 to ∞ = π²/6
        x = sp.symbols('x', positive=True)
        std_int = sp.integrate(x/(sp.exp(x)-1), (x, 0, sp.oo))
        # I(T) = (ℏω_D)^2/2 + 2(kBT)^2 * ∫_0^{∞} x/(e^x-1) dx  (upper bound approximated to infinity)
        I_low_approx = (ℏ*ω_D)**2 / 2 + 2 * (kB*T)**2 * std_int
        expanded = sp.simplify(sp.expand(I_low_approx))
        coeff_T2 = sp.simplify(expanded.coeff(T, 2))
        if coeff_T2 == 0:
            return False, f"Low T expansion gives no T^2 term; I_low_approx = {expanded}"
        if self.scaling_low is None:
            return False, "scaling_low is missing"
        if 'constant' not in self.scaling_low or 'T^2' not in self.scaling_low:
            return False, f"scaling_low '{self.scaling_low}' should contain 'constant' and 'T^2'"
        return True, f"Low-T scaling constant+T^2 confirmed (coefficient of T^2 is {coeff_T2})"

    # ---------- Monotonicity checks (trend via perturbation) ----------

    def test_monotonicity_Cv_low_T_increases_with_T(self) -> Tuple[bool, str]:
        """
        For low temperatures, C_v increases with T (T^3 law).
        Perturb T=20K by +10% and verify estimated C_v increases.
        """
        T_base = 20.0
        C_base = 1.0
        T_pert = T_base * 1.10
        # Using T^3 law: C_v ∝ T^3
        C_pert_estimated = C_base * (T_pert / T_base)**3
        if C_pert_estimated <= C_base:
            return False, f"Estimated C_v at perturbed T {T_pert} is {C_pert_estimated} <= {C_base}"
        # Also check that the given data indicates increase from 20K to 40K
        if not (self.CVALS[3] < self.CVALS[2]):  # 1 < 8
            return False, "C_v at 40K must be larger than at 20K"
        return True, f"Low-T C_v monotonic increasing: base T={T_base} C={C_base}, perturbed T={T_pert} estimated C={C_pert_estimated:.3f}"

    # ---------- Weak answer sanity checks (only one allowed) ----------

    def test_answer_sanity_plausibility(self) -> Tuple[bool, str]:
        """
        Very weak sanity: all required result keys are present, non-empty, and have plausible types.
        No re-derivation of answers or comparison against literal expected results.
        """
        checks = []
        if self.material is None or not isinstance(self.material, str) or len(self.material)==0:
            checks.append("material missing or empty")
        if self.scaling_high is None or not isinstance(self.scaling_high, str) or len(self.scaling_high)==0:
            checks.append("scaling_high missing or empty")
        if self.scaling_low is None or not isinstance(self.scaling_low, str) or len(self.scaling_low)==0:
            checks.append("scaling_low missing or empty")
        if self.avg_U2_zero is None:
            checks.append("avg_U2_zero missing")
        if checks:
            return False, "; ".join(checks)
        return True, "All required keys present and non-empty"

    # ---------- Runner ----------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, return (overall_pass, logs)."""
        logs: List[str] = []
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