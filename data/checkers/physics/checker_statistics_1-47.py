import math
import traceback

class PhysicsChecker:
    """
    Auditor for entropy difference problem: nitrogen gas vs liquid at 1 atm.
    """
    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract givens from results or from solution_instance attributes
        # Assume solution_instance may have attributes: given_m, given_M, given_L, given_Cp,
        # given_T_gas_C, given_T_boil_C, given_P
        # If not, fall back to None.
        self.m = getattr(solution_instance, 'given_m', None) or self.results.get('m', None)
        self.M = getattr(solution_instance, 'given_M', None) or self.results.get('M', None)
        self.L = getattr(solution_instance, 'given_L', None) or self.results.get('L', None)
        self.Cp = getattr(solution_instance, 'given_Cp', None) or self.results.get('Cp', None)
        self.T_gas_C = getattr(solution_instance, 'given_T_gas_C', None) or self.results.get('T_gas_C', None)
        self.T_boil_C = getattr(solution_instance, 'given_T_boil_C', None) or self.results.get('T_boil_C', None)
        self.P = getattr(solution_instance, 'given_P', None) or self.results.get('P', None)

        # Extract emitted result
        self.delta_S_total = self.results.get('delta_S_total', None)

        # Optional: extract intermediate results if present
        self.delta_S_vap = self.results.get('delta_S_vap', None)
        self.delta_S_heat = self.results.get('delta_S_heat', None)
        self.n = self.results.get('n', None)

        # Compute derived values for checks (only if givens are available)
        self.T_boil = None
        self.T_gas = None
        self.n_calc = None
        if self.T_boil_C is not None:
            self.T_boil = self.T_boil_C + 273.0
        if self.T_gas_C is not None:
            self.T_gas = self.T_gas_C + 273.0
        if self.m is not None and self.M is not None and self.M != 0:
            self.n_calc = self.m / self.M

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_delta_S_total_exists(self):
        """Check that the required result key 'delta_S_total' exists."""
        if 'delta_S_total' not in self.results:
            return False, "missing key 'delta_S_total' in results"
        return True, "key present"

    def test_interface_delta_S_total_is_number(self):
        """Check that delta_S_total is a finite number."""
        if self.delta_S_total is None:
            return False, "delta_S_total is None"
        if not isinstance(self.delta_S_total, (int, float)):
            return False, f"delta_S_total has type {type(self.delta_S_total)}"
        if math.isnan(self.delta_S_total) or math.isinf(self.delta_S_total):
            return False, "delta_S_total is NaN or Inf"
        return True, f"delta_S_total = {self.delta_S_total}"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use isclose)
    # ------------------------------------------------------------------
    def test_invariant_entropy_decomposition(self):
        """
        Total entropy change equals sum of vaporization and heating contributions.
        ΔS_total = ΔS_vap + ΔS_heat
        """
        if self.delta_S_total is None:
            return False, "cannot check decomposition: delta_S_total missing"
        # If intermediate values are available, use them; else we must compute them
        vap = self.delta_S_vap
        heat = self.delta_S_heat
        if vap is None and self.m is not None and self.L is not None and self.T_boil is not None:
            vap = self.m * self.L / self.T_boil
        if heat is None and self.n_calc is not None and self.Cp is not None and self.T_gas is not None and self.T_boil is not None:
            heat = self.n_calc * self.Cp * math.log(self.T_gas / self.T_boil)
        if vap is None or heat is None:
            return False, "cannot compute contributions: insufficient givens"
        expected_total = vap + heat
        if math.isclose(self.delta_S_total, expected_total, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ΔS_total = {self.delta_S_total} ≈ {expected_total}"
        else:
            return False, f"ΔS_total {self.delta_S_total} != {expected_total}"

    def test_invariant_delta_S_vap_formula(self):
        """ΔS_vap = mL/T_boil (if ΔS_vap is available)."""
        if self.delta_S_vap is None:
            return False, "cannot check ΔS_vap: not available"
        if self.m is None or self.L is None or self.T_boil is None:
            return False, "cannot compute expected: missing givens"
        expected = self.m * self.L / self.T_boil
        if math.isclose(self.delta_S_vap, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ΔS_vap = {self.delta_S_vap} ≈ {expected}"
        else:
            return False, f"ΔS_vap {self.delta_S_vap} != {expected}"

    def test_invariant_delta_S_heat_formula(self):
        """ΔS_heat = n Cp ln(T_gas/T_boil)."""
        if self.delta_S_heat is None:
            return False, "cannot check ΔS_heat: not available"
        if self.n_calc is None or self.Cp is None or self.T_gas is None or self.T_boil is None:
            return False, "cannot compute expected: missing givens"
        expected = self.n_calc * self.Cp * math.log(self.T_gas / self.T_boil)
        if math.isclose(self.delta_S_heat, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ΔS_heat = {self.delta_S_heat} ≈ {expected}"
        else:
            return False, f"ΔS_heat {self.delta_S_heat} != {expected}"

    def test_invariant_moles(self):
        """n = m/M."""
        if self.n is not None and self.n_calc is not None:
            if math.isclose(self.n, self.n_calc, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"n = {self.n} ≈ {self.n_calc}"
            else:
                return False, f"n {self.n} != {self.n_calc} from m/M"
        return False, "cannot check moles: missing n or givens"

    def test_invariant_temperature_conversions(self):
        """
        Check that temperatures in Kelvin are correctly computed from Celsius.
        Only if both T_boil_C and T_gas_C are available.
        """
        if self.T_boil_C is None or self.T_gas_C is None:
            return False, "cannot check conversions: missing Celsius temperatures"
        if self.T_boil is None or self.T_gas is None:
            return False, "Kelvin values missing"
        # Compare to expected
        if math.isclose(self.T_boil, self.T_boil_C + 273.0, rel_tol=1e-2, abs_tol=1e-2) and \
           math.isclose(self.T_gas, self.T_gas_C + 273.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, "temperature conversions correct"
        else:
            return False, "temperature conversion error"

    # ------------------------------------------------------------------
    # Modeling quality tests (using explanation_steps as guide)
    # ------------------------------------------------------------------
    def test_modeling_entropy_increases_with_heating(self):
        """At constant pressure, heating from T_boil to T_gas increases entropy."""
        if self.delta_S_heat is None:
            return False, "cannot check: ΔS_heat missing"
        if self.delta_S_heat <= 0:
            return False, f"ΔS_heat = {self.delta_S_heat} is not positive (expected increase)"
        return True, f"ΔS_heat = {self.delta_S_heat} > 0"

    def test_modeling_entropy_increases_with_vaporization(self):
        """Vaporization (phase change) increases entropy."""
        if self.delta_S_vap is None:
            return False, "cannot check: ΔS_vap missing"
        if self.delta_S_vap <= 0:
            return False, f"ΔS_vap = {self.delta_S_vap} is not positive"
        return True, f"ΔS_vap = {self.delta_S_vap} > 0"

    def test_modeling_pressure_constant(self):
        """Pressure is constant at 1 atm (check if key exists or value is plausible)."""
        if self.P is None:
            # If not given, we cannot check; skip
            return False, "cannot check: P not provided"
        # Allowed range around 1.0 atm (allow small numeric tolerance)
        if math.isclose(self.P, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"P = {self.P} atm (≈ 1)"
        else:
            return False, f"P = {self.P} atm, expected near 1"

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_T_boil_decreases_total_entropy(self):
        """
        If T_boil increases (other givens unchanged), ΔS_vap decreases,
        and ΔS_heat also decreases (since ln(T_gas/T_boil) decreases).
        Overall ΔS_total should decrease.
        We perturb T_boil by +5% and recompute using current values.
        """
        if self.m is None or self.L is None or self.Cp is None or self.n_calc is None or self.T_gas is None or self.T_boil is None:
            return False, "cannot run monotonicity: missing givens"
        # Baseline
        base_vap = self.m * self.L / self.T_boil
        base_heat = self.n_calc * self.Cp * math.log(self.T_gas / self.T_boil)
        base_total = base_vap + base_heat

        # Increased T_boil by 5%
        T_boil_new = self.T_boil * 1.05
        new_vap = self.m * self.L / T_boil_new
        new_heat = self.n_calc * self.Cp * math.log(self.T_gas / T_boil_new)
        new_total = new_vap + new_heat

        if new_total < base_total:
            return True, f"Total entropy decreased from {base_total:.4f} to {new_total:.4f}"
        else:
            return False, f"Expected decrease, but new total {new_total} >= {base_total}"

    def test_monotonicity_T_gas_increases_total_entropy(self):
        """
        If T_gas increases, ΔS_heat increases, while ΔS_vap unchanged.
        Total entropy increases.
        Perturb T_gas by +5%.
        """
        if self.m is None or self.L is None or self.Cp is None or self.n_calc is None or self.T_gas is None or self.T_boil is None:
            return False, "cannot run monotonicity: missing givens"
        base_vap = self.m * self.L / self.T_boil
        base_heat = self.n_calc * self.Cp * math.log(self.T_gas / self.T_boil)
        base_total = base_vap + base_heat

        T_gas_new = self.T_gas * 1.05
        new_heat = self.n_calc * self.Cp * math.log(T_gas_new / self.T_boil)
        new_total = base_vap + new_heat

        if new_total > base_total:
            return True, f"Total entropy increased from {base_total:.4f} to {new_total:.4f}"
        else:
            return False, f"Expected increase, but new total {new_total} <= {base_total}"

    def test_monotonicity_m_increases_total_entropy(self):
        """
        Doubling mass doubles both contributions, thus total entropy increases.
        """
        if self.m is None or self.L is None or self.Cp is None or self.n_calc is None or self.T_gas is None or self.T_boil is None:
            return False, "cannot run monotonicity: missing givens"
        # Baseline (using current n_calc from given m)
        base_vap = self.m * self.L / self.T_boil
        base_heat = self.n_calc * self.Cp * math.log(self.T_gas / self.T_boil)
        base_total = base_vap + base_heat

        # Double mass
        m_new = self.m * 2
        n_new = m_new / self.M
        new_vap = m_new * self.L / self.T_boil
        new_heat = n_new * self.Cp * math.log(self.T_gas / self.T_boil)
        new_total = new_vap + new_heat

        if new_total > base_total:
            return True, f"Total entropy increased from {base_total:.4f} to {new_total:.4f} (mass doubled)"
        else:
            return False, f"Expected increase, but new total {new_total} <= {base_total}"

    # ------------------------------------------------------------------
    # Answer sanity (weak final check, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """
        Final ΔS_total should be a positive value of order 1 cal/K.
        We accept between 0.5 and 2.0 cal/K (broad range).
        """
        if self.delta_S_total is None:
            return False, "cannot check magnitude: delta_S_total missing"
        if not isinstance(self.delta_S_total, (int, float)):
            return False, "delta_S_total not numeric"
        if math.isnan(self.delta_S_total) or math.isinf(self.delta_S_total):
            return False, "delta_S_total is NaN or Inf"
        if self.delta_S_total <= 0:
            return False, f"delta_S_total = {self.delta_S_total} is not positive"
        if self.delta_S_total < 0.5 or self.delta_S_total > 2.0:
            return False, f"delta_S_total = {self.delta_S_total} is outside plausible range (0.5, 2.0) cal/K"
        return True, f"delta_S_total = {self.delta_S_total} cal/K (plausible)"

    # ------------------------------------------------------------------
    # Run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)