import math
import traceback

class PhysicsChecker:
    """
    Audits solution for a two-part thermodynamics problem:
    - Burning coal (100J) in a Carnot power plant between 1273.15K and 303.15K.
    - Work from plant used either in resistance heating (Q1) or in a heat pump (Q2)
      pumping heat from 273.15K to 303.15K.

    Extracts Q1 and Q2 from solution_instance.results.
    Performs interface, invariant, modeling, and a weak sanity check.
    All tests return (passed_bool, message_string).
    run_all_tests() collects and returns (success, logs).
    """
    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', None) or {}
        self.Q1 = results.get('Q1', None)
        self.Q2 = results.get('Q2', None)

        # Store physical constants (in Kelvin, exactly as given)
        self.T_hot_plant = 1273.15
        self.T_cold_plant = 303.15
        self.T_hot_house = 303.15
        self.T_cold_house = 273.15
        self.Q_coal = 100.0  # J

        # Derive expected efficiency and COP (used in modeling tests)
        self.eta_expected = 1.0 - self.T_cold_plant / self.T_hot_plant
        self.W_expected = self.eta_expected * self.Q_coal
        self.COP_expected = self.T_hot_house / (self.T_hot_house - self.T_cold_house)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_Q1_exists(self):
        if self.Q1 is None:
            return (False, "Q1 not found in results")
        return (True, "Q1 key present")

    def test_interface_Q2_exists(self):
        if self.Q2 is None:
            return (False, "Q2 not found in results")
        return (True, "Q2 key present")

    def test_interface_Q1_is_number(self):
        if self.Q1 is None:
            return (False, "SKIP: Q1 is missing")
        if not isinstance(self.Q1, (int, float)):
            return (False, f"Q1 is not a number, got {type(self.Q1).__name__}")
        return (True, "Q1 is numeric")

    def test_interface_Q2_is_number(self):
        if self.Q2 is None:
            return (False, "SKIP: Q2 is missing")
        if not isinstance(self.Q2, (int, float)):
            return (False, f"Q2 is not a number, got {type(self.Q2).__name__}")
        return (True, "Q2 is numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_Q1_positive(self):
        if self.Q1 is None:
            return (False, "SKIP: Q1 missing")
        if self.Q1 <= 0:
            return (False, f"Q1 must be positive, got {self.Q1}")
        return (True, "Q1 > 0")

    def test_invariant_Q2_positive(self):
        if self.Q2 is None:
            return (False, "SKIP: Q2 missing")
        if self.Q2 <= 0:
            return (False, f"Q2 must be positive, got {self.Q2}")
        return (True, "Q2 > 0")

    def test_invariant_Q1_less_than_Qcoal(self):
        """Carnot efficiency < 1, so work (and Q1) must be < 100 J."""
        if self.Q1 is None:
            return (False, "SKIP: Q1 missing")
        if self.Q1 >= self.Q_coal:
            return (False, f"Q1 = {self.Q1} should be less than 100 J (efficiency < 1)")
        return (True, "Q1 < 100 J")

    def test_invariant_Q2_greater_than_W(self):
        """Heat pump delivers more heat than work input (COP > 1)."""
        if self.Q1 is None or self.Q2 is None:
            return (False, "SKIP: missing Q1 or Q2")
        # Q1 is equal to work
        if self.Q2 <= self.Q1:
            return (False, f"Q2 = {self.Q2} should exceed work = {self.Q1} (COP > 1)")
        return (True, "Q2 > W (heat pump amplifies)")

    # ------------------------------------------------------------------
    # Modeling / trace tests (check consistency with physics formulas)
    # ------------------------------------------------------------------
    def test_modeling_Q1_equals_eta_times_Qcoal(self):
        """Q1 should equal W = η * 100 J."""
        if self.Q1 is None:
            return (False, "SKIP: Q1 missing")
        if not math.isclose(self.Q1, self.W_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q1 = {self.Q1}, expected {self.W_expected} from η = {self.eta_expected}")
        return (True, f"Q1 matches Carnot work: η*100 = {self.W_expected}")

    def test_modeling_Q2_equals_COP_times_W(self):
        """Q2 should equal COP * W, with W = η*100."""
        if self.Q1 is None or self.Q2 is None:
            return (False, "SKIP: Q1 or Q2 missing")
        W = self.Q1  # baseline work
        expected_Q2 = self.COP_expected * W
        if not math.isclose(self.Q2, expected_Q2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q2 = {self.Q2}, expected {expected_Q2} (COP = {self.COP_expected})")
        return (True, "Q2 = COP * W")

    def test_modeling_ratio_Q2_over_Q1_matches_COP(self):
        """The ratio Q2/Q1 should equal the heat pump COP."""
        if self.Q1 is None or self.Q2 is None:
            return (False, "SKIP: missing values")
        if self.Q1 == 0:
            return (False, "Q1 is zero, cannot compute ratio")
        ratio = self.Q2 / self.Q1
        if not math.isclose(ratio, self.COP_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q2/Q1 = {ratio}, expected COP = {self.COP_expected}")
        return (True, f"Q2/Q1 ≈ COP = {self.COP_expected}")

    def test_modeling_efficiency_realistic_bound(self):
        """Carnot efficiency (1 - 303.15/1273.15) ≈ 0.762. Q1/100 should match."""
        if self.Q1 is None:
            return (False, "SKIP: Q1 missing")
        efficiency_from_Q1 = self.Q1 / self.Q_coal
        if not math.isclose(efficiency_from_Q1, self.eta_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Inferred efficiency {efficiency_from_Q1}, expected {self.eta_expected}")
        return (True, f"Q1/Qcoal ≈ η = {self.eta_expected}")

    # ------------------------------------------------------------------
    # Monotonicity tests (we can't perturb input, but we can verify that
    # Q1 and Q2 follow expected monotonic relationships with temperature.)
    # Since only one set of results, we check that if we hypothetically
    # increase T_cold_plant by 5%, the new work would be smaller, i.e.
    # the reported Q1 must be larger than that hypothetical work.
    # Actually, we can't know the reported Q1 for that perturbation, but
    # we can check that the current work is consistent with the given T_cold_plant.
    # Instead, we implement a pseudo-monotonicity by checking that
    # efficiency decreases as T_cold_plant increases, using the current T_cold_plant.
    # This is more like an invariant, but we'll label as monotonicity.
    # ------------------------------------------------------------------
    def test_monotonicity_efficiency_decreases_with_Tcold(self):
        """If T_cold_plant were higher, efficiency would be lower, so W would be lower.
        We check that the current efficiency is less than what it would be if T_cold_plant were 5% lower.
        (If the answer used an even colder reservoir, they'd get higher efficiency.)"""
        if self.Q1 is None:
            return (False, "SKIP: Q1 missing")
        # Compute efficiency today
        eff_current = self.Q1 / self.Q_coal
        # Hypothetical: T_cold_plant reduced by 5%
        Tc_lower = self.T_cold_plant * 0.95
        eff_hyp = 1 - Tc_lower / self.T_hot_plant
        if eff_hyp <= eff_current:
            # This would fail if the solution used a lower Te (should increase eff)
            return (False, f"Efficiency with 5% colder reservoir = {eff_hyp} <= current {eff_current}, expected increase")
        return (True, "Monotonicity consistent: cooler reservoir gives higher efficiency")

    def test_monotonicity_COP_decreases_as_Tch_increases(self):
        """If the cold source for the heat pump were warmer, COP would decrease.
        We check that current COP (from Q2/Q1) is higher than if T_cold_house were 5% higher."""
        if self.Q1 is None or self.Q2 is None:
            return (False, "SKIP: missing Q1 or Q2")
        COP_current = self.Q2 / self.Q1
        # Hypothetical: T_cold_house increased by 5%
        Tch_hotter = self.T_cold_house * 1.05
        COP_hyp = self.T_hot_house / (self.T_hot_house - Tch_hotter)
        if COP_hyp >= COP_current:
            return (False, f"COP with 5% warmer cold source = {COP_hyp} >= current {COP_current}, expected lower")
        return (True, "Monotonicity consistent: warmer cold source gives lower COP")

    # ------------------------------------------------------------------
    # Answer sanity (weak, only one test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic_check(self):
        """Weak sanity: Q1 and Q2 must be finite positive numbers, Q2 > Q1, both < 1000 J."""
        if self.Q1 is None or self.Q2 is None:
            return (False, "SKIP: missing values")
        if math.isnan(self.Q1) or math.isinf(self.Q1):
            return (False, "Q1 is not a finite number")
        if math.isnan(self.Q2) or math.isinf(self.Q2):
            return (False, "Q2 is not a finite number")
        if self.Q1 <= 0 or self.Q2 <= 0:
            return (False, "Both Q1 and Q2 must be positive")
        if self.Q2 <= self.Q1:
            return (False, "Q2 must be larger than Q1 (heat pump amplifies)")
        # Broad magnitude: Q1 between 50 and 100, Q2 between 500 and 1000 (expected ~76 and ~770)
        if not (50 < self.Q1 < 100):
            return (False, f"Q1 = {self.Q1} not in plausible range (50,100)")
        if not (500 < self.Q2 < 1000):
            return (False, f"Q2 = {self.Q2} not in plausible range (500,1000)")
        return (True, "Q1 and Q2 are physically plausible")

    # ------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)