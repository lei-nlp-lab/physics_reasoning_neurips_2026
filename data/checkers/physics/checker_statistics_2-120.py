import re
import math
import traceback

class PhysicsChecker:
    """
    Auditor for thermodynamics problem: third law, negative temperature, example, classical limitation.
    """
    def __init__(self, solution_instance):
        # Extract results safely; default to None for missing keys
        self.results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        # Define each expected result with None default
        self.third_law = self.results.get('third_law', None)
        self.neg_T_definition = self.results.get('neg_T_definition', None)
        self.example = self.results.get('example', None)
        self.classical_limitation = self.results.get('classical_limitation', None)

        # Optional numeric input for monotonicity test (if available)
        self.energy_gap = getattr(solution_instance, 'energy_gap', None)
        self.population_ratio = getattr(solution_instance, 'population_ratio', None)

        # Store full solution_instance for potential future use
        self.solution_instance = solution_instance

    def run_all_tests(self):
        """Discover all test_ methods and run them, collecting logs."""
        logs = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface tests: verify required keys exist and have correct types
    # ------------------------------------------------------------------
    def test_interface_third_law_present(self):
        if self.third_law is None:
            return False, "Missing result key 'third_law'"
        if not isinstance(self.third_law, str):
            return False, f"'third_law' is not a string (type {type(self.third_law).__name__})"
        return True, "Key 'third_law' exists and is a string"

    def test_interface_neg_T_definition_present(self):
        if self.neg_T_definition is None:
            return False, "Missing result key 'neg_T_definition'"
        if not isinstance(self.neg_T_definition, str):
            return False, f"'neg_T_definition' is not a string (type {type(self.neg_T_definition).__name__})"
        return True, "Key 'neg_T_definition' exists and is a string"

    def test_interface_example_present(self):
        if self.example is None:
            return False, "Missing result key 'example'"
        if not isinstance(self.example, str):
            return False, f"'example' is not a string (type {type(self.example).__name__})"
        return True, "Key 'example' exists and is a string"

    def test_interface_classical_limitation_present(self):
        if self.classical_limitation is None:
            return False, "Missing result key 'classical_limitation'"
        if not isinstance(self.classical_limitation, str):
            return False, f"'classical_limitation' is not a string (type {type(self.classical_limitation).__name__})"
        return True, "Key 'classical_limitation' exists and is a string"

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity
    # ------------------------------------------------------------------
    def test_invariant_third_law_statement_contains_third_law(self):
        """Third law statement should mention 'third law' or 'Nernst'."""
        if self.third_law is None:
            return False, "SKIP: third_law not available"
        s_lower = self.third_law.lower()
        if "third law" in s_lower or "nernst" in s_lower:
            return True, "Third law statement contains reference to third law or Nernst"
        else:
            return False, f"Third law statement does not mention 'third law' or 'Nernst': '{self.third_law[:50]}...'"

    def test_invariant_third_law_absolute_zero_reachability(self):
        """Third law should assert that absolute zero is unattainable."""
        if self.third_law is None:
            return False, "SKIP: third_law not available"
        s_lower = self.third_law.lower()
        if "cannot reach" in s_lower or "no system" in s_lower or "impossible" in s_lower or "never" in s_lower:
            return True, "Third law statement correctly indicates absolute zero is unattainable"
        else:
            return False, f"Third law statement does not communicate unattainability of absolute zero: '{self.third_law[:50]}...'"

    def test_invariant_neg_T_definition_inverse_temperature(self):
        """Negative temperature definition should include 1/T = dS/dU relation."""
        if self.neg_T_definition is None:
            return False, "SKIP: neg_T_definition not available"
        s_lower = self.neg_T_definition.lower()
        has_one_over_T = "1/t" in s_lower
        has_derivative = "ds/du" in s_lower or "∂s/∂u" in s_lower or "d s/d u" in s_lower
        if has_one_over_T and has_derivative:
            return True, "Negative temperature definition includes 1/T = dS/dU relation"
        else:
            return False, f"Negative temperature definition missing 1/T and/or dS/dU: '{self.neg_T_definition[:60]}...'"

    def test_invariant_neg_T_population_inversion_or_dSdU_negative(self):
        """Negative temperature definition should mention population inversion or dS/dU < 0."""
        if self.neg_T_definition is None:
            return False, "SKIP: neg_T_definition not available"
        s_lower = self.neg_T_definition.lower()
        if "population inversion" in s_lower or "ds/du < 0" in s_lower or "∂s/∂u < 0" in s_lower:
            return True, "Negative temperature definition correctly identifies condition dS/dU < 0"
        else:
            return False, f"Negative temperature definition missing population inversion or dS/dU < 0: '{self.neg_T_definition[:60]}...'"

    def test_invariant_example_mentions_spin(self):
        """Example should mention spin (or magnetic) as the system."""
        if self.example is None:
            return False, "SKIP: example not available"
        s_lower = self.example.lower()
        if "spin" in s_lower or "magnetic" in s_lower:
            return True, "Example mentions spin or magnetic system"
        else:
            return False, f"Example does not mention spin or magnetic systems: '{self.example[:50]}...'"

    def test_invariant_example_contains_inversion_or_reversal(self):
        """Example should describe population inversion or field reversal."""
        if self.example is None:
            return False, "SKIP: example not available"
        s_lower = self.example.lower()
        if "inversion" in s_lower or "reversal" in s_lower or "flip" in s_lower or "pump" in s_lower:
            return True, "Example describes a mechanism to achieve negative temperature"
        else:
            return False, f"Example missing mechanism (inversion/reversal/pump): '{self.example[:50]}...'"

    def test_invariant_classical_limitation_monotonic(self):
        """Classical limitation should state that dS/dU > 0 is required."""
        if self.classical_limitation is None:
            return False, "SKIP: classical_limitation not available"
        s_lower = self.classical_limitation.lower()
        if "ds/du > 0" in s_lower or "∂s/∂u > 0" in s_lower or "positive derivative" in s_lower or "monotonic" in s_lower:
            return True, "Classical limitation correctly mentions dS/dU > 0 or monotonic entropy"
        else:
            return False, f"Classical limitation missing condition dS/dU > 0 or monotonic: '{self.classical_limitation[:50]}...'"

    # ------------------------------------------------------------------
    # Modeling tests: deeper content checks using explanation steps
    # ------------------------------------------------------------------
    def test_modeling_third_law_perfect_crystal(self):
        """Third law statement should mention perfect crystal or S=0 as T→0."""
        if self.third_law is None:
            return False, "SKIP: third_law not available"
        s_lower = self.third_law.lower()
        if "perfect crystal" in s_lower or "s=0" in s_lower.replace(" ", "") or "entropy zero" in s_lower:
            return True, "Third law statement references perfect crystal or entropy zero at absolute zero"
        else:
            return False, f"Third law statement missing reference to perfect crystal or entropy zero: '{self.third_law[:50]}...'"

    def test_modeling_neg_T_definition_uses_inverse_temperature(self):
        """Definition of negative T should explicitly use 1/T = dS/dU."""
        if self.neg_T_definition is None:
            return False, "SKIP: neg_T_definition not available"
        cleaned = self.neg_T_definition.replace(" ", "")
        if re.search(r"1/T\s*=\s*dS/dU", cleaned, re.IGNORECASE):
            return True, "Neg.T definition contains exact relation 1/T = dS/dU"
        else:
            return False, f"Neg.T definition does not contain the relation 1/T = dS/dU as expected: '{self.neg_T_definition[:60]}...'"

    def test_modeling_example_realization_method(self):
        """Example should describe how to achieve negative T (field reversal, RF pumping)."""
        if self.example is None:
            return False, "SKIP: example not available"
        s_lower = self.example.lower()
        if ("reverse" in s_lower and "magnetic" in s_lower) or ("rf" in s_lower and "pump" in s_lower) or ("invert" in s_lower and "population" in s_lower):
            return True, "Example describes an experimental realization method"
        else:
            return False, f"Example does not describe how to achieve negative T: '{self.example[:50]}...'"

    def test_modeling_classical_limitation_bounded_energy(self):
        """Classical limitation should mention that classical systems have unbounded energy, or negative T requires bounded energy."""
        if self.classical_limitation is None:
            return False, "SKIP: classical_limitation not available"
        s_lower = self.classical_limitation.lower()
        if "bounded" in s_lower or "upper bound" in s_lower or "finite energy" in s_lower or "unbounded" in s_lower:
            return True, "Classical limitation mentions energy bound concept"
        else:
            if "mechanical instability" in s_lower or "instability" in s_lower:
                return True, "Classical limitation mentions mechanical instability (another key point)"
            return False, f"Classical limitation missing energy bound or instability: '{self.classical_limitation[:50]}...'"

    # ------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare results)
    # ------------------------------------------------------------------
    def test_monotonicity_inverse_temperature_with_energy_gap(self):
        """
        If numeric inputs (energy_gap and population_ratio) are provided,
        test that perturbing the energy gap changes the inferred temperature monotonically.
        Uses the Boltzmann relation: 1/T ∝ ln(N_down/N_up) / ΔE.
        Perturbation: increase ΔE by 10% should change 1/T in a predictable direction.
        """
        if self.energy_gap is None or self.population_ratio is None:
            return False, "SKIP: No numeric inputs (energy_gap or population_ratio) available for monotonicity test"
        try:
            Delta_E_base = float(self.energy_gap)
            r_base = float(self.population_ratio)  # N_down/N_up
        except (ValueError, TypeError):
            return False, "SKIP: energy_gap or population_ratio not numeric"
        if Delta_E_base <= 0 or r_base <= 0:
            return False, "SKIP: Non-positive values; cannot perform monotonicity test"
        # Baseline inverse temperature (ignoring kB, since only trend matters)
        inv_T_base = math.log(r_base) / Delta_E_base
        # Perturb: increase ΔE by 10%
        Delta_E_perturbed = Delta_E_base * 1.10
        inv_T_pert = math.log(r_base) / Delta_E_perturbed
        # Check monotonicity direction:
        # If r_base > 1 (population inversion), ln(r_base) > 0, so increasing ΔE should decrease 1/T
        # If r_base < 1 (normal population), ln(r_base) < 0, so increasing ΔE should increase 1/T
        if r_base > 1:
            if inv_T_base > inv_T_pert:
                return True, f"Monotonicity holds: for inverted population (r={r_base:.2f}), increasing gap reduces 1/T"
            else:
                return False, f"Monotonicity failed: inverse temperature increased when gap increased (baseline={inv_T_base:.4f}, perturbed={inv_T_pert:.4f})"
        elif r_base < 1:
            if inv_T_base < inv_T_pert:
                return True, f"Monotonicity holds: for normal population (r={r_base:.2f}), increasing gap increases 1/T"
            else:
                return False, f"Monotonicity failed: inverse temperature decreased when gap increased (baseline={inv_T_base:.4f}, perturbed={inv_T_pert:.4f})"
        else:
            # r_base == 1, temperature infinite, trivial
            return True, "Population ratio =1, temperature infinite; monotonicity trivially satisfied"

    # ------------------------------------------------------------------
    # Answer sanity (one weak method only)
    # ------------------------------------------------------------------
    def test_answer_sanity_general(self):
        """
        Weak sanity checks on answer quality:
        - All four result strings are non-empty.
        - The third law statement does not claim absolute zero is reachable.
        - The negative temperature definition mentions it does not violate the third law.
        """
        # Check non-empty
        checks = []
        all_keys_nonempty = True
        for key, val in [("third_law", self.third_law),
                         ("neg_T_definition", self.neg_T_definition),
                         ("example", self.example),
                         ("classical_limitation", self.classical_limitation)]:
            if val is None or str(val).strip() == "":
                checks.append(f"Missing or empty: {key}")
                all_keys_nonempty = False
        if not all_keys_nonempty:
            return False, "; ".join(checks)

        # Check third law does not claim reachability
        s_lower = self.third_law.lower()
        if "can reach" in s_lower or "possible to reach" in s_lower or "achieve absolute zero" in s_lower:
            checks.append("Third law incorrectly implies absolute zero is reachable")

        # Check negative T definition mentions no violation
        s_lower2 = self.neg_T_definition.lower()
        if not ("not violate" in s_lower2 or "does not violate" in s_lower2 or "no violation" in s_lower2 or "further from" in s_lower2 or "farther from" in s_lower2):
            checks.append("Neg.T definition does not mention it does not violate third law")

        if checks:
            return False, "Sanity issues: " + "; ".join(checks)
        return True, "All weak sanity checks passed"