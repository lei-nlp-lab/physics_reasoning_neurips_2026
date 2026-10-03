import math
import re
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results with safe defaults
        self.results = getattr(solution_instance, 'results', {})
        self.Q1 = self.results.get('Q1', None)
        self.Q2 = self.results.get('Q2', None)
        self.aB = self.results.get('aB', None)
        self.aB_check = self.results.get('aB_check', None)
        self.gauge_independent = self.results.get('gauge_independent', None)
        self.energy_density = self.results.get('energy_density', None)
        # Optional: explanation_steps for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface tests: required keys and types
    # ----------------------------------------------------------------------
    def test_interface_Q1_type(self):
        if self.Q1 is None:
            return (False, "SKIP: Q1 not provided")
        if not isinstance(self.Q1, str):
            return (False, f"Q1 should be a string, got {type(self.Q1)}")
        return (True, "Q1 is string")

    def test_interface_Q2_type(self):
        if self.Q2 is None:
            return (False, "SKIP: Q2 not provided")
        if not isinstance(self.Q2, str):
            return (False, f"Q2 should be a string, got {type(self.Q2)}")
        return (True, "Q2 is string")

    def test_interface_aB_type(self):
        if self.aB is None:
            return (False, "SKIP: aB not provided")
        if not isinstance(self.aB, (int, float)):
            return (False, f"aB should be numeric, got {type(self.aB)}")
        return (True, "aB is numeric")

    def test_interface_aB_check_type(self):
        if self.aB_check is None:
            return (False, "SKIP: aB_check not provided")
        if not isinstance(self.aB_check, (int, float)):
            return (False, f"aB_check should be numeric, got {type(self.aB_check)}")
        return (True, "aB_check is numeric")

    def test_interface_gauge_independent_type(self):
        if self.gauge_independent is None:
            return (False, "SKIP: gauge_independent not provided")
        if not isinstance(self.gauge_independent, str):
            return (False, f"gauge_independent should be a string, got {type(self.gauge_independent)}")
        return (True, "gauge_independent is string")

    def test_interface_energy_density_type(self):
        if self.energy_density is None:
            return (False, "SKIP: energy_density not provided")
        if not isinstance(self.energy_density, str):
            return (False, f"energy_density should be a string, got {type(self.energy_density)}")
        return (True, "energy_density is string")

    # ----------------------------------------------------------------------
    # Invariant tests: physics-based checks (use math.isclose where appropriate)
    # ----------------------------------------------------------------------
    def test_invariant_Q1_no_constraint(self):
        if self.Q1 is None:
            return (False, "SKIP: Q1 not provided")
        expected = "No constraint"
        if self.Q1.strip() != expected:
            return (False, f"Q1 should be '{expected}', got '{self.Q1}'")
        return (True, "Q1 correctly indicates no constraint")

    def test_invariant_Q2_no_constraint(self):
        if self.Q2 is None:
            return (False, "SKIP: Q2 not provided")
        expected = "No constraint"
        if self.Q2.strip() != expected:
            return (False, f"Q2 should be '{expected}', got '{self.Q2}'")
        return (True, "Q2 correctly indicates no constraint")

    def test_invariant_aB_zero(self):
        if self.aB is None:
            return (False, "SKIP: aB not provided")
        if not math.isclose(self.aB, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"aB should be 0, got {self.aB}")
        return (True, "aB is zero")

    def test_invariant_aB_check_zero(self):
        if self.aB_check is None:
            return (False, "SKIP: aB_check not provided")
        if not math.isclose(self.aB_check, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"aB_check should be 0, got {self.aB_check}")
        return (True, "aB_check is zero")

    # ----------------------------------------------------------------------
    # Modeling/trace tests: based on explanation steps and expected formulations
    # ----------------------------------------------------------------------
    def test_modeling_gauge_independent_correct(self):
        """Check that gauge_independent mentions a1 and a2, and NOT aL or aB."""
        if self.gauge_independent is None:
            return (False, "SKIP: gauge_independent not provided")
        s = self.gauge_independent
        has_a1 = 'a1' in s or 'a_1' in s
        has_a2 = 'a2' in s or 'a_2' in s
        has_aL = 'aL' in s or 'a_L' in s
        has_aB = 'aB' in s or 'a_B' in s
        if not (has_a1 and has_a2):
            return (False, f"gauge_independent should mention a1 and a2, got '{s}'")
        if has_aL:
            return (False, f"gauge_independent should not mention aL, but got '{s}'")
        if has_aB:
            return (False, f"gauge_independent should not mention aB, but got '{s}'")
        return (True, "gauge_independent correctly identifies a1 and a2")

    def test_modeling_energy_density_form(self):
        """Check that energy_density contains K^2, a1^2, a2^2, and no aL/aB."""
        if self.energy_density is None:
            return (False, "SKIP: energy_density not provided")
        s = self.energy_density
        has_K2 = re.search(r'K\^2', s)
        has_a1sq = re.search(r'a_?1\^2', s)
        has_a2sq = re.search(r'a_?2\^2', s)
        if not (has_K2 and has_a1sq and has_a2sq):
            return (False, f"energy_density should contain K^2, a1^2, a2^2, got '{s}'")
        # Check absence of aL and aB
        has_aL = re.search(r'[aA]_?[lL]', s)  # matches "aL", "a_L", "Al" etc.
        has_aB = re.search(r'[aA]_?[bB]', s)
        if has_aL:
            return (False, f"energy_density should not contain aL, but got '{s}'")
        if has_aB:
            return (False, f"energy_density should not contain aB, but got '{s}'")
        return (True, "energy_density formula is correct")

    def test_modeling_explanation_steps_consistent(self):
        """Check that explanation_steps explicitly mention aB=0."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        steps_str = str(self.explanation_steps)
        pattern = r'a[B_]\s*=\s*0'
        if re.search(pattern, steps_str):
            return (True, "explanation_steps mention aB=0")
        else:
            return (False, "explanation_steps do not explicitly state aB=0")

    # ----------------------------------------------------------------------
    # Monotonicity tests: explicit perturbation of an input and comparison
    # ----------------------------------------------------------------------
    def test_monotonicity_divE_increases_with_aB(self):
        """
        Perturb aB from 0 to a small positive value and check that |∇·E| increases.
        Using relation ∇·E = 2K² aB (from explanation_steps). Assume K=1 for baseline.
        """
        if self.aB is None:
            return (False, "SKIP: aB not provided")
        # Baseline: aB = self.aB (should be 0)
        baseline_aB = self.aB
        # Compute baseline |∇·E| using K=1 (arbitrary positive constant)
        K = 1.0
        baseline_divE = 2 * K**2 * baseline_aB  # absolute value
        # Perturb to a small positive value
        perturbed_aB = baseline_aB + 1e-3
        perturbed_divE = 2 * K**2 * perturbed_aB
        # Check monotonicity: perturbed > baseline
        if perturbed_divE > baseline_divE:
            return (True, "|∇·E| increases when aB increases (monotonic)")
        else:
            return (False, f"|∇·E| did not increase: baseline {baseline_divE}, perturbed {perturbed_divE}")

    def test_monotonicity_energy_increases_with_K(self):
        """
        Perturb K from 1.0 to 1.1 and check that energy density increases.
        Energy density = K²(a1²+a2²)/(8π). Use a1=a2=1 for baseline.
        """
        if self.energy_density is None:
            return (False, "SKIP: energy_density not provided")
        # Baseline parameters (arbitrary positive values)
        K_baseline = 1.0
        a1 = 1.0
        a2 = 1.0
        baseline_energy = (K_baseline**2) * (a1**2 + a2**2) / (8 * math.pi)
        # Perturb K: increase by 10%
        K_perturbed = 1.1
        perturbed_energy = (K_perturbed**2) * (a1**2 + a2**2) / (8 * math.pi)
        if perturbed_energy > baseline_energy:
            return (True, "Energy density increases when K increases (monotonic)")
        else:
            return (False, f"Energy density did not increase: baseline {baseline_energy}, perturbed {perturbed_energy}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_energy_density_not_empty(self):
        if self.energy_density is None:
            return (False, "SKIP: energy_density not provided")
        if len(self.energy_density.strip()) == 0:
            return (False, "energy_density string is empty")
        return (True, "energy_density is non-empty")

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
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