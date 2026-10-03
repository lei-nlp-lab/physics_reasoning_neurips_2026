import sympy
import math
import traceback

class PhysicsChecker:
    """
    Auditors a solution for the interference of two synchronous tuning forks.
    Checks physical consistency, modeling quality, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safe extraction of all relevant data with None defaults
        self.givens = getattr(solution_instance, 'givens', None)
        self.results = getattr(solution_instance, 'results', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.physics_mapping = getattr(solution_instance, 'physics_mapping', None)

        # Extract specific quantities with None defaults
        self.delta_s = self.results.get('delta_s', None) if self.results else None

        self.f = self.givens.get('f', None) if self.givens else None
        self.c = self.givens.get('c', None) if self.givens else None
        self.s1 = self.givens.get('s1', None) if self.givens else None
        self.s2 = self.givens.get('s2', None) if self.givens else None
        self.A = self.givens.get('A', None) if self.givens else None

        # Derived quantity
        if self.f is not None and self.c is not None:
            self.lambda_ = self.c / self.f
        else:
            self.lambda_ = None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Verifies that all required result and given keys are present."""
        if self.results is None:
            return (False, "No results dictionary found.")
        if 'delta_s' not in self.results:
            return (False, "Required result key 'delta_s' is missing.")
        if self.givens is None:
            return (False, "No givens dictionary found.")
        required = ['f', 'c', 's1', 's2', 'A']
        missing = [k for k in required if k not in self.givens]
        if missing:
            return (False, f"Missing required givens: {missing}")
        return (True, "All required keys are present.")

    def test_interface_results_type(self):
        """Checks that delta_s is a valid expression (string or numeric)."""
        if self.delta_s is None:
            return (False, "delta_s is missing.")
        if not isinstance(self.delta_s, (str, int, float)):
            return (False, f"delta_s has unexpected type {type(self.delta_s)}.")
        return (True, "delta_s type is acceptable.")

    # ------------------------------------------------------------------
    # Theorem-based invariants (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_superposition_principle(self):
        """Explanation steps should invoke superposition of two sine waves."""
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation steps available.")
        joined = ' '.join(self.explanation_steps).lower()
        keywords = ['y1', 'y2', 'sin', 'sum', 'resultant', 'superposition']
        found = [kw for kw in keywords if kw in joined]
        if len(found) < 3:
            return (False, f"Superposition not sufficiently covered. Found only: {found}")
        return (True, "Superposition principle is referenced.")

    def test_invariant_energy_conservation(self):
        """Solution must explain that energy is redistributed, not destroyed."""
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation steps available.")
        joined = ' '.join(self.explanation_steps).lower()
        # multiple possible phrasings
        energy_phrases = [
            'conservation of energy', 'energy is not destroyed',
            'redistributed', 'energy is conserved', 'total energy'
        ]
        found = [ph for ph in energy_phrases if ph in joined]
        if not found:
            return (False, "No mention of energy conservation or redistribution.")
        return (True, f"Energy conservation mentioned: {found}")

    def test_invariant_destructive_condition(self):
        """Explanation should contain the path-difference condition for zero intensity."""
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation steps available.")
        joined = ' '.join(self.explanation_steps).lower()
        # The key equation: omega*(s2-s1)/(2c) = (2n+1)*pi/2
        patterns = [
            'omega', 's2', 's1', 'c', '2n+1', 'pi/2', 'odd multiple',
            'half-wavelength', 'lambda/2', 'path difference'
        ]
        found = [pat for pat in patterns if pat in joined]
        if len(found) < 3:
            return (False, f"Destructive condition not sufficiently described. Found: {found}")
        return (True, "Destructive interference condition is present.")

    # ------------------------------------------------------------------
    # Modeling quality checks (use explanation_steps and physics_mapping)
    # ------------------------------------------------------------------
    def test_modeling_explanation_coverage(self):
        """Check that the explanation covers key conceptual steps."""
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation steps.")
        joined = ' '.join(self.explanation_steps).lower()
        required_concepts = [
            'amplitude', 'intensity', 'nodes', 'antinodes',
            'energy', 'interference', 'constructive', 'destructive'
        ]
        missing = [c for c in required_concepts if c not in joined]
        if missing:
            return (False, f"Missing concepts in explanation: {missing}")
        return (True, "Explanation covers required concepts.")

    def test_modeling_physics_mapping_present(self):
        """Physics mapping should contain the interference condition."""
        if self.physics_mapping is None:
            return (False, "SKIP: No physics_mapping provided.")
        # check that mapping includes the superposition or the condition
        mapping_str = str(self.physics_mapping).lower()
        key_terms = ['phase difference', 'phi', 'pi', 'amplitude squared', 'intensity']
        found = [t for t in key_terms if t in mapping_str]
        if len(found) < 2:
            return (False, f"Physics mapping lacks key interference concepts. Found: {found}")
        return (True, "Physics mapping contains relevant interference details.")

    # ------------------------------------------------------------------
    # Monotonicity checks (if numeric data available)
    # ------------------------------------------------------------------
    def test_monotonicity_intensity_with_path_difference(self):
        """
        As path difference moves away from odd half-wavelength, intensity should increase.
        Check using a small perturbation if numeric delta_s and lambda are available.
        """
        if self.delta_s is None or self.lambda_ is None:
            return (False, "SKIP: Numeric delta_s or lambda required for monotonicity.")
        if not isinstance(self.delta_s, (int, float)):
            return (False, "SKIP: delta_s is symbolic, cannot compute numeric monotonicity.")
        if self.lambda_ <= 0:
            return (False, "Invalid lambda (<=0).")
        # Baseline cos factor for destructive: cos(π/2) = 0
        # For a small change in path difference, cos should become nonzero.
        eps = 1e-3 * self.lambda_
        new_delta = self.delta_s + eps
        # Compute the factor: cos(omega*delta_s/(2c)) = cos(π*delta_s/lambda)
        baseline_factor = math.cos(math.pi * self.delta_s / self.lambda_)
        perturbed_factor = math.cos(math.pi * new_delta / self.lambda_)
        # Expect baseline_factor ≈ 0 (destructive) and perturbed_factor > 0 in magnitude
        if not math.isclose(baseline_factor, 0.0, abs_tol=1e-2):
            return (False, f"Baseline cos factor not close to 0: {baseline_factor}. No destructive interference?")
        if abs(perturbed_factor) <= abs(baseline_factor):
            return (False, f"Perturbed factor magnitude did not increase: baseline {baseline_factor}, perturbed {perturbed_factor}")
        return (True, "Monotonicity confirmed: intensity increases away from destructive condition.")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one such method, no isclose, no re-derivation, no target_* comparisons)
    # ------------------------------------------------------------------
    def test_answer_sanity_final(self):
        """
        Weak checks: results dict is non-empty; if delta_s is numeric it should be non-negative.
        Does NOT use math.isclose or re-derive the condition.
        """
        issues = []
        if self.results is None:
            issues.append("No results dict.")
        elif len(self.results) == 0:
            issues.append("Results dict is empty.")

        if self.delta_s is not None and isinstance(self.delta_s, (int, float)):
            if self.delta_s < 0:
                issues.append(f"delta_s = {self.delta_s} is negative (should be non-negative).")

        if issues:
            return (False, "; ".join(issues))
        return (True, "Weak sanity checks passed: results exist, delta_s non-negative (if numeric).")

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)