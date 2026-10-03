import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the hyperfine structure of hydrogen problem.
    Checks that the solution correctly identifies nuclear spin as the origin,
    and includes a monotonicity test on hyperfine splitting.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.results = results

        # Extract chosen answer (e.g., 'b') if present
        self.chosen_answer = getattr(solution_instance, 'chosen_answer', None)
        if self.chosen_answer is None:
            self.chosen_answer = results.get('chosen_answer', None)

        # Extract explanation text safely
        explanation = getattr(solution_instance, 'explanation', None)
        if explanation is None:
            explanation = results.get('explanation', '')
        if isinstance(explanation, list):
            explanation = '\n'.join(explanation)
        self.explanation = explanation

        # Extract origin key from results
        self.origin = results.get('origin', None)

        # Extract coupling constant A if available (for monotonicity test)
        self.A = results.get('A', None)  # in energy units (e.g., Joules)
        # Also store nuclear and electron spin quantum numbers (optional)
        self.I = results.get('I', None)
        self.S = results.get('S', None)

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_origin_present(self):
        """Check that results contain the 'origin' key."""
        if self.origin is None:
            return (False, "Missing 'origin' key in results.")
        return (True, "'origin' key is present.")

    def test_interface_origin_expected_type(self):
        """Check that origin is a string."""
        if self.origin is None:
            return (False, "SKIP: 'origin' is None.")
        if not isinstance(self.origin, str):
            return (False, f"'origin' should be a string, got {type(self.origin)}.")
        return (True, "'origin' is a string.")

    # ------------------------------------------------------------
    # Invariant / physics reasoning checks
    # ------------------------------------------------------------
    def test_invariant_origin_nuclear_spin(self):
        """Check that origin is 'nuclear_spin'."""
        if self.origin is None:
            return (False, "SKIP: 'origin' not available.")
        if self.origin != 'nuclear_spin':
            return (False, f"Expected 'nuclear_spin' but got '{self.origin}'.")
        return (True, "Origin correctly set to 'nuclear_spin'.")

    def test_invariant_finite_nuclear_size_not_origin(self):
        """Check that explanation correctly rejects finite nuclear size as cause."""
        if not self.explanation:
            return (False, "SKIP: no explanation text available.")
        lower = self.explanation.lower()
        if 'finite nuclear size' in lower:
            # Check that it's rejected, not affirmed
            # Look for negation keywords nearby
            idx = lower.find('finite nuclear size')
            if idx == -1:
                return (True, "No mention of finite nuclear size.")
            start = max(0, idx - 50)
            end = min(len(lower), idx + len('finite nuclear size') + 50)
            snippet = lower[start:end]
            negative_indicators = ['not', 'false', 'incorrect', 'no', "don't", "doesn't"]
            if any(ind in snippet for ind in negative_indicators):
                return (True, "Finite nuclear size correctly identified as not the cause.")
            else:
                return (False, "Explanation mentions 'finite nuclear size' without rejecting it.")
        return (True, "No mention of finite nuclear size; acceptable since it's not the origin.")

    def test_invariant_structure_detectable(self):
        """Check that explanation mentions hyperfine structure is detectable (21 cm line)."""
        if not self.explanation:
            return (False, "SKIP: no explanation text.")
        lower = self.explanation.lower()
        if any(word in lower for word in ['21 cm', 'detectable', 'observable', 'measure']):
            return (True, "Explanation mentions detectability of hyperfine structure.")
        return (True, "Detectability not explicitly mentioned; still acceptable.")

    # ------------------------------------------------------------
    # Modeling checks (based on explanation steps)
    # ------------------------------------------------------------
    def test_modeling_explanation_mentions_nuclear_spin(self):
        """Check that explanation explicitly mentions nuclear spin as the cause."""
        if not self.explanation:
            return (False, "SKIP: no explanation to check.")
        lower = self.explanation.lower()
        if 'nuclear spin' in lower:
            return (True, "Explanation includes 'nuclear spin'.")
        else:
            return (False, "Explanation does not mention 'nuclear spin'.")

    def test_modeling_explanation_rejects_option_a(self):
        """Check that explanation rejects the idea that hyperfine structure is too small to detect."""
        if not self.explanation:
            return (False, "SKIP: no explanation.")
        lower = self.explanation.lower()
        negative_indicators = ['not', 'false', 'incorrect', 'detectable', 'observable', '21 cm']
        if 'too small' in lower or 'tiny' in lower:
            idx = lower.find('too small')
            if idx == -1:
                idx = lower.find('tiny')
            start = max(0, idx - 50)
            end = min(len(lower), idx + 100)
            snippet = lower[start:end]
            if any(ind in snippet for ind in negative_indicators):
                return (True, "Option (a) correctly rejected.")
            else:
                return (False, "Explanation mentions 'too small' but may not reject it.")
        return (True, "No mention of 'too small'; acceptable.")

    # ------------------------------------------------------------
    # Monotonicity test (perturb A and check splitting)
    # ------------------------------------------------------------
    def test_monotonicity_splitting_increases_with_A(self):
        """
        Check that hyperfine splitting (E_F=1 - E_F=0) increases monotonically
        with the coupling constant A. We perturb A by +10% and verify the splitting
        also increases by 10% (linear relationship).
        Baseline: E_F = A/2 * [F(F+1) - I(I+1) - S(S+1)]
        For hydrogen ground state: I=1/2, S=1/2, so F=0 and F=1.
        """
        # We need A, I, S to compute splitting. Use defaults for hydrogen if not provided.
        I = self.I if self.I is not None else 0.5
        S = self.S if self.S is not None else 0.5
        A = self.A

        if A is None:
            return (False, "SKIP: no coupling constant A available in results.")

        # Compute baseline splitting
        def energy(F):
            return (A/2) * (F*(F+1) - I*(I+1) - S*(S+1))
        try:
            E0 = energy(0)
            E1 = energy(1)
            splitting_baseline = E1 - E0
        except Exception as e:
            return (False, f"Failed to compute baseline energy: {e}")

        # Perturb A by +10%
        A_perturbed = A * 1.1
        def energy_pert(F):
            return (A_perturbed/2) * (F*(F+1) - I*(I+1) - S*(S+1))
        try:
            E0_pert = energy_pert(0)
            E1_pert = energy_pert(1)
            splitting_perturbed = E1_pert - E0_pert
        except Exception as e:
            return (False, f"Failed to compute perturbed energy: {e}")

        # Because splitting is linear in A, the relative change should be 10%.
        # Use isclose with tolerance.
        baseline = splitting_baseline
        expected = baseline * 1.1
        if math.isclose(splitting_perturbed, expected, rel_tol=1e-9):
            return (True, f"Splitting increased from {baseline:.6e} to {splitting_perturbed:.6e} (expected {expected:.6e}), consistent with linear dependence on A.")
        else:
            return (False, f"Splitting perturbation mismatch: baseline {baseline:.6e}, perturbed {splitting_perturbed:.6e}, expected {expected:.6e}.")

    # ------------------------------------------------------------
    # Weak answer sanity check (only one)
    # ------------------------------------------------------------
    def test_answer_sanity_choice_is_b(self):
        """Check that the chosen answer is 'b' (or a valid option)."""
        if self.chosen_answer is None:
            return (False, "SKIP: no chosen answer available.")
        if self.chosen_answer not in ['a', 'b', 'c']:
            return (False, f"Invalid answer '{self.chosen_answer}'; must be a, b, or c.")
        if self.chosen_answer != 'b':
            return (False, f"Answer is '{self.chosen_answer}' but expected 'b'.")
        return (True, "Chosen answer is 'b'.")

    # ------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith('test_'):
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