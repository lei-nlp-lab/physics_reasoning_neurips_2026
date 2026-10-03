import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    PhysicsChecker for the nuclear force spin dependence problem.
    Checks that the solution correctly applies the Pauli principle and spin-dependent potential
    to explain the absence of a bound di-neutron state.
    """
    def __init__(self, solution_instance):
        # Try to get results dictionary (common pattern)
        try:
            results = solution_instance.results
        except AttributeError:
            results = {}
        # Fall back to direct attributes if keys are missing
        # Use getattr with None default for each expected key
        self.allowed_nn_spin = results.get('allowed_nn_spin', getattr(solution_instance, 'allowed_nn_spin', None))
        self.deuteron_spin = results.get('deuteron_spin', getattr(solution_instance, 'deuteron_spin', None))
        self.conclusion = results.get('conclusion', getattr(solution_instance, 'conclusion', None))

        # Also store the full explanation steps if provided (for consistency checks)
        try:
            self.explanation_steps = solution_instance.explanation_steps
        except AttributeError:
            self.explanation_steps = None

    # ------------------------------------------------------------------
    # Interface tests: check that required keys exist and have correct types
    # ------------------------------------------------------------------
    def test_interface_allowed_nn_spin(self):
        """Check that allowed_nn_spin is present and is a string."""
        if self.allowed_nn_spin is None:
            return (False, "SKIP: missing allowed_nn_spin")
        if not isinstance(self.allowed_nn_spin, str):
            return (False, f"allowed_nn_spin should be str, got {type(self.allowed_nn_spin)}")
        return (True, "allowed_nn_spin is present and a string")

    def test_interface_deuteron_spin(self):
        """Check that deuteron_spin is present and is a string."""
        if self.deuteron_spin is None:
            return (False, "SKIP: missing deuteron_spin")
        if not isinstance(self.deuteron_spin, str):
            return (False, f"deuteron_spin should be str, got {type(self.deuteron_spin)}")
        return (True, "deuteron_spin is present and a string")

    def test_interface_conclusion(self):
        """Check that conclusion is present and is a string."""
        if self.conclusion is None:
            return (False, "SKIP: missing conclusion")
        if not isinstance(self.conclusion, str):
            return (False, f"conclusion should be str, got {type(self.conclusion)}")
        return (True, "conclusion is present and a string")

    # ------------------------------------------------------------------
    # Invariant tests: fundamental physics principles
    # ------------------------------------------------------------------
    def test_invariant_pauli_nn_singlet(self):
        """
        Pauli principle for identical fermions (nn) at L=0 forces spin singlet.
        The solution must state that nn can only be in singlet state.
        """
        if self.allowed_nn_spin is None:
            return (False, "SKIP: allowed_nn_spin not available")
        # Accept common variants: 'singlet', 'S=0', 'singlet state'
        if self.allowed_nn_spin.lower() not in ['singlet', 's=0', 'singlet state']:
            return (False, f"expected nn allowed spin to be singlet, got '{self.allowed_nn_spin}'")
        return (True, "allowed_nn_spin correctly identifies singlet state")

    def test_invariant_deuteron_triplet(self):
        """
        The known bound state of np (deuteron) has spin 1 (triplet).
        The solution must acknowledge this.
        """
        if self.deuteron_spin is None:
            return (False, "SKIP: deuteron_spin not available")
        if self.deuteron_spin.lower() not in ['triplet', 's=1', 'triplet state']:
            return (False, f"expected deuteron spin to be triplet, got '{self.deuteron_spin}'")
        return (True, "deuteron_spin correctly identifies triplet state")

    def test_invariant_spin_dependence_in_conclusion(self):
        """
        The conclusion must clearly state that the nuclear force is spin dependent.
        """
        if self.conclusion is None:
            return (False, "SKIP: conclusion not available")
        if 'spin' not in self.conclusion.lower():
            return (False, "conclusion does not mention spin dependence")
        # Check for phrases like "spin dependent" or "spin dependence"
        if 'spin depend' not in self.conclusion.lower():
            return (False, "conclusion should explicitly state spin dependence")
        return (True, "conclusion explicitly mentions spin dependence")

    # ------------------------------------------------------------------
    # Modeling/trace tests: consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_charge_independence_acknowledged(self):
        """
        The explanation should acknowledge that the strong force is charge independent.
        Check the conclusion or explanation steps for this.
        """
        # Check conclusion
        if self.conclusion and 'charge independ' in self.conclusion.lower():
            return (True, "conclusion mentions charge independence")
        # Also check explanation steps if available
        if self.explanation_steps is not None and isinstance(self.explanation_steps, (list, str)):
            text = ''.join(self.explanation_steps) if isinstance(self.explanation_steps, list) else self.explanation_steps
            if 'charge independ' in text.lower():
                return (True, "explanation steps mention charge independence")
        return (False, "neither conclusion nor explanation steps mention charge independence of nuclear force")

    def test_modeling_pauli_exclusion_used(self):
        """
        The solution must explicitly invoke the Pauli exclusion principle for identical fermions.
        Check explanation steps.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        text = ''.join(self.explanation_steps) if isinstance(self.explanation_steps, list) else self.explanation_steps
        if 'pauli' in text.lower():
            return (True, "Pauli principle is referenced")
        return (False, "Pauli principle not mentioned in explanation steps")

    def test_modeling_antisymmetry_logic(self):
        """
        Check that the antisymmetry requirement (space symmetric -> spin antisymmetric) is present.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        text = ''.join(self.explanation_steps) if isinstance(self.explanation_steps, list) else self.explanation_steps
        # Look for keywords: antisymmetric, symmetric, L=0, s-wave
        if 'antisymmetric' in text.lower() and 'symmetric' in text.lower():
            return (True, "antisymmetry argument is present")
        return (False, "argument about space symmetric requiring spin antisymmetric is missing")

    # ------------------------------------------------------------------
    # Monotonicity test: perturb a physical parameter and check monotonic behavior
    # ------------------------------------------------------------------
    def test_monotonicity_spin_dependent_potential(self):
        """
        Test that increasing the spin-dependent strength V_sigma increases the gap
        between triplet and singlet potentials.
        Use a simple model: V_singlet = V0 - 3*V_sigma, V_triplet = V0 + V_sigma.
        Perturb V_sigma by +10% and verify that V_triplet - V_singlet increases.
        This does not use any solution data; it's a standalone physical consistency check.
        """
        # Define symbols and baseline
        V0 = 1.0  # central potential strength (arbitrary)
        V_sigma_baseline = 1.0  # spin-dependent strength
        perturb_factor = 1.1
        V_sigma_perturbed = V_sigma_baseline * perturb_factor

        V_singlet_baseline = V0 - 3 * V_sigma_baseline
        V_triplet_baseline = V0 + V_sigma_baseline
        gap_baseline = V_triplet_baseline - V_singlet_baseline

        V_singlet_perturbed = V0 - 3 * V_sigma_perturbed
        V_triplet_perturbed = V0 + V_sigma_perturbed
        gap_perturbed = V_triplet_perturbed - V_singlet_perturbed

        if math.isclose(gap_perturbed, gap_baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Gap did not change significantly after perturbing V_sigma; expected increase")
        # Check that gap increased (stronger spin dependence makes triplet relatively more bound)
        if gap_perturbed <= gap_baseline:
            return (False, f"Gap decreased: baseline {gap_baseline}, perturbed {gap_perturbed}")
        return (True, f"Gap increases from {gap_baseline} to {gap_perturbed} as V_sigma increases")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, non-dominant)
    # ------------------------------------------------------------------
    def test_answer_sanity_conclusion_plausible(self):
        """
        The conclusion should be a non-empty, reasonable statement (no contradiction).
        Only perform a very weak check: it should be a string and not contain obviously wrong statements.
        """
        if self.conclusion is None:
            return (False, "SKIP: conclusion missing")
        if not isinstance(self.conclusion, str):
            return (False, "conclusion is not a string")
        # Check that it does not contain obvious nonsense (e.g., "nn bound state exists")
        if 'bound state' in self.conclusion.lower() and 'nn' in self.conclusion.lower():
            # Could be "no bound state for nn" which is fine, but need to ensure it states absent
            if 'no bound' in self.conclusion.lower() or 'not' in self.conclusion.lower():
                pass
            else:
                return (False, "conclusion might incorrectly imply nn bound state exists")
        return (True, "conclusion is plausible non-empty string")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_ methods and execute them, collecting results."""
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)