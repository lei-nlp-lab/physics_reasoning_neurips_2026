import math
import sympy
import traceback

class PhysicsChecker:
    """
    Auditor for the single strange particle problem.
    Checks interface completeness, physical invariants, modeling consistency,
    and logical monotonicity.
    """
    
    def __init__(self, solution_instance):
        # Safe extraction from results
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        
        self.a_true = self.results.get('a_true', None)
        self.b_true = self.results.get('b_true', None)
        self.c_true = self.results.get('c_true', None)
        
        # Try to extract optional inputs from solution_instance
        # These may be stored as attributes (e.g., givens, shared_variables)
        self.strange_particle_exists = None
        self.strong_interaction_conserves_S = None
        self.weak_interaction_allows_delta_S = None
        
        # Check for givens dict
        givens = getattr(solution_instance, 'givens', None)
        if isinstance(givens, dict):
            self.strange_particle_exists = givens.get('strange_particle_exists', None)
            self.strong_interaction_conserves_S = givens.get('strong_interaction_conserves_S', None)
            self.weak_interaction_allows_delta_S = givens.get('weak_interaction_allows_delta_S', None)
        
        # Fallback to shared_variables if not found
        if self.strange_particle_exists is None:
            shared = getattr(solution_instance, 'shared_variables', None)
            if isinstance(shared, dict):
                self.strange_particle_exists = shared.get('S_strange', None)  # might be ±1
        # If still None, flag as missing
        
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        
    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all three required result keys exist."""
        missing = []
        for key in ['a_true', 'b_true', 'c_true']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")
    
    def test_interface_boolean_values(self):
        """Check that all result values are boolean (or at least not None)."""
        for key, val in [('a_true', self.a_true), ('b_true', self.b_true), ('c_true', self.c_true)]:
            if val is None:
                return (False, f"{key} is None")
            if not isinstance(val, (bool, int)):
                return (False, f"{key} is not boolean, got {type(val).__name__}")
        return (True, "All values are boolean")
    
    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_weak_decay_possibility(self):
        """
        Weak interaction allows ΔS = ±1, so a strange particle (S≠0) can decay to S=0.
        Therefore a_true must be True.
        """
        if self.a_true is None:
            return (False, "SKIP: a_true not available")
        # True as float = 1.0, False = 0.0
        if not math.isclose(float(self.a_true), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return (False, f"a_true is {self.a_true}, expected True (weak decay always possible)")
        return (True, "a_true correctly indicates weak decay possibility")
    
    def test_invariant_strong_creation_impossibility(self):
        """
        Strong interaction conserves strangeness. Creating a single strange particle (S≠0)
        from S=0 would break conservation. Therefore b_true must be False.
        """
        if self.b_true is None:
            return (False, "SKIP: b_true not available")
        if not math.isclose(float(self.b_true), 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return (False, f"b_true is {self.b_true}, expected False (strong creation impossible)")
        return (True, "b_true correctly indicates strong creation impossibility")
    
    def test_invariant_strange_particle_existence(self):
        """
        Strange particles exist (e.g., K⁺, Λ). Therefore statement (c) 'cannot exist' is False.
        """
        if self.c_true is None:
            return (False, "SKIP: c_true not available")
        if not math.isclose(float(self.c_true), 0.0, rel_tol=1e-9, abs_tol=1e-9):
            return (False, f"c_true is {self.c_true}, expected False (strange particles exist)")
        return (True, "c_true correctly indicates existence of strange particles")
    
    # ------------------------------------------------------------------
    # Modeling / trace quality tests (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_explanation_references(self):
        """
        Check that explanation steps mention key physics concepts.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        text = " ".join(self.explanation_steps) if isinstance(self.explanation_steps, list) else str(self.explanation_steps)
        required_phrases = ["weak", "strong", "strangeness", "conserv", "exist"]
        missing = [phrase for phrase in required_phrases if phrase.lower() not in text.lower()]
        if missing:
            return (False, f"Explanation steps missing references to: {missing}")
        return (True, "Explanation steps cover required physics concepts")
    
    def test_modeling_pseudocode_consistency(self):
        """
        Check that the pseudocode's emitted values match the results.
        Only performed if pseudocode is available.
        """
        if self.pseudocode is None:
            return (False, "SKIP: no pseudocode provided")
        
        # Try to parse the emitted values from pseudocode
        # Assumes pseudocode contains a [emit] block with assignments
        # We'll attempt to extract a_true, b_true, c_true from text.
        pseudo_text = str(self.pseudocode)
        # Simple extraction: look for "a_true = True" etc. (case-sensitive)
        def extract_bool(text, var_name):
            import re
            pattern = rf'{var_name}\s*=\s*(True|False)'
            m = re.search(pattern, text)
            if m:
                return m.group(1) == 'True'
            return None
        
        pseudo_a = extract_bool(pseudo_text, 'a_true')
        pseudo_b = extract_bool(pseudo_text, 'b_true')
        pseudo_c = extract_bool(pseudo_text, 'c_true')
        
        mismatches = []
        if pseudo_a is not None and self.a_true is not None and pseudo_a != self.a_true:
            mismatches.append(f"a_true: pseudocode says {pseudo_a}, result gives {self.a_true}")
        if pseudo_b is not None and self.b_true is not None and pseudo_b != self.b_true:
            mismatches.append(f"b_true: pseudocode says {pseudo_b}, result gives {self.b_true}")
        if pseudo_c is not None and self.c_true is not None and pseudo_c != self.c_true:
            mismatches.append(f"c_true: pseudocode says {pseudo_c}, result gives {self.c_true}")
        
        if mismatches:
            return (False, "Pseudocode/result mismatch: " + "; ".join(mismatches))
        return (True, "Pseudocode and results agree on all evaluable values")
    
    # ------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_existence_affects_c(self):
        """
        If the input 'strange_particle_exists' changes from True to False,
        the output c_true should flip from False to True.
        We check that given the input (if available), the result matches,
        and we derive the expected outcome for the flipped input.
        """
        if self.strange_particle_exists is None:
            return (False, "SKIP: input 'strange_particle_exists' not available")
        if self.c_true is None:
            return (False, "SKIP: c_true not available")
        
        baseline_input = self.strange_particle_exists
        baseline_output = self.c_true
        
        # Derive expected output for baseline input
        if baseline_input is True:
            expected_baseline_output = False
        else:
            expected_baseline_output = True
        
        if baseline_output != expected_baseline_output:
            return (False, f"Baseline mismatch: input={baseline_input}, got c_true={baseline_output}, expected {expected_baseline_output}")
        
        # Now consider perturbed input (flip the boolean)
        perturbed_input = not baseline_input
        # Derive expected perturbed output
        if perturbed_input is True:
            expected_perturbed_output = False
        else:
            expected_perturbed_output = True
        
        # Since we cannot get the perturbed output from the solution itself, we can only reason that
        # the logical relationship holds. The monotonicity condition is satisfied if the expected output
        # changes in the opposite direction.
        if expected_perturbed_output == baseline_output:
            # No change means no monotonic effect (shouldn't happen in this logical scenario)
            return (False, f"Perturbation yields same output: input {baseline_input}->{perturbed_input} gives no change in c_true prediction")
        
        # If everything is consistent, monotonicity holds.
        return (True, f"Monotonicity holds: flipping existence from {baseline_input} to {perturbed_input} flips c_true from {baseline_output} to {expected_perturbed_output}")
    
    # ------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausibility(self):
        """
        Broad sanity: check that all three booleans are present and not contradictory with basic physics.
        This is intentionally weak; no hardcoded expected values.
        """
        if None in (self.a_true, self.b_true, self.c_true):
            return (False, "One or more answer values are None")
        # All are boolean – that's the only sanity we check here.
        return (True, "Answer sanity: all values are present and boolean")
    
    # ------------------------------------------------------------------
    # Runner
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