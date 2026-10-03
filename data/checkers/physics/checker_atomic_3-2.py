import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics auditor for the electrostatic force between Earth and Moon problem.
    Checks neutrality reasoning, Coulomb's law, and interface completeness.
    """

    def __init__(self, solution_instance):
        # Safely extract all relevant attributes, defaulting to None
        self.results = getattr(solution_instance, 'results', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.selected_answer = getattr(solution_instance, 'selected_answer', None)

        # Try to extract givens if they exist (not strictly required)
        givens = getattr(solution_instance, 'givens', None) or {}
        # Physical quantities – all default to None (no numeric fallbacks)
        self.Q_e = givens.get('Q_e', None) or getattr(solution_instance, 'Q_e', None)
        self.Q_m = givens.get('Q_m', None) or getattr(solution_instance, 'Q_m', None)
        self.M_e = givens.get('M_e', None) or getattr(solution_instance, 'M_e', None)
        self.M_m = givens.get('M_m', None) or getattr(solution_instance, 'M_m', None)
        self.r = givens.get('r', None) or getattr(solution_instance, 'r', None)
        # Constants – must be explicitly provided; do NOT assign numeric defaults
        self.G = givens.get('G', None) or getattr(solution_instance, 'G', None)
        self.k = givens.get('k', None) or getattr(solution_instance, 'k', None)

        # Result quantity from the pseudocode emit contract
        self.F_e = self.results.get('F_e', None)

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
        return all_pass, logs

    # --- Interface tests ---
    def test_interface_results_dict_exists(self):
        """Check that solution has a results dictionary."""
        if self.results is None:
            return False, "results dictionary is missing"
        return True, "results dictionary present"

    def test_interface_F_e_in_results(self):
        """Check that 'F_e' is present in results."""
        if self.results is None:
            return False, "SKIP: results not available"
        if 'F_e' not in self.results:
            return False, "'F_e' key missing in results"
        return True, "'F_e' key present in results"

    # --- Invariant (physics theorem) tests ---
    def test_invariant_electrostatic_force_zero_by_neutrality(self):
        """
        If net charges of Earth and Moon are given, verify that they are effectively zero,
        which implies electrostatic force is zero.
        """
        if self.Q_e is None or self.Q_m is None:
            return False, "SKIP: Q_e and Q_m not provided, cannot check neutrality"
        # Check neutrality: charges are approximately zero
        if self.Q_e is not None and not math.isclose(self.Q_e, 0.0, abs_tol=1e-20):
            return False, f"Earth net charge Q_e = {self.Q_e} is not negligible"
        if self.Q_m is not None and not math.isclose(self.Q_m, 0.0, abs_tol=1e-20):
            return False, f"Moon net charge Q_m = {self.Q_m} is not negligible"
        return True, "Earth and Moon appear electrically neutral"

    def test_invariant_F_e_negligible(self):
        """
        Check that the reported electrostatic force F_e is essentially zero.
        Use math.isclose with a very tight tolerance because in this ideal case it should be exactly zero.
        """
        if self.F_e is None:
            return False, "SKIP: F_e not available"
        if not math.isclose(self.F_e, 0.0, abs_tol=1e-12):
            return False, f"F_e = {self.F_e} is not negligible (should be ≈ 0)"
        return True, "F_e is effectively zero"

    # --- Modeling / trace test (from explanation steps) ---
    def test_modeling_reasoning_contains_neutrality(self):
        """Ensure that the explanation steps mention the concept of electrical neutrality."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not provided"
        text = self.explanation_steps
        if isinstance(text, list):
            text = " ".join(text)
        if not isinstance(text, str):
            return False, "explanation_steps is not a string or list of strings"
        if "neutral" not in text.lower():
            return False, "explanation does not mention electrical neutrality"
        return True, "Explanation references neutrality"

    def test_modeling_pseudocode_emit_F_e_zero(self):
        """Check that the emitted F_e is exactly zero as per pseudocode contract."""
        if self.F_e is None:
            return False, "SKIP: F_e not available"
        # Compare numerically (exact zero expected, but allow tiny float error)
        if abs(self.F_e) > 1e-12:
            return False, f"F_e = {self.F_e} is not zero as per emit contract"
        return True, "F_e equals zero (pseudocode emit contract satisfied)"

    # --- Monotonicity test (explicit input perturbation) ---
    def test_monotonicity_F_e_increases_with_charge(self):
        """
        Check that increasing the magnitude of net charge increases the magnitude of electrostatic force.
        Perturb Q_e by 10% (or to a small non-zero if it was zero) and recompute.
        """
        if self.k is None or self.r is None:
            return (False, "SKIP: missing Coulomb constant k or distance r")
        if self.Q_e is None and self.Q_m is None:
            return (False, "SKIP: neither Q_e nor Q_m provided, cannot perturb")

        # Use provided charges; treat missing ones as zero
        Q_e_base = self.Q_e if self.Q_e is not None else 0.0
        Q_m_base = self.Q_m if self.Q_m is not None else 0.0

        # Compute baseline force (could be zero)
        try:
            F_e_base = self.k * abs(Q_e_base * Q_m_base) / (self.r ** 2)
        except (ZeroDivisionError, TypeError) as e:
            return (False, f"SKIP: division or type error in baseline: {e}")

        # Perturb one charge (increase magnitude by 10% or set to a small non-zero if zero)
        if Q_e_base != 0.0:
            Q_e_pert = Q_e_base * 1.1   # increase by 10%
        else:
            Q_e_pert = 1e-10            # set to a tiny but non-zero value
        try:
            F_e_pert = self.k * abs(Q_e_pert * Q_m_base) / (self.r ** 2)
        except (ZeroDivisionError, TypeError) as e:
            return (False, f"SKIP: division or type error in perturbed: {e}")

        # Compare magnitudes
        if abs(F_e_pert) >= abs(F_e_base):
            return (True, f"Perturbed |F_e| = {abs(F_e_pert):.6e} >= baseline |F_e| = {abs(F_e_base):.6e}, monotonicity holds")
        else:
            return (False, f"Perturbed |F_e| = {abs(F_e_pert):.6e} < baseline |F_e| = {abs(F_e_base):.6e}")

    # --- Answer sanity test (weak, secondary) ---
    def test_answer_sanity_option_exists(self):
        """Weak sanity: check that an answer option was selected and is a plausible single letter."""
        if self.selected_answer is None:
            return False, "SKIP: selected_answer not available"
        if not isinstance(self.selected_answer, str):
            return False, "selected_answer is not a string"
        if self.selected_answer not in ("a", "b", "c"):
            return False, f"selected_answer '{self.selected_answer}' is not among expected options (a, b, c)"
        return True, f"selected_answer '{self.selected_answer}' is a valid option"