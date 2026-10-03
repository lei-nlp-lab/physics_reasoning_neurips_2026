import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for the hypothetical helium atom degeneracy problem.
    Two identical spin-1 bosons occupy the symmetric spatial ground state.
    The total wavefunction must be symmetric, so only symmetric spin states count.
    Symmetric spin states arise from total spin S=0 and S=2.
    """

    def __init__(self, solution_instance):
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.N_symmetric = self.results.get('N_symmetric', None)

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Givens (optional)
        self.givens = getattr(solution_instance, 'givens', {})

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

    # ------------------------------------------------------------------
    # INTERFACE CHECKS
    # ------------------------------------------------------------------
    def test_interface_required_key_present(self):
        """Check that the emitted key 'N_symmetric' exists in results."""
        if self.N_symmetric is None:
            return (False, "Missing required key 'N_symmetric' in results.")
        return (True, "Key 'N_symmetric' present.")

    def test_interface_numeric_type(self):
        """Check that N_symmetric is a number (int or float)."""
        if self.N_symmetric is None:
            return (False, "SKIP: N_symmetric not available.")
        if not isinstance(self.N_symmetric, (int, float)):
            return (False, f"N_symmetric has type {type(self.N_symmetric)}, expected int or float.")
        return (True, "N_symmetric is numeric.")

    # ------------------------------------------------------------------
    # THEOREM-BASED INVARIANT CHECKS (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_symmetric_spin_count(self):
        """
        Re-derive the number of symmetric spin states for two spin-1 bosons.
        Ground state spatial part is symmetric => spin part must be symmetric.
        For two spin-1 particles, symmetric spin states come from total spin S=0 (dim 1)
        and S=2 (dim 5). Total = 6.
        """
        if self.N_symmetric is None:
            return (False, "SKIP: N_symmetric not available.")

        # Derive expected count from first principles
        dim_S0 = 1  # 2*0+1 = 1
        dim_S2 = 5  # 2*2+1 = 5
        expected = dim_S0 + dim_S2

        if not math.isclose(self.N_symmetric, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"N_symmetric = {self.N_symmetric}, but re-derived value = {expected}.")
        return (True, f"N_symmetric = {self.N_symmetric} matches re-derived value {expected}.")

    def test_invariant_spin_addition_dimensions(self):
        """
        Verify that the total spin values 0,1,2 are possible from spin addition
        of two spin-1 particles by checking dimension consistency.
        """
        dim_total = (2 * 1 + 1) ** 2  # 9
        dim_sum = sum(2 * S + 1 for S in [0, 1, 2])  # 1+3+5 = 9
        if dim_total != dim_sum:
            return (False, f"Spin addition dimensions mismatch: {dim_total} vs {dim_sum}.")
        return (True, "Spin addition dimensions are consistent (9 = 1+3+5).")

    # ------------------------------------------------------------------
    # MODELING / EXPLANATION-STEP CHECKS
    # ------------------------------------------------------------------
    def test_modeling_spatial_symmetry_mentioned(self):
        """Check that explanation steps mention spatial wavefunction symmetry."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided.")
        text = ""
        if isinstance(self.explanation_steps, list):
            text = " ".join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        else:
            return (False, "SKIP: explanation_steps has unsupported type.")

        keywords = ["spatial", "symmetric"]
        if not all(kw in text.lower() for kw in keywords):
            return (False, "Explanation steps do not mention spatial symmetry.")
        return (True, "Explanation steps mention spatial symmetry.")

    def test_modeling_boson_symmetry_mentioned(self):
        """Check that explanation steps mention boson statistics or symmetric total wavefunction."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided.")
        text = ""
        if isinstance(self.explanation_steps, list):
            text = " ".join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        else:
            return (False, "SKIP: explanation_steps has unsupported type.")

        if "boson" not in text.lower() and "bose" not in text.lower():
            return (False, "Explanation steps do not mention boson statistics.")
        if "symmetric" not in text.lower():
            return (False, "Explanation steps do not mention symmetry condition.")
        return (True, "Explanation steps mention boson symmetry requirement.")

    def test_modeling_symmetric_spin_states_listed(self):
        """Check that explanation steps correctly list symmetric spin states S=0 and S=2, and note S=1 as antisymmetric."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided.")
        text = ""
        if isinstance(self.explanation_steps, list):
            text = " ".join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        else:
            return (False, "SKIP: explanation_steps has unsupported type.")

        if "S = 0" not in text and "S=0" not in text:
            return (False, "S=0 not mentioned as a possible total spin.")
        if "S = 2" not in text and "S=2" not in text:
            return (False, "S=2 not mentioned as a possible total spin.")
        if ("S = 1" not in text and "S=1" not in text) or "antisymmetric" not in text.lower():
            return (False, "Antisymmetry of S=1 not clearly stated.")
        return (True, "Symmetric (S=0, S=2) and antisymmetric (S=1) states correctly described.")

    # ------------------------------------------------------------------
    # MONOTONICITY CHECK (explicit perturbation of a parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_spin_increase_increases_degeneracy(self):
        """
        For two identical spin-j bosons in a symmetric spatial state,
        the number of symmetric spin states is N(j) = (j+1)(2j+1), which is
        increasing for j >= 0. Perturb j = 1 to j = 1.1 and verify that
        the derived N increases. This test does not rely on the provided answer.
        """
        # Baseline: j = 1 -> N = 6
        j_base = 1
        N_base = (j_base + 1) * (2 * j_base + 1)

        # Perturb j by +0.1 (approximately +10%)
        j_pert = j_base + 0.1
        N_pert = (j_pert + 1) * (2 * j_pert + 1)

        if N_pert <= N_base:
            return (False, f"Monotonicity violated: N_pert={N_pert} <= N_base={N_base}")
        return (True, f"Degeneracy increases with spin: base={N_base}, perturbed (j=1.1)={N_pert}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY CHECK (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite_reasonable(self):
        """
        Weak sanity: N_symmetric must be positive, finite, and within a
        reasonable broad magnitude range (small integer, not huge).
        """
        if self.N_symmetric is None:
            return (False, "SKIP: N_symmetric not available.")
        if self.N_symmetric <= 0:
            return (False, f"N_symmetric = {self.N_symmetric} is not positive.")
        if math.isnan(self.N_symmetric):
            return (False, "N_symmetric is NaN.")
        if math.isinf(self.N_symmetric):
            return (False, "N_symmetric is infinite.")
        # Reasonable range: between 1 and 10 inclusive
        if not (1 <= self.N_symmetric <= 10):
            return (False, f"N_symmetric = {self.N_symmetric} is outside the reasonable range [1,10].")
        return (True, "N_symmetric is positive, finite, and in reasonable magnitude range.")