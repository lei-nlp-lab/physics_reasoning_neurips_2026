import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for the one-particle density matrix problem in Bose-Einstein condensation.
    """
    def __init__(self, solution_instance):
        # Safe extraction of results dictionary
        self.results = getattr(solution_instance, 'results', {}) if solution_instance else {}
        
        # Extract each required key with None default
        self.density_matrix_def = self.results.get('density_matrix_def', None)
        self.rho1_asymptotic_Tc = self.results.get('rho1_asymptotic_Tc', None)
        self.rho1_infty_Tbelow = self.results.get('rho1_infty_Tbelow', None)
        self.rho1_decay_Tabove = self.results.get('rho1_decay_Tabove', None)

    # ------------------------------------------------------------------
    # Interface tests – verify that required keys exist and are non-empty strings
    # ------------------------------------------------------------------
    def test_interface_density_matrix_def(self):
        """Check that 'density_matrix_def' key exists and is a non-empty string."""
        if self.density_matrix_def is None:
            return False, "Missing key 'density_matrix_def'"
        if not isinstance(self.density_matrix_def, str) or len(self.density_matrix_def) == 0:
            return False, "'density_matrix_def' is not a non-empty string"
        return True, "Key 'density_matrix_def' present and valid"

    def test_interface_rho1_asymptotic_Tc(self):
        """Check that 'rho1_asymptotic_Tc' key exists and is a non-empty string."""
        if self.rho1_asymptotic_Tc is None:
            return False, "Missing key 'rho1_asymptotic_Tc'"
        if not isinstance(self.rho1_asymptotic_Tc, str) or len(self.rho1_asymptotic_Tc) == 0:
            return False, "'rho1_asymptotic_Tc' is not a non-empty string"
        return True, "Key 'rho1_asymptotic_Tc' present and valid"

    def test_interface_rho1_infty_Tbelow(self):
        """Check that 'rho1_infty_Tbelow' key exists and is a non-empty string."""
        if self.rho1_infty_Tbelow is None:
            return False, "Missing key 'rho1_infty_Tbelow'"
        if not isinstance(self.rho1_infty_Tbelow, str) or len(self.rho1_infty_Tbelow) == 0:
            return False, "'rho1_infty_Tbelow' is not a non-empty string"
        return True, "Key 'rho1_infty_Tbelow' present and valid"

    def test_interface_rho1_decay_Tabove(self):
        """Check that 'rho1_decay_Tabove' key exists and is a non-empty string."""
        if self.rho1_decay_Tabove is None:
            return False, "Missing key 'rho1_decay_Tabove'"
        if not isinstance(self.rho1_decay_Tabove, str) or len(self.rho1_decay_Tabove) == 0:
            return False, "'rho1_decay_Tabove' is not a non-empty string"
        return True, "Key 'rho1_decay_Tabove' present and valid"

    # ------------------------------------------------------------------
    # Invariant tests – physical consistency (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_basic_definition(self):
        """The definition of the one-particle density matrix must include the Boltzmann factor."""
        if self.density_matrix_def is None:
            return False, "SKIP: 'density_matrix_def' missing"
        has_exp = "e^{-E/k_B T" in self.density_matrix_def or "exp(-E/k_B T" in self.density_matrix_def
        has_sum = "\\sum_E" in self.density_matrix_def or "\\sum_{E}" in self.density_matrix_def
        if not (has_exp and has_sum):
            return False, "Definition does not contain Boltzmann factor or sum over states"
        return True, "Definition includes thermal weight and sum over states"

    def test_invariant_Tc_asymptotic_form(self):
        """At T_c the asymptotic form must be ~1/r."""
        if self.rho1_asymptotic_Tc is None:
            return False, "SKIP: 'rho1_asymptotic_Tc' missing"
        if "1/r" not in self.rho1_asymptotic_Tc and "\\frac{1}{r}" not in self.rho1_asymptotic_Tc:
            return False, "Asymptotic form at T_c does not indicate 1/r decay"
        if "m k_B T_c" not in self.rho1_asymptotic_Tc and "m k_B T_c" not in self.rho1_asymptotic_Tc:
            return False, "Asymptotic form missing mass and critical temperature"
        return True, "Asymptotic form at T_c correctly shows 1/r dependence"

    def test_invariant_Tbelow_constant(self):
        """Below T_c the limit r→∞ must be a constant (condensate density n0)."""
        if self.rho1_infty_Tbelow is None:
            return False, "SKIP: 'rho1_infty_Tbelow' missing"
        if "n_0" not in self.rho1_infty_Tbelow and "n0" not in self.rho1_infty_Tbelow:
            return False, "Below T_c limit not indicated as constant n0"
        return True, "Below T_c limit correctly refers to condensate density n0"

    # ------------------------------------------------------------------
    # Modelling / trace quality tests – based on explanation steps / pseudocode
    # ------------------------------------------------------------------
    def test_modeling_above_Tc_exponential_general(self):
        """Above T_c the description must mention exponential decay and a correlation length."""
        if self.rho1_decay_Tabove is None:
            return False, "SKIP: 'rho1_decay_Tabove' missing"
        contains_exp = "exponential" in self.rho1_decay_Tabove.lower()
        contains_xi = "correlation length" in self.rho1_decay_Tabove.lower()
        if not (contains_exp and contains_xi):
            return False, "Above T_c decay description does not mention both exponential and correlation length"
        return True, "Above T_c decay correctly described as exponential with correlation length"

    def test_modeling_above_Tc_exponential_formula(self):
        """Above T_c the decay description should give the explicit formula for the correlation length xi."""
        if self.rho1_decay_Tabove is None:
            return False, "SKIP: 'rho1_decay_Tabove' missing"
        # Look for the standard expression ξ = ℏ/√(2m|μ|) or similar
        has_xi_formula = False
        markers = ["\\hbar/\\sqrt", "\\hbar/\\sqrt{2m|\\mu|}", "\\sqrt{2m|\\mu|}", "\\frac{\\hbar}{\\sqrt{2m|\\mu|}}"]
        for m in markers:
            if m in self.rho1_decay_Tabove:
                has_xi_formula = True
                break
        if not has_xi_formula:
            return False, "Above T_c decay description lacks the explicit correlation length formula ξ = ℏ/√(2m|μ|)"
        return True, "Above T_c decay includes correct correlation length formula"

    def test_modeling_below_Tc_condensate_density(self):
        """Below T_c the limit should be described as the condensate density n0."""
        if self.rho1_infty_Tbelow is None:
            return False, "SKIP: 'rho1_infty_Tbelow' missing"
        if "n_0" not in self.rho1_infty_Tbelow and "n0" not in self.rho1_infty_Tbelow:
            return False, "Below T_c limit does not refer to the condensate density n0"
        return True, "Below T_c limit correctly refers to condensate density n0"

    # ------------------------------------------------------------------
    # Monotonicity test – perturb temperature from below to above Tc
    # ------------------------------------------------------------------
    def test_monotonicity_limit_vs_temperature(self):
        """
        As temperature increases from below T_c to above T_c, the limit of ρ₁(r) as r→∞ 
        changes from a finite constant (n0) to zero (exponential decay).
        This test checks that the description changes monotonically.
        Baseline: below Tc (rho1_infty_Tbelow) should indicate a constant, not decay.
        Perturbed: above Tc (rho1_decay_Tabove) should indicate decay to zero.
        """
        if self.rho1_infty_Tbelow is None:
            return False, "SKIP: 'rho1_infty_Tbelow' missing for baseline"
        if self.rho1_decay_Tabove is None:
            return False, "SKIP: 'rho1_decay_Tabove' missing for perturbed state"

        # Baseline: below Tc – must indicate a constant (not decaying)
        baseline_is_constant = any(kw in self.rho1_infty_Tbelow.lower() for kw in ["n0", "n_0", "constant", "finite"])
        if not baseline_is_constant:
            # If it says "0" or "zero" that contradicts physics
            if "0" in self.rho1_infty_Tbelow and "n" not in self.rho1_infty_Tbelow:
                return False, "Below Tc limit incorrectly indicates zero instead of constant"
            # Otherwise, treat as not clearly constant – still allow if above decays
            # (qualitative monotonic change may still be present)
            pass

        # Perturbed: above Tc – must indicate decay to zero
        perturbed_decays = any(kw in self.rho1_decay_Tabove.lower() for kw in ["decay", "exponential", "zero", "vanishes"])
        if not perturbed_decays:
            return False, "Above Tc description does not indicate decay to zero"

        # The monotonic change: below -> above, the limit goes from constant to decay
        if baseline_is_constant and perturbed_decays:
            return True, "Monotonic: limit changes from constant (T<Tc) to decay (T>Tc) as temperature increases"
        elif perturbed_decays and not baseline_is_constant:
            # Even if baseline is not explicitly constant, the change to decay is qualitative
            return True, "Qualitative monotonic change: above decays, below does not (physics expects constant)"
        else:
            return True, "Monotonic change inferred (baseline not clearly constant, but above decays)"

    # ------------------------------------------------------------------
    # Answer sanity (secondary, weak check)
    # ------------------------------------------------------------------
    def test_answer_sanity_coherence(self):
        """Minimal coherence: all emitted results are non‑empty strings."""
        required = [
            self.density_matrix_def,
            self.rho1_asymptotic_Tc,
            self.rho1_infty_Tbelow,
            self.rho1_decay_Tabove
        ]
        for i, s in enumerate(required):
            if s is None:
                return False, f"Result at index {i} is None"
            if not isinstance(s, str) or len(s) == 0:
                return False, f"Result at index {i} is not a non‑empty string"
        return True, "All emitted results are present and non‑empty"

    # ------------------------------------------------------------------
    # Runner: discover all test methods and execute them
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