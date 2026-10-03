import math
import traceback
import sympy as sp
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Audits solutions for electrostatic self-energy of a uniform sphere
    and radius estimate from mirror nuclei mass difference.
    """

    # Fundamental constants (SI units)
    k_e = 8.9875517923e9      # Coulomb constant (N·m²/C²)
    e = 1.602176634e-19       # Elementary charge (C)
    MeV_to_J = 1.602176634e-13  # 1 MeV in Joules
    fm_to_m = 1e-15           # 1 femtometer in meters

    def __init__(self, solution_instance: Any):
        # Safely extract results dictionary, givens, and explanation steps
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Required emitted quantities
        self.U = self.results.get('U', None)
        self.R = self.results.get('R', None)

        # Additional information that might be useful
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        self.pseudocode = getattr(solution_instance, 'pseudocode', "")

    # ----------------------------------------------------------------------
    # Interface tests: ensure required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_results_U(self):
        """Check that results['U'] exists and is a numeric type."""
        if self.U is None:
            return (False, "result 'U' is missing")
        if not isinstance(self.U, (int, float, sp.Expr)):
            return (False, f"U has invalid type {type(self.U)}")
        return (True, "U present and has valid type")

    def test_interface_results_R(self):
        """Check that results['R'] exists and is numeric."""
        if self.R is None:
            return (False, "result 'R' is missing")
        if not isinstance(self.R, (int, float)):
            return (False, f"R has invalid type {type(self.R)}")
        return (True, "R present and numeric")

    # ----------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_U_positive(self):
        """The electrostatic energy U must be positive for a positive Q."""
        if self.U is None:
            return (False, "SKIP: U not available")
        if not isinstance(self.U, (int, float)):
            # Could be symbolic; skip if not numeric
            return (False, "SKIP: U is not numeric")
        if self.U < 0:
            return (False, f"U is negative: {self.U}")
        return (True, f"U is positive ({self.U})")

    def test_invariant_R_positive(self):
        """Radius R must be positive."""
        if self.R is None:
            return (False, "SKIP: R not available")
        if not math.isfinite(self.R):
            return (False, f"R is not finite: {self.R}")
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}")
        return (True, f"R is positive ({self.R} fm)")

    # ----------------------------------------------------------------------
    # Modeling/trace tests: check consistency with explanation steps
    # ----------------------------------------------------------------------
    def test_modeling_U_formula_provided(self):
        """
        If the solution provides a symbolic expression for U,
        verify it matches the standard uniform sphere formula.
        """
        if not isinstance(self.U, sp.Expr):
            return (False, "SKIP: U is not symbolic")
        try:
            # Define symbolic variables
            Q, R = sp.symbols('Q R')
            expected_U = (3/sp.Integer(5)) * self.k_e * Q**2 / R
            # Compare simplified difference
            diff = sp.simplify(self.U - expected_U)
            if diff == 0:
                return (True, "Symbolic U matches expected formula")
            else:
                return (False, f"Symbolic U deviates: difference = {diff}")
        except Exception as e:
            return (False, f"Error during symbolic comparison: {e}")

    def test_modeling_explanation_steps_mention_energy(self):
        """Explanation steps should mention electrostatic energy or Coulomb."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation steps provided")
        steps_text = " ".join(str(s) for s in self.explanation_steps).lower()
        keywords = ['electrostatic', 'coulomb', 'energy', 'u =']
        found = any(kw in steps_text for kw in keywords)
        if found:
            return (True, "Explanation steps include relevant physics terms")
        else:
            return (False, "Explanation steps lack key terms (electrostatic, Coulomb, energy)")

    # ----------------------------------------------------------------------
    # Monotonicity tests: perturb input and check trend
    # ----------------------------------------------------------------------
    def test_monotonicity_R_vs_delta_E(self):
        """
        Increase the mass difference ΔE by 10% -> radius should decrease (inverse proportionality).
        R is proportional to 1/ΔE.  So for δΔE > 0, δR < 0.
        """
        if self.R is None:
            return (False, "SKIP: R not available")
        # Recompute expected R for perturbed ΔE
        delta_Z2 = 14**2 - 13**2
        base_delta_E = 6 * self.MeV_to_J
        # 10% increase
        perturbed_delta_E = base_delta_E * 1.1
        perturbed_R_m = (3.0 / 5.0) * self.k_e * (self.e ** 2) * delta_Z2 / perturbed_delta_E
        perturbed_R_fm = perturbed_R_m / self.fm_to_m
        # Check that perturbed R is smaller than original
        if perturbed_R_fm < self.R:
            return (True, f"Perturbed R ({perturbed_R_fm:.4f} fm) < original ({self.R:.4f} fm) as expected")
        else:
            return (False, f"Perturbed R ({perturbed_R_fm:.4f} fm) not smaller than original ({self.R:.4f} fm)")

    def test_monotonicity_R_vs_Z2_difference(self):
        """
        If the difference in Z² were larger (e.g., 15²-13²=56 instead of 27),
        the radius should be larger (proportional to ΔZ²).
        """
        if self.R is None:
            return (False, "SKIP: R not available")
        # Compute expected R for a larger ΔZ²
        base_delta_E = 6 * self.MeV_to_J
        larger_delta_Z2 = 15**2 - 13**2  # = 56
        larger_R_m = (3.0 / 5.0) * self.k_e * (self.e ** 2) * larger_delta_Z2 / base_delta_E
        larger_R_fm = larger_R_m / self.fm_to_m
        if larger_R_fm > self.R:
            return (True, f"Larger ΔZ² gives larger R ({larger_R_fm:.4f} fm > {self.R:.4f} fm)")
        else:
            return (False, f"Larger ΔZ² does not increase R ({larger_R_fm:.4f} fm vs {self.R:.4f} fm)")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_R_range(self):
        """Radius should be in a physically reasonable range (~1-10 fm)."""
        if self.R is None:
            return (False, "SKIP: R not available")
        if not math.isfinite(self.R):
            return (False, f"R is not finite: {self.R}")
        # Typical nuclear radii ~1-10 fm
        if 1.0 <= self.R <= 10.0:
            return (True, f"R = {self.R} fm is in reasonable range")
        else:
            return (False, f"R = {self.R} fm is outside typical nuclear radius range (1-10 fm)")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover and execute all test_* methods.
        Returns (all_pass: bool, logs: List[str]).
        """
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
                msg = f"{name} CRASHED: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)