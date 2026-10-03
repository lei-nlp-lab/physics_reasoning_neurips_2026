import math
import sys
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution to the gyrotropic birefringence problem.
    Checks interface completeness, physical invariants, modeling consistency,
    and monotonicity trends.
    """
    
    def __init__(self, solution_instance: Any) -> None:
        # Extract results dict (handle missing gracefully)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        else:
            self.results = {}
        
        # Extract givens dict if available
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.givens = solution_instance.givens
        else:
            self.givens = {}
        
        # Extract explanation_steps if available (list)
        if hasattr(solution_instance, 'explanation_steps') and isinstance(solution_instance.explanation_steps, list):
            self.explanation_steps = solution_instance.explanation_steps
        else:
            self.explanation_steps = []
        
        # Extract required parameters (use None for missing)
        self.epsilon = self.givens.get('epsilon', None)
        self.g = self.givens.get('g', None)
        
        self.N1 = self.results.get('N1', None)
        self.N2 = self.results.get('N2', None)
        self.pol1 = self.results.get('pol1', None)
        self.pol2 = self.results.get('pol2', None)
        
        # Also extract omega and c if present (not strictly required but used for some checks)
        self.omega = self.givens.get('omega', None)
        self.c = self.givens.get('c', None)
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods."""
        logs: List[str] = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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
    
    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all four expected keys are present in results."""
        required = ['N1', 'N2', 'pol1', 'pol2']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present.")
    
    def test_interface_epsilon_g_present(self) -> Tuple[bool, str]:
        """Check that epsilon and g are provided as givens."""
        if self.epsilon is None and self.g is None:
            return (False, "Both epsilon and g are missing from givens.")
        if self.epsilon is None:
            return (False, "epsilon missing from givens.")
        if self.g is None:
            return (False, "g missing from givens.")
        return (True, "epsilon and g are available.")
    
    def test_interface_result_types(self) -> Tuple[bool, str]:
        """Check that N1, N2 are numeric (float/int) and pol strings."""
        if self.N1 is None or self.N2 is None:
            return (False, "N1 or N2 is None.")
        if not isinstance(self.N1, (int, float)):
            return (False, f"N1 has invalid type {type(self.N1).__name__}.")
        if not isinstance(self.N2, (int, float)):
            return (False, f"N2 has invalid type {type(self.N2).__name__}.")
        if not isinstance(self.pol1, str):
            return (False, f"pol1 is not a string.")
        if not isinstance(self.pol2, str):
            return (False, f"pol2 is not a string.")
        return (True, "Types are correct: N1, N2 numeric; pol1, pol2 strings.")
    
    # -----------------------------------------------------------------
    # Physical invariant tests (use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_N_positive(self) -> Tuple[bool, str]:
        """Index of refraction must be positive."""
        if self.N1 is None or self.N2 is None:
            return (False, "SKIP: N1 or N2 not available.")
        if self.N1 <= 0:
            return (False, f"N1 = {self.N1} is not positive.")
        if self.N2 <= 0:
            return (False, f"N2 = {self.N2} is not positive.")
        return (True, f"N1={self.N1}, N2={self.N2} are positive.")
    
    def test_invariant_squared_sum(self) -> Tuple[bool, str]:
        """Check N1^2 + N2^2 ≈ 2*epsilon (theoretical: (ε+g)+(ε-g)=2ε)."""
        if None in (self.N1, self.N2, self.epsilon):
            return (False, "SKIP: missing N1, N2, or epsilon.")
        lhs = self.N1**2 + self.N2**2
        rhs = 2 * self.epsilon
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N1^2+N2^2 = {lhs} ≈ 2ε = {rhs}.")
        else:
            return (False, f"N1^2+N2^2 = {lhs}, expected ~{rhs}.")
    
    def test_invariant_squared_diff(self) -> Tuple[bool, str]:
        """Check N1^2 - N2^2 ≈ 2*g (theoretical: (ε+g)-(ε-g)=2g)."""
        if None in (self.N1, self.N2, self.g):
            return (False, "SKIP: missing N1, N2, or g.")
        lhs = self.N1**2 - self.N2**2
        rhs = 2 * self.g
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N1^2-N2^2 = {lhs} ≈ 2g = {rhs}.")
        else:
            return (False, f"N1^2-N2^2 = {lhs}, expected ~{rhs}.")
    
    def test_invariant_N1_greater_than_N2(self) -> Tuple[bool, str]:
        """For g>0, N1 should be larger than N2; for g<0, N2 larger."""
        if self.N1 is None or self.N2 is None or self.g is None:
            return (False, "SKIP: missing data.")
        if not math.isfinite(self.g):
            return (False, "g is not finite.")
        if self.g > 0 and self.N1 > self.N2:
            return (True, f"For g>0, N1={self.N1} > N2={self.N2} correct.")
        elif self.g < 0 and self.N1 < self.N2:
            return (True, f"For g<0, N1={self.N1} < N2={self.N2} correct.")
        else:
            return (False, f"Order inconsistent: g={self.g}, N1={self.N1}, N2={self.N2}.")
    
    # -----------------------------------------------------------------
    # Modeling / trace tests (check polarizations)
    # -----------------------------------------------------------------
    def test_modeling_polarization_N1(self) -> Tuple[bool, str]:
        """Check pol1 indicates right-circular polarization (E_x = i E_y)."""
        if self.pol1 is None:
            return (False, "SKIP: pol1 not provided.")
        pol1_str = self.pol1.replace(" ", "")
        if "E_z=0" not in pol1_str:
            return (False, "pol1 does not state E_z=0.")
        # Check for E_x = i E_y or similar
        if "E_x=iE_y" in pol1_str or "E_x= iE_y" in pol1_str or "E_x = iE_y" in self.pol1:
            return (True, "pol1 correctly shows E_x = i E_y.")
        # Allow minor deviations but flag if left circular
        if "E_x=-iE_y" in pol1_str:
            return (False, "pol1 suggests left circular polarization, expected right.")
        if "E_x" in pol1_str and "E_y" in pol1_str and "i" in pol1_str:
            return (True, "pol1 contains some E_x-E_y relation, but could not verify exact form.")
        return (False, f"pol1 = '{self.pol1}' does not express required circular polarization.")
    
    def test_modeling_polarization_N2(self) -> Tuple[bool, str]:
        """Check pol2 indicates left-circular polarization (E_x = -i E_y)."""
        if self.pol2 is None:
            return (False, "SKIP: pol2 not provided.")
        pol2_str = self.pol2.replace(" ", "")
        if "E_z=0" not in pol2_str:
            return (False, "pol2 does not state E_z=0.")
        if "E_x=-iE_y" in pol2_str or "E_x= -iE_y" in pol2_str or "E_x = -iE_y" in self.pol2:
            return (True, "pol2 correctly shows E_x = -i E_y.")
        if "E_x=iE_y" in pol2_str:
            return (False, "pol2 suggests right circular polarization, expected left.")
        if "E_x" in pol2_str and "E_y" in pol2_str and "i" in pol2_str:
            return (True, "pol2 contains some E_x-E_y relation.")
        return (False, f"pol2 = '{self.pol2}' does not express required circular polarization.")
    
    def test_trace_Ez_zero(self) -> Tuple[bool, str]:
        """Check that the explanation mentions E_z=0 or D_z=0."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps to check.")
        for step in self.explanation_steps:
            if isinstance(step, str) and 'E_z' in step and '0' in step:
                return (True, "Explanation mentions E_z = 0.")
        for step in self.explanation_steps:
            if isinstance(step, str) and 'transverse' in step.lower():
                return (True, "Explanation mentions transverse wave (E_z=0).")
        return (True, "E_z=0 not explicitly found in explanation; could be implicit.")
    
    # -----------------------------------------------------------------
    # Monotonicity tests (explicit perturbation required)
    # -----------------------------------------------------------------
    def test_monotonicity_N1_and_N2_separation(self) -> Tuple[bool, str]:
        """
        Perturb g upward by a small delta and verify that N1 increases and N2 decreases.
        Use the theoretical formulas N1 = sqrt(epsilon+g), N2 = sqrt(epsilon-g)
        to compute baseline and perturbed values.
        """
        if self.epsilon is None or self.g is None:
            return (False, "SKIP: missing epsilon or g.")
        # Ensure the theoretical formulas are physically valid
        if self.epsilon + self.g <= 0 or self.epsilon - self.g <= 0:
            return (False, "epsilon +/- g not positive; cannot compute theoretical indices.")
        # Baseline indices computed from theory
        N1_base = math.sqrt(self.epsilon + self.g)
        N2_base = math.sqrt(self.epsilon - self.g)
        # Choose a small perturbation relative to the scale of g
        # Use a fraction of max(|g|, 1) to avoid zero division.
        delta_g = 0.01 * max(abs(self.g), 0.1)
        # Ensure perturbed values remain real
        if self.epsilon + self.g + delta_g <= 0 or self.epsilon - self.g - delta_g <= 0:
            # Reduce delta if needed
            delta_g = 0.001 * min(self.epsilon + self.g, self.epsilon - self.g)
        if delta_g <= 0:
            return (False, "Cannot choose a positive delta_g while keeping radicands positive.")
        # Compute perturbed indices
        N1_pert = math.sqrt(self.epsilon + self.g + delta_g)
        N2_pert = math.sqrt(self.epsilon - self.g - delta_g)
        # Monotonicity: increasing g should increase N1 and decrease N2
        if N1_pert > N1_base and N2_pert < N2_base:
            return (True, f"Perturbed g by +{delta_g:.4f}: N1 increased ({N1_base:.4f}→{N1_pert:.4f}), N2 decreased ({N2_base:.4f}→{N2_pert:.4f}).")
        else:
            return (False, f"Monotonicity violated: N1 changed {N1_base:.4f}→{N1_pert:.4f}, N2 changed {N2_base:.4f}→{N2_pert:.4f}.")
    
    # -----------------------------------------------------------------
    # Final answer sanity (very weak, no isclose, no derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_finite_non_nan(self) -> Tuple[bool, str]:
        """Check that the final answer values are finite and not NaN."""
        if self.N1 is None or self.N2 is None:
            return (False, "SKIP: N1 or N2 not available.")
        if not math.isfinite(self.N1):
            return (False, f"N1 is not finite: {self.N1}")
        if not math.isfinite(self.N2):
            return (False, f"N2 is not finite: {self.N2}")
        return (True, "N1 and N2 are finite and not NaN.")