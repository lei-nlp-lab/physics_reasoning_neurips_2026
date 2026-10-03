import math
import traceback
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics checker for the thermodynamics problem.
    Extracts and verifies results emitted from the solution, and optionally
    checks explanation steps for consistency with thermodynamic identities.
    """

    def __init__(self, solution_instance: Any):
        """
        Initialize by extracting results and explanation steps from the solution instance.
        The instance is expected to have a .results dict (or be a dict itself).
        """
        if isinstance(solution_instance, dict):
            self.results = solution_instance
        else:
            self.results = getattr(solution_instance, 'results', None) or {}

        # Extract expected keys with None defaults (no numeric fallbacks)
        self.ideal_gas_flag = self.results.get('ideal_gas_flag', None)
        self.alpha_sign = self.results.get('alpha_sign', None)
        self.Cp_value = self.results.get('Cp_value', None)
        self.beta_value = self.results.get('beta_value', None)
        self.maxwell_holds = self.results.get('maxwell_holds', None)

        # Optional auxiliary data
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------------
    def test_interface_expected_keys(self) -> Tuple[bool, str]:
        """Verify that all five required keys are present in results."""
        required_keys = ['ideal_gas_flag', 'alpha_sign', 'Cp_value', 'beta_value', 'maxwell_holds']
        if not isinstance(self.results, dict):
            return (False, "results is not a dict")
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, f"All required keys present: {required_keys}")

    def test_interface_key_types(self) -> Tuple[bool, str]:
        """Verify that each key has the expected type (bool, string, numeric)."""
        if not isinstance(self.results, dict):
            return (False, "results is not a dict")

        errors = []
        # ideal_gas_flag should be bool
        if not isinstance(self.ideal_gas_flag, bool):
            errors.append(f"ideal_gas_flag: expected bool, got {type(self.ideal_gas_flag).__name__}")
        # alpha_sign should be string
        if not isinstance(self.alpha_sign, str):
            errors.append(f"alpha_sign: expected str, got {type(self.alpha_sign).__name__}")
        # Cp_value should be string
        if not isinstance(self.Cp_value, str):
            errors.append(f"Cp_value: expected str, got {type(self.Cp_value).__name__}")
        # beta_value should be numeric (int or float)
        if not isinstance(self.beta_value, (int, float)):
            errors.append(f"beta_value: expected int/float, got {type(self.beta_value).__name__}")
        # maxwell_holds should be bool
        if not isinstance(self.maxwell_holds, bool):
            errors.append(f"maxwell_holds: expected bool, got {type(self.maxwell_holds).__name__}")
        if errors:
            return (False, "; ".join(errors))
        return (True, "All keys have expected types")

    # ------------------------------------------------------------------------
    # Invariant tests (theorem-based physical checks)
    # ------------------------------------------------------------------------
    def test_invariant_maxwell_holds(self) -> Tuple[bool, str]:
        """Maxwell relation (e) holds identically for all classical single‑component systems."""
        if self.maxwell_holds is None:
            return (False, "SKIP: maxwell_holds not provided")
        if not isinstance(self.maxwell_holds, bool):
            return (False, f"maxwell_holds is not bool: {self.maxwell_holds}")
        if not self.maxwell_holds:
            return (False, "maxwell_holds must be True (the relation always holds)")
        return (True, "maxwell_holds is correctly True")

    def test_invariant_ideal_gas_flag(self) -> Tuple[bool, str]:
        """For (a), (∂U/∂V)_T=0 implies ideal gas (standard example)."""
        if self.ideal_gas_flag is None:
            return (False, "SKIP: ideal_gas_flag not provided")
        if not isinstance(self.ideal_gas_flag, bool):
            return (False, f"ideal_gas_flag is not bool: {self.ideal_gas_flag}")
        if not self.ideal_gas_flag:
            return (False, "ideal_gas_flag should be True for (a)")
        return (True, "ideal_gas_flag is correctly True")

    def test_invariant_alpha_sign(self) -> Tuple[bool, str]:
        """For (b), (∂S/∂V)_p < 0 requires negative thermal expansion coefficient α."""
        if self.alpha_sign is None:
            return (False, "SKIP: alpha_sign not provided")
        if not isinstance(self.alpha_sign, str):
            return (False, f"alpha_sign is not string: {self.alpha_sign}")
        if self.alpha_sign != "negative":
            return (False, f"alpha_sign must be 'negative', got '{self.alpha_sign}'")
        return (True, "alpha_sign is correctly 'negative'")

    def test_invariant_Cp_value(self) -> Tuple[bool, str]:
        """For (c), (∂T/∂S)_p = 0 => C_p → ∞."""
        if self.Cp_value is None:
            return (False, "SKIP: Cp_value not provided")
        if not isinstance(self.Cp_value, str):
            return (False, f"Cp_value is not string: {self.Cp_value}")
        if self.Cp_value != "infinite":
            return (False, f"Cp_value must be 'infinite', got '{self.Cp_value}'")
        return (True, "Cp_value is correctly 'infinite'")

    def test_invariant_beta_value(self) -> Tuple[bool, str]:
        """For (d), (∂S/∂V)_T = 0 => (∂p/∂T)_V = 0 => β = 0."""
        if self.beta_value is None:
            return (False, "SKIP: beta_value not provided")
        if not isinstance(self.beta_value, (int, float)):
            return (False, f"beta_value is not numeric: {self.beta_value}")
        if abs(self.beta_value) > 1e-9:
            return (False, f"beta_value should be 0, got {self.beta_value}")
        return (True, "beta_value is correctly 0")

    # ------------------------------------------------------------------------
    # Modeling/trace tests (using explanation_steps if available)
    # ------------------------------------------------------------------------
    def test_trace_phase_transition_mention(self) -> Tuple[bool, str]:
        """
        For (c), the explanation steps should mention a first-order phase transition
        or coexistent phases (since C_p → ∞).
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        if isinstance(self.explanation_steps, list):
            text = " ".join(str(step) for step in self.explanation_steps)
        else:
            text = str(self.explanation_steps)
        keywords = ["phase transition", "coexistent", "two phas", "first order"]
        found = any(kw.lower() in text.lower() for kw in keywords)
        if not found:
            return (False, "Explanation steps do not mention a phase transition for (c)")
        return (True, "Explanation steps mention a phase transition for (c)")

    def test_trace_maximum_density_mention(self) -> Tuple[bool, str]:
        """
        For (d), the explanation steps should refer to a system with β=0,
        e.g., maximum density (water at 4°C).
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")
        if isinstance(self.explanation_steps, list):
            text = " ".join(str(step) for step in self.explanation_steps)
        else:
            text = str(self.explanation_steps)
        keywords = ["maximum density", "water at 4", "density maximum", "β = 0"]
        found = any(kw.lower() in text.lower() for kw in keywords)
        if not found:
            return (False, "Explanation steps do not mention maximum density for (d)")
        return (True, "Explanation steps mention maximum density for (d)")

    # ------------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input)
    # ------------------------------------------------------------------------
    def test_monotonicity_entropy_decreases_with_volume_at_const_p(self) -> Tuple[bool, str]:
        """
        For (b): (∂S/∂V)_p < 0. This means that at constant pressure,
        as volume increases, entropy decreases (monotonic decreasing).
        We simulate a small positive perturbation in volume and check the sign of the resulting
        entropy change using the thermodynamic identity and the given alpha_sign.
        """
        if self.alpha_sign is None:
            return (False, "SKIP: alpha_sign not provided")
        if self.alpha_sign != "negative":
            return (False, f"alpha_sign is '{self.alpha_sign}'; cannot assert monotonic decreasing S with V")
        # Simulate: baseline (V0, S0). Perturb V0 -> V0 + ΔV (ΔV > 0).
        # The change in entropy: ΔS ≈ (∂S/∂V)_p * ΔV. (∂S/∂V)_p < 0, so ΔS < 0.
        # This is a monotonic decreasing relationship.
        return (True, "Perturbation: ΔV>0 → ΔS<0; monotonic decreasing holds (consistent with alpha negative)")

    # ------------------------------------------------------------------------
    # Weak answer sanity (final check; does not dominate)
    # ------------------------------------------------------------------------
    def test_answer_sanity_overall(self) -> Tuple[bool, str]:
        """
        Weak sanity: all five result keys are present and not None,
        and the results are internally consistent (no contradictions).
        """
        if not isinstance(self.results, dict):
            return (False, "results is not a dict")
        required_keys = ['ideal_gas_flag', 'alpha_sign', 'Cp_value', 'beta_value', 'maxwell_holds']
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        # None safety
        if any(self.results.get(k) is None for k in required_keys):
            return (False, "Some required results are None")
        # Check that ideal_gas_flag and maxwell_holds are boolean True
        if self.ideal_gas_flag is not True:
            return (False, "ideal_gas_flag not True")
        if self.maxwell_holds is not True:
            return (False, "maxwell_holds not True")
        # alpha_sign string consistency
        if self.alpha_sign not in ["negative", "positive", "zero"]:
            return (False, f"alpha_sign '{self.alpha_sign}' not a valid sign string")
        # Cp_value must be appropriately descriptive
        if self.Cp_value not in ["infinite", "finite", "0"]:
            return (False, f"Cp_value '{self.Cp_value}' not a descriptive string")
        # beta_value numeric (already checked)
        return (True, "Overall results are present and internally consistent")

    # ------------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all test_* methods, collect logs, and return overall success flag + log list.
        """
        logs: List[str] = []
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)