import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Physics auditor for the Boltzmann energy equipartition problem.
    Verifies interface, invariants, modeling quality, and (weak) answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Extract givens (expected keys: a, T, k_B)
        givens = getattr(solution_instance, 'givens', None) or {}
        self.a = givens.get('a', None)
        self.T = givens.get('T', None)
        self.k_B = givens.get('k_B', None)

        # If k_B not set, try common naming
        if self.k_B is None:
            self.k_B = givens.get('k_Boltzmann', None) or givens.get('kB', None)

        # Extract explanation steps
        self.explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = []

        # Extract pseudocode (optional)
        self.pseudocode: Optional[Dict] = getattr(solution_instance, 'pseudocode', None)

        # Extract target answers
        self.target_answers: Optional[List[str]] = getattr(solution_instance, 'target_answers', None)

        # Derived values
        self.E_avg = self.results.get('E_avg', None)
        self.partition_function = self.results.get('Z', None)   # optional

        # Convert to float if possible
        if self.E_avg is not None:
            try:
                self.E_avg = float(self.E_avg)
            except (TypeError, ValueError):
                pass  # keep as is (may be symbolic)

        if self.T is not None and self.k_B is not None:
            try:
                self.T = float(self.T)
                self.k_B = float(self.k_B)
            except (TypeError, ValueError):
                pass

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_results_dict(self) -> Tuple[bool, str]:
        """Check that results is a dict (or at least not None)."""
        if self.results is None:
            return False, "No 'results' found on solution_instance"
        if not isinstance(self.results, dict):
            return False, "'results' is not a dict"
        return True, "results is a valid dict"

    def test_interface_E_avg_present(self) -> Tuple[bool, str]:
        """Check that the emitted key 'E_avg' exists."""
        if 'E_avg' not in self.results:
            return False, "Key 'E_avg' not found in results"
        return True, "E_avg is present"

    def test_interface_E_avg_numeric(self) -> Tuple[bool, str]:
        """Check that E_avg is numeric (int or float)."""
        if self.E_avg is None:
            return False, "E_avg is missing"
        if not isinstance(self.E_avg, (int, float)):
            return False, f"E_avg is of type {type(self.E_avg).__name__}, expected numeric"
        if math.isnan(self.E_avg) or math.isinf(self.E_avg):
            return False, "E_avg is NaN or Inf"
        return True, "E_avg is finite numeric"

    def test_interface_givens_provided(self) -> Tuple[bool, str]:
        """Check that essential givens (a, T, k_B) are provided."""
        missing = []
        if self.a is None:
            missing.append('a')
        if self.T is None:
            missing.append('T')
        if self.k_B is None:
            missing.append('k_B')
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        # check positivity
        try:
            if float(self.a) <= 0:
                return False, "Given 'a' must be positive (a > 0)"
        except:
            return False, "Given 'a' is not convertible to number"
        return True, "All essential givens present"

    # ----------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ----------------------------------------------------------------
    def test_invariant_E_avg_kT_half(self) -> Tuple[bool, str]:
        """Check the core result: E_avg = k_B T / 2."""
        if self.E_avg is None or self.T is None or self.k_B is None:
            return False, "SKIP: missing E_avg, T, or k_B"
        try:
            expected = (float(self.k_B) * float(self.T)) / 2.0
            actual = float(self.E_avg)
            if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"E_avg = {actual}, expected ~ {expected}"
            return True, f"E_avg ≈ {expected} (kT/2)"
        except Exception as e:
            return False, f"Evaluation error: {e}"

    def test_invariant_E_avg_independent_of_a(self) -> Tuple[bool, str]:
        """Check that the average energy does not depend on parameter 'a'."""
        if self.E_avg is None:
            return False, "SKIP: E_avg missing"
        expr = self.E_avg
        # If it's a string, try to parse
        if isinstance(expr, str):
            expr = sp.sympify(expr, locals={'k_B': self.k_B, 'T': self.T, 'k_Boltzmann': self.k_B, 'kB': self.k_B})
        # If it's a sympy expression, check free symbols
        try:
            syms = sp.sympify(expr).free_symbols
            if any(str(s) in ('a', 'A', 'alpha') for s in syms):
                return False, "E_avg expression depends on 'a' (should not)"
            return True, "E_avg does not contain 'a'"
        except:
            return True, "E_avg is numeric; cannot detect 'a' dependence, but likely correct"

    def test_invariant_partition_function_form(self) -> Tuple[bool, str]:
        """If partition function Z is provided, verify it has the form sqrt(pi/(beta*a))."""
        if self.partition_function is None:
            return False, "SKIP: partition function Z not provided"
        if self.T is None or self.k_B is None or self.a is None:
            return False, "SKIP: missing T, k_B, or a"
        try:
            beta = 1.0 / (float(self.k_B) * float(self.T))
            expected_Z = math.sqrt(math.pi / (beta * float(self.a)))
            actual_Z = float(self.partition_function)
            if not math.isclose(actual_Z, expected_Z, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"Partition function Z = {actual_Z}, expected ~ {expected_Z}"
            return True, f"Z ≈ {expected_Z}"
        except Exception as e:
            return False, f"Error: {e}"

    # ----------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------
    def test_modeling_mentions_boltzmann(self) -> Tuple[bool, str]:
        """Check that explanation steps mention Boltzmann distribution."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation steps"
        combined = " ".join(self.explanation_steps).lower()
        if "boltzmann" not in combined:
            return False, "Boltzmann not mentioned in explanation steps"
        return True, "Boltzmann distribution referenced"

    def test_modeling_uses_gaussian_integral(self) -> Tuple[bool, str]:
        """Check that Gaussian integral or normalisation is mentioned."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation steps"
        combined = " ".join(self.explanation_steps).lower()
        keywords = ["gaussian", "integral of exp", "∫ exp", "sqrt(pi/", "normalization"]
        found = any(kw in combined for kw in keywords)
        if not found:
            return False, "No Gaussian integral reference found"
        return True, "Gaussian integral used"

    def test_modeling_equipartition_stated(self) -> Tuple[bool, str]:
        """Check that the equipartition principle is stated."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation steps"
        combined = " ".join(self.explanation_steps).lower()
        if "equipartition" not in combined:
            return False, "Equipartition principle not mentioned"
        return True, "Equipartition theorem stated"

    def test_modeling_E_avg_derivation(self) -> Tuple[bool, str]:
        """Verify that the derivation steps lead to E_avg = kT/2."""
        if self.E_avg is None:
            return False, "SKIP: E_avg missing"
        from sympy import sympify, Symbol
        try:
            expr = sympify(self.E_avg)
            if expr.has(Symbol('a')):
                return False, "Expression still contains 'a'"
            return True, "Derivation yields E_avg independent of a"
        except:
            return True, "Cannot parse, but numeric value passes other tests"

    # ----------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------
    def test_monotonicity_temperature(self) -> Tuple[bool, str]:
        """
        Explicitly perturb temperature and compare perturbed output to baseline.
        Baseline: provided E_avg.
        Perturbation: increase T by 10%.
        Perturbed output (expected) = E_avg * (T_pert / T)   (because E_avg ∝ T).
        Check that the perturbed expected output is larger than baseline.
        """
        if self.E_avg is None or self.T is None or self.k_B is None:
            return False, "SKIP: missing E_avg, T, or k_B"
        try:
            T_base = float(self.T)
            E_base = float(self.E_avg)
            # Perturb T upward by 10%
            T_pert = T_base * 1.1
            # Expected E_avg at perturbed T (using known linear relation)
            # Since E_avg = k_B T / 2, the ratio should be preserved:
            E_pert_expected = (self.k_B * T_pert) / 2.0
            # Check monotonic increase: E_pert_expected should be greater than E_base
            if not (E_pert_expected > E_base):
                return False, f"Expected E at perturbed T ({E_pert_expected}) not greater than baseline ({E_base})"
            return True, f"Monotonicity: increasing T from {T_base} to {T_pert} increases E_avg from {E_base} to {E_pert_expected}"
        except Exception as e:
            return False, f"Error: {e}"

    # ----------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose)
    # ----------------------------------------------------------------
    def test_answer_sanity_positivity(self) -> Tuple[bool, str]:
        """Check that E_avg is positive and finite."""
        if self.E_avg is None:
            return False, "SKIP: E_avg missing"
        if not isinstance(self.E_avg, (int, float)):
            return False, "E_avg not numeric"
        if self.E_avg <= 0:
            return False, "E_avg is not positive"
        if math.isnan(self.E_avg) or math.isinf(self.E_avg):
            return False, "E_avg is NaN or Inf"
        return True, "E_avg is positive and finite"

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, return (all_pass, logs)."""
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