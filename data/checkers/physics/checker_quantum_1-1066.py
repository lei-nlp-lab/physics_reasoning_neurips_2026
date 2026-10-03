import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics checker for the problem:
    Real operator A satisfying A^2 - 3A + 2 = 0.
    Checks eigenvalues, eigenstates, and observability.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract required data from the solution_instance.
        Use None defaults for missing data – never numeric defaults.
        """
        # Try to get a results dictionary
        results: Dict[str, Any] = getattr(solution_instance, 'results', None) or {}
        # Extract individual keys safely
        self.eigenvalues: Optional[List[float]] = results.get('eigenvalues', None)
        self.eigenstates: Optional[Any] = results.get('eigenstates', None)
        self.observable: Optional[Any] = results.get('observable', None)

        # Also try direct attributes as fallback
        if self.eigenvalues is None:
            self.eigenvalues = getattr(solution_instance, 'eigenvalues', None)
        if self.eigenstates is None:
            self.eigenstates = getattr(solution_instance, 'eigenstates', None)
        if self.observable is None:
            self.observable = getattr(solution_instance, 'observable', None)

    # ------------------------------------------------------------
    # Interface tests – ensure required keys exist with correct types
    # ------------------------------------------------------------
    def test_interface_eigenvalues_present(self) -> Tuple[bool, str]:
        """Check that 'eigenvalues' is provided and is a list/tuple of numbers."""
        if self.eigenvalues is None:
            return (False, "SKIP? eigenvalues missing, but test expects list.")
        if not isinstance(self.eigenvalues, (list, tuple)):
            return (False, f"eigenvalues must be list or tuple, got {type(self.eigenvalues)}")
        if len(self.eigenvalues) < 2:
            return (False, f"expected at least 2 eigenvalues, got {len(self.eigenvalues)}")
        # Check each element is numeric
        for i, val in enumerate(self.eigenvalues):
            if not isinstance(val, (int, float)):
                return (False, f"eigenvalues[{i}] is not numeric, got {type(val)}")
        return (True, "eigenvalues present and valid")

    def test_interface_eigenstates_present(self) -> Tuple[bool, str]:
        """Check that 'eigenstates' is provided and is a string (or suitable representation)."""
        if self.eigenstates is None:
            return (False, "eigenstates missing")
        if not isinstance(self.eigenstates, str):
            return (False, f"eigenstates should be a string description, got {type(self.eigenstates)}")
        if len(self.eigenstates.strip()) == 0:
            return (False, "eigenstates string is empty")
        return (True, "eigenstates present and non-empty string")

    def test_interface_observable_present(self) -> Tuple[bool, str]:
        """Check that 'observable' is provided and is truthy."""
        if self.observable is None:
            return (False, "observable missing")
        if not self.observable:
            # observable could be bool, string "True", etc.
            # Accept truthy values
            return (False, f"observable is not truthy, got {self.observable}")
        return (True, "observable present and truthy")

    # ------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_eigenvalue_polynomial(self) -> Tuple[bool, str]:
        """Each eigenvalue must satisfy λ² - 3λ + 2 ≈ 0."""
        if self.eigenvalues is None:
            return (False, "SKIP: no eigenvalues to check invariant")
        for i, val in enumerate(self.eigenvalues):
            try:
                lhs = val*val - 3*val + 2
                if not math.isclose(lhs, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"eigenvalue {i} = {val} does not satisfy λ²-3λ+2=0 (lhs={lhs})")
            except Exception:
                return (False, f"exception during evaluation of eigenvalue {i}")
        return (True, "all eigenvalues satisfy λ²-3λ+2≈0")

    def test_invariant_eigenvalues_real(self) -> Tuple[bool, str]:
        """Eigenvalues must be real numbers (already numeric but check complex)."""
        if self.eigenvalues is None:
            return (False, "SKIP: no eigenvalues to check realness")
        for i, val in enumerate(self.eigenvalues):
            if isinstance(val, complex):
                # If the problem expects real eigenvalues, complex with zero imag is acceptable
                if abs(val.imag) > 1e-9:
                    return (False, f"eigenvalue {i} has nonzero imaginary part: {val}")
            elif not isinstance(val, (int, float)):
                return (False, f"eigenvalue {i} is not real numeric, got {type(val)}")
        return (True, "all eigenvalues are real (or complex with negligible imag)")

    # ------------------------------------------------------------
    # Modeling tests – check quality of eigenstate description, reasoning
    # ------------------------------------------------------------
    def test_modeling_eigenstate_description(self) -> Tuple[bool, str]:
        """Eigenstates description should mention both eigenvalues or completeness."""
        if self.eigenstates is None:
            return (False, "SKIP: eigenstates not available")
        desc = str(self.eigenstates).lower()
        # Look for keywords
        has_one = '1' in desc or 'one' in desc
        has_two = '2' in desc or 'two' in desc
        if not (has_one and has_two):
            return (False, "eigenstates description does not clearly reference eigenvalues 1 and 2")
        return (True, "eigenstates description references both eigenvalues")

    def test_modeling_observable_reasoning(self) -> Tuple[bool, str]:
        """
        Observability claim is backed by Hermitian property and completeness.
        We check that 'observable' is true and that eigenstates is non-empty (indicating completeness).
        """
        if self.observable is None or self.eigenstates is None:
            return (False, "SKIP: missing observable or eigenstates")
        # If observable is truthy and eigenstates exist, that's a minimal modeling check.
        if not self.observable:
            return (False, "observable is False or falsy")
        if isinstance(self.eigenstates, str) and len(self.eigenstates.strip()) == 0:
            return (False, "eigenstates description empty, missing completeness argument")
        return (True, "observable is true and eigenstates provided")

    # ------------------------------------------------------------
    # Monotonicity test – must perturb an input and compare perturbed output
    # ------------------------------------------------------------
    def test_monotonicity_eigenvalue_linear_term(self) -> Tuple[bool, str]:
        """
        Perturb the linear coefficient of the characteristic polynomial and check
        that the eigenvalues shift monotonically in the correct direction.
        """
        if self.eigenvalues is None:
            return (False, "SKIP: no eigenvalues to benchmark")
        if len(self.eigenvalues) != 2:
            return (False, f"SKIP: expected exactly 2 eigenvalues, got {len(self.eigenvalues)}")
        # Baseline coefficients from problem: λ² - 3λ + 2 = 0
        a, b, c = 1.0, -3.0, 2.0
        # Sort baseline eigenvalues for comparison
        base = sorted(self.eigenvalues)
        # Use small perturbation: 5% of |b|
        delta = 0.05 * abs(b)   # 0.15
        for sign, label in [(1, '+5% (more negative)'), (-1, '-5% (less negative)')]:
            b_new = b + sign * delta
            disc = b_new*b_new - 4*a*c
            if disc < 0:
                return (False, f"Perturbation {label} results in complex eigenvalues (disc={disc})")
            sqrt_disc = math.sqrt(disc)
            new_roots = sorted([ (-b_new - sqrt_disc) / (2*a),
                                 (-b_new + sqrt_disc) / (2*a) ])
            # Check monotonic direction:
            # When b becomes more negative (sign=1), sum increases → smaller root decreases, larger root increases.
            # When b becomes less negative (sign=-1), sum decreases → smaller root increases, larger root decreases.
            if sign == 1:
                if not (new_roots[0] < base[0] - 1e-9 and new_roots[1] > base[1] + 1e-9):
                    return (False, f"Monotonicity violation for {label}: expected new_small<base_small and new_large>base_large, "
                                   f"got base={base}, new={new_roots}")
            else:  # sign == -1
                if not (new_roots[0] > base[0] + 1e-9 and new_roots[1] < base[1] - 1e-9):
                    return (False, f"Monotonicity violation for {label}: expected new_small>base_small and new_large<base_large, "
                                   f"got base={base}, new={new_roots}")
        return (True, "Eigenvalue monotonicity with respect to linear coefficient holds")

    # ------------------------------------------------------------
    # Answer sanity – weak final answer check (no derivation, no isclose)
    # ------------------------------------------------------------
    def test_answer_sanity_observable_positive(self) -> Tuple[bool, str]:
        """Check that observable indicates True/yes. Only a weak sanity."""
        if self.observable is None:
            return (False, "SKIP: observable missing")
        if self.observable:  # truthy
            return (True, "observable is truthy")
        else:
            return (False, "observable is falsy, expected truthy")

    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods.
        Collect results and logs.
        """
        logs: List[str] = []
        all_pass = True
        # Iterate over all attributes
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED with exception: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)