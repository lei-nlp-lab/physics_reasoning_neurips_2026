import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution for the atomic clock time dilation problem.
    Extracts the discrepancy delta_tau and optional givens,
    then runs small, focused tests on interface, invariants,
    modeling quality, monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Extract results dictionary safely ---
        results: Dict = getattr(solution_instance, 'results', None)
        if not isinstance(results, dict):
            results = {}
        self.results = results
        self.delta_tau = results.get('delta_tau', None)

        # --- Extract givens dictionary safely ---
        givens: Dict = getattr(solution_instance, 'givens', None)
        if not isinstance(givens, dict):
            givens = {}
        self.givens = givens

        # --- Extract individual givens with None defaults ---
        self.L: Optional[float] = givens.get('L', None)
        self.v: Optional[float] = givens.get('v', None)
        self.c: Optional[float] = givens.get('c', None)

        # Also try direct attributes (fallback)
        if self.L is None:
            self.L = getattr(solution_instance, 'L', None)
        if self.v is None:
            self.v = getattr(solution_instance, 'v', None)
        if self.c is None:
            self.c = getattr(solution_instance, 'c', None)

        # --- Explanation steps (optional) ---
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_delta_tau_exists(self) -> Tuple[bool, str]:
        """Check that delta_tau is present and not None."""
        if self.delta_tau is None:
            return (False, "SKIP: delta_tau not provided")
        return (True, "delta_tau exists")

    def test_interface_delta_tau_numerical(self) -> Tuple[bool, str]:
        """Check that delta_tau is a finite number."""
        if self.delta_tau is None:
            return (False, "SKIP: delta_tau not provided")
        if not isinstance(self.delta_tau, (int, float)):
            return (False, f"delta_tau is not numeric, got {type(self.delta_tau)}")
        if not math.isfinite(self.delta_tau):
            return (False, f"delta_tau is not finite: {self.delta_tau}")
        return (True, "delta_tau is a finite number")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_delta_tau_positive(self) -> Tuple[bool, str]:
        """The moving clock runs slow, so the discrepancy is positive."""
        if self.delta_tau is None:
            return (False, "SKIP: delta_tau not provided")
        if self.delta_tau <= 0:
            return (False, f"delta_tau must be positive, got {self.delta_tau}")
        return (True, "delta_tau > 0")

    def test_invariant_delta_tau_less_than_T(self) -> Tuple[bool, str]:
        """The discrepancy cannot exceed the total coordinate travel time T.
        (Requires L and v to compute T = L/v.)"""
        if self.delta_tau is None:
            return (False, "SKIP: delta_tau not provided")
        if self.L is None or self.v is None or self.v == 0:
            return (False, "SKIP: L or v missing or v=0; cannot compute T")
        T = self.L / self.v
        if self.delta_tau >= T:
            return (False, f"delta_tau ({self.delta_tau}) must be < T ({T}); "
                           "moving clock cannot run ahead of coordinate time")
        return (True, f"delta_tau ({self.delta_tau}) < T ({T})")

    # ------------------------------------------------------------------
    # Modeling / trace quality tests
    # ------------------------------------------------------------------
    def test_modeling_delta_tau_approx(self) -> Tuple[bool, str]:
        """
        If L, v, c are provided, check that the discrepancy matches
        the low‑speed approximation Δτ ≈ (1/2)(v/c)² (L/v).
        Uses fuzzy matching (math.isclose) because rounding noise is expected.
        """
        if self.delta_tau is None:
            return (False, "SKIP: delta_tau not provided")
        if self.L is None or self.v is None or self.c is None:
            return (False, "SKIP: L, v, or c missing")
        if self.v == 0 or self.c == 0:
            return (False, "SKIP: v or c is zero")
        beta = self.v / self.c
        T = self.L / self.v
        approx = 0.5 * beta * beta * T
        if math.isclose(self.delta_tau, approx, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_tau ({self.delta_tau}) matches approx ({approx}) within 1%")
        else:
            return (False, f"delta_tau ({self.delta_tau}) deviates from approx ({approx}) by >1%")

    def test_modeling_exact_approx_inequality(self) -> Tuple[bool, str]:
        """
        For v << c, the exact time dilation discrepancy
        Δτ_exact = T*(1 - sqrt(1-β²)) is slightly larger than the
        first‑order approximation Δτ_approx = (1/2)β² T.
        Check that exact >= approx when computed from the same givens.
        """
        if self.L is None or self.v is None or self.c is None:
            return (False, "SKIP: L, v, or c missing")
        if self.v == 0 or self.c == 0:
            return (False, "SKIP: v or c is zero")
        beta = self.v / self.c
        T = self.L / self.v
        approx = 0.5 * beta * beta * T
        exact = T * (1.0 - math.sqrt(1.0 - beta*beta))
        if exact >= approx - 1e-15:  # allow tiny rounding
            return (True, f"exact ({exact:.2e}) >= approx ({approx:.2e}) as expected")
        else:
            return (False, f"exact ({exact:.2e}) < approx ({approx:.2e}) violates physics")

    # ------------------------------------------------------------------
    # Monotonicity tests (impact of perturbing a physical parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_v_increases_delta_tau(self) -> Tuple[bool, str]:
        """
        For v << c, the approximate discrepancy is proportional to v.
        Increase v by 10% and verify that the approximate discrepancy increases.
        Requires L, v, c.
        """
        if self.L is None or self.v is None or self.c is None:
            return (False, "SKIP: L, v, or c missing")
        if self.v == 0 or self.c == 0:
            return (False, "SKIP: v or c is zero")
        beta = self.v / self.c
        T = self.L / self.v
        baseline = 0.5 * beta * beta * T   # = (v L)/(2 c²)

        # Perturb v upward by 10%
        v_high = self.v * 1.10
        beta_high = v_high / self.c
        T_high = self.L / v_high
        perturbed = 0.5 * beta_high * beta_high * T_high   # = (v_high L)/(2 c²)

        if perturbed > baseline:
            return (True, f"v increased by 10% → approx discrepancy grows from "
                          f"{baseline:.2e} to {perturbed:.2e}")
        else:
            return (False, f"v increased but approx discrepancy did not increase: "
                           f"baseline {baseline:.2e}, perturbed {perturbed:.2e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (DO NOT compute or re‑derive the target answer)
    # ------------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitude(self) -> Tuple[bool, str]:
        """
        The discrepancy should be on the order of 10⁻⁷ to 10⁻⁶ seconds
        for typical jet speeds. Check that it lies in a broad range.
        """
        if self.delta_tau is None:
            return (False, "SKIP: delta_tau not provided")
        if not math.isfinite(self.delta_tau):
            return (False, f"delta_tau not finite: {self.delta_tau}")
        # Expect between 10⁻⁹ s and 10⁻⁵ s (10 ns to 10 μs)
        if 1e-9 < self.delta_tau < 1e-5:
            return (True, f"delta_tau ({self.delta_tau:.2e} s) is in a plausible range")
        else:
            return (False, f"delta_tau ({self.delta_tau:.2e} s) seems far "
                           f"from expected ~10⁻⁷ s (between 1e-9 and 1e-5 s)")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them
        inside try/except, collect PASS/FAIL messages and return overall
        success flag and log.
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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)