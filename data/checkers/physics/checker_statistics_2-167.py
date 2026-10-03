import math
import sympy as sp
from typing import Any, Callable, Dict, List, Tuple

class PhysicsChecker:
    """
    Auditor for the diffusion coefficient problem from kinetic theory.
    Checks:
      - Presence and type of required result keys.
      - Structural correctness of the formula.
      - Positivity and order-of-magnitude of the numerical value.
      - Consistency of the numerical value with the given physical constants (if provided).
      - Internal consistency among intermediate quantities (if provided).
      - Monotonicity of D with respect to temperature (if givens enable it).
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely (required keys)
        results = getattr(solution_instance, 'results', None) or {}
        self.D_formula = results.get('D_formula', None)
        self.D = results.get('D', None)

        # Extract optional intermediate results (for trace consistency)
        self.v_th = results.get('v_th', None)
        self.lambda_ = results.get('lambda', None)   # avoid shadowing built-in lambda

        # Extract givens safely (optional, used for modeling checks)
        givens = getattr(solution_instance, 'givens', None) or {}
        self.k = givens.get('k', None)          # Boltzmann constant (J/K)
        self.P = givens.get('P', None)          # Pressure (Pa)
        self.T = givens.get('T', None)          # Temperature (K)
        self.m = givens.get('m', None)          # Molecular mass (kg)
        self.d = givens.get('d', None)          # Molecular diameter (m)

        self._has_givens = all(v is not None for v in [self.k, self.P, self.T, self.m, self.d])

        # Explanation steps (for reference, not hardcoded)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_D_formula_exists(self) -> Tuple[bool, str]:
        """Check that D_formula is a non‑empty string."""
        if self.D_formula is None:
            return (False, "Missing D_formula in results")
        if not isinstance(self.D_formula, str) or self.D_formula.strip() == "":
            return (False, "D_formula must be a non‑empty string")
        return (True, "D_formula present and is a non‑empty string")

    def test_interface_D_exists(self) -> Tuple[bool, str]:
        """Check that D is a numeric value."""
        if self.D is None:
            return (False, "Missing D in results")
        if not isinstance(self.D, (int, float)):
            return (False, "D must be numeric (int or float)")
        return (True, "D present and numeric")

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based)
    # ------------------------------------------------------------------
    def test_invariant_D_formula_factor(self) -> Tuple[bool, str]:
        """
        The formula should contain the factor 1/3 and both speed and mean free path.
        Since numerical factors may be omitted, we look for a multiplicative form.
        """
        if self.D_formula is None:
            return (False, "SKIP: D_formula not available")
        formula_str = self.D_formula.lower()
        # Check for '1/3' or '0.333' or '1/3' ignored whitespace
        has_one_third = ('1/3' in formula_str or '0.333' in formula_str)
        # Check for a speed variable (v) and a mean‑free‑path variable (lambda or λ)
        has_speed = ('v' in formula_str)
        has_lambda = ('λ' in formula_str or 'lambda' in formula_str)
        if not has_one_third:
            return (False, "D_formula does not contain factor 1/3")
        if not has_speed:
            return (False, "D_formula lacks a speed variable")
        if not has_lambda:
            return (False, "D_formula lacks a mean‑free‑path variable")
        return (True, "D_formula structurally correct (1/3 * v * λ)")

    def test_invariant_D_positive(self) -> Tuple[bool, str]:
        """Diffusion coefficient must be positive."""
        if self.D is None:
            return (False, "SKIP: D not available")
        if self.D <= 0:
            return (False, f"D = {self.D} is not positive")
        return (True, f"D = {self.D} > 0")

    def test_invariant_D_order_of_magnitude(self) -> Tuple[bool, str]:
        """
        For air at STP, D should be on the order of 10^-5 m^2/s.
        Use a generous bound (1e-6 to 1e-3) that covers typical textbook values.
        """
        if self.D is None:
            return (False, "SKIP: D not available")
        if not (1e-6 <= self.D <= 1e-3):
            return (False, f"D = {self.D} is outside expected range [1e-6, 1e-3] m^2/s")
        return (True, f"D = {self.D} lies within 1e-6 to 1e-3 m^2/s")

    # ------------------------------------------------------------------
    # Modeling / trace quality checks (at least two)
    # ------------------------------------------------------------------
    def test_modeling_compute_D_from_givens(self) -> Tuple[bool, str]:
        """
        If all givens are provided, compute D from the kinetic‑theory formula
        and compare with the reported value using fuzzy matching.
        Because we ignore small numerical factors, a tolerance of ±50% is allowed.
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens (k, P, T, m, d) for recomputation")
        if self.D is None:
            return (False, "SKIP: D not available for comparison")

        try:
            # Compute n = P/(kT)
            k = self.k
            P = self.P
            T = self.T
            m = self.m
            d = self.d
            n = P / (k * T)
            # Thermal speed sqrt(kT/m) – simplest kinetic theory
            v_th = math.sqrt(k * T / m)
            # Collision cross‑section π d^2
            sigma = math.pi * d * d
            # Mean free path λ = 1/(n σ)  (ignoring √2)
            lam = 1.0 / (n * sigma)
            # D = (1/3) v_th λ
            D_computed = (1.0/3.0) * v_th * lam
            # Compare with reported D
            if not math.isclose(D_computed, self.D, rel_tol=0.5, abs_tol=1e-6):
                return (False, f"Computed D = {D_computed:.3e} does not match reported D = {self.D:.3e} "
                               f"(allowed ±50%)")
            return (True, f"Computed D = {D_computed:.3e} agrees with reported D = {self.D:.3e}")
        except Exception as e:
            return (False, f"Computation failed: {e}")

    def test_modeling_trace_consistency(self) -> Tuple[bool, str]:
        """
        If the solution provides intermediate quantities v_th and lambda,
        check that D ≈ (1/3) * v_th * lambda.
        This tests internal consistency of the reasoning trace.
        """
        if self.D is None:
            return (False, "SKIP: D not available")
        if self.v_th is None or self.lambda_ is None:
            return (False, "SKIP: v_th and lambda not available; cannot perform trace consistency check")

        try:
            expected_D = (1.0/3.0) * self.v_th * self.lambda_
            if not math.isclose(self.D, expected_D, rel_tol=0.01, abs_tol=1e-7):
                return (False, f"D = {self.D} does not match (1/3)*v_th*lambda = {expected_D:.3e}")
            return (True, f"Consistent: D = {self.D}, (1/3)*v_th*lambda = {expected_D:.3e}")
        except Exception as e:
            return (False, f"Trace consistency check crashed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------
    def test_monotonicity_D_increases_with_T(self) -> Tuple[bool, str]:
        """
        Check that increasing the temperature leads to a larger diffusion coefficient.
        Use the provided T and recompute D for a small perturbation (e.g., +5%).
        If givens are not present, skip.
        """
        if not self._has_givens:
            return (False, "SKIP: insufficient givens to perturb T")
        if self.D is None:
            return (False, "SKIP: D not available")

        try:
            k = self.k
            P = self.P
            T_base = self.T
            m = self.m
            d = self.d

            # Base D (using same formula as above)
            def compute_D(T_val: float) -> float:
                n = P / (k * T_val)
                v = math.sqrt(k * T_val / m)
                sigma = math.pi * d * d
                lam = 1.0 / (n * sigma)
                return (1.0/3.0) * v * lam

            D_base = compute_D(T_base)
            # Perturb T by +5%
            T_pert = T_base * 1.05
            D_pert = compute_D(T_pert)
            if D_pert <= D_base:
                return (False, f"D did not increase when T increased by 5%: base D={D_base:.3e}, perturbed D={D_pert:.3e}")
            return (True, f"D increases with T: base {D_base:.3e} → perturbed {D_pert:.3e}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Answer sanity (weak – only sign and finiteness)
    # ------------------------------------------------------------------
    def test_answer_sanity_D_finite(self) -> Tuple[bool, str]:
        """D must be finite (not NaN or infinity)."""
        if self.D is None:
            return (False, "SKIP: D not available")
        if math.isnan(self.D) or math.isinf(self.D):
            return (False, f"D = {self.D} is NaN or infinite")
        return (True, f"D = {self.D} is finite")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)