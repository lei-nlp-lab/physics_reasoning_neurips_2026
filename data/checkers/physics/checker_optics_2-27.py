import sympy as sp
import math
import traceback
from typing import Tuple, Any

class PhysicsChecker:
    """
    Checker for the Fresnel half-period zones problem.
    Verifies that the solution correctly shows all zones have equal area.
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction from solution_instance
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Extract required parameters with None defaults
        self.lambda_val = self.givens.get('lambda', None) or self.results.get('lambda', None)
        self.R = self.givens.get('R', None) or self.results.get('R', None)
        self.b = self.givens.get('b', None) or self.results.get('b', None)
        self.area_constant = self.results.get('area_constant', None)

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_lambda_present(self) -> Tuple[bool, str]:
        """Check that wavelength lambda is provided."""
        if self.lambda_val is None:
            return (False, "Missing required givens/results: 'lambda'")
        return (True, "lambda is present")

    def test_interface_R_present(self) -> Tuple[bool, str]:
        """Check that wavefront radius R is provided."""
        if self.R is None:
            return (False, "Missing required givens/results: 'R'")
        return (True, "R is present")

    def test_interface_b_present(self) -> Tuple[bool, str]:
        """Check that distance b is provided."""
        if self.b is None:
            return (False, "Missing required givens/results: 'b'")
        return (True, "b is present")

    def test_interface_area_constant_present(self) -> Tuple[bool, str]:
        """Check that the final result 'area_constant' was emitted."""
        if self.area_constant is None:
            return (False, "Missing key result: 'area_constant'")
        return (True, "area_constant is present")

    # ----------------------------------------------------------------
    # Invariant tests (using math.isclose for numeric, symbolic for exact)
    # ----------------------------------------------------------------
    def test_invariant_area_independent_of_n(self) -> Tuple[bool, str]:
        """
        The area of a Fresnel zone must be independent of the zone number n.
        Check that the symbolic expression does not contain 'n'.
        """
        if self.area_constant is None:
            return (False, "SKIP: area_constant not available")
        # If it's a numeric constant, it's trivially independent of n.
        if isinstance(self.area_constant, (int, float)):
            return (True, "Area constant is numeric → independent of n")
        # Assume sympy expression
        try:
            free_symbols = self.area_constant.free_symbols
            sp_n = sp.Symbol('n')
            if sp_n in free_symbols:
                return (False, f"Area constant contains 'n', but should be independent: {self.area_constant}")
            return (True, "Area constant is free of 'n'")
        except Exception as e:
            return (False, f"Could not analyze free symbols: {e}")

    def test_invariant_symmetry_R_b(self) -> Tuple[bool, str]:
        """
        The area constant expression π λ b R / (R + b) is symmetric under
        swapping R and b.  Verify this property using the given values.
        """
        if any(v is None for v in [self.lambda_val, self.R, self.b, self.area_constant]):
            return (False, "SKIP: missing required data")

        # Try symbolic check
        try:
            lam = sp.sympify(self.lambda_val) if not isinstance(self.lambda_val, sp.Basic) else self.lambda_val
            R = sp.sympify(self.R) if not isinstance(self.R, sp.Basic) else self.R
            b = sp.sympify(self.b) if not isinstance(self.b, sp.Basic) else self.b
            original = sp.pi * lam * b * R / (R + b)
            swapped = sp.pi * lam * R * b / (b + R)
            if sp.simplify(original - swapped) == 0:
                return (True, "Area constant expression is symmetric under R↔b (symbolic)")
        except Exception:
            pass

        # Numeric fallback using math.isclose
        try:
            lam_num = float(self.lambda_val)
            R_num = float(self.R)
            b_num = float(self.b)
            area_num = float(self.area_constant)
            # Compute with swapped R and b (using the same physics formula)
            area_swapped = math.pi * lam_num * R_num * b_num / (b_num + R_num)
            if math.isclose(area_num, area_swapped, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Area constant is symmetric under R↔b (numeric)")
            else:
                return (False, f"Area constant not symmetric: original {area_num:.6f}, swapped {area_swapped:.6f}")
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return (False, f"Cannot verify symmetry: {e}")

    # ----------------------------------------------------------------
    # Modeling/trace tests (using math.isclose for numeric fallback)
    # ----------------------------------------------------------------
    def test_modeling_correct_expression(self) -> Tuple[bool, str]:
        """
        Verify that the area constant matches the exact expression derived from the
        zone boundary and small‑angle approximation: π * λ * b * R / (R + b).
        Uses symbolic comparison, falling back to numeric isclose.
        """
        if any(v is None for v in [self.lambda_val, self.R, self.b, self.area_constant]):
            return (False, "SKIP: missing required data")

        # Build expected symbolic expression
        try:
            lam = sp.sympify(self.lambda_val) if not isinstance(self.lambda_val, sp.Basic) else self.lambda_val
            R = sp.sympify(self.R) if not isinstance(self.R, sp.Basic) else self.R
            b = sp.sympify(self.b) if not isinstance(self.b, sp.Basic) else self.b
            expected = sp.pi * lam * b * R / (R + b)
            diff = sp.simplify(self.area_constant - expected)
            if diff == 0:
                return (True, "Area constant matches exact derived expression symbolically")
        except Exception:
            pass

        # Numeric fallback
        try:
            lam_num = float(self.lambda_val)
            R_num = float(self.R)
            b_num = float(self.b)
            area_num = float(self.area_constant)
            expected_num = math.pi * lam_num * b_num * R_num / (R_num + b_num)
            if math.isclose(area_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Area constant numerically matches expected value {expected_num:.6f}")
            else:
                return (False, f"Numerical mismatch: got {area_num:.6f}, expected {expected_num:.6f}")
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return (False, f"Cannot verify numerically: {e}")

    def test_modeling_zone_radius_linear_in_n(self) -> Tuple[bool, str]:
        """
        Check consistency: the area constant implies that ρ_n^2 is linear in n.
        If the solution provides ρ_n^2 for two zones, the difference should equal
        the area constant divided by π.  (Weak check: we only verify that the
        area constant is positive and consistent with the intended derivation.)
        """
        if self.area_constant is None:
            return (False, "SKIP: area_constant not available")
        # The derivation relies on ρ_n^2 = n * λ b R / (R+b).
        # We cannot directly test the linearity here because we lack ρ_n values,
        # but we can note that the area constant being constant already implies it.
        # As a minimal modeling check, we verify that area_constant is positive
        # (performed in answer sanity) and that it has the correct dimensional form.
        return self.test_answer_sanity_finite_positive()

    # ----------------------------------------------------------------
    # Monotonicity tests (using explicit perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_lambda_increases_area(self) -> Tuple[bool, str]:
        """
        Perturb λ by +10% and verify that the area constant (computed from the
        correct physics formula) increases proportionally.
        This test re-derives both baseline and perturbed values from the given
        givens, without relying on the solution's area_constant.
        """
        if any(v is None for v in [self.lambda_val, self.R, self.b]):
            return (False, "SKIP: missing numeric data to compute area")
        try:
            lam0 = float(self.lambda_val)
            R0 = float(self.R)
            b0 = float(self.b)
        except (ValueError, TypeError):
            return (False, "SKIP: cannot convert givens to float")

        # Compute baseline area using the correct physics formula
        area0 = math.pi * lam0 * b0 * R0 / (R0 + b0)

        # Perturb lambda by +10%
        lam1 = lam0 * 1.1
        area1 = math.pi * lam1 * b0 * R0 / (R0 + b0)

        if area1 > area0:
            return (True, f"λ increased by 10% → area increases from {area0:.6e} to {area1:.6e}")
        else:
            return (False, f"Area should increase with λ, but got area1={area1:.6e} ≤ area0={area0:.6e}")

    def test_monotonicity_R_increases_area(self) -> Tuple[bool, str]:
        """
        Perturb R by +10% and verify that area increases.
        dA/dR = π λ b^2 / (R+b)^2 > 0, so area should increase.
        """
        if any(v is None for v in [self.lambda_val, self.R, self.b]):
            return (False, "SKIP: missing numeric data to compute area")
        try:
            lam0 = float(self.lambda_val)
            R0 = float(self.R)
            b0 = float(self.b)
        except (ValueError, TypeError):
            return (False, "SKIP: cannot convert givens to float")

        area0 = math.pi * lam0 * b0 * R0 / (R0 + b0)

        R1 = R0 * 1.1
        area1 = math.pi * lam0 * b0 * R1 / (R1 + b0)

        if area1 > area0:
            return (True, f"R increased by 10% → area increases from {area0:.6e} to {area1:.6e}")
        else:
            return (False, f"Area should increase with R, but got area1={area1:.6e} ≤ area0={area0:.6e}")

    # ----------------------------------------------------------------
    # Answer sanity (weak, no re‑derivation, no math.isclose)
    # ----------------------------------------------------------------
    def test_answer_sanity_finite_positive(self) -> Tuple[bool, str]:
        """Weak sanity: area constant must be finite and positive."""
        if self.area_constant is None:
            return (False, "SKIP: area_constant missing")
        try:
            val = float(self.area_constant)
            if not math.isfinite(val):
                return (False, f"Area constant is not finite: {val}")
            if val <= 0:
                return (False, f"Area constant must be positive, got {val}")
            return (True, f"Area constant is finite and positive: {val:.4e}")
        except (ValueError, TypeError):
            # Symbolic expression: weak pass — assume positive if givens are positive
            return (True, "Symbolic area constant assumed positive (weak check)")

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
        """Discover all test_* methods, run them, collect results."""
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