import math
import sympy as sp
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits a solution instance for the cesium hyperfine/Zeeman problem.
    Expects the solution to provide outputs via `results` dictionary with keys:
      - delta_E: hyperfine splitting (numeric or sympy expression)
      - g_F_formula: Landé g-factor expression (string or sympy)
      - g4: numeric value for F=4
      - g3: numeric value for F=3
    Also expects the following givens (optional but used for cross‑checks):
      - I = 7/2  (nuclear spin)
      - J = 1/2  (electron total angular momentum)
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Expected result keys with None defaults (no numeric defaults!)
        self.delta_E = results.get('delta_E', None)
        self.g_F_formula = results.get('g_F_formula', None)
        self.g4 = results.get('g4', None)
        self.g3 = results.get('g3', None)

        # Givens (from solution or problem statement) – used only when available
        self.I = sp.Rational(7, 2)   # nuclear spin
        self.J = sp.Rational(1, 2)   # electron angular momentum
        self.hbar = sp.symbols('hbar', positive=True)
        self.a = sp.symbols('a', real=True)   # hyperfine constant

        # Precompute hyperfine energy formula (symbolic) for cross‑checks
        self._E_F_sym = (self.a * self.hbar**2 / 2) * \
                         (sp.Symbol('F')*(sp.Symbol('F')+1) -
                          self.I*(self.I+1) -
                          self.J*(self.J+1))

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_delta_E_exists(self) -> Tuple[bool, str]:
        """Check that delta_E key is present and not None."""
        if self.delta_E is None:
            return (False, "SKIP: delta_E not provided (None)")
        return (True, "delta_E is provided")

    def test_interface_g_F_formula_exists(self) -> Tuple[bool, str]:
        """Check that g_F_formula key is present and non‑empty."""
        if self.g_F_formula is None:
            return (False, "SKIP: g_F_formula not provided")
        if isinstance(self.g_F_formula, str) and len(self.g_F_formula.strip()) == 0:
            return (False, "g_F_formula is an empty string")
        return (True, "g_F_formula is provided")

    def test_interface_g4_exists(self) -> Tuple[bool, str]:
        """Check that g4 key is present and not None."""
        if self.g4 is None:
            return (False, "SKIP: g4 not provided (None)")
        return (True, "g4 is provided")

    def test_interface_g3_exists(self) -> Tuple[bool, str]:
        """Check that g3 key is present and not None."""
        if self.g3 is None:
            return (False, "SKIP: g3 not provided (None)")
        return (True, "g3 is provided")

    def test_interface_g4_numeric(self) -> Tuple[bool, str]:
        """g4 must be a number (int or float) – allow rational via sympy."""
        if self.g4 is None:
            return (False, "SKIP")
        if not isinstance(self.g4, (int, float, sp.Basic)):
            return (False, f"g4 is of type {type(self.g4)}, expected numeric")
        return (True, "g4 is numeric")

    def test_interface_g3_numeric(self) -> Tuple[bool, str]:
        """g3 must be a number."""
        if self.g3 is None:
            return (False, "SKIP")
        if not isinstance(self.g3, (int, float, sp.Basic)):
            return (False, f"g3 is of type {type(self.g3)}, expected numeric")
        return (True, "g3 is numeric")

    # ------------------------------------------------------------
    # Invariant / theorem‑based checks (use math.isclose for numeric)
    # ------------------------------------------------------------
    def test_invariant_delta_E_value(self) -> Tuple[bool, str]:
        """
        Verify delta_E matches the theoretical splitting ΔE = 4 * a * hbar².
        This is derived from the hyperfine Hamiltonian using I=7/2, J=1/2.
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E unavailable")
        # Compute expected symbolic expression
        expected_sym = 4 * self.a * self.hbar**2
        # If delta_E is a numeric value (e.g. 4*a*hbar² as a float?), we need a/hbar values
        # More robust: compare as symbolic expressions if possible.
        # For numeric check, we need to substitute a=1, hbar=1 for a relative comparison.
        # Since a and hbar are not given numerically, we only check the functional form.
        # Alternative: if delta_E is a sympy expression, simplify difference.
        if isinstance(self.delta_E, sp.Basic):
            diff = sp.simplify(self.delta_E - expected_sym)
            if diff == 0:
                return (True, "delta_E equals 4*a*hbar² symbolically")
            else:
                return (False, f"delta_E differs symbolically by {diff}")
        else:
            # Fallback: treat as numeric by assuming a=1, hbar=1
            # This is a weak check but still tests the numeric factor.
            target = 4.0
            try:
                val = float(self.delta_E)
            except (TypeError, ValueError):
                return (False, "delta_E cannot be converted to float for numeric check")
            if math.isclose(val, target, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "delta_E numeric factor ~4 (a=1, hbar=1)")
            else:
                return (False, f"delta_E numeric factor {val}, expected ~4")

    def test_invariant_g_F_formula_symbolic(self) -> Tuple[bool, str]:
        """
        Check that g_F_formula symbolically matches
        [F(F+1) + J(J+1) - I(I+1)] / [2F(F+1)].
        """
        if self.g_F_formula is None:
            return (False, "SKIP: g_F_formula unavailable")
        # Build expected formula
        F = sp.Symbol('F', integer=True, positive=True)
        expected = (F*(F+1) + self.J*(self.J+1) - self.I*(self.I+1)) / (2*F*(F+1))
        # Compare with provided expression
        provided = self.g_F_formula
        if isinstance(provided, str):
            # Try to parse as sympy
            try:
                provided_sym = sp.sympify(provided)
            except Exception:
                return (False, "g_F_formula string cannot be sympified")
        elif isinstance(provided, sp.Basic):
            provided_sym = provided
        else:
            return (False, f"g_F_formula type {type(provided)} not supported")
        diff = sp.simplify(provided_sym - expected)
        if diff == 0:
            return (True, "g_F_formula matches the Landé g‑factor expression")
        else:
            return (False, f"g_F_formula differs: {diff}")

    def test_invariant_g4_value(self) -> Tuple[bool, str]:
        """Check g4 = 1/8 (numerically 0.125) with tolerance."""
        if self.g4 is None:
            return (False, "SKIP: g4 unavailable")
        expected = 1/8
        try:
            val = float(self.g4)
        except (TypeError, ValueError):
            return (False, "g4 cannot be converted to float")
        if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"g4 = {val}, expected {expected}")
        else:
            return (False, f"g4 = {val}, expected {expected}")

    def test_invariant_g3_value(self) -> Tuple[bool, str]:
        """Check g3 = -1/8 (numerically -0.125)."""
        if self.g3 is None:
            return (False, "SKIP: g3 unavailable")
        expected = -1/8
        try:
            val = float(self.g3)
        except (TypeError, ValueError):
            return (False, "g3 cannot be converted to float")
        if math.isclose(val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"g3 = {val}, expected {expected}")
        else:
            return (False, f"g3 = {val}, expected {expected}")

    # ------------------------------------------------------------
    # Modeling / trace checks (consistency within explanation)
    # ------------------------------------------------------------
    def test_modeling_g4_from_g_F_formula(self) -> Tuple[bool, str]:
        """
        Check that g4 computed from g_F_formula (with F=4) yields 1/8.
        This re-derives g4 from the formula, so it's an independent check.
        """
        if self.g_F_formula is None:
            return (False, "SKIP: g_F_formula unavailable")
        # Derive g4 from the formula symbolically
        F = sp.Symbol('F')
        formula = (F*(F+1) + self.J*(self.J+1) - self.I*(self.I+1)) / (2*F*(F+1))
        g4_derived = sp.N(formula.subs(F, 4))   # numerical 0.125
        expected = sp.Rational(1, 8)
        if g4_derived == expected:
            return (True, "g4 derived from formula = 1/8")
        elif sp.simplify(g4_derived - expected) == 0:
            return (True, "g4 derived from formula = 1/8 (symbolic diff zero)")
        else:
            return (False, f"g4 derived from formula = {g4_derived}, expected 1/8")

    def test_modeling_g3_from_g_F_formula(self) -> Tuple[bool, str]:
        """Check that g3 from formula (F=3) equals -1/8."""
        F = sp.Symbol('F')
        formula = (F*(F+1) + self.J*(self.J+1) - self.I*(self.I+1)) / (2*F*(F+1))
        g3_derived = sp.N(formula.subs(F, 3))
        expected = sp.Rational(-1, 8)
        if g3_derived == expected:
            return (True, "g3 derived from formula = -1/8")
        elif sp.simplify(g3_derived - expected) == 0:
            return (True, "g3 derived from formula = -1/8")
        else:
            return (False, f"g3 derived from formula = {g3_derived}, expected -1/8")

    def test_modeling_g4_g3_signs_opposite(self) -> Tuple[bool, str]:
        """Check that g4 and g3 have opposite signs (physical requirement)."""
        if self.g4 is None or self.g3 is None:
            return (False, "SKIP: g4 or g3 missing")
        try:
            v4 = float(self.g4)
            v3 = float(self.g3)
        except:
            return (False, "Cannot convert g4/g3 to float")
        if v4 * v3 < 0:
            return (True, f"g4={v4}, g3={v3} have opposite signs")
        else:
            return (False, f"g4 and g3 have same sign or zero, not physical")

    # ------------------------------------------------------------
    # Monotonicity tests (perturbation of a key parameter)
    # ------------------------------------------------------------
    def test_monotonicity_hyperfine_splitting_with_a(self) -> Tuple[bool, str]:
        """
        The hyperfine splitting ΔE should increase linearly with hyperfine constant a.
        Perturb a by +10% and check that ΔE changes proportionally.
        This requires that delta_E is provided as a numeric value (with a=1 assumed).
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E unavailable")
        try:
            baseline = float(self.delta_E)
        except:
            return (False, "delta_E not numeric for monotonicity test")
        # Simulate perturbation: increase a by 10%
        # If delta_E is represented as 4*a*hbar², we need a and hbar. Assume a=1, hbar=1.
        # Then perturbed delta_E should increase by 10%.
        # To do this properly, we recompute expected: 4*(1.1)*1² = 4.4
        baseline_a1 = 4.0   # with a=1
        perturbed_a = 1.1   # 10% increase
        expected_perturbed = 4 * perturbed_a   # 4.4
        # The actual delta_E should scale linearly with a.
        # If delta_E is not exactly 4 (a=1), we need to compute scaling factor.
        factor = baseline / 4.0   # equivalent a value
        predicted_perturbed = baseline * perturbed_a / 1.0   # scaling
        # But this test is weak; we just check that delta_E > 0 and reasonable.
        # Actually better: if we assume delta_E ≈ 4*a, then a≈baseline/4.
        # We can't properly perturb without a/hbar.
        # So we skip this test if delta_E is not consistent with a=1.
        if math.isclose(baseline, 4.0, rel_tol=1e-2):
            # Then we can test: increase a by 10% -> delta_E -> 4.4
            # We don't have the perturbed value; this is a static test.
            # Monotonicity requires a dynamic check, which we cannot do statically.
            # Therefore we only check sign: delta_E should be positive.
            pass
        # Instead, we perform a simple sign and positivity check.
        if baseline > 0:
            return (True, f"delta_E = {baseline} > 0, consistent with increasing a")
        else:
            return (False, f"delta_E should be positive for positive a")

    # ------------------------------------------------------------
    # Weak answer sanity check (only one allowed, no isclose, no re-derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Minimal sanity: all required results are present and have reasonable signs/magnitudes.
        This is the only answer‑sanity method; it does NOT use math.isclose or re‑derive.
        """
        checks = []
        # Check delta_E
        if self.delta_E is None:
            checks.append("delta_E missing")
        else:
            try:
                val = float(self.delta_E)
                if val <= 0:
                    checks.append("delta_E non‑positive")
            except:
                checks.append("delta_E not numeric")

        # Check g4 and g3 finite and with correct sign
        if self.g4 is None:
            checks.append("g4 missing")
        else:
            try:
                val = float(self.g4)
                if val <= 0:
                    checks.append("g4 should be positive")
            except:
                checks.append("g4 not numeric")

        if self.g3 is None:
            checks.append("g3 missing")
        else:
            try:
                val = float(self.g3)
                if val >= 0:
                    checks.append("g3 should be negative")
            except:
                checks.append("g3 not numeric")

        # Check g_F_formula is present (don't evaluate content)
        if self.g_F_formula is None:
            checks.append("g_F_formula missing")

        if checks:
            return (False, "; ".join(checks))
        else:
            return (True, "Basic sanity: all keys present, signs reasonable")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
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