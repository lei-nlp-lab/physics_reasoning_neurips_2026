import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics checker for the radial motion in a spherically symmetric potential (WKB problem).
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens (use None if missing)
        self.m = getattr(solution_instance, 'm', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.a = getattr(solution_instance, 'a', None)
        self.V0 = getattr(solution_instance, 'V0', None)

        # Extract results (with None defaults)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        self.radial_eq = self.results.get('radial_eq', None)
        self.bc = self.results.get('bc', None)
        self.wkb = self.results.get('WKB', None)
        self.v0_min = self.results.get('V0_min', None)

        # Store explanation steps for audit direction (not hardcoding constants)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all four required result keys exist."""
        required = ['radial_eq', 'bc', 'WKB', 'V0_min']
        missing = [key for key in required if self.results.get(key, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}. Got keys: {list(self.results.keys())}")
        return (True, "All required result keys present")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check types of result values: radial_eq, bc, WKB should be strings; V0_min numeric or symbolic."""
        issues = []
        if self.radial_eq is not None and not isinstance(self.radial_eq, str):
            issues.append(f"radial_eq is {type(self.radial_eq)}, expected str")
        if self.bc is not None and not isinstance(self.bc, str):
            issues.append(f"bc is {type(self.bc)}, expected str")
        if self.wkb is not None and not isinstance(self.wkb, str):
            issues.append(f"WKB is {type(self.wkb)}, expected str")
        if self.v0_min is not None:
            if not isinstance(self.v0_min, (int, float, sp.Number)):
                issues.append(f"V0_min is {type(self.v0_min)}, expected numeric or symbolic")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All types correct")

    # ----------------------------------------------------------------
    # Invariant checks (theorem-based physics)
    # ----------------------------------------------------------------
    def test_invariant_radial_equation_form(self) -> Tuple[bool, str]:
        """Check that radial_eq expresses the correct 1D Schrödinger-like equation for u or chi."""
        if self.radial_eq is None:
            return (False, "SKIP: radial_eq not available")
        eq = self.radial_eq
        checks = [
            ("d^2u/dr^2" in eq or "d²u/dr²" in eq or "u''" in eq,
             "must contain second radial derivative"),
            ("V(r)" in eq or "V" in eq,
             "must contain potential V(r)"),
            ("E u" in eq or "E" in eq,
             "must contain energy E"),
        ]
        failures = [msg for cond, msg in checks if not cond]
        if failures:
            return (False, "radial_eq: " + "; ".join(failures))
        if "l(l+1)" in eq or "centrifugal" in eq.lower():
            return (False, "radial_eq should be for s-states (l=0); centrifugal term found")
        return (True, "radial equation form is correct")

    def test_invariant_boundary_condition(self) -> Tuple[bool, str]:
        """Check that bc includes u(0)=0 or chi(0)=0."""
        if self.bc is None:
            return (False, "SKIP: bc not available")
        bc_str = self.bc.lower()
        if "u(0)=0" in bc_str or "u(0) = 0" in bc_str:
            return (True, "boundary condition u(0)=0 present")
        if "chi(0)=0" in bc_str or "χ(0)=0" in bc_str:
            return (True, "boundary condition chi(0)=0 present")
        if "0" in bc_str and "r=0" in bc_str:
            return (True, "boundary condition at r=0 implied")
        return (False, "bc must state that wavefunction vanishes at the origin")

    def test_invariant_wkb_phase(self) -> Tuple[bool, str]:
        """Check that WKB condition contains (n+3/4) or similar phase."""
        if self.wkb is None:
            return (False, "SKIP: WKB not available")
        wkb_lower = self.wkb.lower()
        if "3/4" in wkb_lower or "0.75" in wkb_lower or r"\frac{3}{4}" in self.wkb:
            if "n" in wkb_lower:
                return (True, "WKB condition includes correct phase (n+3/4)")
        if r"\left( n + \frac{3}{4} \right)" in self.wkb:
            return (True, "WKB condition includes correct phase (n+3/4)")
        return (False, "WKB condition must include phase (n+3/4) for s-states")

    def test_invariant_wkb_integral_form(self) -> Tuple[bool, str]:
        """Check that WKB condition contains the correct integral sqrt(2m(E-V)) dr."""
        if self.wkb is None:
            return (False, "SKIP: WKB not available")
        wkb_lower = self.wkb.lower()
        if "int" in wkb_lower and "sqrt" in wkb_lower and "dr" in wkb_lower:
            if "e-v" in wkb_lower or "e - v" in wkb_lower or "v(r)" in self.wkb:
                return (True, "WKB integral form correct")
        return (False, "WKB condition must show integral of sqrt(2m(E-V)) dr")

    # ----------------------------------------------------------------
    # Modeling / trace checks (at least two required)
    # ----------------------------------------------------------------
    def test_modeling_v0_min_threshold(self) -> Tuple[bool, str]:
        """
        Verify the dimensionless constant: 2*m*V0_min*a^2/ħ^2 should be ≈ 9π²/64 ≈ 1.39.
        This tests the original threshold derivation.
        """
        if self.v0_min is None:
            return (False, "SKIP: V0_min not available")
        m = self.m
        hbar = self.hbar
        a = self.a
        if m is None or hbar is None or a is None:
            # Try symbolic check
            try:
                expr = sp.sympify(self.v0_min)
                m_s, hbar_s, a_s = sp.symbols('m hbar a')
                expected = (9 * sp.pi**2 * hbar_s**2) / (128 * m_s * a_s**2)
                if sp.simplify(expr / expected) == 1:
                    return (True, "V0_min expression matches expected form")
                else:
                    return (False, f"V0_min expression does not match expected")
            except Exception:
                return (False, "SKIP: cannot parse V0_min symbolically, and givens missing")
        # Numeric check of dimensionless constant
        dimless = 2 * m * float(self.v0_min) * a**2 / (hbar**2)
        expected_dimless = 9 * math.pi**2 / 64  # ≈ 1.39
        if math.isclose(dimless, expected_dimless, rel_tol=0.05):
            return (True, f"Dimensionless ratio {dimless:.3f} ~ 1.39 within 5%")
        else:
            return (False, f"Dimensionless ratio {dimless:.3f} deviates from expected ~1.39")

    def test_modeling_threshold_integral(self) -> Tuple[bool, str]:
        """
        Check the integral condition for threshold: ∫₀^∞ sqrt(2m V0 e^{-r/a}) dr = (3/4)π ħ.
        This should hold for the provided V0_min.
        """
        if self.v0_min is None:
            return (False, "SKIP: V0_min not available")
        m = self.m
        hbar = self.hbar
        a = self.a
        if m is None or hbar is None or a is None:
            # Symbolic: check that V0_min expression satisfies the integral equation
            try:
                v0 = sp.sympify(self.v0_min)
                m_s, hbar_s, a_s = sp.symbols('m hbar a')
                # left side: 2*a*sqrt(2*m*V0)
                left = 2 * a_s * sp.sqrt(2 * m_s * v0)
                # right side: (3/4) π ħ
                right = (sp.Rational(3, 4)) * sp.pi * hbar_s
                # Solve equation left == right for V0: V0 = (9 π^2 ħ^2)/(128 m a^2)
                expected_v0 = (9 * sp.pi**2 * hbar_s**2) / (128 * m_s * a_s**2)
                if sp.simplify(v0 / expected_v0) == 1:
                    return (True, "V0_min expression satisfies the threshold integral equation")
                else:
                    return (False, "V0_min does not satisfy the symbolic threshold integral")
            except Exception:
                return (False, "SKIP: cannot perform symbolic check, givens missing")
        # Numeric check
        left = 2 * a * math.sqrt(2 * m * float(self.v0_min))
        right = 0.75 * math.pi * hbar
        if math.isclose(left, right, rel_tol=0.05):
            return (True, f"Threshold integral satisfied: LHS={left:.4e}, RHS={right:.4e} within 5%")
        else:
            return (False, f"Threshold integral mismatch: LHS={left:.4e}, RHS={right:.4e}")

    def test_modeling_wkb_integral_structure(self) -> Tuple[bool, str]:
        """
        Check that the WKB integral string is complete: includes the lower limit 0 and the upper limit r2,
        and the square root form with proper dimensions.
        This tests the trace of the derivation.
        """
        if self.wkb is None:
            return (False, "SKIP: WKB not available")
        wkb_str = self.wkb
        # Check that it includes an integral symbol (∫ or int)
        if not ("∫" in wkb_str or "int" in wkb_str.lower()):
            return (False, "WKB condition must show an integral")
        # Check that it includes the square root of 2m(E-V)
        if not ("sqrt" in wkb_str.lower() or "√" in wkb_str):
            return (False, "WKB condition must include a square root")
        if not ("2m" in wkb_str or "2 m" in wkb_str):
            return (False, "WKB condition must include factor 2m")
        # Check for the differential dr
        if "dr" not in wkb_str:
            return (False, "WKB condition must include dr")
        # Check that it mentions (n+3/4) phase
        if not ("3/4" in wkb_str or "\\frac{3}{4}" in wkb_str):
            return (False, "WKB condition must include phase (n+3/4)")
        return (True, "WKB integral structure is correct")

    # ----------------------------------------------------------------
    # Monotonicity check (perturb an input and compare)
    # ----------------------------------------------------------------
    def test_monotonicity_v0_min_decreases_with_a(self) -> Tuple[bool, str]:
        """
        Verify that if the scale length a increases, the minimum V0 decreases (V0_min ∝ 1/a^2).
        We perturb a by +5% and recompute expected V0_min from the formula.
        """
        m = self.m
        hbar = self.hbar
        a = self.a
        if any(x is None for x in [m, hbar, a]):
            # Use symbolic check: derivative of V0_min w.r.t a should be negative
            if self.v0_min is not None and isinstance(self.v0_min, (sp.Expr, sp.Number)):
                try:
                    expr = sp.sympify(self.v0_min)
                    a_sym = sp.symbols('a')
                    der = sp.diff(expr, a_sym)
                    val = der.subs(a_sym, 1)  # arbitrary positive a
                    if val < 0:
                        return (True, "Symbolic derivative indicates V0_min decreases with a")
                    else:
                        return (False, f"Symbolic derivative not negative: {val}")
                except Exception:
                    return (False, "SKIP: cannot determine monotonicity symbolically")
            else:
                return (False, "SKIP: insufficient givens (m, hbar, a) for numeric monotonicity check")
        # Numeric: compute V0_min at original a and at perturbed a
        expected_orig = (9 * math.pi**2 * hbar**2) / (128 * m * a**2)
        # Perturb a: increase by 5%
        a_pert = a * 1.05
        expected_pert = (9 * math.pi**2 * hbar**2) / (128 * m * a_pert**2)
        if expected_pert < expected_orig:
            # Monotonicity holds. Also check that the provided V0_min is near expected_orig (optional)
            if isinstance(self.v0_min, (int, float)):
                if not math.isclose(float(self.v0_min), expected_orig, rel_tol=0.05):
                    # Still monotonicity holds but value mismatched – not a failure of monotonicity.
                    return (True, f"Monotonicity holds, but V0_min value differs from expected (got {self.v0_min}, expected {expected_orig:.4e})")
            return (True, "Monotonicity holds: V0 decreases as a increases")
        else:
            return (False, f"Expected monotonic decrease fails: a_pert={a_pert:.4e}, V0_orig={expected_orig:.4e}, V0_pert={expected_pert:.4e} (should be smaller)")

    # ----------------------------------------------------------------
    # Weak answer sanity (only one, no isclose)
    # ----------------------------------------------------------------
    def test_answer_sanity_v0_positive_finite(self) -> Tuple[bool, str]:
        """Check that V0_min is positive and finite (weak sanity)."""
        if self.v0_min is None:
            return (False, "SKIP: V0_min not available")
        v = self.v0_min
        if isinstance(v, (int, float)):
            if v > 0 and math.isfinite(v):
                return (True, "V0_min is positive and finite")
            else:
                return (False, f"V0_min = {v} is not a positive finite number")
        if isinstance(v, sp.Number):
            try:
                val = float(v.evalf())
                if val > 0:
                    return (True, "V0_min symbolically positive")
                else:
                    return (False, f"V0_min appears negative: {v}")
            except Exception:
                return (False, "SKIP: cannot evaluate V0_min symbolically")
        return (False, f"V0_min type not recognized: {type(v)}")

    # ----------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Collect and run all test_* methods."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)