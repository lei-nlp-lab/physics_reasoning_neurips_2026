import math
import sympy as sp
from typing import Any, Dict, Optional, Tuple, List

class PhysicsChecker:
    """Audits solution for atomic multipole transition problem."""

    def __init__(self, solution_instance: Any):
        # Extract results dict safely
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        elif isinstance(solution_instance, dict):
            self.results = solution_instance
        else:
            self.results = {}

        # Required result keys from pseudocode [emit] contract
        self.multipole_1 = self.results.get('multipole_1', None)
        self.multipole_2 = self.results.get('multipole_2', None)
        self.multipole_3 = self.results.get('multipole_3', None)
        self.multipole_4 = self.results.get('multipole_4', None)
        self.multipole_5 = self.results.get('multipole_5', None)
        self.Gamma_E1_symbolic = self.results.get('Gamma_E1_symbolic', None)
        self.Gamma_numeric = self.results.get('Gamma_numeric', None)
        self.ratio_8 = self.results.get('ratio_8', None)
        self.ratio_9 = self.results.get('ratio_9', None)
        self.ratio_10 = self.results.get('ratio_10', None)
        self.ratio_11 = self.results.get('ratio_11', None)

        # Optional givens (may not be present)
        self.a = self.results.get('a', None)
        self.c = self.results.get('c', None)
        self.alpha = self.results.get('alpha', None)
        self.hbar = self.results.get('hbar', None)
        self.m_e = self.results.get('m_e', None)
        self.omega_21 = self.results.get('omega_21', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys exist and are not None."""
        required_keys = [
            'multipole_1', 'multipole_2', 'multipole_3', 'multipole_4', 'multipole_5',
            'Gamma_E1_symbolic', 'Gamma_numeric',
            'ratio_8', 'ratio_9', 'ratio_10', 'ratio_11'
        ]
        missing = [k for k in required_keys if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_multipole_types(self) -> Tuple[bool, str]:
        """Check that multipole results are strings."""
        for name in ['multipole_1', 'multipole_2', 'multipole_3', 'multipole_4', 'multipole_5']:
            val = getattr(self, name, None)
            if val is not None and not isinstance(val, str):
                return (False, f"{name} is not string: {type(val)}")
        return (True, "All multipole keys are strings")

    def test_interface_Gamma_numeric_type(self) -> Tuple[bool, str]:
        """Check Gamma_numeric is numeric (int or float)."""
        val = self.Gamma_numeric
        if val is None:
            return (False, "Gamma_numeric not provided")
        if not isinstance(val, (int, float)):
            return (False, f"Gamma_numeric has type {type(val)}")
        return (True, "Gamma_numeric is numeric")

    def test_interface_ratio_types(self) -> Tuple[bool, str]:
        """Check ratios 8-11 are strings (symbolic)."""
        for name in ['ratio_8', 'ratio_9', 'ratio_10', 'ratio_11']:
            val = getattr(self, name, None)
            if val is not None and not isinstance(val, str):
                return (False, f"{name} is not string: {type(val)}")
        return (True, "All ratio keys are strings")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, selection rules, scaling)
    # ------------------------------------------------------------------
    def test_invariant_E1_selection_1(self) -> Tuple[bool, str]:
        """2p1/2 -> 1s1/2 must be E1 (Δl=1, parity change, Δj=0 allowed)."""
        if self.multipole_1 is None:
            return (False, "SKIP: multipole_1 not provided")
        expected = 'E1'
        if self.multipole_1 == expected:
            return (True, f"multipole_1 = {expected}")
        else:
            return (False, f"multipole_1 = {self.multipole_1}, expected {expected}")

    def test_invariant_E1_selection_5(self) -> Tuple[bool, str]:
        """3d3/2 -> 2p1/2 must be E1 (Δl=1, parity change, Δj=1)."""
        if self.multipole_5 is None:
            return (False, "SKIP: multipole_5 not provided")
        expected = 'E1'
        if self.multipole_5 == expected:
            return (True, f"multipole_5 = {expected}")
        else:
            return (False, f"multipole_5 = {self.multipole_5}, expected {expected}")

    def test_invariant_two_photon(self) -> Tuple[bool, str]:
        """2s1/2 -> 1s1/2 must be two-photon E1 (Δl=0, parity same, single-photon forbidden)."""
        if self.multipole_2 is None:
            return (False, "SKIP: multipole_2 not provided")
        expected = 'two-photon E1'
        if self.multipole_2 == expected:
            return (True, f"multipole_2 = {expected}")
        else:
            return (False, f"multipole_2 = {self.multipole_2}, expected '{expected}'")

    def test_invariant_E2_selection(self) -> Tuple[bool, str]:
        """3d3/2 -> 2s1/2 must be E2 (Δl=2, parity same, lowest electric order)."""
        if self.multipole_3 is None:
            return (False, "SKIP: multipole_3 not provided")
        expected = 'E2'
        if self.multipole_3 == expected:
            return (True, f"multipole_3 = {expected}")
        else:
            return (False, f"multipole_3 = {self.multipole_3}, expected {expected}")

    def test_invariant_M1_selection(self) -> Tuple[bool, str]:
        """2p3/2 -> 2p1/2 must be M1 (Δl=0, parity same, Δj=1, magnetic)."""
        if self.multipole_4 is None:
            return (False, "SKIP: multipole_4 not provided")
        expected = 'M1'
        if self.multipole_4 == expected:
            return (True, f"multipole_4 = {expected}")
        else:
            return (False, f"multipole_4 = {self.multipole_4}, expected {expected}")

    def test_invariant_Gamma_E1_symbolic_form(self) -> Tuple[bool, str]:
        """Check that Gamma_E1_symbolic contains expected factors: alpha, omega^3, (a/c)^2."""
        s = self.Gamma_E1_symbolic
        if s is None:
            return (False, "SKIP: Gamma_E1_symbolic not provided")
        s_lower = s.lower().replace(' ', '').replace('*', '')
        expected_parts = ['alpha', 'omega', 'a', 'c']
        missing = [p for p in expected_parts if p not in s_lower]
        if missing:
            return (False, f"Missing symbolic parts: {missing} in '{s}'")
        # Optional sympy parse to verify structure
        try:
            expr = sp.sympify(s)
            symbols = expr.free_symbols
            sym_names = {str(sym) for sym in symbols}
            expected_sym = {'alpha', 'omega', 'a', 'c'}
            if not expected_sym.issubset(sym_names):
                return (False, f"Symbolic expression missing symbols: {expected_sym - sym_names}")
        except Exception:
            pass
        return (True, f"Gamma_E1_symbolic contains required components")

    def test_invariant_Gamma_numeric_magnitude(self) -> Tuple[bool, str]:
        """Weak numerical sanity: Gamma_numeric should be around 10^9 s^-1 (within 2 orders)."""
        val = self.Gamma_numeric
        if val is None:
            return (False, "SKIP: Gamma_numeric not provided")
        if not isinstance(val, (int, float)):
            return (False, f"Gamma_numeric not numeric: {type(val)}")
        if 1e7 <= val <= 1e11:
            return (True, f"Gamma_numeric = {val} is in acceptable range (1e7 to 1e11)")
        else:
            return (False, f"Gamma_numeric = {val} outside range 1e7..1e11")

    def test_invariant_ratio_8_symbolic(self) -> Tuple[bool, str]:
        """Check ratio_8 contains hbar, m_e, c, alpha."""
        s = self.ratio_8
        if s is None:
            return (False, "SKIP: ratio_8 not provided")
        s_lower = s.lower().replace(' ', '')
        if not any(part in s_lower for part in ['hbar', 'h_bar']):
            return (False, f"ratio_8 missing hbar: {s}")
        if 'm_e' not in s_lower and 'm' not in s_lower:
            return (False, f"ratio_8 missing electron mass: {s}")
        if 'c' not in s_lower:
            return (False, f"ratio_8 missing c: {s}")
        if 'alpha' not in s_lower:
            return (False, f"ratio_8 missing alpha: {s}")
        return (True, f"ratio_8 symbolic structure acceptable: {s}")

    def test_invariant_ratio_9_symbolic(self) -> Tuple[bool, str]:
        """Check ratio_9 is (ka)^2 or ka^2."""
        s = self.ratio_9
        if s is None:
            return (False, "SKIP: ratio_9 not provided")
        s_lower = s.lower().replace(' ', '')
        allowed = ['ka^2', '(ka)^2', 'k*a^2', '(k*a)^2']
        if s_lower in allowed or (s_lower.startswith('ka') and '2' in s_lower):
            return (True, f"ratio_9 = {s}")
        else:
            return (False, f"ratio_9 = {s} does not match (ka)^2 format")

    def test_invariant_ratio_10_symbolic(self) -> Tuple[bool, str]:
        """Check ratio_10 is (ka)^2 or ka^2."""
        s = self.ratio_10
        if s is None:
            return (False, "SKIP: ratio_10 not provided")
        s_lower = s.lower().replace(' ', '')
        allowed = ['ka^2', '(ka)^2', 'k*a^2', '(k*a)^2']
        if s_lower in allowed or (s_lower.startswith('ka') and '2' in s_lower):
            return (True, f"ratio_10 = {s}")
        else:
            return (False, f"ratio_10 = {s} does not match (ka)^2 format")

    def test_invariant_ratio_11_symbolic(self) -> Tuple[bool, str]:
        """Check ratio_11 is (omega_32/omega_21)^3 or similar."""
        s = self.ratio_11
        if s is None:
            return (False, "SKIP: ratio_11 not provided")
        s_lower = s.lower().replace(' ', '')
        allowed_phrases = ['omega', '32', '21', '^3']
        if all(phrase in s_lower for phrase in allowed_phrases):
            return (True, f"ratio_11 = {s}")
        else:
            return (False, f"ratio_11 = {s} does not match required form")

    # ------------------------------------------------------------------
    # Modeling/trace tests: check consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_Gamma_E1_uses_E1_formula(self) -> Tuple[bool, str]:
        """Check that Gamma_E1_symbolic corresponds to standard E1 formula (4/3)αω^3(a/c)^2."""
        s = self.Gamma_E1_symbolic
        if s is None:
            return (False, "SKIP: Gamma_E1_symbolic not provided")
        target = sp.sympify('4/3 * alpha * omega**3 * (a/c)**2')
        try:
            expr = sp.sympify(s)
            diff = sp.simplify(expr - target)
            if diff == 0:
                return (True, "Gamma_E1_symbolic matches standard expression")
            # Numerical equivalence test
            subs = {sp.Symbol('alpha'): 0.007, sp.Symbol('omega'): 1e16,
                    sp.Symbol('a'): 5e-11, sp.Symbol('c'): 3e8}
            val_expr = expr.subs(subs).evalf()
            val_target = target.subs(subs).evalf()
            if math.isclose(float(val_expr), float(val_target), rel_tol=1e-6):
                return (True, "Gamma_E1_symbolic numerically equivalent")
            else:
                return (False, f"Gamma_E1_symbolic numerically different: {val_expr} vs {val_target}")
        except Exception:
            if '4/3' in s and 'alpha' in s and 'omega' in s and 'a' in s and 'c' in s:
                return (True, "String match passed")
            return (False, f"Cannot parse Gamma_E1_symbolic: {s}")

    def test_modeling_ratio_8_derivation(self) -> Tuple[bool, str]:
        """Check that ratio_8 contains factor 10 and hbar/(m_e c alpha)."""
        s = self.ratio_8
        if s is None:
            return (False, "SKIP: ratio_8 not provided")
        s_clean = s.replace(' ', '')
        parts = ['10', 'hbar', 'm_e', 'c', 'alpha']
        if all(p in s_clean for p in parts):
            return (True, "ratio_8 contains required parts")
        return (False, f"ratio_8 = {s} does not match expected pattern")

    def test_modeling_ratio_9_10_ka2(self) -> Tuple[bool, str]:
        """Check that ratio_9 and ratio_10 both represent (ka)^2."""
        fail_msgs = []
        for name in ['ratio_9', 'ratio_10']:
            val = getattr(self, name, None)
            if val is None:
                fail_msgs.append(f"{name} missing")
                continue
            s = str(val).lower().replace(' ', '')
            if 'ka' not in s or '^2' not in s:
                fail_msgs.append(f"{name} does not contain ka^2: {val}")
        if fail_msgs:
            return (False, "; ".join(fail_msgs))
        return (True, "ratio_9 and ratio_10 both represent (ka)^2")

    def test_modeling_ratio_11_frequency_cube(self) -> Tuple[bool, str]:
        """Check that ratio_11 is proportional to (omega_32/omega_21)^3."""
        s = self.ratio_11
        if s is None:
            return (False, "SKIP: ratio_11 not provided")
        s_lower = s.lower().replace(' ', '')
        if 'omega' in s_lower and '/' in s_lower and '^3' in s_lower:
            return (True, f"ratio_11 = {s} acceptable")
        return (False, f"ratio_11 = {s} does not match required form")

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare baseline vs perturbed)
    # ------------------------------------------------------------------
    def test_monotonicity_Gamma_E1_with_omega(self) -> Tuple[bool, str]:
        """Increasing photon frequency ω should increase Gamma_E1 (positive exponent)."""
        s = self.Gamma_E1_symbolic
        if s is None:
            return (False, "SKIP: Gamma_E1_symbolic not provided")
        try:
            expr = sp.sympify(s)  # e.g., (4/3)*alpha*omega^3*(a/c)^2
        except Exception:
            return (False, "Cannot parse Gamma_E1_symbolic")
        # Define baseline values (typical atomic transition)
        baseline = {
            sp.Symbol('alpha'): 1/137,
            sp.Symbol('omega'): 1.55e16,  # rad/s (Lyman-alpha)
            sp.Symbol('a'): 5.29e-11,     # Bohr radius in m
            sp.Symbol('c'): 3e8
        }
        # Check that all symbols in expression are covered
        symbols = expr.free_symbols
        missing_syms = symbols - set(baseline.keys())
        if missing_syms:
            return (False, f"Expression uses unknown symbols: {missing_syms}")
        # Baseline value
        val0 = float(expr.subs(baseline).evalf())
        # Perturb omega +10%
        perturbed = baseline.copy()
        perturbed[sp.Symbol('omega')] = 1.1 * baseline[sp.Symbol('omega')]
        val1 = float(expr.subs(perturbed).evalf())
        if val1 > val0:
            return (True, f"Gamma increases with ω: baseline={val0:.3e}, +10% ω → {val1:.3e}")
        else:
            return (False, f"Gamma did not increase: baseline={val0:.3e}, perturbed={val1:.3e}")

    def test_monotonicity_ratio_9_increases_with_ka(self) -> Tuple[bool, str]:
        """Ratio (ka)^2 should increase when a (or ω) increases."""
        s = self.ratio_9
        if s is None:
            return (False, "SKIP: ratio_9 not provided")
        # Parse expression, typically "ka^2" or "(ka)^2". We need to interpret
        # We'll define a synthetic baseline for k and a
        # Common symbols: potentially k and a, or just ka
        try:
            expr = sp.sympify(s)  # e.g., ka^2 -> sympy will treat as k*a^2, unless ka is a single symbol
        except Exception:
            return (False, f"Cannot parse ratio_9: {s}")
        # Determine free symbols
        syms = expr.free_symbols
        # If exactly one symbol like Symbol('ka') then we cannot perturb individually; we skip.
        if len(syms) == 1:
            # Single symbol likely 'ka' – we cannot isolate, skip monotonicity test
            return (False, "SKIP: ratio_9 uses single symbol 'ka', cannot perturb individually")
        # Try to find a and k or omega and c
        # We'll use typical values
        baseline = {}
        if sp.Symbol('k') in syms:
            baseline[sp.Symbol('k')] = 1.55e16 / 3e8  # ω/c
        elif sp.Symbol('omega') in syms:
            baseline[sp.Symbol('omega')] = 1.55e16
        if sp.Symbol('a') in syms:
            baseline[sp.Symbol('a')] = 5.29e-11
        if not baseline:
            return (False, "Cannot identify symbols for perturbation")
        # Fill missing symbols with dummy 1 (if any left)
        for sym in syms:
            if sym not in baseline:
                baseline[sym] = 1.0
        try:
            val0 = float(expr.subs(baseline).evalf())
        except Exception:
            return (False, "Cannot evaluate baseline")
        # Perturb a by +10% if present
        if sp.Symbol('a') in baseline:
            perturbed = baseline.copy()
            perturbed[sp.Symbol('a')] = 1.1 * baseline[sp.Symbol('a')]
            val1 = float(expr.subs(perturbed).evalf())
            if val1 > val0:
                return (True, f"ratio_9 increases with a: baseline={val0:.4e}, +10% a → {val1:.4e}")
            else:
                return (False, f"ratio_9 did not increase: baseline={val0:.4e}, perturbed={val1:.4e}")
        # else try perturb omega/k
        if sp.Symbol('omega') in baseline:
            perturbed = baseline.copy()
            perturbed[sp.Symbol('omega')] = 1.1 * baseline[sp.Symbol('omega')]
            val1 = float(expr.subs(perturbed).evalf())
            if val1 > val0:
                return (True, f"ratio_9 increases with ω: baseline={val0:.4e}, +10% ω → {val1:.4e}")
            else:
                return (False, f"ratio_9 did not increase: baseline={val0:.4e}, perturbed={val1:.4e}")
        return (False, "Cannot find suitable parameters for perturbation")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, broad magnitude)
    # ------------------------------------------------------------------
    def test_answer_sanity_Gamma_numeric_broad(self) -> Tuple[bool, str]:
        """Broad sanity: Gamma_numeric must be between 1e6 and 1e12 s^-1."""
        val = self.Gamma_numeric
        if val is None:
            return (False, "SKIP: Gamma_numeric not provided")
        if 1e6 <= val <= 1e12:
            return (True, f"Gamma_numeric = {val} in broad range")
        else:
            return (False, f"Gamma_numeric = {val} outside 1e6..1e12")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods."""
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# ------------------------------------------------------------------
# Example usage (for testing)
# ------------------------------------------------------------------
if __name__ == "__main__":
    class MockSolution:
        def __init__(self):
            self.results = {
                'multipole_1': 'E1',
                'multipole_2': 'two-photon E1',
                'multipole_3': 'E2',
                'multipole_4': 'M1',
                'multipole_5': 'E1',
                'Gamma_E1_symbolic': '(4/3) * alpha * omega**3 * (a/c)**2',
                'Gamma_numeric': 1.13e9,
                'ratio_8': '10 * hbar/(m_e * c * alpha)',
                'ratio_9': 'ka^2',
                'ratio_10': 'ka^2',
                'ratio_11': '(omega_32 / omega_21)**3'
            }

    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    print("\n".join(logs))
    print(f"\nOverall success: {success}")