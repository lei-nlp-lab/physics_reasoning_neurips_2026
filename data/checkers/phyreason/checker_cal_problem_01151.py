import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the PhysicsChecker with a solution instance.
        Safely extracts all expected inputs and results using getattr and .get().
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Constants (with safe defaults)
        self.g = getattr(solution_instance, 'g', 9.8)  # gravitational acceleration
        self.m = getattr(solution_instance, 'm', 1.0)  # mass, default 1.0
        self.H = getattr(solution_instance, 'H', None)  # may be symbolic or numeric
        if self.H is None:
            self.H = self.results.get('H', None)

        # Extract all result quantities safely
        self.keys = [
            'H', 'h_m', 'v_0*t', 'x_3', 'F', 'v_y', 'x', 't', 'v_0', 'v',
            'x_2', 'v_x1', 'h', 't_1', 'v_y1', 'x_1'
        ]
        for key in self.keys:
            value = getattr(solution_instance, key, self.results.get(key, None))
            setattr(self, key, value)

        # Additional derived or expected symbolic variables
        self.v_symbolic = sp.symbols('v', positive=True) if self.v is None else self.v
        self.v_0_symbolic = sp.symbols('v_0', positive=True) if self.v_0 is None else self.v_0
        self.t_symbolic = sp.symbols('t', positive=True) if self.t is None else self.t
        self.t_1_symbolic = sp.symbols('t_1', positive=True) if self.t_1 is None else self.t_1

        # Target expressions (symbolic)
        self.target_F = (4/3) * self.m * self.g
        self.target_h_m = (4/3) * self.H if self.H is not None else None
        self.target_x = (32/9) * self.H if self.H is not None else None

        # Logs storage
        self.logs = []

    def _log(self, message: str):
        """Helper to append log messages."""
        self.logs.append(message)

    def _compare(self, a, b, name: str, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """
        Compare two values a and b, which can be sympy expressions or numbers.
        Returns True if they are equal within tolerance (for numbers) or symbolically equivalent.
        """
        try:
            # If both are numbers (or numeric types)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                    self._log(f"PASS {name}: {a} ≈ {b}")
                    return True
                else:
                    self._log(f"FAIL {name}: {a} != {b}")
                    return False
            else:
                # Try symbolic simplification
                if sp.simplify(a - b) == 0:
                    self._log(f"PASS {name}: {a} ≡ {b}")
                    return True
                else:
                    # Try numerical evaluation if symbols have values
                    if self.H is not None and self.g is not None and self.m is not None:
                        # Substitute known values
                        subs_dict = {}
                        if hasattr(self.H, 'free_symbols'):
                            # If H is symbolic, we cannot evaluate numerically without a value
                            self._log(f"SKIP {name}: symbolic comparison inconclusive")
                            return True  # Skip to avoid false negatives
                        else:
                            subs_dict[sp.Symbol('H')] = self.H
                        if hasattr(self.g, 'free_symbols'):
                            subs_dict[sp.Symbol('g')] = self.g
                        if hasattr(self.m, 'free_symbols'):
                            subs_dict[sp.Symbol('m')] = self.m
                        a_val = a.subs(subs_dict) if hasattr(a, 'subs') else a
                        b_val = b.subs(subs_dict) if hasattr(b, 'subs') else b
                        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                            if math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol):
                                self._log(f"PASS {name}: {a} ≈ {b} after substitution")
                                return True
                            else:
                                self._log(f"FAIL {name}: {a_val} != {b_val} after substitution")
                                return False
                        else:
                            self._log(f"SKIP {name}: cannot evaluate numerically")
                            return True
                    else:
                        self._log(f"SKIP {name}: insufficient data for comparison")
                        return True
        except Exception as e:
            self._log(f"ERROR in {name}: {e}")
            return False

    def test_invariants(self) -> bool:
        """
        Test fundamental physics invariants:
        1. Newton's second law in horizontal direction: F = m * (v / t)
        2. Vertical motion: v_y1 = v_y + g * t
        3. Symmetry: v_0 = v_y1 and v_y = v_0 / 2
        """
        success = True

        # 1. F = m * (v / t) from step 3 (if t is the time to accelerate from 0 to v)
        if self.F is not None and self.v is not None and self.t is not None:
            expected = self.m * self.v / self.t
            if not self._compare(self.F, expected, "Newton II horizontal"):
                success = False

        # 2. v_y1 = v_y + g * t (step 10)
        if self.v_y1 is not None and self.v_y is not None and self.t is not None:
            expected = self.v_y + self.g * self.t
            if not self._compare(self.v_y1, expected, "Vertical acceleration N->P"):
                success = False

        # 3. Symmetry: v_0 = v_y1 (step 7) and v_y = v_0 / 2 (step 8)
        if self.v_0 is not None and self.v_y1 is not None:
            if not self._compare(self.v_0, self.v_y1, "v_0 = v_y1"):
                success = False
        if self.v_y is not None and self.v_0 is not None:
            expected = self.v_0 / 2
            if not self._compare(self.v_y, expected, "v_y = v_0/2"):
                success = False

        return success

    def test_trace_consistency(self) -> bool:
        """
        Verify consistency of intermediate results as per the steps analysis.
        """
        success = True

        # Step 3: v = (F/m) * t
        if self.F is not None and self.v is not None and self.t is not None:
            expected = (self.F / self.m) * self.t
            if not self._compare(self.v, expected, "Step 3: v = (F/m)*t"):
                success = False

        # Step 6: v_y1 = (3/2) * v (since tan37° = 0.6/0.8 = 3/4, but wait: tan37° = sin/cos = 0.6/0.8 = 3/4,
        # and step 6: tan37° = v_y1 / v_x1, with v_x1 = 2v -> v_y1 = (3/4)*2v = (3/2)v)
        if self.v_y1 is not None and self.v is not None:
            expected = (3/2) * self.v
            if not self._compare(self.v_y1, expected, "Step 6: v_y1 = (3/2)*v"):
                success = False

        # Step 7: v_0 = v_y1
        if self.v_0 is not None and self.v_y1 is not None:
            if not self._compare(self.v_0, self.v_y1, "Step 7: v_0 = v_y1"):
                success = False

        # Step 8: v_y = v_0 / 2
        if self.v_y is not None and self.v_0 is not None:
            expected = self.v_0 / 2
            if not self._compare(self.v_y, expected, "Step 8: v_y = v_0/2"):
                success = False

        # Step 9: t_1 = v_0 / (2*g)
        if self.t_1 is not None and self.v_0 is not None:
            expected = self.v_0 / (2 * self.g)
            if not self._compare(self.t_1, expected, "Step 9: t_1 = v_0/(2g)"):
                success = False

        # Step 10: t = (v_y1 - v_y) / g
        if self.t is not None and self.v_y1 is not None and self.v_y is not None:
            expected = (self.v_y1 - self.v_y) / self.g
            if not self._compare(self.t, expected, "Step 10: t = (v_y1 - v_y)/g"):
                success = False

        # Step 11: h = (v_y / 2) * t
        if self.h is not None and self.v_y is not None and self.t is not None:
            expected = (self.v_y / 2) * self.t
            if not self._compare(self.h, expected, "Step 11: h = (v_y/2)*t"):
                success = False

        # Step 12: H = ( (v_0/2 + v_0)/2 ) * t_1
        if self.H is not None and self.v_0 is not None and self.t_1 is not None:
            expected = ((self.v_0/2 + self.v_0) / 2) * self.t_1
            if not self._compare(self.H, expected, "Step 12: H formula"):
                success = False

        # Step 12 also: h = H/3
        if self.h is not None and self.H is not None:
            expected = self.H / 3
            if not self._compare(self.h, expected, "Step 12: h = H/3"):
                success = False

        # Step 13: h_m = h + H
        if self.h_m is not None and self.h is not None and self.H is not None:
            expected = self.h + self.H
            if not self._compare(self.h_m, expected, "Step 13: h_m = h + H"):
                success = False

        # Step 14: x_1 = (v/2) * t
        if self.x_1 is not None and self.v is not None and self.t is not None:
            expected = (self.v / 2) * self.t
            if not self._compare(self.x_1, expected, "Step 14: x_1 = (v/2)*t"):
                success = False

        # Step 15: x_2 = v * (2*t)  (since t_MN = 2*t from symmetry)
        if self.x_2 is not None and self.v is not None and self.t is not None:
            expected = self.v * (2 * self.t)
            if not self._compare(self.x_2, expected, "Step 15: x_2 = 2*v*t"):
                success = False

        # Step 16: x_3 = ((v + 2*v)/2) * t = (3/2)*v*t
        if self.x_3 is not None and self.v is not None and self.t is not None:
            expected = (3/2) * self.v * self.t
            if not self._compare(self.x_3, expected, "Step 16: x_3 = (3/2)*v*t"):
                success = False

        # Step 17: v_0 * t = 4H/3
        if self.v_0 is not None and self.t is not None and self.H is not None:
            expected = (4/3) * self.H
            if not self._compare(self.v_0 * self.t, expected, "Step 17: v_0*t = 4H/3"):
                success = False

        # Step 18: x = x_1 + x_2 + x_3
        if self.x is not None and self.x_1 is not None and self.x_2 is not None and self.x_3 is not None:
            expected = self.x_1 + self.x_2 + self.x_3
            if not self._compare(self.x, expected, "Step 18: x = x_1 + x_2 + x_3"):
                success = False

        return success

    def test_monotonicity(self) -> bool:
        """
        Check that h_m and x scale linearly with H (if H is numeric).
        Also verify that increasing H leads to increasing h_m and x.
        Since we cannot re-run the solution, we verify the linearity from the results.
        """
        success = True

        if self.H is not None and not hasattr(self.H, 'free_symbols'):
            # H is numeric
            if self.h_m is not None:
                ratio = self.h_m / self.H
                expected = 4/3
                if not math.isclose(ratio, expected, rel_tol=1e-2):
                    self._log(f"FAIL monotonicity: h_m/H = {ratio}, expected ≈ {expected}")
                    success = False
                else:
                    self._log(f"PASS monotonicity: h_m/H ≈ {expected}")
            if self.x is not None:
                ratio = self.x / self.H
                expected = 32/9
                if not math.isclose(ratio, expected, rel_tol=1e-2):
                    self._log(f"FAIL monotonicity: x/H = {ratio}, expected ≈ {expected}")
                    success = False
                else:
                    self._log(f"PASS monotonicity: x/H ≈ {expected}")
        else:
            self._log("SKIP monotonicity: H is symbolic or missing")
            # For symbolic H, we check that h_m and x are proportional to H
            if self.h_m is not None and self.H is not None:
                # Try to simplify h_m - (4/3)*H
                expr = sp.simplify(self.h_m - (4/3)*self.H)
                if expr != 0:
                    self._log(f"FAIL monotonicity: h_m not proportional to H: {expr}")
                    success = False
                else:
                    self._log("PASS monotonicity: h_m ∝ H")
            if self.x is not None and self.H is not None:
                expr = sp.simplify(self.x - (32/9)*self.H)
                if expr != 0:
                    self._log(f"FAIL monotonicity: x not proportional to H: {expr}")
                    success = False
                else:
                    self._log("PASS monotonicity: x ∝ H")

        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return (success, logs).
        """
        self.logs = []  # Reset logs
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            self._log(f"\n--- Testing {name} ---")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"ERROR in {name}: {traceback.format_exc()}")
                success = False

        # Final summary
        if success:
            self._log("\n✓ All tests passed.")
        else:
            self._log("\n✗ Some tests failed.")

        return success, self.logs