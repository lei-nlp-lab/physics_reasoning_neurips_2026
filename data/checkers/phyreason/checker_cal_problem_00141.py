import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Helper to safely get attributes
        def safe_get(key, default=0.0):
            val = getattr(solution_instance, key, None)
            if val is None:
                val = self.results.get(key, default)
            return val

        # Extract inputs (with safe defaults to avoid AttributeError)
        self.m = safe_get('m', 1.0)
        self.v0 = safe_get('v0', 1.0)
        self.q = safe_get('q', 1.0)
        self.E = safe_get('E', 1.0)
        self.l = safe_get('l', 1.0)

        # Extract results (with safe defaults)
        self.s = safe_get('s')
        self.x = safe_get('x')
        self.y = safe_get('y')
        self.r_min = safe_get('r_min')
        self.v_y = safe_get('v_y')
        self.d_min = safe_get('d_min')
        self.v = safe_get('v')
        self.r_max = safe_get('r_max')
        self.B = safe_get('B')  # This might be a string or tuple
        self.F_c = safe_get('F_c')
        self.r_3 = safe_get('r_3')
        self.sintheta = safe_get('sin(theta)')

        # For B range, try to extract min and max if available
        self.B_min = safe_get('B_min')
        self.B_max = safe_get('B_max')
        # If not separate, try to parse from B string (fallback)
        if self.B_min is None and isinstance(self.B, str):
            try:
                parts = self.B.split('<=')  # Expect "expr <= B <= expr"
                if len(parts) == 3:
                    expr_min = parts[0].strip()
                    expr_max = parts[2].strip()
                    # Evaluate expressions symbolically
                    m, v0, q, l = sp.symbols('m v0 q l')
                    expr_min_sym = sp.sympify(expr_min)
                    expr_max_sym = sp.sympify(expr_max)
                    self.B_min = float(expr_min_sym.subs({m: self.m, v0: self.v0, q: self.q, l: self.l}))
                    self.B_max = float(expr_max_sym.subs({m: self.m, v0: self.v0, q: self.q, l: self.l}))
            except:
                pass

        # Store logs for reporting
        self.logs = []

    def _isclose(self, a, b):
        """Fuzzy numerical comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Check velocity decomposition at point P
            v_y_expected = self.v0 * math.tan(math.radians(30))  # tan(30°) = vy/v0
            if not self._isclose(self.v_y, v_y_expected):
                self.logs.append(f"FAIL: v_y mismatch. Expected {v_y_expected}, got {self.v_y}")
                return False

            # 2. Check speed at P: v = v0 / cos(30°)
            v_expected = self.v0 / math.cos(math.radians(30))
            if not self._isclose(self.v, v_expected):
                self.logs.append(f"FAIL: v mismatch. Expected {v_expected}, got {self.v}")
                return False

            # 3. Check time from electric field motion: t = v_y / a, where a = qE/m
            a = self.q * self.E / self.m
            t = self.v_y / a
            # Horizontal displacement x = v0 * t
            x_expected = self.v0 * t
            # Vertical displacement y = 0.5 * a * t^2
            y_expected = 0.5 * a * t**2
            if self.x and not self._isclose(self.x, x_expected):
                self.logs.append(f"FAIL: x mismatch. Expected {x_expected}, got {self.x}")
                return False
            if self.y and not self._isclose(self.y, y_expected):
                self.logs.append(f"FAIL: y mismatch. Expected {y_expected}, got {self.y}")
                return False

            # 4. Check distance s = sqrt(x^2 + y^2)
            s_expected = math.sqrt(x_expected**2 + y_expected**2)
            if not self._isclose(self.s, s_expected):
                self.logs.append(f"FAIL: s mismatch. Expected {s_expected}, got {self.s}")
                return False

            # 5. Check geometric relations for r_min and r_max
            r_min_expected = (self.l / 2) / math.cos(math.radians(30))
            if not self._isclose(self.r_min, r_min_expected):
                self.logs.append(f"FAIL: r_min mismatch. Expected {r_min_expected}, got {self.r_min}")
                return False

            # r_max: half the diagonal of the square of side l, divided by cos(75°)
            half_diag = (self.l * math.sqrt(2)) / 2
            r_max_expected = half_diag / math.cos(math.radians(75))
            if not self._isclose(self.r_max, r_max_expected):
                self.logs.append(f"FAIL: r_max mismatch. Expected {r_max_expected}, got {self.r_max}")
                return False

            # 6. Check Lorentz force relation: qvB = mv^2/r => B = mv/(qr)
            if self.B_min and self.B_max:
                # Check B_min corresponds to r_max
                B_min_expected = self.m * self.v / (self.q * self.r_max)
                if not self._isclose(self.B_min, B_min_expected):
                    self.logs.append(f"FAIL: B_min mismatch. Expected {B_min_expected}, got {self.B_min}")
                    return False
                # Check B_max corresponds to r_min
                B_max_expected = self.m * self.v / (self.q * self.r_min)
                if not self._isclose(self.B_max, B_max_expected):
                    self.logs.append(f"FAIL: B_max mismatch. Expected {B_max_expected}, got {self.B_max}")
                    return False

            self.logs.append("PASS: All invariants satisfied.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {e}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check that s matches the derived formula
            s_expected = (math.sqrt(13) * self.m * self.v0**2) / (6 * self.q * self.E)
            if not self._isclose(self.s, s_expected):
                self.logs.append(f"FAIL: s inconsistent with formula. Expected {s_expected}, got {self.s}")
                return False

            # Check v matches formula
            v_expected = (2 * math.sqrt(3) * self.v0) / 3
            if not self._isclose(self.v, v_expected):
                self.logs.append(f"FAIL: v inconsistent with formula. Expected {v_expected}, got {self.v}")
                return False

            # Check r_min formula
            r_min_expected = (math.sqrt(3) * self.l) / 3
            if not self._isclose(self.r_min, r_min_expected):
                self.logs.append(f"FAIL: r_min inconsistent with formula. Expected {r_min_expected}, got {self.r_min}")
                return False

            # Check r_max formula
            r_max_expected = (math.sqrt(3) + 1) * self.l
            if not self._isclose(self.r_max, r_max_expected):
                self.logs.append(f"FAIL: r_max inconsistent with formula. Expected {r_max_expected}, got {self.r_max}")
                return False

            # Check sin(theta)
            sintheta_expected = math.sqrt(5) / 5
            if self.sintheta and not self._isclose(self.sintheta, sintheta_expected):
                self.logs.append(f"FAIL: sin(theta) inconsistent. Expected {sintheta_expected}, got {self.sintheta}")
                return False

            # Check d_min formula
            if self.d_min:
                d_expected = (39 - 10 * math.sqrt(3)) / 44 * self.l
                if not self._isclose(self.d_min, d_expected):
                    self.logs.append(f"FAIL: d_min inconsistent. Expected {d_expected}, got {self.d_min}")
                    return False

            self.logs.append("PASS: Trace consistency verified.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {e}")
            return False

    def test_monotonicity(self):
        """Perturb an input and check the trend, then restore."""
        try:
            # Save original values
            original_m = self.m
            # Perturb mass by +10%
            self.m = original_m * 1.1

            # Recalculate s using formula (should increase)
            s_perturbed = (math.sqrt(13) * self.m * self.v0**2) / (6 * self.q * self.E)
            if s_perturbed <= self.s:
                self.logs.append(f"FAIL: s did not increase with mass. Original {self.s}, perturbed {s_perturbed}")
                self.m = original_m  # Restore
                return False

            # Recalculate B_min using formula (should increase)
            B_min_perturbed = 2 * self.m * self.v0 / ((3 + math.sqrt(3)) * self.q * self.l)
            if self.B_min and B_min_perturbed <= self.B_min:
                self.logs.append(f"FAIL: B_min did not increase with mass. Original {self.B_min}, perturbed {B_min_perturbed}")
                self.m = original_m
                return False

            # Restore original mass
            self.m = original_m

            # Perturb electric field E by +10%
            original_E = self.E
            self.E = original_E * 1.1
            s_perturbed_E = (math.sqrt(13) * self.m * self.v0**2) / (6 * self.q * self.E)
            if s_perturbed_E >= self.s:
                self.logs.append(f"FAIL: s did not decrease with E. Original {self.s}, perturbed {s_perturbed_E}")
                self.E = original_E
                return False
            self.E = original_E

            self.logs.append("PASS: Monotonicity trends verified.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {e}")
            # Attempt to restore
            self.m = original_m if 'original_m' in locals() else self.m
            self.E = original_E if 'original_E' in locals() else self.E
            return False

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        self.logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            self.logs.append(f"\n--- Running {name} Test ---")
            if not test_func():
                success = False
                self.logs.append(f"FAILED: {name} test.")
            else:
                self.logs.append(f"PASSED: {name} test.")

        return success, self.logs