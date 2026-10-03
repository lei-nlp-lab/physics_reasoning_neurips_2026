import math
import sympy
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens from a dictionary or direct attributes
        givens = getattr(solution_instance, 'givens', {})
        self.i = givens.get('i', None)
        self.dl = givens.get('dl', None)
        self.x = givens.get('x', None)
        self.y = givens.get('y', None)
        self.z = givens.get('z', None)

        # Extract results
        results = getattr(solution_instance, 'results', {})
        self.dB_x = results.get('dB_x', None)
        # Also try direct attribute (e.g., from a solution class)
        if self.dB_x is None:
            self.dB_x = getattr(solution_instance, 'dB_x', None)

        # Explanation steps (list of strings or dicts)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        self.solution_instance = solution_instance

    # ----------------------------------------------------------------------
    # Interface checks: required keys and types
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.dB_x is None:
            return (False, "Missing required key: dB_x in results")
        return (True, "dB_x present")

    def test_interface_givens_available(self):
        missing = []
        if self.i is None: missing.append('i')
        if self.dl is None: missing.append('dl')
        if self.x is None: missing.append('x')
        if self.y is None: missing.append('y')
        if self.z is None: missing.append('z')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens present")

    def test_interface_dB_x_type(self):
        if self.dB_x is None:
            return (False, "SKIP: dB_x not available")
        if not isinstance(self.dB_x, (int, float, sympy.Expr)):
            return (False, f"dB_x type {type(self.dB_x)} not numeric or symbolic")
        return (True, "dB_x has valid type")

    # ----------------------------------------------------------------------
    # Invariant tests: fundamental physics laws
    # ----------------------------------------------------------------------
    def test_invariant_sign_vs_y(self):
        """
        Biot‑Savart: dB_x = - (i y dl) / r^3. Since i>0, dl>0, r>0,
        the sign of dB_x should be opposite to the sign of y.
        """
        if any(v is None for v in [self.i, self.dl, self.y, self.dB_x]):
            return (False, "SKIP: missing data for sign check")
        if math.isclose(self.y, 0, abs_tol=1e-12):
            if math.isclose(self.dB_x, 0, abs_tol=1e-12):
                return (True, "dB_x zero when y=0")
            else:
                return (False, f"Expected dB_x=0 when y=0, got {self.dB_x}")
        if self.dB_x * self.y <= 1e-12:   # product should be ≤ 0
            return (True, "Sign consistent: dB_x * y <= 0")
        else:
            return (False, f"Sign inconsistent: dB_x={self.dB_x}, y={self.y}, product positive")

    def test_invariant_zero_current_or_element(self):
        """
        If i=0 or dl=0, dB_x must be exactly zero (no source).
        """
        if any(v is None for v in [self.i, self.dl, self.dB_x]):
            return (False, "SKIP: missing data for zero‑source check")
        if (math.isclose(self.i, 0) or math.isclose(self.dl, 0)):
            if not math.isclose(self.dB_x, 0, abs_tol=1e-12):
                return (False, f"dB_x should be zero when i=0 or dl=0, got {self.dB_x}")
        return (True, "Zero‑source invariant passed")

    def test_invariant_denominator_structure(self):
        """
        dB_x must be proportional to (x^2+y^2+z^2)^(-3/2).
        For a symbolic expression, check structural equality.
        For numeric values, compute the expected value from givens.
        """
        if self.dB_x is None:
            return (False, "SKIP: no dB_x")
        # ----- Symbolic case -----
        if isinstance(self.dB_x, sympy.Expr):
            x, y, z, i, dl = sympy.symbols('x y z i dl')
            expected = -i*y*dl / (x**2 + y**2 + z**2)**(sympy.Rational(3,2))
            diff = sympy.simplify(self.dB_x - expected)
            if diff == 0:
                return (True, "Symbolic expression matches Biot–Savart form")
            else:
                # Allow for a constant factor (the problem assumes mu0/(4π)=1)
                # Check if the ratio is a numeric constant close to 1
                ratio = sympy.simplify(self.dB_x / expected)
                if ratio.is_Number and math.isclose(float(ratio), 1.0, rel_tol=1e-2):
                    return (True, f"Symbolic expression differs by constant factor {ratio}")
                return (False, f"Symbolic expression differs from expected. Simplified difference: {diff}")
        # ----- Numeric case -----
        if any(v is None for v in [self.i, self.dl, self.x, self.y, self.z]):
            return (False, "SKIP: missing numeric data for denominator check")
        r_sq = self.x**2 + self.y**2 + self.z**2
        if math.isclose(r_sq, 0):
            return (False, "r=0 (field point at source) not allowed")
        expected = -self.i * self.y * self.dl / (r_sq ** 1.5)
        if math.isclose(self.dB_x, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Numeric check matches expected {expected}")
        else:
            return (False, f"Numeric mismatch: got {self.dB_x}, expected {expected} (i={self.i}, dl={self.dl}, x={self.x}, y={self.y}, z={self.z})")

    # ----------------------------------------------------------------------
    # Modeling checks: explanation steps should mention the essential physics
    # ----------------------------------------------------------------------
    def test_modeling_biot_savart_mentioned(self):
        if not self.explanation_steps:
            return (False, "SKIP: no explanation steps")
        text = ' '.join(self.explanation_steps).lower()
        if 'biot' in text or 'savart' in text:
            return (True, "Explanation mentions Biot‑Savart law")
        else:
            return (False, "Explanation does not mention Biot‑Savart law")

    def test_modeling_cross_product_mentioned(self):
        if not self.explanation_steps:
            return (False, "SKIP: no explanation steps")
        text = ' '.join(self.explanation_steps).lower()
        if 'cross product' in text or 'cross' in text:
            return (True, "Explanation mentions cross product")
        else:
            return (False, "Explanation does not mention cross product")

    # ----------------------------------------------------------------------
    # Monotonicity test: perturb y and check dB_x becomes more negative
    # ----------------------------------------------------------------------
    def test_monotonicity_dB_x_with_y(self):
        """
        Check that dB_x is a decreasing function of y (for positive i, dl).
        Perturb y by +10% and verify dB_x becomes more negative.
        Use numeric evaluation of the symbolic dB_x if possible, else SKIP.
        """
        if self.dB_x is None:
            return (False, "SKIP: dB_x not available")
        if not isinstance(self.dB_x, sympy.Expr):
            return (False, "SKIP: dB_x is numeric, cannot perturb (expression needed)")
        # Define symbols used in expression
        syms = self.dB_x.free_symbols
        # Build baseline numeric values, preferring givens and filling defaults where missing
        baseline_vals = {}
        # Helper to get numeric value
        def get_val(given, default):
            if given is not None:
                return float(given) if not isinstance(given, sympy.Basic) else default
            else:
                return default
        baseline_vals[sympy.Symbol('i')] = get_val(self.i, 1.0)
        baseline_vals[sympy.Symbol('y')] = get_val(self.y, 1.0)
        baseline_vals[sympy.Symbol('dl')] = get_val(self.dl, 0.1)
        baseline_vals[sympy.Symbol('x')] = get_val(self.x, 0.0)
        baseline_vals[sympy.Symbol('z')] = get_val(self.z, 1.0)
        # Cover any remaining free symbols with 1.0
        for sym in syms:
            if sym not in baseline_vals:
                baseline_vals[sym] = 1.0
        # Ensure y is not zero to avoid sign ambiguity
        if math.isclose(baseline_vals[sympy.Symbol('y')], 0, abs_tol=1e-9):
            baseline_vals[sympy.Symbol('y')] = 1.0
        # Compute baseline value
        try:
            baseline_val = float(self.dB_x.subs(baseline_vals))
        except Exception as e:
            return (False, f"SKIP: cannot evaluate baseline: {e}")
        # Perturb y by +10%
        perturbed_vals = baseline_vals.copy()
        perturbed_vals[sympy.Symbol('y')] *= 1.1
        try:
            perturbed_val = float(self.dB_x.subs(perturbed_vals))
        except Exception as e:
            return (False, f"SKIP: cannot evaluate perturbed: {e}")
        # Expect: because dB_x ∝ -y, increasing y (positive) should make dB_x more negative.
        # So perturbed_val - baseline_val should be negative.
        diff = perturbed_val - baseline_val
        if diff < 0:
            return (True, f"Perturbation y+10%: dB_x decreased by {abs(diff):.4f} (from {baseline_val:.4f} to {perturbed_val:.4f})")
        else:
            return (False, f"Perturbation y+10%: dB_x increased or unchanged (diff={diff:.4f}). baseline={baseline_val:.4f}, perturbed={perturbed_val:.4f}")

    # ----------------------------------------------------------------------
    # Weak answer‑sanity check (only one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_not_nan(self):
        if self.dB_x is None:
            return (False, "SKIP: no dB_x")
        if isinstance(self.dB_x, (int, float)):
            if not math.isfinite(self.dB_x) or math.isnan(self.dB_x):
                return (False, f"dB_x is not finite or is NaN: {self.dB_x}")
            return (True, "dB_x is finite and not NaN")
        else:
            # symbolic: accept as long as it exists
            return (True, "dB_x is symbolic (assumed finite)")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
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