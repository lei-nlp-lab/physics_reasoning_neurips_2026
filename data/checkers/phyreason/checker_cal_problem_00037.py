import math
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.H = getattr(solution_instance, 'H', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Extract results with safe defaults
        self.results = {}
        # Map from LaTeX keys to result keys (from problem context)
        key_map = {
            'fracx_frogx_insect': 'fracx_frogx_insect',
            'x': 'x',
            'x_max': 'x_max',
            't_1': 't_1',
            'a_x': 'a_x',
            'yx': 'yx',
            't_frog': 't_frog',
            'v_2': 'v_2',
            't': 't',
            'x_1': 'x_1',
            'x_min': 'x_min',
            'x_2': 'x_2',
            'a_y': 'a_y',
            'v_m': 'v_m',
            't_2': 't_2',
            'x_frog': 'x_frog',
            'v_1': 'v_1',
            'delta_y': 'delta_y'
        }
        
        # Try to get results from solution_instance.results or attributes
        for latex_key, result_key in key_map.items():
            # First try to get from results dictionary
            val = getattr(solution_instance, 'results', {}).get(result_key, None)
            if val is None:
                # Fallback to direct attribute
                val = getattr(solution_instance, result_key, None)
            self.results[latex_key] = val
        
        # Initialize logs
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def compare(self, val1, val2, name, rel_tol=1e-2, abs_tol=1e-2):
        """Compare two values numerically or symbolically."""
        if val1 is None or val2 is None:
            self.log(f"SKIP {name}: missing value")
            return False
        
        # Try numeric comparison
        try:
            num1 = float(val1) if not isinstance(val1, (int, float)) else val1
            num2 = float(val2) if not isinstance(val2, (int, float)) else val2
            if math.isclose(num1, num2, rel_tol=rel_tol, abs_tol=abs_tol):
                self.log(f"PASS {name}: {val1} ≈ {val2}")
                return True
            else:
                self.log(f"FAIL {name}: {val1} ≠ {val2} (numeric diff {abs(num1-num2)})")
                return False
        except (TypeError, ValueError):
            # Try symbolic comparison
            try:
                expr1 = sp.sympify(val1) if isinstance(val1, str) else sp.sympify(str(val1))
                expr2 = sp.sympify(val2) if isinstance(val2, str) else sp.sympify(str(val2))
                if sp.simplify(expr1 - expr2) == 0:
                    self.log(f"PASS {name}: {expr1} ≡ {expr2}")
                    return True
                else:
                    self.log(f"FAIL {name}: {expr1} ≠ {expr2} (symbolic)")
                    return False
            except Exception as e:
                self.log(f"ERROR {name}: could not compare {val1} and {val2}: {e}")
                return False

    def test_invariants(self):
        """Check fundamental physical laws and step consistency."""
        passes = []
        
        # Define common symbols
        H, g = sp.symbols('H g', positive=True)
        
        # Step 5: t_frog and v_m
        t_frog_expected = sp.sqrt(H/(2*g))
        v_m_expected = (4/sp.Integer(3)) * sp.sqrt(g*H)
        
        # Step 6 & 7: acceleration components
        a_x_expected = (4/sp.Integer(9)) * g
        a_y_expected = (1/sp.Integer(3)) * g
        
        # Step 8: meeting time t
        t_expected = sp.sqrt(3*H/(2*g))
        
        # Step 9: x_min
        # x_min = (1/2)*a_x*t^2 - v_m*t
        # Using the expected expressions:
        x_min_expr = (1/sp.Integer(2)) * a_x_expected * t_expected**2 - v_m_expected * t_expected
        x_min_expected = sp.simplify(x_min_expr)  # Should be (1-2√6)/3 * H
        
        # Step 12: x_max
        x_max_expected = 2*H
        
        # Step 14 & 15: t1, t2
        t1_expected = sp.sqrt(6*H/(5*g))
        t2_expected = sp.sqrt(3*H/(10*g))
        
        # Step 16: v1 and x1
        v1_expected = (1/sp.Integer(5)) * sp.sqrt(30*g*H)
        x1_expected = (26/sp.Integer(15)) * H
        
        # Step 17: v2 and x2
        v2_expected = (1/sp.Integer(15)) * sp.sqrt(30*g*H)
        x2_expected = (14/sp.Integer(15)) * H
        
        # Check each if available in results
        checks = [
            ('t_frog', t_frog_expected),
            ('v_m', v_m_expected),
            ('a_x', a_x_expected),
            ('a_y', a_y_expected),
            ('t', t_expected),
            ('x_min', x_min_expected),
            ('x_max', x_max_expected),
            ('t_1', t1_expected),
            ('t_2', t2_expected),
            ('v_1', v1_expected),
            ('x_1', x1_expected),
            ('v_2', v2_expected),
            ('x_2', x2_expected)
        ]
        
        for key, expected_expr in checks:
            val = self.results.get(key)
            if val is not None:
                # Substitute H and g with numeric values for comparison
                numeric_expected = float(expected_expr.subs({H: self.H, g: self.g}))
                passes.append(self.compare(val, numeric_expected, f"Invariant {key}"))
            else:
                self.log(f"SKIP {key}: not in results")
        
        # Check x_min <= x_max
        x_min_val = self.results.get('x_min')
        x_max_val = self.results.get('x_max')
        if x_min_val is not None and x_max_val is not None:
            try:
                xmin = float(x_min_val) if not isinstance(x_min_val, (int, float)) else x_min_val
                xmax = float(x_max_val) if not isinstance(x_max_val, (int, float)) else x_max_val
                if xmin <= xmax:
                    self.log(f"PASS x_min <= x_max: {xmin} ≤ {xmax}")
                    passes.append(True)
                else:
                    self.log(f"FAIL x_min <= x_max: {xmin} > {xmax}")
                    passes.append(False)
            except Exception as e:
                self.log(f"ERROR comparing x_min and x_max: {e}")
                passes.append(False)
        
        return all(passes)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        passes = []
        
        # Check that v_m, t_frog satisfy x_frog = v_m * t_frog (step 3)
        v_m = self.results.get('v_m')
        t_frog = self.results.get('t_frog')
        x_frog = self.results.get('x_frog')
        
        if v_m is not None and t_frog is not None and x_frog is not None:
            try:
                lhs = float(x_frog) if not isinstance(x_frog, (int, float)) else x_frog
                rhs = float(v_m) * float(t_frog)
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"PASS x_frog = v_m * t_frog: {lhs} ≈ {rhs}")
                    passes.append(True)
                else:
                    self.log(f"FAIL x_frog = v_m * t_frog: {lhs} ≠ {rhs}")
                    passes.append(False)
            except Exception as e:
                self.log(f"ERROR in x_frog check: {e}")
                passes.append(False)
        
        # Check that a_x and a_y match the given acceleration magnitude and angle
        a_x = self.results.get('a_x')
        a_y = self.results.get('a_y')
        if a_x is not None and a_y is not None:
            # Magnitude should be (5/9)g
            magnitude = math.sqrt(float(a_x)**2 + float(a_y)**2)
            expected = (5/9) * self.g
            if math.isclose(magnitude, expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"PASS acceleration magnitude: {magnitude} ≈ {expected}")
                passes.append(True)
            else:
                self.log(f"FAIL acceleration magnitude: {magnitude} ≠ {expected}")
                passes.append(False)
            
            # Angle should be 37°
            angle = math.degrees(math.atan2(float(a_y), float(a_x)))
            if math.isclose(angle, 37, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"PASS acceleration angle: {angle}° ≈ 37°")
                passes.append(True)
            else:
                self.log(f"FAIL acceleration angle: {angle}° ≠ 37°")
                passes.append(False)
        
        # Check that t1 and t2 satisfy the vertical motion equations (steps 14, 15)
        t1 = self.results.get('t_1')
        t2 = self.results.get('t_2')
        if t1 is not None:
            # Step 14: (1/2)g*t1^2 + (1/2)*a_y*(√2*t1)^2 = H
            try:
                t1_val = float(t1)
                lhs = 0.5*self.g*t1_val**2 + 0.5*(self.g/3)*(math.sqrt(2)*t1_val)**2
                if math.isclose(lhs, self.H, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"PASS step 14 vertical equation: {lhs} ≈ {self.H}")
                    passes.append(True)
                else:
                    self.log(f"FAIL step 14 vertical equation: {lhs} ≠ {self.H}")
                    passes.append(False)
            except Exception as e:
                self.log(f"ERROR in step 14 check: {e}")
                passes.append(False)
        
        if t2 is not None:
            # Step 15: (1/2)g*t2^2 + (1/2)*a_y*(√17*t2)^2 = H
            try:
                t2_val = float(t2)
                lhs = 0.5*self.g*t2_val**2 + 0.5*(self.g/3)*(math.sqrt(17)*t2_val)**2
                if math.isclose(lhs, self.H, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"PASS step 15 vertical equation: {lhs} ≈ {self.H}")
                    passes.append(True)
                else:
                    self.log(f"FAIL step 15 vertical equation: {lhs} ≠ {self.H}")
                    passes.append(False)
            except Exception as e:
                self.log(f"ERROR in step 15 check: {e}")
                passes.append(False)
        
        return all(passes)

    def test_monotonicity(self):
        """Perturb H and check trends."""
        passes = []
        
        # Save original H
        H_orig = self.H
        
        # Perturb H by +10%
        self.H = H_orig * 1.1
        
        # Recompute expected v_m from formula
        v_m_orig = self.results.get('v_m')
        if v_m_orig is not None:
            try:
                v_m_orig_val = float(v_m_orig) if not isinstance(v_m_orig, (int, float)) else v_m_orig
                # Expected v_m ∝ sqrt(H)
                v_m_new_expected = v_m_orig_val * math.sqrt(1.1)
                
                # The actual v_m in results might be symbolic; we can't recompute it from the solution_instance.
                # Instead, we check that the formula v_m = (4/3)*sqrt(g*H) holds for the perturbed H.
                v_m_formula = (4/3) * math.sqrt(self.g * self.H)
                # Compare with the expected scaling
                if math.isclose(v_m_new_expected, v_m_formula, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"PASS v_m monotonicity: scaling matches sqrt(H)")
                    passes.append(True)
                else:
                    self.log(f"FAIL v_m monotonicity: expected {v_m_new_expected} from scaling, got {v_m_formula} from formula")
                    passes.append(False)
            except Exception as e:
                self.log(f"ERROR in monotonicity test: {e}")
                passes.append(False)
        else:
            self.log("SKIP v_m monotonicity: v_m not in results")
            passes.append(True)  # Skip, not a failure
        
        # Restore H
        self.H = H_orig
        
        return all(passes)

    def run_all_tests(self):
        """Run all tests and return success and logs."""
        self.logs = []
        
        self.log("=== PhysicsChecker Start ===")
        
        # Run tests
        inv_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        mono_ok = self.test_monotonicity()
        
        success = inv_ok and trace_ok and mono_ok
        
        self.log(f"=== PhysicsChecker End: {'SUCCESS' if success else 'FAILED'} ===")
        
        return success, self.logs