import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.logs = []
        self.extract_inputs()
        self.extract_results()
        
    def extract_inputs(self):
        """Safely extract input parameters from solution_instance."""
        self.U = getattr(self.solution, 'U', None)
        self.d = getattr(self.solution, 'd', None)
        self.B = getattr(self.solution, 'B', None)
        self.a = getattr(self.solution, 'a', None)
        self.theta = getattr(self.solution, 'theta', None)  # in degrees
        
        # Fallback to results dictionary if attributes not directly available
        if hasattr(self.solution, 'results'):
            results = self.solution.results
            self.U = self.U if self.U is not None else results.get('U', 0.0)
            self.d = self.d if self.d is not None else results.get('d', 0.0)
            self.B = self.B if self.B is not None else results.get('B', 0.0)
            self.a = self.a if self.a is not None else results.get('a', 0.0)
            self.theta = self.theta if self.theta is not None else results.get('theta', 0.0)
            
        # Convert to float if possible for numerical checks
        try:
            self.U = float(self.U) if self.U is not None else 0.0
            self.d = float(self.d) if self.d is not None else 0.0
            self.B = float(self.B) if self.B is not None else 0.0
            self.a = float(self.a) if self.a is not None else 0.0
            self.theta = float(self.theta) if self.theta is not None else 0.0
        except (TypeError, ValueError):
            pass  # Keep as symbolic if conversion fails

    def extract_results(self):
        """Safely extract computed results from solution_instance."""
        results = getattr(self.solution, 'results', {})
        self.v = results.get('v', 0.0)
        self.fracqm = results.get('fracqm', 0.0)
        self.x = results.get('x', 0.0)
        self.y = results.get('y', 0.0)
        
        # Convert to float if possible
        try:
            self.v = float(self.v) if self.v is not None else 0.0
            self.fracqm = float(self.fracqm) if self.fracqm is not None else 0.0
            self.x = float(self.x) if self.x is not None else 0.0
            self.y = float(self.y) if self.y is not None else 0.0
        except (TypeError, ValueError):
            pass

    def log(self, message):
        """Helper to log messages."""
        self.logs.append(message)

    def test_invariants(self):
        """Test fundamental physical laws and given target formulas."""
        try:
            # Convert to sympy symbols for symbolic checks
            U, d, B, a, theta, v, fracqm = sp.symbols('U d B a theta v fracqm')
            
            # Target formula 1: v = U/(d*B)
            expr1 = sp.Eq(v, U/(d*B))
            # Substitute actual values if numerical
            if all(isinstance(val, (int, float)) for val in [self.v, self.U, self.d, self.B]):
                if not math.isclose(self.v, self.U/(self.d*self.B), rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: v = U/(d*B) not satisfied. v={self.v}, U/(d*B)={self.U/(self.d*self.B)}")
                    return False
                else:
                    self.log("PASS: v = U/(d*B) satisfied.")
            else:
                # Symbolic check: plug values into expression and simplify
                sub_dict = {v: self.v, U: self.U, d: self.d, B: self.B}
                if not expr1.subs(sub_dict).simplify():
                    self.log("FAIL: v = U/(d*B) not satisfied symbolically.")
                    return False
                else:
                    self.log("PASS: v = U/(d*B) satisfied symbolically.")
            
            # Target formula 2: fracqm = (U*tanθ)/(a*d*B^2)
            expr2 = sp.Eq(fracqm, U*sp.tan(theta*sp.pi/180) / (a*d*B**2))
            if all(isinstance(val, (int, float)) for val in [self.fracqm, self.U, self.d, self.B, self.a, self.theta]):
                tan_theta = math.tan(math.radians(self.theta))
                expected = (self.U * tan_theta) / (self.a * self.d * self.B**2)
                if not math.isclose(self.fracqm, expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: fracqm = (U*tanθ)/(a*d*B^2) not satisfied. fracqm={self.fracqm}, expected={expected}")
                    return False
                else:
                    self.log("PASS: fracqm = (U*tanθ)/(a*d*B^2) satisfied.")
            else:
                sub_dict = {fracqm: self.fracqm, U: self.U, d: self.d, B: self.B, a: self.a, theta: self.theta}
                if not expr2.subs(sub_dict).simplify():
                    self.log("FAIL: fracqm = (U*tanθ)/(a*d*B^2) not satisfied symbolically.")
                    return False
                else:
                    self.log("PASS: fracqm = (U*tanθ)/(a*d*B^2) satisfied symbolically.")
            
            # Additional invariant: force balance E = v*B (since q cancels)
            if all(isinstance(val, (int, float)) for val in [self.U, self.d, self.v, self.B]):
                E = self.U / self.d
                if not math.isclose(E, self.v * self.B, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Force balance E = v*B not satisfied. E={E}, v*B={self.v*self.B}")
                    return False
                else:
                    self.log("PASS: Force balance E = v*B satisfied.")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}")
            return False

    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        try:
            # Check if intermediate results (x, y) are consistent with v and fracqm
            # From step4: y = 0.5 * (U*q/(d*m)) * t^2, and step3: x = v*t
            # Thus: y = 0.5 * (U*fracqm/d) * (x/v)^2
            if all(isinstance(val, (int, float)) for val in [self.x, self.y, self.U, self.d, self.fracqm, self.v]) and self.v != 0:
                y_expected = 0.5 * (self.U * self.fracqm / self.d) * (self.x / self.v)**2
                if not math.isclose(self.y, y_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Trace consistency y mismatch. y={self.y}, expected={y_expected}")
                    return False
                else:
                    self.log("PASS: Trace consistency (y) satisfied.")
            
            # Check if the deflection angle θ is consistent with geometry
            # From step2: tanθ = y / (a/2)  [since the step analysis gives y = (a/2)*tanθ]
            if all(isinstance(val, (int, float)) for val in [self.y, self.a, self.theta]) and self.a != 0:
                tan_theta_expected = 2 * self.y / self.a
                tan_theta_actual = math.tan(math.radians(self.theta))
                if not math.isclose(tan_theta_actual, tan_theta_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Trace consistency tanθ mismatch. tanθ={tan_theta_actual}, expected={tan_theta_expected}")
                    return False
                else:
                    self.log("PASS: Trace consistency (tanθ) satisfied.")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}")
            return False

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends in outputs."""
        original_inputs = {'U': self.U, 'd': self.d, 'B': self.B, 'a': self.a, 'theta': self.theta}
        
        try:
            # Only perform numerical perturbation if all inputs are numerical
            if all(isinstance(val, (int, float)) for val in original_inputs.values()):
                # Test perturbation of U
                perturbation = 0.01  # 1% increase
                new_U = self.U * (1 + perturbation)
                
                # Recalculate v and fracqm using target formulas
                new_v = new_U / (self.d * self.B)
                tan_theta = math.tan(math.radians(self.theta))
                new_fracqm = (new_U * tan_theta) / (self.a * self.d * self.B**2)
                
                # Check monotonic trends: v and fracqm should increase with U
                if new_v <= self.v or new_fracqm <= self.fracqm:
                    self.log(f"FAIL: Monotonicity w.r.t U failed. v: {self.v} -> {new_v}, fracqm: {self.fracqm} -> {new_fracqm}")
                    return False
                else:
                    self.log("PASS: Monotonicity w.r.t U satisfied.")
                
                # Test perturbation of B
                new_B = self.B * (1 + perturbation)
                new_v_B = self.U / (self.d * new_B)
                new_fracqm_B = (self.U * tan_theta) / (self.a * self.d * new_B**2)
                
                # v and fracqm should decrease as B increases
                if new_v_B >= self.v or new_fracqm_B >= self.fracqm:
                    self.log(f"FAIL: Monotonicity w.r.t B failed. v: {self.v} -> {new_v_B}, fracqm: {self.fracqm} -> {new_fracqm_B}")
                    return False
                else:
                    self.log("PASS: Monotonicity w.r.t B satisfied.")
                
                return True
                
            else:
                # Symbolic monotonicity check using partial derivatives
                U, d, B, a, theta = sp.symbols('U d B a theta', positive=True)
                v_expr = U/(d*B)
                fracqm_expr = U*sp.tan(theta*sp.pi/180) / (a*d*B**2)
                
                # Derivatives of v
                dv_dU = sp.diff(v_expr, U)  # positive
                dv_dB = sp.diff(v_expr, B)  # negative
                
                # Derivatives of fracqm
                dfracqm_dU = sp.diff(fracqm_expr, U)  # positive
                dfracqm_dB = sp.diff(fracqm_expr, B)  # negative
                
                if dv_dU > 0 and dv_dB < 0 and dfracqm_dU > 0 and dfracqm_dB < 0:
                    self.log("PASS: Symbolic monotonicity checks satisfied.")
                    return True
                else:
                    self.log(f"FAIL: Symbolic monotonicity failed. dv/dU={dv_dU}, dv/dB={dv_dB}, d(fracqm)/dU={dfracqm_dU}, d(fracqm)/dB={dfracqm_dB}")
                    return False
                
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} test failed.")
            except Exception as e:
                success = False
                self.log(f"{test_name} test raised exception: {e}\n{traceback.format_exc()}")
        
        return success, self.logs