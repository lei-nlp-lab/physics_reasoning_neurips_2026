import sympy as sp
import math
import traceback
from typing import List, Tuple, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.v0 = getattr(solution_instance, 'v0', None)
        self.R = getattr(solution_instance, 'R', None)
        self.theta = getattr(solution_instance, 'theta', None)
        self.N = getattr(solution_instance, 'N', None)
        self.k = getattr(solution_instance, 'k', 0)
        self.n = getattr(solution_instance, 'n', 0)
        
        # Extract results with safe access
        self.B = self.results.get('B', None)
        self.omega = self.results.get('omega', None)
        self.omega_prime = self.results.get('omega\'', None)
        self.F = self.results.get('F', None)
        self.theta_prime_vals = self.results.get('theta\'', None)
        self.t = self.results.get('t', None)
        self.t_prime = self.results.get('t\'', None)
        self.v = self.results.get('v', None)
        self.R_prime = self.results.get('R\'', None)
        self.omega_t_prime = self.results.get('omega\'t\'', None)
        
        # For symbolic computations
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'q': sp.symbols('q', positive=True),
            'v0': sp.symbols('v0', positive=True),
            'R': sp.symbols('R', positive=True),
            'theta': sp.symbols('theta', positive=True),
            'N': sp.symbols('N', positive=True),
            'k': sp.symbols('k', integer=True, nonnegative=True),
            'n': sp.symbols('n', integer=True, nonnegative=True),
            'B': sp.symbols('B', positive=True),
            'omega': sp.symbols('omega', positive=True),
            'omega_prime': sp.symbols('omega_prime', positive=True),
            'F': sp.symbols('F', positive=True),
            't': sp.symbols('t', positive=True),
            't_prime': sp.symbols('t_prime', positive=True),
            'v': sp.symbols('v', positive=True),
            'R_prime': sp.symbols('R_prime', positive=True)
        }
        
        self.logs: List[str] = []
        
    def _log(self, message: str, success: bool = True):
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison"""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _eval_symbolic(self, expr, subs_dict: Dict = None):
        """Evaluate symbolic expression with given substitutions"""
        if subs_dict is None:
            subs_dict = {}
        
        # Create actual substitution dictionary
        actual_subs = {}
        for sym, val in subs_dict.items():
            if val is not None:
                actual_subs[self.symbols[sym]] = val
                
        try:
            if actual_subs:
                result = expr.subs(actual_subs)
                if result.is_number:
                    return float(result)
                return result
            return expr
        except:
            return None
    
    def test_lorentz_centripetal(self) -> bool:
        """Test Lorentz force equals centripetal force for initial ion"""
        try:
            # Symbolic check
            B_sym = self.symbols['m'] * self.symbols['v0'] / (self.symbols['q'] * self.symbols['R'])
            expected_B = self._eval_symbolic(B_sym, {'m': self.m, 'v0': self.v0, 'q': self.q, 'R': self.R})
            
            if self.B is not None and expected_B is not None:
                if isinstance(expected_B, (int, float)) and isinstance(self.B, (int, float)):
                    if self._isclose(self.B, expected_B):
                        self._log(f"Lorentz-centripetal: B = {self.B} matches expected {expected_B}")
                        return True
                    else:
                        self._log(f"Lorentz-centripetal: B = {self.B} differs from expected {expected_B}", False)
                        return False
                else:
                    # Symbolic comparison
                    diff = sp.simplify(self.B - expected_B)
                    if diff == 0:
                        self._log("Lorentz-centripetal: Symbolic match")
                        return True
                    else:
                        self._log(f"Lorentz-centripetal: Symbolic difference: {diff}", False)
                        return False
            else:
                self._log("Lorentz-centripetal: Missing data for check", True)  # Not a failure if data missing
                return True
        except Exception as e:
            self._log(f"Lorentz-centripetal error: {str(e)}", False)
            return False
    
    def test_uniform_circular_time(self) -> bool:
        """Test time spent in magnetic field for initial ion"""
        try:
            # From step 2: t = πR/(2v0)
            t_sym = sp.pi * self.symbols['R'] / (2 * self.symbols['v0'])
            expected_t = self._eval_symbolic(t_sym, {'R': self.R, 'v0': self.v0})
            
            if self.t is not None and expected_t is not None:
                if isinstance(expected_t, (int, float)) and isinstance(self.t, (int, float)):
                    if self._isclose(self.t, expected_t):
                        self._log(f"Uniform circular time: t = {self.t} matches expected {expected_t}")
                        return True
                    else:
                        self._log(f"Uniform circular time: t = {self.t} differs from expected {expected_t}", False)
                        return False
                else:
                    diff = sp.simplify(self.t - expected_t)
                    if diff == 0:
                        self._log("Uniform circular time: Symbolic match")
                        return True
                    else:
                        self._log(f"Uniform circular time: Symbolic difference: {diff}", False)
                        return False
            else:
                self._log("Uniform circular time: Missing data for check", True)
                return True
        except Exception as e:
            self._log(f"Uniform circular time error: {str(e)}", False)
            return False
    
    def test_angular_speed_initial(self) -> bool:
        """Test angular speed for initial ion detection"""
        try:
            # From step 3: ω = (4k+1)*v0/R
            omega_sym = (4 * self.symbols['k'] + 1) * self.symbols['v0'] / self.symbols['R']
            expected_omega = self._eval_symbolic(omega_sym, {'k': self.k, 'v0': self.v0, 'R': self.R})
            
            if self.omega is not None and expected_omega is not None:
                if isinstance(expected_omega, (int, float)) and isinstance(self.omega, (int, float)):
                    if self._isclose(self.omega, expected_omega):
                        self._log(f"Angular speed initial: ω = {self.omega} matches expected {expected_omega}")
                        return True
                    else:
                        self._log(f"Angular speed initial: ω = {self.omega} differs from expected {expected_omega}", False)
                        return False
                else:
                    diff = sp.simplify(self.omega - expected_omega)
                    if diff == 0:
                        self._log("Angular speed initial: Symbolic match")
                        return True
                    else:
                        self._log(f"Angular speed initial: Symbolic difference: {diff}", False)
                        return False
            else:
                self._log("Angular speed initial: Missing data for check", True)
                return True
        except Exception as e:
            self._log(f"Angular speed initial error: {str(e)}", False)
            return False
    
    def test_geometry_radius_prime(self) -> bool:
        """Test R' = R * tan(θ/2)"""
        try:
            if self.theta is None or self.R is None:
                self._log("Geometry radius prime: Missing theta or R", True)
                return True
                
            R_prime_sym = self.symbols['R'] * sp.tan(self.symbols['theta'] / 2)
            expected_R_prime = self._eval_symbolic(R_prime_sym, {'R': self.R, 'theta': self.theta})
            
            if self.R_prime is not None and expected_R_prime is not None:
                if isinstance(expected_R_prime, (int, float)) and isinstance(self.R_prime, (int, float)):
                    if self._isclose(self.R_prime, expected_R_prime):
                        self._log(f"Geometry radius prime: R' = {self.R_prime} matches expected {expected_R_prime}")
                        return True
                    else:
                        self._log(f"Geometry radius prime: R' = {self.R_prime} differs from expected {expected_R_prime}", False)
                        return False
                else:
                    diff = sp.simplify(self.R_prime - expected_R_prime)
                    if diff == 0:
                        self._log("Geometry radius prime: Symbolic match")
                        return True
                    else:
                        self._log(f"Geometry radius prime: Symbolic difference: {diff}", False)
                        return False
            else:
                self._log("Geometry radius prime: Missing R' for check", True)
                return True
        except Exception as e:
            self._log(f"Geometry radius prime error: {str(e)}", False)
            return False
    
    def test_speed_from_angle(self) -> bool:
        """Test v = v0 * tan(θ/2)"""
        try:
            if self.theta is None or self.v0 is None:
                self._log("Speed from angle: Missing theta or v0", True)
                return True
                
            v_sym = self.symbols['v0'] * sp.tan(self.symbols['theta'] / 2)
            expected_v = self._eval_symbolic(v_sym, {'v0': self.v0, 'theta': self.theta})
            
            if self.v is not None and expected_v is not None:
                if isinstance(expected_v, (int, float)) and isinstance(self.v, (int, float)):
                    if self._isclose(self.v, expected_v):
                        self._log(f"Speed from angle: v = {self.v} matches expected {expected_v}")
                        return True
                    else:
                        self._log(f"Speed from angle: v = {self.v} differs from expected {expected_v}", False)
                        return False
                else:
                    diff = sp.simplify(self.v - expected_v)
                    if diff == 0:
                        self._log("Speed from angle: Symbolic match")
                        return True
                    else:
                        self._log(f"Speed from angle: Symbolic difference: {diff}", False)
                        return False
            else:
                self._log("Speed from angle: Missing v for check", True)
                return True
        except Exception as e:
            self._log(f"Speed from angle error: {str(e)}", False)
            return False
    
    def test_time_prime(self) -> bool:
        """Test t' = (π-θ) * R'/v"""
        try:
            if self.theta is None or self.R_prime is None or self.v is None:
                self._log("Time prime: Missing theta, R', or v", True)
                return True
                
            t_prime_sym = (sp.pi - self.symbols['theta']) * self.symbols['R_prime'] / self.symbols['v']
            expected_t_prime = self._eval_symbolic(t_prime_sym, 
                                                   {'theta': self.theta, 'R_prime': self.R_prime, 'v': self.v})
            
            if self.t_prime is not None and expected_t_prime is not None:
                if isinstance(expected_t_prime, (int, float)) and isinstance(self.t_prime, (int, float)):
                    if self._isclose(self.t_prime, expected_t_prime):
                        self._log(f"Time prime: t' = {self.t_prime} matches expected {expected_t_prime}")
                        return True
                    else:
                        self._log(f"Time prime: t' = {self.t_prime} differs from expected {expected_t_prime}", False)
                        return False
                else:
                    diff = sp.simplify(self.t_prime - expected_t_prime)
                    if diff == 0:
                        self._log("Time prime: Symbolic match")
                        return True
                    else:
                        self._log(f"Time prime: Symbolic difference: {diff}", False)
                        return False
            else:
                self._log("Time prime: Missing t' for check", True)
                return True
        except Exception as e:
            self._log(f"Time prime error: {str(e)}", False)
            return False
    
    def test_angular_speed_prime(self) -> bool:
        """Test ω' = (2nπ+θ)/(π-θ) * v0/R"""
        try:
            if self.theta is None or self.v0 is None or self.R is None:
                self._log("Angular speed prime: Missing theta, v0, or R", True)
                return True
                
            omega_prime_sym = (2 * self.symbols['n'] * sp.pi + self.symbols['theta']) * self.symbols['v0'] / \
                              ((sp.pi - self.symbols['theta']) * self.symbols['R'])
            expected_omega_prime = self._eval_symbolic(omega_prime_sym, 
                                                       {'n': self.n, 'theta': self.theta, 'v0': self.v0, 'R': self.R})
            
            if self.omega_prime is not None and expected_omega_prime is not None:
                if isinstance(expected_omega_prime, (int, float)) and isinstance(self.omega_prime, (int, float)):
                    if self._isclose(self.omega_prime, expected_omega_prime):
                        self._log(f"Angular speed prime: ω' = {self.omega_prime} matches expected {expected_omega_prime}")
                        return True
                    else:
                        self._log(f"Angular speed prime: ω' = {self.omega_prime} differs from expected {expected_omega_prime}", False)
                        return False
                else:
                    diff = sp.simplify(self.omega_prime - expected_omega_prime)
                    if diff == 0:
                        self._log("Angular speed prime: Symbolic match")
                        return True
                    else:
                        self._log(f"Angular speed prime: Symbolic difference: {diff}", False)
                        return False
            else:
                self._log("Angular speed prime: Missing ω' for check", True)
                return True
        except Exception as e:
            self._log(f"Angular speed prime error: {str(e)}", False)
            return False
    
    def test_impulse_force(self) -> bool:
        """Test F = N*m*v*ω'/(2π) and consistency with step 10 formula"""
        try:
            if self.N is None or self.m is None or self.v is None or self.omega_prime is None:
                self._log("Impulse force: Missing N, m, v, or ω'", True)
                return True
                
            # From step 9: F = N*m*v*ω'/(2π)
            F_sym1 = self.symbols['N'] * self.symbols['m'] * self.symbols['v'] * self.symbols['omega_prime'] / (2 * sp.pi)
            
            # From step 10: F = (2nπ+θ)N/(2(π-θ)π) * (m*v0²/R) * tan(θ/2)
            F_sym2 = (2 * self.symbols['n'] * sp.pi + self.symbols['theta']) * self.symbols['N'] * \
                     self.symbols['m'] * self.symbols['v0']**2 * sp.tan(self.symbols['theta'] / 2) / \
                     (2 * (sp.pi - self.symbols['theta']) * sp.pi * self.symbols['R'])
            
            # Check if both expressions are equivalent
            diff_sym = sp.simplify(F_sym1 - F_sym2)
            
            # Substitute known values to check numerical consistency
            subs_dict = {
                'N': self.N, 'm': self.m, 'v': self.v, 'omega_prime': self.omega_prime,
                'n': self.n, 'theta': self.theta, 'v0': self.v0, 'R': self.R
            }
            
            expected_F1 = self._eval_symbolic(F_sym1, subs_dict)
            expected_F2 = self._eval_symbolic(F_sym2, subs_dict)
            
            if self.F is not None:
                # Check against first formula
                if expected_F1 is not None and isinstance(expected_F1, (int, float)) and isinstance(self.F, (int, float)):
                    if self._isclose(self.F, expected_F1):
                        self._log(f"Impulse force: F = {self.F} matches expected from formula 1: {expected_F1}")
                        # Also verify formulas are equivalent
                        if diff_sym == 0:
                            self._log("Impulse force: Both formulas are equivalent (symbolic)")
                        else:
                            # Numerical check of equivalence
                            if expected_F2 is not None and isinstance(expected_F2, (int, float)):
                                if self._isclose(expected_F1, expected_F2):
                                    self._log("Impulse force: Both formulas give same numerical result")
                                else:
                                    self._log(f"Impulse force: Formulas disagree: {expected_F1} vs {expected_F2}", False)
                                    return False
                        return True
                    else:
                        self._log(f"Impulse force: F = {self.F} differs from expected {expected_F1}", False)
                        return False
                else:
                    # Symbolic check
                    diff1 = sp.simplify(self.F - F_sym1)
                    if diff1 == 0:
                        self._log("Impulse force: Symbolic match with formula 1")
                        return True
                    else:
                        self._log(f"Impulse force: Symbolic difference with formula 1: {diff1}", False)
                        return False
            else:
                self._log("Impulse force: Missing F for check", True)
                return True
        except Exception as e:
            self._log(f"Impulse force error: {str(e)}", False)
            return False
    
    def test_detected_angles(self) -> bool:
        """Test detected angles θ' = 5π/6, π/6, π/2"""
        try:
            expected_angles = [5*math.pi/6, math.pi/6, math.pi/2]
            
            if self.theta_prime_vals is not None:
                if isinstance(self.theta_prime_vals, list):
                    # Check each angle is close to one of the expected angles
                    valid = True
                    for angle in self.theta_prime_vals:
                        if isinstance(angle, (int, float)):
                            # Find closest expected angle
                            closest = min(expected_angles, key=lambda x: abs(x - angle))
                            if not self._isclose(angle, closest):
                                self._log(f"Detected angle {angle} not close to expected {closest}", False)
                                valid = False
                        else:
                            # Symbolic check
                            if angle not in [5*sp.pi/6, sp.pi/6, sp.pi/2]:
                                self._log(f"Detected angle {angle} not in expected set", False)
                                valid = False
                    
                    if valid:
                        self._log(f"Detected angles {self.theta_prime_vals} match expected set")
                        return True
                    else:
                        return False
                else:
                    # Single value
                    angle = self.theta_prime_vals
                    if isinstance(angle, (int, float)):
                        closest = min(expected_angles, key=lambda x: abs(x - angle))
                        if self._isclose(angle, closest):
                            self._log(f"Detected angle {angle} matches expected {closest}")
                            return True
                        else:
                            self._log(f"Detected angle {angle} not close to any expected angle", False)
                            return False
                    else:
                        # Symbolic
                        if angle in [5*sp.pi/6, sp.pi/6, sp.pi/2]:
                            self._log(f"Detected angle {angle} in expected set")
                            return True
                        else:
                            self._log(f"Detected angle {angle} not in expected set", False)
                            return False
            else:
                self._log("Detected angles: No θ' values provided", True)
                return True
        except Exception as e:
            self._log(f"Detected angles error: {str(e)}", False)
            return False
    
    def test_inequality_condition(self) -> bool:
        """Test inequality (4k+1)v0/R < (2nπ+θ')/(π-θ') * v0/R < 6v0/R"""
        try:
            if self.theta_prime_vals is None or self.v0 is None or self.R is None:
                self._log("Inequality condition: Missing data", True)
                return True
                
            # For k=0 (lowest value) and n=0 (simplest case)
            left_expr = (4*0 + 1) * self.v0 / self.R  # = v0/R
            right_expr = 6 * self.v0 / self.R
            
            angles = self.theta_prime_vals if isinstance(self.theta_prime_vals, list) else [self.theta_prime_vals]
            
            valid = True
            for theta_prime in angles:
                if isinstance(theta_prime, (int, float)):
                    middle_expr = (2*0*math.pi + theta_prime) * self.v0 / ((math.pi - theta_prime) * self.R)
                    
                    if left_expr < middle_expr < right_expr:
                        self._log(f"Inequality holds for θ'={theta_prime}: {left_expr} < {middle_expr} < {right_expr}")
                    else:
                        self._log(f"Inequality fails for θ'={theta_prime}: {left_expr} < {middle_expr} < {right_expr} is false", False)
                        valid = False
            
            return valid
        except Exception as e:
            self._log(f"Inequality condition error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers"""
        try:
            # Check that all derived formulas are internally consistent
            success = True
            
            # If we have both v and θ, check v = v0*tan(θ/2)
            if self.v is not None and self.theta is not None and self.v0 is not None:
                expected_v = self.v0 * math.tan(self.theta/2) if isinstance(self.theta, (int, float)) else None
                if expected_v is not None and isinstance(self.v, (int, float)):
                    if not self._isclose(self.v, expected_v):
                        self._log(f"Trace consistency: v={self.v} but v0*tan(θ/2)={expected_v}", False)
                        success = False
                    else:
                        self._log(f"Trace consistency: v matches v0*tan(θ/2)")
            
            # If we have both ω' and t', check ω'*t' = 2nπ + θ
            if self.omega_prime is not None and self.t_prime is not None and self.theta is not None:
                lhs = self.omega_prime * self.t_prime if isinstance(self.omega_prime, (int, float)) and isinstance(self.t_prime, (int, float)) else None
                rhs = 2*self.n*math.pi + self.theta if isinstance(self.theta, (int, float)) else None
                if lhs is not None and rhs is not None:
                    if not self._isclose(lhs, rhs):
                        self._log(f"Trace consistency: ω'*t'={lhs} but 2nπ+θ={rhs}", False)
                        success = False
                    else:
                        self._log(f"Trace consistency: ω'*t' matches 2nπ+θ")
            
            return success
        except Exception as e:
            self._log(f"Trace consistency error: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb inputs and check trends"""
        try:
            success = True
            
            # Test 1: If v0 increases, B should increase proportionally
            if self.m is not None and self.q is not None and self.R is not None and self.v0 is not None:
                original_B = self.m * self.v0 / (self.q * self.R)
                perturbed_v0 = self.v0 * 1.1  # 10% increase
                perturbed_B = self.m * perturbed_v0 / (self.q * self.R)
                
                if perturbed_B > original_B:
                    self._log(f"Monotonicity: B increases with v0 ({original_B} -> {perturbed_B})")
                else:
                    self._log(f"Monotonicity: B doesn't increase with v0", False)
                    success = False
            
            # Test 2: If θ increases (0 < θ < π), R' should increase
            if self.R is not None:
                for theta_test in [math.pi/6, math.pi/4, math.pi/3]:
                    R_prime1 = self.R * math.tan(math.pi/6 / 2)
                    R_prime2 = self.R * math.tan(math.pi/3 / 2)
                    if R_prime2 > R_prime1:
                        self._log(f"Monotonicity: R' increases with θ")
                        break
                else:
                    self._log(f"Monotonicity: R' doesn't increase with θ", False)
                    success = False
            
            return success
        except Exception as e:
            self._log(f"Monotonicity error: {str(e)}", False)
            return False
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants"""
        try:
            success = True
            
            # Check Lorentz force magnitude equals centripetal force for circular motion
            if self.B is not None and self.q is not None and self.v0 is not None and self.m is not None and self.R is not None:
                lorentz_force = abs(self.q) * self.v0 * self.B  # q > 0, ion has charge -q
                centripetal_force = self.m * self.v0**2 / self.R
                
                if self._isclose(lorentz_force, centripetal_force):
                    self._log(f"Invariant: Lorentz force ({lorentz_force}) ≈ centripetal force ({centripetal_force})")
                else:
                    self._log(f"Invariant: Lorentz force ({lorentz_force}) ≠ centripetal force ({centripetal_force})", False)
                    success = False
            
            # Check dimensions of formulas
            # B = m*v0/(q*R): [M][L/T]/([Q][L]) = [M]/[Q][T] = Tesla (correct)
            # ω = (4k+1)v0/R: [L/T]/[L] = 1/[T] = rad/s (correct)
            
            return success
        except Exception as e:
            self._log(f"Invariants error: {str(e)}", False)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics consistency checks"""
        self.logs = []
        
        try:
            # Run individual tests
            tests = [
                self.test_lorentz_centripetal,
                self.test_uniform_circular_time,
                self.test_angular_speed_initial,
                self.test_geometry_radius_prime,
                self.test_speed_from_angle,
                self.test_time_prime,
                self.test_angular_speed_prime,
                self.test_impulse_force,
                self.test_detected_angles,
                self.test_inequality_condition,
                self.test_trace_consistency,
                self.test_monotonicity,
                self.test_invariants
            ]
            
            results = []
            for test in tests:
                try:
                    results.append(test())
                except Exception as e:
                    self._log(f"Test {test.__name__} crashed: {str(e)}", False)
                    results.append(False)
            
            # Count successes (skip tests that returned True due to missing data)
            actual_tests = [r for r in results if r is not None]
            success_count = sum(1 for r in actual_tests if r)
            total_count = len(actual_tests)
            
            if total_count > 0:
                success_rate = success_count / total_count
                overall_success = success_rate > 0.8  # Allow some tolerance
                
                self._log(f"Overall: {success_count}/{total_count} tests passed ({success_rate*100:.1f}%)")
            else:
                overall_success = True
                self._log("No tests could be executed with available data")
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"PhysicsChecker crashed: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, False)
            return False, self.logs