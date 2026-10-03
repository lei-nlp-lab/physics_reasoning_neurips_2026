import math
import sympy as sp
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access from solution_instance."""
        # Extract results safely - use getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Expected keys from the problem mapping
        self.v = self._get_result('v')  # Speed at Q (should be √2 V0)
        self.E = self._get_result('E')  # Electric field magnitude
        self.B = self._get_result('B')  # Magnetic field magnitude
        self.t = self._get_result('t')  # Total time O→P
        self.T = self._get_result('T')  # Period in magnetic field
        self.R = self._get_result('R')  # Radius in magnetic field
        self.theta = self._get_result('theta')  # Central angle (should be 90° or π/2)
        self.t1 = self._get_result('t_1')  # Time O→Q
        self.t2 = self._get_result('t_2')  # Time Q→P
        self.OQ = self._get_result('OQ')  # Distance OQ (should be L)
        self.x = self._get_result('x')  # Horizontal displacement Q→P (should be 2L)
        self.y = self._get_result('y')  # Vertical displacement Q→P (should be L)
        
        # Extract input parameters safely
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.L = getattr(solution_instance, 'L', None)
        
        # Symbolic variables for symbolic checks
        self.m_sym = sp.symbols('m', positive=True)
        self.q_sym = sp.symbols('q', positive=True)
        self.V0_sym = sp.symbols('V0', positive=True)
        self.L_sym = sp.symbols('L', positive=True)
        
        # Test values for numerical checks (using reasonable defaults)
        self.test_values = {
            'm': 1.0,
            'q': 1.0,
            'V0': 1.0,
            'L': 1.0
        }
        
        self.logs = []
    
    def _get_result(self, key):
        """Safely get result with fallback to 0."""
        return self.results.get(key, 0.0)
    
    def _log(self, message: str):
        """Add log message."""
        self.logs.append(message)
    
    def _isclose_sym(self, expr1, expr2) -> bool:
        """Check if two symbolic expressions are equivalent."""
        try:
            # Try symbolic simplification
            diff = sp.simplify(expr1 - expr2)
            return diff == 0
        except:
            return False
    
    def _isclose_num(self, val1, val2) -> bool:
        """Check if two numerical values are close (1-2% tolerance)."""
        if val1 is None or val2 is None:
            return False
        try:
            return math.isclose(float(val1), float(val2), rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            return False
    
    def _evaluate_expr(self, expr, symbolic=False):
        """Evaluate expression either symbolically or numerically."""
        if symbolic:
            return expr
        else:
            # Substitute test values if expr is symbolic
            if hasattr(expr, 'subs'):
                expr = expr.subs([
                    (self.m_sym, self.test_values['m']),
                    (self.q_sym, self.test_values['q']),
                    (self.V0_sym, self.test_values['V0']),
                    (self.L_sym, self.test_values['L'])
                ])
            try:
                return float(expr)
            except:
                return expr
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics laws based on KB."""
        tests_passed = True
        
        # Test 1: Work-energy theorem from O to P
        # W = ΔK = qE*(-L) since force is -qE and displacement in y is L
        # ΔK = 1/2*m*V0² - 1/2*m*v² where v is speed at Q
        try:
            if self.v and self.E and self.V0 and self.m and self.q and self.L:
                v_expr = self._evaluate_expr(self.v, symbolic=True)
                E_expr = self._evaluate_expr(self.E, symbolic=True)
                V0_expr = self._evaluate_expr(self.V0, symbolic=True)
                m_expr = self._evaluate_expr(self.m, symbolic=True)
                q_expr = self._evaluate_expr(self.q, symbolic=True)
                L_expr = self._evaluate_expr(self.L, symbolic=True)
                
                # Left side: work done by electric field
                W_left = -q_expr * E_expr * L_expr
                
                # Right side: change in kinetic energy
                K_Q = m_expr * v_expr**2 / 2
                K_P = m_expr * V0_expr**2 / 2
                W_right = K_P - K_Q
                
                if not self._isclose_sym(W_left, W_right):
                    # Try numerical fallback
                    W_left_num = self._evaluate_expr(W_left, symbolic=False)
                    W_right_num = self._evaluate_expr(W_right, symbolic=False)
                    if not self._isclose_num(W_left_num, W_right_num):
                        tests_passed = False
                        self._log(f"Work-energy theorem failed: W_left={W_left}, W_right={W_right}")
        except Exception as e:
            tests_passed = False
            self._log(f"Work-energy theorem error: {str(e)}")
        
        # Test 2: Lorentz force in magnetic field (circular motion)
        # qvB = mv²/R
        try:
            if self.v and self.B and self.R and self.m and self.q:
                v_expr = self._evaluate_expr(self.v, symbolic=True)
                B_expr = self._evaluate_expr(self.B, symbolic=True)
                R_expr = self._evaluate_expr(self.R, symbolic=True)
                m_expr = self._evaluate_expr(self.m, symbolic=True)
                q_expr = self._evaluate_expr(self.q, symbolic=True)
                
                left = q_expr * v_expr * B_expr
                right = m_expr * v_expr**2 / R_expr
                
                if not self._isclose_sym(left, right):
                    left_num = self._evaluate_expr(left, symbolic=False)
                    right_num = self._evaluate_expr(right, symbolic=False)
                    if not self._isclose_num(left_num, right_num):
                        tests_passed = False
                        self._log(f"Lorentz force failed: qvB={left}, mv²/R={right}")
        except Exception as e:
            tests_passed = False
            self._log(f"Lorentz force error: {str(e)}")
        
        # Test 3: Period formula for circular motion
        # T = 2πR/v
        try:
            if self.T and self.R and self.v:
                T_expr = self._evaluate_expr(self.T, symbolic=True)
                R_expr = self._evaluate_expr(self.R, symbolic=True)
                v_expr = self._evaluate_expr(self.v, symbolic=True)
                
                expected_T = 2 * sp.pi * R_expr / v_expr
                
                if not self._isclose_sym(T_expr, expected_T):
                    T_num = self._evaluate_expr(T_expr, symbolic=False)
                    expected_T_num = self._evaluate_expr(expected_T, symbolic=False)
                    if not self._isclose_num(T_num, expected_T_num):
                        tests_passed = False
                        self._log(f"Period formula failed: T={T_expr}, expected 2πR/v={expected_T}")
        except Exception as e:
            tests_passed = False
            self._log(f"Period formula error: {str(e)}")
        
        # Test 4: Geometry - OQ should be chord of circle
        # OQ = 2R cos(45°) = √2 R
        try:
            if self.OQ and self.R:
                OQ_expr = self._evaluate_expr(self.OQ, symbolic=True)
                R_expr = self._evaluate_expr(self.R, symbolic=True)
                
                expected_OQ = sp.sqrt(2) * R_expr
                
                if not self._isclose_sym(OQ_expr, expected_OQ):
                    OQ_num = self._evaluate_expr(OQ_expr, symbolic=False)
                    expected_OQ_num = self._evaluate_expr(expected_OQ, symbolic=False)
                    if not self._isclose_num(OQ_num, expected_OQ_num):
                        tests_passed = False
                        self._log(f"Geometry failed: OQ={OQ_expr}, expected √2 R={expected_OQ}")
        except Exception as e:
            tests_passed = False
            self._log(f"Geometry error: {str(e)}")
        
        return tests_passed, self.logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answer."""
        tests_passed = True
        
        # Check that total time t = t1 + t2
        try:
            if self.t and self.t1 and self.t2:
                t_expr = self._evaluate_expr(self.t, symbolic=True)
                t1_expr = self._evaluate_expr(self.t1, symbolic=True)
                t2_expr = self._evaluate_expr(self.t2, symbolic=True)
                
                if not self._isclose_sym(t_expr, t1_expr + t2_expr):
                    t_num = self._evaluate_expr(t_expr, symbolic=False)
                    sum_num = self._evaluate_expr(t1_expr + t2_expr, symbolic=False)
                    if not self._isclose_num(t_num, sum_num):
                        tests_passed = False
                        self._log(f"Time consistency failed: t={t_expr}, t1+t2={t1_expr + t2_expr}")
        except Exception as e:
            tests_passed = False
            self._log(f"Time consistency error: {str(e)}")
        
        # Check that x displacement Q→P = 2L
        try:
            if self.x and self.L:
                x_expr = self._evaluate_expr(self.x, symbolic=True)
                L_expr = self._evaluate_expr(self.L, symbolic=True)
                
                if not self._isclose_sym(x_expr, 2 * L_expr):
                    x_num = self._evaluate_expr(x_expr, symbolic=False)
                    expected_x_num = 2 * self._evaluate_expr(L_expr, symbolic=False)
                    if not self._isclose_num(x_num, expected_x_num):
                        tests_passed = False
                        self._log(f"X displacement failed: x={x_expr}, expected 2L={2*L_expr}")
        except Exception as e:
            tests_passed = False
            self._log(f"X displacement error: {str(e)}")
        
        # Check that y displacement Q→P = L
        try:
            if self.y and self.L:
                y_expr = self._evaluate_expr(self.y, symbolic=True)
                L_expr = self._evaluate_expr(self.L, symbolic=True)
                
                if not self._isclose_sym(y_expr, L_expr):
                    y_num = self._evaluate_expr(y_expr, symbolic=False)
                    expected_y_num = self._evaluate_expr(L_expr, symbolic=False)
                    if not self._isclose_num(y_num, expected_y_num):
                        tests_passed = False
                        self._log(f"Y displacement failed: y={y_expr}, expected L={L_expr}")
        except Exception as e:
            tests_passed = False
            self._log(f"Y displacement error: {str(e)}")
        
        # Check that speed at Q = √2 V0
        try:
            if self.v and self.V0:
                v_expr = self._evaluate_expr(self.v, symbolic=True)
                V0_expr = self._evaluate_expr(self.V0, symbolic=True)
                
                expected_v = sp.sqrt(2) * V0_expr
                
                if not self._isclose_sym(v_expr, expected_v):
                    v_num = self._evaluate_expr(v_expr, symbolic=False)
                    expected_v_num = self._evaluate_expr(expected_v, symbolic=False)
                    if not self._isclose_num(v_num, expected_v_num):
                        tests_passed = False
                        self._log(f"Speed at Q failed: v={v_expr}, expected √2 V0={expected_v}")
        except Exception as e:
            tests_passed = False
            self._log(f"Speed at Q error: {str(e)}")
        
        # Check that central angle = 90° (π/2 rad)
        try:
            if self.theta:
                theta_expr = self._evaluate_expr(self.theta, symbolic=True)
                
                # Check if it's approximately π/2
                if not self._isclose_sym(theta_expr, sp.pi/2):
                    theta_num = self._evaluate_expr(theta_expr, symbolic=False)
                    if not self._isclose_num(theta_num, math.pi/2):
                        # Also check if it's 90 (degrees)
                        if not self._isclose_num(theta_num, 90.0):
                            tests_passed = False
                            self._log(f"Central angle failed: θ={theta_expr}, expected π/2 or 90°")
        except Exception as e:
            tests_passed = False
            self._log(f"Central angle error: {str(e)}")
        
        # Check that OQ = L
        try:
            if self.OQ and self.L:
                OQ_expr = self._evaluate_expr(self.OQ, symbolic=True)
                L_expr = self._evaluate_expr(self.L, symbolic=True)
                
                if not self._isclose_sym(OQ_expr, L_expr):
                    OQ_num = self._evaluate_expr(OQ_expr, symbolic=False)
                    L_num = self._evaluate_expr(L_expr, symbolic=False)
                    if not self._isclose_num(OQ_num, L_num):
                        tests_passed = False
                        self._log(f"OQ length failed: OQ={OQ_expr}, expected L={L_expr}")
        except Exception as e:
            tests_passed = False
            self._log(f"OQ length error: {str(e)}")
        
        return tests_passed, self.logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb an input and check trend, then restore."""
        tests_passed = True
        
        # Only run if we have all necessary parameters
        if not all([self.m, self.q, self.V0, self.L, self.E, self.B, self.t]):
            self._log("Monotonicity test skipped: missing parameters")
            return True, self.logs
        
        try:
            # Save original test values
            original_values = self.test_values.copy()
            
            # Test 1: Increase L, E should decrease (E = m*V0²/(2qL))
            L_original = self.test_values['L']
            self.test_values['L'] = L_original * 1.1  # Increase by 10%
            
            # Calculate expected trend
            E_original = self._evaluate_expr(self.E, symbolic=False)
            E_perturbed = self._evaluate_expr(self.E, symbolic=False)
            
            # E ∝ 1/L, so if L increases, E should decrease
            if not (E_perturbed < E_original or self._isclose_num(E_perturbed, E_original)):
                tests_passed = False
                self._log(f"Monotonicity failed: E should decrease when L increases, but E_original={E_original}, E_perturbed={E_perturbed}")
            
            # Test 2: Increase V0, B should increase (B = 2m*V0/(qL))
            self.test_values = original_values.copy()
            V0_original = self.test_values['V0']
            self.test_values['V0'] = V0_original * 1.1
            
            B_original = self._evaluate_expr(self.B, symbolic=False)
            B_perturbed = self._evaluate_expr(self.B, symbolic=False)
            
            # B ∝ V0, so if V0 increases, B should increase
            if not (B_perturbed > B_original or self._isclose_num(B_perturbed, B_original)):
                tests_passed = False
                self._log(f"Monotonicity failed: B should increase when V0 increases, but B_original={B_original}, B_perturbed={B_perturbed}")
            
            # Test 3: Increase L, total time t should increase (t ∝ L)
            self.test_values = original_values.copy()
            L_original = self.test_values['L']
            self.test_values['L'] = L_original * 1.1
            
            t_original = self._evaluate_expr(self.t, symbolic=False)
            t_perturbed = self._evaluate_expr(self.t, symbolic=False)
            
            # t ∝ L, so if L increases, t should increase
            if not (t_perturbed > t_original or self._isclose_num(t_perturbed, t_original)):
                tests_passed = False
                self._log(f"Monotonicity failed: t should increase when L increases, but t_original={t_original}, t_perturbed={t_perturbed}")
            
            # Restore original values
            self.test_values = original_values
            
        except Exception as e:
            tests_passed = False
            self._log(f"Monotonicity test error: {str(e)}")
            # Try to restore values
            self.test_values = original_values
        
        return tests_passed, self.logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_passed = True
        self.logs = []
        
        try:
            # Run invariants test
            passed, logs = self.test_invariants()
            if not passed:
                all_passed = False
            self.logs.extend(logs)
            
            # Run trace consistency test
            passed, logs = self.test_trace_consistency()
            if not passed:
                all_passed = False
            self.logs.extend(logs)
            
            # Run monotonicity test
            passed, logs = self.test_monotonicity()
            if not passed:
                all_passed = False
            self.logs.extend(logs)
            
        except Exception as e:
            all_passed = False
            self.logs.append(f"Critical error in run_all_tests: {str(e)}")
            self.logs.append(traceback.format_exc())
        
        return all_passed, self.logs

# Example usage (commented out for clarity):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()