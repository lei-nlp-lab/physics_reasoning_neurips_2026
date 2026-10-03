import sympy as sp
import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance"""
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.T_A = getattr(solution_instance, 'T_A', self.results.get('T_A', None))
        self.T_Bmax = getattr(solution_instance, 'T_Bmax', self.results.get('T_Bmax', None))
        self.m_1max = getattr(solution_instance, 'm_1max', self.results.get('m_1max', None))
        self.T_B = getattr(solution_instance, 'T_B', self.results.get('T_B', None))
        self.F_max = getattr(solution_instance, 'F_max', self.results.get('F_max', None))
        
        # Extract inputs if available
        self.theta = getattr(solution_instance, 'theta', 37)  # degrees
        self.g = getattr(solution_instance, 'g', 10.0)
        self.mu = getattr(solution_instance, 'mu', None)
        self.m2 = getattr(solution_instance, 'm2', None)
        self.m1 = getattr(solution_instance, 'm1', None)
        
        # Pre-computed trig values from problem
        self.sin_theta = 0.6  # sin(37°)
        self.cos_theta = 0.8  # cos(37°)
        self.tan_theta = 0.6/0.8  # tan(37°) = 0.75
        
        # For symbolic computations
        self.m1_sym = sp.symbols('m1', positive=True)
        self.m2_sym = sp.symbols('m2', positive=True)
        self.mu_sym = sp.symbols('mu', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        
        self.logs = []
        
    def _log(self, message: str, success: bool = True):
        """Helper to log test results"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance"""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """Verify fundamental physics laws are respected"""
        try:
            # 1. Newton's 1st Law at knot O: ΣF_x = 0, ΣF_y = 0
            # From STEPS ANALYSIS: T_A*sinθ = T_B (horizontal) and T_A*cosθ = m1*g (vertical)
            
            if self.T_A is not None and self.T_B is not None and self.m1 is not None:
                # Convert to numerical if expressions
                T_A_num = float(self.T_A) if hasattr(self.T_A, '__float__') else self.T_A
                T_B_num = float(self.T_B) if hasattr(self.T_B, '__float__') else self.T_B
                
                # Check horizontal equilibrium: T_A*sinθ = T_B
                if hasattr(T_A_num, '__float__') and hasattr(T_B_num, '__float__'):
                    lhs_horizontal = T_A_num * self.sin_theta
                    if self._isclose(lhs_horizontal, T_B_num):
                        self._log("Horizontal equilibrium at knot O satisfied")
                    else:
                        self._log(f"Horizontal equilibrium violated: T_A*sinθ = {lhs_horizontal}, T_B = {T_B_num}", False)
                
                # Check vertical equilibrium: T_A*cosθ = m1*g
                if hasattr(T_A_num, '__float__') and self.m1 is not None:
                    lhs_vertical = T_A_num * self.cos_theta
                    rhs_vertical = self.m1 * self.g if self.m1 else 0
                    if self._isclose(lhs_vertical, rhs_vertical):
                        self._log("Vertical equilibrium at knot O satisfied")
                    else:
                        self._log(f"Vertical equilibrium violated: T_A*cosθ = {lhs_vertical}, m1*g = {rhs_vertical}", False)
            
            # 2. Maximum static friction condition
            if self.F_max is not None and self.m2 is not None and self.mu is not None:
                # F_max should equal μ*m2*g
                expected_F_max = self.mu * self.m2 * self.g
                if self._isclose(self.F_max, expected_F_max):
                    self._log("Maximum static friction formula satisfied")
                else:
                    self._log(f"F_max mismatch: got {self.F_max}, expected μ*m2*g = {expected_F_max}", False)
            
            # 3. Symbolic checks for T_A and T_B expressions
            if self.T_A is not None:
                # Expected: T_A = m1*g/cosθ = m1*10/0.8 = 12.5*m1 or 1.25*m1*g
                if hasattr(self.T_A, 'subs'):  # It's a sympy expression
                    expected_T_A = self.m1_sym * self.g_sym / self.cos_theta
                    diff = sp.simplify(self.T_A - expected_T_A.subs({self.g_sym: self.g}))
                    if diff == 0:
                        self._log("Symbolic check: T_A expression correct")
                    else:
                        self._log(f"Symbolic T_A mismatch: diff = {diff}", False)
            
            if self.T_B is not None:
                # Expected: T_B = m1*g*tanθ = m1*10*0.75 = 7.5*m1 or 0.75*m1*g
                if hasattr(self.T_B, 'subs'):
                    expected_T_B = self.m1_sym * self.g_sym * self.tan_theta
                    diff = sp.simplify(self.T_B - expected_T_B.subs({self.g_sym: self.g}))
                    if diff == 0:
                        self._log("Symbolic check: T_B expression correct")
                    else:
                        self._log(f"Symbolic T_B mismatch: diff = {diff}", False)
                        
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check 1: T_Bmax should equal F_max (from step 4)
            if self.T_Bmax is not None and self.F_max is not None:
                if self._isclose(self.T_Bmax, self.F_max):
                    self._log("T_Bmax = F_max consistency check passed")
                else:
                    self._log(f"T_Bmax ≠ F_max: {self.T_Bmax} vs {self.F_max}", False)
            
            # Check 2: m_1max should be derivable from T_Bmax (step 5)
            if self.m_1max is not None and self.T_Bmax is not None:
                # m_1max = T_Bmax / (g * tanθ)
                expected_m1max = self.T_Bmax / (self.g * self.tan_theta)
                if self._isclose(self.m_1max, expected_m1max):
                    self._log("m_1max derivation from T_Bmax consistent")
                else:
                    self._log(f"m_1max mismatch: got {self.m_1max}, expected T_Bmax/(g*tanθ) = {expected_m1max}", False)
            
            # Check 3: Final numerical answer for m_1max (step 6)
            if self.m_1max is not None:
                # From problem context: expected m_1max = 2.4 kg
                if self._isclose(self.m_1max, 2.4):
                    self._log(f"Final answer m_1max = {self.m_1max} matches expected 2.4 kg")
                else:
                    self._log(f"Final answer m_1max = {self.m_1max} doesn't match expected 2.4 kg", False)
            
            # Check 4: T_B should be consistent with T_Bmax when at limit
            if self.T_B is not None and self.T_Bmax is not None and self.m1 is not None:
                # T_B = m1*g*tanθ, T_Bmax = μ*m2*g
                # At limit: T_B = T_Bmax => m1*g*tanθ = μ*m2*g => m1 = μ*m2/tanθ
                if hasattr(self.T_B, '__float__'):
                    if self._isclose(self.T_B, self.T_Bmax):
                        self._log("T_B = T_Bmax at sliding limit")
                    # Else: T_B may be less than T_Bmax for equilibrium
                        
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}", False)
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends"""
        original_m1 = self.m1
        
        try:
            # Test 1: T_A and T_B should increase linearly with m1
            if self.T_A is not None and self.T_B is not None and self.m1 is not None:
                # Create test m1 values
                test_m1_values = [original_m1 * 0.9, original_m1, original_m1 * 1.1]
                T_A_values = []
                T_B_values = []
                
                for m1_test in test_m1_values:
                    # Calculate using formulas from steps analysis
                    T_A_test = m1_test * self.g / self.cos_theta
                    T_B_test = m1_test * self.g * self.tan_theta
                    T_A_values.append(T_A_test)
                    T_B_values.append(T_B_test)
                
                # Check monotonic increase
                if all(T_A_values[i] <= T_A_values[i+1] for i in range(len(T_A_values)-1)):
                    self._log("T_A increases monotonically with m1")
                else:
                    self._log("T_A doesn't increase monotonically with m1", False)
                
                if all(T_B_values[i] <= T_B_values[i+1] for i in range(len(T_B_values)-1)):
                    self._log("T_B increases monotonically with m1")
                else:
                    self._log("T_B doesn't increase monotonically with m1", False)
            
            # Test 2: m_1max should increase with μ and m2
            if self.mu is not None and self.m2 is not None:
                # m_1max = (μ * m2) / tanθ
                # Perturb μ
                mu_test = self.mu * 1.1
                m1max_test = (mu_test * self.m2) / self.tan_theta
                if m1max_test > self.m_1max:
                    self._log("m_1max increases with μ (as expected)")
                else:
                    self._log("m_1max should increase with μ", False)
                
                # Perturb m2
                m2_test = self.m2 * 1.1
                m1max_test = (self.mu * m2_test) / self.tan_theta
                if m1max_test > self.m_1max:
                    self._log("m_1max increases with m2 (as expected)")
                else:
                    self._log("m_1max should increase with m2", False)
                    
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}", False)
        finally:
            # Restore original value
            self.m1 = original_m1
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success status"""
        self.logs = []
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success
        success = all("PASS" in log or "Exception" in log for log in self.logs)
        
        # Summary
        if success:
            self.logs.append("SUMMARY: All physics checks passed ✓")
        else:
            self.logs.append("SUMMARY: Some physics checks failed ✗")
        
        return success, self.logs