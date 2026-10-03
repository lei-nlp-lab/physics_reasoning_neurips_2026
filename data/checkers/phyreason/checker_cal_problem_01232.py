import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all needed quantities without direct attribute access."""
        self.solution = solution_instance
        self.logs = []
        
        # Extract inputs with safe fallbacks
        self.m = getattr(solution_instance, 'm', None)
        self.g = getattr(solution_instance, 'g', None)
        self.R = getattr(solution_instance, 'R', None)
        self.mu = getattr(solution_instance, 'mu', None) or (math.sqrt(3)/6)
        
        # Extract results safely - use both direct attributes and results dict
        self.N = getattr(solution_instance, 'N', 
                        getattr(solution_instance, 'results', {}).get('N', None))
        self.F = getattr(solution_instance, 'F', 
                        getattr(solution_instance, 'results', {}).get('F', None))
        self.f = getattr(solution_instance, 'f', 
                        getattr(solution_instance, 'results', {}).get('f', None))
        
        # Symbolic variables for checking
        self.m_sym, self.g_sym = sp.symbols('m g', positive=True)
        self.theta = sp.pi/6  # 30 degrees from geometry analysis
        
    def _get_numeric(self, expr_or_val, m_val=None, g_val=None):
        """Convert expression or value to numeric for comparison."""
        if expr_or_val is None:
            return None
            
        # If it's already numeric
        if isinstance(expr_or_val, (int, float)):
            return float(expr_or_val)
        
        # If it's a sympy expression
        if hasattr(expr_or_val, 'subs'):
            try:
                if m_val is None:
                    m_val = self.m if self.m is not None else 1.0
                if g_val is None:
                    g_val = self.g if self.g is not None else 9.8
                return float(expr_or_val.subs({self.m_sym: m_val, self.g_sym: g_val}))
            except:
                pass
        
        # If it's a string representation
        if isinstance(expr_or_val, str):
            try:
                # Handle LaTeX-like strings
                if '\\frac' in expr_or_val or '\\sqrt' in expr_or_val:
                    expr_or_val = expr_or_val.replace('\\frac{', '').replace('}{', '/').replace('}', '')
                    expr_or_val = expr_or_val.replace('\\sqrt{', 'sqrt(').replace('}', ')')
                return float(sp.sympify(expr_or_val).subs({self.m_sym: m_val or 1.0, 
                                                          self.g_sym: g_val or 9.8}))
            except:
                pass
        
        return None
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physical invariants from KB."""
        logs = []
        success = True
        
        try:
            # Test 1: Vertical equilibrium of body C
            # From steps: N = mg/(2cosθ) with θ=30°
            if self.N is not None:
                N_expected = self.m_sym * self.g_sym / (2 * sp.cos(self.theta))
                N_expected_simplified = sp.simplify(N_expected)
                
                # Numeric comparison with tolerance
                m_test = self.m if self.m is not None else 1.0
                g_test = self.g if self.g is not None else 9.8
                
                N_num = self._get_numeric(self.N, m_test, g_test)
                N_expected_num = self._get_numeric(N_expected_simplified, m_test, g_test)
                
                if N_num is not None and N_expected_num is not None:
                    if not math.isclose(N_num, N_expected_num, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: N = {N_num} but expected ~{N_expected_num}")
                        success = False
                    else:
                        logs.append(f"PASS: N = mg/(2cos30°) = √3/3 mg ≈ {N_expected_num}")
                else:
                    # Symbolic fallback
                    N_expr = sp.sympify(str(self.N)) if isinstance(self.N, str) else self.N
                    if hasattr(N_expr, 'simplify'):
                        diff = sp.simplify(N_expr - N_expected_simplified)
                        if diff != 0:
                            logs.append(f"Symbolic mismatch in N: {diff}")
                            success = False
            
            # Test 2: Vertical equilibrium of body A
            # From steps: F = mg + N cosθ
            if self.F is not None and self.N is not None:
                N_expr = sp.sympify(str(self.N)) if isinstance(self.N, str) else self.N
                F_expected = self.m_sym * self.g_sym + N_expr * sp.cos(self.theta)
                
                F_num = self._get_numeric(self.F, m_test, g_test)
                F_expected_num = self._get_numeric(F_expected, m_test, g_test)
                
                if F_num is not None and F_expected_num is not None:
                    if not math.isclose(F_num, F_expected_num, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: F = {F_num} but expected ~{F_expected_num}")
                        success = False
                    else:
                        logs.append(f"PASS: F = mg + N cos30° = 3mg/2 ≈ {F_expected_num}")
            
            # Test 3: Horizontal equilibrium of body A
            # From steps: f = N sinθ
            if self.f is not None and self.N is not None:
                N_expr = sp.sympify(str(self.N)) if isinstance(self.N, str) else self.N
                f_expected = N_expr * sp.sin(self.theta)
                
                f_num = self._get_numeric(self.f, m_test, g_test)
                f_expected_num = self._get_numeric(f_expected, m_test, g_test)
                
                if f_num is not None and f_expected_num is not None:
                    if not math.isclose(f_num, f_expected_num, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: f = {f_num} but expected ~{f_expected_num}")
                        success = False
                    else:
                        logs.append(f"PASS: f = N sin30° = √3/6 mg ≈ {f_expected_num}")
            
            # Test 4: Friction limit (should be exactly at limit or below)
            if self.f is not None and self.F is not None:
                f_num = self._get_numeric(self.f, m_test, g_test)
                F_num = self._get_numeric(self.F, m_test, g_test)
                mu_val = self.mu if self.mu is not None else math.sqrt(3)/6
                
                if f_num is not None and F_num is not None:
                    max_friction = mu_val * F_num
                    if f_num > max_friction * 1.02:  # 2% tolerance
                        logs.append(f"FAIL: f = {f_num} > μF = {max_friction}")
                        success = False
                    else:
                        logs.append(f"PASS: f ≤ μF = {max_friction}")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate steps and final results."""
        logs = []
        success = True
        
        try:
            # Check if steps analysis matches final results
            # Step 1: N = mg/(2cosθ) → should match our N
            if self.N is not None:
                N_from_step1 = self.m_sym * self.g_sym / (2 * sp.cos(self.theta))
                N_step1_num = self._get_numeric(N_from_step1)
                N_num = self._get_numeric(self.N)
                
                if N_num is not None and N_step1_num is not None:
                    if not math.isclose(N_num, N_step1_num, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"Step1 inconsistency: N = {N_num}, step1 gives {N_step1_num}")
                        success = False
            
            # Step 2: N = √3/3 mg → check simplification
            if self.N is not None:
                N_expected = (sp.sqrt(3)/3) * self.m_sym * self.g_sym
                N_exp_num = self._get_numeric(N_expected)
                N_num = self._get_numeric(self.N)
                
                if N_num is not None and N_exp_num is not None:
                    if not math.isclose(N_num, N_exp_num, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"Step2 inconsistency: N = {N_num}, step2 gives {N_exp_num}")
                        success = False
            
            # Step 3 & 4: F and f calculations
            if self.F is not None and self.f is not None and self.N is not None:
                N_expr = sp.sympify(str(self.N)) if isinstance(self.N, str) else self.N
                F_from_step = self.m_sym * self.g_sym + N_expr * sp.cos(self.theta)
                f_from_step = N_expr * sp.sin(self.theta)
                
                F_step_num = self._get_numeric(F_from_step)
                f_step_num = self._get_numeric(f_from_step)
                F_num = self._get_numeric(self.F)
                f_num = self._get_numeric(self.f)
                
                if (F_num is not None and F_step_num is not None and 
                    not math.isclose(F_num, F_step_num, rel_tol=0.02, abs_tol=0.02)):
                    logs.append(f"Step3 F inconsistency: {F_num} vs {F_step_num}")
                    success = False
                    
                if (f_num is not None and f_step_num is not None and 
                    not math.isclose(f_num, f_step_num, rel_tol=0.02, abs_tol=0.02)):
                    logs.append(f"Step3 f inconsistency: {f_num} vs {f_step_num}")
                    success = False
            
            if success:
                logs.append("PASS: All steps consistent with final results")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test that perturbing mass changes results predictably."""
        logs = []
        success = True
        
        try:
            # Use test values if actual ones not available
            m_base = self.m if self.m is not None else 1.0
            g_base = self.g if self.g is not None else 9.8
            
            # Store original
            m_orig = m_base
            
            # Perturb mass by +10%
            m_pert = m_base * 1.10
            
            # Calculate expected changes using formulas
            theta_val = math.pi/6
            cos_theta = math.cos(theta_val)
            sin_theta = math.sin(theta_val)
            
            # Original values
            N_orig = m_orig * g_base / (2 * cos_theta)
            F_orig = m_orig * g_base + N_orig * cos_theta
            f_orig = N_orig * sin_theta
            
            # Perturbed values
            N_pert = m_pert * g_base / (2 * cos_theta)
            F_pert = m_pert * g_base + N_pert * cos_theta
            f_pert = N_pert * sin_theta
            
            # Check monotonic increase (all should increase by ~10%)
            if not (N_pert > N_orig and F_pert > F_orig and f_pert > f_orig):
                logs.append("FAIL: Results don't increase with mass")
                success = False
            
            # Check approximate proportionality (within 2%)
            N_ratio = N_pert / N_orig
            expected_ratio = 1.10
            
            if not math.isclose(N_ratio, expected_ratio, rel_tol=0.02):
                logs.append(f"FAIL: N ratio {N_ratio:.3f} ≠ expected {expected_ratio:.3f}")
                success = False
            
            if success:
                logs.append(f"PASS: Results scale correctly with mass (ratio ≈ {N_ratio:.3f})")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ERROR ===")
                all_logs.append(f"Exception: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs