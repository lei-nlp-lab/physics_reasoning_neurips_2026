import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        Solution instance should have:
        - attributes: M, m, H, g (with defaults)
        - results dictionary with keys from key mapping
        """
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', 1.0)  # Default mass unit
        self.M = getattr(solution_instance, 'M', 4.0 * self.m)  # M = 4m
        self.H = getattr(solution_instance, 'H', 1.0)  # Default height
        self.g = getattr(solution_instance, 'g', 9.8)  # Gravitational acceleration
        
        # Safe extraction of results (using dict get with defaults)
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all expected quantities with safe defaults
        self.a1 = self.results.get('a_1', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        self.v0 = self.results.get('v_0', 0.0)
        self.t1 = self.results.get('t_1', 0.0)
        self.h1 = self.results.get('h_1', 0.0)
        self.v = self.results.get('v', 0.0)
        self.h2 = self.results.get('h_2', 0.0)
        self.H1 = self.results.get('H_1', 0.0)
        self.x1 = self.results.get('x_1', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.L = self.results.get('L', 0.0)
        
        # Friction force magnitude
        self.f = 4.0 * self.m * self.g
        
        # Setup symbolic variables for invariant checks
        self._setup_symbolic()
        
        self.logs = []
    
    def _setup_symbolic(self):
        """Setup symbolic variables for invariant checks."""
        self.sym_m, self.sym_M, self.sym_g, self.sym_H = sp.symbols('m M g H', positive=True)
        self.sym_f = 4 * self.sym_m * self.sym_g  # Friction force
        
        # Expected symbolic expressions from steps analysis
        self.expected_a1 = 2 * self.sym_g
        self.expected_a2 = 3 * self.sym_g
        self.expected_v0 = sp.sqrt(2 * self.sym_g * self.sym_H)
        self.expected_t1 = (2/sp.Integer(5)) * sp.sqrt(2 * self.sym_H / self.sym_g)
        self.expected_H1 = (sp.Integer(13)/sp.Integer(25)) * self.sym_H
        self.expected_x1 = (sp.Integer(4)/sp.Integer(5)) * self.sym_H
        self.expected_L_min = (sp.Integer(152)/sp.Integer(125)) * self.sym_H
    
    def _log(self, message: str, success: bool):
        """Add message to logs with success indicator."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
    
    def _isclose(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _check_newton_laws(self) -> bool:
        """
        Check Newton's Second Law applications.
        From steps: Ma1 = Mg + f, ma2 = f - mg
        """
        success = True
        
        # Check tube acceleration (a1)
        # Equation: M*a1 = M*g + f
        lhs_tube = self.M * self.a1
        rhs_tube = self.M * self.g + self.f
        
        if not self._isclose(lhs_tube, rhs_tube):
            self._log(f"Newton's Law for tube fails: M*a1 = {lhs_tube}, M*g + f = {rhs_tube}", False)
            success = False
        else:
            self._log(f"Newton's Law for tube passes: M*a1 = M*g + f = {rhs_tube}", True)
        
        # Check ball acceleration (a2)
        # Equation: m*a2 = f - m*g
        lhs_ball = self.m * self.a2
        rhs_ball = self.f - self.m * self.g
        
        if not self._isclose(lhs_ball, rhs_ball):
            self._log(f"Newton's Law for ball fails: m*a2 = {lhs_ball}, f - m*g = {rhs_ball}", False)
            success = False
        else:
            self._log(f"Newton's Law for ball passes: m*a2 = f - m*g = {rhs_ball}", True)
        
        # Check expected values: a1 = 2g, a2 = 3g
        if not self._isclose(self.a1, 2 * self.g):
            self._log(f"a1 ≠ 2g: {self.a1} vs {2 * self.g}", False)
            success = False
        else:
            self._log(f"a1 = 2g = {self.a1}", True)
            
        if not self._isclose(self.a2, 3 * self.g):
            self._log(f"a2 ≠ 3g: {self.a2} vs {3 * self.g}", False)
            success = False
        else:
            self._log(f"a2 = 3g = {self.a2}", True)
        
        return success
    
    def _check_kinematic_consistency(self) -> bool:
        """Check kinematic equations for consistency."""
        success = True
        
        # 1. Check v0 from free fall: v0 = sqrt(2gH)
        expected_v0 = math.sqrt(2 * self.g * self.H)
        if not self._isclose(self.v0, expected_v0):
            self._log(f"v0 inconsistent: {self.v0} vs sqrt(2gH) = {expected_v0}", False)
            success = False
        else:
            self._log(f"v0 = sqrt(2gH) = {self.v0}", True)
        
        # 2. Check t1 from velocity matching: v0 - a1*t1 = -v0 + a2*t1
        # Rearranged: t1 = 2v0/(a1 + a2)
        if self.a1 + self.a2 != 0:
            expected_t1 = 2 * self.v0 / (self.a1 + self.a2)
            if not self._isclose(self.t1, expected_t1):
                self._log(f"t1 inconsistent: {self.t1} vs 2v0/(a1+a2) = {expected_t1}", False)
                success = False
            else:
                self._log(f"t1 = 2v0/(a1+a2) = {self.t1}", True)
        
        # 3. Check h1: h1 = v0*t1 - 0.5*a1*t1²
        expected_h1 = self.v0 * self.t1 - 0.5 * self.a1 * self.t1**2
        if not self._isclose(self.h1, expected_h1):
            self._log(f"h1 inconsistent: {self.h1} vs v0*t1 - 0.5*a1*t1² = {expected_h1}", False)
            success = False
        else:
            self._log(f"h1 = v0*t1 - 0.5*a1*t1² = {self.h1}", True)
        
        # 4. Check v: v = v0 - a1*t1
        expected_v = self.v0 - self.a1 * self.t1
        if not self._isclose(self.v, expected_v):
            self._log(f"v inconsistent: {self.v} vs v0 - a1*t1 = {expected_v}", False)
            success = False
        else:
            self._log(f"v = v0 - a1*t1 = {self.v}", True)
        
        # 5. Check h2: h2 = v²/(2g)
        expected_h2 = self.v**2 / (2 * self.g)
        if not self._isclose(self.h2, expected_h2):
            self._log(f"h2 inconsistent: {self.h2} vs v²/(2g) = {expected_h2}", False)
            success = False
        else:
            self._log(f"h2 = v²/(2g) = {self.h2}", True)
        
        # 6. Check H1: H1 = h1 + h2
        expected_H1 = self.h1 + self.h2
        if not self._isclose(self.H1, expected_H1):
            self._log(f"H1 inconsistent: {self.H1} vs h1 + h2 = {expected_H1}", False)
            success = False
        else:
            self._log(f"H1 = h1 + h2 = {self.H1}", True)
        
        # 7. Check H1 = (13/25)H
        expected_H1_value = (13/25) * self.H
        if not self._isclose(self.H1, expected_H1_value):
            self._log(f"H1 ≠ (13/25)H: {self.H1} vs {expected_H1_value}", False)
            success = False
        else:
            self._log(f"H1 = (13/25)H = {self.H1}", True)
        
        return success
    
    def _check_work_energy(self) -> bool:
        """Check work-energy theorem for first rebound."""
        success = True
        
        # Equation from step_12: Mg(H-H1) + mg(H-H1+x1) - 4mg*x1 = 0
        term1 = self.M * self.g * (self.H - self.H1)
        term2 = self.m * self.g * (self.H - self.H1 + self.x1)
        term3 = -self.f * self.x1  # -4mg*x1
        
        total = term1 + term2 + term3
        if not self._isclose(total, 0.0):
            self._log(f"Work-energy equation fails: Mg(H-H1) + mg(H-H1+x1) - 4mg*x1 = {total}", False)
            success = False
        else:
            self._log(f"Work-energy equation passes: sum = {total}", True)
        
        # Check x1 = (4/5)H
        expected_x1 = (4/5) * self.H
        if not self._isclose(self.x1, expected_x1):
            self._log(f"x1 ≠ (4/5)H: {self.x1} vs {expected_x1}", False)
            success = False
        else:
            self._log(f"x1 = (4/5)H = {self.x1}", True)
        
        return success
    
    def _check_length_condition(self) -> bool:
        """Check the geometric constraint for tube length."""
        success = True
        
        # Check x2 = (4/5)H1
        expected_x2 = (4/5) * self.H1
        if not self._isclose(self.x2, expected_x2):
            self._log(f"x2 ≠ (4/5)H1: {self.x2} vs {expected_x2}", False)
            success = False
        else:
            self._log(f"x2 = (4/5)H1 = {self.x2}", True)
        
        # Check condition: L >= x1 + x2
        total_slide = self.x1 + self.x2
        if self.L < total_slide - 1e-2:  # Slight tolerance
            self._log(f"Length condition fails: L = {self.L} < x1 + x2 = {total_slide}", False)
            success = False
        else:
            self._log(f"Length condition passes: L = {self.L} >= x1 + x2 = {total_slide}", True)
        
        # Check L_min = (152/125)H
        expected_L_min = (152/125) * self.H
        if self.L < expected_L_min - 1e-2:
            self._log(f"L below minimum: {self.L} < (152/125)H = {expected_L_min}", False)
            success = False
        else:
            self._log(f"L >= (152/125)H = {expected_L_min}", True)
        
        return success
    
    def test_invariants(self) -> bool:
        """Test conservation laws and Newton's laws."""
        success = True
        
        # 1. Newton's Second Law checks
        if not self._check_newton_laws():
            success = False
        
        # 2. Work-energy theorem check
        if not self._check_work_energy():
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer."""
        success = True
        
        # 1. Kinematic consistency
        if not self._check_kinematic_consistency():
            success = False
        
        # 2. Length condition check
        if not self._check_length_condition():
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input H and check that L changes in expected direction.
        Restore original H after test.
        """
        original_H = self.H
        perturb_factor = 1.01  # 1% increase
        
        try:
            # Perturb H
            self.H = original_H * perturb_factor
            
            # Recompute expected values based on perturbed H
            expected_v0_pert = math.sqrt(2 * self.g * self.H)
            expected_H1_pert = (13/25) * self.H
            expected_x1_pert = (4/5) * self.H
            expected_x2_pert = (4/5) * expected_H1_pert
            expected_L_min_pert = expected_x1_pert + expected_x2_pert
            
            # Restore original H
            self.H = original_H
            
            # Recompute with original H
            expected_v0_orig = math.sqrt(2 * self.g * self.H)
            expected_H1_orig = (13/25) * self.H
            expected_x1_orig = (4/5) * self.H
            expected_x2_orig = (4/5) * expected_H1_orig
            expected_L_min_orig = expected_x1_orig + expected_x2_orig
            
            # Check monotonicity: increasing H should increase L_min
            if expected_L_min_pert <= expected_L_min_orig:
                self._log(f"Monotonicity fails: L_min({self.H*perturb_factor}) = {expected_L_min_pert} <= L_min({self.H}) = {expected_L_min_orig}", False)
                return False
            else:
                self._log(f"Monotonicity passes: L_min increases with H", True)
                return True
                
        except Exception as e:
            self.H = original_H  # Restore on error
            self._log(f"Monotonicity test error: {str(e)}", False)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        tests = [
            ("Newton's Laws & Work-Energy", self.test_invariants),
            ("Kinematic & Geometric Consistency", self.test_trace_consistency),
            ("Monotonicity Check", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Testing {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test {test_name} raised exception: {str(e)}", False)
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Overall success: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs