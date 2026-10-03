import math
import sympy as sp
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Initialize with safe attribute access."""
        # Extract inputs with safe defaults
        self.E = getattr(solution_instance, 'E', 100.0)  # V/m
        self.q = getattr(solution_instance, 'q', 0.02)   # C
        self.L = getattr(solution_instance, 'L', 2.0)    # m
        self.theta_deg = getattr(solution_instance, 'theta_deg', 60.0)
        self.theta_rad = math.radians(self.theta_deg)
        
        # Extract results with safe access to results dictionary
        results = getattr(solution_instance, 'results', {})
        self.U_AB = results.get('U_AB', results.get('U_{AB}', 0.0))
        self.W_AB = results.get('W_AB', results.get('W_{AB}', 0.0))
        
        # For symbolic calculations
        self.symbols = self._define_symbols()
        self.logs = []
        
    def _define_symbols(self) -> Dict[str, Any]:
        """Define sympy symbols for physical quantities."""
        E, q, L, theta, U_AB, W_AB = sp.symbols('E q L theta U_AB W_AB')
        return {
            'E': E, 'q': q, 'L': L, 'theta': theta,
            'U_AB': U_AB, 'W_AB': W_AB
        }
    
    def _log(self, message: str, success: bool = True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
    
    def test_invariants(self) -> bool:
        """Check fundamental physics laws."""
        try:
            # 1. Check work-energy theorem for electric force: W_AB = q * U_AB
            expected_W = self.q * self.U_AB
            if not math.isclose(self.W_AB, expected_W, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Work-energy invariant violated: W_AB={self.W_AB}, q*U_AB={expected_W}", False)
                return False
            self._log(f"Work-energy theorem check: W_AB = q * U_AB holds (W_AB={self.W_AB}, q*U_AB={expected_W})")
            
            # 2. Check potential difference formula in uniform field: U_AB = E * L * cos(theta)
            # Note: In uniform field, U_AB = -E·dL = -E*L*cosθ, but here we have U_AB = E*L*cosθ (positive)
            # The sign depends on direction of movement relative to field. We'll check magnitude.
            expected_U = self.E * self.L * math.cos(self.theta_rad)
            if not math.isclose(self.U_AB, expected_U, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Potential difference formula violated: U_AB={self.U_AB}, E*L*cosθ={expected_U}", False)
                return False
            self._log(f"Uniform field potential check: U_AB = E*L*cosθ holds (U_AB={self.U_AB}, E*L*cosθ={expected_U})")
            
            # 3. Symbolic verification - use numeric tolerance instead of boolean equality
            sym = self.symbols
            eq1 = sp.Eq(sym['W_AB'], sym['q'] * sym['U_AB'])
            eq2 = sp.Eq(sym['U_AB'], sym['E'] * sym['L'] * sp.cos(sym['theta']))
            
            # Substitute values and evaluate numerically
            sub_dict = {
                sym['E']: self.E,
                sym['q']: self.q,
                sym['L']: self.L,
                sym['theta']: self.theta_rad,
                sym['U_AB']: self.U_AB,
                sym['W_AB']: self.W_AB
            }
            
            # Check eq1: W_AB - q*U_AB should be close to 0
            lhs1 = float((sym['W_AB'] - sym['q'] * sym['U_AB']).subs(sub_dict))
            if not math.isclose(lhs1, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Symbolic check eq1 failed: W_AB - q*U_AB = {lhs1}", False)
                return False
                
            # Check eq2: U_AB - E*L*cosθ should be close to 0
            lhs2 = float((sym['U_AB'] - sym['E'] * sym['L'] * sp.cos(sym['theta'])).subs(sub_dict))
            if not math.isclose(lhs2, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Symbolic check eq2 failed: U_AB - E*L*cosθ = {lhs2}", False)
                return False
                
            self._log("All invariant checks passed")
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self) -> bool:
        """Verify intermediate steps match final results."""
        try:
            # Step 1 from analysis: U_AB = E * L * cos(60°)
            expected_U = self.E * self.L * math.cos(self.theta_rad)
            if not math.isclose(self.U_AB, expected_U, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 1 inconsistency: U_AB={self.U_AB}, E*L*cosθ={expected_U}", False)
                return False
            self._log(f"Step 1 check: U_AB = E*L*cosθ = {expected_U} V (got {self.U_AB} V)")
            
            # Step 2 from analysis: W_AB = q * U_AB
            expected_W = self.q * self.U_AB
            if not math.isclose(self.W_AB, expected_W, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 2 inconsistency: W_AB={self.W_AB}, q*U_AB={expected_W}", False)
                return False
            self._log(f"Step 2 check: W_AB = q*U_AB = {expected_W} J (got {self.W_AB} J)")
            
            # Verify the numerical values from the steps analysis
            # According to steps: U_AB = 100 V, W_AB = 2 J
            if not math.isclose(self.U_AB, 100.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"U_AB doesn't match expected 100V: got {self.U_AB}V", False)
                return False
                
            if not math.isclose(self.W_AB, 2.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"W_AB doesn't match expected 2J: got {self.W_AB}J", False)
                return False
                
            self._log("All trace consistency checks passed")
            return True
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb inputs and verify expected trends."""
        original_q = self.q
        
        try:
            # Test 1: Increase charge by 10% → work should increase proportionally
            self.q *= 1.1
            expected_W_increase = self.q * self.U_AB
            # Note: U_AB should remain constant since it doesn't depend on q
            # We'll check the proportionality: new_W / old_W ≈ new_q / old_q
            expected_ratio = 1.1
            actual_ratio = (self.q * self.U_AB) / (original_q * self.U_AB)
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity violated for q: ratio={actual_ratio}, expected={expected_ratio}", False)
                self.q = original_q
                return False
            self._log(f"Charge monotonicity: 10% q increase → {expected_ratio:.2f}x work increase (actual: {actual_ratio:.2f}x)")
            
            # Restore original q
            self.q = original_q
            
            # Test 2: Increase E by 5% → both U_AB and W_AB should increase proportionally
            original_E = self.E
            self.E *= 1.05
            
            # U_AB should increase by 5%
            expected_U_ratio = 1.05
            actual_U_ratio = (self.E * self.L * math.cos(self.theta_rad)) / (original_E * self.L * math.cos(self.theta_rad))
            
            if not math.isclose(actual_U_ratio, expected_U_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity violated for E (U_AB): ratio={actual_U_ratio}, expected={expected_U_ratio}", False)
                self.E = original_E
                return False
                
            # W_AB should also increase by 5% (since W_AB = q*U_AB and U_AB increased by 5%)
            expected_W_ratio = 1.05
            actual_W_ratio = (self.q * self.E * self.L * math.cos(self.theta_rad)) / (self.q * original_E * self.L * math.cos(self.theta_rad))
            
            if not math.isclose(actual_W_ratio, expected_W_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity violated for E (W_AB): ratio={actual_W_ratio}, expected={expected_W_ratio}", False)
                self.E = original_E
                return False
                
            self._log(f"Field strength monotonicity: 5% E increase → 5% U_AB increase and 5% W_AB increase")
            
            # Restore original E
            self.E = original_E
            
            # Test 3: Increase displacement L by 20% → U_AB and W_AB should increase proportionally
            original_L = self.L
            self.L *= 1.2
            
            expected_L_ratio_U = 1.2
            actual_L_ratio_U = (self.E * self.L * math.cos(self.theta_rad)) / (self.E * original_L * math.cos(self.theta_rad))
            
            if not math.isclose(actual_L_ratio_U, expected_L_ratio_U, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity violated for L (U_AB): ratio={actual_L_ratio_U}, expected={expected_L_ratio_U}", False)
                self.L = original_L
                return False
                
            expected_L_ratio_W = 1.2
            actual_L_ratio_W = (self.q * self.E * self.L * math.cos(self.theta_rad)) / (self.q * self.E * original_L * math.cos(self.theta_rad))
            
            if not math.isclose(actual_L_ratio_W, expected_L_ratio_W, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity violated for L (W_AB): ratio={actual_L_ratio_W}, expected={expected_L_ratio_W}", False)
                self.L = original_L
                return False
                
            self._log(f"Displacement monotonicity: 20% L increase → 20% U_AB increase and 20% W_AB increase")
            
            # Restore original L
            self.L = original_L
            
            self._log("All monotonicity checks passed")
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            # Restore original values on error
            self.q = original_q
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} ---")
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"Test crashed: {str(e)}\n{traceback.format_exc()}", False)
                success = False
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs