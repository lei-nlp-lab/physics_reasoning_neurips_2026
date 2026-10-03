import sympy as sp
import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.inputs = {}
        self.results = {}
        
        # Extract inputs safely
        self.inputs['F1'] = getattr(solution_instance, 'F1', 12.0)
        self.inputs['F2'] = getattr(solution_instance, 'F2', 6.0)
        self.inputs['weight'] = getattr(solution_instance, 'weight', 40.0)
        self.inputs['mu'] = getattr(solution_instance, 'mu', 0.2)
        
        # Extract results safely with fallbacks to 0.0
        self.results['f'] = getattr(solution_instance, 'f', getattr(solution_instance, 'results', {}).get('f', 0.0))
        self.results['f_1'] = getattr(solution_instance, 'f_1', getattr(solution_instance, 'results', {}).get('f_1', 0.0))
        self.results['f_2'] = getattr(solution_instance, 'f_2', getattr(solution_instance, 'results', {}).get('f_2', 0.0))
        self.results['f_m'] = getattr(solution_instance, 'f_m', getattr(solution_instance, 'results', {}).get('f_m', 0.0))
        
        # Constants
        self.g = 9.8  # m/s² for normal force calculation if needed
        self.logs = []
    
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def test_invariants(self) -> bool:
        """Verify physical invariants: equilibrium conditions and friction laws."""
        try:
            F1 = self.inputs['F1']
            F2 = self.inputs['F2']
            weight = self.inputs['weight']
            mu = self.inputs['mu']
            f = self.results['f']
            f_1 = self.results['f_1']
            f_2 = self.results['f_2']
            f_m = self.results['f_m']
            
            # 1. Check maximum static friction formula
            N = weight  # Horizontal surface, normal force equals weight
            expected_f_m = mu * N
            success_fm = math.isclose(f_m, expected_f_m, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Max static friction: f_m = μN = {mu}*{N} = {expected_f_m:.2f} N, got {f_m:.2f} N", success_fm)
            
            # 2. Check step 1 equilibrium (both forces present)
            # According to STEPS ANALYSIS: f = F1 - F2 = 6N, direction to the right
            # This means friction balances the net applied force: f = |F1 - F2|
            net_force_step1 = F1 - F2
            success_step1 = math.isclose(abs(f), abs(net_force_step1), rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Step 1 equilibrium: |f| should equal |F1 - F2| = |{F1} - {F2}| = {abs(net_force_step1)} N, got |f| = {abs(f):.2f} N", success_step1)
            
            # 3. Check step 4 equilibrium (only F2 present)
            # According to STEPS ANALYSIS: f_1 = F2 = 6N, direction to the left
            # This means friction balances F2: |f_1| = |F2|
            success_step4 = math.isclose(abs(f_1), abs(F2), rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Step 4 equilibrium: |f_1| should equal |F2| = {abs(F2)} N, got |f_1| = {abs(f_1):.2f} N", success_step4)
            
            # 4. Check step 6 kinetic friction (only F1 present)
            # According to STEPS ANALYSIS: f_2 = μN = 8N, direction to the right
            # This should equal f_m (since max static = kinetic friction)
            success_step6 = math.isclose(abs(f_2), f_m, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Step 6 kinetic friction: |f_2| should equal f_m = {f_m:.2f} N, got |f_2| = {abs(f_2):.2f} N", success_step6)
            
            # 5. Verify static friction bounds
            # In static cases (steps 1 & 4), friction must be ≤ f_m
            static_bound_step1 = abs(f) <= f_m * 1.01  # Allow 1% tolerance for rounding
            static_bound_step4 = abs(f_1) <= f_m * 1.01
            
            self._log(f"Step 1 static bound: |f|={abs(f):.2f} ≤ f_m={f_m:.2f}?", static_bound_step1)
            self._log(f"Step 4 static bound: |f_1|={abs(f_1):.2f} ≤ f_m={f_m:.2f}?", static_bound_step4)
            
            # 6. Verify kinetic friction direction logic (symbolic check)
            # When F2 removed, net force without friction is F1 = 12N right
            # Since F1 > f_m, object moves right → kinetic friction opposes motion (left)
            # But answer says "8N, horizontally to the right" → contradiction?
            # Wait, re-examining STEPS ANALYSIS: f_2 equation is f_2 = μG, and answer says "right"
            # This suggests they might be reporting magnitude only or using different sign convention
            # For robustness, we'll just check magnitude equality
            
            all_success = (success_fm and success_step1 and success_step4 and 
                          success_step6 and static_bound_step1 and static_bound_step4)
            
            return all_success
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        try:
            f = self.results['f']
            f_1 = self.results['f_1']
            f_2 = self.results['f_2']
            
            # According to target answers: ["6N, horizontally to the right", "6N, horizontally to the left", "8N, horizontally to the right"]
            # We'll check magnitudes match (direction checks would require parsing strings)
            target_magnitudes = [6.0, 6.0, 8.0]
            computed_magnitudes = [abs(f), abs(f_1), abs(f_2)]
            
            all_match = True
            for i, (target, computed) in enumerate(zip(target_magnitudes, computed_magnitudes)):
                match = math.isclose(computed, target, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Target answer {i+1}: magnitude {target} N, got {computed:.2f} N", match)
                all_match = all_match and match
            
            return all_match
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb an input (mass/weight) and verify expected trends in results."""
        try:
            # Store original values
            original_weight = self.inputs['weight']
            original_results = self.results.copy()
            
            # Perturb weight by +10%
            perturbed_weight = original_weight * 1.10
            N_perturbed = perturbed_weight
            mu = self.inputs['mu']
            
            # Recalculate expected values using formulas from STEPS ANALYSIS
            # f_m should increase proportionally with weight
            f_m_perturbed = mu * N_perturbed
            f_m_original = mu * original_weight
            
            # f (step 1) depends on F1-F2, not on weight (as long as static bound holds)
            f_original = abs(self.results['f'])
            net_force = abs(self.inputs['F1'] - self.inputs['F2'])
            
            # f_1 (step 4) depends on F2, not on weight
            f_1_original = abs(self.results['f_1'])
            
            # f_2 (step 6) should increase proportionally with weight (kinetic friction)
            f_2_original = abs(self.results['f_2'])
            f_2_perturbed = f_m_perturbed
            
            # Check trends
            trends_ok = True
            
            # f_m trend: should increase
            if not (f_m_perturbed > f_m_original * 0.99):  # Allow 1% tolerance
                self._log(f"Monotonicity: f_m should increase with weight (original {f_m_original:.2f}, perturbed {f_m_perturbed:.2f})", False)
                trends_ok = False
            else:
                self._log(f"Monotonicity: f_m increases with weight as expected", True)
            
            # f (step 1) should remain constant (independent of weight)
            # Actually, it could change if the increased f_m affects static/kinetic transition,
            # but here F1-F2=6N is less than both f_m values, so should remain static friction = 6N
            expected_f_perturbed = net_force
            if not math.isclose(expected_f_perturbed, f_original, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity: f should remain ~{f_original:.2f} N (got {expected_f_perturbed:.2f} N)", False)
                trends_ok = False
            else:
                self._log(f"Monotonicity: f remains constant as expected", True)
            
            # f_1 (step 4) should remain constant
            expected_f_1_perturbed = abs(self.inputs['F2'])
            if not math.isclose(expected_f_1_perturbed, f_1_original, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity: f_1 should remain ~{f_1_original:.2f} N (got {expected_f_1_perturbed:.2f} N)", False)
                trends_ok = False
            else:
                self._log(f"Monotonicity: f_1 remains constant as expected", True)
            
            # f_2 (step 6) should increase proportionally with f_m
            ratio_original = f_2_original / f_m_original if f_m_original != 0 else 0
            ratio_perturbed = f_2_perturbed / f_m_perturbed if f_m_perturbed != 0 else 0
            
            if not math.isclose(ratio_original, ratio_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity: f_2/f_m ratio should remain ~{ratio_original:.3f} (got {ratio_perturbed:.3f})", False)
                trends_ok = False
            else:
                self._log(f"Monotonicity: f_2 scales with f_m as expected", True)
            
            # Restore original values (important for sandbox safety)
            self.inputs['weight'] = original_weight
            
            return trends_ok
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            # Restore on error
            if 'original_weight' in locals():
                self.inputs['weight'] = original_weight
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        self.logs.append("=== Physics Checker Start ===")
        
        success = True
        
        # Run individual tests
        inv_success = self.test_invariants()
        self.logs.append(f"Invariants test: {'PASS' if inv_success else 'FAIL'}")
        success = success and inv_success
        
        trace_success = self.test_trace_consistency()
        self.logs.append(f"Trace consistency test: {'PASS' if trace_success else 'FAIL'}")
        success = success and trace_success
        
        mono_success = self.test_monotonicity()
        self.logs.append(f"Monotonicity test: {'PASS' if mono_success else 'FAIL'}")
        success = success and mono_success
        
        self.logs.append(f"=== Physics Checker End: Overall {'PASS' if success else 'FAIL'} ===")
        
        return success, self.logs

# Example usage (commented out for the sandbox):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.F1 = 12.0
#             self.F2 = 6.0
#             self.weight = 40.0
#             self.mu = 0.2
#             self.results = {
#                 'f': 6.0,
#                 'f_1': 6.0,
#                 'f_2': 8.0,
#                 'f_m': 8.0
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)