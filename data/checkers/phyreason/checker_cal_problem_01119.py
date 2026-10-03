import sympy as sp
import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance
        self.M = getattr(solution_instance, 'M', 1.0)  # kg
        self.m = getattr(solution_instance, 'm', 1.0)  # kg
        self.mu1 = getattr(solution_instance, 'mu1', 0.2)
        self.mu2 = getattr(solution_instance, 'mu2', 0.3)
        self.F = getattr(solution_instance, 'F', 11.0)  # N
        self.t0 = getattr(solution_instance, 't0', 1.0)  # s
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²
        
        # Safely extract results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        
        # Map keys from problem to safe names
        self.a5 = results.get('a_5', 0.0)
        self.a1 = results.get('a_1', 0.0)
        self.a0_prime = results.get('a_0\'', results.get('a_0_prime', 0.0))
        self.x1 = results.get('x_1', 0.0)
        self.L = results.get('L', 0.0)
        self.t2 = results.get('t_2', 0.0)
        self.Deltax1 = results.get('Deltax_1', 0.0)
        self.v_all = results.get('v_all', 0.0)
        self.F_f = results.get('F_f', 0.0)
        self.t_object = results.get('t_object', 0.0)
        self.F_fm = results.get('F_fm', 0.0)
        self.Deltax2 = results.get('Deltax_2', 0.0)
        self.a6 = results.get('a_6', 0.0)
        self.t3 = results.get('t_3', 0.0)
        self.a2 = results.get('a_2', 0.0)
        self.x2 = results.get('x_2', 0.0)
        self.t1 = results.get('t_1', 0.0)
        self.x5 = results.get('x_5', 0.0)
        self.v2 = results.get('v_2', 0.0)
        self.x3 = results.get('x_3', 0.0)
        self.x4 = results.get('x_4', 0.0)
        self.t_board = results.get('t_board', 0.0)
        self.a4 = results.get('a_4', 0.0)
        self.a0 = results.get('a_0', 0.0)
        self.x6 = results.get('x_6', 0.0)
        self.v1 = results.get('v_1', 0.0)
        
        # For convenience, define total mass
        self.M_total = self.M + self.m
        
        # Store logs
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(f"  {message}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Safe numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify Newton's Laws and conservation principles."""
        success = True
        
        # 1. Check step1: F - μ₂(M+m)g = (M+m)a₀
        expected_a0 = (self.F - self.mu2 * self.M_total * self.g) / self.M_total
        if not self._is_close(self.a0, expected_a0):
            self._log(f"Step1: a0 expected {expected_a0:.3f}, got {self.a0:.3f}", False)
            success = False
        else:
            self._log(f"Step1: a0 = {self.a0:.3f} matches Newton's 2nd Law", True)
        
        # 2. Check step2: μ₁mg = ma₁ → a₁ = μ₁g
        expected_a1 = self.mu1 * self.g
        if not self._is_close(self.a1, expected_a1):
            self._log(f"Step2: a1 expected {expected_a1:.3f}, got {self.a1:.3f}", False)
            success = False
        else:
            self._log(f"Step2: a1 = {self.a1:.3f} matches friction law", True)
        
        # 3. Check step3: F - μ₂(M+m)g - μ₁mg = Ma₂
        expected_a2 = (self.F - self.mu2 * self.M_total * self.g - self.mu1 * self.m * self.g) / self.M
        if not self._is_close(self.a2, expected_a2):
            self._log(f"Step3: a2 expected {expected_a2:.3f}, got {self.a2:.3f}", False)
            success = False
        else:
            self._log(f"Step3: a2 = {self.a2:.3f} matches Newton's 2nd Law", True)
        
        # 4. Check step4: v₁ = a₁t₀
        expected_v1 = self.a1 * self.t0
        if not self._is_close(self.v1, expected_v1):
            self._log(f"Step4: v1 expected {expected_v1:.3f}, got {self.v1:.3f}", False)
            success = False
        else:
            self._log(f"Step4: v1 = {self.v1:.3f} matches kinematics", True)
        
        # 5. Check step5: v₂ = a₂t₀
        expected_v2 = self.a2 * self.t0
        if not self._is_close(self.v2, expected_v2):
            self._log(f"Step5: v2 expected {expected_v2:.3f}, got {self.v2:.3f}", False)
            success = False
        else:
            self._log(f"Step5: v2 = {self.v2:.3f} matches kinematics", True)
        
        # 6. Check step7: μ₂(M+m)g + μ₁mg = Ma₄
        expected_a4 = (self.mu2 * self.M_total * self.g + self.mu1 * self.m * self.g) / self.M
        if not self._is_close(self.a4, expected_a4):
            self._log(f"Step7: a4 expected {expected_a4:.3f}, got {self.a4:.3f}", False)
            success = False
        else:
            self._log(f"Step7: a4 = {self.a4:.3f} matches Newton's 2nd Law", True)
        
        # 7. Check step9: μ₂(M+m)g = (M+m)a₀'
        expected_a0_prime = self.mu2 * self.g
        if not self._is_close(self.a0_prime, expected_a0_prime):
            self._log(f"Step9: a0' expected {expected_a0_prime:.3f}, got {self.a0_prime:.3f}", False)
            success = False
        else:
            self._log(f"Step9: a0' = {self.a0_prime:.3f} matches friction law", True)
        
        # 8. Check step10: F_f = ma₀'
        expected_F_f = self.m * self.a0_prime
        if not self._is_close(self.F_f, expected_F_f):
            self._log(f"Step10: F_f expected {expected_F_f:.3f}, got {self.F_f:.3f}", False)
            success = False
        else:
            self._log(f"Step10: F_f = {self.F_f:.3f} matches Newton's 2nd Law", True)
        
        # 9. Check step11: F_fm = μ₁mg
        expected_F_fm = self.mu1 * self.m * self.g
        if not self._is_close(self.F_fm, expected_F_fm):
            self._log(f"Step11: F_fm expected {expected_F_fm:.3f}, got {self.F_fm:.3f}", False)
            success = False
        else:
            self._log(f"Step11: F_fm = {self.F_fm:.3f} matches friction formula", True)
        
        # 10. Check step12: a₅ = μ₁g
        expected_a5 = self.mu1 * self.g
        if not self._is_close(self.a5, expected_a5):
            self._log(f"Step12: a5 expected {expected_a5:.3f}, got {self.a5:.3f}", False)
            success = False
        else:
            self._log(f"Step12: a5 = {self.a5:.3f} matches friction law", True)
        
        # 11. Check step15: μ₂(M+m)g - μ₁mg = Ma₆
        expected_a6 = (self.mu2 * self.M_total * self.g - self.mu1 * self.m * self.g) / self.M
        if not self._is_close(self.a6, expected_a6):
            self._log(f"Step15: a6 expected {expected_a6:.3f}, got {self.a6:.3f}", False)
            success = False
        else:
            self._log(f"Step15: a6 = {self.a6:.3f} matches Newton's 2nd Law", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        success = True
        
        # 1. Check step8: v_all = v₁ + a₁*t₁ (using a₁ as a₃ = μ₁g = 2 m/s²)
        # During t₁, block accelerates at a₁ = 2 m/s² (same as step2)
        expected_v_all_from_block = self.v1 + self.a1 * self.t1
        expected_v_all_from_board = self.v2 - self.a4 * self.t1
        if not (self._is_close(self.v_all, expected_v_all_from_block) and 
                self._is_close(self.v_all, expected_v_all_from_board)):
            self._log(f"Step8: v_all inconsistent. From block: {expected_v_all_from_block:.3f}, "
                     f"from board: {expected_v_all_from_board:.3f}, got {self.v_all:.3f}", False)
            success = False
        else:
            self._log(f"Step8: v_all = {self.v_all:.3f} consistent from both equations", True)
        
        # 2. Check step13: t₂ = v_all / a₅
        expected_t2 = self.v_all / self.a5
        if not self._is_close(self.t2, expected_t2):
            self._log(f"Step13: t2 expected {expected_t2:.3f}, got {self.t2:.3f}", False)
            success = False
        else:
            self._log(f"Step13: t2 = {self.t2:.3f} matches kinematics", True)
        
        # 3. Check step14: t_object = t₀ + t₁ + t₂
        expected_t_object = self.t0 + self.t1 + self.t2
        if not self._is_close(self.t_object, expected_t_object):
            self._log(f"Step14: t_object expected {expected_t_object:.3f}, got {self.t_object:.3f}", False)
            success = False
        else:
            self._log(f"Step14: t_object = {self.t_object:.3f} consistent", True)
        
        # 4. Check step16: t₃ = v_all / a₆
        expected_t3 = self.v_all / self.a6
        if not self._is_close(self.t3, expected_t3):
            self._log(f"Step16: t3 expected {expected_t3:.3f}, got {self.t3:.3f}", False)
            success = False
        else:
            self._log(f"Step16: t3 = {self.t3:.3f} matches kinematics", True)
        
        # 5. Check step17: t_board = t₀ + t₁ + t₃
        expected_t_board = self.t0 + self.t1 + self.t3
        if not self._is_close(self.t_board, expected_t_board):
            self._log(f"Step17: t_board expected {expected_t_board:.3f}, got {self.t_board:.3f}", False)
            success = False
        else:
            self._log(f"Step17: t_board = {self.t_board:.3f} consistent", True)
        
        # 6. Check step18: x₁ = (v₁/2)*t₀
        expected_x1 = (self.v1 / 2) * self.t0
        if not self._is_close(self.x1, expected_x1):
            self._log(f"Step18: x1 expected {expected_x1:.3f}, got {self.x1:.3f}", False)
            success = False
        else:
            self._log(f"Step18: x1 = {self.x1:.3f} matches kinematics", True)
        
        # 7. Check step19: x₂ = (v₂/2)*t₀
        expected_x2 = (self.v2 / 2) * self.t0
        if not self._is_close(self.x2, expected_x2):
            self._log(f"Step19: x2 expected {expected_x2:.3f}, got {self.x2:.3f}", False)
            success = False
        else:
            self._log(f"Step19: x2 = {self.x2:.3f} matches kinematics", True)
        
        # 8. Check step20: x₃ = (v₁ + v_all)/2 * t₁
        expected_x3 = (self.v1 + self.v_all) / 2 * self.t1
        if not self._is_close(self.x3, expected_x3):
            self._log(f"Step20: x3 expected {expected_x3:.3f}, got {self.x3:.3f}", False)
            success = False
        else:
            self._log(f"Step20: x3 = {self.x3:.3f} matches kinematics", True)
        
        # 9. Check step21: x₄ = (v₂ + v_all)/2 * t₁
        expected_x4 = (self.v2 + self.v_all) / 2 * self.t1
        if not self._is_close(self.x4, expected_x4):
            self._log(f"Step21: x4 expected {expected_x4:.3f}, got {self.x4:.3f}", False)
            success = False
        else:
            self._log(f"Step21: x4 = {self.x4:.3f} matches kinematics", True)
        
        # 10. Check step22: Δx₁ = x₂ + x₄ - (x₁ + x₃)
        expected_Deltax1 = self.x2 + self.x4 - (self.x1 + self.x3)
        if not self._is_close(self.Deltax1, expected_Deltax1):
            self._log(f"Step22: Δx1 expected {expected_Deltax1:.3f}, got {self.Deltax1:.3f}", False)
            success = False
        else:
            self._log(f"Step22: Δx1 = {self.Deltax1:.3f} consistent", True)
        
        # 11. Check step23: x₅ = v_all²/(2a₅)
        expected_x5 = self.v_all**2 / (2 * self.a5)
        if not self._is_close(self.x5, expected_x5):
            self._log(f"Step23: x5 expected {expected_x5:.3f}, got {self.x5:.3f}", False)
            success = False
        else:
            self._log(f"Step23: x5 = {self.x5:.3f} matches kinematics", True)
        
        # 12. Check step24: x₆ = v_all²/(2a₆)
        expected_x6 = self.v_all**2 / (2 * self.a6)
        if not self._is_close(self.x6, expected_x6):
            self._log(f"Step24: x6 expected {expected_x6:.3f}, got {self.x6:.3f}", False)
            success = False
        else:
            self._log(f"Step24: x6 = {self.x6:.3f} matches kinematics", True)
        
        # 13. Check step25: Δx₂ = x₅ - x₆
        expected_Deltax2 = self.x5 - self.x6
        if not self._is_close(self.Deltax2, expected_Deltax2):
            self._log(f"Step25: Δx2 expected {expected_Deltax2:.3f}, got {self.Deltax2:.3f}", False)
            success = False
        else:
            self._log(f"Step25: Δx2 = {self.Deltax2:.3f} consistent", True)
        
        # 14. Check step26: L = Δx₂
        if not self._is_close(self.L, self.Deltax2):
            self._log(f"Step26: L expected {self.Deltax2:.3f}, got {self.L:.3f}", False)
            success = False
        else:
            self._log(f"Step26: L = {self.L:.3f} matches Δx2", True)
        
        # Final check: Target answers from problem
        target_v1_v2 = f"${self.v1:.1f}\\mathrm{{m/s}}$, ${self.v2:.1f}\\mathrm{{m/s}}$"
        target_times = f"${self.t_object:.2f}\\mathrm{{s}}$, ${self.t_board:.2f}\\mathrm{{s}}$"
        target_L = f"${self.L:.3f}\\mathrm{{m}}$"
        
        expected_v1_v2 = "$2\\mathrm{m/s}$, $3\\mathrm{m/s}$"
        expected_times = "$2.2\\mathrm{s}$, $1.65\\mathrm{s}$"
        expected_L = "$0.605\\mathrm{m}$"
        
        if not (self._is_close(self.v1, 2.0) and self._is_close(self.v2, 3.0)):
            self._log(f"Target v1,v2 mismatch: expected {expected_v1_v2}, got {target_v1_v2}", False)
            success = False
        else:
            self._log(f"Target v1,v2 matches: {target_v1_v2}", True)
        
        if not (self._is_close(self.t_object, 2.2) and self._is_close(self.t_board, 1.65)):
            self._log(f"Target times mismatch: expected {expected_times}, got {target_times}", False)
            success = False
        else:
            self._log(f"Target times matches: {target_times}", True)
        
        if not self._is_close(self.L, 0.605):
            self._log(f"Target L mismatch: expected {expected_L}, got {target_L}", False)
            success = False
        else:
            self._log(f"Target L matches: {target_L}", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb an input and check trend, then restore."""
        success = True
        
        # Perturb μ₁ by +10%
        mu1_original = self.mu1
        mu1_perturbed = mu1_original * 1.10
        
        # Recalculate key quantity: F_fm = μ₁mg
        F_fm_original = mu1_original * self.m * self.g
        F_fm_perturbed = mu1_perturbed * self.m * self.g
        
        # Check monotonicity: increasing μ₁ should increase F_fm
        if F_fm_perturbed <= F_fm_original:
            self._log(f"Monotonicity test failed: F_fm should increase when μ₁ increases", False)
            success = False
        else:
            self._log(f"Monotonicity: μ₁↑ {mu1_original:.3f}→{mu1_perturbed:.3f} "
                     f"⇒ F_fm↑ {F_fm_original:.3f}→{F_fm_perturbed:.3f}", True)
        
        # Perturb F by +10%
        F_original = self.F
        F_perturbed = F_original * 1.10
        
        # Recalculate a₂ = [F - μ₂(M+m)g - μ₁mg] / M
        a2_original = (F_original - self.mu2 * self.M_total * self.g - self.mu1 * self.m * self.g) / self.M
        a2_perturbed = (F_perturbed - self.mu2 * self.M_total * self.g - self.mu1 * self.m * self.g) / self.M
        
        # Check monotonicity: increasing F should increase a₂
        if a2_perturbed <= a2_original:
            self._log(f"Monotonicity test failed: a2 should increase when F increases", False)
            success = False
        else:
            self._log(f"Monotonicity: F↑ {F_original:.3f}→{F_perturbed:.3f} "
                     f"⇒ a₂↑ {a2_original:.3f}→{a2_perturbed:.3f}", True)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs = []
        overall_success = True
        
        try:
            self._log("=== PhysicsChecker Starting ===")
            
            # Run tests
            tests = [
                ("Invariant Tests", self.test_invariants),
                ("Trace Consistency Tests", self.test_trace_consistency),
                ("Monotonicity Tests", self.test_monotonicity)
            ]
            
            for test_name, test_func in tests:
                self._log(f"\n--- Running {test_name} ---")
                try:
                    if not test_func():
                        overall_success = False
                except Exception as e:
                    self._log(f"Error in {test_name}: {str(e)}", False)
                    self._log(traceback.format_exc(), False)
                    overall_success = False
            
            self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
            
        except Exception as e:
            self._log(f"Critical error: {str(e)}", False)
            self._log(traceback.format_exc(), False)
            overall_success = False
        
        return overall_success, self.logs