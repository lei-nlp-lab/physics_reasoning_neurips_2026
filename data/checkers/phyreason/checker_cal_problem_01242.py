import math
import sympy as sp
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all needed inputs and results from solution_instance.
        Never access solution_instance.attr directly.
        """
        # Input parameters (provided in problem statement)
        self.L = 10.0          # m, distance AB
        self.v_belt = 2.0      # m/s, conveyor belt speed
        self.R = 4.0           # m, radius of turntable
        self.mu1 = 0.1         # coefficient of friction on belt
        self.mu2 = 0.4         # coefficient of friction on turntable
        self.g = 10.0          # m/s², gravity
        self.pi = 3.14         # given value of π
        
        # Results extraction with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Key results for total time calculation (first answer)
        self.v = getattr(solution_instance, 'v', self.results.get('v', None))
        self.t1 = getattr(solution_instance, 't₁', self.results.get('t₁', None))
        self.t2 = getattr(solution_instance, 't₂', self.results.get('t₂', None))
        self.t3 = getattr(solution_instance, 't₃', self.results.get('t₃', None))
        self.t_total = getattr(solution_instance, 't', self.results.get('t', None))
        self.a_belt = getattr(solution_instance, 'a', self.results.get('a', None))
        self.s_acc = getattr(solution_instance, 's', self.results.get('s', None))
        
        # Key results for maximum speed (second answer)
        self.v2 = getattr(solution_instance, 'v₂', self.results.get('v₂', None))
        self.v_adjusted = getattr(solution_instance, 'v_adjusted', self.results.get('v_adjusted', None))
        self.v1_max = getattr(solution_instance, 'v₁', self.results.get('v₁', None))
        
        # Log storage
        self.logs = []
        
    def _log(self, message: str, success: bool = None):
        """Add a message to logs with optional success flag."""
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(f"  {message}")
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physical laws:
        1. Newton's 2nd Law on conveyor belt (μ₁mg = ma)
        2. Newton's 2nd Law on turntable (μ₂mg = ma)
        3. Centripetal acceleration relation (a_c = v²/R)
        """
        success = True
        
        # 1. Check acceleration on conveyor belt
        expected_a_belt = self.mu1 * self.g  # μ₁g
        if self.a_belt is not None:
            if not math.isclose(self.a_belt, expected_a_belt, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Acceleration on belt mismatch: expected {expected_a_belt:.3f}, got {self.a_belt:.3f}", False)
                success = False
            else:
                self._log(f"Belt acceleration correct: {self.a_belt:.3f} m/s²", True)
        else:
            self._log("Warning: Belt acceleration not provided", None)
        
        # 2. Check maximum friction acceleration on turntable
        expected_a_turntable_max = self.mu2 * self.g  # μ₂g
        # This is the maximum tangential acceleration the turntable can provide without slipping
        self._log(f"Maximum turntable acceleration (μ₂g): {expected_a_turntable_max:.3f} m/s²", None)
        
        # 3. Verify centripetal relation for maximum speed
        if self.v2 is not None:
            expected_v2 = math.sqrt(self.mu2 * self.g * self.R)  # √(μ₂gR)
            if not math.isclose(self.v2, expected_v2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Maximum speed v₂ mismatch: expected {expected_v2:.3f}, got {self.v2:.3f}", False)
                success = False
            else:
                self._log(f"Maximum speed v₂ correct: {self.v2:.3f} m/s", True)
            
            # Verify v_adjusted equals v2
            if self.v_adjusted is not None and not math.isclose(self.v_adjusted, self.v2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v_adjusted ({self.v_adjusted:.3f}) ≠ v₂ ({self.v2:.3f})", False)
                success = False
            elif self.v_adjusted is not None:
                self._log(f"v_adjusted matches v₂: {self.v_adjusted:.3f} m/s", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Based on the steps analysis.
        """
        success = True
        
        # Part 1: Total time calculation (first answer: 12.28 s)
        if all(x is not None for x in [self.t1, self.t2, self.t3, self.t_total]):
            # Check t1 = v/a_belt
            if self.a_belt is not None:
                expected_t1 = self.v_belt / self.a_belt
                if not math.isclose(self.t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"t1 mismatch: expected v/a = {expected_t1:.3f}, got {self.t1:.3f}", False)
                    success = False
                else:
                    self._log(f"t1 correct: {self.t1:.3f} s", True)
            
            # Check distance during acceleration: s = 0.5*a*t1²
            if self.a_belt is not None and self.s_acc is not None:
                expected_s = 0.5 * self.a_belt * self.t1**2
                if not math.isclose(self.s_acc, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Acceleration distance mismatch: expected {expected_s:.3f}, got {self.s_acc:.3f}", False)
                    success = False
                else:
                    self._log(f"Acceleration distance correct: {self.s_acc:.3f} m", True)
            
            # Check t2 = (L - s)/v
            if self.s_acc is not None:
                expected_t2 = (self.L - self.s_acc) / self.v_belt
                if not math.isclose(self.t2, expected_t2, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"t2 mismatch: expected (L-s)/v = {expected_t2:.3f}, got {self.t2:.3f}", False)
                    success = False
                else:
                    self._log(f"t2 correct: {self.t2:.3f} s", True)
            
            # Check t3 = πR/v
            expected_t3 = self.pi * self.R / self.v_belt
            if not math.isclose(self.t3, expected_t3, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"t3 mismatch: expected πR/v = {expected_t3:.3f}, got {self.t3:.3f}", False)
                success = False
            else:
                self._log(f"t3 correct: {self.t3:.3f} s", True)
            
            # Check total time sum
            expected_total = self.t1 + self.t2 + self.t3
            if not math.isclose(self.t_total, expected_total, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Total time mismatch: expected sum = {expected_total:.3f}, got {self.t_total:.3f}", False)
                success = False
            else:
                self._log(f"Total time correct: {self.t_total:.3f} s", True)
        
        # Part 2: Maximum speed verification
        if self.v1_max is not None and self.a_belt is not None:
            # v1_max from acceleration over full belt length: v₁² = 2aL
            expected_v1_max = math.sqrt(2 * self.a_belt * self.L)
            if not math.isclose(self.v1_max, expected_v1_max, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Maximum achievable speed on belt mismatch: expected √(2aL) = {expected_v1_max:.3f}, got {self.v1_max:.3f}", False)
                success = False
            else:
                self._log(f"Maximum achievable speed on belt correct: {self.v1_max:.3f} m/s", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check if trends match physical intuition.
        We'll test two perturbations:
        1. Increase μ1 → should decrease total time (faster acceleration)
        2. Increase μ2 → should increase maximum allowed speed v2
        """
        success = True
        
        # Test 1: Effect of μ1 on total time
        original_mu1 = self.mu1
        perturbed_mu1 = original_mu1 * 1.1  # Increase by 10%
        
        # Recalculate with perturbed μ1 (keeping other parameters fixed)
        a_belt_new = perturbed_mu1 * self.g
        t1_new = self.v_belt / a_belt_new
        s_new = 0.5 * a_belt_new * t1_new**2
        t2_new = (self.L - s_new) / self.v_belt
        t3_new = self.pi * self.R / self.v_belt  # unchanged
        total_new = t1_new + t2_new + t3_new
        
        # Original total time (if provided, else compute from original μ1)
        if self.t_total is not None:
            original_total = self.t_total
        else:
            a_orig = original_mu1 * self.g
            t1_orig = self.v_belt / a_orig
            s_orig = 0.5 * a_orig * t1_orig**2
            t2_orig = (self.L - s_orig) / self.v_belt
            t3_orig = self.pi * self.R / self.v_belt
            original_total = t1_orig + t2_orig + t3_orig
        
        # Expect total time to decrease when μ1 increases (faster acceleration)
        if total_new < original_total:
            self._log(f"Monotonicity check 1 (μ1↑ → time↓): PASS (new μ1={perturbed_mu1:.3f}, new time={total_new:.3f} < original={original_total:.3f})", True)
        else:
            self._log(f"Monotonicity check 1 (μ1↑ → time↓): FAIL (new time={total_new:.3f} ≥ original={original_total:.3f})", False)
            success = False
        
        # Test 2: Effect of μ2 on maximum allowed speed v2
        original_mu2 = self.mu2
        perturbed_mu2 = original_mu2 * 1.1  # Increase by 10%
        
        # Recalculate v2 = √(μ₂gR)
        v2_original = math.sqrt(original_mu2 * self.g * self.R)
        v2_new = math.sqrt(perturbed_mu2 * self.g * self.R)
        
        # Expect v2 to increase when μ2 increases (more friction allows higher speed)
        if v2_new > v2_original:
            self._log(f"Monotonicity check 2 (μ2↑ → v2↑): PASS (new μ2={perturbed_mu2:.3f}, new v2={v2_new:.3f} > original={v2_original:.3f})", True)
        else:
            self._log(f"Monotonicity check 2 (μ2↑ → v2↑): FAIL (new v2={v2_new:.3f} ≤ original={v2_original:.3f})", False)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = ["Physics Checker Report:"]
        overall_success = True
        
        try:
            # Run each test
            if not self.test_invariants():
                overall_success = False
            self._log("", None)
            
            if not self.test_trace_consistency():
                overall_success = False
            self._log("", None)
            
            if not self.test_monotonicity():
                overall_success = False
            
            # Final summary
            self._log("", None)
            if overall_success:
                self._log("All physics checks passed!", True)
            else:
                self._log("Some physics checks failed.", False)
                
        except Exception as e:
            self._log(f"Error during testing: {str(e)}", False)
            self._log(traceback.format_exc(), False)
            overall_success = False
        
        return overall_success, self.logs