import math
import sympy as sp
from typing import Dict, Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with safe defaults
        self.R = getattr(solution_instance, 'R', 5.0)
        self.s = getattr(solution_instance, 's', 9.5)
        self.mu = getattr(solution_instance, 'mu', 0.1)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe fallbacks
        self.f = self.results.get('f', 0.0)
        self.n = self.results.get('n', 0)
        self.v_C = self.results.get('v_C', 0.0)
        self.v_B = self.results.get('v_B', 0.0)
        self.total_distance = self.results.get('total_distance', 0.0)
        self.final_rest_distance = self.results.get('final_rest_distance', 0.0)
        
        # Define symbolic variables for physics laws
        self.m = sp.symbols('m', positive=True)  # mass (will cancel out)
        self.v = sp.symbols('v', positive=True)
        self.h = sp.symbols('h', positive=True)
        self.f_friction = sp.symbols('f_friction')
        self.W_friction = sp.symbols('W_friction')
        self.W_gravity = sp.symbols('W_gravity')
        
        # Store test logs
        self.logs = []
        
    def _log(self, message: str):
        """Helper to log test messages"""
        self.logs.append(message)
    
    def test_invariants(self) -> bool:
        """
        Test fundamental physics invariants:
        1. Energy conservation from A to B (no friction on arc)
        2. Work-energy theorem from A to C (with friction)
        3. Friction force magnitude
        """
        success = True
        
        # 1. Energy conservation from A to B (smooth arc)
        # Potential energy at A = Kinetic energy at B
        theoretical_v_B = math.sqrt(2 * self.g * self.R)
        if not math.isclose(self.v_B, theoretical_v_B, rel_tol=0.02, abs_tol=0.02):
            self._log(f"FAIL: v_B = {self.v_B}, theoretical v_B = {theoretical_v_B}")
            success = False
        else:
            self._log(f"PASS: v_B matches energy conservation (v_B = {self.v_B})")
        
        # 2. Work-energy theorem from B to C
        # Kinetic energy at B - Work done by friction = Kinetic energy at C
        work_friction = self.mu * self.g * self.s  # f = μmg, W = f·s
        theoretical_v_C_squared = self.v_B**2 - 2 * work_friction
        theoretical_v_C = math.sqrt(max(0, theoretical_v_C_squared))
        
        if not math.isclose(self.v_C, theoretical_v_C, rel_tol=0.02, abs_tol=0.02):
            self._log(f"FAIL: v_C = {self.v_C}, theoretical v_C = {theoretical_v_C}")
            success = False
        else:
            self._log(f"PASS: v_C matches work-energy theorem (v_C = {self.v_C})")
        
        # 3. Friction force magnitude (if provided)
        if self.f > 0:
            theoretical_f = self.mu * self.g  # f = μmg, but mass cancels in ratios
            # Since f is force per unit mass in the steps
            if not math.isclose(self.f, self.mu * self.g, rel_tol=0.02, abs_tol=0.02):
                self._log(f"FAIL: f = {self.f}, theoretical f = {theoretical_f}")
                success = False
            else:
                self._log(f"PASS: Friction force magnitude correct")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Test consistency between intermediate results and final answers.
        Based on the steps analysis:
        - Step 1: v_B from energy conservation
        - Step 4: v_C from simultaneous equations
        - Step 5: Total distance until rest = 50m
        - Step 6: Number of collisions = 3
        - Step 7: Final rest position distance from B = 7m
        """
        success = True
        
        # Check step 1 consistency: v_B should be 10 m/s
        if not math.isclose(self.v_B, 10.0, rel_tol=0.02, abs_tol=0.02):
            self._log(f"FAIL: v_B inconsistent with step 1 (expected 10, got {self.v_B})")
            success = False
        
        # Check step 4 consistency: v_C should be 9 m/s
        if not math.isclose(self.v_C, 9.0, rel_tol=0.02, abs_tol=0.02):
            self._log(f"FAIL: v_C inconsistent with step 4 (expected 9, got {self.v_C})")
            success = False
        
        # Check step 5 consistency: total distance until rest
        theoretical_total_dist = self.R / self.mu  # mgh = f·s, h = R, f = μmg => R = μs => s = R/μ
        if not math.isclose(theoretical_total_dist, 50.0, rel_tol=0.02, abs_tol=0.02):
            self._log(f"FAIL: Total distance calculation inconsistent")
            success = False
        
        # Check step 6 consistency: number of collisions = 3
        if self.n != 3:
            self._log(f"FAIL: Number of collisions inconsistent (expected 3, got {self.n})")
            success = False
        
        # Check step 7 consistency: final rest position 7m from B
        # Distance traveled until last collision: 9.5 + 2*19 = 47.5m
        # Remaining distance: 50 - 47.5 = 2.5m from C
        # Distance from B: 9.5 - 2.5 = 7m
        if 'final_rest_distance' in self.results:
            if not math.isclose(self.final_rest_distance, 7.0, rel_tol=0.02, abs_tol=0.02):
                self._log(f"FAIL: Final rest distance inconsistent (expected 7, got {self.final_rest_distance})")
                success = False
        
        if success:
            self._log("PASS: All steps are internally consistent")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and check monotonic trends:
        1. Increase μ → decrease v_C, decrease number of collisions
        2. Increase R → increase v_B, increase total distance
        """
        success = True
        
        # Store original values
        original_mu = self.mu
        original_R = self.R
        
        # Test 1: Increase friction coefficient
        self.mu = original_mu * 1.5  # Increase by 50%
        
        # Recalculate v_C using the formula
        v_B_perturbed = math.sqrt(2 * self.g * original_R)
        v_C_perturbed = math.sqrt(max(0, v_B_perturbed**2 - 2 * self.mu * self.g * self.s))
        
        # Check monotonicity: higher μ should give lower v_C
        if v_C_perturbed >= self.v_C:
            self._log(f"FAIL: v_C should decrease when μ increases (original: {self.v_C}, perturbed: {v_C_perturbed})")
            success = False
        
        # Recalculate total distance
        total_dist_perturbed = original_R / self.mu
        
        # Check monotonicity: higher μ should give shorter total distance
        if total_dist_perturbed >= 50.0:  # Original total distance is 50m
            self._log(f"FAIL: Total distance should decrease when μ increases")
            success = False
        
        # Restore original μ
        self.mu = original_mu
        
        # Test 2: Increase radius R
        self.R = original_R * 1.2  # Increase by 20%
        
        # Recalculate v_B
        v_B_perturbed = math.sqrt(2 * self.g * self.R)
        
        # Check monotonicity: higher R should give higher v_B
        if v_B_perturbed <= self.v_B:
            self._log(f"FAIL: v_B should increase when R increases (original: {self.v_B}, perturbed: {v_B_perturbed})")
            success = False
        
        # Recalculate total distance
        total_dist_perturbed = self.R / original_mu
        
        # Check monotonicity: higher R should give longer total distance
        if total_dist_perturbed <= 50.0:  # Original total distance is 50m
            self._log(f"FAIL: Total distance should increase when R increases")
            success = False
        
        # Restore original R
        self.R = original_R
        
        if success:
            self._log("PASS: All monotonicity tests passed")
        
        return success
    
    def test_target_answers(self) -> bool:
        """Directly compare with target answers from the problem"""
        success = True
        target_answers = [10.0, 9.0, 3, 7.0]
        
        # Compare v_B
        if not math.isclose(self.v_B, target_answers[0], rel_tol=0.02, abs_tol=0.02):
            self._log(f"FAIL: v_B mismatch (target: {target_answers[0]}, got: {self.v_B})")
            success = False
        
        # Compare v_C
        if not math.isclose(self.v_C, target_answers[1], rel_tol=0.02, abs_tol=0.02):
            self._log(f"FAIL: v_C mismatch (target: {target_answers[1]}, got: {self.v_C})")
            success = False
        
        # Compare number of collisions
        if self.n != target_answers[2]:
            self._log(f"FAIL: n mismatch (target: {target_answers[2]}, got: {self.n})")
            success = False
        
        # Compare final rest distance (if available)
        if 'final_rest_distance' in self.results:
            if not math.isclose(self.final_rest_distance, target_answers[3], rel_tol=0.02, abs_tol=0.02):
                self._log(f"FAIL: final_rest_distance mismatch (target: {target_answers[3]}, got: {self.final_rest_distance})")
                success = False
        
        if success:
            self._log("PASS: All target answers match")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag with logs"""
        self.logs = []
        self.logs.append("=== Physics Checker Results ===")
        
        # Run individual tests
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Target Answers", self.test_target_answers)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
                    self.logs.append(f"❌ {test_name} FAILED")
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            self.logs.append("✅ All tests passed!")
        else:
            self.logs.append("❌ Some tests failed")
        
        return overall_success, self.logs