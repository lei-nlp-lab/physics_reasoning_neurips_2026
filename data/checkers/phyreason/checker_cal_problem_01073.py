import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        
        Args:
            solution_instance: The student's solution object containing inputs and results.
        """
        # Safe extraction of input parameters with fallbacks
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'M', 1.0))
        self.mu = getattr(solution_instance, 'mu', 0.3)
        self.F = getattr(solution_instance, 'F', 6.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safe extraction of results using the provided key mapping
        self.results = getattr(solution_instance, 'results', {})
        self.a = self.results.get('a', 0.0)        # Initial acceleration
        self.x = self.results.get('x', 0.0)        # Distance to reach v=12 m/s
        self.v1 = self.results.get('v_1', 0.0)     # Velocity at t=4s
        self.a1 = self.results.get('a_1', 0.0)     # Deceleration after force removal
        self.x1 = self.results.get('x_1', 0.0)     # Additional displacement
        
        # Target answers from the problem
        self.target_a = 3.0     # Initial acceleration
        self.target_x = 24.0    # Distance to reach v=12 m/s
        self.target_x1 = 24.0   # Additional displacement
        
        # Store logs for all tests
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper method to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """
        Test fundamental physics invariants:
        1. Newton's Second Law during force application
        2. Newton's Second Law after force removal
        3. Kinematic consistency
        """
        self._log("Testing physics invariants...")
        
        # 1. Calculate friction force from parameters
        f_friction = self.mu * self.m * self.g
        
        # Test 1: Newton's Second Law during force application
        # F_net = F - f = m*a
        expected_a = (self.F - f_friction) / self.m
        newton2_success = self._is_close(self.a, expected_a)
        self._log(f"Newton's 2nd Law (force on): F_net = {self.F} - {f_friction} = {self.F-f_friction}, "
                  f"m*a = {self.m}*{self.a} = {self.m*self.a}. Expected a={expected_a}, got {self.a}",
                  newton2_success)
        
        # Test 2: Newton's Second Law after force removal
        # F_net = -f = m*a1
        expected_a1 = -f_friction / self.m
        newton2_off_success = self._is_close(self.a1, expected_a1)
        self._log(f"Newton's 2nd Law (force off): F_net = -{f_friction}, m*a1 = {self.m}*{self.a1} = {self.m*self.a1}. "
                  f"Expected a1={expected_a1}, got {self.a1}",
                  newton2_off_success)
        
        # Test 3: Kinematic consistency for v^2 = 2*a*x
        if self._is_close(self.a, 0.0, abs_tol=1e-6):
            kinematic_success = False
            self._log("Acceleration is zero, cannot apply v^2 = 2*a*x", False)
        else:
            expected_x = (12.0**2) / (2 * self.a)  # Using v=12 m/s from problem
            kinematic_success = self._is_close(self.x, expected_x)
            self._log(f"Kinematic: v²=2ax => 144=2*{self.a}*x => x={expected_x}, got {self.x}",
                      kinematic_success)
        
        return newton2_success and newton2_off_success and kinematic_success
    
    def test_trace_consistency(self):
        """
        Test consistency between intermediate results and final answers.
        Based on the steps analysis provided.
        """
        self._log("Testing trace consistency...")
        
        successes = []
        
        # Step 1: Check acceleration matches target
        step1_success = self._is_close(self.a, self.target_a)
        successes.append(step1_success)
        self._log(f"Step 1 (acceleration): target={self.target_a}, got={self.a}", step1_success)
        
        # Step 2: Check distance x matches target
        step2_success = self._is_close(self.x, self.target_x)
        successes.append(step2_success)
        self._log(f"Step 2 (distance to v=12m/s): target={self.target_x}, got={self.x}", step2_success)
        
        # Step 3: Check velocity at 4 seconds (should be 12 m/s)
        # v = a*t with t=4s
        expected_v1 = self.a * 4.0
        step3_success = self._is_close(self.v1, expected_v1)
        successes.append(step3_success)
        self._log(f"Step 3 (velocity at t=4s): v=a*t={self.a}*4={expected_v1}, got={self.v1}", step3_success)
        
        # Step 4: Check deceleration after force removal
        # a1 should be -μ*g
        expected_a1 = -self.mu * self.g
        step4_success = self._is_close(self.a1, expected_a1)
        successes.append(step4_success)
        self._log(f"Step 4 (deceleration): a1=-μ*g=-{self.mu}*{self.g}={expected_a1}, got={self.a1}", step4_success)
        
        # Step 5: Check additional displacement
        # 0 = v1² + 2*a1*x1 => x1 = -v1²/(2*a1)
        if self._is_close(self.a1, 0.0, abs_tol=1e-6):
            step5_success = False
            self._log("Deceleration is zero, cannot compute x1 from kinematic equation", False)
        else:
            expected_x1 = -(self.v1**2) / (2 * self.a1)
            step5_success = self._is_close(self.x1, expected_x1)
            successes.append(step5_success)
            self._log(f"Step 5 (additional displacement): x1=-v1²/(2*a1)=-{self.v1**2}/(2*{self.a1})={expected_x1}, got={self.x1}",
                      step5_success)
        
        return all(successes)
    
    def test_monotonicity(self):
        """
        Test that increasing mass decreases acceleration and increases stopping distance.
        Perturb mass by +10%, recalculate using physics formulas, check trends.
        CORRECTED: Deceleration magnitude |a1| = μ*g is independent of mass.
        """
        self._log("Testing monotonicity with mass perturbation...")
        
        original_m = self.m
        perturbed_m = original_m * 1.1  # +10%
        
        # Save original results
        original_a = self.a
        original_x = self.x
        original_x1 = self.x1
        
        try:
            # Recalculate with perturbed mass using physics formulas
            f_friction = self.mu * perturbed_m * self.g
            perturbed_a = (self.F - f_friction) / perturbed_m
            
            # Calculate perturbed x using v^2 = 2*a*x (with v=12 m/s)
            perturbed_x = (12.0**2) / (2 * perturbed_a) if perturbed_a > 0 else 0.0
            
            # Calculate perturbed v1 at t=4s
            perturbed_v1 = perturbed_a * 4.0
            
            # Calculate perturbed a1 after force removal
            perturbed_a1 = -f_friction / perturbed_m
            
            # Calculate perturbed x1
            perturbed_x1 = -(perturbed_v1**2) / (2 * perturbed_a1) if perturbed_a1 < 0 else 0.0
            
            # Check monotonic trends
            # 1. Larger mass should give smaller acceleration (since F-friction decreases proportionally)
            trend1_success = perturbed_a < original_a
            self._log(f"Mass increase from {original_m} to {perturbed_m}: "
                      f"a decreases from {original_a} to {perturbed_a}", trend1_success)
            
            # 2. Larger mass with smaller acceleration means longer distance to reach 12 m/s
            trend2_success = perturbed_x > original_x
            self._log(f"Mass increase: distance to v=12m/s increases from {original_x} to {perturbed_x}",
                      trend2_success)
            
            # 3. CORRECTED: Deceleration magnitude |a1| = μ*g is independent of mass, should stay constant
            trend3_success = self._is_close(abs(perturbed_a1), abs(self.a1))
            self._log(f"Mass increase: |deceleration| remains constant at μ*g = {self.mu*self.g}",
                      trend3_success)
            
            # 4. With smaller initial velocity (due to smaller a) but same deceleration magnitude,
            # stopping distance should decrease
            trend4_success = perturbed_x1 < original_x1
            self._log(f"Mass increase: stopping distance decreases from {original_x1} to {perturbed_x1}",
                      trend4_success)
            
            return trend1_success and trend2_success and trend3_success and trend4_success
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
        finally:
            # Restore original values
            self.m = original_m
    
    def test_target_answers(self):
        """Direct comparison with target answers from the problem."""
        self._log("Testing target answers...")
        
        successes = []
        
        # Check initial acceleration
        target1_success = self._is_close(self.a, self.target_a)
        successes.append(target1_success)
        self._log(f"Target 1 (acceleration): {self.target_a} m/s², got {self.a} m/s²", target1_success)
        
        # Check distance to reach v=12 m/s
        target2_success = self._is_close(self.x, self.target_x)
        successes.append(target2_success)
        self._log(f"Target 2 (distance to v=12m/s): {self.target_x} m, got {self.x} m", target2_success)
        
        # Check additional displacement after force removal
        target3_success = self._is_close(self.x1, self.target_x1)
        successes.append(target3_success)
        self._log(f"Target 3 (additional displacement): {self.target_x1} m, got {self.x1} m", target3_success)
        
        return all(successes)
    
    def run_all_tests(self):
        """
        Run all physics checks and return overall success and logs.
        
        Returns:
            Tuple of (success_bool, logs_list)
        """
        self.logs = []
        overall_success = True
        
        try:
            # Test 1: Physics invariants
            if not self.test_invariants():
                overall_success = False
                self._log("Physics invariants test failed")
            
            # Test 2: Trace consistency
            if not self.test_trace_consistency():
                overall_success = False
                self._log("Trace consistency test failed")
            
            # Test 3: Monotonicity
            if not self.test_monotonicity():
                overall_success = False
                self._log("Monotonicity test failed")
            
            # Test 4: Target answers
            if not self.test_target_answers():
                overall_success = False
                self._log("Target answers test failed")
            
        except Exception as e:
            self._log(f"Unexpected error during testing: {str(e)}\n{traceback.format_exc()}", False)
            overall_success = False
        
        # Summary
        if overall_success:
            self._log("All physics checks passed!", True)
        else:
            self._log("Some physics checks failed", False)
        
        return overall_success, self.logs