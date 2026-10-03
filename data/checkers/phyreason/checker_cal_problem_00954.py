import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs (with safe defaults)
        self.P_gen = getattr(solution_instance, 'P_gen', self.results.get('P_gen', 500e3))  # 500 kW in watts
        self.U1 = getattr(solution_instance, 'U1', self.results.get('U1', 500.0))  # Generator terminal voltage
        self.turns_ratio_step_up = getattr(solution_instance, 'turns_ratio_step_up', self.results.get('turns_ratio_step_up', 5.0))  # n2/n1 = 5
        self.R_line = getattr(solution_instance, 'R_line', self.results.get('R_line', 1.5))  # Line resistance
        self.U4 = getattr(solution_instance, 'U4', self.results.get('U4', 220.0))  # Step-down output voltage
        
        # Extract computed results (with safe defaults)
        self.U2 = self.results.get('U_2', 0.0)
        self.I = self.results.get('I', 0.0)
        self.DeltaP = self.results.get('DeltaP', 0.0)
        self.DeltaU = self.results.get('DeltaU', 0.0)
        self.U3 = self.results.get('U_3', 0.0)
        self.frac_n3_n4 = self.results.get('fracn_3n_4', 0.0)
        self.P4 = self.results.get('P_4', 0.0)
        
        # Store logs
        self.logs = []

    def log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")

    def test_invariants(self):
        """Check fundamental physical laws."""
        tests_passed = True
        
        try:
            # 1. Transformer ratio for step-up: U2/U1 = n2/n1
            expected_U2 = self.U1 * self.turns_ratio_step_up
            if not math.isclose(self.U2, expected_U2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Step-up transformer ratio mismatch. Expected U2={expected_U2:.1f}, got {self.U2:.1f}", False)
                tests_passed = False
            else:
                self.log(f"Step-up transformer ratio correct: {self.U2:.1f} V ≈ {self.U1} * {self.turns_ratio_step_up}", True)
            
            # 2. Power transmission: P_gen = U2 * I (ideal transformer)
            expected_I = self.P_gen / self.U2
            if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Current mismatch. Expected I={expected_I:.1f}, got {self.I:.1f}", False)
                tests_passed = False
            else:
                self.log(f"Current correct: {self.I:.1f} A ≈ {self.P_gen} / {self.U2}", True)
            
            # 3. Joule's law: ΔP = I² * R
            expected_DeltaP = self.I**2 * self.R_line
            if not math.isclose(self.DeltaP, expected_DeltaP, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Power loss mismatch. Expected ΔP={expected_DeltaP:.1f}, got {self.DeltaP:.1f}", False)
                tests_passed = False
            else:
                self.log(f"Power loss correct: {self.DeltaP:.1f} W ≈ ({self.I})² * {self.R_line}", True)
            
            # 4. Ohm's law: ΔU = I * R
            expected_DeltaU = self.I * self.R_line
            if not math.isclose(self.DeltaU, expected_DeltaU, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Voltage drop mismatch. Expected ΔU={expected_DeltaU:.1f}, got {self.DeltaU:.1f}", False)
                tests_passed = False
            else:
                self.log(f"Voltage drop correct: {self.DeltaU:.1f} V ≈ {self.I} * {self.R_line}", True)
            
            # 5. Voltage subtraction: U3 = U2 - ΔU
            expected_U3 = self.U2 - self.DeltaU
            if not math.isclose(self.U3, expected_U3, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Step-down primary voltage mismatch. Expected U3={expected_U3:.1f}, got {self.U3:.1f}", False)
                tests_passed = False
            else:
                self.log(f"Step-down primary voltage correct: {self.U3:.1f} V ≈ {self.U2} - {self.DeltaU}", True)
            
            # 6. Transformer ratio for step-down: n3/n4 = U3/U4
            expected_ratio = self.U3 / self.U4
            if not math.isclose(self.frac_n3_n4, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Step-down turns ratio mismatch. Expected {expected_ratio:.2f}, got {self.frac_n3_n4:.2f}", False)
                tests_passed = False
            else:
                self.log(f"Step-down turns ratio correct: {self.frac_n3_n4:.2f} ≈ {self.U3} / {self.U4}", True)
            
            # 7. Power conservation: P4 = P_gen - ΔP
            expected_P4 = self.P_gen - self.DeltaP
            if not math.isclose(self.P4, expected_P4, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"User power mismatch. Expected P4={expected_P4:.1f}, got {self.P4:.1f}", False)
                tests_passed = False
            else:
                self.log(f"User power correct: {self.P4:.1f} W ≈ {self.P_gen} - {self.DeltaP}", True)
                
        except Exception as e:
            self.log(f"Exception in test_invariants: {str(e)}", False)
            tests_passed = False
            
        return tests_passed

    def test_trace_consistency(self):
        """Verify intermediate results match final answer."""
        tests_passed = True
        
        # Check against the target answers (converted to appropriate units)
        target_values = [2500.0, 60000.0, 10.0, 440000.0]  # V, W, ratio, W
        
        try:
            # U2 should be 2500 V
            if not math.isclose(self.U2, target_values[0], rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"U2 mismatch with target. Expected {target_values[0]} V, got {self.U2} V", False)
                tests_passed = False
            else:
                self.log(f"U2 matches target: {self.U2} V ≈ {target_values[0]} V", True)
            
            # ΔP should be 60 kW = 60000 W
            if not math.isclose(self.DeltaP, target_values[1], rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"ΔP mismatch with target. Expected {target_values[1]} W, got {self.DeltaP} W", False)
                tests_passed = False
            else:
                self.log(f"ΔP matches target: {self.DeltaP} W ≈ {target_values[1]} W", True)
            
            # n3/n4 should be 10/1 = 10.0
            if not math.isclose(self.frac_n3_n4, target_values[2], rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"n3/n4 mismatch with target. Expected {target_values[2]}, got {self.frac_n3_n4}", False)
                tests_passed = False
            else:
                self.log(f"n3/n4 matches target: {self.frac_n3_n4} ≈ {target_values[2]}", True)
            
            # P4 should be 440 kW = 440000 W
            if not math.isclose(self.P4, target_values[3], rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"P4 mismatch with target. Expected {target_values[3]} W, got {self.P4} W", False)
                tests_passed = False
            else:
                self.log(f"P4 matches target: {self.P4} W ≈ {target_values[3]} W", True)
                
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {str(e)}", False)
            tests_passed = False
            
        return tests_passed

    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        tests_passed = True
        
        try:
            # Perturb transmission line resistance by +10%
            original_R = self.R_line
            perturbed_R = original_R * 1.10
            
            # Recalculate expected ΔP using Joule's law
            new_DeltaP = self.I**2 * perturbed_R
            
            # Trend: If R increases, ΔP should increase
            if new_DeltaP <= self.DeltaP:
                self.log(f"Monotonicity violated: ΔP did not increase when R increased (from {self.DeltaP:.1f} to {new_DeltaP:.1f})", False)
                tests_passed = False
            else:
                self.log(f"Monotonicity holds: ΔP increases from {self.DeltaP:.1f} to {new_DeltaP:.1f} when R increases", True)
            
            # Recalculate expected P4 using power conservation
            new_P4 = self.P_gen - new_DeltaP
            
            # Trend: If R increases, P4 should decrease
            if new_P4 >= self.P4:
                self.log(f"Monotonicity violated: P4 did not decrease when R increased (from {self.P4:.1f} to {new_P4:.1f})", False)
                tests_passed = False
            else:
                self.log(f"Monotonicity holds: P4 decreases from {self.P4:.1f} to {new_P4:.1f} when R increases", True)
                
            # Restore original R (conceptually)
            self.log(f"Original R restored: {original_R} Ω", None)
            
        except Exception as e:
            self.log(f"Exception in test_monotonicity: {str(e)}", False)
            tests_passed = False
            
        return tests_passed

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        all_passed = True
        
        self.log("Starting Physics Audit", None)
        
        # Run invariant tests
        self.log("\n--- Testing Physical Invariants ---", None)
        if not self.test_invariants():
            all_passed = False
            
        # Run trace consistency
        self.log("\n--- Testing Trace Consistency ---", None)
        if not self.test_trace_consistency():
            all_passed = False
            
        # Run monotonicity
        self.log("\n--- Testing Monotonicity ---", None)
        if not self.test_monotonicity():
            all_passed = False
            
        # Summary
        self.log(f"\n--- Audit Complete: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'} ---", None)
        
        return all_passed, self.logs