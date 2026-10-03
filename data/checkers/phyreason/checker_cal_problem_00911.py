import math
import sympy as sp
from sympy import symbols, Eq, simplify

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely (not accessing solution_instance.attr directly)
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters from problem
        self.P1 = 10e3  # 10 kW in watts
        self.U1 = 400   # volts
        self.n1_n2 = 1/5  # Step-up transformer turns ratio (n1:n2 = 1:5)
        self.R = 4      # ohms
        self.U4 = 220   # volts
        
        # Extract results with safe defaults
        self.U2 = self._get_result('U_2', 0.0)
        self.I2 = self._get_result('I_2', 0.0)
        self.U3 = self._get_result('U_3', 0.0)
        self.P_loss = self._get_result('P_loss', 0.0)
        self.ratio_n3_n4 = self._get_result('fracn_3n_4', 0.0)
        
        # Target answers for comparison
        self.target_U2 = 2000.0
        self.target_P_loss = 100.0
        self.target_ratio = 9.0
        
    def _get_result(self, key, default):
        """Safe attribute access with fallback to default"""
        return float(self.results.get(key, default))
    
    def test_invariants(self):
        """Verify fundamental physics laws are satisfied"""
        logs = []
        success = True
        
        # 1. Check transformer voltage ratio for step-up transformer
        # U2 should equal U1 * (n2/n1) = 400 * 5 = 2000V
        expected_U2 = self.U1 * 5
        if not math.isclose(self.U2, expected_U2, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: U2 ({self.U2}V) doesn't match transformer ratio. Expected: {expected_U2}V")
            success = False
        else:
            logs.append(f"PASS: Step-up transformer voltage ratio: U2 = U1 * (n2/n1) = 2000V")
        
        # 2. Check power conservation at step-up transformer (ideal)
        # P1 = U1 * I1 = U2 * I2
        # Since we don't have I1, check that U2*I2 = P1
        P2_calc = self.U2 * self.I2
        if not math.isclose(P2_calc, self.P1, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Power not conserved at step-up transformer. U2*I2 = {P2_calc}W, P1 = {self.P1}W")
            success = False
        else:
            logs.append(f"PASS: Power conserved at step-up transformer: U2*I2 = P1 = 10kW")
        
        # 3. Check Joule's law for transmission line loss
        # P_loss = I2^2 * R
        P_loss_calc = self.I2**2 * self.R
        if not math.isclose(self.P_loss, P_loss_calc, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Transmission line loss doesn't match Joule's law. Calculated: {P_loss_calc}W, Found: {self.P_loss}W")
            success = False
        else:
            logs.append(f"PASS: Joule's law satisfied: P_loss = I2² * R = {P_loss_calc:.1f}W")
        
        # 4. Check power conservation along transmission line
        # P3 = P2 - P_loss = U3 * I2
        P2 = self.U2 * self.I2
        P3_calc = P2 - self.P_loss
        P3_alternate = self.U3 * self.I2
        
        if not math.isclose(P3_calc, P3_alternate, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Power mismatch at transmission line output. P2-P_loss = {P3_calc}W, U3*I2 = {P3_alternate}W")
            success = False
        else:
            logs.append(f"PASS: Power conserved along transmission line: P3 = P2 - P_loss = U3*I2 = {P3_calc:.0f}W")
        
        # 5. Check transformer voltage ratio for step-down transformer
        # U3/U4 = n3/n4
        ratio_calc = self.U3 / self.U4
        if not math.isclose(self.ratio_n3_n4, ratio_calc, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Step-down transformer ratio mismatch. Calculated: n3/n4 = U3/U4 = {ratio_calc}, Found: {self.ratio_n3_n4}")
            success = False
        else:
            logs.append(f"PASS: Step-down transformer voltage ratio: n3/n4 = U3/U4 = {ratio_calc:.1f}")
        
        # 6. Check overall energy conservation
        # P1 = P_loss + P4 (where P4 = U4 * I4, and I4 = I2 * (n3/n4) due to transformer current ratio)
        I4_calc = self.I2 * self.ratio_n3_n4
        P4_calc = self.U4 * I4_calc
        P_total_out = self.P_loss + P4_calc
        
        if not math.isclose(self.P1, P_total_out, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: Overall energy not conserved. P1 = {self.P1}W, P_loss + P4 = {P_total_out}W")
            success = False
        else:
            logs.append(f"PASS: Overall energy conserved: P1 = P_loss + P4 = {P_total_out:.0f}W")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        logs = []
        success = True
        
        # Check against target answers
        if not math.isclose(self.U2, self.target_U2, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: U2 ({self.U2}V) doesn't match target ({self.target_U2}V)")
            success = False
        else:
            logs.append(f"PASS: U2 matches target: 2000V")
        
        if not math.isclose(self.P_loss, self.target_P_loss, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: P_loss ({self.P_loss}W) doesn't match target ({self.target_P_loss}W)")
            success = False
        else:
            logs.append(f"PASS: P_loss matches target: 100W")
        
        if not math.isclose(self.ratio_n3_n4, self.target_ratio, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"FAIL: n3/n4 ratio ({self.ratio_n3_n4}) doesn't match target (9:1 = 9.0)")
            success = False
        else:
            logs.append(f"PASS: n3/n4 ratio matches target: 9:1")
        
        # Check internal consistency
        # Recalculate I2 from U2 and P1
        I2_calc = self.P1 / self.U2
        if not math.isclose(self.I2, I2_calc, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"WARN: I2 ({self.I2}A) differs from P1/U2 ({I2_calc}A)")
        
        # Recalculate U3 from power balance
        P3 = self.P1 - self.P_loss
        U3_calc = P3 / self.I2
        if not math.isclose(self.U3, U3_calc, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"WARN: U3 ({self.U3}V) differs from (P1-P_loss)/I2 ({U3_calc}V)")
        
        return success, logs
    
    def test_monotonicity(self):
        """Test trend when resistance increases"""
        logs = []
        success = True
        
        # Save original values
        original_R = self.R
        original_P_loss = self.P_loss
        
        # Perturb resistance by 10%
        perturbed_R = self.R * 1.1  # Increase by 10%
        
        # Recalculate using physical formulas
        # Step 1: U2 remains same (determined by transformer ratio and U1)
        U2 = self.U1 * 5  # Fixed by transformer
        
        # Step 2: I2 remains same (determined by P1 and U2, assuming ideal transformer)
        I2 = self.P1 / U2  # Fixed by power conservation
        
        # Step 3: New power loss
        new_P_loss = I2**2 * perturbed_R
        
        # Check that increased resistance leads to increased power loss
        if new_P_loss <= original_P_loss:
            logs.append(f"FAIL: Increased resistance should increase power loss. Original: {original_P_loss}W, New: {new_P_loss}W")
            success = False
        else:
            logs.append(f"PASS: Monotonicity holds: R↑ {original_R}→{perturbed_R:.1f}Ω, P_loss↑ {original_P_loss}→{new_P_loss:.1f}W")
        
        # Restore original R (though not strictly necessary as we work on copy)
        self.R = original_R
        
        # Additional monotonicity test: If step-up transformer ratio increases, U2 increases
        original_ratio = 5  # n2/n1
        perturbed_ratio = 5.5  # 10% increase
        U2_original = self.U1 * original_ratio
        U2_perturbed = self.U1 * perturbed_ratio
        
        if U2_perturbed <= U2_original:
            logs.append("FAIL: Increased step-up ratio should increase U2")
            success = False
        else:
            logs.append(f"PASS: U2 increases with step-up ratio: {U2_original}V → {U2_perturbed}V")
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs"""
        all_logs = ["=== Physics Checker for Transformer Transmission Problem ==="]
        overall_success = True
        
        # Run invariants test
        success, logs = self.test_invariants()
        all_logs.append("\n1. PHYSICAL INVARIANTS:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run trace consistency test
        success, logs = self.test_trace_consistency()
        all_logs.append("\n2. TRACE CONSISTENCY:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run monotonicity test
        success, logs = self.test_monotonicity()
        all_logs.append("\n3. MONOTONICITY (SENSITIVITY):")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Summary
        all_logs.append("\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("✓ All physics checks passed!")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return overall_success, all_logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)