import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction using getattr with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.R_1 = self._get_result('R_1', 0.0)
        self.R_2 = self._get_result('R_2', 0.0)
        self.P_1 = 3.0  # From problem: L1 lights normally at 3W
        self.P_2_prime = self._get_result('P_2\'', 0.0)
        self.P_out = self.P_1 + self.P_2_prime
        self.P_in = self._get_result('P_in', 0.0)
        self.P_a = self._get_result('P_a', 0.0)
        self.U_out = self._get_result('U_out', 0.0)
        self.U_in = self._get_result('U_in', 0.0)
        self.E = self._get_result('E', 0.0)
        self.v = self._get_result('v', 0.0)
        
        # Fixed problem parameters
        self.l = 0.6  # m
        self.B = 1.0  # T
        self.r = 1.0  # Ohm
        self.R1_rated = 3.0  # From step1: R_1 = 3 Ohm
        self.R2_rated = 6.0  # From step2: R_2 = 6 Ohm
        
        # For monotonicity perturbation
        self.original_v = self.v
        self.original_E = self.E
        
    def _get_result(self, key: str, default: float) -> float:
        """Safely extract result from solution instance."""
        # First try results dictionary
        val = self.results.get(key, default)
        if val is None:
            return default
        # Handle ratio strings like "2:1"
        if isinstance(val, str) and ':' in val:
            parts = val.split(':')
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except:
                    return default
        # Convert to float if possible
        try:
            return float(val)
        except:
            return default
    
    def test_invariants(self) -> List[str]:
        """Verify fundamental physics laws."""
        logs = []
        
        # 1. Ohm's Law checks
        # For lamp L1 (should be at rated voltage when lit normally)
        I1_calc = math.sqrt(self.P_1 / self.R1_rated) if self.R1_rated > 0 else 0
        U1_calc = self.R1_rated * I1_calc if I1_calc > 0 else 0
        if not math.isclose(self.U_out, 3.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: U_out should be ~3.0V (L1 normal), got {self.U_out}")
        
        # For lamp L2
        if self.R_2 > 0:
            I2_calc = self.U_out / self.R_2 if self.R_2 > 0 else 0
            P2_calc = I2_calc * self.U_out
            if not math.isclose(self.P_2_prime, P2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: P_2' mismatch. Calculated {P2_calc}, got {self.P_2_prime}")
        
        # 2. Series/Parallel circuit laws
        # Equivalent resistance check
        R_out_calc = (self.R1_rated * self.R2_rated) / (self.R1_rated + self.R2_rated)
        if not math.isclose(R_out_calc, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: R_out should be ~2.0Ω, calculated {R_out_calc}")
        
        # Voltage divider check
        if self.U_out > 0 and self.U_in > 0:
            ratio_calc = self.U_out / self.U_in
            expected_ratio = R_out_calc / self.r if self.r > 0 else 0
            if not math.isclose(ratio_calc, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_out:U_in ratio should be ~{expected_ratio}, got {ratio_calc}")
        
        # 3. Motional EMF law
        E_calc = self.B * self.l * self.v
        if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: E = B*l*v = {E_calc}, but got {self.E}")
        
        # 4. Energy conservation (power balance)
        # Total power dissipated in circuit
        P_total_circuit = self.P_1 + self.P_2_prime + self.P_in
        if not math.isclose(self.P_a, P_total_circuit, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: P_a should equal sum of all powers. P_total={P_total_circuit}, P_a={self.P_a}")
        
        # Alternative: P_a = E²/(R_out + r)
        R_total = R_out_calc + self.r
        P_a_calc = (self.E ** 2) / R_total if R_total > 0 else 0
        if not math.isclose(self.P_a, P_a_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: P_a = E²/(R_out+r) = {P_a_calc}, got {self.P_a}")
        
        if not logs:
            logs.append("PASS: All invariants satisfied")
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # Check step 3 ratio
        if self.R_1 > 0 and self.R_2 > 0:
            expected_ratio = self.R_2 / self.R_1  # P1:P2' = R2:R1
            actual_ratio = self.P_1 / self.P_2_prime if self.P_2_prime > 0 else 0
            if not math.isclose(expected_ratio, actual_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: P1:P2' ratio. Expected {expected_ratio}, got {actual_ratio}")
        
        # Check step 6 ratio
        if self.U_in > 0:
            expected_ratio2 = 2.0  # From step6: U_out:U_in = 2:1
            actual_ratio2 = self.U_out / self.U_in
            if not math.isclose(expected_ratio2, actual_ratio2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_out:U_in ratio. Expected 2.0, got {actual_ratio2}")
        
        # Check step 8: E = U_in + U_out
        E_calc = self.U_in + self.U_out
        if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: E = U_in + U_out = {E_calc}, got {self.E}")
        
        # Check step 10: v = E/(B*l)
        v_calc = self.E / (self.B * self.l)
        if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v = E/(B*l) = {v_calc}, got {self.v}")
        
        # Check step 11: P_out:P_in ratio
        if self.P_in > 0:
            expected_ratio3 = 2.0  # From step11: P_out:P_in = 2:1
            actual_ratio3 = self.P_out / self.P_in
            if not math.isclose(expected_ratio3, actual_ratio3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: P_out:P_in ratio. Expected 2.0, got {actual_ratio3}")
        
        # Check step 13: P_a = P_1 + P_2' + P_in
        P_total = self.P_1 + self.P_2_prime + self.P_in
        if not math.isclose(self.P_a, P_total, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: P_a = sum of powers = {P_total}, got {self.P_a}")
        
        if not logs:
            logs.append("PASS: All trace consistency checks passed")
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Perturb magnetic field B and check trend in velocity."""
        logs = []
        
        # Store original values
        B_original = self.B
        v_original = self.v
        
        # Perturb B by +10%
        B_perturbed = B_original * 1.1
        
        # From motional EMF: v = E/(B*l)
        # But E is not constant! When B changes, the circuit conditions change.
        # However, to keep L1 lit normally (U_out=3V), the induced EMF must adjust.
        
        # Let's analyze: The circuit equation is E = I*(R_out + r)
        # And U_out = I*R_out = 3V (fixed for normal lighting)
        # So I = 3/R_out = 3/2 = 1.5A (fixed)
        # Thus E = 1.5*(R_out + r) = 1.5*3 = 4.5V (fixed!)
        
        # Therefore, to maintain normal lighting, E must remain constant at 4.5V.
        # From E = B*l*v, if B increases, v must decrease to keep E constant.
        
        E_fixed = 4.5  # From circuit analysis
        v_expected = E_fixed / (B_perturbed * self.l)
        
        # Check trend: increasing B should decrease v
        if v_expected >= v_original:
            logs.append(f"FAIL: Monotonicity violated. B increased by 10%, v should decrease but calculated v={v_expected}")
        
        # Restore original values (in practice, just check logic)
        self.v = v_original
        
        if not logs:
            logs.append("PASS: Monotonicity check passed (v ∝ 1/B for fixed normal lighting)")
        return logs
    
    def test_final_answers(self) -> List[str]:
        """Verify the three target answers."""
        logs = []
        targets = [1.5, 7.5, 6.75]  # P_2', v, P_a
        
        # Check P_2'
        if not math.isclose(self.P_2_prime, targets[0], rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: P_2' should be ~{targets[0]}W, got {self.P_2_prime}")
        
        # Check v
        if not math.isclose(self.v, targets[1], rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v should be ~{targets[1]}m/s, got {self.v}")
        
        # Check P_a
        if not math.isclose(self.P_a, targets[2], rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: P_a should be ~{targets[2]}W, got {self.P_a}")
        
        if not logs:
            logs.append("PASS: All final answers match targets")
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status with logs."""
        all_logs = []
        
        # Run tests
        all_logs.append("=== Testing Invariants ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== Testing Trace Consistency ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== Testing Monotonicity ===")
        all_logs.extend(self.test_monotonicity())
        
        all_logs.append("\n=== Testing Final Answers ===")
        all_logs.extend(self.test_final_answers())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("FAIL:"):
                success = False
                break
        
        return success, all_logs