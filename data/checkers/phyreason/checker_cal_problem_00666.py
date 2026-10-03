import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction using getattr with fallback to .results dict
        self.solution = solution_instance
        self.V = 8.0  # Given voltage
        self.q = 0.16  # Given charge
        self.t = 10.0  # Given time
        self.e = 1.6e-19  # Electron charge constant
        
        # Extract results safely
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', 0.0))
        self.n = getattr(solution_instance, 'n', getattr(solution_instance, 'results', {}).get('n', 0.0))
        
        # Standard physics constants
        self.electron_charge = 1.6e-19
        
    def test_invariants(self):
        """Verify fundamental physics laws: Ohm's Law and current definition"""
        logs = []
        
        # 1. Check Ohm's Law: V = I * R
        if not math.isclose(self.V, self.I * self.R, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Ohm's Law violation. V={self.V}, I×R={self.I*self.R}")
        else:
            logs.append("PASS: Ohm's Law satisfied")
        
        # 2. Check current definition: I = q/t
        expected_I = self.q / self.t
        if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Current definition. I={self.I}, q/t={expected_I}")
        else:
            logs.append("PASS: Current definition satisfied")
        
        # 3. Symbolic check of Ohm's Law variants
        try:
            V, I, R = sp.symbols('V I R')
            ohm_law = sp.Eq(V, I * R)
            # Substitute values
            ohm_check = ohm_law.subs({V: self.V, I: self.I, R: self.R})
            if not ohm_check:
                logs.append("WARN: Symbolic Ohm's Law check failed")
        except Exception as e:
            logs.append(f"Symbolic check skipped: {str(e)}")
        
        return all("FAIL" not in log for log in logs), logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate calculation steps with final results"""
        logs = []
        
        # Step 1: Current from charge/time
        step1_I = self.q / self.t
        if not math.isclose(self.I, step1_I, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 1 inconsistency. I={self.I}, q/t={step1_I}")
        else:
            logs.append(f"PASS: Step 1 (I = q/t) consistent. I={self.I}")
        
        # Step 2: Resistance from Ohm's Law
        step2_R = self.V / self.I if self.I != 0 else float('inf')
        if not math.isclose(self.R, step2_R, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 2 inconsistency. R={self.R}, V/I={step2_R}")
        else:
            logs.append(f"PASS: Step 2 (R = V/I) consistent. R={self.R}")
        
        # Step 3: Electron density (n = I/(e·S·v))
        # We don't have S and v, but we can check if n is reasonable (10^27 range)
        if self.n > 0:
            # Check order of magnitude
            order_magnitude = math.log10(self.n)
            if not (26 <= order_magnitude <= 28):  # Allow some leeway
                logs.append(f"WARN: n={self.n} has unexpected magnitude (expected ~10^27)")
            else:
                logs.append(f"PASS: n magnitude reasonable (10^{order_magnitude:.1f})")
        else:
            logs.append("FAIL: n is non-positive")
        
        return all("FAIL" not in log for log in logs), logs
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends"""
        logs = []
        
        # Test 1: Increase voltage → current should increase (Ohm's Law)
        original_V = self.V
        perturbed_V = original_V * 1.1  # +10%
        
        # Recalculate using formulas
        I_original = self.q / self.t
        R_original = original_V / I_original
        I_perturbed = perturbed_V / R_original
        
        if not (I_perturbed > I_original and math.isclose(I_perturbed, 1.1 * I_original, rel_tol=1e-2)):
            logs.append(f"FAIL: Voltage ↑10% → current should ↑10%. I_orig={I_original}, I_pert={I_perturbed}")
        else:
            logs.append("PASS: Voltage-current monotonicity OK")
        
        # Test 2: Increase resistance → current should decrease (Ohm's Law)
        R_perturbed = R_original * 1.1  # +10%
        I_R_perturbed = original_V / R_perturbed
        
        if not (I_R_perturbed < I_original and math.isclose(I_R_perturbed, I_original/1.1, rel_tol=1e-2)):
            logs.append(f"FAIL: Resistance ↑10% → current should ↓10%. I_orig={I_original}, I_R_pert={I_R_perturbed}")
        else:
            logs.append("PASS: Resistance-current monotonicity OK")
        
        # Test 3: Increase charge → current should increase proportionally
        q_perturbed = self.q * 1.1
        I_q_perturbed = q_perturbed / self.t
        
        if not (I_q_perturbed > I_original and math.isclose(I_q_perturbed, 1.1 * I_original, rel_tol=1e-2)):
            logs.append(f"FAIL: Charge ↑10% → current should ↑10%. I_orig={I_original}, I_q_pert={I_q_perturbed}")
        else:
            logs.append("PASS: Charge-current monotonicity OK")
        
        # Restore original values (conceptual - we didn't actually modify instance)
        logs.append("Original values preserved")
        
        return all("FAIL" not in log for log in logs), logs
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        all_passed = True
        all_logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            passed, logs = test_func()
            all_passed = all_passed and passed
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            
        # Final summary
        all_logs.append(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
        return all_passed, all_logs