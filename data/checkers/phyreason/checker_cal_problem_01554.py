import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.E = self.results.get('E', 0.0)
        self.U_CD = self.results.get('U_CD', 0.0)
        self.U_CB = self.results.get('U_CB', 0.0)
        self.varphi_C = self.results.get('varphi_C', 0.0)
        self.E_pC = self.results.get('E_pC', 0.0)
        
        # Constants from problem statement
        self.e = 1.6e-19  # C
        self.W_AB = 4.0e-17  # J
        self.d_AB = 0.10  # m (10 cm)
        self.d_AC = 0.02  # m (2 cm)
        self.d_CB = 0.08  # m (8 cm)
        self.d_CD = 0.08  # m (8 cm)
        self.theta_CD = 60  # degrees
        
        # Reference values from steps analysis
        self.ref_E = 2500.0  # V/m
        self.ref_U_CD = 100.0  # V
        self.ref_U_CB = 200.0  # V
        self.ref_varphi_C = 200.0  # V
        self.ref_E_pC = 3.2e-17  # J
        
    def test_invariants(self):
        logs = []
        
        # Test 1: Work-energy theorem for A→B
        try:
            # W = q * E * d in uniform field
            W_calculated = self.e * self.E * self.d_AB
            if not math.isclose(self.W_AB, W_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy invariant. W_AB={self.W_AB:.2e}, "
                           f"calculated={W_calculated:.2e}, E={self.E}")
            else:
                logs.append("PASS: Work-energy theorem consistent for A→B")
        except Exception as e:
            logs.append(f"ERROR in work-energy test: {str(e)}")
        
        # Test 2: Relationship between potential difference and electric field for U_CB
        try:
            # U_CB = E * d_CB
            U_CB_calculated = self.E * self.d_CB
            if not math.isclose(self.U_CB, U_CB_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_CB invariant. U_CB={self.U_CB}, "
                           f"calculated={U_CB_calculated}, E={self.E}, d_CB={self.d_CB}")
            else:
                logs.append("PASS: U_CB = E * d_CB holds")
        except Exception as e:
            logs.append(f"ERROR in U_CB invariant test: {str(e)}")
        
        # Test 3: Relationship between potential difference and electric field for U_CD
        try:
            # U_CD = E * d_CD * cos(60°)
            U_CD_calculated = self.E * self.d_CD * math.cos(math.radians(self.theta_CD))
            if not math.isclose(self.U_CD, U_CD_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_CD invariant. U_CD={self.U_CD}, "
                           f"calculated={U_CD_calculated}, E={self.E}")
            else:
                logs.append("PASS: U_CD = E * d_CD * cos(θ) holds")
        except Exception as e:
            logs.append(f"ERROR in U_CD invariant test: {str(e)}")
        
        # Test 4: Potential energy at C
        try:
            # E_pC = e * φ_C
            E_pC_calculated = self.e * self.varphi_C
            if not math.isclose(self.E_pC, E_pC_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_pC invariant. E_pC={self.E_pC:.2e}, "
                           f"calculated={E_pC_calculated:.2e}, varphi_C={self.varphi_C}")
            else:
                logs.append("PASS: E_pC = e * φ_C holds")
        except Exception as e:
            logs.append(f"ERROR in E_pC invariant test: {str(e)}")
        
        # Test 5: Consistency between potential differences (φ_C relative to B)
        try:
            # φ_C = U_CB (if φ_B = 0)
            if not math.isclose(self.varphi_C, self.U_CB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: φ_C consistency. φ_C={self.varphi_C}, U_CB={self.U_CB}")
            else:
                logs.append("PASS: φ_C = U_CB (with φ_B = 0)")
        except Exception as e:
            logs.append(f"ERROR in φ_C consistency test: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        
        # Compare each result with reference values from steps analysis
        checks = [
            ('E', self.E, self.ref_E, "V/m"),
            ('U_CD', self.U_CD, self.ref_U_CD, "V"),
            ('U_CB', self.U_CB, self.ref_U_CB, "V"),
            ('varphi_C', self.varphi_C, self.ref_varphi_C, "V"),
            ('E_pC', self.E_pC, self.ref_E_pC, "J")
        ]
        
        for name, value, ref, unit in checks:
            try:
                if math.isclose(value, ref, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"PASS: {name}={value} matches reference {ref} {unit}")
                else:
                    logs.append(f"FAIL: {name}={value} differs from reference {ref} {unit}")
            except Exception as e:
                logs.append(f"ERROR comparing {name}: {str(e)}")
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        
        # Perturb the work W_AB by +10% and check if E changes proportionally
        try:
            W_original = self.W_AB
            W_perturbed = W_original * 1.10  # +10%
            
            # Recalculate E using formula E = W / (e * d_AB)
            E_original = self.E
            E_perturbed = W_perturbed / (self.e * self.d_AB)
            
            # Expected: E should increase by 10%
            expected_ratio = 1.10
            actual_ratio = E_perturbed / E_original
            
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: E increases proportionally with W_AB (+10% perturbation)")
            else:
                logs.append(f"FAIL: E monotonicity. Expected ratio={expected_ratio:.3f}, "
                           f"actual={actual_ratio:.3f}")
            
            # Restore original values
            self.W_AB = W_original
            self.E = E_original
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            traceback.print_exc()
        
        # Perturb distance d_AB and check trend
        try:
            d_original = self.d_AB
            d_perturbed = d_original * 0.90  # -10%
            
            # Recalculate E using formula E = W / (e * d_AB)
            E_original = self.E
            E_perturbed = self.W_AB / (self.e * d_perturbed)
            
            # Expected: E should increase when d decreases (inverse relationship)
            expected_ratio = 1.0 / 0.90  # ~1.111
            actual_ratio = E_perturbed / E_original
            
            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: E inversely proportional to d_AB (-10% perturbation)")
            else:
                logs.append(f"FAIL: d_AB monotonicity. Expected ratio={expected_ratio:.3f}, "
                           f"actual={actual_ratio:.3f}")
            
            # Restore original values
            self.d_AB = d_original
            self.E = E_original
            
        except Exception as e:
            logs.append(f"ERROR in d_AB monotonicity test: {str(e)}")
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        success = True
        
        try:
            # Run each test suite
            all_logs.append("=== Testing Physics Invariants ===")
            inv_logs = self.test_invariants()
            all_logs.extend(inv_logs)
            
            all_logs.append("\n=== Testing Trace Consistency ===")
            trace_logs = self.test_trace_consistency()
            all_logs.extend(trace_logs)
            
            all_logs.append("\n=== Testing Monotonicity ===")
            mono_logs = self.test_monotonicity()
            all_logs.extend(mono_logs)
            
            # Check for failures
            for log in inv_logs + trace_logs + mono_logs:
                if log.startswith("FAIL") or log.startswith("ERROR"):
                    success = False
                    break
                    
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR in test suite: {str(e)}")
            success = False
            traceback.print_exc()
        
        return success, all_logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)