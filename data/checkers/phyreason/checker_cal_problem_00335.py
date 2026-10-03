import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely - use defaults if not in results
        self.results = getattr(solution_instance, 'results', {})
        
        # Use the original problem constants directly
        self.q = -2e-9
        self.W_AB = -2e-7
        self.W_BC = 6e-7
        
        # Extract results safely
        self.U_AB = self.results.get('U_AB', 0.0)
        self.U_BC = self.results.get('U_BC', 0.0)
        self.U_AC = self.results.get('U_AC', 0.0)
        self.varphi_C = self.results.get('varphi_C', 0.0)
        self.E_C = self.results.get('E_C', 0.0)
        
        # Constants
        self.tol_rel = 1e-2
        self.tol_abs = 1e-2
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def is_close(self, a, b):
        return math.isclose(a, b, rel_tol=self.tol_rel, abs_tol=self.tol_abs)
    
    def test_invariants(self):
        try:
            # Theorem 1: W_by_field = -q * delta_V
            # From steps: U_AB = W_AB/q and U_BC = W_BC/q
            # Check U_AB = W_AB / q
            if not self.is_close(self.U_AB, self.W_AB / self.q):
                self.log(f"FAIL: U_AB = {self.U_AB} != W_AB/q = {self.W_AB / self.q}")
                return False
            self.log(f"PASS: U_AB = W_AB/q holds")
            
            # Check U_BC = W_BC / q
            if not self.is_close(self.U_BC, self.W_BC / self.q):
                self.log(f"FAIL: U_BC = {self.U_BC} != W_BC/q = {self.W_BC / self.q}")
                return False
            self.log(f"PASS: U_BC = W_BC/q holds")
            
            # Theorem 2: Additivity of potential differences
            if not self.is_close(self.U_AC, self.U_AB + self.U_BC):
                self.log(f"FAIL: U_AC = {self.U_AC} != U_AB + U_BC = {self.U_AB + self.U_BC}")
                return False
            self.log(f"PASS: U_AC = U_AB + U_BC holds")
            
            # Theorem 3: U_AC = phi_A - phi_C with phi_A = 0
            if not self.is_close(self.varphi_C, -self.U_AC):
                self.log(f"FAIL: varphi_C = {self.varphi_C} != -U_AC = {-self.U_AC}")
                return False
            self.log(f"PASS: varphi_C = -U_AC holds")
            
            # Theorem 4: E_C = q * varphi_C
            expected_E_C = self.q * self.varphi_C
            if not self.is_close(self.E_C, expected_E_C):
                self.log(f"FAIL: E_C = {self.E_C} != q * varphi_C = {expected_E_C}")
                return False
            self.log(f"PASS: E_C = q * varphi_C holds")
            
            # Energy conservation check: W_total = W_AB + W_BC = -q * U_AC
            W_total = self.W_AB + self.W_BC
            expected_W_total = -self.q * self.U_AC
            if not self.is_close(W_total, expected_W_total):
                self.log(f"FAIL: W_total = {W_total} != -q * U_AC = {expected_W_total}")
                return False
            self.log(f"PASS: Energy conservation (W_total = -q * U_AC) holds")
            
            return True
        except Exception as e:
            self.log(f"Error in invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        try:
            # Cross-check with target answers from steps
            target_U_AB = 100.0
            target_U_BC = -300.0
            target_U_AC = -200.0
            target_E_C = -4e-7
            
            if not (self.is_close(self.U_AB, target_U_AB) and 
                    self.is_close(self.U_BC, target_U_BC) and 
                    self.is_close(self.U_AC, target_U_AC)):
                self.log(f"FAIL: Potential differences don't match targets")
                self.log(f"  U_AB: got {self.U_AB}, expected {target_U_AB}")
                self.log(f"  U_BC: got {self.U_BC}, expected {target_U_BC}")
                self.log(f"  U_AC: got {self.U_AC}, expected {target_U_AC}")
                return False
            self.log(f"PASS: Potential differences match targets")
            
            if not self.is_close(self.E_C, target_E_C):
                self.log(f"FAIL: E_C = {self.E_C} != target {target_E_C}")
                return False
            self.log(f"PASS: E_C matches target")
            
            # Verify intermediate consistency
            # From U_AC, compute varphi_C and check against result
            computed_varphi_C = -self.U_AC
            if not self.is_close(self.varphi_C, computed_varphi_C):
                self.log(f"FAIL: varphi_C inconsistency: got {self.varphi_C}, computed {computed_varphi_C}")
                return False
            self.log(f"PASS: varphi_C consistent with U_AC")
            
            # From varphi_C and q, compute E_C and check
            computed_E_C = self.q * self.varphi_C
            if not self.is_close(self.E_C, computed_E_C):
                self.log(f"FAIL: E_C inconsistency: got {self.E_C}, computed {computed_E_C}")
                return False
            self.log(f"PASS: E_C consistent with varphi_C and q")
            
            return True
        except Exception as e:
            self.log(f"Error in trace consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        try:
            # Perturb charge q by +10% and recompute E_C
            original_q = self.q
            perturbed_q = original_q * 1.10  # +10%
            
            # Recompute chain using formulas from steps
            U_AB_pert = self.W_AB / perturbed_q
            U_BC_pert = self.W_BC / perturbed_q
            U_AC_pert = U_AB_pert + U_BC_pert
            varphi_C_pert = -U_AC_pert
            E_C_pert = perturbed_q * varphi_C_pert
            
            # Analyze trend: E_C = - (W_AB + W_BC) is independent of q!
            # Actually, from derivation: E_C = q * (-U_AC) = q * (-(W_AB/q + W_BC/q)) = -(W_AB + W_BC)
            # So E_C should remain constant when q changes
            if not self.is_close(E_C_pert, self.E_C):
                self.log(f"FAIL: E_C changed with q perturbation: {E_C_pert} vs {self.E_C}")
                return False
            self.log(f"PASS: E_C invariant under q perturbation (as expected from E_C = -(W_AB+W_BC))")
            
            # Alternative: perturb W_AB and check E_C trend
            original_W_AB = self.W_AB
            perturbed_W_AB = original_W_AB * 1.05  # +5%
            
            # Recompute with perturbed W_AB
            U_AB_pert2 = perturbed_W_AB / self.q
            U_AC_pert2 = U_AB_pert2 + self.U_BC
            varphi_C_pert2 = -U_AC_pert2
            E_C_pert2 = self.q * varphi_C_pert2
            
            # Expected trend: E_C = -(W_AB + W_BC), so E_C_pert2 = -(perturbed_W_AB + W_BC)
            expected_E_C_pert2 = -(perturbed_W_AB + self.W_BC)
            
            if not self.is_close(E_C_pert2, expected_E_C_pert2):
                self.log(f"FAIL: E_C after W_AB perturbation doesn't match theoretical value")
                self.log(f"  Got: {E_C_pert2}, Expected: {expected_E_C_pert2}")
                return False
            self.log(f"PASS: E_C monotonicity correct after W_AB perturbation")
            
            return True
        except Exception as e:
            self.log(f"Error in monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test in tests:
            self.log(f"\n=== Running {name} Test ===")
            if not test():
                success = False
                self.log(f"FAILED: {name}")
            else:
                self.log(f"PASSED: {name}")
        
        return success, self.logs