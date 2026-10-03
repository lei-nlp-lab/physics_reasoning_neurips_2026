import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults
        self.L = getattr(solution_instance, 'L', None)
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Safe extraction of results with fallback to 0.0
        results = getattr(solution_instance, 'results', {})
        self.E = results.get('E', 0.0)
        self.varphi_B = results.get('varphi_B', 0.0)
        self.E_p = results.get('E_p', 0.0)
        self.U_OB = results.get('U_OB', 0.0)
        self.U_BA = results.get('U_BA', 0.0)
        self.U_AB = results.get('U_AB', 0.0)
        
        # Define symbolic variables for invariant checks
        self.sym_L, self.sym_m, self.sym_q, self.sym_g = sp.symbols('L m q g', positive=True)
        self.sym_theta = sp.rad(60)  # 60 degrees in radians
        
        # Expected symbolic expressions from the solution steps
        self.expected_U_AB = -sp.sqrt(3) * self.sym_m * self.sym_g * self.sym_L / (2 * self.sym_q)
        self.expected_U_BA = -self.expected_U_AB
        self.expected_E = sp.sqrt(3) * self.sym_m * self.sym_g / self.sym_q
        self.expected_U_OB = self.expected_E * self.sym_L * sp.cos(self.sym_theta)
        self.expected_varphi_B = -self.expected_U_OB
        self.expected_E_p = self.sym_q * self.expected_varphi_B
        
        # Store logs for all tests
        self.logs = []
        
    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((success, message))
        
    def test_invariants(self):
        """Check fundamental physical laws and consistency of intermediate steps."""
        success = True
        
        # 1. Work-energy theorem: m*g*L*sin(60°) + q*U_AB = 0 (since initial and final kinetic energy are zero)
        work_gravity = self.m * self.g * self.L * math.sin(math.radians(60))
        work_electric = self.q * self.U_AB
        net_work = work_gravity + work_electric
        if not math.isclose(net_work, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Work-energy theorem violation: m*g*L*sin60° + q*U_AB = {net_work:.4f} ≠ 0", False)
            success = False
        else:
            self._log(f"Work-energy theorem satisfied: m*g*L*sin60° + q*U_AB ≈ 0", True)
            
        # 2. Potential difference antisymmetry: U_BA = -U_AB
        if not math.isclose(self.U_BA, -self.U_AB, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Potential difference antisymmetry violation: U_BA = {self.U_BA:.4f}, -U_AB = {-self.U_AB:.4f}", False)
            success = False
        else:
            self._log(f"Potential difference antisymmetry satisfied: U_BA = -U_AB", True)
            
        # 3. Electric field from potential difference: E = U_BA / (L * (1 - cos60°))
        expected_E_from_U_BA = self.U_BA / (self.L * (1 - math.cos(math.radians(60))))
        if not math.isclose(self.E, expected_E_from_U_BA, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"E from U_BA inconsistent: E = {self.E:.4f}, computed = {expected_E_from_U_BA:.4f}", False)
            success = False
        else:
            self._log(f"E consistent with U_BA and geometry", True)
            
        # 4. Potential difference U_OB = E * L * cos60°
        expected_U_OB = self.E * self.L * math.cos(math.radians(60))
        if not math.isclose(self.U_OB, expected_U_OB, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"U_OB from E inconsistent: U_OB = {self.U_OB:.4f}, computed = {expected_U_OB:.4f}", False)
            success = False
        else:
            self._log(f"U_OB consistent with E and geometry", True)
            
        # 5. Electric potential at B: φ_B = -U_OB (assuming φ_O = 0)
        if not math.isclose(self.varphi_B, -self.U_OB, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"φ_B from U_OB inconsistent: φ_B = {self.varphi_B:.4f}, -U_OB = {-self.U_OB:.4f}", False)
            success = False
        else:
            self._log(f"φ_B consistent with U_OB (φ_O=0 assumed)", True)
            
        # 6. Electric potential energy: E_p = q * φ_B
        expected_E_p = self.q * self.varphi_B
        if not math.isclose(self.E_p, expected_E_p, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"E_p from φ_B inconsistent: E_p = {self.E_p:.4f}, computed = {expected_E_p:.4f}", False)
            success = False
        else:
            self._log(f"E_p consistent with φ_B", True)
            
        # 7. Symbolic cross-check with derived formulas (numerical substitution)
        sym_vals = {self.sym_L: self.L, self.sym_m: self.m, self.sym_q: self.q, self.sym_g: self.g}
        expected_E_val = float(self.expected_E.subs(sym_vals))
        expected_varphi_B_val = float(self.expected_varphi_B.subs(sym_vals))
        expected_E_p_val = float(self.expected_E_p.subs(sym_vals))
        
        if not (math.isclose(self.E, expected_E_val, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.varphi_B, expected_varphi_B_val, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.E_p, expected_E_p_val, rel_tol=1e-2, abs_tol=1e-2)):
            self._log("Symbolic-numerical mismatch with derived formulas", False)
            success = False
        else:
            self._log("All results match symbolic derivations", True)
            
        return success
    
    def test_trace_consistency(self):
        """Verify that intermediate results align with final answers."""
        success = True
        
        # Check that U_AB leads to correct E via steps
        # From step 1: U_AB = -√3 m g L / (2q)
        expected_U_AB = -math.sqrt(3) * self.m * self.g * self.L / (2 * self.q)
        if not math.isclose(self.U_AB, expected_U_AB, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"U_AB inconsistent with input parameters: got {self.U_AB:.4f}, expected {expected_U_AB:.4f}", False)
            success = False
        else:
            self._log(f"U_AB matches formula from step 1", True)
            
        # Check that E matches the target expression √3 m g / q
        expected_E = math.sqrt(3) * self.m * self.g / self.q
        if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"E inconsistent with target expression: got {self.E:.4f}, expected {expected_E:.4f}", False)
            success = False
        else:
            self._log(f"E matches target expression √3 m g / q", True)
            
        # Check that φ_B matches -√3 m g L / (2q)
        expected_varphi_B = -math.sqrt(3) * self.m * self.g * self.L / (2 * self.q)
        if not math.isclose(self.varphi_B, expected_varphi_B, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"φ_B inconsistent with target expression: got {self.varphi_B:.4f}, expected {expected_varphi_B:.4f}", False)
            success = False
        else:
            self._log(f"φ_B matches target expression -√3 m g L / (2q)", True)
            
        # Check that E_p matches -√3 m g L / 2
        expected_E_p = -math.sqrt(3) * self.m * self.g * self.L / 2
        if not math.isclose(self.E_p, expected_E_p, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"E_p inconsistent with target expression: got {self.E_p:.4f}, expected {expected_E_p:.4f}", False)
            success = False
        else:
            self._log(f"E_p matches target expression -√3 m g L / 2", True)
            
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and verify that E changes as expected (E ∝ m)."""
        if self.m is None or self.q == 0:
            self._log("Cannot test monotonicity: missing m or q=0", False)
            return False
            
        success = True
        original_m = self.m
        perturbed_m = original_m * 1.05  # +5% perturbation
        
        # Recalculate expected E using the formula E = √3 * m * g / q
        original_E = math.sqrt(3) * original_m * self.g / self.q
        perturbed_E = math.sqrt(3) * perturbed_m * self.g / self.q
        
        # Check monotonic trend: if m increases, E should increase (since q ≠ 0)
        # But note: if q is negative, E will decrease when m increases. So we check the sign.
        expected_trend = (perturbed_E > original_E) if self.q > 0 else (perturbed_E < original_E)
        actual_trend = (self.E > 0 and perturbed_E > original_E) or (self.E < 0 and perturbed_E < original_E)
        
        if not expected_trend == actual_trend:
            self._log(f"Monotonicity violation: E should {'increase' if self.q>0 else 'decrease'} with m, but perturbed E = {perturbed_E:.4f}, original E = {original_E:.4f}", False)
            success = False
        else:
            self._log(f"Monotonicity holds: E ∝ m (sign depends on q)", True)
            
        # Restore original mass
        self.m = original_m
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        test_methods = [
            ("Invariant Checks", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in test_methods:
            try:
                success = test_func()
                if not success:
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
                
        return overall_success, self.logs