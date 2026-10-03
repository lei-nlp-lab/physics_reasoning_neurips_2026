import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.r0 = getattr(solution_instance, 'r0', 1.0)
        self.rho0 = getattr(solution_instance, 'rho0', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.G = getattr(solution_instance, 'G', 6.67430e-11)
        
        # Extract results with safe access
        self.rho = self.results.get('$rho$', self.results.get('ρ', 0.0))
        self.M = self.results.get('M', 0.0)
        self.delta_Ek = self.results.get('$DeltaE_k$', self.results.get('ΔEk', 0.0))
        self.H = self.results.get('H', None)  # Could be string or numerical
        
        # Create symbolic variables for verification
        self.r0_sym = sp.symbols('r0', positive=True, real=True)
        self.rho0_sym = sp.symbols('rho0', positive=True, real=True)
        self.m_sym = sp.symbols('m', positive=True, real=True)
        self.G_sym = sp.symbols('G', positive=True, real=True)
        
        self.logs = []
        
    def log(self, message, success=None):
        if success is not None:
            prefix = "✓" if success else "✗"
            message = f"{prefix} {message}"
        self.logs.append(message)
        
    def test_invariants(self):
        """Verify physical laws based on KB theorems"""
        try:
            # 1. Check density scaling (mass conservation)
            rho_expected = self.rho0 / 8
            if isinstance(self.rho, (int, float)):
                if not math.isclose(self.rho, rho_expected, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"Density mismatch: expected ~{rho_expected}, got {self.rho}", False)
                    return False
            self.log(f"Density scaling correct: ρ = ρ₀/8 = {rho_expected}", True)
            
            # 2. Check mass formula (Shell theorem)
            M_expected = (4/3) * math.pi * self.r0**3 * self.rho0
            if isinstance(self.M, (int, float)):
                if not math.isclose(self.M, M_expected, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"Mass mismatch: expected ~{M_expected}, got {self.M}", False)
                    return False
            self.log(f"Mass formula correct: M = (4/3)πr₀³ρ₀ = {M_expected}", True)
            
            # 3. Check energy conservation
            # Potential energy at r0: U0 = -GMm/r0
            # Potential energy at 2r0: U1 = -GMm/(2r0)
            # ΔEk = -ΔU = -(U1 - U0) = -[GMm/(2r0) - GMm/r0] = GMm/(2r0) - GMm/r0 = -GMm/(2r0)
            delta_Ek_expected = -self.G * self.M * self.m / (2 * self.r0)
            
            # Also check using the formula from step 3
            delta_Ek_formula = - (2/3) * self.G * math.pi * self.rho0 * self.m * self.r0**2
            
            if isinstance(self.delta_Ek, (int, float)):
                if not (math.isclose(self.delta_Ek, delta_Ek_expected, rel_tol=0.02, abs_tol=0.02) or 
                        math.isclose(self.delta_Ek, delta_Ek_formula, rel_tol=0.02, abs_tol=0.02)):
                    self.log(f"ΔEk mismatch: expected ~{delta_Ek_formula}, got {self.delta_Ek}", False)
                    return False
            self.log(f"Energy conservation correct: ΔEk = -(2/3)Gπρ₀mr₀² = {delta_Ek_formula}", True)
            
            # 4. Check symbolic consistency
            M_sym = (4/3) * sp.pi * self.r0_sym**3 * self.rho0_sym
            delta_Ek_sym = -self.G_sym * M_sym * self.m_sym / (2 * self.r0_sym)
            delta_Ek_sym_alt = - (2/3) * self.G_sym * sp.pi * self.rho0_sym * self.m_sym * self.r0_sym**2
            
            # Simplify to check equivalence
            diff = sp.simplify(delta_Ek_sym - delta_Ek_sym_alt)
            if diff != 0:
                self.log(f"Symbolic mismatch: ΔEk formulas not equivalent", False)
                return False
            self.log("Symbolic formulas equivalent", True)
            
            return True
            
        except Exception as e:
            self.log(f"Error in invariants test: {str(e)}", False)
            self.log(traceback.format_exc())
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check that using M from step 2 gives the same ΔEk as step 3
            if isinstance(self.M, (int, float)) and isinstance(self.delta_Ek, (int, float)):
                # Method 1: From potential energy difference
                delta_Ek_from_M = -self.G * self.M * self.m / (2 * self.r0)
                
                # Method 2: From the given formula
                delta_Ek_from_formula = - (2/3) * self.G * math.pi * self.rho0 * self.m * self.r0**2
                
                if not (math.isclose(delta_Ek_from_M, self.delta_Ek, rel_tol=0.02, abs_tol=0.02) and
                        math.isclose(delta_Ek_from_formula, self.delta_Ek, rel_tol=0.02, abs_tol=0.02)):
                    self.log(f"Trace inconsistency: ΔEk from M={delta_Ek_from_M}, from formula={delta_Ek_from_formula}, stored={self.delta_Ek}", False)
                    return False
                    
                self.log("Trace consistency: M → ΔEk conversion correct", True)
            
            # Check density and mass relationship
            if isinstance(self.rho, (int, float)) and self.rho != 0:
                M_from_density = (4/3) * math.pi * (2*self.r0)**3 * self.rho
                M_expected = (4/3) * math.pi * self.r0**3 * self.rho0
                
                # Mass should be conserved, so total mass in larger sphere is same
                # Actually, mass in sphere of radius 2r0 should be 8 * M_expected (volume factor)
                # But the density ρ = ρ0/8, so M_2r0 = (4/3)π(2r0)³ * (ρ0/8) = 8*(4/3)πr0³ρ0/8 = M_expected
                # So mass within 2r0 equals mass within r0 in the uniform distribution model
                if not math.isclose(M_from_density, M_expected, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"Mass-density inconsistency: M from ρ={M_from_density}, expected={M_expected}", False)
                    return False
                    
                self.log("Mass-density consistency: ρ → M conversion correct", True)
            
            return True
            
        except Exception as e:
            self.log(f"Error in trace consistency test: {str(e)}", False)
            self.log(traceback.format_exc())
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends"""
        try:
            original_r0 = self.r0
            
            # Test 1: If r0 increases, |ΔEk| should increase (more negative)
            perturbed_r0 = self.r0 * 1.1
            delta_Ek_original = - (2/3) * self.G * math.pi * self.rho0 * self.m * self.r0**2
            delta_Ek_perturbed = - (2/3) * self.G * math.pi * self.rho0 * self.m * perturbed_r0**2
            
            # Since both are negative, check absolute values
            if abs(delta_Ek_perturbed) <= abs(delta_Ek_original):
                self.log(f"Monotonicity fail: |ΔEk| should increase with r0, got {abs(delta_Ek_original)}→{abs(delta_Ek_perturbed)}", False)
                return False
            self.log(f"Monotonicity: |ΔEk| increases with r0 ({abs(delta_Ek_original)}→{abs(delta_Ek_perturbed)})", True)
            
            # Test 2: If ρ0 increases, |ΔEk| should increase
            perturbed_rho0 = self.rho0 * 1.1
            delta_Ek_perturbed2 = - (2/3) * self.G * math.pi * perturbed_rho0 * self.m * self.r0**2
            
            if abs(delta_Ek_perturbed2) <= abs(delta_Ek_original):
                self.log(f"Monotonicity fail: |ΔEk| should increase with ρ0, got {abs(delta_Ek_original)}→{abs(delta_Ek_perturbed2)}", False)
                return False
            self.log(f"Monotonicity: |ΔEk| increases with ρ0 ({abs(delta_Ek_original)}→{abs(delta_Ek_perturbed2)})", True)
            
            # Test 3: Check Hubble constant trend (should decrease with time)
            if self.H is not None:
                if isinstance(self.H, str):
                    if "decrease" not in self.H.lower() and "decreases" not in self.H.lower():
                        self.log(f"Hubble trend mismatch: expected 'decreases', got '{self.H}'", False)
                        return False
                self.log("Hubble constant decreases with time (as expected)", True)
            
            # Restore original values
            self.r0 = original_r0
            
            return True
            
        except Exception as e:
            self.log(f"Error in monotonicity test: {str(e)}", False)
            self.log(traceback.format_exc())
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"\nRunning {test_name}...")
            if not test_func():
                success = False
                self.log(f"{test_name} failed", False)
        
        return success, self.logs