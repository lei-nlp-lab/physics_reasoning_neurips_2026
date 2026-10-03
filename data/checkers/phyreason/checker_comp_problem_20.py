import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallback
        self.a_w = self.results.get('a_w', 0.0)
        self.b_w = self.results.get('b_w', 0.0)
        self.d_w = self.results.get('d_w', 0.0)
        self.V_G = self.results.get('V_G', 0.0)
        self.V_G0 = self.results.get('V_G0', 0.0)
        self.DeltaV_G_over_V_G0 = self.results.get('DeltaV_G/V_G0', 0.0)
        self.V_Gmin = self.results.get('V_Gmin', 0.0)
        self.V_G_over_V_Gmin = self.results.get('V_G/V_Gmin', 0.0)
        self.V_L = self.results.get('V_L', 0.0)
        self.rho_L = self.results.get('rho_L', 0.0)
        self.alpha = self.results.get('alpha', 0.0)
        self.L = self.results.get('L', 0.0)
        self.sigma = self.results.get('sigma', 0.0)
        self.Delta_p_T = self.results.get('Deltap_T', 0.0)
        self.r = self.results.get('r', 0.0)
        
        # Extract inputs with safe fallback
        self.T = getattr(solution_instance, 'T', 373.15)  # Assume boiling point if not provided
        self.p0 = getattr(solution_instance, 'p0', 101325.0)  # Standard atmospheric pressure
        self.R = getattr(solution_instance, 'R', 8.314462618)
        self.N_A = getattr(solution_instance, 'N_A', 6.02214076e23)
        self.mu = getattr(solution_instance, 'mu', 0.018015)  # Molar mass of water in kg/mol
        
        # Constants
        self.g = 9.80665
        self.logs = []
        
    def log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def test_invariants(self):
        """Check fundamental physical invariants from KB theorems"""
        success = True
        
        try:
            # 1. Check van der Waals equation consistency
            n = 1.0  # One mole
            V_G_calc = (self.R * self.T) / (2 * self.p0) * (1 + math.sqrt(1 - (4 * self.a_w * self.p0) / (self.R**2 * self.T**2)))
            if not math.isclose(self.V_G, V_G_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Van der Waals V_G mismatch: {self.V_G} vs {V_G_calc}", False) and success
            
            # 2. Check ideal gas law V_G0 = R*T/p0
            V_G0_calc = self.R * self.T / self.p0
            if not math.isclose(self.V_G0, V_G0_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Ideal gas V_G0 mismatch: {self.V_G0} vs {V_G0_calc}", False) and success
            
            # 3. Check relative decrease formula
            rel_decrease_calc = 0.5 * (1 - math.sqrt(1 - (4 * self.a_w * self.p0) / (self.R**2 * self.T**2)))
            if not math.isclose(self.DeltaV_G_over_V_G0, rel_decrease_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Relative decrease mismatch: {self.DeltaV_G_over_V_G0} vs {rel_decrease_calc}", False) and success
            
            # 4. Check V_Gmin formula
            V_Gmin_calc = 2 * self.a_w / (self.R * self.T)
            if not math.isclose(self.V_Gmin, V_Gmin_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"V_Gmin mismatch: {self.V_Gmin} vs {V_Gmin_calc}", False) and success
            
            # 5. Check ratio V_G/V_Gmin
            ratio_calc = self.V_G / self.V_Gmin
            if not math.isclose(self.V_G_over_V_Gmin, ratio_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"V_G/V_Gmin ratio mismatch: {self.V_G_over_V_Gmin} vs {ratio_calc}", False) and success
            
            # 6. Check liquid volume formula
            term = 1 - math.sqrt(1 - (4 * self.b_w * self.R * self.T) / self.a_w)
            V_L_calc = (self.a_w / (2 * self.R * self.T)) * term
            if not math.isclose(self.V_L, V_L_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Liquid volume mismatch: {self.V_L} vs {V_L_calc}", False) and success
            
            # 7. Check density from volume
            rho_L_calc = self.mu / self.V_L
            if not math.isclose(self.rho_L, rho_L_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Density mismatch: {self.rho_L} vs {rho_L_calc}", False) and success
            
            # 8. Check thermal expansion coefficient
            alpha_calc = (self.b_w * self.R) / (self.a_w + self.b_w * self.R * self.T)
            if not math.isclose(self.alpha, alpha_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Thermal expansion mismatch: {self.alpha} vs {alpha_calc}", False) and success
            
            # 9. Check latent heat formula
            L_calc = self.a_w / (self.mu * self.b_w * (1 + (self.b_w * self.R * self.T) / self.a_w))
            if not math.isclose(self.L, L_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Latent heat mismatch: {self.L} vs {L_calc}", False) and success
            
            # 10. Check surface tension formula
            sigma_calc = self.a_w * self.d_w / (2 * self.b_w**2)
            if not math.isclose(self.sigma, sigma_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Surface tension mismatch: {self.sigma} vs {sigma_calc}", False) and success
            
            # 11. Check molecular diameter from b
            d_w_calc = (self.b_w / self.N_A)**(1/3)
            if not math.isclose(self.d_w, d_w_calc, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Molecular diameter mismatch: {self.d_w} vs {d_w_calc}", False) and success
            
        except Exception as e:
            self.log(f"Invariant test failed with error: {str(e)}", False)
            return False
        
        if success:
            self.log("All invariant tests passed", True)
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        success = True
        
        try:
            # Check that V_G calculation is consistent with steps
            # From step 9: V_G = (RT/2p0) * (1 ± sqrt(1 - 4a p0/(R²T²)))
            disc = 1 - (4 * self.a_w * self.p0) / (self.R**2 * self.T**2)
            if disc < 0:
                success = self.log("Discriminant negative in V_G formula", False) and success
            
            # Check V_G > V_Gmin (physical condition)
            if self.V_G <= self.V_Gmin:
                success = self.log(f"V_G ({self.V_G}) not > V_Gmin ({self.V_Gmin})", False) and success
            
            # Check V_L is much smaller than V_G (liquid vs gas)
            if self.V_L >= self.V_G:
                success = self.log(f"Liquid volume {self.V_L} not < gas volume {self.V_G}", False) and success
            
            # Check relative decrease is positive
            if self.DeltaV_G_over_V_G0 < 0:
                success = self.log("Relative decrease should be positive", False) and success
            
            # Check ratio V_G/V_Gmin > 1
            if self.V_G_over_V_Gmin <= 1:
                success = self.log(f"Ratio V_G/V_Gmin should be >1, got {self.V_G_over_V_Gmin}", False) and success
            
            # Check density is reasonable for van der Waals model
            # Van der Waals gives lower density than real water
            # Accept range from 400 to 1000 kg/m³
            if not (400 < self.rho_L < 1000):
                success = self.log(f"Density {self.rho_L} outside reasonable van der Waals range (400-1000 kg/m³)", False) and success
            
            # Check thermal expansion positive
            if self.alpha <= 0:
                success = self.log(f"Thermal expansion should be positive, got {self.alpha}", False) and success
            
            # Check latent heat positive
            if self.L <= 0:
                success = self.log(f"Latent heat should be positive, got {self.L}", False) and success
            
            # Check surface tension positive
            if self.sigma <= 0:
                success = self.log(f"Surface tension should be positive, got {self.sigma}", False) and success
            
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}", False)
            return False
        
        if success:
            self.log("All trace consistency tests passed", True)
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends"""
        success = True
        
        try:
            # Save original values
            original_T = self.T
            
            # Perturb temperature upward by 1%
            self.T *= 1.01
            
            # Recalculate using formulas
            V_G_perturbed = (self.R * self.T) / (2 * self.p0) * (1 + math.sqrt(1 - (4 * self.a_w * self.p0) / (self.R**2 * self.T**2)))
            V_G0_perturbed = self.R * self.T / self.p0
            V_L_perturbed = (self.a_w / (2 * self.R * self.T)) * (1 - math.sqrt(1 - (4 * self.b_w * self.R * self.T) / self.a_w))
            alpha_perturbed = (self.b_w * self.R) / (self.a_w + self.b_w * self.R * self.T)
            
            # Check trends (gas volume should increase with T)
            if V_G_perturbed <= self.V_G:
                success = self.log(f"V_G should increase with T: {V_G_perturbed} vs {self.V_G}", False) and success
            
            # Ideal gas volume should increase
            if V_G0_perturbed <= self.V_G0:
                success = self.log(f"V_G0 should increase with T: {V_G0_perturbed} vs {self.V_G0}", False) and success
            
            # Liquid volume should increase (thermal expansion)
            if V_L_perturbed <= self.V_L:
                success = self.log(f"V_L should increase with T: {V_L_perturbed} vs {self.V_L}", False) and success
            
            # Thermal expansion coefficient might change slightly
            if abs(alpha_perturbed - self.alpha) > 0.1 * self.alpha:
                success = self.log(f"Alpha change too large: {alpha_perturbed} vs {self.alpha}", False) and success
            
            # Restore original temperature
            self.T = original_T
            
            # Perturb 'a' parameter
            original_a = self.a_w
            self.a_w *= 1.01
            
            # Recalculate with higher 'a'
            rel_decrease_perturbed = 0.5 * (1 - math.sqrt(1 - (4 * self.a_w * self.p0) / (self.R**2 * self.T**2)))
            sigma_perturbed = self.a_w * self.d_w / (2 * self.b_w**2)
            
            # With higher 'a', relative decrease should increase (stronger attraction)
            if rel_decrease_perturbed <= self.DeltaV_G_over_V_G0:
                success = self.log(f"Relative decrease should increase with a: {rel_decrease_perturbed} vs {self.DeltaV_G_over_V_G0}", False) and success
            
            # Surface tension should increase with 'a'
            if sigma_perturbed <= self.sigma:
                success = self.log(f"Sigma should increase with a: {sigma_perturbed} vs {self.sigma}", False) and success
            
            # Restore 'a'
            self.a_w = original_a
            
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}", False)
            # Try to restore original values
            self.T = getattr(self, '_original_T', self.T)
            self.a_w = getattr(self, '_original_a', self.a_w)
            return False
        
        if success:
            self.log("All monotonicity tests passed", True)
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status with logs"""
        self.logs = []
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        success = test1 and test2 and test3
        
        # Summary
        self.logs.append(f"\nSUMMARY: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
        self.logs.append(f"Invariants: {'PASS' if test1 else 'FAIL'}")
        self.logs.append(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
        self.logs.append(f"Monotonicity: {'PASS' if test3 else 'FAIL'}")
        
        return success, self.logs

# Example usage (for testing only):
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'a_w': 0.56,
                'b_w': 3.1e-5,
                'd_w': 3.7e-10,
                'V_G': 0.0306,
                'V_G0': 0.0308,
                'DeltaV_G/V_G0': 0.0058,
                'V_Gmin': 0.000355,
                'V_G/V_Gmin': 86.2,
                'V_L': 1.82e-5,
                'rho_L': 990.0,
                'alpha': 4.6e-4,
                'L': 1.0e6,
                'sigma': 0.012,
                'Deltap_T': 100.0,
                'r': 1.5e-8
            }
            self.T = 373.15
            self.p0 = 101325.0
            self.R = 8.314462618
            self.N_A = 6.02214076e23
            self.mu = 0.018015
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)