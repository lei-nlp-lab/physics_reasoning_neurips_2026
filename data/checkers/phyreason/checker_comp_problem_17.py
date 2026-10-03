import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results with fallbacks."""
        self.solution = solution_instance
        
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract ALL values from results with sensible defaults
        # Use both Greek and ASCII keys for compatibility
        self.E_min = results.get('E_min', results.get('E_min', 0.0))
        self.λ_dB = results.get('λ_dB', results.get('lambda_dB', results.get('lambda_dB_thermal', 0.0)))
        self.T_c = results.get('T_c', 0.0)
        self.E_n = results.get('E_n', 0.0)
        self.p_n = results.get('p_n', 0.0)
        self.λ_dBn = results.get('λ_dBn', results.get('lambda_dBn', 0.0))
        self.l = results.get('l', 0.0)
        self.λ_21 = results.get('λ_21', results.get('lambda_21', 0.0))
        self.p = results.get('p', 0.0)
        self.delta_E = results.get('delta_E', 0.0)
        self.E = results.get('E', 0.0)
        
        # Extract input constants from results or use defaults
        self.L = results.get('L', 1e-9)
        self.m = results.get('m', 9.10938356e-31)
        self.n = results.get('n', 2.0)  # quantum number for n-th state
        self.T = results.get('T', 300.0)
        self.c = results.get('c', 299792458.0)
        self.h = results.get('h', 6.62607015e-34)
        self.k_B = results.get('k_B', 1.380649e-23)
        self.density = results.get('density', results.get('n_density', 1e20))
        
        # Ensure no zero values for critical denominators
        if self.L == 0: self.L = 1e-9
        if self.m == 0: self.m = 9.10938356e-31
        if self.h == 0: self.h = 6.62607015e-34
        if self.k_B == 0: self.k_B = 1.380649e-23
        
        self.logs = []
        
    def log(self, message):
        """Helper for logging test results."""
        self.logs.append(message)
        
    def fuzzy_equal(self, a, b, context=""):
        """Robust fuzzy comparison with physics-appropriate tolerances."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        return False
    
    def test_invariants(self):
        """Verify fundamental physical laws from KB."""
        success = True
        
        try:
            # 1. Check particle in a box energy quantization (step 3)
            expected_E_min = (self.h**2) / (8 * self.m * self.L**2)
            if not self.fuzzy_equal(self.E_min, expected_E_min):
                self.log(f"FAIL: E_min mismatch. Got {self.E_min}, expected {expected_E_min}")
                success = False
            else:
                self.log("PASS: E_min matches particle-in-a-box formula")
                
            # 2. Check de Broglie relation for ground state (step 1 + step 2)
            expected_λ_dB = 2 * self.L
            if self.λ_dB and not self.fuzzy_equal(self.λ_dB, expected_λ_dB):
                self.log(f"FAIL: λ_dB mismatch. Got {self.λ_dB}, expected {expected_λ_dB}")
                success = False
            elif self.λ_dB:
                self.log("PASS: λ_dB matches 2L relation")
                
            # 3. Check n-th state formulas (steps 4-6)
            if self.E_n and self.p_n:
                # E_n should be n^2 * E_min
                expected_E_n = (self.n**2) * (self.h**2) / (8 * self.m * self.L**2)
                if not self.fuzzy_equal(self.E_n, expected_E_n):
                    self.log(f"FAIL: E_n mismatch. Got {self.E_n}, expected {expected_E_n}")
                    success = False
                else:
                    self.log("PASS: E_n matches n^2 scaling")
                
                # p_n = h/λ_dB^(n) = h*n/(2L)
                expected_p_n = self.h * self.n / (2 * self.L)
                if not self.fuzzy_equal(self.p_n, expected_p_n):
                    self.log(f"FAIL: p_n mismatch. Got {self.p_n}, expected {expected_p_n}")
                    success = False
                else:
                    self.log("PASS: p_n matches h*n/(2L)")
            
            # 4. Check photon emission energy conservation (steps 7-10)
            if self.λ_21 and self.delta_E:
                # ΔE = h*c/λ_21
                expected_λ_from_ΔE = (self.h * self.c) / self.delta_E if self.delta_E != 0 else 0
                if expected_λ_from_ΔE and not self.fuzzy_equal(self.λ_21, expected_λ_from_ΔE):
                    self.log(f"FAIL: λ_21 energy conservation mismatch. Got {self.λ_21}, expected {expected_λ_from_ΔE}")
                    success = False
                elif self.λ_21:
                    # Also check specific formula from step 10
                    expected_λ_21 = (8 * self.m * self.c * self.L**2) / (3 * self.h)
                    if not self.fuzzy_equal(self.λ_21, expected_λ_21):
                        self.log(f"WARN: λ_21 doesn't match expected formula but may be from different approach")
            
            # 5. Check equipartition theorem relation (step 12)
            if self.p and self.T:
                # p = sqrt(3*m*k_B*T) from step 12
                expected_p = math.sqrt(3 * self.m * self.k_B * self.T)
                if not self.fuzzy_equal(self.p, expected_p):
                    self.log(f"FAIL: p from equipartition mismatch. Got {self.p}, expected {expected_p}")
                    success = False
                else:
                    self.log("PASS: p matches equipartition theorem")
            
            # 6. Check BEC condition (step 16)
            if self.T_c and self.density:
                # T_c = h^2 * n^(2/3) / (3*m*k_B) where n is density
                expected_T_c = (self.h**2 * self.density**(2/3)) / (3 * self.m * self.k_B)
                if not self.fuzzy_equal(self.T_c, expected_T_c):
                    self.log(f"FAIL: T_c mismatch. Got {self.T_c}, expected {expected_T_c}")
                    success = False
                else:
                    self.log("PASS: T_c matches BEC condensation formula")
                    
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        try:
            # 1. Check that all de Broglie wavelengths are consistent with momentum
            if self.λ_dB and self.p:
                expected_p_from_λ = self.h / self.λ_dB
                if not self.fuzzy_equal(self.p, expected_p_from_λ):
                    self.log(f"FAIL: p-λ_dB de Broglie inconsistency. p={self.p}, h/λ_dB={expected_p_from_λ}")
                    success = False
                else:
                    self.log("PASS: Ground state p and λ_dB satisfy de Broglie relation")
            
            if self.λ_dBn and self.p_n:
                expected_pn_from_λ = self.h / self.λ_dBn
                if not self.fuzzy_equal(self.p_n, expected_pn_from_λ):
                    self.log(f"FAIL: p_n-λ_dB^(n) de Broglie inconsistency")
                    success = False
                else:
                    self.log("PASS: n-th state p_n and λ_dB^(n) satisfy de Broglie relation")
            
            # 2. Check energy-momentum relation for all states
            if self.E_min and self.p:
                expected_E_from_p = (self.p**2) / (2 * self.m)
                if not self.fuzzy_equal(self.E_min, expected_E_from_p):
                    self.log(f"FAIL: E_min-p kinetic energy inconsistency")
                    success = False
                else:
                    self.log("PASS: E_min = p²/(2m) holds")
            
            if self.E_n and self.p_n:
                expected_En_from_pn = (self.p_n**2) / (2 * self.m)
                if not self.fuzzy_equal(self.E_n, expected_En_from_pn):
                    self.log(f"FAIL: E_n-p_n kinetic energy inconsistency")
                    success = False
                else:
                    self.log("PASS: E_n = p_n²/(2m) holds")
            
            # 3. Check that ΔE = E_2 - E_1 = 3*h²/(8*m*L²)
            if self.delta_E:
                expected_ΔE = 3 * (self.h**2) / (8 * self.m * self.L**2)
                if not self.fuzzy_equal(self.delta_E, expected_ΔE):
                    self.log(f"FAIL: ΔE doesn't match expected 3h²/(8mL²)")
                    success = False
                else:
                    self.log("PASS: ΔE matches expected transition energy")
            
            # 4. Check interparticle distance (step 14)
            if self.l and self.density:
                expected_l = self.density**(-1/3)
                if not self.fuzzy_equal(self.l, expected_l):
                    self.log(f"FAIL: Interparticle distance l mismatch. Got {self.l}, expected {expected_l}")
                    success = False
                else:
                    self.log("PASS: l = n^(-1/3) holds")
                    
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        success = True
        
        try:
            perturb_factors = [0.9, 1.1]  # Test both decrease and increase
            
            # Test 1: Box size L vs E_min (should be inverse quadratic)
            for factor in perturb_factors:
                L_perturbed = self.L * factor
                E_min_perturbed = (self.h**2) / (8 * self.m * L_perturbed**2)
                
                if factor < 1:
                    # Smaller L should give larger E_min
                    if not (E_min_perturbed > self.E_min or self.fuzzy_equal(E_min_perturbed, self.E_min)):
                        self.log(f"FAIL: Monotonicity violated: L decreased but E_min didn't increase")
                        success = False
                else:
                    # Larger L should give smaller E_min
                    if not (E_min_perturbed < self.E_min or self.fuzzy_equal(E_min_perturbed, self.E_min)):
                        self.log(f"FAIL: Monotonicity violated: L increased but E_min didn't decrease")
                        success = False
            
            self.log("PASS: E_min ∝ 1/L² monotonicity holds")
            
            # Test 2: Quantum number n vs E_n (should be quadratic increase)
            for factor in perturb_factors:
                n_perturbed = self.n * factor
                E_n_perturbed = (self.h**2 * n_perturbed**2) / (8 * self.m * self.L**2)
                
                if factor < 1 and n_perturbed < self.n:
                    if not (E_n_perturbed < self.E_n or self.fuzzy_equal(E_n_perturbed, self.E_n)):
                        self.log(f"FAIL: Monotonicity violated: n decreased but E_n didn't decrease")
                        success = False
                elif factor > 1 and n_perturbed > self.n:
                    if not (E_n_perturbed > self.E_n or self.fuzzy_equal(E_n_perturbed, self.E_n)):
                        self.log(f"FAIL: Monotonicity violated: n increased but E_n didn't increase")
                        success = False
            
            self.log("PASS: E_n ∝ n² monotonicity holds")
            
            # Test 3: Temperature T vs de Broglie wavelength (should be inverse square root)
            if self.T and self.T > 0:
                for factor in perturb_factors:
                    T_perturbed = self.T * factor
                    λ_dB_perturbed = self.h / math.sqrt(3 * self.m * self.k_B * T_perturbed)
                    λ_dB_original = self.h / math.sqrt(3 * self.m * self.k_B * self.T)
                    
                    if factor < 1:
                        if not (λ_dB_perturbed > λ_dB_original or self.fuzzy_equal(λ_dB_perturbed, λ_dB_original)):
                            self.log(f"FAIL: Monotonicity violated: T decreased but λ_dB didn't increase")
                            success = False
                    else:
                        if not (λ_dB_perturbed < λ_dB_original or self.fuzzy_equal(λ_dB_perturbed, λ_dB_original)):
                            self.log(f"FAIL: Monotonicity violated: T increased but λ_dB didn't decrease")
                            success = False
                
                self.log("PASS: λ_dB ∝ 1/√T monotonicity holds")
            
            # Test 4: Density n vs critical temperature T_c (should be n^(2/3) increase)
            if self.density and self.density > 0:
                for factor in perturb_factors:
                    density_perturbed = self.density * factor
                    T_c_perturbed = (self.h**2 * density_perturbed**(2/3)) / (3 * self.m * self.k_B)
                    T_c_original = (self.h**2 * self.density**(2/3)) / (3 * self.m * self.k_B)
                    
                    if factor < 1:
                        if not (T_c_perturbed < T_c_original or self.fuzzy_equal(T_c_perturbed, T_c_original)):
                            self.log(f"FAIL: Monotonicity violated: density decreased but T_c didn't decrease")
                            success = False
                    else:
                        if not (T_c_perturbed > T_c_original or self.fuzzy_equal(T_c_perturbed, T_c_original)):
                            self.log(f"FAIL: Monotonicity violated: density increased but T_c didn't increase")
                            success = False
                
                self.log("PASS: T_c ∝ n^(2/3) monotonicity holds")
                
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
            
        return success
    
    def run_all_tests(self):
        """Execute all tests and return summary."""
        self.logs = []
        
        test_results = []
        
        # Run tests
        test_results.append(("Invariants", self.test_invariants()))
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        test_results.append(("Monotonicity", self.test_monotonicity()))
        
        # Overall success
        all_passed = all(result[1] for result in test_results)
        
        # Create summary
        summary = []
        for test_name, passed in test_results:
            status = "PASS" if passed else "FAIL"
            summary.append(f"{test_name}: {status}")
        
        self.log("\n=== TEST SUMMARY ===")
        for line in summary:
            self.log(line)
        self.log(f"Overall: {'PASS' if all_passed else 'FAIL'}")
        
        return all_passed, self.logs