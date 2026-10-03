import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs from solution_instance or its results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.t = getattr(solution_instance, 't', 27.0)  # Initial temperature in Celsius
        self.p = getattr(solution_instance, 'p', 1.050e5)  # Initial pressure in Pa
        
        # Extract results with safe access
        self.T = getattr(solution_instance, 'T', self.results.get('T', 0.0))  # Initial temp in K
        self.T_prime = getattr(solution_instance, 'T_prime', 
                               self.results.get("T'", self.results.get('T_prime', 0.0)))  # Final temp in K
        self.p_prime = getattr(solution_instance, 'p_prime', 
                               self.results.get("p'", self.results.get('p_prime', 0.0)))  # Final pressure
        self.ratio = getattr(solution_instance, 'ratio', 
                             self.results.get("V'/V", self.results.get('ratio', 0.0)))  # Volume ratio
        
        # Expected values from steps analysis for cross-checking
        self.expected_T = 300.0  # From step 1: 27 + 273
        self.expected_T_prime = 310.0  # From step 2: 37 + 273 (implied)
        self.expected_p_prime = 1.085e5  # From step 3
        self.expected_ratio = 0.97  # From step 4
        
    def test_invariants(self):
        """Test conservation laws and physical principles."""
        logs = []
        success = True
        
        try:
            # 1. Check temperature conversions
            # The checker should accept either 273 or 273.15 conversion
            T_calc_273 = self.t + 273.0
            T_calc_27315 = self.t + 273.15
            if not (math.isclose(self.T, T_calc_273, rel_tol=1e-2, abs_tol=1.0) or 
                    math.isclose(self.T, T_calc_27315, rel_tol=1e-2, abs_tol=1.0)):
                logs.append(f"Temperature conversion mismatch: T={self.T} vs possible {T_calc_273} or {T_calc_27315}")
                success = False
            else:
                logs.append("✓ Temperature conversion correct")
            
            # 2. Check Charles's Law (isochoric process)
            # p/T should remain constant for constant volume
            if self.T > 0 and self.T_prime > 0:
                charles_ratio_initial = self.p / self.T
                charles_ratio_final = self.p_prime / self.T_prime
                
                if not math.isclose(charles_ratio_initial, charles_ratio_final, 
                                   rel_tol=1e-2, abs_tol=1.0):
                    logs.append(f"Charles's Law violation: p/T ratios differ by {abs(charles_ratio_initial - charles_ratio_final)}")
                    success = False
                else:
                    logs.append("✓ Charles's Law satisfied (isochoric process)")
            
            # 3. Check Boyle's Law (isothermal process)
            # For the ratio calculation, Boyle's Law should hold: p*V = p'*V'
            # So V'/V = p/p'
            if self.p_prime > 0:
                boyle_ratio = self.p / self.p_prime
                if not math.isclose(self.ratio, boyle_ratio, rel_tol=1e-2, abs_tol=0.01):
                    logs.append(f"Boyle's Law violation: ratio={self.ratio} vs expected {boyle_ratio}")
                    success = False
                else:
                    logs.append("✓ Boyle's Law satisfied (isothermal process)")
            
            # 4. Check ideal gas law consistency for the compression step
            # The compression is isothermal at temperature T (300 K)
            # Initial state before compression: p', V, T (after heating)
            # Final state after compression: p, V', T (compressed to original pressure)
            # So p' * V / T = p * V' / T  => p' * V = p * V' => V'/V = p'/p
            # But we already have V'/V = p/p' from Boyle's Law test above
            # Actually, both are correct depending on interpretation
            # The important check is that the product p*V/T is constant for each process
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final results."""
        logs = []
        success = True
        
        try:
            # Check against expected values from steps analysis
            # Allow either 273 or 273.15 conversion
            if not (math.isclose(self.T, 300.0, rel_tol=1e-2, abs_tol=1.0) or 
                    math.isclose(self.T, 300.15, rel_tol=1e-2, abs_tol=1.0)):
                logs.append(f"Initial temperature mismatch: {self.T} vs expected ~300")
                success = False
            else:
                logs.append("✓ Initial temperature matches step 1")
            
            if not (math.isclose(self.T_prime, 310.0, rel_tol=1e-2, abs_tol=1.0) or 
                    math.isclose(self.T_prime, 310.15, rel_tol=1e-2, abs_tol=1.0)):
                logs.append(f"Final temperature mismatch: {self.T_prime} vs expected ~310")
                success = False
            else:
                logs.append("✓ Final temperature matches step 2")
            
            # Check Charles's Law calculation from step 3
            p_prime_calc = self.p * (self.T_prime / self.T)
            if not math.isclose(self.p_prime, p_prime_calc, rel_tol=1e-2, abs_tol=100.0):
                logs.append(f"Pressure calculation mismatch: {self.p_prime} vs expected {p_prime_calc}")
                success = False
            else:
                logs.append("✓ Pressure calculation matches step 3 (Charles's Law)")
            
            # Check against expected pressure (allow slight rounding differences)
            if not math.isclose(self.p_prime, self.expected_p_prime, rel_tol=1e-2, abs_tol=100.0):
                logs.append(f"Final pressure mismatch: {self.p_prime} vs expected {self.expected_p_prime}")
                success = False
            else:
                logs.append("✓ Final pressure matches expected value")
            
            # Check Boyle's Law calculation from step 4
            ratio_calc = self.p / self.p_prime
            if not math.isclose(self.ratio, ratio_calc, rel_tol=1e-2, abs_tol=0.01):
                logs.append(f"Volume ratio mismatch: {self.ratio} vs calculated {ratio_calc}")
                success = False
            else:
                logs.append("✓ Volume ratio matches step 4 (Boyle's Law)")
            
            # Check against expected ratio (allow slight rounding)
            if not math.isclose(self.ratio, self.expected_ratio, rel_tol=1e-2, abs_tol=0.01):
                logs.append(f"Volume ratio mismatch: {self.ratio} vs expected {self.expected_ratio}")
                success = False
            else:
                logs.append("✓ Volume ratio matches expected value")
            
            # Check that ratio is less than 1 (volume decreased)
            if self.ratio >= 1.0:
                logs.append(f"Warning: Volume ratio {self.ratio} suggests volume didn't decrease")
                # Not a failure, just a warning
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test that perturbations produce physically expected trends."""
        logs = []
        success = True
        
        try:
            # Save original values
            original_T_prime = self.T_prime
            original_p_prime = self.p_prime
            original_ratio = self.ratio
            
            # Perturbation: increase final temperature by 10K
            perturbed_T_prime = original_T_prime + 10.0
            
            # Recalculate using physical laws
            # Charles's Law: p'/p = T'/T  (constant volume)
            perturbed_p_prime = self.p * (perturbed_T_prime / self.T)
            
            # Boyle's Law: V'/V = p/p'  (constant temperature)
            perturbed_ratio = self.p / perturbed_p_prime
            
            # Check trends
            # 1. Higher temperature should give higher pressure (Charles's Law)
            if not perturbed_p_prime > original_p_prime:
                logs.append(f"Monotonicity violation: Higher temperature {perturbed_T_prime}K "
                           f"did not increase pressure: {perturbed_p_prime} vs {original_p_prime}")
                success = False
            else:
                logs.append("✓ Higher temperature increases pressure (Charles's Law trend)")
            
            # 2. Higher pressure should give smaller volume ratio (Boyle's Law)
            if not perturbed_ratio < original_ratio:
                logs.append(f"Monotonicity violation: Higher pressure {perturbed_p_prime}Pa "
                           f"did not decrease volume ratio: {perturbed_ratio} vs {original_ratio}")
                success = False
            else:
                logs.append("✓ Higher pressure decreases volume (Boyle's Law trend)")
            
            # Test opposite perturbation: decrease temperature
            perturbed_T_prime2 = original_T_prime - 10.0
            perturbed_p_prime2 = self.p * (perturbed_T_prime2 / self.T)
            perturbed_ratio2 = self.p / perturbed_p_prime2
            
            # Lower temperature should give lower pressure
            if not perturbed_p_prime2 < original_p_prime:
                logs.append(f"Monotonicity violation: Lower temperature {perturbed_T_prime2}K "
                           f"did not decrease pressure: {perturbed_p_prime2} vs {original_p_prime}")
                success = False
            else:
                logs.append("✓ Lower temperature decreases pressure")
            
            # Lower pressure should give larger volume ratio
            if not perturbed_ratio2 > original_ratio:
                logs.append(f"Monotonicity violation: Lower pressure {perturbed_p_prime2}Pa "
                           f"did not increase volume ratio: {perturbed_ratio2} vs {original_ratio}")
                success = False
            else:
                logs.append("✓ Lower pressure increases volume")
            
            # Restore original values (important for subsequent tests)
            self.T_prime = original_T_prime
            self.p_prime = original_p_prime
            self.ratio = original_ratio
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            # Restore on error
            if 'original_T_prime' in locals():
                self.T_prime = original_T_prime
                self.p_prime = original_p_prime
                self.ratio = original_ratio
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return summary."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                all_logs.append(f"Test crashed: {traceback.format_exc()}")
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall result: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        return overall_success, all_logs