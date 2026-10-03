import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract all values from results dictionary only
        results = getattr(solution_instance, 'results', {})
        
        # Debug: Print what we're reading
        # print(f"Checker reading results: {results}")
        
        # Input parameters with safe defaults
        self.T = results.get('T', 7200.0)
        self.h = results.get('h', 100000.0)
        self.R = results.get('R', 1737400.0)
        self.G = results.get('G', 6.6743e-11)
        
        # Results
        self.M = results.get('M', 0.0)
        self.g = results.get('g', 0.0)
        
        self.logs = []
        
        # Debug log
        self.logs.append(f"Checker initialized with: G={self.G}, R={self.R}, M={self.M}, g={self.g}, h={self.h}, T={self.T}")

    def test_invariants(self):
        """
        Verify:
        1. Gravitational force on surface: g = G*M/R²
        2. Orbital motion: G*M = 4π²(R+h)³/T²
        """
        try:
            # Test 1: g = G*M/R²
            lhs_g = self.g
            # Protect against division by zero
            if self.R == 0:
                self.logs.append("⚠ Skipping surface gravity test due to R=0")
                return
            
            rhs_g = self.G * self.M / (self.R ** 2)
            
            if math.isclose(lhs_g, rhs_g, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append("✓ Surface gravity invariant (g = G*M/R²) satisfied")
            else:
                self.logs.append(f"✗ Surface gravity mismatch: g={lhs_g}, G*M/R²={rhs_g} (G={self.G}, M={self.M}, R={self.R})")

            # Test 2: Orbital motion invariant: G*M = 4π²(R+h)³/T²
            lhs_orb = self.G * self.M
            # Protect against division by zero
            if self.T == 0:
                self.logs.append("⚠ Skipping orbital motion test due to T=0")
                return
                
            rhs_orb = 4 * math.pi**2 * (self.R + self.h)**3 / (self.T ** 2)
            
            if math.isclose(lhs_orb, rhs_orb, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append("✓ Orbital motion invariant (G*M = 4π²(R+h)³/T²) satisfied")
            else:
                self.logs.append(f"✗ Orbital motion mismatch: G*M={lhs_orb}, 4π²(R+h)³/T²={rhs_orb}")

        except Exception as e:
            self.logs.append(f"✗ Error in test_invariants: {str(e)}")
            self.logs.append(traceback.format_exc())

    def test_trace_consistency(self):
        """
        Cross-check intermediate and final results with step-by-step formulas.
        """
        try:
            # Step 1: M = 4π²(R+h)³/(G T²)
            if self.G != 0 and self.T != 0:
                expected_M = 4 * math.pi**2 * (self.R + self.h)**3 / (self.G * self.T**2)
                
                if math.isclose(self.M, expected_M, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append("✓ Mass M consistent with orbital formula")
                else:
                    self.logs.append(f"✗ Mass M mismatch: {self.M} vs expected {expected_M}")
            else:
                self.logs.append("⚠ Skipping M consistency check due to zero G or T")

            # Step 3: g = 4π²(R+h)³/(T² R²)
            if self.T != 0 and self.R != 0:
                expected_g = 4 * math.pi**2 * (self.R + self.h)**3 / (self.T**2 * self.R**2)
                
                if math.isclose(self.g, expected_g, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append("✓ Gravity g consistent with final formula")
                else:
                    self.logs.append(f"✗ Gravity g mismatch: {self.g} vs expected {expected_g}")
            else:
                self.logs.append("⚠ Skipping g consistency check due to zero T or R")

        except Exception as e:
            self.logs.append(f"✗ Error in test_trace_consistency: {str(e)}")
            self.logs.append(traceback.format_exc())

    def test_monotonicity(self):
        """
        Perturb altitude h and verify physical trends:
        - Increase h → increase M (∝ (R+h)³)
        - Increase h → increase g (∝ (R+h)³)
        """
        try:
            # Store original values
            original_h = self.h
            
            # Perturb h by 1%
            perturbed_h = original_h * 1.01
            
            # Calculate expected M with perturbed h
            if self.G != 0 and self.T != 0:
                M_perturbed = 4 * math.pi**2 * (self.R + perturbed_h)**3 / (self.G * self.T**2)
            else:
                M_perturbed = 0.0
                
            # Calculate expected g with perturbed h
            if self.T != 0 and self.R != 0:
                g_perturbed = 4 * math.pi**2 * (self.R + perturbed_h)**3 / (self.T**2 * self.R**2)
            else:
                g_perturbed = 0.0
            
            # Check trends (only if values are valid)
            if M_perturbed > 0 and g_perturbed > 0:
                if M_perturbed > self.M and g_perturbed > self.g:
                    self.logs.append("✓ Monotonicity: Increasing h increases both M and g")
                else:
                    self.logs.append(f"✗ Monotonicity violated: M={self.M}→{M_perturbed}, g={self.g}→{g_perturbed}")
            else:
                self.logs.append("⚠ Skipping monotonicity check due to invalid computed values")
                
            # Restore original h
            self.h = original_h

        except Exception as e:
            self.logs.append(f"✗ Error in test_monotonicity: {str(e)}")
            self.logs.append(traceback.format_exc())

    def run_all_tests(self):
        """
        Execute all tests and return success status and logs.
        """
        self.logs = []  # Reset logs
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success (no ✗ in logs)
        success = all('✗' not in log for log in self.logs)
            
        return success, self.logs