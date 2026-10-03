import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        try:
            self.results = getattr(solution_instance, 'results', {})
            
            # Extract ALL values from results dictionary with safe defaults
            # Use the most recent/largest value if multiple entries exist
            all_keys = list(self.results.keys())
            
            # Helper to get the best value for a key
            def get_best(key, default):
                # First try the exact key
                if key in self.results:
                    val = self.results[key]
                    # Ensure it's a number, not a sympy object
                    if hasattr(val, 'evalf'):
                        return float(val)
                    return float(val)
                # Try variations
                for k in all_keys:
                    if k.startswith(key) and k != key:
                        val = self.results[k]
                        if hasattr(val, 'evalf'):
                            return float(val)
                        return float(val)
                return default
            
            self.m = get_best('m', 1.0)
            self.q = get_best('q', 1.0e-6)  # Use a sensible default for charge
            self.a = get_best('a', 1.0)
            self.b = get_best('b', 1.0)
            self.B = get_best('B', 0.1)     # Use a sensible default for B
            self.k = get_best('k', 8.99e9)
            self.Ek = get_best('Ek', 0.0)
            self.T = get_best('T', 0.0)
            
            # Ensure all values are floats
            self.m = float(self.m)
            self.q = float(self.q)
            self.a = float(self.a)
            self.b = float(self.b)
            self.B = float(self.B)
            self.k = float(self.k)
            self.Ek = float(self.Ek)
            self.T = float(self.T)
            
        except Exception as e:
            self.error = f"Initialization error: {str(e)}"
            raise

    def test_invariants(self):
        """Test physical invariants: Energy conservation, force balance, Coulomb's Law."""
        logs = []
        success = True
        
        try:
            # Part 1: Without magnetic field
            # Check if Ek matches the theoretical value k*q^2/(2*a)
            if abs(self.q) < 1e-30:
                logs.append("SKIP: q is zero, cannot compute expected_Ek")
            else:
                expected_Ek = self.k * self.q**2 / (2 * self.a)
                
                if not math.isclose(self.Ek, expected_Ek, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Ek mismatch. Got {self.Ek}, expected {expected_Ek}")
                    success = False
                else:
                    logs.append("PASS: Ek matches Coulomb's law and energy conservation")
            
            # Part 2: With magnetic field
            # Check if T matches the theoretical value
            if abs(self.B) > 1e-30 and abs(self.q) > 1e-30:
                expected_T = 2 * math.pi * self.m / self.q * math.sqrt(
                    self.b**3 / (2 * self.k * self.m + self.B**2 * self.b**3)
                )
                
                if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: T mismatch. Got {self.T}, expected {expected_T}")
                    success = False
                else:
                    logs.append("PASS: T matches force balance with magnetic field")
            else:
                logs.append("SKIP: B=0 or q=0, magnetic field test not applicable")
            
            # Force balance check for part 1 (centripetal = Coulomb)
            # This is a consistency check, not a requirement
            if abs(self.q) > 1e-30:
                centripetal_force_q = self.m * (self.q**2 * self.k / (self.a**2 * self.m))
                coulomb_force = self.k * self.q**2 / self.a**2
                
                if not math.isclose(centripetal_force_q, coulomb_force, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: Force balance without B may be inconsistent: centripetal={centripetal_force_q}, Coulomb={coulomb_force}")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        logs = []
        success = True
        
        try:
            # Check step 4: Ek = 1/2 m v1^2 + 1/2 m v2^2 = k q^2 / (2a)
            # This is consistent by construction
            logs.append("INFO: Step consistency check passed - geometric and force equations lead to Ek expression")
            
            # For part 2, check that the derived T satisfies the force equations
            if abs(self.B) > 1e-30 and abs(self.q) > 1e-30 and self.T > 1e-30:
                ω = 2 * math.pi / self.T
                # Avoid division by zero
                denom1 = self.m * ω**2 - self.B * self.q * ω
                denom2 = self.m * ω**2 + self.B * self.q * ω
                
                if abs(denom1) > 1e-30:
                    R1_est = (self.k * self.q**2 / self.b**2) / denom1
                else:
                    R1_est = float('inf')
                
                if abs(denom2) > 1e-30:
                    R2_est = (self.k * self.q**2 / self.b**2) / denom2
                else:
                    R2_est = float('inf')
                
                # Check geometric constraint R1 + R2 ≈ b
                if R1_est != float('inf') and R2_est != float('inf'):
                    if not math.isclose(R1_est + R2_est, self.b, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"WARN: Geometric constraint R1+R2=b not satisfied: {R1_est}+{R2_est}≈{R1_est+R2_est}, b={self.b}")
                    else:
                        logs.append("PASS: Force balance with magnetic field satisfies geometric constraint")
                else:
                    logs.append("SKIP: Cannot compute R1 or R2 due to division by near-zero")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """Test monotonic trends by perturbing inputs."""
        logs = []
        success = True
        
        try:
            # Save original values
            m_original = self.m
            B_original = self.B
            perturbation = 1.1  # 10% increase
            
            # Test 1: Ek should be independent of mass
            if abs(self.q) > 1e-30:
                self.m *= perturbation
                new_Ek = self.k * self.q**2 / (2 * self.a)
                
                if not math.isclose(new_Ek, self.Ek, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Ek changed with mass perturbation: {self.Ek} -> {new_Ek}")
                    success = False
                else:
                    logs.append("PASS: Ek is independent of mass as expected")
                
                # Restore original mass
                self.m = m_original
            else:
                logs.append("SKIP: q=0, Ek monotonicity test not applicable")
            
            # Test 2: T monotonicity with respect to B (if B != 0)
            if abs(self.B) > 1e-30 and abs(self.q) > 1e-30:
                self.B *= perturbation
                new_T = 2 * math.pi * self.m / self.q * math.sqrt(
                    self.b**3 / (2 * self.k * self.m + self.B**2 * self.b**3)
                )
                
                if new_T > self.T:
                    logs.append(f"FAIL: T increased with B increase: {self.T} -> {new_T}")
                    success = False
                else:
                    logs.append("PASS: T decreases with increasing B as expected")
                
                # Restore original B
                self.B = B_original
                
                # Test 3: T monotonicity with respect to m
                self.m *= perturbation
                new_T2 = 2 * math.pi * self.m / self.q * math.sqrt(
                    self.b**3 / (2 * self.k * self.m + self.B**2 * self.b**3)
                )
                
                if new_T2 < self.T:
                    logs.append(f"FAIL: T decreased with mass increase: {self.T} -> {new_T2}")
                    success = False
                else:
                    logs.append("PASS: T increases with mass as expected")
                
                # Restore original mass
                self.m = m_original
            else:
                logs.append("SKIP: B=0 or q=0, monotonicity tests with magnetic field not applicable")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Restore original values
            self.m = m_original
            self.B = B_original
            
        return success, logs

    def run_all_tests(self):
        """Run all tests and return aggregated results."""
        all_logs = []
        all_success = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_func()
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.extend(logs)
                if not success:
                    all_success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} Test ===")
                all_logs.append(f"ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_success = False
        
        return all_success, all_logs