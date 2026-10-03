import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution instance.
        """
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Physical constants
        self.g = 9.8  # m/s²
        self.k = 8.99e9  # N·m²/C²
        
        # Problem-specific inputs with safe defaults
        self.h = 0.1  # 10 cm in meters
        self.angle_deg = 30  # degrees
        self.theta = math.radians(self.angle_deg)  # radians
        
        # Extract mass from results or use default
        self.m = self.results.get('m', 30*math.sqrt(3)/1000)  # 30√3 g = 0.03√3 kg
        
        # Extract results with safe defaults
        self.L = float(self.results.get('L', 0.0))
        self.F = float(self.results.get('F', 0.0))
        self.Q = float(self.results.get('Q', 0.0))
        
        # For L, if provided in cm (as per steps), convert to meters
        if self.L > 10:  # Heuristic: if L > 10, likely in cm
            self.L = self.L / 100.0
        
        # Store original mass for monotonicity test
        self._original_m = float(self.m)
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison for physics data."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics laws: equilibrium and Coulomb's law."""
        logs = []
        success = True
        
        try:
            # 1. Test equilibrium condition: F = m*g*tan(θ)
            expected_F = self.m * self.g * math.tan(self.theta)
            if not self._is_close(self.F, expected_F):
                logs.append(f"FAIL: Equilibrium condition violated. F={self.F:.3e}, "
                          f"m*g*tan(θ)={expected_F:.3e}, diff={abs(self.F-expected_F):.2e}")
                success = False
            else:
                logs.append(f"PASS: Equilibrium condition satisfied (F ≈ m*g*tan(θ))")
            
            # 2. Test Coulomb's law: F = k*Q²/L²
            if self.L > 0:
                expected_F_coulomb = self.k * self.Q**2 / self.L**2
                if not self._is_close(self.F, expected_F_coulomb):
                    logs.append(f"FAIL: Coulomb's law violated. F={self.F:.3e}, "
                              f"kQ²/L²={expected_F_coulomb:.3e}, diff={abs(self.F-expected_F_coulomb):.2e}")
                    success = False
                else:
                    logs.append(f"PASS: Coulomb's law satisfied (F ≈ kQ²/L²)")
            else:
                logs.append("SKIP: Coulomb's law test (L=0)")
                success = False
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results (L, F) lead to consistent final Q."""
        logs = []
        success = True
        
        try:
            # Step 3: L = h / tan(θ)
            expected_L = self.h / math.tan(self.theta)
            if not self._is_close(self.L, expected_L):
                logs.append(f"FAIL: L inconsistent. Computed L={self.L:.3e}, "
                          f"Expected h/tan(θ)={expected_L:.3e}")
                success = False
            else:
                logs.append(f"PASS: L = h/tan(θ) = {self.L:.3e} m")
            
            # Step 4: F = m*g*tan(θ) (already tested in invariants)
            expected_F = self.m * self.g * math.tan(self.theta)
            if not self._is_close(self.F, expected_F):
                logs.append(f"FAIL: F inconsistent. Computed F={self.F:.3e}, "
                          f"Expected m*g*tan(θ)={expected_F:.3e}")
                success = False
            else:
                logs.append(f"PASS: F = m*g*tan(θ) = {self.F:.3e} N")
            
            # Step 5: Q = sqrt(F*L²/k)
            if self.F > 0 and self.L > 0:
                expected_Q = math.sqrt(self.F * self.L**2 / self.k)
                if not self._is_close(self.Q, expected_Q):
                    logs.append(f"FAIL: Q inconsistent. Computed Q={self.Q:.3e}, "
                              f"Expected sqrt(F*L²/k)={expected_Q:.3e}")
                    success = False
                else:
                    logs.append(f"PASS: Q = sqrt(F*L²/k) = {self.Q:.3e} C")
                    
                # Cross-check with target answer
                target_Q = 1.0e-6
                if not self._is_close(self.Q, target_Q):
                    logs.append(f"WARN: Q differs from target {target_Q:.1e} by "
                              f"{abs(self.Q-target_Q)/target_Q*100:.1f}%")
            else:
                logs.append("SKIP: Q consistency test (F or L invalid)")
                success = False
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb mass m and verify expected trend in Q.
        Restore original mass after test.
        """
        logs = []
        success = True
        
        try:
            # Store original state
            original_m = self.m
            original_Q = self.Q
            
            # Perturb mass by +10%
            self.m = original_m * 1.1
            
            # Recalculate expected quantities using physics formulas
            new_F = self.m * self.g * math.tan(self.theta)  # From equilibrium
            new_Q = math.sqrt(new_F * self.L**2 / self.k)   # From Coulomb's law
            
            # Check monotonicity: if m increases, Q should increase
            if new_Q <= original_Q:
                logs.append(f"FAIL: Monotonicity violated. m↑ {original_m:.3e}→{self.m:.3e}, "
                          f"but Q {original_Q:.3e}→{new_Q:.3e}")
                success = False
            else:
                logs.append(f"PASS: Monotonicity holds. m↑ {original_m:.3e}→{self.m:.3e}, "
                          f"Q↑ {original_Q:.3e}→{new_Q:.3e} (Δ={new_Q-original_Q:.2e})")
            
            # Restore original mass
            self.m = original_m
            
            # Additional test: perturb height h
            original_h = self.h
            original_L = self.L
            
            # Increase h by 10%
            self.h = original_h * 1.1
            new_L = self.h / math.tan(self.theta)
            
            # With fixed F (from original m), Q should increase when L increases
            # From Coulomb: Q ∝ √(F*L²) ∝ L (since F constant for fixed m)
            if new_L <= original_L:
                logs.append(f"WARN: L not monotonic with h. h↑ but L {original_L:.3e}→{new_L:.3e}")
            else:
                logs.append(f"INFO: L monotonic with h. h↑ {original_h:.3e}→{self.h:.3e}, "
                          f"L↑ {original_L:.3e}→{new_L:.3e}")
            
            # Restore h
            self.h = original_h
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            # Ensure we restore even on error
            self.m = original_m
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                for log in logs:
                    all_logs.append(f"  {log}")
                if not success:
                    overall_success = False
                    all_logs.append(f"  RESULT: FAILED")
                else:
                    all_logs.append(f"  RESULT: PASSED")
            except Exception as e:
                all_logs.append(f"  ERROR: Test crashed - {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append(f"Charge Q: {self.Q:.3e} C")
        all_logs.append(f"Target Q: 1.0e-6 C")
        all_logs.append(f"Difference: {abs(self.Q-1e-6):.2e} C ({abs(self.Q-1e-6)/1e-6*100:.1f}%)")
        
        return overall_success, all_logs