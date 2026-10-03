import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results without direct attribute access."""
        self.solution = solution_instance
        
        # Extract inputs from solution instance using safe getattr
        self.Q = getattr(solution_instance, 'Q', 
                        getattr(solution_instance, 'inputs', {}).get('Q', -5e-8))
        self.q = getattr(solution_instance, 'q',
                        getattr(solution_instance, 'inputs', {}).get('q', 4e-6))
        self.r = getattr(solution_instance, 'r',
                        getattr(solution_instance, 'inputs', {}).get('r', 0.3))
        self.theta = getattr(solution_instance, 'theta',
                            getattr(solution_instance, 'inputs', {}).get('theta', 45))
        self.g = getattr(solution_instance, 'g',
                        getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        self.k = getattr(solution_instance, 'k',
                        getattr(solution_instance, 'inputs', {}).get('k', 9.0e9))
        
        # Extract results safely - using .results if available
        results_dict = getattr(solution_instance, 'results', {})
        self.m = results_dict.get('m', 
                                 getattr(solution_instance, 'm', 0.0))
        self.F_coulomb = results_dict.get('F_Coulomb',
                                         results_dict.get('F_{Coulomb}',
                                                         getattr(solution_instance, 'F_coulomb', 0.0)))
        
        # Convert angle to radians for calculations
        self.theta_rad = math.radians(self.theta)
        
        self.logs = []
        
    def _log(self, message, status="INFO"):
        """Helper to log messages with status."""
        self.logs.append(f"[{status}] {message}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison following robustness rules."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physics invariants: Coulomb's Law and equilibrium conditions."""
        try:
            # Test 1: Coulomb's Law
            q1, q2, r, k = sp.symbols('q1 q2 r k')
            F_symbolic = k * abs(q1 * q2) / r**2  # Magnitude only
            
            # Numerical check
            F_expected = self.k * abs(self.Q * self.q) / (self.r ** 2)
            
            if not self._is_close(self.F_coulomb, F_expected):
                self._log(f"Coulomb's Law violation: F_calculated={F_expected:.6f}, F_result={self.F_coulomb:.6f}", "ERROR")
                return False
            else:
                self._log(f"Coulomb's Law verified: F={self.F_coulomb:.6f} N", "PASS")
            
            # Test 2: Equilibrium condition (force balance)
            # Horizontal: T*sinθ = F_coulomb
            # Vertical: T*cosθ = m*g
            # Therefore: F_coulomb = m*g*tanθ
            
            # Symbolic equilibrium
            m_sym, g_sym, theta_sym, F_sym = sp.symbols('m g theta F')
            equilibrium_eq = sp.Eq(F_sym, m_sym * g_sym * sp.tan(theta_sym))
            
            # Numerical check
            F_from_equilibrium = self.m * self.g * math.tan(self.theta_rad)
            
            if not self._is_close(self.F_coulomb, F_from_equilibrium):
                self._log(f"Equilibrium violation: F_from_eq={F_from_equilibrium:.6f}, F_coulomb={self.F_coulomb:.6f}", "ERROR")
                return False
            else:
                self._log(f"Equilibrium condition verified: m*g*tanθ={F_from_equilibrium:.6f} N", "PASS")
            
            return True
            
        except Exception as e:
            self._log(f"Invariants test failed with error: {str(e)}", "ERROR")
            self._log(traceback.format_exc(), "DEBUG")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # According to steps analysis:
            # Step 1: r = 0.3 m (already in inputs)
            # Step 2: F_coulomb = 0.02 N
            # Step 3: m = 0.002 kg
            
            # Check Step 2: F_coulomb = k*|Q*q|/r²
            F_step2 = self.k * abs(self.Q * self.q) / (self.r ** 2)
            
            if not self._is_close(F_step2, 0.02):
                self._log(f"Step 2 inconsistency: Calculated F={F_step2:.6f}, Expected 0.02", "ERROR")
                return False
            else:
                self._log(f"Step 2 consistent: F_coulomb={F_step2:.6f} N ≈ 0.02 N", "PASS")
            
            # Check Step 3: m = F_coulomb / (g * tanθ)
            m_step3 = F_step2 / (self.g * math.tan(self.theta_rad))
            
            if not self._is_close(m_step3, 0.002):
                self._log(f"Step 3 inconsistency: Calculated m={m_step3:.6f}, Expected 0.002", "ERROR")
                return False
            else:
                self._log(f"Step 3 consistent: m={m_step3:.6f} kg ≈ 0.002 kg", "PASS")
            
            # Final check: compare with results
            if not self._is_close(self.m, m_step3):
                self._log(f"Final mass inconsistency: result m={self.m:.6f}, trace m={m_step3:.6f}", "ERROR")
                return False
            else:
                self._log(f"Trace consistency fully verified", "PASS")
            
            return True
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", "ERROR")
            self._log(traceback.format_exc(), "DEBUG")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends, then restore."""
        try:
            original_m = self.m
            original_F = self.F_coulomb
            
            # Test 1: Increase |Q| by 10%, Coulomb force should increase
            Q_perturbed = self.Q * 1.1  # 10% increase
            F_perturbed = self.k * abs(Q_perturbed * self.q) / (self.r ** 2)
            
            if F_perturbed <= self.F_coulomb:
                self._log(f"Monotonicity violation: |Q|↑ should cause F↑, but F={F_perturbed:.6f} <= {self.F_coulomb:.6f}", "ERROR")
                return False
            else:
                self._log(f"Monotonicity 1 passed: |Q|↑ ({self.Q:.2e}→{Q_perturbed:.2e}) causes F↑ ({self.F_coulomb:.6f}→{F_perturbed:.6f})", "PASS")
            
            # Test 2: Increase distance r, Coulomb force should decrease
            r_perturbed = self.r * 1.1  # 10% increase
            F_perturbed_r = self.k * abs(self.Q * self.q) / (r_perturbed ** 2)
            
            if F_perturbed_r >= self.F_coulomb:
                self._log(f"Monotonicity violation: r↑ should cause F↓, but F={F_perturbed_r:.6f} >= {self.F_coulomb:.6f}", "ERROR")
                return False
            else:
                self._log(f"Monotonicity 2 passed: r↑ ({self.r:.3f}→{r_perturbed:.3f}) causes F↓ ({self.F_coulomb:.6f}→{F_perturbed_r:.6f})", "PASS")
            
            # Test 3: Increase angle θ (keeping F_coulomb fixed), mass should decrease
            # From equilibrium: m = F_coulomb / (g * tanθ)
            theta_perturbed = self.theta * 1.1  # 10% increase
            theta_perturbed_rad = math.radians(theta_perturbed)
            m_perturbed = self.F_coulomb / (self.g * math.tan(theta_perturbed_rad))
            
            if m_perturbed >= self.m:
                self._log(f"Monotonicity violation: θ↑ should cause m↓, but m={m_perturbed:.6f} >= {self.m:.6f}", "ERROR")
                return False
            else:
                self._log(f"Monotonicity 3 passed: θ↑ ({self.theta:.1f}°→{theta_perturbed:.1f}°) causes m↓ ({self.m:.6f}→{m_perturbed:.6f})", "PASS")
            
            # Verify original values unchanged (sanity check)
            if not (self._is_close(self.m, original_m) and self._is_close(self.F_coulomb, original_F)):
                self._log("Warning: State may have been altered during monotonicity test", "WARN")
            
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", "ERROR")
            self._log(traceback.format_exc(), "DEBUG")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []  # Reset logs
        
        self._log("Starting Physics Audit for Electrostatic Equilibrium Problem", "INFO")
        self._log(f"Inputs: Q={self.Q:.2e} C, q={self.q:.2e} C, r={self.r:.3f} m, θ={self.theta}°, g={self.g} m/s², k={self.k:.2e} N·m²/C²", "INFO")
        self._log(f"Results: m={self.m:.6f} kg, F_coulomb={self.F_coulomb:.6f} N", "INFO")
        
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} Test ---", "INFO")
            if not test_func():
                success = False
                self._log(f"{test_name} test FAILED", "ERROR")
            else:
                self._log(f"{test_name} test PASSED", "PASS")
        
        # Final summary
        self._log("\n" + "="*50, "INFO")
        if success:
            self._log("ALL PHYSICS CHECKS PASSED", "SUCCESS")
        else:
            self._log("SOME PHYSICS CHECKS FAILED", "FAILURE")
        
        return success, self.logs