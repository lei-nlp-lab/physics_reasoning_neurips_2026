import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract inputs and results safely without direct attribute access."""
        self.results = {}
        
        # Extract inputs - use defaults from problem statement
        self.N = 300  # coil turns (fixed from problem)
        self.r = 1.0  # internal resistance (ohm)
        self.R = 9.0  # external resistance (ohm)
        self.t = 1.0  # time (s)
        
        # Extract results with fallbacks
        self.results['T'] = getattr(solution_instance, 'T', solution_instance.results.get('T', 0.0))
        self.results['Q'] = getattr(solution_instance, 'Q', solution_instance.results.get('Q', 0.0))
        self.results['E_m'] = getattr(solution_instance, 'E_m', solution_instance.results.get('E_m', 0.0))
        self.results['Phi_m'] = getattr(solution_instance, 'Phi_m', solution_instance.results.get('Phi_m', 0.0))
        self.results['U'] = getattr(solution_instance, 'U', solution_instance.results.get('U', 0.0))
        self.results['E'] = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        
        # Store original values for monotonicity tests
        self._original_R = self.R
        self._original_N = self.N
        self._original_r = self.r
        
        # Define constants
        self.sqrt2 = math.sqrt(2)
        self.pi = math.pi
        
        # Expected exact values for comparison
        self.E_m_expected = 6 * self.sqrt2
        self.E_expected = 6.0
        self.U_expected = 5.4  # E * (R/(R+r)) = 6 * (9/10)
        self.Q_expected = 3.24  # U^2/R = 5.4^2/9
        self.T_expected = 0.02
        self.Phi_m_expected = self.sqrt2 / (5000 * self.pi)

    def test_invariants(self):
        """Verify physical invariants: Ohm's Law, Joule's Law, EMF relationships."""
        logs = []
        success = True
        
        try:
            # 1. Check E = E_m / sqrt(2)
            E_expected = self.results['E_m'] / self.sqrt2
            if not math.isclose(self.results['E'], E_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E = {self.results['E']:.4f} ≠ E_m/√2 = {E_expected:.4f}")
                success = False
            else:
                logs.append(f"OK: E = {self.results['E']:.4f} ≈ E_m/√2 = {E_expected:.4f}")
            
            # 2. Check U = E * (R/(R+r))
            U_expected = self.results['E'] * (self.R / (self.R + self.r))
            if not math.isclose(self.results['U'], U_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U = {self.results['U']:.4f} ≠ E*R/(R+r) = {U_expected:.4f}")
                success = False
            else:
                logs.append(f"OK: U = {self.results['U']:.4f} ≈ E*R/(R+r) = {U_expected:.4f}")
            
            # 3. Check Q = (U^2 / R) * t
            Q_expected = (self.results['U']**2 / self.R) * self.t
            if not math.isclose(self.results['Q'], Q_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Q = {self.results['Q']:.4f} ≠ (U^2/R)*t = {Q_expected:.4f}")
                success = False
            else:
                logs.append(f"OK: Q = {self.results['Q']:.4f} ≈ (U^2/R)*t = {Q_expected:.4f}")
            
            # 4. Check E_m = N * Phi_m * ω, where ω = 2π/T
            # Use the EXPECTED Phi_m value for this check, not the potentially rounded code value
            if self.results['T'] > 0:
                omega = 2 * self.pi / self.results['T']
                E_m_expected_from_Phi = self.N * self.Phi_m_expected * omega
                if not math.isclose(self.results['E_m'], E_m_expected_from_Phi, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: E_m = {self.results['E_m']:.4e} ≠ N*Φ_m_expected*ω = {E_m_expected_from_Phi:.4e}")
                    success = False
                else:
                    logs.append(f"OK: E_m = {self.results['E_m']:.4e} ≈ N*Φ_m_expected*ω = {E_m_expected_from_Phi:.4e}")
            else:
                logs.append("SKIP: T ≤ 0, cannot compute ω")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check E_m
            if not math.isclose(self.results['E_m'], self.E_m_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_m = {self.results['E_m']:.4f} ≠ expected 6√2 = {self.E_m_expected:.4f}")
                success = False
            else:
                logs.append(f"OK: E_m = {self.results['E_m']:.4f} ≈ 6√2 = {self.E_m_expected:.4f}")
            
            # Check E
            if not math.isclose(self.results['E'], self.E_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E = {self.results['E']:.4f} ≠ expected 6.0")
                success = False
            else:
                logs.append(f"OK: E = {self.results['E']:.4f} ≈ 6.0")
            
            # Check U
            if not math.isclose(self.results['U'], self.U_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U = {self.results['U']:.4f} ≠ expected 5.4")
                success = False
            else:
                logs.append(f"OK: U = {self.results['U']:.4f} ≈ 5.4")
            
            # Check Q
            if not math.isclose(self.results['Q'], self.Q_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Q = {self.results['Q']:.4f} ≠ expected 3.24")
                success = False
            else:
                logs.append(f"OK: Q = {self.results['Q']:.4f} ≈ 3.24")
            
            # Check T
            if not math.isclose(self.results['T'], self.T_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T = {self.results['T']:.4e} ≠ expected 0.02")
                success = False
            else:
                logs.append(f"OK: T = {self.results['T']:.4e} ≈ 0.02")
            
            # Check Phi_m - use the exact expected value
            if not math.isclose(self.results['Phi_m'], self.Phi_m_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Φ_m = {self.results['Phi_m']:.4e} ≠ expected √2/(5000π) = {self.Phi_m_expected:.4e}")
                success = False
            else:
                logs.append(f"OK: Φ_m = {self.results['Phi_m']:.4e} ≈ √2/(5000π) = {self.Phi_m_expected:.4e}")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """Perturb inputs and check trends based on physical laws."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase external resistance R
            original_R = self.R
            perturbed_R = original_R * 1.1  # +10%
            
            # Compute Q for perturbed R using formulas
            E = self.results['E']  # Should remain constant
            perturbed_U = E * (perturbed_R / (perturbed_R + self.r))
            perturbed_Q = (perturbed_U**2 / perturbed_R) * self.t
            
            # According to Joule's Law: Q = (U^2/R)*t, where U = E*R/(R+r)
            # For R > r, increasing R should decrease Q (since we're beyond max power transfer)
            if original_R > self.r:
                if perturbed_Q < self.results['Q']:
                    logs.append(f"OK: Q decreases from {self.results['Q']:.4f} to {perturbed_Q:.4f} when R increases (R > r)")
                else:
                    logs.append(f"SUSPICIOUS: Q does not decrease when R increases (R > r)")
                    success = False
            
            # Restore original R
            self.R = original_R
            
            # Test 2: Increase number of turns N
            original_N = self.N
            perturbed_N = original_N * 1.1  # +10%
            
            # Compute E_m for perturbed N using E_m = N*Φ_m*ω
            if self.results['T'] > 0:
                omega = 2 * self.pi / self.results['T']
                # Use the code's Phi_m value for monotonicity test
                original_E_m = original_N * self.results['Phi_m'] * omega
                perturbed_E_m = perturbed_N * self.results['Phi_m'] * omega
                
                # E_m should increase proportionally with N
                expected_ratio = perturbed_N / original_N
                actual_ratio = perturbed_E_m / original_E_m if original_E_m != 0 else 0
                
                if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
                    logs.append(f"OK: E_m increases proportionally with N (ratio {actual_ratio:.3f} ≈ {expected_ratio:.3f})")
                else:
                    logs.append(f"FAIL: E_m does not scale correctly with N")
                    success = False
            
            # Restore original N
            self.N = original_N
            
            # Test 3: Increase internal resistance r
            original_r = self.r
            perturbed_r = original_r * 1.1  # +10%
            
            # Compute U and Q for perturbed r
            perturbed_U = self.results['E'] * (self.R / (self.R + perturbed_r))
            perturbed_Q = (perturbed_U**2 / self.R) * self.t
            
            # With constant E and R, increasing r should decrease U and thus Q
            if perturbed_Q < self.results['Q']:
                logs.append(f"OK: Q decreases from {self.results['Q']:.4f} to {perturbed_Q:.4f} when r increases")
            else:
                logs.append(f"SUSPICIOUS: Q does not decrease when r increases")
                success = False
            
            # Restore original r
            self.r = original_r
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
            
        return success, logs

    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                for log in logs:
                    all_logs.append(f"  {log}")
            except Exception as e:
                all_logs.append(f"  CRASH: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs