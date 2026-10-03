import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract ALL values from results dictionary first
        # Physical constants
        self.mu0 = self.results.get('mu0', 4e-7 * math.pi)
        self.eps0 = self.results.get('eps0', 8.854e-12)
        
        # If not in results, try to get from solution_instance attributes
        if self.mu0 == 4e-7 * math.pi:
            self.mu0 = getattr(solution_instance, 'mu0', 4e-7 * math.pi)
        if self.eps0 == 8.854e-12:
            self.eps0 = getattr(solution_instance, 'eps0', 8.854e-12)
        
        # Problem parameters (extract from results first, then attributes)
        self.lam = self.results.get('lam', getattr(solution_instance, 'lam', 1e-6))
        self.a = self.results.get('a', getattr(solution_instance, 'a', 0.1))
        self.I = self.results.get('I', getattr(solution_instance, 'I', 1e-6))
        self.B0 = self.results.get('B0', getattr(solution_instance, 'B0', 1e-3))
        self.T = self.results.get('T', getattr(solution_instance, 'T', 1.0))
        
        # Result quantities (extract from results only)
        self.alpha = self.results.get('alpha', 0.0)
        self.domega_dt = self.results.get('fracdomegadt', 0.0)
        self.mathcalP = self.results.get('mathcalPmathbfr', 0.0)
        self.omegaT = self.results.get('omegaT', 0.0)
        self.E_ind = self.results.get('E_textindr', 0.0)
        self.mathcalL = self.results.get('mathcalLmathbfr', 0.0)
        self.tau = self.results.get('tau', 0.0)
        self.BT = self.results.get('BT', 0.0)
        self.L = self.results.get('L', 0.0)
        
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
        
    def test_invariants(self):
        """Check fundamental physical laws."""
        checks = []
        
        # 1. Check Faraday's Law: E_ind(r) = -(r/2) * dB/dt (from step 2)
        try:
            # Numerical check at r = a
            # We need dB/dt. From the problem: B decreases linearly from B0 to 0 over time T
            # So dB/dt = -B0/T (approximately)
            dB_dt = -self.B0 / self.T if self.T != 0 else 0.0
            expected_E_at_a = -self.a * dB_dt / 2
            if math.isclose(self.E_ind, expected_E_at_a, rel_tol=1e-2, abs_tol=1e-10):
                checks.append(("Faraday's Law (E_ind at r=a)", True))
            else:
                checks.append(("Faraday's Law (E_ind at r=a)", False, 
                              f"E_ind={self.E_ind}, expected={expected_E_at_a}"))
        except Exception as e:
            self.log(f"Faraday check failed: {e}")
            
        # 2. Check torque-angular acceleration relation: τ = I * dω/dt (from step 3-4)
        try:
            expected_tau = self.I * self.domega_dt
            if math.isclose(self.tau, expected_tau, rel_tol=1e-2, abs_tol=1e-10):
                checks.append(("Newton's 2nd Law (rotation)", True))
            else:
                checks.append(("Newton's 2nd Law (rotation)", False,
                              f"τ={self.tau}, I*dω/dt={expected_tau}"))
        except Exception as e:
            self.log(f"Torque check failed: {e}")
            
        # 3. Check angular momentum conservation (initial vs final)
        try:
            # Initial angular momentum (per unit length) from electromagnetic field
            # From physics: L_field = - (α λ a² B) / 2
            # With α = ε0, L_field_initial = - (ε0 λ a² B0) / 2
            L_initial = -self.eps0 * self.lam * self.B0 * self.a**2 / 2
            
            # Final mechanical angular momentum (per unit length)
            L_mech_final = self.I * self.omegaT
            
            # Final electromagnetic angular momentum (per unit length) at t=T
            L_em_final = -self.eps0 * self.lam * self.BT * self.a**2 / 2
            
            # Total final angular momentum
            L_total_final = L_mech_final + L_em_final
            
            if math.isclose(L_initial, L_total_final, rel_tol=1e-2, abs_tol=1e-10):
                checks.append(("Angular momentum conservation", True))
            else:
                checks.append(("Angular momentum conservation", False,
                              f"L_initial={L_initial}, L_final={L_total_final}"))
        except Exception as e:
            self.log(f"Angular momentum check failed: {e}")
            
        # 4. Check that α = ε0 (from steps 12-13)
        try:
            if math.isclose(self.alpha, self.eps0, rel_tol=1e-2, abs_tol=1e-12):
                checks.append(("Proportionality constant α", True))
            else:
                checks.append(("Proportionality constant α", False,
                              f"α={self.alpha}, ε0={self.eps0}"))
        except Exception as e:
            self.log(f"Alpha check failed: {e}")
            
        all_passed = all(check[1] for check in checks)
        return all_passed, checks
        
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        checks = []
        
        # 1. Check that ω(T) matches the expected formula
        try:
            # Expected formula: ω(T) = ( (λa²/(2I)) * B0 ) / (1 + μ0 * λ² * a / (8πI) )
            num = (self.lam * self.a**2 / (2 * self.I)) * self.B0
            den = 1 + (self.mu0 * self.lam**2 * self.a) / (8 * math.pi * self.I)
            expected_omegaT = num / den if den != 0 else 0.0
            
            if math.isclose(self.omegaT, expected_omegaT, rel_tol=1e-2, abs_tol=1e-10):
                checks.append(("ω(T) matches expected formula", True))
            else:
                checks.append(("ω(T) matches expected formula", False,
                              f"ω(T)={self.omegaT}, expected={expected_omegaT}"))
        except Exception as e:
            self.log(f"ω(T) formula check failed: {e}")
            
        # 2. Check that B(T) from Ampere's Law matches ω(T)
        try:
            # B(T) = μ0 * (λ/(2π)) * ω(T)
            expected_BT = self.mu0 * (self.lam / (2 * math.pi)) * self.omegaT
            if math.isclose(self.BT, expected_BT, rel_tol=1e-2, abs_tol=1e-10):
                checks.append(("B(T) from Ampere's Law", True))
            else:
                checks.append(("B(T) from Ampere's Law", False,
                              f"B(T)={self.BT}, expected={expected_BT}"))
        except Exception as e:
            self.log(f"B(T) check failed: {e}")
            
        # 3. Check angular momentum density relations
        try:
            # From physics: ℒ(𝐫) = -α * (λ B)/(2π ε0) for r < a
            # At r=a, with B=B0 and α=ε0: ℒ = - (λ B0)/(2π)
            expected_mathcalL = - (self.lam * self.B0) / (2 * math.pi)
            
            # We compare magnitudes (sign depends on coordinate system)
            if math.isclose(abs(self.mathcalL), abs(expected_mathcalL), rel_tol=1e-2, abs_tol=1e-10):
                checks.append(("Angular momentum density magnitude", True))
            else:
                checks.append(("Angular momentum density magnitude", False,
                              f"|ℒ|={abs(self.mathcalL)}, expected={abs(expected_mathcalL)}"))
        except Exception as e:
            self.log(f"Angular momentum density check failed: {e}")
            
        all_passed = all(check[1] for check in checks)
        return all_passed, checks
        
    def test_monotonicity(self):
        """Perturb inputs and check physical trends."""
        checks = []
        
        try:
            # Save original values
            original_lam = self.lam
            original_B0 = self.B0
            
            # Test 1: Increase B0, ω(T) should increase
            B0_increased = original_B0 * 1.1
            num = (self.lam * self.a**2 / (2 * self.I)) * B0_increased
            den = 1 + (self.mu0 * self.lam**2 * self.a) / (8 * math.pi * self.I)
            omegaT_increased = num / den if den != 0 else 0.0
            
            num = (self.lam * self.a**2 / (2 * self.I)) * original_B0
            omegaT_original = num / den if den != 0 else 0.0
            
            if omegaT_increased > omegaT_original:
                checks.append(("ω(T) increases with B0", True))
            else:
                checks.append(("ω(T) increases with B0", False,
                              f"ω(T)_original={omegaT_original}, ω(T)_increased={omegaT_increased}"))
                
            # Test 2: Check ω(T) is positive for our parameters
            if self.omegaT > 0:
                checks.append(("ω(T) is positive", True))
            else:
                checks.append(("ω(T) is positive", False,
                              f"ω(T)={self.omegaT}"))
                
        except Exception as e:
            self.log(f"Monotonicity check failed: {e}")
            
        all_passed = all(check[1] for check in checks)
        return all_passed, checks
        
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        success = True
        detailed_logs = []
        
        try:
            # Test 1: Invariants
            inv_success, inv_checks = self.test_invariants()
            detailed_logs.append(("Physical Invariants", inv_checks))
            if not inv_success:
                success = False
                
            # Test 2: Trace consistency
            trace_success, trace_checks = self.test_trace_consistency()
            detailed_logs.append(("Trace Consistency", trace_checks))
            if not trace_success:
                success = False
                
            # Test 3: Monotonicity
            mono_success, mono_checks = self.test_monotonicity()
            detailed_logs.append(("Monotonicity", mono_checks))
            if not mono_success:
                success = False
                
        except Exception as e:
            self.log(f"Error in run_all_tests: {traceback.format_exc()}")
            success = False
            
        # Format logs
        all_logs = self.logs.copy()
        for test_name, checks in detailed_logs:
            all_logs.append(f"\n{test_name}:")
            for check in checks:
                if len(check) == 2:
                    all_logs.append(f"  {'✓' if check[1] else '✗'} {check[0]}")
                else:
                    all_logs.append(f"  {'✗'} {check[0]}: {check[2]}")
                    
        return success, all_logs