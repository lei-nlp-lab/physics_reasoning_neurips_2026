import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', 1.0)
        self.k1 = getattr(solution_instance, 'k1', 1.0)
        self.k2 = getattr(solution_instance, 'k2', 1.0)
        self.A = getattr(solution_instance, 'A', 1.0)
        
        # Extract results from solution_instance
        results = getattr(solution_instance, 'results', {})
        
        # Extract all required variables as floats
        self.P = float(results.get('P', 0.0))
        self.t1 = float(results.get('t_1', 0.0))
        self.T = float(results.get('T', 0.0))
        self.Deltat = float(results.get('Deltat', 0.0))
        self.F_return = float(results.get('F_return', 0.0))
        self.t2 = float(results.get('t_2', 0.0))
        self.k = float(results.get('k', 0.0))
        self.phi = float(results.get('phi', 0.0))
        
        # Initialize logs
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append({"test": msg, "success": success})
    
    def _fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical invariants from KB"""
        success = True
        
        try:
            # 1. Check equivalent spring constant (parallel springs)
            k_eq_expected = self.k1 + self.k2
            success_k = self._fuzzy_equal(self.k, k_eq_expected)
            self._log(f"Equivalent spring constant: k = {self.k:.4f}, expected k1+k2 = {k_eq_expected:.4f}", success_k)
            success = success and success_k
            
            # 2. Check restoring force at x = 0.5A
            F_expected = -(self.k1 + self.k2) * (0.5 * self.A)
            success_F = self._fuzzy_equal(self.F_return, F_expected)
            self._log(f"Restoring force at x=0.5A: F = {self.F_return:.4f}, expected -k*(0.5A) = {F_expected:.4f}", success_F)
            success = success and success_F
            
            # 3. Check period formula
            T_expected = 2 * math.pi * math.sqrt(self.m / (self.k1 + self.k2))
            success_T = self._fuzzy_equal(self.T, T_expected)
            self._log(f"Period: T = {self.T:.4f}, expected 2π√(m/(k1+k2)) = {T_expected:.4f}", success_T)
            success = success and success_T
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final probability"""
        success = True
        
        try:
            # Step 4: Period calculation (already verified in invariants)
            T_expected = 2 * math.pi * math.sqrt(self.m / (self.k1 + self.k2))
            success_T = self._fuzzy_equal(self.T, T_expected)
            self._log(f"Period T consistent with formula: {self.T:.4f} ≈ {T_expected:.4f}", success_T)
            success = success and success_T
            
            # Step 5: Phase at 0.5A - check sin(φ) = 0.5 (since using sine function)
            sin_val = math.sin(self.phi)
            success_phi = self._fuzzy_equal(sin_val, 0.5)
            self._log(f"Phase φ gives sin(φ) ≈ 0.5 for x=0.5A: sin({self.phi:.4f}) = {sin_val:.4f}", success_phi)
            success = success and success_phi
            
            # Step 6: Times t1 and t2
            # Angular frequency ω = 2π/T
            omega = 2 * math.pi / self.T
            # Expected phases: φ1 = π/6, φ2 = 5π/6
            phi1_expected = math.pi/6
            phi2_expected = 5*math.pi/6
            t1_expected = phi1_expected / omega
            t2_expected = phi2_expected / omega
            
            success_t1 = self._fuzzy_equal(self.t1, t1_expected)
            self._log(f"Time t1 = {self.t1:.4f}, expected π/(6ω) = {t1_expected:.4f}", success_t1)
            success = success and success_t1
            
            success_t2 = self._fuzzy_equal(self.t2, t2_expected)
            self._log(f"Time t2 = {self.t2:.4f}, expected 5π/(6ω) = {t2_expected:.4f}", success_t2)
            success = success and success_t2
            
            # Step 7: Time interval Δt
            # Δt = (π/3)/ω = T/6
            Deltat_expected = self.T / 6
            success_Deltat = self._fuzzy_equal(self.Deltat, Deltat_expected)
            self._log(f"Δt = {self.Deltat:.4f}, expected T/6 = {Deltat_expected:.4f}", success_Deltat)
            success = success and success_Deltat
            
            # Step 8: Probability P
            P_expected = 1/6
            success_P = self._fuzzy_equal(self.P, P_expected)
            self._log(f"Probability P = {self.P:.4f}, expected 1/6 = {P_expected:.4f}", success_P)
            success = success and success_P
            
            # Also check P = Δt/T
            P_ratio = self.Deltat / self.T
            success_ratio = self._fuzzy_equal(self.P, P_ratio)
            self._log(f"P = Δt/T consistency: {self.P:.4f} = {self.Deltat:.4f}/{self.T:.4f} = {P_ratio:.4f}", success_ratio)
            success = success and success_ratio
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb input parameters and check trends"""
        success = True
        
        try:
            # Test 1: Period vs mass (T ∝ √m)
            m_orig = self.m
            T_orig = 2 * math.pi * math.sqrt(m_orig / (self.k1 + self.k2))
            
            m_perturbed = m_orig * 1.5
            T_perturbed = 2 * math.pi * math.sqrt(m_perturbed / (self.k1 + self.k2))
            
            expected_ratio = math.sqrt(m_perturbed / m_orig)
            actual_ratio = T_perturbed / T_orig
            success_m = self._fuzzy_equal(actual_ratio, expected_ratio)
            self._log(f"T ∝ √m: T({m_perturbed:.2f})/T({m_orig:.2f}) = {actual_ratio:.4f} ≈ √({m_perturbed:.2f}/{m_orig:.2f}) = {expected_ratio:.4f}", success_m)
            success = success and success_m
            
            # Test 2: Period vs spring constant (T ∝ 1/√k)
            k1_orig = self.k1
            T_k1_orig = 2 * math.pi * math.sqrt(m_orig / (k1_orig + self.k2))
            
            k1_perturbed = k1_orig * 2.0
            T_k1_perturbed = 2 * math.pi * math.sqrt(m_orig / (k1_perturbed + self.k2))
            
            expected_ratio_k = math.sqrt((k1_orig + self.k2) / (k1_perturbed + self.k2))
            actual_ratio_k = T_k1_perturbed / T_k1_orig
            success_k = self._fuzzy_equal(actual_ratio_k, expected_ratio_k)
            self._log(f"T ∝ 1/√k: T(k1={k1_perturbed:.2f})/T(k1={k1_orig:.2f}) = {actual_ratio_k:.4f} ≈ √({k1_orig+self.k2:.2f}/{k1_perturbed+self.k2:.2f}) = {expected_ratio_k:.4f}", success_k)
            success = success and success_k
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Run all checks and return summary"""
        self.logs = []  # Reset logs
        
        try:
            success_invariants = self.test_invariants()
            success_trace = self.test_trace_consistency()
            success_monotonicity = self.test_monotonicity()
            
            overall_success = success_invariants and success_trace and success_monotonicity
            
            # Summary log
            summary = f"PhysicsChecker summary: Invariants={success_invariants}, Trace={success_trace}, Monotonicity={success_monotonicity}"
            self._log(summary, overall_success)
            
        except Exception as e:
            self._log(f"PhysicsChecker crashed: {traceback.format_exc()}", False)
            overall_success = False
        
        return overall_success, self.logs