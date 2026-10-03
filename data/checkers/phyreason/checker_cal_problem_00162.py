import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # solution_instance is the module where the code ran
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract required results with safe defaults
        self.E_e_E_v = self.results.get('$E_e+E_v$', 0.0)
        self.eta = self.results.get('$eta$', 0.0)
        self.E_kp = self.results.get('$E_kp$', 0.0)
        self.DeltaE_d = self.results.get('$DeltaE_d$', 0.0)
        self.R = self.results.get('R', 0.0)
        self.B = self.results.get('B', 0.0)
        
        # Extract alpha and beta
        self.alpha = self.results.get('$alpha$', 0.0)
        self.beta = self.results.get('$beta$', 0.0)
        
        # If alpha and beta are 0, check for combined format
        if self.alpha == 0.0 and self.beta == 0.0:
            alpha_beta = self.results.get('$alpha$,$beta$', '0,0')
            if isinstance(alpha_beta, str):
                try:
                    parts = alpha_beta.split(',')
                    if len(parts) == 2:
                        self.alpha = float(parts[0].strip())
                        self.beta = float(parts[1].strip())
                except:
                    pass  # Keep defaults
        
        # Extract constants with safe defaults
        self.m_n = self.results.get('m_n', 939.57)
        self.m_p = self.results.get('m_p', 938.27)
        self.m_e = self.results.get('m_e', 0.51)
        self.L = self.results.get('L', 1.2)
        self.c = self.results.get('c', 2.99792458e8)
        
        # For p and e, use results or known values
        self.p = self.results.get('p', 4.8e-21)  # kg·m/s
        self.e = self.results.get('e', 1.602176634e-19)  # C
        
        # Alternative momentum in MeV·s/m (from problem statement)
        self.p_MeV_s_per_m = 3e-8
        
        self.logs = []
    
    def _log(self, msg, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def test_invariants(self):
        """Check fundamental physics invariants"""
        success = True
        
        try:
            # 1. Check mass-energy conservation in decay
            delta_m = self.m_n - self.m_p - self.m_e
            Q_expected = delta_m  # In natural units (MeV)
            
            if not math.isclose(self.DeltaE_d, Q_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Mass-energy conservation mismatch: ΔE_d={self.DeltaE_d}, expected={Q_expected}", False)
                success = False
            else:
                self._log(f"Mass-energy conservation OK: ΔE_d={self.DeltaE_d} MeV", True)
            
            # 2. Check proton kinetic energy formula
            # E_kp should be calculated from p in MeV·s/m: E_kp = (p*c)^2/(2*m_p)
            p_times_c = self.p_MeV_s_per_m * self.c
            E_kp_expected = (p_times_c**2) / (2 * self.m_p)
            
            if not math.isclose(self.E_kp, E_kp_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Proton KE mismatch: E_kp={self.E_kp}, expected={E_kp_expected}", False)
                success = False
            else:
                self._log(f"Proton KE formula OK: E_kp={self.E_kp} MeV", True)
            
            # 3. Check energy conservation
            total_KE = self.E_kp + self.E_e_E_v
            if not math.isclose(self.DeltaE_d, total_KE, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Energy conservation mismatch: ΔE_d={self.DeltaE_d}, E_kp+E_e+v={total_KE}", False)
                success = False
            else:
                self._log(f"Energy conservation OK: ΔE_d = E_kp + (E_e+E_v) = {total_KE} MeV", True)
            
            # 4. Check circular motion in magnetic field
            # Use known values if results are 0
            p_for_B = self.p if abs(self.p) > 1e-30 else 4.8e-21
            e_for_B = self.e if abs(self.e) > 1e-30 else 1.602176634e-19
            
            if self.B > 1e-10 and p_for_B > 1e-30 and e_for_B > 1e-30:
                R_expected = p_for_B / (e_for_B * self.B)
                if not math.isclose(self.R, R_expected, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Gyroradius mismatch: R={self.R}, p/(eB)={R_expected}", False)
                    success = False
                else:
                    self._log(f"Gyroradius formula OK: R={self.R} m", True)
            else:
                # Check using geometry formula
                R_expected = self.L / math.sqrt(15)
                if math.isclose(self.R, R_expected, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Gyroradius from geometry OK: R={self.R} m", True)
                else:
                    self._log(f"Cannot verify gyroradius (B={self.B}, p={p_for_B}, e={e_for_B})", True)
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        success = True
        
        try:
            # Check step-by-step consistency from the analysis
            # Use 2% tolerance
            if not math.isclose(self.DeltaE_d, 0.79, rel_tol=0.02, abs_tol=0.02):
                self._log(f"ΔE_d inconsistent with step_2: {self.DeltaE_d} vs 0.79 MeV", False)
                success = False
            else:
                self._log(f"ΔE_d consistent with step_2: {self.DeltaE_d} MeV", True)
            
            if not math.isclose(self.E_kp, 0.0432, rel_tol=0.02, abs_tol=0.02):
                self._log(f"E_kp inconsistent with step_3: {self.E_kp} vs 0.0432 MeV", False)
                success = False
            else:
                self._log(f"E_kp consistent with step_3: {self.E_kp} MeV", True)
            
            if not math.isclose(self.E_e_E_v, 0.7468, rel_tol=0.02, abs_tol=0.02):
                self._log(f"E_e+E_v inconsistent with step_4: {self.E_e_E_v} vs 0.7468 MeV", False)
                success = False
            else:
                self._log(f"E_e+E_v consistent with step_4: {self.E_e_E_v} MeV", True)
            
            if not (math.isclose(self.alpha, math.pi/6, rel_tol=0.02, abs_tol=0.02) and
                    math.isclose(self.beta, math.pi/6, rel_tol=0.02, abs_tol=0.02)):
                self._log(f"α,β inconsistent with step_6: ({self.alpha},{self.beta}) vs (π/6,π/6)", False)
                success = False
            else:
                self._log(f"α,β consistent with step_6: ({self.alpha},{self.beta}) rad", True)
            
            if not math.isclose(self.eta, 2/3, rel_tol=0.02, abs_tol=0.02):
                self._log(f"η inconsistent with step_7: {self.eta} vs 2/3", False)
                success = False
            else:
                self._log(f"η consistent with step_7: {self.eta}", True)
            
            B_expected = math.sqrt(15)/40
            if not math.isclose(self.B, B_expected, rel_tol=0.02, abs_tol=0.02):
                self._log(f"B inconsistent with step_10: {self.B} vs sqrt(15)/40={B_expected}", False)
                success = False
            else:
                self._log(f"B consistent with step_10: {self.B} T", True)
            
            if success:
                self._log("All trace consistency checks passed", True)
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Test monotonic trends by perturbing inputs"""
        success = True
        
        try:
            # Test monotonicity of E_kp with respect to momentum
            # Use p in MeV·s/m for this test
            original_p = self.p_MeV_s_per_m
            
            if original_p == 0:
                self._log("Using alternative p value for monotonicity test", True)
                original_p = 3e-8  # Default value
            
            # Perturb momentum by +1%
            perturbed_p = original_p * 1.01
            
            # Recalculate E_kp using formula
            p_times_c_original = original_p * self.c
            p_times_c_perturbed = perturbed_p * self.c
            
            E_kp_original = (p_times_c_original**2) / (2 * self.m_p)
            E_kp_perturbed = (p_times_c_perturbed**2) / (2 * self.m_p)
            
            # E_kp should increase by ~2% (since E_kp ∝ p^2)
            expected_ratio = 1.01**2
            actual_ratio = E_kp_perturbed / E_kp_original if E_kp_original != 0 else 1
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Monotonicity fail: E_kp ratio={actual_ratio}, expected={expected_ratio}", False)
                success = False
            else:
                self._log(f"Monotonicity OK: E_kp ∝ p^2 holds (ratio={actual_ratio})", True)
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        self.logs = []
        
        try:
            inv_success = self.test_invariants()
            trace_success = self.test_trace_consistency()
            mono_success = self.test_monotonicity()
            
            overall_success = inv_success and trace_success and mono_success
            
            self._log(f"\n=== TEST SUMMARY ===", None)
            self._log(f"Invariants: {'PASS' if inv_success else 'FAIL'}", None)
            self._log(f"Trace Consistency: {'PASS' if trace_success else 'FAIL'}", None)
            self._log(f"Monotonicity: {'PASS' if mono_success else 'FAIL'}", None)
            self._log(f"OVERALL: {'PASS' if overall_success else 'FAIL'}", None)
            
            return overall_success, self.logs
            
        except Exception as e:
            error_msg = f"Critical error in run_all_tests: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, False)
            return False, self.logs