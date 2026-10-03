import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results without direct attribute access."""
        # Only use results, not inputs
        self.results = getattr(solution_instance, 'results', {})
        
        # Safely extract all expected results using KB mapping
        self.S_I_over_I_B = self.results.get('S_I/I_B', 0.0)
        self.frac_N_Nphi = self.results.get('fracNN_phi', 0.0)
        self.alpha = self.results.get('alpha', 0.0)
        self.KE = self.results.get('KE', 0.0)
        self.v_s = self.results.get('v_s', 0.0)
        self.S_I = self.results.get('S_I', 0.0)
        self.R_H = self.results.get('R_H', 0.0)
        self.l_0 = self.results.get('l_0', 0.0)
        self.DeltaB = self.results.get('DeltaB', 0.0)
        self.V_H = self.results.get('V_H', 0.0)
        self.E_th = self.results.get('E_th', 0.0)
        self.I = self.results.get('I', 0.0)
        self.I_B = self.results.get('I_B', 0.0)
        self.vecv_s = self.results.get('vecv_s', 0.0)
        self.DeltaU = self.results.get('DeltaU', 0.0)
        self.Deltaphi = self.results.get('Deltaphi', 0.0)
        
        # Extract key parameters from results with reasonable defaults
        # These should come from results if available, otherwise use typical values
        self.B = self.results.get('B', 15.0)
        self.W = self.results.get('W', 1e-6)
        self.L = self.results.get('L', 1e-6)
        
        # For E, we need to compute it from V_H and W if available
        if self.V_H != 0 and self.W != 0:
            self.E = self.V_H / self.W
        else:
            self.E = 1.0  # default
            
        # Use standard constants directly for comparison
        self.e = 1.602176634e-19
        self.h = 6.62607015e-34
        self.k_B = 1.380649e-23
        self.m_star = 9.10938356e-31 * 0.067  # GaAs effective mass
        
        self.logs = []
        
    def _log(self, message, success=None):
        """Helper to log test results."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _is_close(self, a, b, name=""):
        """Robust fuzzy comparison with physics-aware tolerances."""
        if a == 0 and b == 0:
            return True
        # Use larger tolerance for very small numbers
        if abs(a) < 1e-20 or abs(b) < 1e-20:
            abs_tol = 1e-23
            rel_tol = 1e-1  # 10% for quantum energies
        else:
            abs_tol = 1e-2
            rel_tol = 1e-2  # 1%
        
        close = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        if not close:
            self._log(f"{name}: {a} != {b} (diff={abs(a-b)})", False)
        return close
    
    def test_invariants(self):
        """Verify fundamental physical laws from KB."""
        success = True
        
        # 1. Lorentz force stationary velocity: v_s = E/B
        if self.B != 0 and self.E != 0:
            expected_v_s = self.E / self.B
            if not self._is_close(self.v_s, expected_v_s, "v_s = E/B"):
                success = False
        else:
            self._log("Skipping v_s = E/B test due to zero values", None)
        
        # 2. Hall resistance definition: R_H = V_H / I
        if self.I != 0:
            expected_R_H = self.V_H / self.I
            if not self._is_close(self.R_H, expected_R_H, "R_H = V_H/I"):
                success = False
        else:
            self._log("Skipping R_H = V_H/I test due to I=0", None)
        
        # 3. Hall resistance from magnetic flux: R_H = (1/e) * (phi/N)
        # This test requires n (electron density) which is not given.
        # We'll skip this test since n is arbitrary.
        
        # 4. Ratio N/N_phi = 1/3 from Hall plateau
        if not self._is_close(self.frac_N_Nphi, 1/3, "N/N_phi = 1/3"):
            success = False
        
        # 5. Energy scaling exponent alpha = 1/2
        if not self._is_close(self.alpha, 0.5, "alpha = 1/2"):
            success = False
        
        # 6. Delta B for one quasihole: ΔB = h/(e W L)
        if self.e != 0 and self.W != 0 and self.L != 0:
            expected_DeltaB = self.h / (self.e * self.W * self.L)
            if not self._is_close(self.DeltaB, expected_DeltaB, "ΔB = h/(e W L)"):
                success = False
        else:
            self._log("Skipping ΔB test due to zero values", None)
        
        # 7. Thermal energy: E_th = k_B * T (T=1K from step_14)
        expected_E_th = self.k_B * 1.0  # T = 1K
        if not self._is_close(self.E_th, expected_E_th, "E_th = k_B*T"):
            success = False
        
        # 8. Kinetic energy from uncertainty: KE = (h/2)*(eB/m*)
        if self.m_star != 0 and self.B != 0:
            expected_KE = (self.h/2) * (self.e * self.B / self.m_star)
            if not self._is_close(self.KE, expected_KE, "KE = (h/2)*(eB/m*)"):
                success = False
        else:
            self._log("Skipping KE test due to zero values", None)
        
        # 9. Vortex size: l_0 = sqrt(h/(eB))
        if self.e != 0 and self.B != 0:
            expected_l_0 = math.sqrt(self.h / (self.e * self.B))
            if not self._is_close(self.l_0, expected_l_0, "l_0 = sqrt(h/(eB))"):
                success = False
        else:
            self._log("Skipping l_0 test due to zero values", None)
        
        # 10. Noise-to-current ratio: S_I/I_B = νe (with ν=1/3 from Hall plateau)
        expected_ratio = (1/3) * self.e  # ν = 1/3 from step_8
        if not self._is_close(self.S_I_over_I_B, expected_ratio, "S_I/I_B = νe"):
            success = False
        
        self._log(f"Invariant tests completed: {success}")
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps from analysis."""
        success = True
        
        # Check step chain: v_s → V_H → I → R_H → N/N_phi
        # Step 3: v_s = E/B
        if self.B != 0:
            step3_v_s = self.E / self.B
            if not self._is_close(self.v_s, step3_v_s, "Step 3: v_s"):
                success = False
        
        # Step 4: V_H = E * W (assuming E_y = E)
        step4_V_H = self.E * self.W
        if not self._is_close(self.V_H, step4_V_H, "Step 4: V_H"):
            success = False
        
        # Step 5: I = (N e / L) * (E/B) = e * (N/phi) * V_H
        # This requires n (electron density) which is not given, so skip
        
        # Step 6: R_H = V_H / I = (1/e) * (phi/N)
        # This also requires n, so skip
        
        # Step 7: R_H = (h/e^2) * (N_phi/N)
        # This requires n, so skip
        
        # Step 11 & 12: Δφ = ΔB * (W L) = h/e
        if self.W != 0 and self.L != 0:
            step11_Deltaphi = self.DeltaB * (self.W * self.L)
        else:
            step11_Deltaphi = 0.0
            
        step12_Deltaphi = self.h / self.e if self.e != 0 else 0.0
        
        if not (self._is_close(self.Deltaphi, step11_Deltaphi, "Step 11: Δφ") and
                self._is_close(self.Deltaphi, step12_Deltaphi, "Step 12: Δφ")):
            success = False
        
        self._log(f"Trace consistency tests completed: {success}")
        return success
    
    def test_monotonicity(self):
        """Perturb magnetic field B and check monotonic trends."""
        success = True
        original_B = self.B
        
        try:
            # Only run if we have valid values
            if original_B == 0 or self.E == 0 or self.e == 0 or self.m_star == 0:
                self._log("Skipping monotonicity test due to zero values", None)
                return True
                
            # Perturb B by +1%
            self.B = original_B * 1.01
            
            # Recompute quantities that should scale monotonically with B
            new_v_s = self.E / self.B
            new_l_0 = math.sqrt(self.h / (self.e * self.B))
            new_KE = (self.h/2) * (self.e * self.B / self.m_star)
            
            # Check trends - only if original values are non-zero
            if self.v_s != 0 and not (new_v_s < self.v_s):  # v_s ∝ 1/B
                self._log("Monotonicity: v_s should decrease with increased B", False)
                success = False
            
            if self.l_0 != 0 and not (new_l_0 < self.l_0):  # l_0 ∝ 1/sqrt(B)
                self._log("Monotonicity: l_0 should decrease with increased B", False)
                success = False
            
            if self.KE != 0 and not (new_KE > self.KE):  # KE ∝ B
                self._log("Monotonicity: KE should increase with increased B", False)
                success = False
            
        except Exception as e:
            self._log(f"Monotonicity test error: {str(e)}", False)
            success = False
        finally:
            # Restore original B
            self.B = original_B
        
        self._log(f"Monotonicity tests completed: {success}")
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Starting {test_name} test...")
                test_result = test_func()
                if not test_result:
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} test crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs