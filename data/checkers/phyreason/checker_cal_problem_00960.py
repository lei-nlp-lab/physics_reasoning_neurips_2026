import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.L = 0.5
        self.R = 3.0
        self.B = 2.0
        self.v = 6.0
        
        # Extract results safely
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        self.I = getattr(solution_instance, 'I', solution_instance.results.get('I', 0.0))
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', 0.0))
        self.W_F = getattr(solution_instance, 'W_F', solution_instance.results.get('W_{F}', 0.0))
        self.U = getattr(solution_instance, 'U', solution_instance.results.get('U', 0.0))
        self.I_prime = getattr(solution_instance, 'I_prime', solution_instance.results.get("I'", 0.0))
        
        # Internal resistance computed from step analysis
        self.r = 1.0  # From step_5: I' = E/(R+r) => 1.5 = 6/(3+r) => r=1Ω
        
        self.logs = []
        
    def _log(self, msg, success=None):
        """Helper for logging"""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")
    
    def test_invariants(self):
        """Test fundamental physical laws"""
        try:
            # 1. Faraday's Law: E = B*L*v
            expected_E = self.B * self.L * self.v
            faraday_ok = math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Faraday's Law: E = B*L*v = {expected_E:.3f}, got {self.E:.3f}", faraday_ok)
            
            # 2. Ohm's Law (ideal): I = E/R
            expected_I = self.E / self.R
            ohm_ok = math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Ohm's Law (ideal): I = E/R = {expected_I:.3f}, got {self.I:.3f}", ohm_ok)
            
            # 3. Ampere's Force: F = B*I*L
            expected_F = self.B * self.I * self.L
            ampere_ok = math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Ampere's Force: F = B*I*L = {expected_F:.3f}, got {self.F:.3f}", ampere_ok)
            
            # 4. Power balance (work-energy): F*v = I²*R (mechanical power = electrical power)
            mech_power = self.F * self.v
            elec_power = self.I**2 * self.R
            power_ok = math.isclose(mech_power, elec_power, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Power balance: F*v = I²*R = {elec_power:.3f}, got {mech_power:.3f}", power_ok)
            
            # 5. Work done in Δt=0.1s: W_F = F*v*Δt
            Δt = 0.1  # Inferred from step_4 (1.2J = 2N * 6m/s * 0.1s)
            expected_W_F = self.F * self.v * Δt
            work_ok = math.isclose(self.W_F, expected_W_F, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"Work in Δt=0.1s: W_F = F*v*0.1 = {expected_W_F:.3f}, got {self.W_F:.3f}", work_ok)
            
            # 6. Circuit with internal resistance: I' = E/(R+r)
            if self.I_prime != 0:  # Only check if I' is provided
                expected_I_prime = self.E / (self.R + self.r)
                i_prime_ok = math.isclose(self.I_prime, expected_I_prime, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"I' = E/(R+r) = {expected_I_prime:.3f}, got {self.I_prime:.3f}", i_prime_ok)
                
                # Voltage across the external resistor R (as per target answer 4.5V)
                expected_U = self.I_prime * self.R
                u_ok = math.isclose(self.U, expected_U, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"U = I'*R = {expected_U:.3f}, got {self.U:.3f}", u_ok)
            else:
                i_prime_ok = u_ok = True  # Skip if not applicable
                self._log("Skipping internal resistance tests (I' not provided)")
            
            return all([faraday_ok, ohm_ok, ampere_ok, power_ok, work_ok, i_prime_ok, u_ok])
            
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Check E against target answer 6V
            e_ok = math.isclose(self.E, 6.0, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"E matches target 6V: {self.E:.3f}", e_ok)
            
            # Check I against target answer 2A
            i_ok = math.isclose(self.I, 2.0, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"I matches target 2A: {self.I:.3f}", i_ok)
            
            # Check W_F against target answer 1.2J
            w_ok = math.isclose(self.W_F, 1.2, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"W_F matches target 1.2J: {self.W_F:.3f}", w_ok)
            
            # Check U against target answer 4.5V
            u_ok = math.isclose(self.U, 4.5, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"U matches target 4.5V: {self.U:.3f}", u_ok)
            
            # Consistency: F should be 2.0N (from step_3)
            f_ok = math.isclose(self.F, 2.0, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"F matches step_3 value 2.0N: {self.F:.3f}", f_ok)
            
            # Consistency: I' should be 1.5A (from step_5)
            if self.I_prime != 0:
                i_prime_ok = math.isclose(self.I_prime, 1.5, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"I' matches step_5 value 1.5A: {self.I_prime:.3f}", i_prime_ok)
            else:
                i_prime_ok = True
                self._log("Skipping I' consistency check (not provided)")
            
            return all([e_ok, i_ok, w_ok, u_ok, f_ok, i_prime_ok])
            
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb velocity and check response trends"""
        original_v = self.v
        
        try:
            # Perturb v by +10%
            self.v *= 1.1
            
            # Recalculate expected values
            new_E = self.B * self.L * self.v
            new_I = new_E / self.R
            new_F = self.B * new_I * self.L
            new_W_F = new_F * self.v * 0.1
            new_I_prime = new_E / (self.R + self.r)
            new_U = new_I_prime * self.R  # U across external resistor R
            
            # Check monotonic relationships
            # 1. E should increase with v
            e_trend = new_E > self.E
            self._log(f"E increases with v: {self.E:.3f} → {new_E:.3f}", e_trend)
            
            # 2. I should increase with E (and thus with v)
            i_trend = new_I > self.I
            self._log(f"I increases with v: {self.I:.3f} → {new_I:.3f}", i_trend)
            
            # 3. F should increase with I (and thus with v)
            f_trend = new_F > self.F
            self._log(f"F increases with v: {self.F:.3f} → {new_F:.3f}", f_trend)
            
            # 4. W_F should increase with F and v (quadratic in v)
            w_trend = new_W_F > self.W_F
            self._log(f"W_F increases with v: {self.W_F:.3f} → {new_W_F:.3f}", w_trend)
            
            # 5. I' and U should increase with E (and thus with v)
            if self.I_prime != 0:
                i_prime_trend = new_I_prime > self.I_prime
                self._log(f"I' increases with v: {self.I_prime:.3f} → {new_I_prime:.3f}", i_prime_trend)
                
                u_trend = new_U > self.U
                self._log(f"U increases with v: {self.U:.3f} → {new_U:.3f}", u_trend)
            else:
                i_prime_trend = u_trend = True
                self._log("Skipping I' and U trend checks (not provided)")
            
            # Restore original v
            self.v = original_v
            
            return all([e_trend, i_trend, f_trend, w_trend, i_prime_trend, u_trend])
            
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}", False)
            self.v = original_v  # Ensure restoration even on error
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)"""
        self.logs = []
        success = True
        
        try:
            self._log("=== Starting Physics Audit ===")
            
            # Run tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = all([inv_ok, trace_ok, mono_ok])
            
            self._log(f"=== Audit Complete: {'PASS' if success else 'FAIL'} ===")
            
        except Exception as e:
            self._log(f"Fatal error in run_all_tests: {str(e)}\n{traceback.format_exc()}", False)
            success = False
        
        return success, self.logs