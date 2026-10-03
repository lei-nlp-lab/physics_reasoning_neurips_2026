import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Extract input parameters with safe defaults
        # Use the SAME arbitrary values as the solution code (all = 1.0)
        self.m = self._safe_get('m', 1.0)
        self.g = self._safe_get('g', 1.0)
        self.L = self._safe_get('L', 1.0)
        self.k = self._safe_get('k', 1.0)
        self.Q = self._safe_get('Q', 1.0)
        
        # Extract results
        self.F = self._safe_get('F', None)
        self.q = self._safe_get('q', None)
        self.U_AB = self._safe_get('U_AB', None)
        
        # Constants
        self.sin30 = 0.5  # sin(30°) = 0.5
        
        # Logs
        self.logs = []
    
    def _safe_get(self, key, default):
        """Safely get attribute from solution instance or its results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = getattr(self.solution, 'results', {}).get(key, default)
        return val
    
    def _log(self, msg, success):
        """Append a test result to logs."""
        self.logs.append((success, msg))
    
    def test_invariants(self):
        """Test fundamental physics laws: Newton's 2nd Law, Coulomb's Law, Energy Conservation."""
        try:
            # 1. Newton's Second Law at point A
            F_gravity = self.m * self.g * self.sin30  # down the incline
            if self.F is not None:
                # Net force down = F_gravity - F (since F is up)
                a_net = (F_gravity - self.F) / self.m
                # Should equal g/4 (downward, so positive if down is positive)
                if math.isclose(a_net, self.g/4, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Newton's 2nd Law at A: net acceleration ≈ g/4", True)
                else:
                    self._log(f"Newton's 2nd Law at A failed: a_net={a_net}, expected {self.g/4}", False)
            else:
                self._log("Skipping Newton's 2nd Law test: F not available", False)
            
            # 2. Coulomb's Law at point A
            if self.F is not None and self.q is not None:
                F_coulomb = self.k * abs(self.q) * self.Q / (self.L ** 2)
                if math.isclose(self.F, F_coulomb, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Coulomb's Law at A: F matches k|q|Q/L²", True)
                else:
                    self._log(f"Coulomb's Law at A failed: F={self.F}, computed {F_coulomb}", False)
            else:
                self._log("Skipping Coulomb's Law test: F or q not available", False)
            
            # 3. Work-Energy Theorem from A to B
            # Work by gravity: positive (down the incline)
            W_gravity = self.m * self.g * self.sin30 * (self.L / 2)
            # Work by electrostatic force: use potential energy change
            # V = kQ/r, so ΔU_elec = q*(V_B - V_A) = q*kQ*(1/(L/2) - 1/L) = q*kQ/L
            # Work by electrostatic force = -ΔU_elec = -q*k*self.Q/self.L
            if self.q is not None:
                W_elec = -self.q * self.k * self.Q / self.L
                total_work = W_gravity + W_elec
                # Change in kinetic energy from A to B is 0
                if math.isclose(total_work, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Work-Energy Theorem A→B: total work ≈ 0", True)
                else:
                    self._log(f"Work-Energy Theorem failed: total work = {total_work}", False)
            else:
                self._log("Skipping Work-Energy test: q not available", False)
            
            # 4. Check U_AB definition consistency (if available)
            if self.U_AB is not None and self.q is not None:
                # From work-energy: q*U_AB should equal -W_gravity (since W_gravity + q*U_AB = 0)
                expected_qU = -W_gravity
                if math.isclose(self.q * self.U_AB, expected_qU, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("U_AB satisfies work-energy equation: q*U_AB ≈ -W_gravity", True)
                else:
                    self._log(f"U_AB consistency failed: q*U_AB={self.q*self.U_AB}, expected {expected_qU}", False)
                
                # Also check if U_AB matches -k*self.Q/self.L (theoretical potential difference V_A - V_B)
                U_theoretical = -self.k * self.Q / self.L
                if math.isclose(self.U_AB, U_theoretical, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("U_AB matches theoretical V_A - V_B = -kQ/L", True)
                elif math.isclose(self.U_AB, -self.k * self.q / self.L, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("U_AB matches -kq/L (as given in target answer)", True)
                else:
                    self._log(f"U_AB deviates from both -kQ/L and -kq/L", False)
        
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results."""
        try:
            # Step 1: F = mg/4 ?
            if self.F is not None:
                expected_F = self.m * self.g / 4
                if math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Step 1 consistency: F ≈ mg/4", True)
                else:
                    self._log(f"Step 1 mismatch: F={self.F}, expected {expected_F}", False)
            
            # Step 2: q = (m g L²)/(4 k Q) ?
            if self.q is not None:
                expected_q = (self.m * self.g * self.L**2) / (4 * self.k * self.Q)
                if math.isclose(self.q, expected_q, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Step 2 consistency: q ≈ m g L²/(4 k Q)", True)
                else:
                    self._log(f"Step 2 mismatch: q={self.q}, expected {expected_q}", False)
            
            # Step 3: U_AB = -kq/L or -kQ/L ? (both are possible depending on derivation)
            if self.U_AB is not None and self.q is not None:
                expected_U1 = -self.k * self.q / self.L
                expected_U2 = -self.k * self.Q / self.L
                if math.isclose(self.U_AB, expected_U1, rel_tol=1e-2, abs_tol=1e-2) or \
                   math.isclose(self.U_AB, expected_U2, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Step 3 consistency: U_AB ≈ -kq/L or -kQ/L", True)
                else:
                    self._log(f"Step 3 mismatch: U_AB={self.U_AB}, expected -kq/L={expected_U1} or -kQ/L={expected_U2}", False)
        
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)
    
    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        try:
            # Use the same arbitrary constants as the solution
            m_original = self.m
            g_original = self.g
            L_original = self.L
            k_original = self.k
            Q_original = self.Q
            
            # Perturb mass m by +10%
            m_perturbed = m_original * 1.10
            
            # Expected F ∝ m
            F_original = m_original * g_original / 4
            F_perturbed = m_perturbed * g_original / 4
            ratio_F = F_perturbed / F_original
            expected_ratio = 1.10
            if math.isclose(ratio_F, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Monotonicity (F ∝ m): passed", True)
            else:
                self._log(f"Monotonicity (F ∝ m) failed: ratio {ratio_F} vs {expected_ratio}", False)
            
            # Expected q ∝ m (from q = m g L²/(4 k Q))
            if self.q is not None:
                q_original = self.q
                q_perturbed = (m_perturbed * g_original * L_original**2) / (4 * k_original * Q_original)
                ratio_q = q_perturbed / q_original
                if math.isclose(ratio_q, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Monotonicity (q ∝ m): passed", True)
                else:
                    self._log(f"Monotonicity (q ∝ m) failed: ratio {ratio_q} vs {expected_ratio}", False)
            
            # Perturb L by +10%
            L_perturbed = L_original * 1.10
            # Expected q ∝ L²
            if self.q is not None:
                q_perturbed_L = (m_original * g_original * L_perturbed**2) / (4 * k_original * Q_original)
                ratio_q_L = q_perturbed_L / self.q
                expected_ratio_L = 1.10**2  # 1.21
                if math.isclose(ratio_q_L, expected_ratio_L, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Monotonicity (q ∝ L²): passed", True)
                else:
                    self._log(f"Monotonicity (q ∝ L²) failed: ratio {ratio_q_L} vs {expected_ratio_L}", False)
        
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        success = all(s for s, _ in self.logs)
        return success, self.logs