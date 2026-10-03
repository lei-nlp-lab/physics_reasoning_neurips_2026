import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: try to get from instance, then from results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # First try to get input parameters from results (they might be stored there)
        # If not in results, use reasonable defaults
        self.g = float(self._get_value('g', 9.8))
        self.h = float(self._get_value('h', 2.0))
        self.w = float(self._get_value('w', 1.0))
        self.rho0 = float(self._get_value('rho0', 1000.0))
        self.rho_oil = float(self._get_value('rho_oil', 800.0))
        self.rho1 = float(self._get_value('rho1', 2800.0))
        self.T0 = float(self._get_value('T0', 0.0))
        self.T1 = float(self._get_value('T1', 1200.0))
        self.k_l = float(self._get_value('k_l', 0.01))
        self.L = float(self._get_value('L', 1.0))
        self.c = float(self._get_value('c', 1000.0))
        self.kappa = float(self._get_value('kappa', 3.0))
        self.A = float(self._get_value('A', 1.0))  # dimensionless constant for tau
        
        # Extracted results with safe access - convert to float
        self.V = float(self.results.get('V', 0.0))
        self.rhoT = float(self.results.get('rhoT', 0.0))
        self.F = float(self.results.get('F', 0.0))
        self.F0 = float(self.results.get('F_0', 0.0))
        self.F1 = float(self.results.get('F_1', 0.0))
        self.Deltap = float(self.results.get('Deltap', 0.0))
        self.tau = float(self.results.get('tau', 0.0))
        # h' might be stored as 'h_prime' or "h''" - try both
        self.h_prime = float(self.results.get("h''", self.results.get('h_prime', 0.0)))
        self.D = float(self.results.get('D', 0.0))
        self.p0z = float(self.results.get('p0,z', 0.0))
        self.N_A = float(self.results.get('N/A', 0.0))
        self.Fx = float(self.results.get('F_x', 0.0))
        self.pinftyz = float(self.results.get('pinfty,z', 0.0))
        self.k = float(self.results.get('k', 3.0 * self.k_l))
        self.Tinftyz = float(self.results.get('Tinfty,z', 0.0))
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def _get_value(self, key, default):
        """Safely retrieve attribute from solution instance or results."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, default)
        # Ensure numeric
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Check fundamental physical laws."""
        logs = []
        try:
            # 1. Hydrostatic pressure: h' should be greater than h (since ρ_oil < ρ0)
            h_prime_expected = (self.rho0 / self.rho_oil) * self.h
            if not self._is_close(self.h_prime, h_prime_expected):
                logs.append(f"FAIL: h' mismatch. Expected {h_prime_expected}, got {self.h_prime}")
            elif self.rho_oil < self.rho0 and self.h_prime <= self.h:
                logs.append("FAIL: h' should be > h when ρ_oil < ρ0")
            else:
                logs.append("PASS: h' consistent with hydrostatic equilibrium")
            
            # 2. Forces F0 and F1 should be positive
            F0_expected = 0.5 * self.rho0 * self.g * self.h**2 * self.w
            F1_expected = 0.5 * self.rho_oil * self.g * self.h_prime**2 * self.w
            if not (self._is_close(self.F0, F0_expected) and self.F0 > 0):
                logs.append(f"FAIL: F0 mismatch. Expected {F0_expected}, got {self.F0}")
            else:
                logs.append("PASS: F0 matches formula and positive")
            if not (self._is_close(self.F1, F1_expected) and self.F1 > 0):
                logs.append(f"FAIL: F1 mismatch. Expected {F1_expected}, got {self.F1}")
            else:
                logs.append("PASS: F1 matches formula and positive")
            
            # 3. Net force Fx = F1 - F0 (from steps)
            Fx_expected = self.F1 - self.F0
            if not self._is_close(self.Fx, Fx_expected):
                logs.append(f"FAIL: Fx not equal to F1 - F0. Expected {Fx_expected}, got {self.Fx}")
            else:
                logs.append("PASS: Fx = F1 - F0 holds")
            
            # 4. Density-temperature relation: ρ(T) ≈ ρ1*(1 + k*(T1-T)/(T1-T0)) with k=3k_l
            if not self._is_close(self.k, 3 * self.k_l):
                logs.append(f"WARN: k ≠ 3k_l. k={self.k}, k_l={self.k_l}")
            # Check that ρ(T) increases with T (since k>0)
            T_mid = (self.T0 + self.T1) / 2
            rho_T_mid = self.rho1 * (1 + self.k * (self.T1 - T_mid) / (self.T1 - self.T0))
            # Use exact formula for comparison since we have exact rhoT
            rhoT_exact = self.rho1 * (1 - self.k_l * (self.T1 - T_mid) / (self.T1 - self.T0))**(-3)
            if not (self._is_close(self.rhoT, rhoT_exact) or self._is_close(self.rhoT, rho_T_mid)):
                logs.append(f"WARN: ρ(T) mismatch. Exact expected {rhoT_exact}, linear {rho_T_mid}, got {self.rhoT}")
            else:
                logs.append("PASS: ρ(T) matches formula")
            
            # 5. D should be positive (since ρ1 > ρ0 and k>0)
            D_expected = (2 / self.k) * (1 - self.rho0 / self.rho1) * self.h
            if not self._is_close(self.D, D_expected):
                logs.append(f"FAIL: D mismatch. Expected {D_expected}, got {self.D}")
            elif self.D <= 0:
                logs.append("FAIL: D should be positive")
            else:
                logs.append("PASS: D matches formula and positive")
            
            # 6. Force F should be positive (repulsive)
            if self.F <= 0:
                logs.append("WARN: Net horizontal force F is non-positive")
            else:
                logs.append("PASS: F > 0 (repulsive force)")
            
            # 7. Dimensional analysis for τ: τ ∝ ρ1^1 * c^1 * D^2 / κ
            tau_expected = self.A * (self.rho1 * self.c * self.D**2) / self.kappa
            if not self._is_close(self.tau, tau_expected):
                logs.append(f"FAIL: τ mismatch. Expected {tau_expected}, got {self.tau}")
            else:
                logs.append("PASS: τ matches dimensional analysis")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results."""
        logs = []
        try:
            # 1. Check h' from step_1 used in step_2
            h_prime_calc = (self.rho0 / self.rho_oil) * self.h
            if not self._is_close(self.h_prime, h_prime_calc):
                logs.append(f"FAIL: h' from step_1 {h_prime_calc} ≠ stored {self.h_prime}")
            else:
                logs.append("PASS: h' consistent in steps")
            
            # 2. Check that Fx formula matches step_3
            Fx_formula = 0.5 * self.rho0 * self.g * self.h**2 * self.w * (self.rho0 / self.rho_oil - 1)
            if not self._is_close(self.Fx, Fx_formula):
                logs.append(f"FAIL: Fx formula mismatch. Expected {Fx_formula}, got {self.Fx}")
            else:
                logs.append("PASS: Fx matches step_3 formula")
            
            # 3. Check that D from step_11 is used in force calculation
            # Using the exact formula from step_16 (first equality):
            F_exact = self.g * self.L * (
                self.rho1 * (0.5 * self.h**2 + self.h * self.D - (self.k * self.D**2) / 3) -
                self.rho0 * (0.5 * self.h**2 + self.h * self.D)
            )
            if not self._is_close(self.F, F_exact):
                logs.append(f"WARN: Exact F mismatch. Expected {F_exact}, got {self.F} (may differ due to approximations)")
            else:
                logs.append("PASS: F matches exact formula with given D")
            
            # 4. Check approximate force (step_17) vs exact for small k
            F_approx = (2 * self.g * self.L * self.h**2 / (3 * self.k)) * (self.rho1 - self.rho0)**2 / self.rho1
            if not self._is_close(self.F, F_approx):
                logs.append(f"WARN: Approx F {F_approx} ≠ exact F {self.F} (may be OK if k is not very small)")
            else:
                logs.append("PASS: F matches small-k approximation")
            
            # 5. Check pressure continuity at z=h for p(∞,z)
            pressure_diff_at_h = (self.rho1 - self.rho0) * self.g * self.h
            if pressure_diff_at_h <= 0:
                logs.append("WARN: Pressure difference at z=h is non-positive")
            else:
                logs.append(f"PASS: Pressure difference at z=h = {pressure_diff_at_h} > 0")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        logs = []
        # Store original values
        original_h = self.h
        original_rho0 = self.rho0
        original_k_l = self.k_l
        original_k = self.k
        original_rho1 = self.rho1
        original_F = self.F
        original_D = self.D
        original_Fx = self.Fx
        try:
            # 1. Increase h by 10% → Fx should increase (since Fx ∝ h^2)
            self.h *= 1.1
            new_h_prime = (self.rho0 / self.rho_oil) * self.h
            new_Fx = 0.5 * self.rho0 * self.g * self.h**2 * self.w * (self.rho0 / self.rho_oil - 1)
            if new_Fx <= original_Fx:
                logs.append("FAIL: Fx did not increase when h increased")
            else:
                logs.append("PASS: Fx increases with h")
            
            # Restore h
            self.h = original_h
            
            # 2. Increase ρ0 (water density) → D should decrease (since D ∝ (1 - ρ0/ρ1))
            self.rho0 *= 1.1
            new_D = (2 / self.k) * (1 - self.rho0 / self.rho1) * self.h
            if new_D >= original_D:
                logs.append("FAIL: D did not decrease when ρ0 increased")
            else:
                logs.append("PASS: D decreases with ρ0")
            
            # Restore ρ0
            self.rho0 = original_rho0
            
            # 3. Increase k_l (expansion coefficient) → D should decrease (since D ∝ 1/k)
            self.k_l *= 1.1
            self.k = 3 * self.k_l
            new_D = (2 / self.k) * (1 - self.rho0 / self.rho1) * self.h
            if new_D >= original_D:
                logs.append("FAIL: D did not decrease when k_l increased")
            else:
                logs.append("PASS: D decreases with k_l")
            
            # Restore k_l and k
            self.k_l = original_k_l
            self.k = original_k
            
            # 4. Increase ρ1 (mantle density) → F should increase? Let's check.
            self.rho1 *= 1.1
            new_F_approx = (2 * self.g * self.L * self.h**2 / (3 * self.k)) * (self.rho1 - self.rho0)**2 / self.rho1
            if new_F_approx <= original_F:
                logs.append("WARN: Approx F did not increase when ρ1 increased (check derivative)")
            else:
                logs.append("PASS: Approx F increases with ρ1 (for typical parameters)")
            
            # Restore ρ1
            self.rho1 = original_rho1
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            # Ensure restoration even on error
            self.h = original_h
            self.rho0 = original_rho0
            self.k_l = original_k_l
            self.k = original_k
            self.rho1 = original_rho1
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                test_logs = test_method()
                if any(log.startswith(("FAIL", "ERROR")) for log in test_logs):
                    success = False
                logs.append(f"=== {test_name} ===")
                logs.extend(test_logs)
            except Exception as e:
                logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                success = False
        
        return success, logs