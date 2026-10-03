import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs with fallbacks
        self.B = 5.0  # Magnetic field (T)
        self.L = 0.2  # Conductor length (m)
        self.R_rod = 1.0  # Rod resistance (Ω)
        self.V_cell = 4.0  # Battery cell voltage (V)
        self.F_resistance = 400.0  # Resistance force (N)
        
        # Initialize results extraction
        self.results = {}
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        
        # Extract numeric values with safe conversion
        self.I = self._extract_numeric('I')
        self.U = self._extract_numeric('U')
        self.n = self._extract_numeric('n')
        self.direction = self._extract_string('direction')
        
        # Store for monotonicity test backup
        self._original_I = float(self.I) if isinstance(self.I, (int, float)) else 0.0
        self._original_U = float(self.U) if isinstance(self.U, (int, float)) else 0.0
        self._original_n = float(self.n) if isinstance(self.n, (int, float)) else 0.0
        
    def _extract_numeric(self, key):
        """Extract numeric value from results dict."""
        val = self.results.get(key)
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return 0.0
        return 0.0
    
    def _extract_string(self, key):
        """Extract string value from results dict."""
        val = self.results.get(key)
        if isinstance(val, str):
            return val
        if val is not None:
            return str(val)
        return ""
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison with physics tolerance."""
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
        except (ValueError, TypeError):
            return False
    
    def test_invariants(self):
        """Verify fundamental physics invariants."""
        logs = []
        
        try:
            # 1. Force equilibrium (constant velocity → net force = 0)
            # Magnetic force should equal resistance force
            F_calc = self.B * self.I * self.L
            if not self._is_close(F_calc, self.F_resistance):
                logs.append(f"FAIL: Force equilibrium violated. F_magnetic = {F_calc:.2f} N, expected {self.F_resistance:.2f} N")
            else:
                logs.append(f"PASS: Force equilibrium F = BIL = {F_calc:.2f} N ≈ {self.F_resistance:.2f} N")
            
            # 2. Ohm's Law
            U_calc = self.I * self.R_rod
            if not self._is_close(U_calc, self.U):
                logs.append(f"FAIL: Ohm's Law violated. U = IR = {U_calc:.2f} V, but reported U = {self.U:.2f} V")
            else:
                logs.append(f"PASS: Ohm's Law U = IR = {U_calc:.2f} V ≈ {self.U:.2f} V")
            
            # 3. Series connection law
            n_calc = self.U / self.V_cell
            if not self._is_close(n_calc, self.n):
                logs.append(f"FAIL: Series connection violated. n = U/V_cell = {n_calc:.2f}, but reported n = {self.n:.2f}")
            else:
                logs.append(f"PASS: Series connection n = U/V_cell = {n_calc:.2f} ≈ {self.n:.2f}")
            
            # 4. Direction consistency (from left-hand rule)
            expected_dir = "rightward"
            if self.direction and isinstance(self.direction, str):
                if expected_dir.lower() not in self.direction.lower():
                    logs.append(f"WARN: Direction mismatch. Expected '{expected_dir}', got '{self.direction}'")
                else:
                    logs.append(f"PASS: Motion direction '{self.direction}' matches left-hand rule")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate calculations with final results."""
        logs = []
        
        try:
            # Step 2 to Step 3: F = BIL → I = F/(BL)
            I_expected = self.F_resistance / (self.B * self.L)
            if not self._is_close(self.I, I_expected):
                logs.append(f"FAIL: Current inconsistent. I = F/(BL) = {I_expected:.2f} A, got {self.I:.2f} A")
            else:
                logs.append(f"PASS: Current I = {self.I:.2f} A matches F/(BL) = {I_expected:.2f} A")
            
            # Step 3 to Step 4: U = IR
            U_expected = self.I * self.R_rod
            if not self._is_close(self.U, U_expected):
                logs.append(f"FAIL: Voltage inconsistent. U = IR = {U_expected:.2f} V, got {self.U:.2f} V")
            else:
                logs.append(f"PASS: Voltage U = {self.U:.2f} V matches IR = {U_expected:.2f} V")
            
            # Step 4 to Step 5: n = U/V_cell
            n_expected = self.U / self.V_cell
            if not self._is_close(self.n, n_expected):
                logs.append(f"FAIL: Battery count inconsistent. n = U/V_cell = {n_expected:.2f}, got {self.n:.2f}")
            else:
                logs.append(f"PASS: Battery count n = {self.n:.2f} matches U/V_cell = {n_expected:.2f}")
            
            # Verify the chain: F → I → U → n
            n_from_F = self.F_resistance / (self.B * self.L) * self.R_rod / self.V_cell
            if not self._is_close(self.n, n_from_F):
                logs.append(f"FAIL: End-to-end chain broken. n(F) = {n_from_F:.2f}, got {self.n:.2f}")
            else:
                logs.append(f"PASS: End-to-end consistency: F → I → U → n chain holds")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            
        return logs
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs."""
        logs = []
        
        try:
            # Perturb resistance force by +10%
            F_perturbed = self.F_resistance * 1.10
            
            # Calculate expected changes using physics formulas
            I_new = F_perturbed / (self.B * self.L)  # I ∝ F
            U_new = I_new * self.R_rod  # U ∝ I
            n_new = U_new / self.V_cell  # n ∝ U
            
            # Check monotonic relationships
            if not (I_new > self._original_I):
                logs.append(f"FAIL: Current should increase when F increases, but I_new = {I_new:.2f} ≤ {self._original_I:.2f}")
            else:
                logs.append(f"PASS: I increases with F (ΔI = {I_new - self._original_I:.2f} A)")
            
            if not (U_new > self._original_U):
                logs.append(f"FAIL: Voltage should increase when F increases, but U_new = {U_new:.2f} ≤ {self._original_U:.2f}")
            else:
                logs.append(f"PASS: U increases with F (ΔU = {U_new - self._original_U:.2f} V)")
            
            if not (n_new > self._original_n):
                logs.append(f"FAIL: Battery count should increase when F increases, but n_new = {n_new:.2f} ≤ {self._original_n:.2f}")
            else:
                logs.append(f"PASS: n increases with F (Δn = {n_new - self._original_n:.2f})")
            
            # Verify proportional relationships
            ratio_F = F_perturbed / self.F_resistance
            ratio_I = I_new / self._original_I
            ratio_U = U_new / self._original_U
            ratio_n = n_new / self._original_n
            
            if not self._is_close(ratio_F, ratio_I):
                logs.append(f"WARN: I ∝ F proportionality weak: ΔF = {ratio_F:.3f}, ΔI = {ratio_I:.3f}")
            if not self._is_close(ratio_I, ratio_U):
                logs.append(f"WARN: U ∝ I proportionality weak: ΔI = {ratio_I:.3f}, ΔU = {ratio_U:.3f}")
            if not self._is_close(ratio_U, ratio_n):
                logs.append(f"WARN: n ∝ U proportionality weak: ΔU = {ratio_U:.3f}, Δn = {ratio_n:.3f}")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            
        return logs
    
    def run_all_tests(self):
        """Run all physics audits and return success status with logs."""
        all_logs = []
        success = True
        
        try:
            # Run each test suite
            invariant_logs = self.test_invariants()
            trace_logs = self.test_trace_consistency()
            monotonicity_logs = self.test_monotonicity()
            
            all_logs.append("=== PHYSICS INVARIANTS ===")
            all_logs.extend(invariant_logs)
            
            all_logs.append("\n=== TRACE CONSISTENCY ===")
            all_logs.extend(trace_logs)
            
            all_logs.append("\n=== MONOTONICITY TESTS ===")
            all_logs.extend(monotonicity_logs)
            
            # Check for critical failures
            critical_failures = [
                log for log in invariant_logs + trace_logs 
                if log.startswith('FAIL:') or log.startswith('ERROR')
            ]
            
            if critical_failures:
                success = False
                all_logs.append(f"\n❌ AUDIT FAILED: {len(critical_failures)} critical issue(s)")
            else:
                all_logs.append(f"\n✅ AUDIT PASSED: All physics checks satisfied")
                
        except Exception as e:
            success = False
            all_logs.append(f"CRITICAL: Unhandled exception in audit: {str(e)}")
            all_logs.append(traceback.format_exc())
            
        return success, all_logs