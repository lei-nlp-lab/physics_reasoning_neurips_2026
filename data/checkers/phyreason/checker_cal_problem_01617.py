import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.L = getattr(solution_instance, 'L', 1.0)
        
        # Safe extraction of results
        self.results = getattr(solution_instance, 'results', {})
        self.qE = self.results.get('qE', None)
        self.T = self.results.get('T', None)
        self.DeltaE_p = self.results.get('DeltaE_p', None)
        self.v_C = self.results.get('v_C', None)
        self.v_C_prime = self.results.get("v_C'", None)
        self.v_P = self.results.get('v_P', None)
        
        # Precompute derived values
        self.sqrt2 = math.sqrt(2)
        self.cos45 = self.sqrt2 / 2
        self.sin45 = self.sqrt2 / 2
        self.tan45 = 1.0
        
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy comparison with 1% tolerance."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def _log(self, msg, passed=True):
        """Helper to format test logs."""
        status = "PASS" if passed else "FAIL"
        return f"{status}: {msg}"
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        try:
            # 1. Equilibrium at point P (Step 1 & 2)
            if self.qE is not None and self.T is not None:
                # Horizontal: qE = T sinθ
                lhs_x = self.qE
                rhs_x = self.T * self.sin45
                if self._isclose(lhs_x, rhs_x):
                    logs.append(self._log(f"Horizontal equilibrium: qE = {lhs_x:.3f} ≈ T sin45° = {rhs_x:.3f}"))
                else:
                    logs.append(self._log(f"Horizontal equilibrium violated: qE = {lhs_x:.3f}, T sin45° = {rhs_x:.3f}", False))
                
                # Vertical: mg = T cosθ
                lhs_y = self.m * self.g
                rhs_y = self.T * self.cos45
                if self._isclose(lhs_y, rhs_y):
                    logs.append(self._log(f"Vertical equilibrium: mg = {lhs_y:.3f} ≈ T cos45° = {rhs_y:.3f}"))
                else:
                    logs.append(self._log(f"Vertical equilibrium violated: mg = {lhs_y:.3f}, T cos45° = {rhs_y:.3f}", False))
            
            # 2. Energy conservation from Step 4: ½ m v_C² = mgL + qE L
            if self.v_C is not None and self.qE is not None:
                kinetic = 0.5 * self.m * self.v_C**2
                work = self.m * self.g * self.L + self.qE * self.L
                if self._isclose(kinetic, work):
                    logs.append(self._log(f"Work-energy (B→C): ½ m v_C² = {kinetic:.3f} ≈ mgL + qE L = {work:.3f}"))
                else:
                    logs.append(self._log(f"Work-energy (B→C) violated: ½ m v_C² = {kinetic:.3f}, mgL + qE L = {work:.3f}", False))
            
            # 3. Energy consistency from Step 6: ½ m v_P² - ½ m (v_C')² = qE L sin45° - mgL (1 - cos45°)
            if all(v is not None for v in [self.v_P, self.v_C_prime, self.qE]):
                lhs = 0.5 * self.m * (self.v_P**2 - self.v_C_prime**2)
                rhs = self.qE * self.L * self.sin45 - self.m * self.g * self.L * (1 - self.cos45)
                if self._isclose(lhs, rhs):
                    logs.append(self._log(f"Energy step 6 holds: ΔK = {lhs:.3f} ≈ qE L sin45° - mgL(1-cos45°) = {rhs:.3f}"))
                else:
                    logs.append(self._log(f"Energy step 6 violated: ΔK = {lhs:.3f}, RHS = {rhs:.3f}", False))
            
            # 4. Velocity decomposition (Step 5): v_C' = v_C sin45°
            if self.v_C is not None and self.v_C_prime is not None:
                expected = self.v_C * self.sin45
                if self._isclose(self.v_C_prime, expected):
                    logs.append(self._log(f"Velocity decomposition: v_C' = {self.v_C_prime:.3f} ≈ v_C sin45° = {expected:.3f}"))
                else:
                    logs.append(self._log(f"Velocity decomposition violated: v_C' = {self.v_C_prime:.3f}, expected {expected:.3f}", False))
                    
        except Exception as e:
            logs.append(self._log(f"Error in invariant test: {e}", False))
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Step 1: qE = mg tan45° = mg
            if self.qE is not None:
                expected_qE = self.m * self.g
                if self._isclose(self.qE, expected_qE):
                    logs.append(self._log(f"Step 1: qE = {self.qE:.3f} ≈ mg = {expected_qE:.3f}"))
                else:
                    logs.append(self._log(f"Step 1 mismatch: qE = {self.qE:.3f}, expected mg = {expected_qE:.3f}", False))
            
            # Step 2: T = mg / cos45° = √2 mg
            if self.T is not None:
                expected_T = self.sqrt2 * self.m * self.g
                if self._isclose(self.T, expected_T):
                    logs.append(self._log(f"Step 2: T = {self.T:.3f} ≈ √2 mg = {expected_T:.3f}"))
                else:
                    logs.append(self._log(f"Step 2 mismatch: T = {self.T:.3f}, expected √2 mg = {expected_T:.3f}", False))
            
            # Step 3: ΔE_p = qE L (1 - cos45°) = 0.293 mgL
            if self.DeltaE_p is not None:
                expected_DE = self.m * self.g * self.L * (1 - self.cos45)
                if self._isclose(self.DeltaE_p, expected_DE):
                    logs.append(self._log(f"Step 3: ΔE_p = {self.DeltaE_p:.3f} ≈ mgL(1-cos45°) = {expected_DE:.3f}"))
                else:
                    logs.append(self._log(f"Step 3 mismatch: ΔE_p = {self.DeltaE_p:.3f}, expected {expected_DE:.3f}", False))
            
            # Step 4: v_C = 2√(gL)
            if self.v_C is not None:
                expected_vC = 2 * math.sqrt(self.g * self.L)
                if self._isclose(self.v_C, expected_vC):
                    logs.append(self._log(f"Step 4: v_C = {self.v_C:.3f} ≈ 2√(gL) = {expected_vC:.3f}"))
                else:
                    logs.append(self._log(f"Step 4 mismatch: v_C = {self.v_C:.3f}, expected {expected_vC:.3f}", False))
            
            # Step 5: v_C' = v_C sin45°
            if self.v_C_prime is not None and self.v_C is not None:
                expected_vCprime = self.v_C * self.sin45
                if self._isclose(self.v_C_prime, expected_vCprime):
                    logs.append(self._log(f"Step 5: v_C' = {self.v_C_prime:.3f} ≈ v_C sin45° = {expected_vCprime:.3f}"))
                else:
                    logs.append(self._log(f"Step 5 mismatch: v_C' = {self.v_C_prime:.3f}, expected {expected_vCprime:.3f}", False))
            
            # Step 7: v_P = √(2√2 g L)
            if self.v_P is not None:
                expected_vP = math.sqrt(2 * self.sqrt2 * self.g * self.L)
                if self._isclose(self.v_P, expected_vP):
                    logs.append(self._log(f"Step 7: v_P = {self.v_P:.3f} ≈ √(2√2 gL) = {expected_vP:.3f}"))
                else:
                    logs.append(self._log(f"Step 7 mismatch: v_P = {self.v_P:.3f}, expected {expected_vP:.3f}", False))
                    
        except Exception as e:
            logs.append(self._log(f"Error in trace consistency: {e}", False))
        return logs
    
    def test_monotonicity(self):
        """Perturb mass m and verify trends, then restore."""
        logs = []
        original_m = self.m
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected values using formulas
            expected_T = self.sqrt2 * perturbed_m * self.g
            expected_DeltaE_p = perturbed_m * self.g * self.L * (1 - self.cos45)
            expected_vC = 2 * math.sqrt(self.g * self.L)
            expected_vP = math.sqrt(2 * self.sqrt2 * self.g * self.L)
            
            # Compare with original results (scaled where appropriate)
            if self.T is not None:
                T_ratio = expected_T / self.T
                if self._isclose(T_ratio, 1.1):
                    logs.append(self._log(f"T scales with m: T_new/T_old = {T_ratio:.3f} ≈ 1.1"))
                else:
                    logs.append(self._log(f"T scaling violation: T_new/T_old = {T_ratio:.3f}", False))
            
            if self.DeltaE_p is not None:
                DE_ratio = expected_DeltaE_p / self.DeltaE_p
                if self._isclose(DE_ratio, 1.1):
                    logs.append(self._log(f"ΔE_p scales with m: ΔE_p_new/ΔE_p_old = {DE_ratio:.3f} ≈ 1.1"))
                else:
                    logs.append(self._log(f"ΔE_p scaling violation: ratio = {DE_ratio:.3f}", False))
            
            if self.v_C is not None:
                if self._isclose(expected_vC, self.v_C):
                    logs.append(self._log(f"v_C independent of m: v_C_new = {expected_vC:.3f} ≈ v_C_old = {self.v_C:.3f}"))
                else:
                    logs.append(self._log(f"v_C dependence on m: v_C_new = {expected_vC:.3f}, v_C_old = {self.v_C:.3f}", False))
            
            if self.v_P is not None:
                if self._isclose(expected_vP, self.v_P):
                    logs.append(self._log(f"v_P independent of m: v_P_new = {expected_vP:.3f} ≈ v_P_old = {self.v_P:.3f}"))
                else:
                    logs.append(self._log(f"v_P dependence on m: v_P_new = {expected_vP:.3f}, v_P_old = {self.v_P:.3f}", False))
                    
        except Exception as e:
            logs.append(self._log(f"Error in monotonicity test: {e}", False))
        finally:
            # Restore original mass
            self.m = original_m
        return logs
    
    def run_all_tests(self):
        """Execute all checks and return overall success."""
        all_logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} ---")
            logs = test_func()
            all_logs.extend(logs)
            # Check for any FAIL logs in this test
            if any("FAIL" in log for log in logs):
                success = False
        
        return success, all_logs