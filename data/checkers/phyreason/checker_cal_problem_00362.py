import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants (known from context)
        self.E = 7.5          # V
        self.r = 1.0          # Ω
        self.R = 0.5          # Ω (motor coil resistance)
        self.U_L = 3.0        # V (bulb normal voltage)
        self.P_L = 4.5        # W (bulb normal power)
        
        # Extract results with safe access
        self.I = self._safe_get('I')
        self.U = self._safe_get('U')
        self.P_out = self._safe_get('P_out')
        
        # Setup symbolic variables for laws
        self._setup_symbols()
        
        self.logs = []
    
    def _safe_get(self, key):
        """Safe extraction with multiple fallback strategies."""
        # Try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return float(val) if isinstance(val, (int, float, str)) else val
        
        # Try results dict
        if key in self.results:
            val = self.results[key]
            try:
                return float(val)
            except:
                return val
        
        # Try parsing from string if needed
        if hasattr(self.solution, 'answer'):
            ans_str = str(self.solution.answer)
            if key == 'I' and 'A' in ans_str:
                try:
                    return float(ans_str.split('A')[0].strip())
                except:
                    pass
            if key == 'P_out' and 'W' in ans_str:
                try:
                    return float(ans_str.split('W')[0].strip())
                except:
                    pass
        
        return 0.0
    
    def _setup_symbols(self):
        """Initialize symbolic variables for law checking."""
        self.I_sym, self.U_sym, self.P_out_sym = sp.symbols('I U P_out')
        self.E_sym, self.r_sym, self.R_sym = sp.symbols('E r R')
        self.U_L_sym, self.P_L_sym = sp.symbols('U_L P_L')
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            return False
    
    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def test_invariants(self):
        """Check conservation laws and fundamental principles."""
        checks = []
        
        # 1. Kirchhoff's Voltage Law (KVL) check
        # E = I*r + U_L + U (from steps analysis)
        if self.I and self.U:
            lhs = self.E
            rhs = self.I * self.r + self.U_L + self.U
            kvl_ok = self._is_close(lhs, rhs)
            checks.append(self._log(
                f"KVL: E({self.E}) = I*r({self.I*self.r}) + U_L({self.U_L}) + U({self.U}) = {rhs}",
                kvl_ok
            ))
            
            # Symbolic KVL check
            kvl_eq = sp.Eq(self.E_sym, self.I_sym*self.r_sym + self.U_L_sym + self.U_sym)
            if not kvl_ok:
                self.logs.append(f"DEBUG: Symbolic KVL: {kvl_eq}")
        
        # 2. Power balance (energy conservation)
        # Battery power output = sum of power consumptions
        if self.I and self.P_out:
            # Battery power: E*I
            # Power losses: I²*r (internal) + P_L (bulb) + I²*R (motor coil) + P_out (mechanical)
            P_battery = self.E * self.I
            P_losses = (self.I**2 * self.r) + self.P_L + (self.I**2 * self.R) + self.P_out
            
            power_ok = self._is_close(P_battery, P_losses)
            checks.append(self._log(
                f"Power balance: E*I({P_battery}) = I²*r({self.I**2*self.r}) + P_L({self.P_L}) + "
                f"I²*R({self.I**2*self.R}) + P_out({self.P_out}) = {P_losses}",
                power_ok
            ))
        
        # 3. Bulb power relation: P_L = U_L * I (should hold exactly for normal operation)
        if self.I:
            bulb_relation = self.P_L / self.U_L  # Should equal I
            bulb_ok = self._is_close(self.I, bulb_relation)
            checks.append(self._log(
                f"Bulb relation: I({self.I}) = P_L/U_L({self.P_L}/{self.U_L} = {bulb_relation})",
                bulb_ok
            ))
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Verify step-by-step consistency with given formulas."""
        checks = []
        
        # Step 1: I = P_L / U_L
        if self.I:
            expected_I = self.P_L / self.U_L  # 4.5/3.0 = 1.5
            step1_ok = self._is_close(self.I, expected_I)
            checks.append(self._log(
                f"Step1: I = P_L/U_L = {self.P_L}/{self.U_L} = {expected_I} ≈ {self.I}",
                step1_ok
            ))
        
        # Step 2: U = E - I*r - U_L
        if self.I and self.U:
            expected_U = self.E - self.I * self.r - self.U_L  # 7.5 - 1.5*1 - 3 = 3.0
            step2_ok = self._is_close(self.U, expected_U)
            checks.append(self._log(
                f"Step2: U = E - I*r - U_L = {self.E} - {self.I}*{self.r} - {self.U_L} = {expected_U} ≈ {self.U}",
                step2_ok
            ))
        
        # Step 3: P_out = U*I - I²*R
        if self.I and self.U and self.P_out:
            expected_P_out = self.U * self.I - (self.I**2) * self.R  # 3*1.5 - 1.5²*0.5 = 3.375
            step3_ok = self._is_close(self.P_out, expected_P_out)
            checks.append(self._log(
                f"Step3: P_out = U*I - I²*R = {self.U}*{self.I} - ({self.I}²)*{self.R} = {expected_P_out} ≈ {self.P_out}",
                step3_ok
            ))
        
        return all(checks)
    
    def test_monotonicity(self):
        """Check that perturbing inputs produces physically plausible changes."""
        checks = []
        
        if not (self.I and self.U and self.P_out):
            return self._log("Monotonicity: Missing values for test", False)
        
        # Store original values
        orig_r = self.r
        
        try:
            # Perturb internal resistance r by +10%
            perturbed_r = self.r * 1.1  # 1.1 Ω instead of 1.0 Ω
            
            # Recalculate using the same formulas (keeping I fixed from bulb constraint)
            # Since bulb operates normally, I remains P_L/U_L = 1.5 A
            I_perturbed = self.P_L / self.U_L  # Still 1.5 A
            
            # New voltage across motor: U' = E - I'*r' - U_L
            U_perturbed = self.E - I_perturbed * perturbed_r - self.U_L
            
            # New output power: P_out' = U'*I' - I'²*R
            P_out_perturbed = U_perturbed * I_perturbed - (I_perturbed**2) * self.R
            
            # Expected monotonic trend: Increasing r should decrease P_out
            # (more voltage drop in internal resistance leaves less for motor)
            monotonic_ok = P_out_perturbed < self.P_out
            
            # Also check that U decreased
            voltage_ok = U_perturbed < self.U
            
            # Log the perturbation results
            self.logs.append(f"MONOTONICITY TEST:")
            self.logs.append(f"  Original: r={self.r}, I={self.I}, U={self.U}, P_out={self.P_out}")
            self.logs.append(f"  Perturbed (+10% r): r={perturbed_r}, I={I_perturbed}, U={U_perturbed}, P_out={P_out_perturbed}")
            
            checks.append(self._log(
                f"Monotonic: Increasing r decreases P_out ({P_out_perturbed} < {self.P_out})",
                monotonic_ok
            ))
            checks.append(self._log(
                f"Monotonic: Increasing r decreases U ({U_perturbed} < {self.U})",
                voltage_ok
            ))
            
            # Restore original r (in memory)
            self.r = orig_r
            
        except Exception as e:
            self.r = orig_r  # Ensure restoration
            checks.append(self._log(
                f"Monotonicity test failed with error: {str(e)}",
                False
            ))
        
        return all(checks)
    
    def run_all_tests(self):
        """Execute all physics checks and return summary."""
        self.logs = []  # Reset logs
        
        try:
            # Run individual tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            # Overall success
            success = inv_ok and trace_ok and mono_ok
            
            # Summary
            self.logs.append("\n" + "="*50)
            self.logs.append(f"SUMMARY:")
            self.logs.append(f"  Invariants: {'PASS' if inv_ok else 'FAIL'}")
            self.logs.append(f"  Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
            self.logs.append(f"  Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
            self.logs.append(f"  OVERALL: {'PASS' if success else 'FAIL'}")
            
            # Target answer check (bonus)
            if self.P_out:
                target_ok = self._is_close(self.P_out, 3.375)
                self.logs.append(f"\nTARGET CHECK: P_out ≈ 3.375W -> {'MATCH' if target_ok else 'MISMATCH'}")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs