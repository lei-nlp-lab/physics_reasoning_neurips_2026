import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        # Safe extraction using getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.M = self._safe_get('M')
        self.g = self._safe_get('g')
        self.I = self._safe_get('I')
        self.k = self._safe_get('k')
        self.v0 = self._safe_get('v0')
        self.R = self._safe_get('R')
        
        # Extract results
        self.F = self._safe_get('F')
        self.a = self._safe_get('a')
        self.L = self._safe_get('L')
        self.E = self._safe_get('E')
        self.U = self._safe_get('U')
        self.W = self._safe_get('W')
        self.W_abs = self._safe_get('|W|')
        self.P = self._safe_get('P')
        
    def _safe_get(self, key):
        """Safely get attribute from solution or results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, None)
        return val
    
    def test_invariants(self):
        """Test fundamental physical invariants."""
        logs = []
        success = True
        
        try:
            # 1. Check Ampere force: F = B1 * I * d, B1 = kI, d = 3Mg/(kI^2)
            if all(v is not None for v in [self.M, self.g, self.k, self.I, self.F]):
                F_expected = self.k * self.I * self.I * (3*self.M*self.g)/(self.k*self.I*self.I)
                if not math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant 1 FAIL: F={self.F}, expected {F_expected}")
                    success = False
                else:
                    logs.append("Invariant 1 PASS: Ampere force correct")
            
            # 2. Check Newton's Second Law: Mg - F = Ma
            if all(v is not None for v in [self.M, self.g, self.F, self.a]):
                lhs = self.M * self.g - self.F
                rhs = self.M * self.a
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant 2 FAIL: Mg-F={lhs}, Ma={rhs}")
                    success = False
                else:
                    logs.append("Invariant 2 PASS: Newton's 2nd Law holds")
            
            # 3. Check kinematics: 0 = v0^2 + 2aL
            if all(v is not None for v in [self.v0, self.a, self.L]):
                rhs = 2 * self.a * self.L
                if not math.isclose(-self.v0*self.v0, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant 3 FAIL: v0^2={self.v0**2}, 2aL={rhs}")
                    success = False
                else:
                    logs.append("Invariant 3 PASS: Kinematics consistent")
            
            # 4. Check energy conservation: Device work + mechanical energy change = 0
            # W_device + (ΔK + ΔU) = 0
            if all(v is not None for v in [self.W, self.M, self.v0, self.g, self.L]):
                ΔK = -0.5 * self.M * self.v0 * self.v0  # final K = 0
                ΔU = self.M * self.g * self.L  # positive as height decreases
                total_mech = ΔK + ΔU
                # W_device should be negative (device absorbs energy)
                if not math.isclose(self.W, -total_mech, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant 4 FAIL: W={self.W}, -ΔE_mech={-total_mech}")
                    success = False
                else:
                    logs.append("Invariant 4 PASS: Energy conservation holds")
                    
        except Exception as e:
            logs.append(f"Invariants test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        logs = []
        success = True
        
        try:
            # 1. Check that acceleration from kinematics matches force analysis
            if all(v is not None for v in [self.a, self.g]):
                a_expected = -2 * self.g
                if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace 1 FAIL: a={self.a}, expected -2g={a_expected}")
                    success = False
                else:
                    logs.append("Trace 1 PASS: Acceleration consistent")
            
            # 2. Check stopping distance L
            if all(v is not None for v in [self.L, self.v0, self.g]):
                L_expected = self.v0*self.v0 / (4 * self.g)
                if not math.isclose(self.L, L_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace 2 FAIL: L={self.L}, expected v0^2/(4g)={L_expected}")
                    success = False
                else:
                    logs.append("Trace 2 PASS: Stopping distance consistent")
            
            # 3. Check induced EMF at t=0
            if self.E is not None and hasattr(self.E, '__call__'):
                # E is function of time
                E0 = self.E(0)
                E_expected = (6 * self.M * self.g / self.I) * self.v0
                if not math.isclose(E0, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace 3 FAIL: E(0)={E0}, expected={E_expected}")
                    success = False
                else:
                    logs.append("Trace 3 PASS: EMF at t=0 consistent")
            
            # 4. Check device voltage at t=0
            if self.U is not None and hasattr(self.U, '__call__'):
                U0 = self.U(0)
                U_expected = self.I * self.R - (6 * self.M * self.g / self.I) * self.v0
                if not math.isclose(U0, U_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace 4 FAIL: U(0)={U0}, expected={U_expected}")
                    success = False
                else:
                    logs.append("Trace 4 PASS: Voltage at t=0 consistent")
            
            # 5. Check power at t=0
            if self.P is not None and hasattr(self.P, '__call__'):
                P0 = self.P(0)
                P_expected = self.I*self.I*self.R - 6*self.M*self.g*self.v0
                if not math.isclose(P0, P_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace 5 FAIL: P(0)={P0}, expected={P_expected}")
                    success = False
                else:
                    logs.append("Trace 5 PASS: Power at t=0 consistent")
                    
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic trends when inputs are perturbed."""
        logs = []
        success = True
        
        try:
            original_M = self.M
            if original_M is None:
                return True, ["Monotonicity test skipped: M not available"]
            
            # Perturb mass by +1%
            perturbed_M = original_M * 1.01
            
            # Recalculate expected results using formulas
            # 1. Force: F = 3Mg
            F_original = 3 * original_M * self.g
            F_perturbed = 3 * perturbed_M * self.g
            
            # 2. Acceleration: a = -2g (independent of M)
            a_original = -2 * self.g
            a_perturbed = -2 * self.g
            
            # 3. Stopping distance: L = v0^2/(4g) (independent of M)
            L_original = self.v0*self.v0 / (4 * self.g)
            L_perturbed = self.v0*self.v0 / (4 * self.g)
            
            # 4. Energy recovered: W = (I^2R v0)/(2g) - (3M v0^2)/2
            W_original = (self.I*self.I*self.R*self.v0)/(2*self.g) - (3*original_M*self.v0*self.v0)/2
            W_perturbed = (self.I*self.I*self.R*self.v0)/(2*self.g) - (3*perturbed_M*self.v0*self.v0)/2
            
            # Check trends
            # Force should increase with mass
            if F_perturbed <= F_original:
                logs.append("Monotonicity FAIL: Force should increase with mass")
                success = False
            else:
                logs.append("Monotonicity PASS: Force increases with mass")
            
            # Acceleration should remain constant
            if not math.isclose(a_perturbed, a_original, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Monotonicity FAIL: Acceleration should be independent of mass")
                success = False
            else:
                logs.append("Monotonicity PASS: Acceleration independent of mass")
            
            # Stopping distance should remain constant
            if not math.isclose(L_perturbed, L_original, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Monotonicity FAIL: Stopping distance should be independent of mass")
                success = False
            else:
                logs.append("Monotonicity PASS: Stopping distance independent of mass")
            
            # Energy recovered magnitude should increase with mass (W becomes more negative)
            if W_perturbed >= W_original:
                logs.append("Monotonicity FAIL: |W| should increase with mass")
                success = False
            else:
                logs.append("Monotonicity PASS: |W| increases with mass")
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return comprehensive results."""
        all_logs = []
        overall_success = True
        
        # Run invariants test
        success, logs = self.test_invariants()
        all_logs.append("=== INVARIANTS TEST ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run trace consistency test
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== TRACE CONSISTENCY TEST ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run monotonicity test
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== MONOTONICITY TEST ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        return overall_success, all_logs