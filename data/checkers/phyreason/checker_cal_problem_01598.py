import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely with fallbacks
        self.E = getattr(solution_instance, 'E', 5.0)
        self.r = getattr(solution_instance, 'r', 10.0)
        self.R = getattr(solution_instance, 'R', 90.0)
        self.R0_max = 400.0  # given in problem statement
        
        # Extract results safely from instance or its results dictionary
        self.results = getattr(solution_instance, 'results', {})
        self.P_min = getattr(solution_instance, 'P_min', self.results.get('P_min', None))
        self.P_0 = getattr(solution_instance, 'P_0', self.results.get('P_0', None))
        self.P_R = getattr(solution_instance, 'P_R', self.results.get('P_R', None))
        self.P_max = getattr(solution_instance, 'P_max', self.results.get('P_max', None))
        
        # Optional: R0 values at critical points (may not be present)
        self.R0_at_Pmax = getattr(solution_instance, 'R0_at_Pmax', self.results.get('R0_at_Pmax', None))
        self.R0_at_P_Rmax = getattr(solution_instance, 'R0_at_P_Rmax', self.results.get('R0_at_P_Rmax', None))
        
        self.logs = []
    
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify Joule's Law, Ohm's Law, and power conservation."""
        logs = []
        try:
            # Define symbols
            R0 = sp.symbols('R0', positive=True, real=True)
            I = self.E / (R0 + self.R + self.r)
            P_total_battery = self.E * I
            P_total_resistors = I**2 * (self.R + self.r + R0)
            
            # Symbolic check: total power from battery equals total dissipated power
            expr = sp.simplify(P_total_battery - P_total_resistors)
            if expr != 0:
                logs.append(f"Power conservation symbolic check: {expr} (should be zero)")
            else:
                logs.append("Power conservation holds symbolically.")
            
            # Numerical checks at key R0 values
            test_points = [0.0, 100.0, 400.0]
            for r0 in test_points:
                I_num = self.E / (r0 + self.R + self.r)
                P_batt = self.E * I_num
                P_res = I_num**2 * (self.R + self.r + r0)
                if not math.isclose(P_batt, P_res, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Power conservation fails at R0={r0}: battery={P_batt:.4f} W, resistors={P_res:.4f} W")
                else:
                    logs.append(f"Power conservation OK at R0={r0}.")
                    
                # Check Joule's Law for each resistor
                if r0 > 0:
                    P0 = I_num**2 * r0
                    # Compare with formula from step 1 (if P_0 is a function, we can't directly here)
                Pr = I_num**2 * self.r
                PR = I_num**2 * self.R
                logs.append(f"  At R0={r0}: I={I_num:.4f} A, P_r={Pr:.4f} W, P_R={PR:.4f} W")
            
            # Check maximum power transfer for R0 (should be at R0 = R + r = 100 Ohm)
            if self.R0_at_Pmax is not None:
                if not math.isclose(self.R0_at_Pmax, 100.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"R0 at P0 max is {self.R0_at_Pmax} Ohm, expected 100 Ohm")
                else:
                    logs.append("R0 for maximum P0 matches theoretical 100 Ohm.")
            
            # Check P_max value (should be E^2/(4*(R+r)) = 25/400 = 0.0625)
            if self.P_max is not None:
                expected_Pmax = self.E**2 / (4 * (self.R + self.r))
                if not math.isclose(self.P_max, expected_Pmax, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"P_max is {self.P_max} W, expected {expected_Pmax} W")
                else:
                    logs.append(f"P_max matches theoretical value {expected_Pmax} W.")
            
            # Check P_min (should be at R0=400)
            if self.P_min is not None:
                I_min = self.E / (self.R0_max + self.R + self.r)
                expected_Pmin = I_min**2 * (self.R + self.r)
                if not math.isclose(self.P_min, expected_Pmin, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"P_min is {self.P_min} W, expected {expected_Pmin} W")
                else:
                    logs.append(f"P_min matches theoretical value {expected_Pmin} W.")
            
            # Check P_R maximum (should be at R0=0)
            if self.P_R is not None:
                I_for_PRmax = self.E / (self.R + self.r)  # R0=0
                expected_PRmax = I_for_PRmax**2 * self.R
                if not math.isclose(self.P_R, expected_PRmax, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"P_R is {self.P_R} W, expected {expected_PRmax} W")
                else:
                    logs.append(f"P_R matches theoretical maximum {expected_PRmax} W.")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # If the solution_instance has intermediate steps, we could check them.
            # For now, we'll check consistency between P_0, P_max, and P_min.
            # P_0 is the power on R0 as a function. We can check that at R0=100, P_0 = P_max.
            if self.P_max is not None and self.R0_at_Pmax is not None:
                I_at_max = self.E / (self.R0_at_Pmax + self.R + self.r)
                P0_at_max = I_at_max**2 * self.R0_at_Pmax
                if not math.isclose(P0_at_max, self.P_max, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Inconsistency: P0 at R0={self.R0_at_Pmax} is {P0_at_max} W, but P_max is {self.P_max} W")
                else:
                    logs.append(f"P0 at R0={self.R0_at_Pmax} matches P_max.")
            
            # Check that P_min is indeed the minimum of P_sum (r and R) over [0,400]
            # We can check a few points around the claimed minimum (at R0=400)
            if self.P_min is not None:
                R0_test = [350.0, 400.0, 450.0]  # 450 is outside range but still informative
                P_sum_vals = []
                for r0 in R0_test:
                    I_test = self.E / (r0 + self.R + self.r)
                    P_sum = I_test**2 * (self.R + self.r)
                    P_sum_vals.append(P_sum)
                # Check that at R0=400, P_sum is the smallest among the tested
                min_val = min(P_sum_vals)
                idx_min = P_sum_vals.index(min_val)
                if not math.isclose(P_sum_vals[1], self.P_min, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"P_min does not match computed P_sum at R0=400: {P_sum_vals[1]} vs {self.P_min}")
                else:
                    logs.append(f"P_min matches computed P_sum at R0=400.")
                if idx_min != 1 and R0_test[idx_min] < 400:
                    logs.append(f"Warning: P_sum might be lower at R0={R0_test[idx_min]} than at 400.")
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs
    
    def test_monotonicity(self):
        """Perturb R0 and check expected trends."""
        logs = []
        try:
            # Test P_sum (power on r and R) decreases as R0 increases
            R0_vals = [0.0, 200.0, 400.0]
            P_sum_vals = []
            for r0 in R0_vals:
                I = self.E / (r0 + self.R + self.r)
                P_sum = I**2 * (self.R + self.r)
                P_sum_vals.append(P_sum)
            if not (P_sum_vals[0] > P_sum_vals[1] and P_sum_vals[1] > P_sum_vals[2]):
                logs.append(f"P_sum (r+R) not monotonically decreasing: {P_sum_vals}")
            else:
                logs.append("P_sum (r+R) decreases with increasing R0 (OK).")
            
            # Test P_R (power on R) decreases as R0 increases
            P_R_vals = []
            for r0 in R0_vals:
                I = self.E / (r0 + self.R + self.r)
                P_R = I**2 * self.R
                P_R_vals.append(P_R)
            if not (P_R_vals[0] > P_R_vals[1] and P_R_vals[1] > P_R_vals[2]):
                logs.append(f"P_R not monotonically decreasing: {P_R_vals}")
            else:
                logs.append("P_R decreases with increasing R0 (OK).")
            
            # Test P_0 (power on R0) has a maximum at R0=100
            R0_for_P0 = [50.0, 100.0, 150.0]
            P0_vals = []
            for r0 in R0_for_P0:
                I = self.E / (r0 + self.R + self.r)
                P0 = I**2 * r0
                P0_vals.append(P0)
            if not (P0_vals[0] < P0_vals[1] and P0_vals[1] > P0_vals[2]):
                logs.append(f"P0 does not peak near R0=100: {P0_vals}")
            else:
                logs.append("P0 peaks near R0=100 (OK).")
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
        return logs
    
    def run_all_tests(self):
        """Run all checks and return (success_bool, logs)."""
        self.logs = []
        self.log("=== PhysicsChecker started ===")
        
        # Run tests
        inv_logs = self.test_invariants()
        self.log("--- test_invariants ---")
        self.log("\n".join(inv_logs))
        
        trace_logs = self.test_trace_consistency()
        self.log("--- test_trace_consistency ---")
        self.log("\n".join(trace_logs))
        
        mono_logs = self.test_monotonicity()
        self.log("--- test_monotonicity ---")
        self.log("\n".join(mono_logs))
        
        # Determine overall success: no severe errors and all key results match
        success = True
        # Check if any of the key results are missing
        key_results = [self.P_min, self.P_R, self.P_max]
        if any(r is None for r in key_results):
            self.log("WARNING: Some key results are missing.")
            success = False
        else:
            # Check numerical values against theory with tolerance
            expected_Pmax = self.E**2 / (4 * (self.R + self.r))
            if not math.isclose(self.P_max, expected_Pmax, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: P_max mismatch: got {self.P_max}, expected {expected_Pmax}")
                success = False
            
            I_min = self.E / (self.R0_max + self.R + self.r)
            expected_Pmin = I_min**2 * (self.R + self.r)
            if not math.isclose(self.P_min, expected_Pmin, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: P_min mismatch: got {self.P_min}, expected {expected_Pmin}")
                success = False
            
            I_for_PRmax = self.E / (self.R + self.r)
            expected_PRmax = I_for_PRmax**2 * self.R
            if not math.isclose(self.P_R, expected_PRmax, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: P_R mismatch: got {self.P_R}, expected {expected_PRmax}")
                success = False
        
        self.log(f"=== Overall success: {success} ===")
        return success, self.logs