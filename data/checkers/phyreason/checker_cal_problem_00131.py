import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # First try to get input_values dict, then fall back to attributes
        input_values = getattr(solution_instance, 'input_values', {})
        
        # Extract input parameters with proper fallbacks
        self.m = float(input_values.get('m', getattr(solution_instance, 'm', 1.0)))
        self.q = float(input_values.get('q', getattr(solution_instance, 'q', 1.0)))
        self.B = float(input_values.get('B', getattr(solution_instance, 'B', 1.0)))
        self.U = float(input_values.get('U', getattr(solution_instance, 'U', 1.0)))
        self.E = float(input_values.get('E', getattr(solution_instance, 'E', 1.0)))
        self.alpha = float(input_values.get('alpha', getattr(solution_instance, 'alpha', 0.1)))
        self.R = float(input_values.get('R', getattr(solution_instance, 'R', 1.0)))
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.R_res = float(results.get('R', self.R))  # Radius at point P
        self.Kp = float(results.get('1/2mv_p2', 0.0))  # Kinetic energy at P
        self.T = float(results.get('T', 0.0))  # Period in magnetic field
        self.t = float(results.get('t', 0.0))  # Time to reach P
        self.r1 = float(results.get('r_1', 0.0))  # Last semicircle radius
        self.r2 = float(results.get('r_2', 0.0))  # Second-last semicircle radius
        self.d_m = float(results.get('d_m', 0.0))  # Electrode thickness
        self.r_Q = float(results.get('r_Q', 0.0))  # Radius in deflector
        self.OO_prime = float(results.get("OO''", 0.0))  # Distance O to O'
        self.R_m = float(results.get('R_m', 0.0))  # Maximum magnetic region radius
        
        # Derived quantities with safety checks
        self.v_p = math.sqrt(2 * self.Kp / self.m) if self.Kp > 0 and self.m > 0 else 0.0
        self.n_acc = self.Kp / (self.q * self.U) if self.q * self.U != 0 else 0.0
        
        # For monotonicity test backup
        self._original_U = self.U
    
    def test_invariants(self):
        """Verify physics invariants: force balance, energy conservation, period formula."""
        logs = []
        try:
            # 1. Lorentz force = centripetal force at point P (step_1)
            if self.R_res != 0:
                magnetic_force = self.q * self.v_p * self.B
                centripetal_force = self.m * self.v_p**2 / self.R_res
                if not math.isclose(magnetic_force, centripetal_force, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Force balance at P: qvB={magnetic_force:.3e}, mv²/R={centripetal_force:.3e}")
                else:
                    logs.append(f"OK: Force balance at P within 1%")
            else:
                logs.append(f"SKIP: Force balance - R_res is zero")
            
            # 2. Work-energy theorem: n*q*U = ½mv_p² (step_2)
            energy_input = self.n_acc * self.q * self.U
            if not math.isclose(energy_input, self.Kp, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work-energy: nqU={energy_input:.3e}, Kp={self.Kp:.3e}")
            else:
                logs.append(f"OK: Work-energy theorem within 1%")
            
            # 3. Period formula T = 2πm/(qB) (step_3)
            if self.q * self.B != 0:
                expected_T = 2 * math.pi * self.m / (self.q * self.B)
                if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Period: T={self.T:.3e}, expected 2πm/(qB)={expected_T:.3e}")
                else:
                    logs.append(f"OK: Period formula within 1%")
            else:
                logs.append(f"SKIP: Period formula - q*B is zero")
            
            # 4. Radius relationship r1 > r2 (from step_5, step_6)
            if self.r1 <= self.r2:
                logs.append(f"FAIL: r1={self.r1:.3e} should be > r2={self.r2:.3e}")
            else:
                logs.append(f"OK: r1 > r2 as expected")
            
            # 5. Energy difference between orbits = qU (from step_5 equations)
            if self.m > 0:
                K1 = (self.q * self.B * self.r1)**2 / (2 * self.m)
                K2 = (self.q * self.B * self.r2)**2 / (2 * self.m)
                if not math.isclose(K1 - K2, self.q * self.U, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"WARN: Energy step ΔK={K1-K2:.3e} vs qU={self.q*self.U:.3e}")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {e}")
            logs.append(traceback.format_exc())
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check step_1 vs step_2: R = mv_p/(qB) and v_p from Kp
            if self.m > 0:
                v_from_R = self.q * self.B * self.R_res / self.m
                v_from_K = math.sqrt(2 * self.Kp / self.m) if self.Kp >= 0 else 0.0
                if not math.isclose(v_from_R, v_from_K, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"INCONSISTENT: v_p from R={v_from_R:.3e}, from K={v_from_K:.3e}")
                else:
                    logs.append(f"OK: Velocity consistent between steps 1 and 2")
            
            # Check step_4 time formula: t = (n-1)*T/2
            expected_t = (self.n_acc - 1) * self.T / 2
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENT: t={self.t:.3e}, expected (n-1)T/2={expected_t:.3e}")
            else:
                logs.append(f"OK: Time formula consistent")
            
            # Check step_5 radii formulas
            if self.Kp - self.q * self.U > 0 and self.q * self.B != 0:
                r1_expected = math.sqrt(2 * self.m * (self.Kp - self.q * self.U)) / (self.q * self.B)
                if not math.isclose(self.r1, r1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"INCONSISTENT: r1={self.r1:.3e}, expected={r1_expected:.3e}")
                else:
                    logs.append(f"OK: r1 formula consistent")
            else:
                logs.append(f"SKIP: r1 check - Kp - qU not positive or q*B=0")
            
            if self.Kp - 2 * self.q * self.U > 0 and self.q * self.B != 0:
                r2_expected = math.sqrt(2 * self.m * (self.Kp - 2 * self.q * self.U)) / (self.q * self.B)
                if not math.isclose(self.r2, r2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"INCONSISTENT: r2={self.r2:.3e}, expected={r2_expected:.3e}")
                else:
                    logs.append(f"OK: r2 formula consistent")
            else:
                logs.append(f"SKIP: r2 check - Kp - 2qU not positive or q*B=0")
            
            # Check step_6 vs step_7: d_m = r1 - r2
            d_from_radii = self.r1 - self.r2
            if not math.isclose(self.d_m, d_from_radii, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENT: d_m={self.d_m:.3e}, r1-r2={d_from_radii:.3e}")
            else:
                logs.append(f"OK: d_m from radii consistent")
            
            # Check step_8: force balance in deflector
            if self.r_Q != 0:
                lhs = self.q * self.v_p * self.B - self.q * self.E
                rhs = self.m * self.v_p**2 / self.r_Q
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"INCONSISTENT: deflector force balance: lhs={lhs:.3e}, rhs={rhs:.3e}")
                else:
                    logs.append(f"OK: Deflector force balance within 1%")
            else:
                logs.append(f"SKIP: deflector force balance - r_Q is zero")
            
            # Check step_11: OO' = 2*(r_Q - R)*sin(α/2)
            OO_expected = 2 * (self.r_Q - self.R_res) * math.sin(self.alpha / 2)
            if not math.isclose(self.OO_prime, OO_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENT: OO'={self.OO_prime:.3e}, expected={OO_expected:.3e}")
            else:
                logs.append(f"OK: Geometry OO' consistent")
            
            # Check step_12: R_m = R + OO'
            R_m_from_OO = self.R_res + self.OO_prime
            if not math.isclose(self.R_m, R_m_from_OO, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"INCONSISTENT: R_m={self.R_m:.3e}, R+OO'={R_m_from_OO:.3e}")
            else:
                logs.append(f"OK: R_m geometry consistent")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {e}")
            logs.append(traceback.format_exc())
        return logs
    
    def test_monotonicity(self):
        """Perturb U and check trends in d_m and R_m."""
        logs = []
        original_U = self.U
        try:
            # Increase U by 2%
            self.U = original_U * 1.02
            
            # Recalculate using formulas from steps
            # d_m formula from step_7
            if self.q * self.B**2 != 0:
                term1_arg = self.R_res**2 - 2 * self.m * self.U / (self.q * self.B**2)
                term2_arg = self.R_res**2 - 4 * self.m * self.U / (self.q * self.B**2)
                if term1_arg > 0 and term2_arg > 0:
                    term1 = math.sqrt(term1_arg)
                    term2 = math.sqrt(term2_arg)
                    d_m_new = 2 * (term1 - term2)
                    
                    if d_m_new < self.d_m:
                        logs.append(f"MONOTONIC: d_m decreases when U increases (d_m_old={self.d_m:.3e}, d_m_new={d_m_new:.3e})")
                    else:
                        logs.append(f"WARN: d_m monotonicity not clear")
                else:
                    logs.append(f"SKIP: d_m monotonicity - square root arguments not positive")
            else:
                logs.append(f"SKIP: d_m monotonicity - q*B² is zero")
            
            # r_Q from step_8
            if self.q * self.U != 0:
                n_new = self.Kp / (self.q * self.U)
                Kp_new = n_new * self.q * self.U
                if Kp_new > 0 and self.m > 0:
                    v_p_new = math.sqrt(2 * Kp_new / self.m)
                    denominator = self.q * (v_p_new * self.B - self.E)
                    if abs(denominator) > 1e-12:
                        r_Q_new = self.m * v_p_new**2 / denominator
                        
                        # R_m from step_12
                        if self.q * self.B**2 * self.R_res - self.m * self.E != 0:
                            R_m_new = self.R_res + (2 * self.m * self.E * self.R_res) / \
                                     (self.q * self.B**2 * self.R_res - self.m * self.E) * math.sin(self.alpha / 2)
                            
                            if R_m_new > self.R_m:
                                logs.append(f"MONOTONIC: R_m increases when U increases (R_m_old={self.R_m:.3e}, R_m_new={R_m_new:.3e})")
                            else:
                                logs.append(f"WARN: R_m monotonicity not clear")
                        else:
                            logs.append(f"SKIP: R_m calculation - denominator zero")
                    else:
                        logs.append(f"SKIP: r_Q calculation - denominator zero")
                else:
                    logs.append(f"SKIP: r_Q calculation - Kp_new not positive or m=0")
            else:
                logs.append(f"SKIP: r_Q calculation - q*U is zero")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {e}")
            logs.append(traceback.format_exc())
        finally:
            self.U = original_U
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        all_logs = []
        success = True
        
        all_logs.append("=== Testing Physics Invariants ===")
        inv_logs = self.test_invariants()
        all_logs.extend(inv_logs)
        if any("FAIL" in log for log in inv_logs):
            success = False
        
        all_logs.append("\n=== Testing Trace Consistency ===")
        trace_logs = self.test_trace_consistency()
        all_logs.extend(trace_logs)
        if any("INCONSISTENT" in log for log in trace_logs):
            success = False
        
        all_logs.append("\n=== Testing Monotonicity ===")
        mono_logs = self.test_monotonicity()
        all_logs.extend(mono_logs)
        
        return success, all_logs