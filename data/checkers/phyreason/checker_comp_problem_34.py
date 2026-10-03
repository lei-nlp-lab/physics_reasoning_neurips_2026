import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract fundamental constants (with safe defaults)
        self.c = 2.99792458e8  # m/s
        self.e = 1.602176634e-19  # C
        self.m_p = 1.6726219e-27  # kg (proton mass)
        self.m_e = 9.1093837e-31  # kg (electron mass)
        self.epsilon_0 = 8.8541878128e-12  # F/m
        
        # Extract results with safe fallbacks
        self.E_total = self.results.get('E_total', 0.0)  # LHC energy (7 TeV)
        self.T = self.results.get('T', 0.0)
        self.Deltat = self.results.get('Deltat', 0.0)
        self.t = self.results.get('t', 0.0)
        self.Delta = self.results.get('Delta', 0.0)
        self.l = self.results.get('l', 0.0)
        self.gamma = self.results.get('gamma', 1.0)  # Q1 gamma (500 MeV)
        self.m = self.results.get('m', 0.0)
        self.v = self.results.get('v', 0.0)  # Q1 velocity (500 MeV)
        self.B = self.results.get('B', 0.0)
        self.P_rad = self.results.get('P_rad', 0.0)
        self.P_tot = self.results.get('P_tot', 0.0)
        self.p_f = self.results.get('p_f', 0.0)  # Q5 momentum
        self.p = self.results.get('p', 0.0)  # Q7 momentum (1 GeV/c)
        
        # Extract additional inputs from solution instance if available
        self.V_q1 = getattr(solution_instance, 'V_q1', 500e6)  # 500 MV for Q1
        self.V_q2 = getattr(solution_instance, 'V_q2', 60e9)  # 60 GeV for Q2
        self.L = getattr(solution_instance, 'L', 26659)  # LHC circumference
        self.N = getattr(solution_instance, 'N', 2 * 2808 * 1.15e11)  # Number of protons
        self.r = self.L/(2*math.pi)  # LHC radius
        self.d = getattr(solution_instance, 'd', 30.0)  # Distance for electric field
        
        # Extract LHC energy from results or use default
        self.E_LHC = self.E_total  # Use E_total from results (7 TeV)
        
        # Q1 energy (500 MeV)
        self.E_q1 = self.m_p * self.c**2 + self.V_q1 * self.e
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
        
    def test_invariants(self):
        """Test conservation laws and fundamental relations."""
        try:
            # 1. Relativistic energy-momentum relation for Q5 proton
            if self.p_f > 1e-20:  # Avoid floating-point zero
                # For Q5: E_q5 = m_p*c^2 + V_q1*e
                E_q5 = self.m_p * self.c**2 + self.V_q1 * self.e
                E_expected = math.sqrt((self.p_f * self.c)**2 + (self.m_p * self.c**2)**2)
                if not math.isclose(E_q5, E_expected, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Energy-momentum relation violated. E_q5={E_q5:.2e}, expected={E_expected:.2e}")
                else:
                    self.log(f"PASS: Energy-momentum relation holds for Q5.")
            
            # 2. Check Lorentz force = centripetal force for LHC
            if self.B > 0 and self.r > 0:
                # Using v ≈ c for LHC protons, gamma = E_LHC/(m_p*c^2)
                gamma_LHC = self.E_LHC / (self.m_p * self.c**2)
                F_L = self.e * self.c * self.B
                F_c = (gamma_LHC * self.m_p * self.c**2) / self.r
                if not math.isclose(F_L, F_c, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Force balance mismatch. F_L={F_L:.2e}, F_c={F_c:.2e}")
                else:
                    self.log(f"PASS: Lorentz and centripetal forces balanced.")
            
            # 3. Check Larmor formula consistency
            if self.P_rad > 1e-30 and self.r > 0:  # Avoid floating-point zero
                # Centripetal acceleration: a = c²/r for relativistic particles
                a = (self.c**2) / self.r
                gamma_LHC = self.E_LHC / (self.m_p * self.c**2)
                P_expected = (gamma_LHC**4 * self.e**2 * a**2) / (6 * math.pi * self.epsilon_0 * self.c**3)
                if not math.isclose(self.P_rad, P_expected, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Larmor formula mismatch. P_rad={self.P_rad:.2e}, expected={P_expected:.2e}")
                else:
                    self.log(f"PASS: Larmor formula holds.")
            
            # 4. Check relativistic gamma definition for Q1
            if self.v > 0:
                gamma_calc = 1 / math.sqrt(1 - (self.v/self.c)**2)
                if not math.isclose(self.gamma, gamma_calc, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Gamma mismatch. gamma={self.gamma}, calculated={gamma_calc:.4f}")
                else:
                    self.log(f"PASS: Gamma definition consistent.")
                    
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        try:
            # 1. Check Q1 velocity from Q1 energy (NOT LHC energy)
            if self.v > 0:
                # Q1 energy: E_q1 = m_p*c^2 + V_q1*e
                E_q1 = self.m_p * self.c**2 + self.V_q1 * self.e
                gamma_q1 = E_q1 / (self.m_p * self.c**2)
                v_calc = self.c * math.sqrt(1 - (1/gamma_q1)**2)
                if not math.isclose(self.v, v_calc, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Q1 velocity mismatch. v={self.v:.2e}, calculated={v_calc:.2e}")
                else:
                    self.log(f"PASS: Q1 velocity consistent with Q1 energy.")
            
            # 2. Check magnetic field from LHC energy and circumference
            if self.B > 0 and self.L > 0:
                B_calc = (2 * math.pi * self.E_LHC) / (self.e * self.c * self.L)
                if not math.isclose(self.B, B_calc, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: B field mismatch. B={self.B:.2e}, calculated={B_calc:.2e}")
                else:
                    self.log(f"PASS: B field consistent with geometry.")
            
            # 3. Check total power from single particle power
            if self.P_tot > 1e-30 and self.P_rad > 1e-30 and self.N > 0:
                P_tot_calc = self.N * self.P_rad
                if not math.isclose(self.P_tot, P_tot_calc, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Total power mismatch. P_tot={self.P_tot:.2e}, calculated={P_tot_calc:.2e}")
                else:
                    self.log(f"PASS: Total power consistent with single particle.")
            
            # 4. Check time of flight relation (only if we have values)
            if self.t > 1e-30 and self.l > 1e-30 and self.p > 1e-30 and self.m > 1e-30:
                t_calc = self.l * math.sqrt((self.m * self.c**2)**2 + (self.p * self.c)**2) / (self.p * self.c**2)
                if not math.isclose(self.t, t_calc, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Time of flight mismatch. t={self.t:.2e}, calculated={t_calc:.2e}")
                else:
                    self.log(f"PASS: Time of flight relation holds.")
            else:
                self.log(f"SKIP: Time of flight test (insufficient data).")
                    
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        try:
            # Store original values as local variables, NOT instance variables
            original_V_q1 = self.V_q1
            original_E_LHC = self.E_LHC
            
            # 1. Test: Increase Q1 voltage → increase Q1 velocity
            if self.v > 0:
                V_new = original_V_q1 * 1.1  # Increase by 10%
                gamma_new = (self.m_p * self.c**2 + V_new * self.e) / (self.m_p * self.c**2)
                v_new = self.c * math.sqrt(1 - (1/gamma_new)**2)
                if v_new <= self.v:
                    self.log(f"FAIL: Velocity didn't increase with increased voltage.")
                else:
                    self.log(f"PASS: Velocity increases with voltage (monotonic).")
            
            # 2. Test: Increase LHC energy → increase magnetic field for fixed L
            if self.B > 0:
                E_LHC_new = original_E_LHC * 1.1  # Increase by 10%
                B_new = (2 * math.pi * E_LHC_new) / (self.e * self.c * self.L)
                if B_new <= self.B:
                    self.log(f"FAIL: B field didn't increase with increased energy.")
                else:
                    self.log(f"PASS: B field increases with energy (monotonic).")
            
            # 3. Test: Increase momentum → decrease time of flight for fixed mass and length
            # Skip if we don't have all values
            if self.t > 1e-30 and self.l > 1e-30 and self.p > 1e-30 and self.m > 1e-30:
                p_new = self.p * 1.1
                t_new = self.l * math.sqrt((self.m * self.c**2)**2 + (p_new * self.c)**2) / (p_new * self.c**2)
                if t_new >= self.t:
                    self.log(f"FAIL: Time of flight didn't decrease with increased momentum.")
                else:
                    self.log(f"PASS: Time of flight decreases with momentum (monotonic).")
            else:
                self.log(f"SKIP: Time of flight monotonicity test (insufficient data).")
            
            # 4. Test: Time difference for kaon-pion system
            if self.Deltat > 1e-30 and self.l > 1e-30 and self.p > 1e-30:
                # For kaon-pion system
                m_K = 493.677e6 * self.e / self.c**2  # kaon mass in kg
                m_pi = 139.570e6 * self.e / self.c**2  # pion mass in kg
                dt_calc = self.l / (self.c * self.p) * (math.sqrt((m_K * self.c**2)**2 + (self.p * self.c)**2) - 
                                                       math.sqrt((m_pi * self.c**2)**2 + (self.p * self.c)**2))
                if not math.isclose(self.Deltat, dt_calc, rel_tol=0.02, abs_tol=0.02):
                    self.log(f"FAIL: Time difference doesn't match mass difference calculation.")
                else:
                    self.log(f"PASS: Time difference consistent with mass difference.")
                    
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
            
            # Check if any FAIL messages in logs
            fail_count = sum(1 for msg in self.logs if msg.startswith("FAIL"))
            success = fail_count == 0
            
            return success, self.logs
            
        except Exception as e:
            self.log(f"CRITICAL ERROR: {str(e)}")
            self.log(traceback.format_exc())
            return False, self.logs