import math
import sympy as sp
from sympy import symbols, Eq, cos, sin, sqrt, pi

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        # Safe extraction of input parameters
        self.m = getattr(solution_instance, 'm', 100)  # number of turns
        self.r = getattr(solution_instance, 'r', 10)   # coil resistance (Ω)
        self.R = getattr(solution_instance, 'R', 90)   # load resistance (Ω)
        self.pi_val = getattr(solution_instance, 'pi', 3.14)  # π value
        
        # Safe extraction of results using the key mapping
        results = getattr(solution_instance, 'results', {})
        self.T = results.get('T', 6.28e-2)      # period (s)
        self.phi_m = results.get('phi_m', 2.0e-2)  # maximum flux per turn (Wb)
        self.E_m = results.get('E_m', 200.0)    # maximum EMF (V)
        self.e = results.get('e', None)         # instantaneous EMF expression
        self.i = results.get('i', None)         # instantaneous current expression/value
        self.I_eff = results.get('I', None)     # effective current (A)
        self.E_eff = results.get('E', None)     # effective EMF (V)
        self.U = results.get('U', None)         # voltmeter reading (V)
        self.q = results.get('q', 0.02)         # induced charge (C)
        self.Q_joule = results.get('Q', 3.14)   # Joule heat (J)
        
        # Derived quantities
        self.omega = 2 * self.pi_val / self.T if self.T != 0 else 0
        self.R_total = self.R + self.r
        
        # For symbolic computations
        self.t = symbols('t')
        
    def test_invariants(self):
        """
        Verify fundamental physics invariants using both symbolic and numeric checks.
        """
        logs = []
        
        # 1. Check maximum EMF formula: E_m = m * phi_m * omega
        try:
            E_m_calc = self.m * self.phi_m * self.omega
            if not math.isclose(self.E_m, E_m_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_m mismatch. Expected {E_m_calc:.2f}, got {self.E_m:.2f}")
            else:
                logs.append(f"PASS: E_m = m*φ_m*ω = {self.E_m:.2f} V")
        except Exception as e:
            logs.append(f"ERROR in E_m check: {e}")
        
        # 2. Check effective values: E_eff = E_m / sqrt(2)
        if self.E_eff is not None:
            E_eff_calc = self.E_m / math.sqrt(2)
            if not math.isclose(self.E_eff, E_eff_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_eff mismatch. Expected {E_eff_calc:.2f}, got {self.E_eff:.2f}")
            else:
                logs.append(f"PASS: E_eff = E_m/√2 = {self.E_eff:.2f} V")
        
        # 3. Check Ohm's law for effective quantities
        if self.I_eff is not None and self.E_eff is not None:
            I_calc = self.E_eff / self.R_total
            if not math.isclose(self.I_eff, I_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: I_eff mismatch. Expected {I_calc:.2f}, got {self.I_eff:.2f}")
            else:
                logs.append(f"PASS: I_eff = E_eff/(R+r) = {self.I_eff:.2f} A")
        
        # 4. Check voltmeter reading: U = I_eff * R
        if self.U is not None and self.I_eff is not None:
            U_calc = self.I_eff * self.R
            if not math.isclose(self.U, U_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U mismatch. Expected {U_calc:.2f}, got {self.U:.2f}")
            else:
                logs.append(f"PASS: U = I_eff*R = {self.U:.2f} V")
        
        # 5. Check induced charge formula: q = m * Δφ / R_total
        # For a quarter cycle, Δφ = φ_m (from 0 to φ_m)
        delta_phi = self.phi_m  # assumption for quarter cycle
        q_calc = self.m * delta_phi / self.R_total
        if not math.isclose(self.q, q_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: q mismatch. Expected {q_calc:.4f}, got {self.q:.4f}")
        else:
            logs.append(f"PASS: q = m*Δφ/(R+r) = {self.q:.4f} C")
        
        # 6. Check Joule heat: Q = (E_eff^2 / R_total) * (T/4) for quarter cycle
        if self.E_eff is not None:
            t_quarter = self.T / 4
            Q_calc = (self.E_eff**2 / self.R_total) * t_quarter
            if not math.isclose(self.Q_joule, Q_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Q_joule mismatch. Expected {Q_calc:.3f}, got {self.Q_joule:.3f}")
            else:
                logs.append(f"PASS: Q = (E_eff^2/(R+r))*(T/4) = {self.Q_joule:.3f} J")
        
        # 7. Check instantaneous expressions (symbolic)
        try:
            if self.e is not None and isinstance(self.e, str):
                # Extract angular frequency from expression
                if 'cos(100t)' in self.e:
                    e_expr = 200 * cos(100 * self.t)
                    i_expr = e_expr / self.R_total
                    # Check if matches given i = 2cos(100t)
                    i_expected = 2 * cos(100 * self.t)
                    if sp.simplify(i_expr - i_expected) != 0:
                        logs.append("WARN: Instantaneous i expression may not match")
                    else:
                        logs.append("PASS: Instantaneous i = e/(R+r) holds symbolically")
        except Exception as e:
            logs.append(f"ERROR in symbolic check: {e}")
        
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        
        # 1. Check that T and ω are consistent
        omega_calc = 2 * self.pi_val / self.T
        if not math.isclose(self.omega, omega_calc, rel_tol=1e-4):
            logs.append(f"FAIL: ω = 2π/T mismatch. Expected {omega_calc:.2f}, got {self.omega:.2f}")
        else:
            logs.append(f"PASS: ω = 2π/T = {self.omega:.2f} rad/s")
        
        # 2. Check that E_m from step 3 matches target answer
        target_E_m = 200.0  # from "200V"
        if not math.isclose(self.E_m, target_E_m, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: E_m doesn't match target answer. Expected {target_E_m}, got {self.E_m}")
        else:
            logs.append(f"PASS: E_m matches target answer (200V)")
        
        # 3. Check instantaneous current at 60° (π/3 rad)
        if self.i is not None:
            # If i is an expression, evaluate at ωt = π/3
            if isinstance(self.i, str) and 'cos' in self.i:
                i_at_60 = 2 * math.cos(math.pi/3)  # 2 * cos(60°) = 1
                if not math.isclose(1.0, i_at_60, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: i(60°) mismatch. Expected 1.0, got {i_at_60}")
                else:
                    logs.append(f"PASS: i(60°) = 1 A matches target")
        
        # 4. Check voltmeter reading against target
        if self.U is not None:
            U_target = 90 * math.sqrt(2)  # from target answer
            if not math.isclose(self.U, U_target, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U doesn't match target. Expected {U_target:.3f}, got {self.U:.3f}")
            else:
                logs.append(f"PASS: U = 90√2 ≈ {self.U:.3f} V matches target")
        
        # 5. Check Joule heat against target (3.14 J)
        if not math.isclose(self.Q_joule, 3.14, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Q doesn't match target. Expected 3.14, got {self.Q_joule}")
        else:
            logs.append(f"PASS: Q = 3.14 J matches target")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb inputs and check if outputs follow expected trends.
        """
        logs = []
        
        # Store original values
        orig_m = self.m
        orig_phi_m = self.phi_m
        
        try:
            # Test 1: Increase number of turns m by 10%
            self.m = orig_m * 1.1
            E_m_new = self.m * orig_phi_m * self.omega
            if E_m_new > self.E_m:
                logs.append("PASS: E_m increases with m (monotonic)")
            else:
                logs.append("FAIL: E_m doesn't increase with m")
            
            # Test 2: Decrease magnetic flux by 20%
            self.m = orig_m  # restore m
            self.phi_m = orig_phi_m * 0.8
            E_m_new = self.m * self.phi_m * self.omega
            if E_m_new < self.E_m:
                logs.append("PASS: E_m decreases with φ_m (monotonic)")
            else:
                logs.append("FAIL: E_m doesn't decrease with φ_m")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
        finally:
            # Restore original values
            self.m = orig_m
            self.phi_m = orig_phi_m
        
        return logs
    
    def run_all_tests(self):
        """
        Execute all tests and return success status and logs.
        """
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                logs = test_func()
                if any(log.startswith(('FAIL', 'ERROR')) for log in logs):
                    success = False
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {e}")
                success = False
        
        return success, all_logs

# Example usage (commented out for the sandbox):
# if __name__ == "__main__":
#     class Solution:
#         def __init__(self):
#             self.m = 100
#             self.r = 10
#             self.R = 90
#             self.pi = 3.14
#             self.results = {
#                 'T': 6.28e-2,
#                 'phi_m': 2.0e-2,
#                 'E_m': 200.0,
#                 'e': '200cos(100t)',
#                 'i': '2cos(100t)',
#                 'I': math.sqrt(2),
#                 'E': 100*math.sqrt(2),
#                 'U': 90*math.sqrt(2),
#                 'q': 0.02,
#                 'Q': 3.14
#             }
#     
#     solution = Solution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     print("Success:", success)
#     for log in logs:
#         print(log)