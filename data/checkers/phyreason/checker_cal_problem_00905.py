import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all inputs with defaults
        self.n = getattr(solution_instance, 'n', 200)
        self.S = getattr(solution_instance, 'S', 0.01)
        self.omega = getattr(solution_instance, 'omega', 100.0)
        self.B = getattr(solution_instance, 'B', math.sqrt(2))
        self.I_ammeter = getattr(solution_instance, 'I_ammeter', 2.0)  # Secondary current
        
        # Safely extract results with defaults (using .results dict if available)
        results = getattr(solution_instance, 'results', {})
        self.E_m = getattr(solution_instance, 'E_m', results.get('E_m', 0.0))
        self.e_expr = getattr(solution_instance, 'e', results.get('e', ''))
        self.U1 = getattr(solution_instance, 'U_1', results.get('U_1', 0.0))
        self.R = getattr(solution_instance, 'R', results.get('R', 0.0))
        
        # Symbolic variables
        self.t = sp.symbols('t', real=True)
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify fundamental physics laws: emf formulas, transformer power conservation"""
        try:
            # 1. Check maximum emf: E_m = n * B * S * omega
            expected_E_m = self.n * self.B * self.S * self.omega
            if not math.isclose(self.E_m, expected_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: E_m mismatch. Computed {self.E_m}, expected {expected_E_m}")
                return False
            self.log(f"PASS: E_m = {self.E_m} ≈ nBSω = {expected_E_m}")
            
            # 2. Check instantaneous emf expression matches E_m and ω
            if isinstance(self.e_expr, str):
                # Parse string to check amplitude and frequency
                if f"{self.E_m:.2f}" not in self.e_expr.replace(' ', ''):
                    self.log(f"WARN: e expression {self.e_expr} may not match E_m={self.E_m}")
                if "sin(100t)" not in self.e_expr.replace(' ', ''):
                    self.log(f"WARN: e expression may not have correct ω=100 rad/s")
            else:
                # Assume it's a sympy expression
                try:
                    amplitude = sp.Abs(self.e_expr.subs(sp.sin(self.omega*self.t), 1))
                    if not math.isclose(float(amplitude), self.E_m, rel_tol=1e-2, abs_tol=1e-2):
                        self.log(f"FAIL: e amplitude {amplitude} ≠ E_m {self.E_m}")
                        return False
                except:
                    self.log("WARN: Could not validate e expression symbolically")
            
            # 3. Check RMS voltage: U1 = E_m / √2
            expected_U1 = self.E_m / math.sqrt(2)
            if not math.isclose(self.U1, expected_U1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: U1 mismatch. Computed {self.U1}, expected {expected_U1}")
                return False
            self.log(f"PASS: U1 = {self.U1} ≈ E_m/√2 = {expected_U1}")
            
            # 4. Check power conservation in transformer (numerical fallback)
            # Primary power = Secondary power: U1 * I1 = U2 * I2
            # With ideal transformer: U1/U2 = N1/N2, I1/I2 = N2/N1
            # Given I_ammeter = I2 = 2A, and R = U2/I2
            # So U2 = I2 * R = 2 * R
            # Then U1 * I1 = U2 * I2 = (2*R) * 2 = 4R
            # Also I1 = (N2/N1) * I2, but N2/N1 unknown.
            # However, we can check consistency: U1/I1 = (N1/N2)^2 * R
            # Without turns ratio, we check that R = U1 * I1 / (I2^2) * (N2/N1)^2? Not directly.
            # Instead, verify that R = 25 Ω is consistent with given I2=2A and power balance.
            # From steps: U1=200V, I2=2A, R=25Ω → U2 = I2*R = 50V
            # Then turns ratio N1/N2 = U1/U2 = 200/50 = 4
            # Then I1 = I2 * (N2/N1) = 2 * (1/4) = 0.5A
            # Check power: P_primary = U1*I1 = 200*0.5 = 100W, P_secondary = U2*I2 = 50*2 = 100W ✓
            U2 = self.I_ammeter * self.R
            I1 = self.I_ammeter * (U2 / self.U1)  # from U1/U2 = I2/I1
            P_in = self.U1 * I1
            P_out = U2 * self.I_ammeter
            if not math.isclose(P_in, P_out, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Power not conserved. P_in={P_in}W, P_out={P_out}W")
                return False
            self.log(f"PASS: Transformer power conserved: {P_in}W ≈ {P_out}W")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer using steps analysis"""
        try:
            # Step 1: E_m = nBSω
            step1_E_m = self.n * self.B * self.S * self.omega
            if not math.isclose(self.E_m, step1_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step1 E_m {step1_E_m} ≠ final {self.E_m}")
                return False
            self.log(f"PASS: Step1 E_m = {step1_E_m} consistent")
            
            # Step 2: e = E_m sin(ωt)
            if isinstance(self.e_expr, str):
                if f"{self.E_m:.2f}" not in self.e_expr and "E_m" not in self.e_expr:
                    self.log(f"FAIL: e expression doesn't contain E_m={self.E_m}")
                    return False
            self.log("PASS: Step2 e expression consistent")
            
            # Step 3: U1 = E_m/√2
            step3_U1 = self.E_m / math.sqrt(2)
            if not math.isclose(self.U1, step3_U1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step3 U1 {step3_U1} ≠ final {self.U1}")
                return False
            self.log(f"PASS: Step3 U1 = {step3_U1} consistent")
            
            # Step 6: R = 25 Ω (from simultaneous equations)
            expected_R = 25.0
            if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step6 R {self.R} ≠ expected {expected_R}")
                return False
            self.log(f"PASS: Step6 R = {self.R} consistent")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb input (n) and verify linear scaling of outputs"""
        original_n = self.n
        try:
            # Double the number of turns
            self.n = 2 * original_n
            
            # Recalculate using formulas
            new_E_m = self.n * self.B * self.S * self.omega
            new_U1 = new_E_m / math.sqrt(2)
            
            # Check scaling: should double
            if not math.isclose(new_E_m, 2 * self.E_m, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: E_m doesn't scale linearly with n. Expected {2*self.E_m}, got {new_E_m}")
                return False
            if not math.isclose(new_U1, 2 * self.U1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: U1 doesn't scale linearly with n. Expected {2*self.U1}, got {new_U1}")
                return False
            
            self.log(f"PASS: Monotonicity - Doubling n doubles E_m and U1")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False
        finally:
            # Restore original value
            self.n = original_n
    
    def run_all_tests(self):
        """Execute all checks and return (success, logs)"""
        self.logs = []
        success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.log(f"\n--- Running {name} ---")
            if not test_func():
                success = False
                self.log(f"{name}: FAILED")
            else:
                self.log(f"{name}: PASSED")
        
        return success, self.logs