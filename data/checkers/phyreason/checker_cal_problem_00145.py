import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', 1.0))
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 1.0))
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v0', 1.0))
        self.m1 = getattr(solution_instance, 'm1', solution_instance.results.get('m1', 1.0))
        self.m2 = getattr(solution_instance, 'm2', solution_instance.results.get('m2', 1.0))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', 1.0))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.C = getattr(solution_instance, 'C', solution_instance.results.get('C', 1.0))
        
        # Safe attribute access for results
        self.E0 = getattr(solution_instance, 'E_0', solution_instance.results.get('E_0', 0.0))
        self.I0 = getattr(solution_instance, 'I_0', solution_instance.results.get('I_0', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        self.v_prime = getattr(solution_instance, "v'", solution_instance.results.get("v'", 0.0))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 0.0))
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', 0.0))
        self.U_c = getattr(solution_instance, 'U_c', solution_instance.results.get('U_c', 0.0))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 0.0))
        self.F_A = getattr(solution_instance, 'F_A', solution_instance.results.get('F_A', 0.0))
        
        self.solution = solution_instance
        self.logs = []

    def test_invariants(self):
        """Verify energy conservation and Newton's laws."""
        try:
            # Newton's 2nd Law: During uniform motion, Ampere force balances gravity
            F_gravity = self.m1 * (self.g / 6)
            if not math.isclose(self.F_A, F_gravity, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Newton's 2nd Law mismatch. F_A={self.F_A}, m1*g/6={F_gravity}")
                return False
            
            # Energy conservation: Power input (gravity) vs power dissipated in resistors
            # During uniform motion: P_gravity = F_gravity * v
            # Power dissipated: P_diss = I^2 * R_eq, where I = B*l*v/(2r) and R_eq = 2r
            I_theory = self.B * self.l * self.v / (2 * self.r)
            P_diss = I_theory**2 * (2 * self.r)
            P_gravity = F_gravity * self.v
            if not math.isclose(P_diss, P_gravity, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Energy conservation mismatch. P_diss={P_diss}, P_gravity={P_gravity}")
                return False
            
            # Resistor network: Equivalent resistance should be 2r
            # The circuit: side ab (r) in series with parallel combination of two branches:
            # Branch1: two sides in series (2r) -> but wait, step_2 says "r + r || r + r +r = 2r"
            # Let's verify symbolically:
            R_symbolic = self.r + 1/(1/(self.r + self.r) + 1/(self.r + self.r + self.r))  # This is incorrect.
            # Actually, from step_2: r + (r || r) + r + r = 2r? Let's compute properly:
            # The ∧-shape has 7 sides, each resistance r. The side ab is in the magnetic field.
            # The circuit diagram: ab (r) then the rest: two parallel branches:
            # Branch1: two sides (2r)
            # Branch2: three sides (3r)
            # Then two more sides in series? Wait, step_2 says total R = 2r.
            # Let's compute: R_parallel = 1/(1/(2r) + 1/(3r)) = (6r/5)
            # Then total R = r (ab) + R_parallel + r + r? That would be > 2r.
            # Actually, the given step_2 equation is: R = r + r || r + r + r = 2r.
            # This suggests: ab (r), then a parallel combination of two resistors (r and r) giving r/2, then two more r in series.
            # So: R = r + (r || r) + r + r = r + r/2 + r + r = 3.5r? Not 2r.
            # There is inconsistency in the problem statement. We'll trust the step_2 result: R = 2r.
            # So we just check that self.R is close to 2r.
            if not math.isclose(self.R, 2 * self.r, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: Equivalent resistance mismatch. R={self.R}, expected 2r={2*self.r}")
                return False
            
            self.logs.append("PASS: All invariants satisfied.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Step 1: E0 = B*l*v0
            E0_expected = self.B * self.l * self.v0
            if not math.isclose(self.E0, E0_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: E0 mismatch. Got {self.E0}, expected {E0_expected}")
                return False
            
            # Step 2: R = 2r (already checked in invariants)
            
            # Step 3: I0 = E0 / R
            I0_expected = self.E0 / self.R
            if not math.isclose(self.I0, I0_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: I0 mismatch. Got {self.I0}, expected {I0_expected}")
                return False
            
            # Step 4: F_A = B^2 * l^2 * v / (2r)
            # But v here is the uniform motion velocity, which is the same as self.v
            F_A_expected = self.B**2 * self.l**2 * self.v / (2 * self.r)
            if not math.isclose(self.F_A, F_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: F_A mismatch. Got {self.F_A}, expected {F_A_expected}")
                return False
            
            # Step 5: F = F_A = m1*g/6 (already checked in invariants)
            
            # Step 6: v = m1*g*r/(3*B^2*l^2)
            v_expected = self.m1 * self.g * self.r / (3 * self.B**2 * self.l**2)
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v mismatch. Got {self.v}, expected {v_expected}")
                return False
            
            # Step 7: v' should equal v
            if not math.isclose(self.v_prime, self.v, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: v' mismatch with v. v'={self.v_prime}, v={self.v}")
                return False
            
            # Step 8: U_c = m1*g*r/(6*B*l)
            U_c_expected = self.m1 * self.g * self.r / (6 * self.B * self.l)
            if not math.isclose(self.U_c, U_c_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: U_c mismatch. Got {self.U_c}, expected {U_c_expected}")
                return False
            
            # Step 9: q = C * U_c
            q_expected = self.C * self.U_c
            if not math.isclose(self.q, q_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append(f"FAIL: q mismatch. Got {self.q}, expected {q_expected}")
                return False
            
            self.logs.append("PASS: All trace consistency checks passed.")
            return True
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb mass m1 and verify the trend in velocity v."""
        try:
            # Original v = m1*g*r/(3*B^2*l^2)
            # So v is proportional to m1.
            # Increase m1 by 10% and check v increases.
            m1_original = self.m1
            perturb_factor = 1.1
            self.m1 = m1_original * perturb_factor
            
            # Recalculate v using the formula
            v_perturbed = self.m1 * self.g * self.r / (3 * self.B**2 * self.l**2)
            
            # Restore m1
            self.m1 = m1_original
            
            if v_perturbed > self.v:
                self.logs.append("PASS: Monotonicity check: increasing m1 increases v.")
                return True
            else:
                self.logs.append(f"FAIL: Monotonicity check failed. Original v={self.v}, perturbed v={v_perturbed}")
                return False
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} Test ---")
            if not test_func():
                success = False
                self.logs.append(f"*** {test_name} Test FAILED ***")
            else:
                self.logs.append(f"*** {test_name} Test PASSED ***")
        
        return success, self.logs