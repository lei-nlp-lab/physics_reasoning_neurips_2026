import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr and .get() to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Extract input parameters with safe defaults
        self.L = getattr(solution_instance, 'L', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.U = getattr(solution_instance, 'U', 10.0)
        self.B = getattr(solution_instance, 'B', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Extract results with safe defaults
        results = getattr(solution_instance, 'results', {})
        self.T1 = results.get('T_1', 0.0)
        self.T2 = results.get('T_2', 0.0)
        self.F1 = results.get('F_1', 0.0)
        self.F2 = results.get('F_2', 0.0)
        self.I1 = results.get('I_1', 0.0)
        self.I2 = results.get('I_2', 0.0)
        self.theta = results.get('theta', 0.0)  # in degrees
        
        # Handle ratio: could be float or string
        ratio_val = results.get('T_1:T_2', 0.0)
        if isinstance(ratio_val, str):
            # Try to parse string like "5:3"
            try:
                if ':' in ratio_val:
                    a, b = map(float, ratio_val.split(':'))
                    self.ratio_val = a / b if b != 0 else 0.0
                else:
                    self.ratio_val = float(ratio_val)
            except:
                self.ratio_val = 0.0
        else:
            # Assume it's already a numeric value
            self.ratio_val = float(ratio_val)
    
    def test_invariants(self):
        """Test physical invariants: force equilibrium, Ohm's law, Ampere's law."""
        logs = []
        success = True
        
        # 1. Ohm's Law for Figure A: I1 = U / (3R)
        I1_expected = self.U / (3 * self.R)
        if not math.isclose(self.I1, I1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Ohm's Law for Figure A: I1 = {self.I1}, expected {I1_expected}")
            success = False
        else:
            logs.append("PASS: Ohm's Law for Figure A")
        
        # 2. Ampere's Law for Figure A: F1 = B * (3L) * I1 = B*U*L/R
        F1_expected = self.B * self.U * self.L / self.R
        if not math.isclose(self.F1, F1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Ampere's Law for Figure A: F1 = {self.F1}, expected {F1_expected}")
            success = False
        else:
            logs.append("PASS: Ampere's Law for Figure A")
        
        # 3. Ohm's Law for Figure B: I2 = U / R
        I2_expected = self.U / self.R
        if not math.isclose(self.I2, I2_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Ohm's Law for Figure B: I2 = {self.I2}, expected {I2_expected}")
            success = False
        else:
            logs.append("PASS: Ohm's Law for Figure B")
        
        # 4. Ampere's Law for Figure B: F2 = B * L * I2 = B*U*L/R
        F2_expected = self.B * self.U * self.L / self.R
        if not math.isclose(self.F2, F2_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Ampere's Law for Figure B: F2 = {self.F2}, expected {F2_expected}")
            success = False
        else:
            logs.append("PASS: Ampere's Law for Figure B")
        
        # 5. Force equilibrium for Figure A: 3*m*g + F1 = 2*T1*sin(theta)
        # sin(theta) from geometry: tanθ = 3/4 → sinθ = 3/5 = 0.6
        sin_theta = 3/5
        left_side_A = 3 * self.m * self.g + self.F1
        right_side_A = 2 * self.T1 * sin_theta
        if not math.isclose(left_side_A, right_side_A, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Force equilibrium for Figure A: left={left_side_A}, right={right_side_A}")
            success = False
        else:
            logs.append("PASS: Force equilibrium for Figure A")
        
        # 6. Force equilibrium for Figure B: 3*m*g + F2 = 2*T2
        left_side_B = 3 * self.m * self.g + self.F2
        right_side_B = 2 * self.T2
        if not math.isclose(left_side_B, right_side_B, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Force equilibrium for Figure B: left={left_side_B}, right={right_side_B}")
            success = False
        else:
            logs.append("PASS: Force equilibrium for Figure B")
        
        # 7. Check that the angle theta is consistent with geometry: tan(theta) = 3/4
        theta_expected_rad = math.atan(3/4)
        theta_expected_deg = math.degrees(theta_expected_rad)
        if not math.isclose(self.theta, theta_expected_deg, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"WARN: Theta = {self.theta}°, expected ~{theta_expected_deg}° (from tan θ=3/4)")
            # Not a failure
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with each other and the final answer."""
        logs = []
        success = True
        
        # 1. Check that T1 and T2 are consistent with the ratio
        if self.T2 != 0:
            ratio_computed = self.T1 / self.T2
            # The expected ratio is 5/3 from step9
            if not math.isclose(ratio_computed, 5/3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Ratio T1/T2 = {ratio_computed}, expected 5/3 ≈ {5/3}")
                success = False
            else:
                logs.append("PASS: Ratio T1:T2 = 5:3")
        
        # 2. Check that the stored ratio value matches 5/3
        if not math.isclose(self.ratio_val, 5/3, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Stored ratio value = {self.ratio_val}, expected 5/3 ≈ {5/3}")
            success = False
        else:
            logs.append("PASS: Stored ratio value matches 5:3")
        
        # 3. Check that the magnetic forces F1 and F2 are equal (both = B*U*L/R)
        if not math.isclose(self.F1, self.F2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F1 ({self.F1}) != F2 ({self.F2})")
            success = False
        else:
            logs.append("PASS: F1 = F2 = B*U*L/R")
        
        # 4. Check that I1 and I2 follow the resistance ratios: I2 = 3 * I1
        if self.I1 != 0:
            if not math.isclose(self.I2 / self.I1, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: I2/I1 = {self.I2/self.I1}, expected 3.0")
                success = False
            else:
                logs.append("PASS: I2 = 3 * I1")
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter (e.g., B) and check that the tensions T1 and T2
        change in the expected direction (both increase with B).
        """
        logs = []
        success = True
        
        # Original values
        B_original = self.B
        T1_original = self.T1
        T2_original = self.T2
        
        # Perturb B by +10%
        self.B = B_original * 1.1
        
        # Recalculate T1 and T2 using formulas derived from physics
        # T1 = (3*m*g + B*U*L/R) / (2 * sinθ) with sinθ = 0.6
        new_T1 = (3 * self.m * self.g + self.B * self.U * self.L / self.R) / (2 * 0.6)
        # T2 = (3*m*g + B*U*L/R) / 2
        new_T2 = (3 * self.m * self.g + self.B * self.U * self.L / self.R) / 2
        
        # Restore original B
        self.B = B_original
        
        # Check monotonicity: both should increase when B increases
        if new_T1 <= T1_original or new_T2 <= T2_original:
            logs.append(f"FAIL: Monotonicity w.r.t B: T1 changed from {T1_original} to {new_T1}, T2 from {T2_original} to {new_T2} (both should increase)")
            success = False
        else:
            logs.append("PASS: T1 and T2 increase when B increases")
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"--- {name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, all_logs