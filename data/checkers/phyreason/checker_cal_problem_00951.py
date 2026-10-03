import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results using getattr to avoid AttributeError."""
        # Inputs from the problem - use getattr with sensible defaults
        # But if not found, we will try to get from results or use safe defaults
        self.U0 = getattr(solution_instance, 'U0', 
                         getattr(solution_instance, 'U', 100.0))  # Primary voltage amplitude (using sample value)
        self.omega = getattr(solution_instance, 'omega', 0.0)    # Angular frequency (not used)
        self.n = getattr(solution_instance, 'n', 5.0)            # Turns ratio (using sample value)
        self.R = getattr(solution_instance, 'R', 4.0)            # Motor coil resistance (using sample value)
        self.m = getattr(solution_instance, 'm', 10.0)           # Mass lifted (using sample value)
        self.I = getattr(solution_instance, 'I', 2.0)            # Secondary current (using sample value)
        self.g = getattr(solution_instance, 'g', 9.8)            # Gravity (using sample value)
        
        # Results from the solution
        results = getattr(solution_instance, 'results', {})
        self.I_0 = results.get('I_0', 0.0)        # Primary current
        self.U_2 = results.get('U_2', 0.0)        # Motor terminal voltage
        self.P = results.get('P', 0.0)            # Electrical power to motor
        self.P_thermal = results.get('P_thermal', 0.0)  # Thermal power in motor
        self.P_prime = results.get("P'", 
                                  results.get('P_prime', 0.0))  # Mechanical output power
        self.v = results.get('v', 0.0)            # Lifting velocity
        
    def test_invariants(self):
        """Check physical laws: transformer relationships, power balance, energy conservation."""
        logs = []
        
        # Guard against zero denominators
        if self.n == 0:
            logs.append("SKIP: n is zero, cannot check transformer relationships.")
            return logs
        if self.m == 0 or self.g == 0:
            logs.append("SKIP: m or g is zero, cannot check mechanical power.")
            # Continue with other checks
        
        # 1. Transformer current relationship: I_0 = I/n
        expected_I0 = self.I / self.n
        if not math.isclose(self.I_0, expected_I0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Transformer current invariant. Expected I_0={expected_I0:.3g}, got {self.I_0:.3g}")
        
        # 2. Transformer voltage relationship (RMS): U_2 = U0/(sqrt(2)*n)
        expected_U2 = self.U0 / (math.sqrt(2) * self.n)
        if not math.isclose(self.U_2, expected_U2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Transformer voltage invariant. Expected U_2={expected_U2:.3g}, got {self.U_2:.3g}")
        
        # 3. Electrical power: P = U_2 * I
        expected_P = self.U_2 * self.I
        if not math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Electrical power invariant. Expected P={expected_P:.3g}, got {self.P:.3g}")
        
        # 4. Thermal power: P_thermal = I^2 * R
        expected_Pth = self.I**2 * self.R
        if not math.isclose(self.P_thermal, expected_Pth, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Joule's law invariant. Expected P_thermal={expected_Pth:.3g}, got {self.P_thermal:.3g}")
        
        # 5. Power balance: P = P_prime + P_thermal (energy conservation)
        if not math.isclose(self.P, self.P_prime + self.P_thermal, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Power balance invariant. P={self.P:.3g} != P'+P_thermal={self.P_prime + self.P_thermal:.3g}")
        
        # 6. Mechanical power: P_prime = m * g * v (constant velocity)
        if self.m != 0 and self.g != 0:
            expected_Pprime = self.m * self.g * self.v
            if not math.isclose(self.P_prime, expected_Pprime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Mechanical power invariant. Expected P'={expected_Pprime:.3g}, got {self.P_prime:.3g}")
        
        return logs
    
    def test_trace_consistency(self):
        """Check that intermediate calculations match the final answer."""
        logs = []
        
        # Guard against zero denominators
        if self.n == 0 or self.m == 0 or self.g == 0:
            logs.append("SKIP: n, m, or g is zero, cannot compute expected v.")
            return logs
        
        # Recompute v from the derived formula and compare
        # v = (I * (sqrt(2)*U0 - 2*n*I*R)) / (2*n*m*g)
        numerator = self.I * (math.sqrt(2) * self.U0 - 2 * self.n * self.I * self.R)
        denominator = 2 * self.n * self.m * self.g
        if denominator == 0:
            logs.append("SKIP: denominator is zero, cannot compute expected v.")
            return logs
        expected_v = numerator / denominator
        
        if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Final velocity mismatch. Expected v={expected_v:.3g}, got {self.v:.3g}")
        
        # Check that P matches the expression from step_4
        if self.n == 0:
            logs.append("SKIP: n is zero, cannot compute expected P.")
            return logs
        expected_P_expr = math.sqrt(2) * self.U0 * self.I / (2 * self.n)
        if not math.isclose(self.P, expected_P_expr, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Power expression mismatch. Expected P={expected_P_expr:.3g}, got {self.P:.3g}")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb mass m and check that velocity v decreases."""
        logs = []
        
        if self.n == 0 or self.m == 0 or self.g == 0:
            logs.append("SKIP: n, m, or g is zero, cannot check monotonicity.")
            return logs
        
        original_m = self.m
        
        # Perturb mass upwards by 10%
        perturbed_m = self.m * 1.1
        # Recompute v using the formula with perturbed mass
        numerator = self.I * (math.sqrt(2) * self.U0 - 2 * self.n * self.I * self.R)
        denominator = 2 * self.n * perturbed_m * self.g
        if denominator == 0:
            logs.append("SKIP: denominator is zero after perturbation.")
            return logs
        perturbed_v = numerator / denominator
        
        # v should decrease when m increases (inverse relationship)
        if perturbed_v >= self.v:
            logs.append(f"FAIL: Monotonicity check. v did not decrease when m increased. Original v={self.v:.3g}, perturbed v={perturbed_v:.3g}")
        
        return logs
    
    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        logs = []
        
        # Run each test
        try:
            logs.extend(self.test_invariants())
            logs.extend(self.test_trace_consistency())
            logs.extend(self.test_monotonicity())
        except Exception as e:
            logs.append(f"ERROR during testing: {str(e)}")
            traceback.print_exc()
        
        success = (len(logs) == 0)
        if success:
            logs.append("All physics checks passed.")
        
        return success, logs