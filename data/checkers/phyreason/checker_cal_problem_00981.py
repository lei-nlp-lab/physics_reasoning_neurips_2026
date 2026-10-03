import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker by safely extracting inputs and results from solution_instance.
        """
        # Safely extract inputs (assuming they are stored in solution_instance.inputs)
        self.inputs = getattr(solution_instance, 'inputs', {})
        
        # Safely extract results (assuming they are stored in solution_instance.results)
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key parameters from inputs with defaults
        # Use reasonable defaults to avoid division by zero
        self.C = self.inputs.get('C', 1.0)
        self.B = self.inputs.get('B', 1.0)
        self.L = self.inputs.get('L', 1.0)
        self.d = self.inputs.get('d', 1.0)
        self.m = self.inputs.get('m', 1.0)
        self.R = self.inputs.get('R', 1.0)
        self.F0 = self.inputs.get('F0', 1.0)
        self.v1 = self.inputs.get('v1', 2.0)  # velocity of rod a when entering region I
        self.v2 = self.inputs.get('v2', 3.0)  # velocity of rod a at collision with rod b
        
        # Extract results with safe fallbacks
        self.barI = self.results.get('barI', 0.0)
        self.mv = self.results.get('mv', 0.0)
        self.U1 = self.results.get('U_1', 0.0)
        self.F = self.results.get('F', 0.0)
        self.x_b = self.results.get('x_b', 0.0)
        self.v_b = self.results.get('v_b', 0.0)
        self.mv_mv1 = self.results.get('mv-mv_1', 0.0)
        self.q = self.results.get('q', 0.0)
        self.U2 = self.results.get('U_2', 0.0)
        self.BLv = self.results.get('BLv', 0.0)
        self.v = self.results.get('v', 0.0)  # velocity of rod a (steady-state)
        self.q1 = self.results.get('q_1', 0.0)
        self.E0 = self.results.get('E_0', 0.0)
        self.Q_a = self.results.get('Q_a', 0.0)
        
        # For region I exit velocity, check if 'v_exit' exists in results
        self.v_exit = self.results.get('v_exit', self.v1 - (self.B**2 * self.L**2 * self.d) / (2 * self.m * self.R))
        
        self.logs = []
        
    def log(self, msg, is_error=False):
        """Add a log message."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {msg}")
        
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Energy conservation for rod a in magnetic field region I.
        2. Impulse-momentum theorem for rod a.
        3. Magnetic force equals thread tension for rod b.
        """
        self.log("Testing invariants...")
        
        # 1. Energy conservation for rod a (step 12)
        # Q_a should be half the kinetic energy loss (since half goes to capacitor)
        K1 = 0.5 * self.m * self.v1**2  # initial KE in region I
        K_exit = 0.5 * self.m * self.v_exit**2  # final KE after region I
        expected_Q_a = 0.5 * (K1 - K_exit)   # from step 12
        
        if not math.isclose(self.Q_a, expected_Q_a, rel_tol=0.02, abs_tol=0.02):
            self.log(f"Energy conservation violation: Q_a={self.Q_a}, expected {expected_Q_a}", is_error=True)
            return False
        
        # 2. Impulse-momentum theorem for rod a (step 3)
        # Magnetic impulse = change in momentum for steady-state acceleration
        # B*L*q = m*v (from step 3)
        expected_q = self.m * self.v / (self.B * self.L)
        if not math.isclose(self.q, expected_q, rel_tol=0.02, abs_tol=0.02):
            self.log(f"Impulse-momentum violation: q={self.q}, expected m*v/(B*L)={expected_q}", is_error=True)
            return False
        
        # 3. Magnetic force on rod b (step 6)
        # At breaking point: B * I * L = F0
        # We can compute I from circuit equation: BLv = 2*I*R => I = BLv/(2R)
        I_from_circuit = self.B * self.L * self.v / (2 * self.R)
        expected_F = self.B * I_from_circuit * self.L
        if not math.isclose(self.F, expected_F, rel_tol=0.02, abs_tol=0.02):
            self.log(f"Magnetic force mismatch: F={self.F}, expected {expected_F}", is_error=True)
            return False
        
        self.log("All invariants passed.")
        return True
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        """
        self.log("Testing trace consistency...")
        
        # Check step 1: U1 = E0
        if not math.isclose(self.U1, self.E0, rel_tol=0.02, abs_tol=0.02):
            self.log(f"U1 mismatch: U1={self.U1}, E0={self.E0}", is_error=True)
            return False
        
        # Check step 2: U2 = B * L * v (steady-state v)
        expected_U2 = self.B * self.L * self.v
        if not math.isclose(self.U2, expected_U2, rel_tol=0.02, abs_tol=0.02):
            self.log(f"U2 mismatch: U2={self.U2}, B*L*v={expected_U2}", is_error=True)
            return False
        
        # Check step 4: q = C * (U1 - U2)
        expected_q = self.C * (self.U1 - self.U2)
        if not math.isclose(self.q, expected_q, rel_tol=0.02, abs_tol=0.02):
            self.log(f"q mismatch: q={self.q}, C*(U1-U2)={expected_q}", is_error=True)
            return False
        
        # Check step 5: B * L * v = 2 * I * R (after insulation)
        # Compute I from F = B*I*L = F0
        I_from_force = self.F0 / (self.B * self.L)
        expected_BLv = 2 * I_from_force * self.R
        if not math.isclose(self.BLv, expected_BLv, rel_tol=0.02, abs_tol=0.02):
            self.log(f"BLv mismatch: BLv={self.BLv}, expected 2*I*R={expected_BLv}", is_error=True)
            return False
        
        # Check step 7: E0 expression
        # E0 = (2*R*F0*(C*B^2*L^2 + m)) / (C*B^2*L^2)
        denominator = self.C * self.B**2 * self.L**2
        if abs(denominator) < 1e-12:
            self.log("Denominator too small for E0 check, skipping", is_error=False)
        else:
            expected_E0 = (2 * self.R * self.F0 * (self.C * self.B**2 * self.L**2 + self.m)) / denominator
            if not math.isclose(self.E0, expected_E0, rel_tol=0.02, abs_tol=0.02):
                self.log(f"E0 mismatch: E0={self.E0}, formula gives {expected_E0}", is_error=True)
                return False
        
        # Check step 11: v_exit = v1 - (B^2 * L^2 * d) / (2 * m * R)
        expected_v_exit = self.v1 - (self.B**2 * self.L**2 * self.d) / (2 * self.m * self.R)
        if not math.isclose(self.v_exit, expected_v_exit, rel_tol=0.02, abs_tol=0.02):
            self.log(f"v_exit mismatch: v_exit={self.v_exit}, formula gives {expected_v_exit}", is_error=True)
            return False
        
        # Check step 13: Q_a expression
        expected_Q_a = (self.B**2 * self.L**2 * self.d) / (8 * self.R) * (2 * self.v1 - (self.B**2 * self.L**2 * self.d) / (2 * self.m * self.R))
        if not math.isclose(self.Q_a, expected_Q_a, rel_tol=0.02, abs_tol=0.02):
            self.log(f"Q_a mismatch: Q_a={self.Q_a}, formula gives {expected_Q_a}", is_error=True)
            return False
        
        # Check step 14/16: v_b = v2 / 6
        expected_v_b = self.v2 / 6
        if not math.isclose(self.v_b, expected_v_b, rel_tol=0.02, abs_tol=0.02):
            self.log(f"v_b mismatch: v_b={self.v_b}, v2/6={expected_v_b}", is_error=True)
            return False
        
        self.log("All trace consistency checks passed.")
        return True
    
    def test_monotonicity(self):
        """
        Perturb an input parameter (mass m) and verify the trend in outputs.
        """
        self.log("Testing monotonicity by perturbing mass m...")
        
        # Save original values
        original_m = self.m
        original_E0 = self.E0
        original_Q_a = self.Q_a
        original_v_b = self.v_b
        
        # Perturb m by +10%
        perturbed_m = original_m * 1.1
        
        # Recalculate expected values using formulas from steps
        # E0 = (2*R*F0*(C*B^2*L^2 + m)) / (C*B^2*L^2)
        denominator = self.C * self.B**2 * self.L**2
        if abs(denominator) < 1e-12:
            self.log("Denominator too small for monotonicity check, skipping E0", is_error=False)
            new_E0 = original_E0
        else:
            new_E0 = (2 * self.R * self.F0 * (self.C * self.B**2 * self.L**2 + perturbed_m)) / denominator
        
        # Q_a = (B^2 L^2 d)/(8R) * (2v1 - (B^2 L^2 d)/(2mR))
        new_Q_a = (self.B**2 * self.L**2 * self.d) / (8 * self.R) * (2 * self.v1 - (self.B**2 * self.L**2 * self.d) / (2 * perturbed_m * self.R))
        
        # v_b = v2 / 6 (independent of m, but v2 might change; we assume v2 constant for this check)
        new_v_b = self.v2 / 6  # unchanged if v2 constant
        
        # Check trends
        # E0 should increase with m (since it's additive in numerator)
        if denominator > 1e-12 and new_E0 <= original_E0:
            self.log(f"E0 monotonicity violation: m increased but E0 did not increase", is_error=True)
            return False
        
        # Q_a should increase with m (since the subtracted term decreases)
        if new_Q_a <= original_Q_a:
            self.log(f"Q_a monotonicity violation: m increased but Q_a did not increase", is_error=True)
            return False
        
        # v_b should remain constant (if v2 constant)
        if not math.isclose(new_v_b, original_v_b, rel_tol=0.02, abs_tol=0.02):
            self.log(f"v_b changed unexpectedly with m", is_error=True)
            return False
        
        # Restore original values
        self.m = original_m
        self.E0 = original_E0
        self.Q_a = original_Q_a
        self.v_b = original_v_b
        
        self.log("Monotonicity test passed.")
        return True
    
    def run_all_tests(self):
        """
        Run all tests and return success boolean and logs.
        """
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.log(f"Test '{test_name}' failed.", is_error=True)
            except Exception as e:
                success = False
                self.log(f"Test '{test_name}' crashed: {str(e)}\n{traceback.format_exc()}", is_error=True)
        
        return success, self.logs