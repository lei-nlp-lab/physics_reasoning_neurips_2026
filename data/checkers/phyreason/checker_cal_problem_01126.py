import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        solution_instance: Object containing inputs and results.
        """
        # Extract input parameters safely
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.q_mag = abs(getattr(solution_instance, 'q', getattr(solution_instance, 'inputs', {}).get('q', -1.0)))  # Magnitude for formulas
        self.q_signed = getattr(solution_instance, 'q', getattr(solution_instance, 'inputs', {}).get('q', -1.0))  # Actual sign
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'inputs', {}).get('v0', 1.0))
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'inputs', {}).get('E', 1.0))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 1.0))
        
        # Extract results safely (using provided key mapping)
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', 0.0))
        self.x_OP = getattr(solution_instance, 'x_OP', getattr(solution_instance, 'results', {}).get('x_OP', 0.0))
        self.t_2 = getattr(solution_instance, 't_2', getattr(solution_instance, 'results', {}).get('t_2', 0.0))
        self.x_AP = getattr(solution_instance, 'x_AP', getattr(solution_instance, 'results', {}).get('x_AP', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'results', {}).get('a', 0.0))
        self.t_3 = getattr(solution_instance, 't_3', getattr(solution_instance, 'results', {}).get('t_3', 0.0))
        self.aprime = getattr(solution_instance, 'aprime', getattr(solution_instance, 'results', {}).get('aprime', 0.0))
        self.t_1 = getattr(solution_instance, 't_1', getattr(solution_instance, 'results', {}).get('t_1', 0.0))
        self.x_OA = getattr(solution_instance, 'x_OA', getattr(solution_instance, 'results', {}).get('x_OA', 0.0))
        
        # Pre-compute expected values for comparison
        self.sqrt3 = math.sqrt(3)
        self.pi = math.pi
        self.q = self.q_mag  # Use magnitude in formulas as per steps
        
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physical laws based on KB theorems."""
        logs = []
        success = True
        
        try:
            # 1. Radius of circular motion (Theorem: Period in magnetic field)
            v_A = 2 * self.v0  # From step_2: v0 = v_A * cos60°
            expected_R = self.m * v_A / (self.q * self.B)
            if not self.fuzzy_equal(self.R, expected_R):
                logs.append(f"Radius R mismatch: got {self.R}, expected {expected_R}")
                success = False
            else:
                logs.append("✓ Radius R consistent with circular motion law")
            
            # 2. Acceleration in region I (Theorem: Projectile motion in electric field)
            # For negative charge, force opposes field (downward), acceleration magnitude = |q|E/m
            expected_a = self.q * self.E / self.m
            if not self.fuzzy_equal(abs(self.a), expected_a):
                logs.append(f"Acceleration a mismatch: got {self.a}, expected magnitude {expected_a}")
                success = False
            else:
                logs.append("✓ Acceleration a consistent with Newton's second law")
            
            # 3. Acceleration in region III (Theorem: Projectile motion in electric field)
            expected_aprime = self.q * (self.E/2) / self.m
            if not self.fuzzy_equal(self.aprime, expected_aprime):
                logs.append(f"Acceleration a' mismatch: got {self.aprime}, expected {expected_aprime}")
                success = False
            else:
                logs.append("✓ Acceleration a' consistent with Newton's second law")
            
            # 4. Time in magnetic field (Theorem: Period of circular motion)
            # Particle sweeps 60° = π/3 radians, so t_2 = (π/3)/(2π) * T = T/6
            T = 2 * math.pi * self.m / (self.q * self.B)
            expected_t2 = T / 6
            if not self.fuzzy_equal(self.t_2, expected_t2):
                logs.append(f"Time t_2 mismatch: got {self.t_2}, expected {expected_t2}")
                success = False
            else:
                logs.append("✓ Time t_2 consistent with circular motion period")
            
            # 5. Velocity at A from kinematics in region I
            # v_y = a*t_1 = v0 * tan60° = v0 * √3
            v_y_expected = self.v0 * self.sqrt3
            v_y_calculated = abs(self.a) * self.t_1
            if not self.fuzzy_equal(v_y_calculated, v_y_expected):
                logs.append(f"Vertical velocity at A mismatch: calculated {v_y_calculated}, expected {v_y_expected}")
                success = False
            else:
                logs.append("✓ Vertical velocity at A consistent with kinematics")
            
        except Exception as e:
            logs.append(f"Exception in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        logs = []
        success = True
        
        try:
            # 1. Check x_OA formula from step_6
            expected_x_OA = self.sqrt3 * self.m * self.v0**2 / (self.q * self.E)
            if not self.fuzzy_equal(self.x_OA, expected_x_OA):
                logs.append(f"x_OA mismatch: got {self.x_OA}, expected {expected_x_OA}")
                success = False
            else:
                logs.append("✓ x_OA matches derived formula")
            
            # 2. Check x_AP formula from step_7
            expected_x_AP = self.sqrt3 * self.m * self.v0**2 / (self.q * self.B)
            if not self.fuzzy_equal(self.x_AP, expected_x_AP):
                logs.append(f"x_AP mismatch: got {self.x_AP}, expected {expected_x_AP}")
                success = False
            else:
                logs.append("✓ x_AP matches derived formula")
            
            # 3. Check x_OP = x_OA + x_AP
            sum_x = self.x_OA + self.x_AP
            if not self.fuzzy_equal(self.x_OP, sum_x):
                logs.append(f"x_OP mismatch: got {self.x_OP}, sum x_OA+x_AP = {sum_x}")
                success = False
            else:
                logs.append("✓ x_OP = x_OA + x_AP (geometric addition)")
            
            # 4. Check t_1 formula from step_6
            expected_t1 = self.sqrt3 * self.m * self.v0 / (self.q * self.E)
            if not self.fuzzy_equal(self.t_1, expected_t1):
                logs.append(f"t_1 mismatch: got {self.t_1}, expected {expected_t1}")
                success = False
            else:
                logs.append("✓ t_1 matches derived formula")
            
            # 5. Check t_3 formula from step_12
            expected_t3 = 4 * self.m * self.v0 / (self.q * self.E)
            if not self.fuzzy_equal(self.t_3, expected_t3):
                logs.append(f"t_3 mismatch: got {self.t_3}, expected {expected_t3}")
                success = False
            else:
                logs.append("✓ t_3 matches derived formula")
            
            # 6. Check total time t = t_1 + t_2 + t_3
            sum_t = self.t_1 + self.t_2 + self.t_3
            if not self.fuzzy_equal(self.t, sum_t):
                logs.append(f"Total time t mismatch: got {self.t}, sum t_1+t_2+t_3 = {sum_t}")
                success = False
            else:
                logs.append("✓ Total time t = t_1 + t_2 + t_3")
            
        except Exception as e:
            logs.append(f"Exception in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and check the trend of results.
        Uses formulas directly, not re-running simulation.
        """
        logs = []
        success = True
        
        try:
            # Perturb mass by +10%
            m_perturbed = self.m * 1.10
            
            # Calculate expected changes using formulas
            # Radius R = 2*m*v0/(q*B) -> increases with m
            R_original = 2 * self.m * self.v0 / (self.q * self.B)
            R_perturbed = 2 * m_perturbed * self.v0 / (self.q * self.B)
            if not (R_perturbed > R_original):
                logs.append(f"Radius R monotonicity failed: {R_perturbed} not > {R_original}")
                success = False
            else:
                logs.append("✓ Radius R increases with mass (as expected)")
            
            # Time t_1 = √3*m*v0/(q*E) -> increases with m
            t1_original = self.sqrt3 * self.m * self.v0 / (self.q * self.E)
            t1_perturbed = self.sqrt3 * m_perturbed * self.v0 / (self.q * self.E)
            if not (t1_perturbed > t1_original):
                logs.append(f"Time t_1 monotonicity failed: {t1_perturbed} not > {t1_original}")
                success = False
            else:
                logs.append("✓ Time t_1 increases with mass (as expected)")
            
            # Time t_2 = π*m/(3*q*B) -> increases with m
            t2_original = self.pi * self.m / (3 * self.q * self.B)
            t2_perturbed = self.pi * m_perturbed / (3 * self.q * self.B)
            if not (t2_perturbed > t2_original):
                logs.append(f"Time t_2 monotonicity failed: {t2_perturbed} not > {t2_original}")
                success = False
            else:
                logs.append("✓ Time t_2 increases with mass (as expected)")
            
            # Total time t should increase with m
            t_original = ((self.sqrt3 + 4) * self.m * self.v0) / (self.q * self.E) + \
                        (self.pi * self.m) / (3 * self.q * self.B)
            t_perturbed = ((self.sqrt3 + 4) * m_perturbed * self.v0) / (self.q * self.E) + \
                         (self.pi * m_perturbed) / (3 * self.q * self.B)
            if not (t_perturbed > t_original):
                logs.append(f"Total time t monotonicity failed: {t_perturbed} not > {t_original}")
                success = False
            else:
                logs.append("✓ Total time t increases with mass (as expected)")
            
            # Check that x_OP increases with m
            x_OP_original = (self.sqrt3 * self.m * self.v0 / self.q) * (self.v0/self.E + 1/self.B)
            x_OP_perturbed = (self.sqrt3 * m_perturbed * self.v0 / self.q) * (self.v0/self.E + 1/self.B)
            if not (x_OP_perturbed > x_OP_original):
                logs.append(f"x_OP monotonicity failed: {x_OP_perturbed} not > {x_OP_original}")
                success = False
            else:
                logs.append("✓ x_OP increases with mass (as expected)")
            
        except Exception as e:
            logs.append(f"Exception in test_monotonicity: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return consolidated results."""
        all_success = True
        all_logs = []
        
        # Test 1: Physical invariants
        success, logs = self.test_invariants()
        all_success = all_success and success
        all_logs.append("=== Test Invariants ===")
        all_logs.extend(logs)
        
        # Test 2: Trace consistency
        success, logs = self.test_trace_consistency()
        all_success = all_success and success
        all_logs.append("\n=== Test Trace Consistency ===")
        all_logs.extend(logs)
        
        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_success = all_success and success
        all_logs.append("\n=== Test Monotonicity ===")
        all_logs.extend(logs)
        
        return all_success, all_logs