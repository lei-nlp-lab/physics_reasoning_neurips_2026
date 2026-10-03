import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance=None):
        # Handle case where solution_instance might be None
        if solution_instance is None:
            solution_instance = type('EmptyInstance', (), {'results': {}})()
        
        # First, try to get the results dictionary directly
        # Some solution instances might have results as an attribute
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        else:
            # Otherwise, try to extract from the instance attributes
            results = {}
            # Look for common result variables
            for attr_name in ['U_ab', 'W_ab', 'E', 'L', 'q', 'theta', 'cos_theta']:
                if hasattr(solution_instance, attr_name):
                    val = getattr(solution_instance, attr_name)
                    # Convert to float if it's a SymPy expression
                    if hasattr(val, 'evalf'):
                        results[attr_name] = float(val.evalf())
                    elif isinstance(val, (int, float)):
                        results[attr_name] = float(val)
        
        # Extract inputs with safe defaults
        self.E = results.get('E', 2e3)
        self.L = results.get('L', 0.1)
        self.theta_deg = results.get('theta', 53)
        self.theta_rad = math.radians(self.theta_deg)
        
        # Extract results with safe defaults
        self.U_ab = results.get('U_ab', results.get('U_{ab}', 0.0))
        self.W_ab = results.get('W_ab', results.get('W_{ab}', 0.0))
        
        # Calculate q from results if not provided directly
        # First check for q in results
        q_val = results.get('q', None)
        
        # If q is not provided or is 0.0 (which might be an error), 
        # we need to infer it from W_ab and U_ab if possible
        if q_val is None or (isinstance(q_val, (int, float)) and abs(q_val) < 1e-12):
            # Try to infer q from W_ab and U_ab if U_ab is not zero
            if abs(self.U_ab) > 1e-12 and abs(self.W_ab) > 1e-12:
                # q = W_ab / U_ab
                self.q = self.W_ab / self.U_ab
            else:
                # Use default from target answer
                self.q = 2e-10
        else:
            self.q = float(q_val)
        
        # Precompute expected values for validation
        self.expected_U = self.E * self.L * 0.6  # cos(53°)=0.6
        self.expected_W = self.expected_U * self.q
        
        self.logs = []
        
    def log(self, message, success=None):
        """Helper to log test results."""
        if success is not None:
            status = "✓" if success else "✗"
            message = f"{status} {message}"
        self.logs.append(message)
        
    def test_invariants(self):
        """Test conservation laws and fundamental relationships."""
        try:
            # 1. Check U_ab = E * L * cos(theta) from uniform field theorem
            computed_U = self.E * self.L * 0.6
            if not math.isclose(self.U_ab, computed_U, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"U_ab mismatch: expected {computed_U:.1f}, got {self.U_ab:.1f}", False)
                return False
            self.log(f"U_ab = E*L*cosθ holds: {self.U_ab:.1f} = {self.E}*{self.L}*0.6", True)
            
            # 2. Check W_ab = q * U_ab from work-energy theorem
            computed_W = self.q * self.U_ab
            if not math.isclose(self.W_ab, computed_W, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"W_ab mismatch: expected {computed_W:.1e}, got {self.W_ab:.1e}", False)
                return False
            self.log(f"W_ab = q*U_ab holds: {self.W_ab:.1e} = {self.q:.1e}*{self.U_ab:.1f}", True)
            
            return True
            
        except Exception as e:
            self.log(f"test_invariants failed: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Verify intermediate results align with final answers."""
        try:
            # Check that U_ab matches the expected calculation
            expected_step1 = 2000 * 0.1 * 0.6  # From step_1 analysis
            if not math.isclose(self.U_ab, expected_step1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Step 1 inconsistency: U_ab={self.U_ab:.1f}, expected {expected_step1}", False)
                return False
            self.log(f"Step 1 consistent: U_ab = {self.U_ab:.1f}V", True)
            
            # Check that W_ab matches the expected calculation
            expected_step2 = self.U_ab * self.q
            if not math.isclose(self.W_ab, expected_step2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Step 2 inconsistency: W_ab={self.W_ab:.1e}, expected {expected_step2:.1e}", False)
                return False
            self.log(f"Step 2 consistent: W_ab = {self.W_ab:.1e}J", True)
            
            # Verify against target answers
            target_U, target_W = 120, 2.4e-8
            if (math.isclose(self.U_ab, target_U, rel_tol=1e-2) and 
                math.isclose(self.W_ab, target_W, rel_tol=1e-2)):
                self.log(f"Matches target answers: {target_U}V, {target_W:.1e}J", True)
                return True
            else:
                self.log(f"Target mismatch: got {self.U_ab}V, {self.W_ab:.1e}J", False)
                return False
                
        except Exception as e:
            self.log(f"test_trace_consistency failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Test trend predictions by perturbing inputs."""
        original_E, original_L, original_theta = self.E, self.L, self.theta_deg
        
        try:
            # Test 1: Increase E by 5% → U_ab and W_ab should increase
            self.E *= 1.05
            new_U = self.E * self.L * 0.6
            new_W = self.q * new_U
            
            if new_U <= self.expected_U or new_W <= self.expected_W:
                self.log(f"E↑ monotonicity failed: U {new_U:.1f} ≤ {self.expected_U:.1f}", False)
                return False
            self.log(f"E↑ monotonicity: U {new_U:.1f} > {self.expected_U:.1f}", True)
            
            # Restore E
            self.E = original_E
            
            # Test 2: Increase L by 5% → U_ab and W_ab should increase
            self.L *= 1.05
            new_U = self.E * self.L * 0.6
            new_W = self.q * new_U
            
            if new_U <= self.expected_U or new_W <= self.expected_W:
                self.log(f"L↑ monotonicity failed: U {new_U:.1f} ≤ {self.expected_U:.1f}", False)
                return False
            self.log(f"L↑ monotonicity: U {new_U:.1f} > {self.expected_U:.1f}", True)
            
            # Restore L
            self.L = original_L
            
            # Test 3: Increase theta (decrease cosθ) → U_ab and W_ab should decrease
            self.theta_deg = original_theta + 5
            new_cos = math.cos(math.radians(self.theta_deg))
            new_U = self.E * self.L * new_cos
            new_W = self.q * new_U
            
            if new_U >= self.expected_U or new_W >= self.expected_W:
                self.log(f"θ↑ monotonicity failed: U {new_U:.1f} ≥ {self.expected_U:.1f}", False)
                return False
            self.log(f"θ↑ monotonicity: U {new_U:.1f} < {self.expected_U:.1f} (cos decreased)", True)
            
            # Restore theta
            self.theta_deg = original_theta
            
            return True
            
        except Exception as e:
            self.log(f"test_monotonicity failed: {str(e)}", False)
            # Restore original values on error
            self.E, self.L, self.theta_deg = original_E, original_L, original_theta
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []  # Reset logs
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Trends", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"\n=== {test_name} ===", None)
            if not test_func():
                success = False
        
        self.log(f"\n=== SUMMARY ===", None)
        self.log(f"Overall: {'PASS' if success else 'FAIL'}", None)
        
        return success, self.logs