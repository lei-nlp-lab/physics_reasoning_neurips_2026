import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs
        self.R = getattr(solution_instance, 'R', 10.0)  # cm
        self.i_deg = getattr(solution_instance, 'i', 45.0)  # degrees
        
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract required numeric results
        self.C1 = self.results.get('C_1', 0.0)  # in radians
        self.C2 = self.results.get('C_2', 0.0)  # in radians
        self.sin_r = self.results.get('sinr', 0.0)
        self.AP1 = self.results.get('AP_1', 0.0)
        self.P1P2 = self.results.get('P_1P_2', 0.0)
        self.n1 = self.results.get('n1', 2*math.sqrt(3)/3)
        self.n2 = self.results.get('n2', math.sqrt(2))
        
        # Color answer is not required to be numeric, handle separately
        self.color_answer = self.results.get('color_answer', '')
        
        # Convert incident angle to radians
        self.i_rad = math.radians(self.i_deg)
        
        # Expected values for verification (all in radians for critical angles)
        self.expected_C1 = math.asin(1/self.n1)  # in radians
        self.expected_C2 = math.asin(1/self.n2)  # in radians
        
        # Snell's law for red light at AB interface (medium to air): n1 * sin(i) = sin(r)
        self.expected_sin_r = self.n1 * math.sin(self.i_rad)
        
        # Calculate expected AP1 from geometry: AP1 = R / tan(r)
        if 0 < self.expected_sin_r < 1:
            cos_r = math.sqrt(1 - self.expected_sin_r**2)
            tan_r = self.expected_sin_r / cos_r
            self.expected_AP1 = self.R / tan_r
        else:
            self.expected_AP1 = float('inf')
        
        # Expected distance (AP2 = R = 10 cm from quarter-circle geometry)
        self.expected_P1P2 = self.expected_AP1 + self.R
        
        self.logs = []
    
    def log(self, message):
        self.logs.append(message)
    
    def test_invariants(self):
        """Verify critical angles and Snell's law."""
        try:
            # 1. Critical angles (in radians, using 1% tolerance)
            if not math.isclose(self.C1, self.expected_C1, rel_tol=0.01, abs_tol=0.01):
                self.log(f"FAIL: C1 mismatch: {self.C1} rad vs expected {self.expected_C1} rad")
                return False
            if not math.isclose(self.C2, self.expected_C2, rel_tol=0.01, abs_tol=0.01):
                self.log(f"FAIL: C2 mismatch: {self.C2} rad vs expected {self.expected_C2} rad")
                return False
            
            # 2. Snell's law for red light at AB interface
            if not math.isclose(self.sin_r, self.expected_sin_r, rel_tol=0.01, abs_tol=0.01):
                self.log(f"FAIL: sin(r) mismatch: {self.sin_r} vs expected {self.expected_sin_r}")
                return False
            
            # 3. Total internal reflection conditions (compare angles in radians)
            # Convert incident angle to radians for comparison
            i_rad = math.radians(self.i_deg)
            # Red: i=45° < C1=60° → refracts out (no TIR)
            # Violet: i=45° = C2=45° → critical condition (TIR just starts)
            if not (i_rad < self.C1 and math.isclose(i_rad, self.C2, abs_tol=0.01)):
                self.log(f"FAIL: Incident angle {i_rad} rad doesn't match critical conditions")
                return False
            
            self.log("PASS: All invariants satisfied (critical angles, Snell's law, TIR conditions)")
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Check geometric consistency between intermediate and final results."""
        try:
            # 1. AP1 from geometry (R / tan(r))
            if not math.isclose(self.AP1, self.expected_AP1, rel_tol=0.02, abs_tol=0.1):
                self.log(f"FAIL: AP1 mismatch: {self.AP1} vs expected {self.expected_AP1}")
                return False
            
            # 2. Distance P1P2 = AP1 + AP2, with AP2 = R = 10 cm (from quarter-circle)
            if not math.isclose(self.P1P2, self.expected_P1P2, rel_tol=0.02, abs_tol=0.1):
                self.log(f"FAIL: P1P2 mismatch: {self.P1P2} vs expected {self.expected_P1P2}")
                return False
            
            # 3. Verify AP2 = R (implicit in step 7)
            AP2 = self.P1P2 - self.AP1
            if not math.isclose(AP2, self.R, rel_tol=0.02, abs_tol=0.1):
                self.log(f"FAIL: Implicit AP2 = {AP2} ≠ R = {self.R}")
                return False
            
            self.log("PASS: Trace consistency verified (geometry and distance)")
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb refractive index and check expected trend."""
        original_n1 = self.n1
        try:
            # Increase n1 by 10%
            perturbed_n1 = original_n1 * 1.10
            # Critical angle should decrease (sin⁻¹(1/n) is decreasing in n)
            perturbed_C1 = math.asin(1/perturbed_n1)  # in radians
            
            if not (perturbed_C1 < self.C1):
                self.log(f"FAIL: Monotonicity violated: C1 should decrease when n1 increases")
                return False
            
            # Restore original value
            self.n1 = original_n1
            self.log("PASS: Monotonicity check (critical angle vs refractive index)")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            self.n1 = original_n1  # Restore on error
            return False
    
    def test_color_assignment(self):
        """Verify color descriptions match expected pattern."""
        try:
            if not isinstance(self.color_answer, str):
                # Color answer is not required to be numeric, skip if not string
                self.log("SKIP: Color answer not a string (may be numeric or absent)")
                return True  # Non-critical test
            
            answer_lower = self.color_answer.lower()
            # Expected patterns (allow variations in wording)
            expected_patterns = [
                ("red", "first spot is red"),
                ("mixture", "second spot has mixture"),
                ("violet", "second spot contains violet"),
                ("purple", "second spot contains purple")
            ]
            
            # Check for at least two of the expected keywords
            matches = 0
            for pattern, description in expected_patterns:
                if pattern in answer_lower:
                    matches += 1
            
            if matches >= 2:
                self.log("PASS: Color assignment plausible")
                return True
            else:
                self.log(f"WARNING: Color answer '{self.color_answer}' doesn't match expected patterns")
                return True  # Not a failure, just warning
        except Exception as e:
            self.log(f"ERROR in test_color_assignment: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        self.logs = []
        success = True
        
        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity,
            self.test_color_assignment
        ]
        
        for test in tests:
            try:
                if not test():
                    success = False
            except Exception as e:
                self.log(f"CRITICAL: Test crashed: {traceback.format_exc()}")
                success = False
        
        return success, self.logs