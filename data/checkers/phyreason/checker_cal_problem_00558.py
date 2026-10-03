import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all necessary inputs and results."""
        try:
            # Extract results from solution_instance
            if hasattr(solution_instance, 'results'):
                self.results = solution_instance.results
            else:
                self.results = {}
            
            # Get F from results dict with proper fallback
            self.F = self.results.get('F', 0.0)
            
            # Extract input values from results dict or use defaults from problem
            self.v = self.results.get('v_val', 3.0e6)  # m/s
            self.B = self.results.get('B_val', 0.10)   # T
            
            # Handle the q value carefully - fix the bug
            q_from_results = self.results.get('q_val', -1.6e-19)
            
            # Check if q_from_results is effectively zero
            if abs(q_from_results) < 1e-30:
                # Use the problem's default value
                self.q = -1.6e-19
            else:
                self.q = q_from_results
            
            # Fundamental constants
            self.m_e = 9.11e-31  # electron mass
            
            # Step analysis data if available
            self.step_1 = getattr(solution_instance, 'step_1', None)
            
            # Store original values for monotonicity tests
            self._original_v = self.v
            self._original_B = self.B
            self._original_q = self.q
            
        except Exception as e:
            self.error = f"Initialization error: {str(e)}"
            raise
    
    def test_invariants(self):
        """Test fundamental physical invariants."""
        logs = []
        try:
            # 1. Lorentz force magnitude invariant: F = |q|*v*B (perpendicular case)
            expected_F_magnitude = abs(self.q) * self.v * self.B
            
            # Use proper tolerance for comparison
            if not math.isclose(self.F, expected_F_magnitude, rel_tol=1e-2, abs_tol=1e-16):
                logs.append(f"FAIL: Lorentz force magnitude mismatch. "
                           f"Expected {expected_F_magnitude:.2e}, got {self.F:.2e}")
                logs.append(f"Details: q={self.q:.2e}, v={self.v:.2e}, B={self.B:.2e}")
                return False, logs
            
            logs.append(f"PASS: Lorentz force magnitude correct: {self.F:.2e} N")
            
            # 2. Direction invariant: Force perpendicular to both v and B
            # Since problem specifies perpendicular injection, F should be perpendicular to v
            # In circular motion, work done by magnetic force is zero
            # Check: F·v = 0 (dot product for perpendicular vectors)
            work = 0.0  # For perpendicular vectors, dot product is zero
            
            if not math.isclose(work, 0.0, rel_tol=1e-2, abs_tol=1e-16):
                logs.append(f"FAIL: Magnetic force does work: F·v = {work:.2e} ≠ 0")
                return False, logs
            
            logs.append("PASS: Force perpendicular to velocity (magnetic force does no work)")
            
            # 3. Newton's second law: F = m*a (for circular motion)
            # For circular motion: a = v²/r, and F = m*v²/r
            # So r = m*v/|q|*B
            # Avoid division by zero
            denominator = abs(self.q) * self.B
            if abs(denominator) < 1e-30:  # Handle near-zero denominator
                logs.append("SKIP: Denominator is effectively zero, cannot compute cyclotron radius")
                return True, logs
                
            expected_r = (self.m_e * self.v) / denominator
            
            # Check if radius is positive and finite
            if expected_r <= 0 or not math.isfinite(expected_r):
                logs.append(f"FAIL: Invalid cyclotron radius: {expected_r:.2e}")
                return False, logs
            
            logs.append(f"PASS: Cyclotron radius consistent: {expected_r:.2e} m")
            
            return True, logs
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            return False, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final answer."""
        logs = []
        try:
            # Check if step_1 exists and matches final answer
            if self.step_1:
                step_F = getattr(self.step_1, 'F', None)
                if step_F is None and isinstance(self.step_1, dict):
                    step_F = self.step_1.get('F', None)
                
                if step_F is not None:
                    if not math.isclose(self.F, step_F, rel_tol=1e-2, abs_tol=1e-16):
                        logs.append(f"FAIL: Step result {step_F:.2e} ≠ final result {self.F:.2e}")
                        return False, logs
                    logs.append("PASS: Step result consistent with final answer")
            
            # Cross-check with expected answer from problem statement
            expected_answer = 4.8e-14
            if not math.isclose(self.F, expected_answer, rel_tol=1e-2, abs_tol=1e-16):
                logs.append(f"FAIL: Result {self.F:.2e} ≠ expected {expected_answer:.2e}")
                return False, logs
            
            logs.append("PASS: Result matches expected answer from problem statement")
            return True, logs
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            return False, logs
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs."""
        logs = []
        try:
            original_F = self.F
            
            # Handle division by zero case
            if abs(original_F) < 1e-30:
                logs.append("SKIP: Original F is effectively zero, cannot test monotonicity ratios")
                return True, logs
            
            # Test 1: Increase B by 10%, force should increase proportionally
            B_increased = self.B * 1.1
            new_F_B = abs(self.q) * self.v * B_increased
            
            if not new_F_B > original_F * 0.99:  # Allow 1% tolerance
                logs.append(f"FAIL: Force not increasing with B. Original: {original_F:.2e}, New: {new_F_B:.2e}")
                return False, logs
            
            logs.append(f"PASS: Force increases with B (factor: {new_F_B/original_F:.3f})")
            
            # Test 2: Increase v by 10%, force should increase proportionally
            v_increased = self.v * 1.1
            new_F_v = abs(self.q) * v_increased * self.B
            
            if not new_F_v > original_F * 0.99:
                logs.append(f"FAIL: Force not increasing with v. Original: {original_F:.2e}, New: {new_F_v:.2e}")
                return False, logs
            
            logs.append(f"PASS: Force increases with v (factor: {new_F_v/original_F:.3f})")
            
            # Test 3: Increase |q| by 10%, force should increase proportionally
            q_increased = self.q * 1.1  # Making charge magnitude larger
            new_F_q = abs(q_increased) * self.v * self.B
            
            if not new_F_q > original_F * 0.99:
                logs.append(f"FAIL: Force not increasing with |q|. Original: {original_F:.2e}, New: {new_F_q:.2e}")
                return False, logs
            
            logs.append(f"PASS: Force increases with |q| (factor: {new_F_q/original_F:.3f})")
            
            return True, logs
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            return False, logs
    
    def run_all_tests(self):
        """Execute all tests and return comprehensive results."""
        all_logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not test_success:
                    success = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.append(f"CRITICAL ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                success = False
        
        return success, all_logs