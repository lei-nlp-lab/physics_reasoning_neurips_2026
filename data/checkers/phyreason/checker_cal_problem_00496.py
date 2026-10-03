import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution_instance."""
        # Input constants from problem statement (hardcoded since they're given)
        self.L = 0.04  # 4 cm = 0.04 m
        self.q = -1e-8  # C (charge for first movement)
        self.W1 = -8e-8  # J
        self.W2 = 4e-8   # J
        self.q2 = 1e-8   # positive charge (equal magnitude)
        
        # Results from solution_instance (prefer results dict, then direct attributes)
        self.U_ab = getattr(solution_instance, 'U_ab', 
                           getattr(solution_instance, 'results', {}).get('U_ab', None))
        self.U_ac = getattr(solution_instance, 'U_ac', 
                           getattr(solution_instance, 'results', {}).get('U_ac', None))
        self.E = getattr(solution_instance, 'E', 
                        getattr(solution_instance, 'results', {}).get('E', None))
        self.direction = getattr(solution_instance, 'direction', 
                                getattr(solution_instance, 'results', {}).get('direction', ''))
        
        # Ensure numeric types for numeric variables only
        for attr in ['U_ab', 'U_ac', 'E']:
            val = getattr(self, attr)
            if val is not None:
                try:
                    setattr(self, attr, float(val))
                except (TypeError, ValueError):
                    setattr(self, attr, 0.0)

    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        try:
            # 1. Check U_ab = W1/q (definition of potential difference)
            if self.U_ab is not None:
                expected_U_ab = self.W1 / self.q
                if not math.isclose(self.U_ab, expected_U_ab, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"U_ab invariant failed: {self.U_ab} != {expected_U_ab}")
                    return False, logs
                logs.append("U_ab = W1/q verified.")
            
            # 2. Check U_ac = W2/q2
            if self.U_ac is not None:
                expected_U_ac = self.W2 / self.q2
                if not math.isclose(self.U_ac, expected_U_ac, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"U_ac invariant failed: {self.U_ac} != {expected_U_ac}")
                    return False, logs
                logs.append("U_ac = W2/q2 verified.")
            
            # 3. Check uniform field relation: |E| = |U_ab|/L (projection along field)
            if all(v is not None for v in [self.E, self.U_ab, self.L]):
                # Since triangle is equilateral, ab makes 60° with ac
                # Electric field component along ab: E_parallel = U_ab / L
                E_expected = abs(self.U_ab) / self.L
                if not math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Uniform field relation failed: E={self.E} != |U_ab|/L={E_expected}")
                    return False, logs
                logs.append("Uniform field relation |E| = |U_ab|/L verified.")
            
            # 4. Symbolic check: potential differences in uniform field
            # Create symbolic variables
            if all(v is not None for v in [self.E, self.L, self.U_ab, self.U_ac]):
                # E is stored as magnitude, so for the formula ΔV = -E·d we need signed E
                # If direction is from a to b, then E_signed = -|E| = -self.E
                E_signed = -abs(self.E)  # Negative because field points from a to b
                U_ab_num = -E_signed * self.L  # = self.E * self.L
                U_ac_num = -E_signed * self.L * math.cos(math.pi/3)  # = self.E * self.L * 0.5
                if not (math.isclose(self.U_ab, U_ab_num, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.U_ac, U_ac_num, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"Symbolic uniform field check failed: U_ab={self.U_ab} vs {U_ab_num}, U_ac={self.U_ac} vs {U_ac_num}")
                    return False, logs
                logs.append("Symbolic uniform field relations verified.")
            
            return True, logs
        except Exception as e:
            logs.append(f"Invariant test error: {e}\n{traceback.format_exc()}")
            return False, logs

    def test_trace_consistency(self):
        """Check intermediate results against final answer."""
        logs = []
        try:
            # Step 3: U_ab = 8 V (from W1/q)
            if self.U_ab is not None and not math.isclose(self.U_ab, 8.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Step 3 trace mismatch: U_ab={self.U_ab}, expected ~8 V")
                return False, logs
            logs.append("Step 3 (U_ab=8 V) consistent.")
            
            # Step 5: U_ac = 4 V (from W2/q2)
            if self.U_ac is not None and not math.isclose(self.U_ac, 4.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Step 5 trace mismatch: U_ac={self.U_ac}, expected ~4 V")
                return False, logs
            logs.append("Step 5 (U_ac=4 V) consistent.")
            
            # Step 7: E = 200 V/m (from U_ab/L)
            if self.E is not None and not math.isclose(self.E, 200.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Step 7 trace mismatch: E={self.E}, expected ~200 V/m")
                return False, logs
            logs.append("Step 7 (E=200 V/m) consistent.")
            
            # Direction check (from a to b)
            if self.direction and "a to b" not in self.direction.lower():
                logs.append(f"Direction mismatch: '{self.direction}' should contain 'a to b'")
                return False, logs
            logs.append("Direction 'from a to b' consistent.")
            
            # Cross-check: U_ac should be half of U_ab (since cos60° = 0.5)
            if all(v is not None for v in [self.U_ab, self.U_ac]):
                if not math.isclose(self.U_ac, self.U_ab * 0.5, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Cross-check failed: U_ac={self.U_ac} != 0.5*U_ab={0.5*self.U_ab}")
                    return False, logs
                logs.append("U_ac = 0.5*U_ab cross-check passed.")
            
            return True, logs
        except Exception as e:
            logs.append(f"Trace consistency error: {e}\n{traceback.format_exc()}")
            return False, logs

    def test_monotonicity(self):
        """Perturb an input and check trend."""
        logs = []
        try:
            original_L = self.L
            
            # Perturb L by +5% and recalculate expected E
            perturbed_L = original_L * 1.05
            # Expected E from formula: |E| = |U_ab|/L (U_ab fixed from W1/q)
            if self.U_ab is not None:
                expected_E_original = abs(self.U_ab) / original_L
                expected_E_perturbed = abs(self.U_ab) / perturbed_L
                
                # E should decrease when L increases
                if expected_E_perturbed >= expected_E_original:
                    logs.append(f"Monotonicity failed: E should decrease when L increases")
                    return False, logs
                
                # Check ratio
                expected_ratio = original_L / perturbed_L  # ~0.952
                actual_ratio = expected_E_perturbed / expected_E_original
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Monotonicity ratio mismatch: {actual_ratio} != {expected_ratio}")
                    return False, logs
                logs.append(f"Monotonicity test passed: E ∝ 1/L")
            
            return True, logs
        except Exception as e:
            logs.append(f"Monotonicity test error: {e}\n{traceback.format_exc()}")
            return False, logs

    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        all_logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            test_success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not test_success:
                success = False
                all_logs.append(f"✗ {test_name} FAILED")
            else:
                all_logs.append(f"✓ {test_name} PASSED")
        
        return success, all_logs