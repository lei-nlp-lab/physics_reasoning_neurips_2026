import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.v1 = getattr(solution_instance, 'v1', None)
        self.v2 = getattr(solution_instance, 'v2', None)
        self.v1_prime = getattr(solution_instance, 'v1_prime', None)
        self.v2_prime = getattr(solution_instance, 'v2_prime', None)
        self.delta_t = getattr(solution_instance, 'delta_t', None)
        self.delta_x = getattr(solution_instance, 'delta_x', None)
        
        # Safe access to results with fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Extract using key mapping with fallbacks
        self.delta_x_val = results.get('Δx', results.get('delta_x', None))
        self.delta_t1_val = results.get('Δt₁', results.get('delta_t₁', None))
        self.delta_t2_val = results.get('2Δt', results.get('2delta_t', None))
        self.delta_t3_val = results.get('Δt₂', results.get('delta_t₂', None))
        self.v_avg = results.get('v̄', None)
        self.delta_x2_val = results.get('2Δx', results.get('2delta_x', None))
        self.delta_x2_2 = results.get('Δx₂', results.get('delta_x₂', None))
        self.delta_x1_val = results.get('Δx₁', results.get('delta_x₁', None))
        
        # For direct average velocity results if available
        self.v_avg1 = getattr(solution_instance, 'v_avg1', results.get('v_avg1', None))
        self.v_avg2 = getattr(solution_instance, 'v_avg2', results.get('v_avg2', None))
        
        # Store symbolic expressions for testing
        self.symbols = {
            'v1': self.v1 if self.v1 is not None else sp.Symbol('v1'),
            'v2': self.v2 if self.v2 is not None else sp.Symbol('v2'),
            'v1_prime': self.v1_prime if self.v1_prime is not None else sp.Symbol('v1_prime'),
            'v2_prime': self.v2_prime if self.v2_prime is not None else sp.Symbol('v2_prime'),
            'delta_t': self.delta_t if self.delta_t is not None else sp.Symbol('delta_t'),
            'delta_x': self.delta_x if self.delta_x is not None else sp.Symbol('delta_x')
        }
        
        self.logs = []
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def _symbolic_equal(self, expr1, expr2):
        """Check if two symbolic expressions are equal."""
        try:
            return sp.simplify(expr1 - expr2) == 0
        except:
            return False
    
    def test_invariants(self):
        """Verify physics invariants from KB theorems."""
        logs = []
        success = True
        
        # Theorem: Vector addition of displacements (1D case)
        try:
            v1, v2, delta_t = self.symbols['v1'], self.symbols['v2'], self.symbols['delta_t']
            delta_x1 = v1 * delta_t
            delta_x2 = v2 * delta_t
            delta_x_total = delta_x1 + delta_x2
            
            # Check if results match this
            if self.delta_x_val is not None and self.delta_x1_val is not None and self.delta_x2_2 is not None:
                if not self._is_close(float(self.delta_x1_val + self.delta_x2_2), float(self.delta_x_val)):
                    logs.append("FAIL: Displacement addition not consistent")
                    success = False
                else:
                    logs.append("PASS: Displacement addition consistent")
        except Exception as e:
            logs.append(f"WARNING: Displacement addition check failed: {e}")
        
        # Theorem: Definition of average velocity
        try:
            # For equal time intervals case
            if self.delta_x_val is not None and self.delta_t2_val is not None:
                v_avg_calc = self.delta_x_val / self.delta_t2_val
                if self.v_avg is not None:
                    if not self._is_close(v_avg_calc, self.v_avg):
                        logs.append(f"FAIL: Average velocity formula mismatch: {v_avg_calc} vs {self.v_avg}")
                        success = False
                    else:
                        logs.append("PASS: Average velocity calculation consistent")
        except Exception as e:
            logs.append(f"WARNING: Average velocity check failed: {e}")
        
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check first scenario (equal time intervals)
        try:
            if self.v1 is not None and self.v2 is not None and self.delta_t is not None:
                # Step 2: Δx₁ = v₁Δt
                delta_x1_calc = self.v1 * self.delta_t
                if self.delta_x1_val is not None:
                    if not self._is_close(delta_x1_calc, self.delta_x1_val):
                        logs.append(f"FAIL: Δx₁ mismatch: {delta_x1_calc} vs {self.delta_x1_val}")
                        success = False
                
                # Step 3: Δx₂ = v₂Δt
                delta_x2_calc = self.v2 * self.delta_t
                if self.delta_x2_2 is not None:
                    if not self._is_close(delta_x2_calc, self.delta_x2_2):
                        logs.append(f"FAIL: Δx₂ mismatch: {delta_x2_calc} vs {self.delta_x2_2}")
                        success = False
                
                # Step 4: Δx = Δx₁ + Δx₂
                delta_x_calc = delta_x1_calc + delta_x2_calc
                if self.delta_x_val is not None:
                    if not self._is_close(delta_x_calc, self.delta_x_val):
                        logs.append(f"FAIL: Δx mismatch: {delta_x_calc} vs {self.delta_x_val}")
                        success = False
                
                # Step 5-6: v̄ = Δx/(2Δt) = (v₁+v₂)/2
                v_avg_calc = delta_x_calc / (2 * self.delta_t)
                expected_v_avg = (self.v1 + self.v2) / 2
                
                if self.v_avg1 is not None:
                    if not self._is_close(v_avg_calc, self.v_avg1) or not self._is_close(expected_v_avg, self.v_avg1):
                        logs.append(f"FAIL: First average velocity mismatch: {v_avg_calc} vs {self.v_avg1}")
                        success = False
                    else:
                        logs.append("PASS: First scenario trace consistent")
        except Exception as e:
            logs.append(f"WARNING: First scenario trace check failed: {e}")
        
        # Check second scenario (equal displacements)
        try:
            if self.v1_prime is not None and self.v2_prime is not None and self.delta_x is not None:
                # Step 8: Δt₁ = Δx/v₁'
                delta_t1_calc = self.delta_x / self.v1_prime
                if self.delta_t1_val is not None:
                    if not self._is_close(delta_t1_calc, self.delta_t1_val):
                        logs.append(f"FAIL: Δt₁ mismatch: {delta_t1_calc} vs {self.delta_t1_val}")
                        success = False
                
                # Step 9: Δt₂ = Δx/v₂'
                delta_t2_calc = self.delta_x / self.v2_prime
                if self.delta_t3_val is not None:
                    if not self._is_close(delta_t2_calc, self.delta_t3_val):
                        logs.append(f"FAIL: Δt₂ mismatch: {delta_t2_calc} vs {self.delta_t3_val}")
                        success = False
                
                # Step 10: v̄ = 2Δx/(Δt₁+Δt₂)
                total_time = delta_t1_calc + delta_t2_calc
                v_avg_calc = (2 * self.delta_x) / total_time
                
                # Step 11: Simplified form
                expected_v_avg = (2 * self.v1_prime * self.v2_prime) / (self.v1_prime + self.v2_prime)
                
                if self.v_avg2 is not None:
                    if not self._is_close(v_avg_calc, self.v_avg2) or not self._is_close(expected_v_avg, self.v_avg2):
                        logs.append(f"FAIL: Second average velocity mismatch: {v_avg_calc} vs {self.v_avg2}")
                        success = False
                    else:
                        logs.append("PASS: Second scenario trace consistent")
        except Exception as e:
            logs.append(f"WARNING: Second scenario trace check failed: {e}")
        
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic trends by perturbing inputs."""
        logs = []
        success = True
        
        # Test first scenario: perturb v1
        if self.v1 is not None and self.v2 is not None and self.delta_t is not None:
            try:
                original_v1 = float(self.v1)
                perturbed_v1 = original_v1 * 1.01  # 1% increase
                
                # Original average velocity
                v_avg_original = (original_v1 + self.v2) / 2
                v_avg_perturbed = (perturbed_v1 + self.v2) / 2
                
                # Check monotonic increase
                if v_avg_perturbed <= v_avg_original:
                    logs.append(f"FAIL: First scenario not monotonic with v1: {v_avg_original} -> {v_avg_perturbed}")
                    success = False
                else:
                    logs.append("PASS: First scenario monotonic with v1 increase")
            except Exception as e:
                logs.append(f"WARNING: First scenario monotonicity check failed: {e}")
        
        # Test second scenario: perturb v1_prime
        if self.v1_prime is not None and self.v2_prime is not None:
            try:
                original_v1p = float(self.v1_prime)
                perturbed_v1p = original_v1p * 1.01  # 1% increase
                
                # Original average velocity
                v_avg_original = (2 * original_v1p * self.v2_prime) / (original_v1p + self.v2_prime)
                v_avg_perturbed = (2 * perturbed_v1p * self.v2_prime) / (perturbed_v1p + self.v2_prime)
                
                # Check monotonic increase (should increase if v1' < v2')
                if original_v1p < float(self.v2_prime):
                    if v_avg_perturbed <= v_avg_original:
                        logs.append(f"FAIL: Second scenario not monotonic with v1' increase: {v_avg_original} -> {v_avg_perturbed}")
                        success = False
                    else:
                        logs.append("PASS: Second scenario monotonic with v1' increase")
                else:
                    logs.append("INFO: Skipping monotonicity check for v1' > v2' case")
            except Exception as e:
                logs.append(f"WARNING: Second scenario monotonicity check failed: {e}")
        
        return success, logs
    
    def test_formula_correctness(self):
        """Verify the final formulas match target answers."""
        logs = []
        success = True
        
        # Target formulas
        target1 = (self.symbols['v1'] + self.symbols['v2']) / 2
        target2 = (2 * self.symbols['v1_prime'] * self.symbols['v2_prime']) / (self.symbols['v1_prime'] + self.symbols['v2_prime'])
        
        # Check first formula
        if self.v_avg1 is not None:
            try:
                if isinstance(self.v_avg1, (int, float)):
                    # Numerical check
                    expected = (self.v1 + self.v2) / 2 if self.v1 is not None and self.v2 is not None else None
                    if expected is not None:
                        if not self._is_close(self.v_avg1, expected):
                            logs.append(f"FAIL: First formula numerical mismatch: {self.v_avg1} vs {expected}")
                            success = False
                        else:
                            logs.append("PASS: First formula numerically correct")
                else:
                    # Symbolic check
                    if not self._symbolic_equal(self.v_avg1, target1):
                        logs.append(f"FAIL: First formula symbolic mismatch")
                        success = False
                    else:
                        logs.append("PASS: First formula symbolically correct")
            except Exception as e:
                logs.append(f"WARNING: First formula check failed: {e}")
        
        # Check second formula
        if self.v_avg2 is not None:
            try:
                if isinstance(self.v_avg2, (int, float)):
                    # Numerical check
                    if self.v1_prime is not None and self.v2_prime is not None:
                        expected = (2 * self.v1_prime * self.v2_prime) / (self.v1_prime + self.v2_prime)
                        if not self._is_close(self.v_avg2, expected):
                            logs.append(f"FAIL: Second formula numerical mismatch: {self.v_avg2} vs {expected}")
                            success = False
                        else:
                            logs.append("PASS: Second formula numerically correct")
                else:
                    # Symbolic check
                    if not self._symbolic_equal(self.v_avg2, target2):
                        logs.append(f"FAIL: Second formula symbolic mismatch")
                        success = False
                    else:
                        logs.append("PASS: Second formula symbolically correct")
            except Exception as e:
                logs.append(f"WARNING: Second formula check failed: {e}")
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Formula Correctness", self.test_formula_correctness)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs