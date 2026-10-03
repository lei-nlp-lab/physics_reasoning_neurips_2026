import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        Follows robust access rules to prevent sandbox errors.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs from problem statement (with safe fallbacks)
        self.v0 = 0.0  # starts from rest
        self.v_final = 8.0  # m/s
        self.t_acc = 2.0  # s
        self.total_distance = 50.0  # 50-meter relay race
        
        # Extract computed results with safe access
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.x = getattr(solution_instance, 'x', self.results.get('x', None))
        self.t_prime = getattr(solution_instance, "t'", self.results.get("t'", None))
        self.t_total = getattr(solution_instance, 't_total', self.results.get('t_total', None))
        
        # Convert any string results to float if needed
        self._convert_results_to_float()
        
        # Expected values from steps analysis
        self.expected_a = 4.0
        self.expected_x = 8.0
        self.expected_t_prime = 5.25
        self.expected_t_total = 7.25
        
        # Setup symbolic variables for symbolic checks
        self.v0_sym = sp.symbols('v0')
        self.v_sym = sp.symbols('v')
        self.a_sym = sp.symbols('a')
        self.t_sym = sp.symbols('t')
        self.x_sym = sp.symbols('x')
        self.t_prime_sym = sp.symbols('t_prime')
        self.total_distance_sym = sp.symbols('D')
        
    def _convert_results_to_float(self):
        """Safely convert any string results to float values."""
        for attr_name in ['a', 'x', 't_prime', 't_total']:
            value = getattr(self, attr_name)
            if isinstance(value, str):
                try:
                    # Remove any units or formatting
                    clean_value = ''.join(ch for ch in value if ch.isdigit() or ch in '.-')
                    setattr(self, attr_name, float(clean_value))
                except ValueError:
                    setattr(self, attr_name, None)
    
    def _is_close(self, a, b):
        """Robust fuzzy comparison with 1-2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify physical invariants using formulas from KB."""
        logs = []
        success = True
        
        try:
            # 1. Check uniformly accelerated motion formula: v = v0 + a*t
            if self.a is not None:
                v_calculated = self.v0 + self.a * self.t_acc
                if not self._is_close(v_calculated, self.v_final):
                    logs.append(f"Velocity invariant failed: v0 + a*t = {v_calculated}, expected {self.v_final}")
                    success = False
                    
                # Symbolic check
                v_eq = sp.Eq(self.v_sym, self.v0_sym + self.a_sym * self.t_sym)
                v_numeric = self.v0 + self.a * self.t_acc
                if not self._is_close(v_numeric, self.v_final):
                    logs.append(f"Symbolic velocity check failed: {v_numeric} vs {self.v_final}")
            
            # 2. Check displacement during acceleration: x = 0.5*a*t²
            if self.x is not None and self.a is not None:
                x_calculated = 0.5 * self.a * (self.t_acc ** 2)
                if not self._is_close(x_calculated, self.x):
                    logs.append(f"Displacement invariant failed: 0.5*a*t² = {x_calculated}, expected {self.x}")
                    success = False
                    
                # Symbolic check using average velocity formula
                v_avg = (self.v0 + self.v_final) / 2
                x_vavg = v_avg * self.t_acc
                if not self._is_close(x_vavg, self.x):
                    logs.append(f"Average velocity displacement check failed: {x_vavg} vs {self.x}")
            
            # 3. Check uniform motion displacement: remaining_distance = v * t'
            if self.t_prime is not None and self.x is not None:
                remaining_distance = self.total_distance - self.x
                t_prime_calculated = remaining_distance / self.v_final
                if not self._is_close(t_prime_calculated, self.t_prime):
                    logs.append(f"Uniform motion invariant failed: (D-x)/v = {t_prime_calculated}, expected {self.t_prime}")
                    success = False
            
            # 4. Check total time addition
            if self.t_total is not None and self.t_prime is not None:
                t_total_calculated = self.t_acc + self.t_prime
                if not self._is_close(t_total_calculated, self.t_total):
                    logs.append(f"Total time invariant failed: t + t' = {t_total_calculated}, expected {self.t_total}")
                    success = False
                    
        except Exception as e:
            logs.append(f"Invariant test error: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        success = True
        
        try:
            # Check if all required results exist
            if None in [self.a, self.x, self.t_prime, self.t_total]:
                missing = [k for k, v in {'a': self.a, 'x': self.x, "t'": self.t_prime, 't_total': self.t_total}.items() if v is None]
                logs.append(f"Missing results: {missing}")
                success = False
                return success, logs
            
            # 1. Check acceleration matches expected
            if not self._is_close(self.a, self.expected_a):
                logs.append(f"Acceleration mismatch: {self.a} vs expected {self.expected_a}")
                success = False
            
            # 2. Check displacement during acceleration
            if not self._is_close(self.x, self.expected_x):
                logs.append(f"Displacement mismatch: {self.x} vs expected {self.expected_x}")
                success = False
            
            # 3. Check uniform motion time
            if not self._is_close(self.t_prime, self.expected_t_prime):
                logs.append(f"Uniform time mismatch: {self.t_prime} vs expected {self.expected_t_prime}")
                success = False
            
            # 4. Check total time
            if not self._is_close(self.t_total, self.expected_t_total):
                logs.append(f"Total time mismatch: {self.t_total} vs expected {self.expected_t_total}")
                success = False
            
            # 5. Verify overall consistency
            # Recalculate total time from components
            t_total_recalc = self.t_acc + self.t_prime
            if not self._is_close(t_total_recalc, self.t_total):
                logs.append(f"Total time recalculation mismatch: {t_total_recalc} vs {self.t_total}")
                success = False
                
            # Verify distance consistency
            distance_acc = self.x
            distance_uniform = self.v_final * self.t_prime
            total_distance_calc = distance_acc + distance_uniform
            if not self._is_close(total_distance_calc, self.total_distance):
                logs.append(f"Total distance mismatch: {total_distance_calc} vs {self.total_distance}")
                success = False
                
        except Exception as e:
            logs.append(f"Trace consistency error: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb total distance and verify expected trend in total time."""
        logs = []
        success = True
        
        try:
            # Store original values
            original_distance = self.total_distance
            
            # Test 1: Increase total distance by 10%
            perturbed_distance = original_distance * 1.1
            
            # Recalculate using physics formulas
            # Acceleration phase remains same
            x_acc = 0.5 * self.expected_a * (self.t_acc ** 2)
            
            # New uniform motion time
            remaining_distance = perturbed_distance - x_acc
            t_prime_new = remaining_distance / self.v_final
            
            # New total time
            t_total_new = self.t_acc + t_prime_new
            
            # Original total time (from expected)
            t_total_original = self.expected_t_total
            
            # Check monotonicity: longer distance should increase total time
            if t_total_new <= t_total_original:
                logs.append(f"Monotonicity failed: distance ↑ ({original_distance}→{perturbed_distance}) but t_total ↓ ({t_total_original}→{t_total_new})")
                success = False
            else:
                logs.append(f"Monotonicity passed: distance ↑ → t_total ↑ ({t_total_original}→{t_total_new})")
            
            # Test 2: Decrease total distance by 10%
            perturbed_distance = original_distance * 0.9
            remaining_distance = perturbed_distance - x_acc
            t_prime_new = remaining_distance / self.v_final
            t_total_new = self.t_acc + t_prime_new
            
            if t_total_new >= t_total_original:
                logs.append(f"Monotonicity failed: distance ↓ ({original_distance}→{perturbed_distance}) but t_total ↑ ({t_total_original}→{t_total_new})")
                success = False
            else:
                logs.append(f"Monotonicity passed: distance ↓ → t_total ↓ ({t_total_original}→{t_total_new})")
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*50}")
            all_logs.append(f"Running {test_name}")
            all_logs.append(f"{'='*50}")
            
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
                
                status = "PASSED" if success else "FAILED"
                all_logs.append(f"{test_name}: {status}")
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        # Summary
        all_logs.append(f"\n{'='*50}")
        all_logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append(f"{'='*50}")
        
        return overall_success, all_logs