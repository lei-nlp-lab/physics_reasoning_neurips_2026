import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        self.solution = solution_instance
        
        # Extract from results dict if present, else from attributes, else default
        self.results = getattr(solution_instance, 'results', {})
        self.T_max = float(self.results.get('T_max', 0.0))
        self.delta_x = float(self.results.get('delta_x', 0.0))
        self.x = float(self.results.get('x', 0.0))
        
        # Handle direction: could be string or numeric code
        direction_val = self.results.get('direction', '')
        if isinstance(direction_val, (int, float)):
            # Convert numeric code to string: -1 -> "left", 1 -> "right"
            self.direction = "Propagating to the left" if float(direction_val) < 0 else "Propagating to the right"
        else:
            self.direction = str(direction_val)
        
        # Input parameters (with safe defaults)
        self.time = float(getattr(solution_instance, 'time', 0.2))  # Δt = 0.2s
        self.wavelength = float(getattr(solution_instance, 'wavelength', 4.0))  # λ = 4m
        
        self.logs = []

    def _log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify wave physics invariants from KB."""
        success = True
        
        # 1. Check distance-time-speed relationship: Δx = v * Δt
        try:
            # From steps: Δx = 7m, Δt = 0.2s → v = 35 m/s
            v_calculated = self.delta_x / self.time if self.time != 0 else 0
            
            if not self._is_close(v_calculated, 35.0):
                self._log(f"Wave speed invariant: v_calc={v_calculated}, expected~35 m/s", False)
                success = False
            else:
                self._log(f"Wave speed consistent: v={v_calculated:.2f} m/s", True)
        except Exception as e:
            self._log(f"Wave speed check failed: {e}", False)
            success = False
        
        # 2. Check waveform distance condition: x = nλ + 3
        try:
            # For n=1: x = 4*1 + 3 = 7, which matches Δx from step4
            x_n1 = 4*1 + 3  # n=1 case
            
            if not self._is_close(self.delta_x, x_n1):
                self._log(f"Distance form mismatch: Δx={self.delta_x}, 4n+3 with n=1 gives {x_n1}", False)
                success = False
            else:
                self._log(f"Distance matches 4n+3 form (n=1): {self.delta_x} m", True)
                
            # Check that direction 'left' is consistent with distance > λ
            if 'left' in self.direction.lower():
                if self.delta_x <= self.wavelength:
                    self._log(f"Left propagation but Δx={self.delta_x} ≤ λ={self.wavelength}", False)
                    success = False
                else:
                    self._log("Left propagation consistent with Δx > λ", True)
        except Exception as e:
            self._log(f"Waveform distance check failed: {e}", False)
            success = False
        
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Check T_max from step3: T_max = 0.2 / (1/4) = 0.8 s
        try:
            T_max_calculated = 0.2 / (1/4)
            if not self._is_close(self.T_max, T_max_calculated):
                self._log(f"T_max mismatch: {self.T_max} vs {T_max_calculated}", False)
                success = False
            else:
                self._log(f"T_max consistent with step3: {self.T_max} s", True)
        except Exception as e:
            self._log(f"T_max consistency check failed: {e}", False)
            success = False
        
        # Check Δx from step4: Δx = 7 m = 1.75λ
        try:
            lambda_multiple = self.delta_x / self.wavelength
            if not self._is_close(lambda_multiple, 1.75):
                self._log(f"Δx not 1.75λ: Δx/λ={lambda_multiple}", False)
                success = False
            else:
                self._log(f"Δx = {self.delta_x} m = 1.75λ (λ={self.wavelength} m)", True)
        except Exception as e:
            self._log(f"Δx consistency check failed: {e}", False)
            success = False
        
        # Check direction matches step5
        if self.direction:
            dir_str = self.direction.lower()
            if 'left' not in dir_str:
                self._log(f"Direction mismatch: got '{self.direction}', expected 'left'", False)
                success = False
            else:
                self._log("Direction consistent with step5: left", True)
        
        return success

    def test_monotonicity(self):
        """Perturb input time and verify period trends."""
        original_time = self.time
        success = True
        
        try:
            # For right propagation: T = Δt / (k + 1/4)
            # Maximum T occurs at k=0 → T_max = 4*Δt
            # So T_max should scale linearly with Δt
            
            # Test perturbation: increase Δt by 10%
            perturbed_time = original_time * 1.1
            expected_T_max_ratio = perturbed_time / original_time
            
            # Calculate new T_max using the formula
            new_T_max_calc = 4 * perturbed_time  # From T_max = 4*Δt
            
            # Check trend: increasing Δt should increase T_max proportionally
            if not self._is_close(new_T_max_calc / self.T_max, expected_T_max_ratio):
                self._log(f"Monotonicity failed: T_max doesn't scale linearly with Δt", False)
                success = False
            else:
                self._log(f"Monotonicity: T_max ∝ Δt (increased Δt by 10% → T_max increased by {expected_T_max_ratio:.1f}x)", True)
                
        except Exception as e:
            self._log(f"Monotonicity test failed: {e}", False)
            success = False
        
        return success

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self._log(f"\n=== Testing {name} ===")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{name}' crashed: {e}\n{traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs