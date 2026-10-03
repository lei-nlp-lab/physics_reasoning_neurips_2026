import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization using safe attribute access.
        """
        # Extract inputs directly from solution_instance attributes
        # Use getattr with sensible defaults
        self.m = getattr(solution_instance, 'm', 9.10938356e-31)
        self.e = getattr(solution_instance, 'e', -1.602176634e-19)
        self.U = getattr(solution_instance, 'U', 100.0)
        self.L = getattr(solution_instance, 'L', 0.1)
        self.d = getattr(solution_instance, 'd', 0.01)
        self.v0 = getattr(solution_instance, 'v0', 1e6)
        
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.t = self.results.get('t', 0.0)
        self.F = self.results.get('F', 0.0)
        self.a = self.results.get('a', 0.0)
        self.delta_y = self.results.get('delta_y', 0.0)
        
        self.logs = []
    
    def _log(self, msg, success=True):
        """Helper to log test results."""
        self.logs.append((msg, success))
    
    def _is_close(self, a, b):
        """Robust fuzzy numerical comparison with adaptive tolerance."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # Handle very small numbers common in physics
            max_abs = max(abs(a), abs(b))
            if max_abs < 1e-10:
                # For very small values, use very small absolute tolerance
                return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-20)
            elif max_abs < 1:
                # For small values
                return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
            else:
                # For larger values
                return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-9)
        return False
    
    def _safe_divide(self, numerator, denominator, default=0.0):
        """Safe division with zero check."""
        if denominator == 0:
            return default
        return numerator / denominator
    
    def test_invariants(self):
        """
        Test fundamental physics invariants:
        1. Newton's second law: F = m*a
        2. Electric force magnitude: |F| = |e| * (U/d)
        3. Time of flight: t = L/v0
        4. Kinematic relation: Δy = 0.5*a*t²
        5. Energy/work consistency: e*U*Δy/d = 0.5*m*(a*t)²
        """
        try:
            # 1. Newton's second law (F = m*a)
            F_from_ma = self.m * self.a
            if self._is_close(abs(self.F), abs(F_from_ma)):
                self._log("✓ Newton's second law: F = m*a holds")
            else:
                self._log(f"✗ Newton's second law violated: F={self.F}, m*a={F_from_ma}", False)
            
            # 2. Electric force magnitude (|F| = |e|*(U/d))
            if self.d == 0:
                self._log("✗ Electric force magnitude: division by zero (d=0)", False)
            else:
                F_from_field = abs(self.e) * self.U / self.d
                if self._is_close(abs(self.F), F_from_field):
                    self._log("✓ Electric force magnitude: |F| = |e|*U/d holds")
                else:
                    self._log(f"✗ Electric force magnitude violated: |F|={abs(self.F)}, |e|*U/d={F_from_field}", False)
            
            # 3. Time of flight (t = L/v0)
            if self.v0 == 0:
                self._log("✗ Time of flight: division by zero (v0=0)", False)
            else:
                t_calculated = self.L / self.v0
                if self._is_close(self.t, t_calculated):
                    self._log("✓ Time of flight: t = L/v0 holds")
                else:
                    self._log(f"✗ Time of flight violated: t={self.t}, L/v0={t_calculated}", False)
            
            # 4. Kinematic relation (Δy = 0.5*a*t²)
            delta_y_calculated = 0.5 * self.a * self.t**2
            if self._is_close(self.delta_y, delta_y_calculated):
                self._log("✓ Kinematic relation: Δy = 0.5*a*t² holds")
            else:
                self._log(f"✗ Kinematic relation violated: Δy={self.delta_y}, 0.5*a*t²={delta_y_calculated}", False)
            
            # 5. Energy/work consistency (work by electric field = change in y-kinetic energy)
            work = abs(self.F) * self.delta_y
            delta_ke = 0.5 * self.m * (self.a * self.t)**2
            if self._is_close(work, delta_ke):
                self._log("✓ Energy/work consistency: |F|*Δy = 0.5*m*(a*t)² holds")
            else:
                self._log(f"✗ Energy/work consistency violated: |F|*Δy={work}, 0.5*m*(a*t)²={delta_ke}", False)
                
        except Exception as e:
            self._log(f"✗ Invariant test failed with error: {str(e)}", False)
    
    def test_trace_consistency(self):
        """
        Verify that intermediate results are consistent with final answer.
        Steps from KB:
        1. F = eU/d (magnitude)
        2. F = m*a
        3. a = eU/(m*d)
        4. t = L/v0
        5. Δy = 0.5*a*t²
        6. Δy = eUL²/(2*m*d*v0²)
        """
        try:
            # Check step 3: a = eU/(m*d) (using magnitude of e)
            if self.m == 0 or self.d == 0:
                self._log("✗ Step consistency (a): division by zero (m=0 or d=0)", False)
            else:
                a_calculated = abs(self.e) * self.U / (self.m * self.d)
                if self._is_close(self.a, a_calculated):
                    self._log("✓ Step consistency: a = |e|*U/(m*d) holds")
                else:
                    self._log(f"✗ Step consistency (a) violated: a={self.a}, |e|*U/(m*d)={a_calculated}", False)
            
            # Check step 6: Δy = eUL²/(2*m*d*v0²) (using magnitude of e)
            if self.m == 0 or self.d == 0 or self.v0 == 0:
                self._log("✗ Step consistency (Δy): division by zero (m=0, d=0, or v0=0)", False)
            else:
                delta_y_final = abs(self.e) * self.U * self.L**2 / (2 * self.m * self.d * self.v0**2)
                if self._is_close(self.delta_y, delta_y_final):
                    self._log("✓ Step consistency: Δy = |e|*U*L²/(2*m*d*v0²) holds")
                else:
                    self._log(f"✗ Step consistency (Δy) violated: Δy={self.delta_y}, formula={delta_y_final}", False)
            
            # Cross-check using chain of steps
            if self.v0 == 0:
                self._log("✗ Trace chain consistency: division by zero (v0=0)", False)
            else:
                t_check = self.L / self.v0
                if self.m == 0 or self.d == 0:
                    self._log("✗ Trace chain consistency: division by zero (m=0 or d=0)", False)
                else:
                    a_check = abs(self.e) * self.U / (self.m * self.d)
                    delta_y_chain = 0.5 * a_check * t_check**2
                    if self._is_close(self.delta_y, delta_y_chain):
                        self._log("✓ Trace chain consistency: Δy from a and t matches")
                    else:
                        self._log(f"✗ Trace chain consistency violated: Δy={self.delta_y}, chain={delta_y_chain}", False)
                
        except Exception as e:
            self._log(f"✗ Trace consistency test failed with error: {str(e)}", False)
    
    def test_monotonicity(self):
        """
        Perturb input parameters and check if Δy changes in expected direction.
        Monotonicity rules from KB:
        - Δy ∝ U (increase U → increase Δy)
        - Δy ∝ 1/m (increase m → decrease Δy)
        - Δy ∝ L² (increase L → increase Δy)
        - Δy ∝ 1/d (increase d → decrease Δy)
        - Δy ∝ 1/v0² (increase v0 → decrease Δy)
        """
        original_params = {
            'U': self.U,
            'm': self.m,
            'L': self.L,
            'd': self.d,
            'v0': self.v0
        }
        
        perturbations = [
            ('U', 1.1, 'increase', True),
            ('U', 0.9, 'decrease', False),
            ('m', 1.1, 'increase', False),
            ('m', 0.9, 'decrease', True),
            ('L', 1.1, 'increase', True),
            ('L', 0.9, 'decrease', False),
            ('d', 1.1, 'increase', False),
            ('d', 0.9, 'decrease', True),
            ('v0', 1.1, 'increase', False),
            ('v0', 0.9, 'decrease', True)
        ]
        
        original_delta_y = self.delta_y
        
        try:
            # Check if we can compute delta_y formula
            if self.m == 0 or self.d == 0 or self.v0 == 0:
                self._log("✗ Monotonicity test: cannot compute Δy formula due to division by zero", False)
                return
                
            for param, factor, direction, expected_increase in perturbations:
                # Store original value
                original_value = getattr(self, param)
                
                # Apply perturbation
                setattr(self, param, original_value * factor)
                
                # Recalculate Δy using the formula
                new_delta_y = abs(self.e) * self.U * self.L**2 / (2 * self.m * self.d * self.v0**2)
                
                # Check monotonicity
                if expected_increase:
                    if new_delta_y > original_delta_y and not self._is_close(new_delta_y, original_delta_y):
                        self._log(f"✓ Monotonicity: {direction} {param} gives expected Δy change")
                    else:
                        self._log(f"✗ Monotonicity failed: {direction} {param} should increase Δy, but Δy didn't increase sufficiently", False)
                else:
                    if new_delta_y < original_delta_y and not self._is_close(new_delta_y, original_delta_y):
                        self._log(f"✓ Monotonicity: {direction} {param} gives expected Δy change")
                    else:
                        self._log(f"✗ Monotonicity failed: {direction} {param} should decrease Δy, but Δy didn't decrease sufficiently", False)
                
                # Restore original value
                setattr(self, param, original_value)
                
        except Exception as e:
            self._log(f"✗ Monotonicity test failed with error: {str(e)}", False)
            
            # Restore all original parameters
            for param, value in original_params.items():
                setattr(self, param, value)
    
    def run_all_tests(self):
        """
        Run all physics checks and return success status and logs.
        """
        self.logs = []
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success
        all_success = all(success for _, success in self.logs)
        
        # Add summary
        success_count = sum(1 for _, success in self.logs if success)
        total_count = len(self.logs)
        summary = f"Physics Checker Summary: {success_count}/{total_count} tests passed"
        self._log(summary, all_success)
        
        return all_success, self.logs