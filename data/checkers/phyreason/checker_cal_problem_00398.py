import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs from results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # ALWAYS use the original problem constants, regardless of what's in results
        self.R = 0.1
        self.B = 0.5
        self.q = 1.6e-19
        self.v0 = 1.6e6
        self.m = 1.0e-26
        
        # Extract results from the dictionary
        self.t_m = results.get('t_m', None)
        self.y_A = results.get('y_A', None)
        self.r = results.get('r', None)
        self.alpha_m = results.get('alpha_m', None)
        self.T = results.get('T', None)
        self.y_D_prime = results.get('y_D_prime', results.get('y_D\'', None))
        
        # Store original inputs for restoration after perturbations
        self._original_m = self.m
        self._original_q = self.q
        self._original_B = self.B
        
    def _isclose(self, a, b):
        """Robust fuzzy comparison for physics quantities."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physics laws and derived formulas."""
        logs = []
        success = True
        
        try:
            # 1. Lorentz force = centripetal force
            r_from_formula = self.m * self.v0 / (self.q * self.B)
            if self.r is not None and not self._isclose(self.r, r_from_formula):
                logs.append(f"Radius invariant failed: r={self.r}, expected {r_from_formula}")
                success = False
            else:
                logs.append("✓ Lorentz-centripetal force invariant satisfied")
            
            # 2. Period formula T = 2πm/(qB)
            if self.T is not None:
                T_expected = 2 * math.pi * self.m / (self.q * self.B)
                if not self._isclose(self.T, T_expected):
                    logs.append(f"Period invariant failed: T={self.T}, expected {T_expected}")
                    success = False
                else:
                    logs.append("✓ Period formula invariant satisfied")
            
            # 3. Force direction: circular motion should be clockwise for positive charge
            if self.y_A is not None and self.y_A < 0:
                logs.append(f"Warning: y_A={self.y_A} negative - check geometry")
            
            # 4. Geometry consistency: y_A = R²/(2r)
            if self.y_A is not None and self.r is not None:
                y_A_expected = self.R**2 / (2 * self.r)
                if not self._isclose(self.y_A, y_A_expected):
                    logs.append(f"Geometry invariant failed: y_A={self.y_A}, expected {y_A_expected}")
                    success = False
                else:
                    logs.append("✓ Geometry invariant satisfied")
            
            # 5. Time-angle relationship: t_m = (α_m/2π) * T
            if all(x is not None for x in [self.t_m, self.alpha_m, self.T]):
                t_m_expected = (self.alpha_m / (2 * math.pi)) * self.T
                if not self._isclose(self.t_m, t_m_expected):
                    logs.append(f"Time-angle invariant failed: t_m={self.t_m}, expected {t_m_expected}")
                    success = False
                else:
                    logs.append("✓ Time-angle invariant satisfied")
            
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Step 1: r should be 0.2m
            if self.r is not None:
                if not self._isclose(self.r, 0.2):
                    logs.append(f"Step 1 inconsistency: r={self.r}, expected 0.2")
                    success = False
                else:
                    logs.append("✓ Step 1 (radius) consistent")
            
            # Step 4: y_A should be 0.025m
            if self.y_A is not None:
                if not self._isclose(self.y_A, 0.025):
                    logs.append(f"Step 4 inconsistency: y_A={self.y_A}, expected 0.025")
                    success = False
                else:
                    logs.append("✓ Step 4 (y_A) consistent")
            
            # Step 6: α_m should be π/6 ≈ 0.5236 rad
            if self.alpha_m is not None:
                expected_alpha = math.pi / 6
                if not self._isclose(self.alpha_m, expected_alpha):
                    logs.append(f"Step 6 inconsistency: α_m={self.alpha_m}, expected {expected_alpha}")
                    success = False
                else:
                    logs.append("✓ Step 6 (α_m) consistent")
            
            # Step 8-9: t_m should be approximately 6.5e-8 s
            # Compute expected t_m using the correct formula: t_m = πm/(6qB)
            expected_t_m = (math.pi * self.m) / (6 * self.q * self.B)
            if self.t_m is not None:
                if not self._isclose(self.t_m, expected_t_m):
                    logs.append(f"Step 9 inconsistency: t_m={self.t_m}, expected {expected_t_m}")
                    success = False
                else:
                    logs.append("✓ Step 9 (t_m) consistent")
            
            # Step 10: y_D' should be approximately -0.027 m
            if self.y_D_prime is not None:
                if not self._isclose(self.y_D_prime, -0.027):
                    logs.append(f"Step 10 inconsistency: y_D'={self.y_D_prime}, expected -0.027")
                    success = False
                else:
                    logs.append("✓ Step 10 (y_D') consistent")
            
            # Cross-check: y_D' = -r*(1 - cos(α_m))
            if all(x is not None for x in [self.y_D_prime, self.r, self.alpha_m]):
                y_D_calc = -self.r * (1 - math.cos(self.alpha_m))
                if not self._isclose(self.y_D_prime, y_D_calc):
                    logs.append(f"Cross-check failed: y_D'={self.y_D_prime}, calculated {y_D_calc}")
                    success = False
                else:
                    logs.append("✓ Cross-check (y_D' formula) consistent")
                    
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase mass → period should increase
            original_T = 2 * math.pi * self.m / (self.q * self.B)
            perturbed_m = self.m * 1.1  # +10%
            perturbed_T = 2 * math.pi * perturbed_m / (self.q * self.B)
            
            if perturbed_T <= original_T:
                logs.append(f"Monotonicity failed: T({perturbed_m})={perturbed_T} ≤ T({self.m})={original_T}")
                success = False
            else:
                logs.append(f"✓ Mass-T monotonicity: T increases with m")
            
            # Test 2: Increase charge magnitude → period should decrease
            perturbed_q = self.q * 1.1  # +10%
            perturbed_T_q = 2 * math.pi * self.m / (perturbed_q * self.B)
            
            if perturbed_T_q >= original_T:
                logs.append(f"Monotonicity failed: T with q+10%={perturbed_T_q} ≥ original T={original_T}")
                success = False
            else:
                logs.append(f"✓ Charge-T monotonicity: T decreases with |q|")
            
            # Test 3: Increase B → radius should decrease
            original_r = self.m * self.v0 / (self.q * self.B)
            perturbed_B = self.B * 1.1  # +10%
            perturbed_r = self.m * self.v0 / (self.q * perturbed_B)
            
            if perturbed_r >= original_r:
                logs.append(f"Monotonicity failed: r with B+10%={perturbed_r} ≥ original r={original_r}")
                success = False
            else:
                logs.append(f"✓ B-r monotonicity: r decreases with B")
            
            # Restore original values
            self.m = self._original_m
            self.q = self._original_q
            self.B = self._original_B
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False
            # Restore on error
            self.m = self._original_m
            self.q = self._original_q
            self.B = self._original_B
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return aggregated results."""
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                for log in logs:
                    all_logs.append(f"  {log}")
            except Exception as e:
                all_logs.append(f"  Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== Overall Result ===")
        all_logs.append(f"Physics audit {'PASSED' if overall_success else 'FAILED'}")
        
        return overall_success, all_logs