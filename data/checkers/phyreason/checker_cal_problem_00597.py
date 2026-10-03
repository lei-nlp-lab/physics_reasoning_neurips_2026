import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization with safe attribute access.
        Extracts inputs and results from solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe extraction of input parameters with fallbacks - use numeric defaults
        self.m = float(getattr(solution_instance, 'm', self.results.get('m', 1.0)))
        self.mu = float(getattr(solution_instance, 'mu', self.results.get('mu', 0.3)))
        self.theta = float(getattr(solution_instance, 'theta', self.results.get('theta', 0.5236)))  # 30° in radians
        self.g = float(getattr(solution_instance, 'g', self.results.get('g', 9.8)))
        
        # Safe extraction of target results
        self.F_coeff = self.results.get('F', None)  # Coefficient (dimensionless)
        self.tantheta_0 = self.results.get('tantheta_0', None)  # tan(critical angle)
        
        # Compute actual force magnitude for testing
        if self.F_coeff is not None:
            self.F_actual = self.F_coeff * self.m * self.g
        else:
            self.F_actual = None
    
    def _is_close(self, a, b):
        """
        Robust numerical comparison with 2% tolerance.
        """
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        try:
            a_val = float(a) if hasattr(a, '__float__') else a
            b_val = float(b) if hasattr(b, '__float__') else b
            if isinstance(a_val, float) and isinstance(b_val, float):
                return math.isclose(a_val, b_val, rel_tol=1e-2, abs_tol=1e-2)
        except (TypeError, ValueError):
            pass
        return False
    
    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Force equilibrium in both directions (using actual force)
        2. Critical angle condition tanθ₀ = μ
        """
        logs = []
        success = True
        
        try:
            # Check 1: Force equilibrium (using actual force if available)
            if self.F_actual is not None:
                # Compute normal force and friction
                N_calc = self.F_actual * math.cos(self.theta) + self.m * self.g
                f_calc = self.F_actual * math.sin(self.theta)
                
                # Equilibrium check: f = μN should hold
                right_eq = self.mu * N_calc
                
                if not self._is_close(f_calc, right_eq):
                    logs.append(f"FAIL: Equilibrium/friction law violated. f={f_calc:.3f}, μN={right_eq:.3f}")
                    success = False
                else:
                    logs.append("PASS: Force equilibrium and friction law satisfied")
            else:
                logs.append("SKIP: Force coefficient not available for equilibrium check")
            
            # Check 2: Critical angle condition
            if self.tantheta_0 is not None:
                if not self._is_close(self.tantheta_0, self.mu):
                    logs.append(f"FAIL: Critical angle condition violated. tanθ₀={self.tantheta_0}, μ={self.mu}")
                    success = False
                else:
                    logs.append("PASS: Critical angle condition tanθ₀ = μ satisfied")
            else:
                logs.append("SKIP: tanθ₀ not available for critical angle check")
            
            # Check 3: Motion condition (denominator positive for given theta)
            motion_cond = math.sin(self.theta) - self.mu * math.cos(self.theta)
            if motion_cond <= 0:
                logs.append(f"WARN: Motion condition may not be satisfied: sinθ - μcosθ = {motion_cond:.3f}")
            else:
                logs.append(f"NOTE: Motion condition satisfied: sinθ - μcosθ = {motion_cond:.3f} > 0")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        try:
            # Check if F_coeff matches the derived formula
            if self.F_coeff is not None:
                # Expected coefficient: μ/(sinθ - μcosθ)
                expected_coeff = self.mu / (math.sin(self.theta) - self.mu * math.cos(self.theta))
                
                if not self._is_close(self.F_coeff, expected_coeff):
                    logs.append(f"FAIL: Coefficient inconsistent. Got {self.F_coeff:.6f}, expected {expected_coeff:.6f}")
                    success = False
                else:
                    logs.append("PASS: Coefficient matches derived formula")
            
            # Check if tanθ₀ = μ
            if self.tantheta_0 is not None:
                if not self._is_close(self.tantheta_0, self.mu):
                    logs.append(f"FAIL: tanθ₀ inconsistent. Got {self.tantheta_0}, expected μ={self.mu}")
                    success = False
                else:
                    logs.append("PASS: tanθ₀ = μ consistent")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Test monotonic trends by perturbing inputs.
        """
        logs = []
        success = True
        
        try:
            # Store original values
            orig_m = self.m
            orig_mu = self.mu
            orig_theta = self.theta
            orig_g = self.g
            
            # Test 1: Increasing mass should increase actual force
            new_m = orig_m * 1.05  # +5%
            coeff_orig = orig_mu / (math.sin(orig_theta) - orig_mu * math.cos(orig_theta))
            F_orig = coeff_orig * orig_m * orig_g
            F_new = coeff_orig * new_m * orig_g
            
            if not (F_new > F_orig):
                logs.append(f"FAIL: Force does not increase with mass. F(m)={F_orig:.3f}, F(1.05m)={F_new:.3f}")
                success = False
            else:
                logs.append(f"PASS: Force increases with mass (monotonic)")
            
            # Test 2: Increasing μ should increase coefficient (when denominator remains positive)
            new_mu = orig_mu * 1.05  # +5%
            # Ensure denominator stays positive
            denom_orig = math.sin(orig_theta) - orig_mu * math.cos(orig_theta)
            denom_new = math.sin(orig_theta) - new_mu * math.cos(orig_theta)
            
            if denom_orig > 0 and denom_new > 0:
                coeff_new = new_mu / denom_new
                if not (coeff_new > coeff_orig):
                    logs.append(f"FAIL: Coefficient does not increase with μ. C(μ)={coeff_orig:.3f}, C(1.05μ)={coeff_new:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Coefficient increases with μ (monotonic)")
            else:
                logs.append(f"SKIP: μ perturbation invalidates motion condition")
            
            # Test 3: Increasing θ (above critical) should decrease coefficient
            # First find critical angle
            theta_crit = math.atan(orig_mu)
            if orig_theta > theta_crit:
                # Increase θ by 1% while staying above critical
                theta_increase = min(0.05, (math.pi/2 - orig_theta) * 0.5)  # Small increase
                new_theta = orig_theta + theta_increase
                coeff_new_theta = orig_mu / (math.sin(new_theta) - orig_mu * math.cos(new_theta))
                if not (coeff_new_theta < coeff_orig):
                    logs.append(f"FAIL: Coefficient does not decrease with θ. C(θ)={coeff_orig:.3f}, C(θ+Δ)={coeff_new_theta:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: Coefficient decreases with θ (monotonic)")
            else:
                logs.append(f"SKIP: θ ({orig_theta:.3f}) not above critical angle ({theta_crit:.3f})")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                for log in logs:
                    all_logs.append(f"  {log}")
                if not success:
                    overall_success = False
                    all_logs.append(f"  Result: FAILED")
                else:
                    all_logs.append(f"  Result: PASSED")
            except Exception as e:
                all_logs.append(f"  ERROR: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs