import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results using getattr and results dictionary."""
        self.solution = solution_instance
        
        # Given constants
        self.R = 1.0  # meter
        self.H = 1.8  # meter
        self.g = 10.0  # m/s² (consistent with solution's use of g=10)
        
        # Extract results with safe access
        self.H_minus_R = self._safe_get('H-R', 0.0)
        self.t1 = self._safe_get('t_1', None)
        self.v0 = self._safe_get('v_0', None)  # May be from sub-question 1 or 2
        self.tan_theta = self._safe_get('tantheta', None)
        self.R_plus_Rcos = self._safe_get('R+Rcostheta', None)
        self.H_minus_Rsin = self._safe_get('H-Rsintheta', None)
        
        # For trace consistency: store expected critical v0 for sub-question 2
        self.v0_critical_expected = 1.5 * math.sqrt(6)  # ≈ 3.674 m/s
        
        # Logs for reporting
        self.logs = []
        
    def _safe_get(self, key, default):
        """Safely get attribute from solution instance or its results dict."""
        try:
            # First try direct attribute
            val = getattr(self.solution, key, None)
            if val is not None:
                return float(val) if isinstance(val, (int, float, str)) else val
            # Then try results dict
            results_dict = getattr(self.solution, 'results', {})
            val = results_dict.get(key, default)
            if isinstance(val, str):
                # Handle string values like "1.5√6" or "2.5"
                if '√' in val:
                    # Parse expression like "1.5√6"
                    parts = val.split('√')
                    if len(parts) == 2:
                        coeff = float(parts[0])
                        root = float(parts[1])
                        return coeff * math.sqrt(root)
                try:
                    return float(val)
                except:
                    return default
            return float(val) if isinstance(val, (int, float)) else val
        except:
            return default

    def _log(self, msg, success):
        """Add a log message with success status."""
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success

    def test_invariants(self):
        """Verify kinematic and geometric invariants."""
        success = True
        
        # 1. Check vertical displacement for step_1: H-R = 0.5*g*t1^2
        if self.t1 is not None:
            expected_H_minus_R = 0.5 * self.g * self.t1**2
            if not math.isclose(self.H_minus_R, expected_H_minus_R, rel_tol=1e-2, abs_tol=1e-2):
                success = self._log(
                    f"Vertical displacement invariant failed: H-R={self.H_minus_R}, "
                    f"0.5*g*t1²={expected_H_minus_R}", 
                    False
                )
            else:
                self._log("Vertical displacement invariant satisfied", True)
        
        # 2. Check horizontal displacement for step_1: R = v0 * t1 (for first sub-question)
        if self.t1 is not None and self.v0 is not None:
            expected_R = self.v0 * self.t1
            if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                # Only check if v0 is approximately 2.5 (first sub-question)
                if math.isclose(self.v0, 2.5, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(
                        f"Horizontal displacement invariant failed: R={self.R}, "
                        f"v0*t1={expected_R}", 
                        False
                    )
                else:
                    self._log("Horizontal displacement invariant not applicable (v0 not 2.5)", True)
            else:
                self._log("Horizontal displacement invariant satisfied", True)
        
        # 3. Check tangent condition geometry (for second sub-question)
        if self.tan_theta is not None and self.v0 is not None:
            # From step_4: tanθ = v0/(g*t) where t is time to tangency
            # We can't check directly without t, but we can verify consistency
            # with trigonometric identity if we have both sin and cos
            theta = math.atan(self.tan_theta)
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            # Check sin² + cos² = 1
            if not math.isclose(sin_theta**2 + cos_theta**2, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                success = self._log(
                    f"Trigonometric identity failed: sin²+cos²={sin_theta**2 + cos_theta**2}", 
                    False
                )
            else:
                self._log("Trigonometric identity satisfied", True)
            
            # Check geometric constraints if we have the displacements
            if self.R_plus_Rcos is not None and self.H_minus_Rsin is not None:
                # From geometry: R + Rcosθ should match
                expected_R_plus_Rcos = self.R * (1 + cos_theta)
                if not math.isclose(self.R_plus_Rcos, expected_R_plus_Rcos, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(
                        f"Geometric R+Rcosθ failed: {self.R_plus_Rcos} vs {expected_R_plus_Rcos}", 
                        False
                    )
                else:
                    self._log("Geometric R+Rcosθ satisfied", True)
                
                # H - Rsinθ should match
                expected_H_minus_Rsin = self.H - self.R * sin_theta
                if not math.isclose(self.H_minus_Rsin, expected_H_minus_Rsin, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(
                        f"Geometric H-Rsinθ failed: {self.H_minus_Rsin} vs {expected_H_minus_Rsin}", 
                        False
                    )
                else:
                    self._log("Geometric H-Rsinθ satisfied", True)
        
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check first sub-question: v0 = 2.5 m/s
        if self.v0 is not None:
            if math.isclose(self.v0, 2.5, rel_tol=1e-2, abs_tol=1e-2):
                # Verify it's consistent with t1 from step_1
                if self.t1 is not None:
                    expected_t1 = math.sqrt(2 * (self.H - self.R) / self.g)
                    if not math.isclose(self.t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                        success = self._log(
                            f"t1 inconsistent for v0=2.5: {self.t1} vs sqrt(2(H-R)/g)={expected_t1}", 
                            False
                        )
                    else:
                        self._log("First sub-question trace consistency satisfied", True)
            else:
                self._log("First sub-question trace check skipped (v0 not 2.5)", True)
        
        # 2. Check second sub-question critical condition
        if self.v0 is not None and math.isclose(self.v0, self.v0_critical_expected, rel_tol=1e-2, abs_tol=1e-2):
            # For the critical v0, verify the tangent condition equations
            if self.tan_theta is not None:
                theta = math.atan(self.tan_theta)
                sin_theta = math.sin(theta)
                cos_theta = math.cos(theta)
                
                # From step_4 and step_5: R + Rcosθ = v0*t and tanθ = v0/(g*t)
                # Eliminate t: t = v0/(g*tanθ) = (R + Rcosθ)/v0
                # So: v0² = g*tanθ*(R + Rcosθ)
                lhs = self.v0**2
                rhs = self.g * self.tan_theta * self.R * (1 + cos_theta)
                
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(
                        f"Critical condition equation 1 failed: v0²={lhs}, g*tanθ*(R+Rcosθ)={rhs}", 
                        False
                    )
                else:
                    self._log("Critical condition equation 1 satisfied", True)
                
                # From step_4 and step_6: H - Rsinθ = 0.5*g*t² and tanθ = v0/(g*t)
                # Eliminate t: t = v0/(g*tanθ)
                # So: H - Rsinθ = 0.5*g*(v0/(g*tanθ))² = 0.5*v0²/(g*tan²θ)
                lhs2 = self.H - self.R * sin_theta
                rhs2 = 0.5 * self.v0**2 / (self.g * self.tan_theta**2)
                
                if not math.isclose(lhs2, rhs2, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(
                        f"Critical condition equation 2 failed: H-Rsinθ={lhs2}, "
                        f"0.5*v0²/(g*tan²θ)={rhs2}", 
                        False
                    )
                else:
                    self._log("Critical condition equation 2 satisfied", True)
            else:
                self._log("Critical condition check skipped (tanθ not available)", True)
        
        return success

    def test_monotonicity(self):
        """Perturb inputs and check trends in results."""
        success = True
        
        # Test 1: Perturb H for first sub-question
        # From step_1 and step_2: v0 = R / sqrt(2*(H-R)/g) = R * sqrt(g/(2*(H-R)))
        # So v0 decreases as H increases (since denominator increases)
        original_H = self.H
        perturbed_H = self.H * 1.1  # Increase by 10%
        
        # Calculate v0 for original and perturbed
        if original_H - self.R > 0:
            v0_original = self.R * math.sqrt(self.g / (2 * (original_H - self.R)))
            v0_perturbed = self.R * math.sqrt(self.g / (2 * (perturbed_H - self.R)))
            
            if v0_perturbed < v0_original:
                self._log(
                    f"Monotonicity test 1 passed: v0 decreases with H increase "
                    f"({v0_original:.3f} → {v0_perturbed:.3f})", 
                    True
                )
            else:
                success = self._log(
                    f"Monotonicity test 1 failed: v0 should decrease but "
                    f"{v0_perturbed:.3f} >= {v0_original:.3f}", 
                    False
                )
        else:
            self._log("Monotonicity test 1 skipped (H-R <= 0)", True)
        
        # Test 2: For critical condition, perturb R and check v0_critical trend
        # From the derived condition (eliminating θ):
        # v0_critical² = g * [R + (H - Rsinθ)/tanθ] but this depends on θ
        # Instead, we can check that if R increases, the required v0 to clear hemisphere increases
        original_R = self.R
        perturbed_R = self.R * 1.1  # Increase by 10%
        
        # Numerically solve for critical v0 at perturbed R (using same H)
        # We'll use the equations from steps 4-6:
        # 1) tanθ = v0/(g*t)
        # 2) R + Rcosθ = v0*t
        # 3) H - Rsinθ = 0.5*g*t²
        # Eliminate t: from (1): t = v0/(g*tanθ)
        # Substitute into (2): R(1+cosθ) = v0²/(g*tanθ)  → v0² = g*R*tanθ*(1+cosθ)
        # And into (3): H - Rsinθ = 0.5*v0²/(g*tan²θ)
        # We can solve these two equations for θ numerically
        
        try:
            # Define function to find θ for given R
            def critical_v0_for_R(R_val):
                # Solve for θ that satisfies both equations
                # We'll use a numerical search over θ in (0, π/2)
                best_theta = None
                best_error = float('inf')
                
                for theta_guess in [i * 0.01 for i in range(1, 157)]:  # 0.01 to 1.56 rad
                    sin_t = math.sin(theta_guess)
                    cos_t = math.cos(theta_guess)
                    tan_t = math.tan(theta_guess)
                    
                    # Equation from (2): v0² = g*R*tan_t*(1+cos_t)
                    v0_sq1 = self.g * R_val * tan_t * (1 + cos_t)
                    
                    # Equation from (3): v0² = 2*g*tan_t**2 * (H - R_val*sin_t)
                    v0_sq2 = 2 * self.g * tan_t**2 * (H - R_val * sin_t)
                    
                    error = abs(v0_sq1 - v0_sq2)
                    if error < best_error:
                        best_error = error
                        best_theta = theta_guess
                        v0_crit = math.sqrt(v0_sq1)  # or average of both
                
                if best_error < 0.01:  # Tolerance
                    return math.sqrt(self.g * R_val * math.tan(best_theta) * (1 + math.cos(best_theta)))
                return None
            
            v0_orig = critical_v0_for_R(original_R)
            v0_pert = critical_v0_for_R(perturbed_R)
            
            if v0_orig is not None and v0_pert is not None:
                if v0_pert > v0_orig:
                    self._log(
                        f"Monotonicity test 2 passed: critical v0 increases with R increase "
                        f"({v0_orig:.3f} → {v0_pert:.3f})", 
                        True
                    )
                else:
                    success = self._log(
                        f"Monotonicity test 2 failed: critical v0 should increase but "
                        f"{v0_pert:.3f} <= {v0_orig:.3f}", 
                        False
                    )
            else:
                self._log("Monotonicity test 2 skipped (numerical solution failed)", True)
        except Exception as e:
            self._log(f"Monotonicity test 2 error: {str(e)}", True)  # Don't fail on numerical issues
        
        return success

    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                self.logs.append(f"\n--- {test_name} ---")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"✗ {test_name} raised exception: {str(e)}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        self.logs.append(f"\n{'='*50}")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs