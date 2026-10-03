import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallback to results dictionary
        self.solution = solution_instance
        
        # Extract inputs - use defaults from problem statement
        self.theta = math.radians(getattr(solution_instance, 'theta', 37.0))
        self.L = getattr(solution_instance, 'L', 0.25)
        self.E = getattr(solution_instance, 'E', 12.0)
        self.r = getattr(solution_instance, 'r', 2.0)
        self.m = getattr(solution_instance, 'm', 0.1)
        self.B = getattr(solution_instance, 'B', 0.8)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        self.N = getattr(solution_instance, 'N', self.results.get('N', None))
        self.f_m = getattr(solution_instance, 'f_m', self.results.get('f_m', None))
        self.F_a = getattr(solution_instance, 'F_a', self.results.get('F_a', None))
        self.I = getattr(solution_instance, 'I', self.results.get('I', None))
        self.R_max = getattr(solution_instance, 'R_max', self.results.get('R_max', None))
        self.R_min = getattr(solution_instance, 'R_min', self.results.get('R_min', None))
        self.P = getattr(solution_instance, 'P', self.results.get('P', None))
        self.P_max = getattr(solution_instance, 'P_max', self.results.get('P_max', None))
        
        # Derived quantities for checks - use problem's g=10
        self.mg = self.m * self.g
        self.mg_sin = self.mg * math.sin(self.theta)
        self.mg_cos = self.mg * math.cos(self.theta)
        
    def test_invariants(self):
        """Check fundamental physics invariants"""
        logs = []
        success = True
        
        try:
            # 1. Check normal force: N = mg cosθ
            expected_N = self.mg_cos
            if self.N is not None:
                if not math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Normal force mismatch: expected {expected_N:.4f}, got {self.N:.4f}")
                    success = False
            else:
                # Calculate from formula
                self.N = expected_N
                logs.append(f"Normal force computed as {self.N:.4f} N")
            
            # 2. Check friction force: f_m = μN
            expected_f_m = self.mu * self.N
            if self.f_m is not None:
                if not math.isclose(self.f_m, expected_f_m, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Friction force mismatch: expected {expected_f_m:.4f}, got {self.f_m:.4f}")
                    success = False
            else:
                self.f_m = expected_f_m
                logs.append(f"Friction force computed as {self.f_m:.4f} N")
            
            # 3. Check equilibrium conditions for R_max and R_min
            # CORRECT PHYSICS:
            # For R_max (rod about to slide DOWN, friction up opposes motion)
            # Equation: mg sinθ = B*I*L + f_m  → B*I*L = mg sinθ - f_m
            # This requires mg sinθ > f_m
            if self.mg_sin > self.f_m:
                I_min_expected = (self.mg_sin - self.f_m) / (self.B * self.L)
            else:
                # If mg sinθ ≤ f_m, then rod won't slide down even with I=0
                # Minimum current needed is 0
                I_min_expected = 0.0
            
            if self.R_max is not None:
                # Avoid division by zero
                if self.R_max + self.r == 0:
                    logs.append("R_max + r is zero, cannot compute I")
                    success = False
                else:
                    I_from_Rmax = self.E / (self.R_max + self.r)
                    if not math.isclose(I_from_Rmax, I_min_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"R_max equilibrium mismatch: I from R_max = {I_from_Rmax:.4f}, expected {I_min_expected:.4f}")
                        success = False
            
            # For R_min (rod about to slide UP, friction down opposes motion)
            # Equation: mg sinθ + f_m = B*I*L  → B*I*L = mg sinθ + f_m
            I_max_expected = (self.mg_sin + self.f_m) / (self.B * self.L)
            
            if self.R_min is not None:
                # Avoid division by zero
                if self.R_min + self.r == 0:
                    logs.append("R_min + r is zero, cannot compute I")
                    success = False
                else:
                    I_from_Rmin = self.E / (self.R_min + self.r)
                    if not math.isclose(I_from_Rmin, I_max_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"R_min equilibrium mismatch: I from R_min = {I_from_Rmin:.4f}, expected {I_max_expected:.4f}")
                        success = False
            
            # 4. Check Ohm's Law consistency
            if self.I is not None:
                # Compute R from I: R = E/I - r
                if math.isclose(self.I, 0, abs_tol=1e-9):
                    logs.append("Current I is approximately zero, cannot compute R from I")
                else:
                    R_from_I = self.E / self.I - self.r
                    # Check if this R is within [R_min, R_max]
                    if self.R_min is not None and self.R_max is not None:
                        if R_from_I < self.R_min - 1e-2 or R_from_I > self.R_max + 1e-2:
                            logs.append(f"Current I={self.I:.4f}A gives R={R_from_I:.4f}Ω, outside range [{self.R_min:.4f}, {self.R_max:.4f}]")
                            success = False
            
            # 5. Check power formula consistency - only if P is provided
            if self.P is not None:
                # P should be I²R for some R in [R_min, R_max]
                # We can't check this directly without knowing which R was used
                logs.append("Power P provided, formula check skipped (requires specific R)")
            
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate results lead to final answers"""
        logs = []
        success = True
        
        try:
            # CORRECT PHYSICS:
            # Recompute R_max and R_min from correct equations
            if self.mg_sin > self.f_m:
                I_min = (self.mg_sin - self.f_m) / (self.B * self.L)
                if I_min > 0:
                    R_max_expected = self.E / I_min - self.r
                else:
                    # If I_min = 0, then R → ∞
                    R_max_expected = 1e6  # Very large
            else:
                # If mg sinθ ≤ f_m, rod won't slide down even with I=0
                # Maximum R corresponds to I=0, which requires R → ∞
                R_max_expected = 1e6  # Very large
            
            I_max = (self.mg_sin + self.f_m) / (self.B * self.L)
            if I_max > 0:
                R_min_expected = self.E / I_max - self.r
            else:
                R_min_expected = 0.0
            
            # Check against provided values
            if self.R_max is not None:
                # Allow for the case where R_max is very large
                if R_max_expected > 1e5:  # If expected is "infinite"
                    if self.R_max < 1e5:  # But got finite
                        logs.append(f"R_max trace mismatch: expected very large (>1e5), got {self.R_max:.4f}")
                        success = False
                elif not math.isclose(self.R_max, R_max_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"R_max trace mismatch: expected {R_max_expected:.4f}, got {self.R_max:.4f}")
                    success = False
            else:
                self.R_max = R_max_expected
                logs.append(f"R_max computed as {self.R_max:.4f} Ω")
            
            if self.R_min is not None:
                if not math.isclose(self.R_min, R_min_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"R_min trace mismatch: expected {R_min_expected:.4f}, got {self.R_min:.4f}")
                    success = False
            else:
                self.R_min = R_min_expected
                logs.append(f"R_min computed as {self.R_min:.4f} Ω")
            
            # Check range consistency
            if self.R_min is not None and self.R_max is not None:
                if self.R_min > self.R_max + 1e-2:  # Allow small tolerance
                    logs.append(f"Range invalid: R_min={self.R_min:.4f} > R_max={self.R_max:.4f}")
                    success = False
                else:
                    logs.append(f"Valid range: {self.R_min:.4f} Ω ≤ R ≤ {self.R_max:.4f} Ω")
            
            # Check maximum power
            # Maximum power transfer occurs when R = r (internal resistance) if within range
            R_for_Pmax = self.r
            if self.R_min <= R_for_Pmax <= self.R_max:
                P_max_expected = (self.E**2 * R_for_Pmax) / ((R_for_Pmax + self.r)**2)
            else:
                # If r is outside range, maximum power is at one endpoint
                P_at_min = (self.E**2 * self.R_min) / ((self.R_min + self.r)**2)
                P_at_max = (self.E**2 * self.R_max) / ((self.R_max + self.r)**2)
                P_max_expected = max(P_at_min, P_at_max)
            
            if self.P_max is not None:
                if not math.isclose(self.P_max, P_max_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"P_max trace mismatch: expected {P_max_expected:.4f}, got {self.P_max:.4f}")
                    success = False
            else:
                self.P_max = P_max_expected
                logs.append(f"P_max computed as {self.P_max:.4f} W")
            
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test how outputs change with input perturbations"""
        logs = []
        success = True
        
        try:
            # Store original values
            orig_m = self.m
            orig_B = self.B
            
            # Test 1: Increase mass by 10%
            self.m = orig_m * 1.1
            self.mg = self.m * self.g
            self.mg_sin = self.mg * math.sin(self.theta)
            self.mg_cos = self.mg * math.cos(self.theta)
            
            # Recompute derived quantities
            N_new = self.mg_cos
            f_m_new = self.mu * N_new
            
            # New equilibrium currents
            if self.mg_sin > f_m_new:
                I_min_new = (self.mg_sin - f_m_new) / (self.B * self.L)
                if I_min_new > 0:
                    R_max_new = self.E / I_min_new - self.r
                else:
                    R_max_new = 1e6
            else:
                R_max_new = 1e6
                
            I_max_new = (self.mg_sin + f_m_new) / (self.B * self.L)
            if I_max_new > 0:
                R_min_new = self.E / I_max_new - self.r
            else:
                R_min_new = 0.0
            
            logs.append(f"With mass +10%: R_min={R_min_new:.4f} Ω, R_max={R_max_new:.4f} Ω")
            
            # Restore original mass
            self.m = orig_m
            self.mg = self.m * self.g
            self.mg_sin = self.mg * math.sin(self.theta)
            self.mg_cos = self.mg * math.cos(self.theta)
            
            # Test 2: Increase magnetic field by 10%
            self.B = orig_B * 1.1
            
            # Recompute with new B
            if self.mg_sin > self.f_m:
                I_min_B = (self.mg_sin - self.f_m) / (self.B * self.L)
                if I_min_B > 0:
                    R_max_B = self.E / I_min_B - self.r
                else:
                    R_max_B = 1e6
            else:
                R_max_B = 1e6
                
            I_max_B = (self.mg_sin + self.f_m) / (self.B * self.L)
            if I_max_B > 0:
                R_min_B = self.E / I_max_B - self.r
            else:
                R_min_B = 0.0
            
            logs.append(f"With B +10%: R_min={R_min_B:.4f} Ω, R_max={R_max_B:.4f} Ω")
            
            # Check monotonicity qualitatively
            if self.R_min is not None and self.R_max is not None:
                # Stronger B means less current needed for same force, so R increases
                if R_min_B > self.R_min + 1e-2:
                    logs.append("R_min increased with stronger B (expected)")
                if R_max_B > self.R_max + 1e-2:
                    logs.append("R_max increased with stronger B (expected)")
            
            # Restore B
            self.B = orig_B
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return summary"""
        all_logs = []
        overall_success = True
        
        # Run invariant tests
        success, logs = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        return overall_success, all_logs