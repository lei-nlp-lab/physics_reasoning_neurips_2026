import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.alpha = math.radians(getattr(solution_instance, 'alpha', 60.0))
        self.beta = math.radians(getattr(solution_instance, 'beta', 30.0))
        
        # Extract results with safe fallbacks
        self.omega = self.results.get('omega', 0.0)
        self.N_A = self.results.get('N', 0.0)
        self.F_B = self.results.get('F', 0.0)
        self.f_B = self.results.get('f', 0.0)
        self.r = self.results.get('r', 0.0)
        
        self.logs = []
    
    def _log(self, msg, success=None):
        """Helper for logging with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {msg}")
        else:
            self.logs.append(f"[INFO] {msg}")
        return success
    
    def test_invariants(self):
        """Check fundamental physical laws."""
        success = True
        
        # 1. Geometric relationship: r = R * sin(alpha)
        r_expected = self.R * math.sin(self.alpha)
        if not math.isclose(self.r, r_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Geometry mismatch: r={self.r:.3f}, expected R*sin(α)={r_expected:.3f}", False) and success
        
        # 2. Newton's 2nd Law at point A (centripetal force)
        # From step_3: mg*tan(α) = mω²r
        lhs = self.m * self.g * math.tan(self.alpha)
        rhs = self.m * self.omega**2 * self.r
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Newton's 2nd Law at A fails: mg*tanα={lhs:.3f}, mω²r={rhs:.3f}", False) and success
        
        # 3. Newton's 2nd Law at point A (normal force)
        # From step_1: N = mg*cos(α) + m(v²)/R, where v = ωr
        v = self.omega * self.r
        N_expected = self.m * self.g * math.cos(self.alpha) + self.m * v**2 / self.R
        if not math.isclose(self.N_A, N_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Normal force at A mismatch: N={self.N_A:.3f}, expected={N_expected:.3f}", False) and success
        
        # 4. Newton's 2nd Law at point B (both components)
        # Horizontal: Fsinβ + fcosβ = mω²Rsinβ
        lhs_horiz = self.F_B * math.sin(self.beta) + self.f_B * math.cos(self.beta)
        rhs_horiz = self.m * self.omega**2 * self.R * math.sin(self.beta)
        if not math.isclose(lhs_horiz, rhs_horiz, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Horizontal force at B fails: {lhs_horiz:.3f} vs {rhs_horiz:.3f}", False) and success
        
        # Vertical: Fcosβ - fsinβ = mg
        lhs_vert = self.F_B * math.cos(self.beta) - self.f_B * math.sin(self.beta)
        rhs_vert = self.m * self.g
        if not math.isclose(lhs_vert, rhs_vert, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"Vertical force at B fails: {lhs_vert:.3f} vs {rhs_vert:.3f}", False) and success
        
        return success
    
    def test_trace_consistency(self):
        """Check intermediate results match final answers."""
        success = True
        
        # 1. Check angular velocity ω = sqrt(2g/R)
        omega_expected = math.sqrt(2 * self.g / self.R)
        if not math.isclose(self.omega, omega_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"ω mismatch: {self.omega:.3f} vs sqrt(2g/R)={omega_expected:.3f}", False) and success
        
        # 2. Check normal force at A = mg*cos(α) + mω²r
        v = self.omega * self.r
        N_expected = self.m * self.g * math.cos(self.alpha) + self.m * v**2 / self.R
        if not math.isclose(self.N_A, N_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"N_A mismatch: {self.N_A:.3f} vs expected={N_expected:.3f}", False) and success
        
        # 3. Check frictional force at B = (√3-1)/2 * mg
        f_expected = (math.sqrt(3) - 1) / 2 * self.m * self.g
        if not math.isclose(self.f_B, f_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"f_B mismatch: {self.f_B:.3f} vs expected={(math.sqrt(3)-1)/2*self.m*self.g:.3f}", False) and success
        
        # 4. Check normal force at B = (√3+1)/2 * mg
        F_expected = (math.sqrt(3) + 1) / 2 * self.m * self.g
        if not math.isclose(self.F_B, F_expected, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(f"F_B mismatch: {self.F_B:.3f} vs expected={(math.sqrt(3)+1)/2*self.m*self.g:.3f}", False) and success
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and check output trends."""
        success = True
        original_m = self.m
        original_omega = self.omega
        original_N_A = self.N_A
        original_f_B = self.f_B
        original_F_B = self.F_B
        
        try:
            # Perturb mass by +10%
            perturb_factor = 1.1
            self.m = original_m * perturb_factor
            
            # Recalculate expected values based on correct formulas
            # ω is independent of m: ω = sqrt(2g/R)
            new_omega = math.sqrt(2 * self.g / self.R)
            
            # N_A = mg*cos(α) + mω²r, where ω is constant, r = R*sin(α)
            r = self.R * math.sin(self.alpha)
            # Exact formula: N_A = m(g*cos(α) + ω²r) = m(g*cos(α) + (2g/R)*R*sin²(α))
            # = mg(cos(α) + 2sin²(α))
            new_N_A = self.m * self.g * (math.cos(self.alpha) + 2 * math.sin(self.alpha)**2)
            
            # f_B = (√3-1)/2 * mg (proportional to m)
            new_f_B = (math.sqrt(3) - 1) / 2 * self.m * self.g
            
            # F_B = (√3+1)/2 * mg (proportional to m)
            new_F_B = (math.sqrt(3) + 1) / 2 * self.m * self.g
            
            # Check trends - ω should be independent of m
            if not math.isclose(new_omega, original_omega, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"ω should be independent of m: {new_omega:.3f} vs {original_omega:.3f}", True)
            
            # N_A should scale exactly with m: N_A = m * constant
            # Compute the constant from original values
            if original_m != 0:
                N_constant = original_N_A / original_m
                expected_new_N_A = self.m * N_constant
                if not math.isclose(new_N_A, expected_new_N_A, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(f"N_A scaling check: {new_N_A:.3f} vs {expected_new_N_A:.3f}", False) and success
                
                # Ratio should be exactly perturb_factor
                N_ratio = new_N_A / original_N_A
                if not math.isclose(N_ratio, perturb_factor, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(f"N_A should scale with m: ratio={N_ratio:.3f}, expected={perturb_factor}", False) and success
            else:
                success = self._log("Original mass is zero, skipping N_A scaling", False) and success
            
            # f_B and F_B should scale exactly with m
            if original_f_B != 0:
                f_ratio = new_f_B / original_f_B
                if not math.isclose(f_ratio, perturb_factor, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(f"f_B should scale with m: ratio={f_ratio:.3f}, expected={perturb_factor}", False) and success
            
            if original_F_B != 0:
                F_ratio = new_F_B / original_F_B
                if not math.isclose(F_ratio, perturb_factor, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(f"F_B should scale with m: ratio={F_ratio:.3f}, expected={perturb_factor}", False) and success
                
        except Exception as e:
            success = self._log(f"Monotonicity test failed: {e}", False)
        finally:
            # Restore original values
            self.m = original_m
            
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name}...")
                success = test_func()
                overall_success = overall_success and success
                self._log(f"{test_name} completed", success)
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"All tests {'PASSED' if overall_success else 'FAILED'}")
        return overall_success, self.logs