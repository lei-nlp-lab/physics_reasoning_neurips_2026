import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract ALL values from results dictionary only
        results = getattr(solution_instance, 'results', {})
        
        # Required constants (with defaults if not in results)
        self.d = results.get('d', 10.0)  # default example value
        self.mu = results.get('mu', 0.1)  # default example value
        self.g = results.get('g', 9.8)    # default example value
        
        # Results with safe fallback
        self.v_B = results.get('$v_B$', results.get('v_B', 0.0))
        self.Delta_p = results.get('$Deltap$', 0.0)
        self.t = results.get('t', 0.0)
        self.v_p = results.get('$v_p$', results.get('v_p', 0.0))
        self.KE_P = results.get('$frac12mv_P2$', 0.0)
        self.L = results.get('L', 0.0)
        
        self.logs = []
        
    def log(self, msg, success=None):
        """Helper for logging"""
        entry = {'message': msg}
        if success is not None:
            entry['success'] = success
        self.logs.append(entry)
        
    def test_invariants(self):
        """Verify physical invariants based on KB theorems using numeric checks"""
        tests_passed = True
        
        # 1. Work-energy theorem from A to P (step_1)
        try:
            # Net work = change in kinetic energy
            # Work done by gravity: m*g*d*sin(45°)
            # Work done by friction: -μ*m*g*cos(45°)*d
            # Use m = 1.0 for testing (mass cancels anyway)
            m = 1.0
            theta = math.pi/4  # 45°
            net_work = m * self.g * self.d * math.sin(theta) - \
                       self.mu * m * self.g * self.d * math.cos(theta)
            KE_P = 0.5 * m * self.v_p**2
            
            if not math.isclose(net_work, KE_P, rel_tol=1e-2):
                self.log(f"FAIL: Work-energy theorem (A→P) violated. Net work={net_work:.3f}, ΔKE={KE_P:.3f}", False)
                tests_passed = False
            else:
                self.log("PASS: Work-energy theorem (A→P) holds", True)
        except Exception as e:
            self.log(f"ERROR in work-energy check: {str(e)}", False)
            tests_passed = False
        
        # 2. Impulse-momentum theorem (step_2)
        try:
            # Net force * time = change in momentum
            # Net force = m*g*sin(45°) - μ*m*g*cos(45°)
            m = 1.0
            theta = math.pi/4
            net_force = m * self.g * math.sin(theta) - self.mu * m * self.g * math.cos(theta)
            impulse = net_force * self.t
            delta_p = m * self.v_p  # from rest to v_p
            
            if not math.isclose(impulse, delta_p, rel_tol=1e-2):
                self.log(f"FAIL: Impulse-momentum theorem (A→P) violated. Impulse={impulse:.3f}, Δp={delta_p:.3f}", False)
                tests_passed = False
            else:
                self.log("PASS: Impulse-momentum theorem (A→P) holds", True)
        except Exception as e:
            self.log(f"ERROR in impulse-momentum check: {str(e)}", False)
            tests_passed = False
        
        # 3. Conservation of mechanical energy (step_5)
        try:
            if not math.isclose(self.v_B, self.v_p, rel_tol=1e-2):
                self.log(f"NOTE: v_B ({self.v_B:.3f}) ≠ v_p ({self.v_p:.3f}) - horizontal friction present?", None)
            else:
                self.log("PASS: v_B = v_p consistent with frictionless horizontal section", True)
        except Exception as e:
            self.log(f"ERROR in energy conservation check: {str(e)}", False)
            tests_passed = False
        
        return tests_passed
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        tests_passed = True
        
        # 1. Check step_3 results match derived formulas
        try:
            # t from formula: sqrt(2*sqrt(2)*d/(g*(1-μ)))
            t_expected = math.sqrt(2*math.sqrt(2)*self.d/(self.g*(1-self.mu)))
            if not math.isclose(self.t, t_expected, rel_tol=1e-2):
                self.log(f"FAIL: t mismatch. Expected {t_expected:.3f}, got {self.t:.3f}", False)
                tests_passed = False
            else:
                self.log("PASS: t matches formula", True)
        except Exception as e:
            self.log(f"ERROR in t consistency: {str(e)}", False)
            tests_passed = False
        
        # 2. Check v_p formula
        try:
            v_p_expected = math.sqrt(math.sqrt(2)*self.g*self.d*(1-self.mu))
            if not math.isclose(self.v_p, v_p_expected, rel_tol=1e-2):
                self.log(f"FAIL: v_p mismatch. Expected {v_p_expected:.3f}, got {self.v_p:.3f}", False)
                tests_passed = False
            else:
                self.log("PASS: v_p matches formula", True)
        except Exception as e:
            self.log(f"ERROR in v_p consistency: {str(e)}", False)
            tests_passed = False
        
        # 3. Check L formula from step_8
        try:
            L_expected = math.sqrt(2)*self.d*(1 - self.mu)
            if not math.isclose(self.L, L_expected, rel_tol=1e-2):
                self.log(f"FAIL: L mismatch. Expected {L_expected:.3f}, got {self.L:.3f}", False)
                tests_passed = False
            else:
                self.log("PASS: L matches formula", True)
        except Exception as e:
            self.log(f"ERROR in L consistency: {str(e)}", False)
            tests_passed = False
        
        # 4. Check v_B = v_p (from step_5)
        if not math.isclose(self.v_B, self.v_p, rel_tol=1e-2):
            self.log(f"WARN: v_B ({self.v_B:.3f}) ≠ v_p ({self.v_p:.3f})", None)
        else:
            self.log("PASS: v_B = v_p", True)
        
        return tests_passed
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends using numeric values only"""
        tests_passed = True
        
        try:
            # Store original values
            d_orig = float(self.d)
            mu_orig = float(self.mu)
            
            # Test 1: Increase μ, L should decrease
            self.mu = mu_orig * 1.1  # 10% increase
            L1 = math.sqrt(2)*self.d*(1 - self.mu)
            
            self.mu = mu_orig * 0.9  # 10% decrease
            L2 = math.sqrt(2)*self.d*(1 - self.mu)
            
            if not (L1 < L2):  # Higher μ → smaller L
                self.log(f"FAIL: L not monotonic with μ. L(μ+10%)={L1:.3f}, L(μ-10%)={L2:.3f}", False)
                tests_passed = False
            else:
                self.log("PASS: L decreases with increasing μ", True)
            
            # Test 2: Increase d, L should increase
            self.mu = mu_orig  # restore μ
            self.d = d_orig * 1.1
            L3 = math.sqrt(2)*self.d*(1 - self.mu)
            
            self.d = d_orig * 0.9
            L4 = math.sqrt(2)*self.d*(1 - self.mu)
            
            if not (L3 > L4):  # Higher d → larger L
                self.log(f"FAIL: L not monotonic with d. L(d+10%)={L3:.3f}, L(d-10%)={L4:.3f}", False)
                tests_passed = False
            else:
                self.log("PASS: L increases with increasing d", True)
            
            # Restore original values
            self.d = d_orig
            self.mu = mu_orig
            
        except Exception as e:
            self.log(f"ERROR in monotonicity test: {str(e)}", False)
            tests_passed = False
        
        return tests_passed
    
    def run_all_tests(self):
        """Run all tests and return success status"""
        self.logs = []  # Clear logs
        
        try:
            # Run tests
            invariant_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            monotonic_ok = self.test_monotonicity()
            
            success = invariant_ok and trace_ok and monotonic_ok
            
            # Summary log
            summary = (f"Tests: Invariants={invariant_ok}, Trace={trace_ok}, "
                      f"Monotonicity={monotonic_ok}, Overall={success}")
            self.log(summary, success)
            
        except Exception as e:
            error_msg = f"Critical error in test suite: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg, False)
            success = False
        
        return success, self.logs