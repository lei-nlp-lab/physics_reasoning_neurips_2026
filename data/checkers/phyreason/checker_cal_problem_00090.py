import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization with safe attribute access.
        Uses getattr to avoid AttributeError in sandboxed environments.
        """
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Safely extract input parameters (with defaults)
        self.mu = float(getattr(solution_instance, 'mu', self.results.get('mu', 0.1)))
        self.g = float(getattr(solution_instance, 'g', self.results.get('g', 9.8)))
        self.s = float(getattr(solution_instance, 's', self.results.get('s', 1.0)))
        
        # Safely extract computed quantities
        self.v_0 = float(self.results.get('v_0', 0.0))  # velocity of block a before collision
        self.v_b = float(self.results.get('v_b', 0.0))  # velocity of block b after collision
        self.L = float(self.results.get('L', 0.0))      # length of rope
        
        # Expected expressions from KB (numeric)
        self.v_b_expected = math.sqrt(2 * self.mu * self.g * self.s)
        self.L_expected = 4 * self.mu * self.s
        
        # For numeric checking, use m = 1.0 (arbitrary, cancels out in ratios)
        self.m = 1.0
        self.m_b = 3.0
        
    def test_invariants(self):
        """Test conservation laws based on KB theorems using numeric values."""
        logs = []
        success = True
        
        try:
            # 1. Check conservation of mechanical energy for the swing
            # Before: U = m*g*L, K = 0
            # After (at lowest point): U = 0, K = 0.5*m*v_0^2
            energy_initial = self.m * self.g * self.L
            energy_final = 0.5 * self.m * self.v_0**2
            
            if not math.isclose(energy_initial, energy_final, rel_tol=1e-2):
                logs.append(f"FAIL: Swing energy not conserved. Initial: {energy_initial}, Final: {energy_final}")
                success = False
            else:
                logs.append("PASS: Swing energy conservation")
            
            # 2. Check conservation of momentum during collision
            # Before: p_total = m*v_0 (block b at rest)
            # After: p_total = m*v_a + 3m*v_b
            # v_a after collision from momentum conservation: v_a = v_0 - 3*v_b
            v_a = self.v_0 - 3 * self.v_b
            momentum_before = self.m * self.v_0
            momentum_after = self.m * v_a + 3 * self.m * self.v_b
            
            if not math.isclose(momentum_before, momentum_after, rel_tol=1e-2):
                logs.append(f"FAIL: Momentum not conserved in collision. Before: {momentum_before}, After: {momentum_after}")
                success = False
            else:
                logs.append("PASS: Momentum conservation in collision")
            
            # 3. Check conservation of kinetic energy during elastic collision
            # Before: KE = 0.5*m*v_0^2
            # After: KE = 0.5*m*v_a^2 + 0.5*3m*v_b^2
            KE_before = 0.5 * self.m * self.v_0**2
            KE_after = 0.5 * self.m * v_a**2 + 0.5 * 3 * self.m * self.v_b**2
            
            if not math.isclose(KE_before, KE_after, rel_tol=1e-2):
                logs.append(f"FAIL: Kinetic energy not conserved in collision. Before: {KE_before}, After: {KE_after}")
                success = False
            else:
                logs.append("PASS: Kinetic energy conservation in elastic collision")
            
            # 4. Check work-energy theorem for block b sliding
            # Work done by friction = -μ*3m*g*s
            # Change in KE = 0 - 0.5*3m*v_b^2
            work_friction = -self.mu * 3 * self.m * self.g * self.s
            delta_KE_b = 0 - 0.5 * 3 * self.m * self.v_b**2
            
            if not math.isclose(work_friction, delta_KE_b, rel_tol=1e-2):
                logs.append(f"FAIL: Work-energy theorem for block b. Work: {work_friction}, ΔKE: {delta_KE_b}")
                success = False
            else:
                logs.append("PASS: Work-energy theorem for sliding block")
            
            # 5. Verify elastic collision condition (v_0 = 2*v_b for m_b = 3m_a)
            if not math.isclose(self.v_0, 2 * self.v_b, rel_tol=1e-2):
                logs.append(f"FAIL: Elastic collision condition violated. v_0={self.v_0}, 2*v_b={2*self.v_b}")
                success = False
            else:
                logs.append("PASS: Elastic collision condition (v_0 = 2*v_b)")
                
        except Exception as e:
            logs.append(f"ERROR in invariants test: {e}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answer."""
        logs = []
        success = True
        
        try:
            # From KB analysis:
            # Step 1: v_b = sqrt(2*μ*g*s)
            if not math.isclose(self.v_b, self.v_b_expected, rel_tol=1e-2):
                logs.append(f"FAIL: v_b mismatch. Got: {self.v_b}, Expected: {self.v_b_expected}")
                success = False
            else:
                logs.append(f"PASS: v_b = sqrt(2μgs)")
            
            # Step 2: v_0 = sqrt(2*g*L)
            v_0_from_L = math.sqrt(2 * self.g * self.L)
            if not math.isclose(self.v_0, v_0_from_L, rel_tol=1e-2):
                logs.append(f"FAIL: v_0 from L mismatch. v_0={self.v_0}, sqrt(2gL)={v_0_from_L}")
                success = False
            else:
                logs.append("PASS: v_0 = sqrt(2gL)")
            
            # Step 5: L = 4μs
            if not math.isclose(self.L, self.L_expected, rel_tol=1e-2):
                logs.append(f"FAIL: L mismatch. Got: {self.L}, Expected: {self.L_expected}")
                success = False
            else:
                logs.append(f"PASS: L = 4μs = {self.L}")
            
            # Check that v_0 = 2*v_b leads to L = 4μs
            # v_0 = sqrt(2gL) = 2*v_b = 2*sqrt(2μgs)
            # Squaring: 2gL = 4*(2μgs) = 8μgs => L = 4μs
            left_side = 2 * self.g * self.L
            right_side = 8 * self.mu * self.g * self.s
            
            if not math.isclose(left_side, right_side, rel_tol=1e-2):
                logs.append(f"FAIL: Chain consistency broken. 2gL={left_side}, 8μgs={right_side}")
                success = False
            else:
                logs.append("PASS: Full chain consistency (v_0=2v_b → L=4μs)")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {e}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb input parameters and verify expected trends.
        Restores original values after test.
        """
        logs = []
        success = True
        
        try:
            # Store original values
            mu_orig = self.mu
            s_orig = self.s
            g_orig = self.g
            
            # Test 1: Increase μ by 10%
            self.mu = mu_orig * 1.1
            v_b_new = math.sqrt(2 * self.mu * self.g * self.s)
            L_new = 4 * self.mu * self.s
            
            # v_b should increase with μ (since sqrt(μ))
            if v_b_new <= self.v_b:
                logs.append(f"FAIL: v_b not monotonic with μ. Original: {self.v_b}, New: {v_b_new}")
                success = False
            else:
                logs.append(f"PASS: v_b increases with μ ({self.v_b} → {v_b_new})")
            
            # L should increase with μ (linear)
            if L_new <= self.L:
                logs.append(f"FAIL: L not monotonic with μ. Original: {self.L}, New: {L_new}")
                success = False
            else:
                logs.append(f"PASS: L increases with μ ({self.L} → {L_new})")
            
            # Restore μ
            self.mu = mu_orig
            
            # Test 2: Increase s by 10%
            self.s = s_orig * 1.1
            v_b_new = math.sqrt(2 * self.mu * self.g * self.s)
            L_new = 4 * self.mu * self.s
            
            # Both v_b and L should increase with s
            if v_b_new <= self.v_b:
                logs.append(f"FAIL: v_b not monotonic with s. Original: {self.v_b}, New: {v_b_new}")
                success = False
            else:
                logs.append(f"PASS: v_b increases with s ({self.v_b} → {v_b_new})")
            
            if L_new <= self.L:
                logs.append(f"FAIL: L not monotonic with s. Original: {self.L}, New: {L_new}")
                success = False
            else:
                logs.append(f"PASS: L increases with s ({self.L} → {L_new})")
            
            # Restore s
            self.s = s_orig
            
            # Test 3: Increase g by 10%
            self.g = g_orig * 1.1
            v_b_new = math.sqrt(2 * self.mu * self.g * self.s)
            # L should be independent of g (L = 4μs)
            L_new = 4 * self.mu * self.s
            
            # v_b should increase with g (sqrt(g))
            if v_b_new <= self.v_b:
                logs.append(f"FAIL: v_b not monotonic with g. Original: {self.v_b}, New: {v_b_new}")
                success = False
            else:
                logs.append(f"PASS: v_b increases with g ({self.v_b} → {v_b_new})")
            
            # L should remain constant with g change
            if not math.isclose(L_new, self.L, rel_tol=1e-2):
                logs.append(f"FAIL: L depends on g. Original: {self.L}, New: {L_new}")
                success = False
            else:
                logs.append(f"PASS: L independent of g (remains {self.L})")
            
            # Restore all original values
            self.mu = mu_orig
            self.s = s_orig
            self.g = g_orig
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            logs.append(traceback.format_exc())
            success = False
            # Try to restore on error
            try:
                self.mu = mu_orig
                self.s = s_orig
                self.g = g_orig
            except:
                pass
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run individual tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} Test ===")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
                all_logs.append(f"{test_name}: FAILED")
            else:
                all_logs.append(f"{test_name}: PASSED")
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs