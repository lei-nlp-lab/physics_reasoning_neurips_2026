import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results with fallbacks."""
        # Safe attribute access with fallbacks
        self.R = getattr(solution_instance, 'R', 1.0)  # Default 1.0 if missing
        self.g = getattr(solution_instance, 'g', 9.81)
        self.m_A = getattr(solution_instance, 'm_A', 1.0)
        self.m_B = getattr(solution_instance, 'm_B', 1.0)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.v_C = results.get('v_C', 0.0)
        self.E_p = results.get('E_p', 0.0)
        self.v_A = results.get('v_A', 0.0)  # Velocity at highest point
        self.v_B = results.get('v_B', 0.0)
        self.v_A_spring = results.get('v_A_spring', self.v_C)  # Velocity when leaving spring
        
        # Expected values based on correct physics
        # v_A at highest point: √(gR)
        self.v_A_expected = math.sqrt(self.g * self.R)
        # v_C (velocity at point C) = v_A_spring: √(5gR)
        self.v_C_expected = math.sqrt(5 * self.g * self.R)
        # v_B: √(20gR) = 2√(5gR)
        self.v_B_expected = math.sqrt(20 * self.g * self.R)
        # E_p: 2.5*m_A*g*R + 10*m_B*g*R
        self.E_p_expected = 2.5 * self.m_A * self.g * self.R + 10 * self.m_B * self.g * self.R
    
    def test_invariants(self):
        """Test physics invariants: Energy conservation and Newton's laws."""
        logs = []
        try:
            # 1. Newton's second law at highest point for ball A
            # m_A*g = m_A*v_A^2/R
            lhs = self.m_A * self.g
            rhs = self.m_A * self.v_A**2 / self.R
            newton_ok = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
            
            if newton_ok:
                logs.append("✓ Newton's second law at highest point for ball A holds")
            else:
                logs.append(f"✗ Newton's second law fails: m*g={lhs:.3f}, m*v²/R={rhs:.3f}")
            
            # 2. Energy conservation for ball A from spring to highest point
            # At spring release: K_spring = 0.5*m_A*v_A_spring^2, U_spring = 0
            # At highest point: K_high = 0.5*m_A*v_A^2, U_high = m_A*g*(2R)
            K_spring = 0.5 * self.m_A * self.v_A_spring**2
            K_high = 0.5 * self.m_A * self.v_A**2
            U_high = self.m_A * self.g * (2 * self.R)
            
            energy_conserved = math.isclose(
                K_spring, 
                K_high + U_high, 
                rel_tol=1e-2, 
                abs_tol=1e-2
            )
            
            if energy_conserved:
                logs.append("✓ Energy conservation for ball A holds")
            else:
                logs.append(f"✗ Energy conservation for ball A fails: K_spring={K_spring:.3f}, K_high+U_high={K_high+U_high:.3f}")
            
            # 3. Energy conservation for ball B
            # Spring to point B: K_spring = 0.5*m_B*v_B^2, U_B = m_B*g*(10R)
            K_B_spring = 0.5 * self.m_B * self.v_B**2
            U_B = self.m_B * self.g * (10 * self.R)
            energy_B_conserved = math.isclose(K_B_spring, U_B, rel_tol=1e-2, abs_tol=1e-2)
            
            if energy_B_conserved:
                logs.append("✓ Energy conservation for ball B holds")
            else:
                logs.append(f"✗ Energy conservation for ball B fails: K_spring={K_B_spring:.3f}, U_B={U_B:.3f}")
            
            return all([newton_ok, energy_conserved, energy_B_conserved]), logs
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        logs = []
        try:
            # Check v_A = √(gR)
            v_A_ok = math.isclose(self.v_A, self.v_A_expected, rel_tol=1e-2, abs_tol=1e-2)
            if v_A_ok:
                logs.append(f"✓ v_A matches √(gR): {self.v_A:.3f} ≈ {self.v_A_expected:.3f}")
            else:
                logs.append(f"✗ v_A mismatch: {self.v_A:.3f} ≠ {self.v_A_expected:.3f}")
            
            # Check v_C = √(5gR)
            v_C_ok = math.isclose(self.v_C, self.v_C_expected, rel_tol=1e-2, abs_tol=1e-2)
            if v_C_ok:
                logs.append(f"✓ v_C matches √(5gR): {self.v_C:.3f} ≈ {self.v_C_expected:.3f}")
            else:
                logs.append(f"✗ v_C mismatch: {self.v_C:.3f} ≠ {self.v_C_expected:.3f}")
            
            # Check v_B = √(20gR) = 2√(5gR)
            v_B_ok = math.isclose(self.v_B, self.v_B_expected, rel_tol=1e-2, abs_tol=1e-2)
            if v_B_ok:
                logs.append(f"✓ v_B matches √(20gR): {self.v_B:.3f} ≈ {self.v_B_expected:.3f}")
            else:
                logs.append(f"✗ v_B mismatch: {self.v_B:.3f} ≠ {self.v_B_expected:.3f}")
            
            # Check v_A_spring = v_C
            v_A_spring_ok = math.isclose(self.v_A_spring, self.v_C, rel_tol=1e-2, abs_tol=1e-2)
            if v_A_spring_ok:
                logs.append(f"✓ v_A_spring = v_C: {self.v_A_spring:.3f} ≈ {self.v_C:.3f}")
            else:
                logs.append(f"✗ v_A_spring ≠ v_C: {self.v_A_spring:.3f} ≠ {self.v_C:.3f}")
            
            # Check E_p = 0.5*m_A*v_A_spring^2 + 0.5*m_B*v_B^2
            E_p_calculated = 0.5 * self.m_A * self.v_A_spring**2 + 0.5 * self.m_B * self.v_B**2
            E_p_ok = math.isclose(self.E_p, E_p_calculated, rel_tol=1e-2, abs_tol=1e-2)
            if E_p_ok:
                logs.append(f"✓ E_p calculation consistent: {self.E_p:.3f} ≈ {E_p_calculated:.3f}")
            else:
                logs.append(f"✗ E_p inconsistent: {self.E_p:.3f} ≠ {E_p_calculated:.3f}")
            
            # Check against expected expression
            E_p_symbolic_ok = math.isclose(self.E_p, self.E_p_expected, rel_tol=1e-2, abs_tol=1e-2)
            if E_p_symbolic_ok:
                logs.append(f"✓ E_p matches expected expression: {self.E_p:.3f} ≈ {self.E_p_expected:.3f}")
            else:
                logs.append(f"✗ E_p differs from expected: {self.E_p:.3f} ≠ {self.E_p_expected:.3f}")
            
            return all([v_A_ok, v_C_ok, v_B_ok, v_A_spring_ok, E_p_ok, E_p_symbolic_ok]), logs
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
    
    def test_monotonicity(self):
        """Test monotonic behavior when mass parameters change."""
        logs = []
        original_m_A = self.m_A
        original_m_B = self.m_B
        
        try:
            # Original E_p
            E_p_orig = self.E_p_expected
            
            # Test 1: Increase m_A by 10%, check E_p increases proportionally
            self.m_A *= 1.1
            E_p_inc = 2.5 * self.m_A * self.g * self.R + 10 * self.m_B * self.g * self.R
            self.m_A = original_m_A  # Restore
            
            # Test 2: Increase m_B by 10%, check E_p increases proportionally
            self.m_B *= 1.1
            E_p_inc2 = 2.5 * self.m_A * self.g * self.R + 10 * self.m_B * self.g * self.R
            self.m_B = original_m_B  # Restore
            
            # Test 3: Decrease m_A by 10%, check E_p decreases
            self.m_A *= 0.9
            E_p_dec = 2.5 * self.m_A * self.g * self.R + 10 * self.m_B * self.g * self.R
            self.m_A = original_m_A  # Restore
            
            # Check monotonicity
            monotonic_ok = (
                E_p_inc > E_p_orig and 
                E_p_inc2 > E_p_orig and 
                E_p_dec < E_p_orig
            )
            
            if monotonic_ok:
                logs.append("✓ Monotonicity test passed: E_p increases with mass increase")
                logs.append(f"  E_p_orig={E_p_orig:.3f}, E_p_+10%m_A={E_p_inc:.3f}, E_p_+10%m_B={E_p_inc2:.3f}, E_p_-10%m_A={E_p_dec:.3f}")
            else:
                logs.append("✗ Monotonicity test failed")
                logs.append(f"  E_p_orig={E_p_orig:.3f}, E_p_+10%m_A={E_p_inc:.3f}, E_p_+10%m_B={E_p_inc2:.3f}, E_p_-10%m_A={E_p_dec:.3f}")
            
            return monotonic_ok, logs
            
        except Exception as e:
            # Restore original values in case of error
            self.m_A = original_m_A
            self.m_B = original_m_B
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            return False, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        success = True
        
        # Run invariant tests
        inv_success, inv_logs = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(inv_logs)
        success = success and inv_success
        
        # Run trace consistency tests
        trace_success, trace_logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(trace_logs)
        success = success and trace_success
        
        # Run monotonicity tests
        mono_success, mono_logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(mono_logs)
        success = success and mono_success
        
        # Summary
        all_logs.append("\n=== Summary ===")
        if success:
            all_logs.append("✓ All physics checks passed")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return success, all_logs