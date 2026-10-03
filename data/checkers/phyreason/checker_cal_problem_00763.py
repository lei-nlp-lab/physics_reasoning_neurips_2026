import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Constants from problem statement (kg, m/s)
        self.m_A = 30.0
        self.m_B = 30.0
        self.m_person = 40.0
        self.v_0 = 2.0
        
        # Extract results safely
        self.P = self._safe_get('P')
        self.v_person = self._safe_get('v_person')
        
        # Symbolic variables for physics laws
        self.m_A_sym = sp.symbols('m_A', positive=True, real=True)
        self.m_B_sym = sp.symbols('m_B', positive=True, real=True)
        self.m_person_sym = sp.symbols('m_person', positive=True, real=True)
        self.v_0_sym = sp.symbols('v_0', real=True)
        self.v1_sym = sp.symbols('v1', real=True)
        self.v2_sym = sp.symbols('v2', real=True)
        self.v_person_sym = sp.symbols('v_person', real=True)
        
        self.logs = []
        
    def _safe_get(self, key):
        """Safe extraction of attributes/results with fallback to None."""
        try:
            # Try direct attribute
            val = getattr(self.solution, key, None)
            if val is not None:
                return float(val)
            # Try results dictionary
            if hasattr(self.solution, 'results'):
                val = self.solution.results.get(key, None)
                if val is not None:
                    return float(val)
        except (AttributeError, ValueError, TypeError) as e:
            self.logs.append(f"Warning: Could not extract {key}: {str(e)}")
        return None
    
    def _is_close(self, a, b):
        """Robust fuzzy numerical comparison with 2% tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self):
        """Test momentum conservation and physics invariants."""
        logs = []
        success = True
        
        # 1. Check initial momentum calculation from Step 1
        if self.P is not None:
            expected_P = (self.m_A + self.m_person) * self.v_0 - self.m_B * self.v_0
            if not self._is_close(self.P, expected_P):
                logs.append(f"FAIL: Initial momentum mismatch. Got {self.P:.2f}, expected {expected_P:.2f}")
                success = False
            else:
                logs.append(f"PASS: Initial momentum {self.P:.2f} matches expected {expected_P:.2f}")
        else:
            logs.append("SKIP: Initial momentum P not provided")
        
        # 2. Symbolic momentum conservation check (Step 2)
        try:
            # Step 2: (m_A + m_person)*v_0 = m_A*v1 + m_person*v_person
            lhs_step2 = (self.m_A_sym + self.m_person_sym) * self.v_0_sym
            rhs_step2 = self.m_A_sym * self.v1_sym + self.m_person_sym * self.v_person_sym
            eq_step2 = sp.Eq(lhs_step2, rhs_step2)
            
            # Solve for v1 symbolically
            v1_expr = sp.solve(eq_step2, self.v1_sym)[0]
            
            # Numerically evaluate with given v_person
            if self.v_person is not None:
                v1_num = v1_expr.subs({
                    self.m_A_sym: self.m_A,
                    self.m_person_sym: self.m_person,
                    self.v_0_sym: self.v_0,
                    self.v_person_sym: self.v_person
                })
                v1_val = float(v1_num)
                logs.append(f"INFO: From Step 2, v1 = {v1_val:.2f} m/s")
                
                # Store for later use
                self.v1_computed = v1_val
            else:
                logs.append("SKIP: v_person missing, cannot compute v1")
        except Exception as e:
            logs.append(f"ERROR in Step 2 symbolic check: {str(e)}")
            success = False
        
        # 3. Symbolic momentum conservation check (Step 3)
        try:
            # Step 3: m_person*v_person - m_B*v_0 = (m_person + m_B)*v2
            lhs_step3 = self.m_person_sym * self.v_person_sym - self.m_B_sym * self.v_0_sym
            rhs_step3 = (self.m_person_sym + self.m_B_sym) * self.v2_sym
            eq_step3 = sp.Eq(lhs_step3, rhs_step3)
            
            # Solve for v2 symbolically
            v2_expr = sp.solve(eq_step3, self.v2_sym)[0]
            
            # Numerically evaluate with given v_person
            if self.v_person is not None:
                v2_num = v2_expr.subs({
                    self.m_B_sym: self.m_B,
                    self.m_person_sym: self.m_person,
                    self.v_0_sym: self.v_0,
                    self.v_person_sym: self.v_person
                })
                v2_val = float(v2_num)
                logs.append(f"INFO: From Step 3, v2 = {v2_val:.2f} m/s")
                
                # Store for later use
                self.v2_computed = v2_val
            else:
                logs.append("SKIP: v_person missing, cannot compute v2")
        except Exception as e:
            logs.append(f"ERROR in Step 3 symbolic check: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answer."""
        logs = []
        success = True
        
        # Need both v1 and v2 computed from previous test
        if not hasattr(self, 'v1_computed') or not hasattr(self, 'v2_computed'):
            logs.append("SKIP: v1 or v2 not computed. Run test_invariants first.")
            self.logs.extend(logs)
            return False, logs
        
        # Check condition for no collision: v1 >= v2 (Step 4)
        if self.v1_computed >= self.v2_computed - 0.02:  # Allow small tolerance
            logs.append(f"PASS: No collision condition satisfied: v1={self.v1_computed:.2f} >= v2={self.v2_computed:.2f}")
            
            # For minimum v_person, v1 should be approximately equal to v2
            if self._is_close(self.v1_computed, self.v2_computed):
                logs.append("PASS: v1 ≈ v2, indicating minimum v_person condition")
            else:
                logs.append(f"INFO: v1 > v2 by {self.v1_computed - self.v2_computed:.2f} m/s")
        else:
            logs.append(f"FAIL: Collision condition violated: v1={self.v1_computed:.2f} < v2={self.v2_computed:.2f}")
            success = False
        
        # Check final answer against target
        target_v_person = 2.9
        if self.v_person is not None:
            if self._is_close(self.v_person, target_v_person):
                logs.append(f"PASS: v_person={self.v_person:.2f} matches target {target_v_person:.2f} m/s")
            else:
                logs.append(f"FAIL: v_person={self.v_person:.2f} differs from target {target_v_person:.2f} m/s")
                success = False
        else:
            logs.append("SKIP: v_person not provided for comparison")
        
        self.logs.extend(logs)
        return success, logs
    
    def test_monotonicity(self):
        """Test that perturbing mass changes v_person in expected direction."""
        logs = []
        
        # Original parameters
        m_person_orig = self.m_person
        
        # Perturb mass by +10%
        m_person_perturbed = m_person_orig * 1.10  # 44.0 kg
        
        # Solve for v_person using the condition v1 = v2
        # From Steps 2 and 3 with v1 = v2 = v_eq:
        # (m_A + m_person)*v_0 - m_person*v_person = m_A * v_eq
        # m_person*v_person - m_B*v_0 = (m_person + m_B) * v_eq
        
        # Eliminate v_eq to get:
        # v_person = [(m_A + m_person)*v_0*(m_person + m_B) + m_A*m_B*v_0] / [m_person*(m_A + m_B + m_person)]
        
        try:
            # Calculate v_person for original mass
            num_orig = (self.m_A + m_person_orig) * self.v_0 * (m_person_orig + self.m_B) + self.m_A * self.m_B * self.v_0
            den_orig = m_person_orig * (self.m_A + self.m_B + m_person_orig)
            v_person_orig_calc = num_orig / den_orig
            
            # Calculate v_person for perturbed mass
            num_pert = (self.m_A + m_person_perturbed) * self.v_0 * (m_person_perturbed + self.m_B) + self.m_A * self.m_B * self.v_0
            den_pert = m_person_perturbed * (self.m_A + self.m_B + m_person_perturbed)
            v_person_pert_calc = num_pert / den_pert
            
            logs.append(f"INFO: Original mass {m_person_orig:.1f} kg → v_person = {v_person_orig_calc:.3f} m/s")
            logs.append(f"INFO: Perturbed mass {m_person_perturbed:.1f} kg → v_person = {v_person_pert_calc:.3f} m/s")
            
            # Check trend: increasing mass should decrease required v_person
            if v_person_pert_calc < v_person_orig_calc:
                logs.append(f"PASS: Monotonicity trend correct: increasing mass decreases required v_person")
                success = True
            else:
                logs.append(f"FAIL: Unexpected monotonicity trend")
                success = False
                
            # Compare with provided v_person if available
            if self.v_person is not None:
                if self._is_close(self.v_person, v_person_orig_calc):
                    logs.append(f"PASS: Provided v_person matches calculated value")
                else:
                    logs.append(f"WARN: Provided v_person {self.v_person:.3f} differs from calculated {v_person_orig_calc:.3f}")
        
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            success = False
        
        self.logs.extend(logs)
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        try:
            # Test 1: Invariants
            success1, logs1 = self.test_invariants()
            all_logs.extend(["=== Invariant Tests ==="] + logs1)
            if not success1:
                overall_success = False
            
            # Test 2: Trace Consistency
            success2, logs2 = self.test_trace_consistency()
            all_logs.extend(["", "=== Trace Consistency Tests ==="] + logs2)
            if not success2:
                overall_success = False
            
            # Test 3: Monotonicity
            success3, logs3 = self.test_monotonicity()
            all_logs.extend(["", "=== Monotonicity Tests ==="] + logs3)
            if not success3:
                overall_success = False
            
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR in test suite: {str(e)}")
            all_logs.append(traceback.format_exc())
            overall_success = False
        
        return overall_success, all_logs

# Example usage stub for the sandbox environment
if __name__ == "__main__":
    # This would be replaced by the actual solution instance in the sandbox
    class MockSolution:
        def __init__(self):
            self.results = {
                'P': 80.0,  # (30+40)*2 - 30*2 = 80
                'v_person': 2.9
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    print("\n".join(logs))
    print(f"\nOverall success: {success}")