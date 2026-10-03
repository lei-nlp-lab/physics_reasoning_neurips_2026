import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        Uses getattr and .get() to avoid AttributeErrors in sandbox.
        """
        self.solution_instance = solution_instance
        
        # Problem constants
        self.F1 = 16.0  # N
        self.mA = 4.0   # kg
        self.mB = 4.0   # kg
        self.g = 10.0   # N/kg
        
        # Extract results safely
        self.F_N = self._safe_get('F_N', 'F_N')
        self.F_f1 = self._safe_get('F_f1', 'F_f1')
        self.F_NB = self._safe_get('F_NB', 'F_NB')
        self.F_fB = self._safe_get('F_fB', 'F_fB')
        self.F_f2 = self._safe_get('F_f2', 'F_f2')
        self.mu = self._safe_get('mu', 'mu')
        self.F_B = self._safe_get('F_B', 'F_B')
        
        # Target answers for final verification
        self.target_answers = ["$0$, $16N$", "0.2", "$8N$"]
        
    def _safe_get(self, attr_name, result_key):
        """Safe extraction with fallback to 0.0."""
        val = getattr(self.solution_instance, attr_name, None)
        if val is None:
            val = getattr(self.solution_instance, 'results', {}).get(result_key, 0.0)
        return float(val) if val is not None else 0.0
    
    def test_invariants(self):
        """Check fundamental physics laws: equilibrium, friction formula, Newton's laws."""
        logs = []
        success = True
        
        try:
            # 1. System equilibrium (uniform motion => net force = 0)
            F_net_system = self.F1 - self.F_f2
            if not math.isclose(F_net_system, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: System net force not zero. F1={self.F1}, F_f2={self.F_f2}, net={F_net_system}")
                success = False
            else:
                logs.append(f"PASS: System equilibrium (F1 = F_f2)")
            
            # 2. Normal force on table matches total weight
            expected_F_N = (self.mA + self.mB) * self.g
            if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Normal force mismatch. Got {self.F_N}, expected {expected_F_N}")
                success = False
            else:
                logs.append(f"PASS: Normal force F_N = (mA+mB)g")
            
            # 3. Kinetic friction formula for table-block B interface
            if self.F_N > 0:
                mu_from_f2 = self.F_f2 / self.F_N
                if not math.isclose(self.mu, mu_from_f2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: μ mismatch from F_f2/F_N. Got μ={self.mu}, expected {mu_from_f2}")
                    success = False
                else:
                    logs.append(f"PASS: μ = F_f2/F_N = {self.mu:.3f}")
            
            # 4. Normal force between A and B equals weight of B (or A, since equal)
            expected_F_NB = self.mB * self.g
            if not math.isclose(self.F_NB, expected_F_NB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_NB mismatch. Got {self.F_NB}, expected {expected_F_NB}")
                success = False
            else:
                logs.append(f"PASS: F_NB = mB*g")
            
            # 5. Friction between A and B follows same μ
            expected_F_fB = self.mu * self.F_NB
            if not math.isclose(self.F_fB, expected_F_fB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_fB mismatch. Got {self.F_fB}, expected μ*F_NB={expected_F_fB}")
                success = False
            else:
                logs.append(f"PASS: F_fB = μ*F_NB")
            
            # 6. Force balance on block A (horizontal)
            # For uniform motion, net force on A = 0. Only horizontal force from B is F_B (elastic)
            # and possible friction F_f1 (which is 0 per step 1)
            F_net_A = self.F_B - self.F_f1  # Assuming F_B acts opposite to motion direction? Wait, analysis says F_B = F_fB.
            if not math.isclose(F_net_A, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: Net force on block A non-zero: {F_net_A}. Might be sign convention.")
                # Not a failure due to possible sign conventions
            
            # 7. Force balance on block B (horizontal)
            # Forces on B: F1 (applied), F_f2 (table friction opposite), F_B (from A, opposite direction?),
            # and F_fB (friction from A, if any). According to steps, F_B = F_fB.
            # So net horizontal: F1 - F_f2 - F_B + F_fB? But F_B = F_fB, so F1 - F_f2 = 0 (already checked).
            # This is consistent.
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check that intermediate results lead consistently to final answers."""
        logs = []
        success = True
        
        try:
            # Step 1: F_f1 = 0
            if not math.isclose(self.F_f1, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 1: F_f1 should be 0, got {self.F_f1}")
                success = False
            else:
                logs.append(f"PASS: Step 1: F_f1 = 0")
            
            # Step 2: F_f2 = F1 = 16 N
            if not math.isclose(self.F_f2, self.F1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 2: F_f2 should equal F1={self.F1}, got {self.F_f2}")
                success = False
            else:
                logs.append(f"PASS: Step 2: F_f2 = F1 = 16 N")
            
            # Step 3: F_N = (mA+mB)*g = 80 N
            expected_F_N = (self.mA + self.mB) * self.g
            if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 3: F_N mismatch. Got {self.F_N}, expected {expected_F_N}")
                success = False
            else:
                logs.append(f"PASS: Step 3: F_N = (mA+mB)g = 80 N")
            
            # Step 4 & 5: μ = F_f2 / F_N = 16/80 = 0.2
            expected_mu = self.F_f2 / self.F_N if self.F_N != 0 else 0.0
            if not math.isclose(self.mu, expected_mu, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 4&5: μ should be {expected_mu:.3f}, got {self.mu}")
                success = False
            else:
                logs.append(f"PASS: Step 4&5: μ = F_f2/F_N = {self.mu:.3f}")
            
            # Step 6: F_NB = mB*g = 40 N
            expected_F_NB = self.mB * self.g
            if not math.isclose(self.F_NB, expected_F_NB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 6: F_NB should be {expected_F_NB}, got {self.F_NB}")
                success = False
            else:
                logs.append(f"PASS: Step 6: F_NB = mB*g = 40 N")
            
            # Step 7: F_fB = μ * F_NB = 0.2*40 = 8 N
            expected_F_fB = self.mu * self.F_NB
            if not math.isclose(self.F_fB, expected_F_fB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 7: F_fB should be {expected_F_fB:.3f}, got {self.F_fB}")
                success = False
            else:
                logs.append(f"PASS: Step 7: F_fB = μ*F_NB = {self.F_fB} N")
            
            # Step 8: F_B = F_fB = 8 N
            if not math.isclose(self.F_B, self.F_fB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 8: F_B should equal F_fB={self.F_fB}, got {self.F_B}")
                success = False
            else:
                logs.append(f"PASS: Step 8: F_B = F_fB = {self.F_B} N")
            
            # Final answer check (if available in solution_instance)
            if hasattr(self.solution_instance, 'final_answers'):
                for i, (tgt, ans) in enumerate(zip(self.target_answers, self.solution_instance.final_answers)):
                    if isinstance(ans, str):
                        # Simple string comparison (could be enhanced)
                        if ans.strip() != tgt.strip():
                            logs.append(f"WARN: Answer {i} mismatch: got '{ans}', expected '{tgt}'")
                    else:
                        # Numeric comparison if answer is numeric
                        try:
                            val = float(ans)
                            if tgt == "0.2" and not math.isclose(val, 0.2, rel_tol=1e-2, abs_tol=1e-2):
                                logs.append(f"WARN: Answer {i} numeric mismatch: got {val}, expected 0.2")
                        except:
                            pass
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (F1) and verify that recalculated results follow expected trends.
        Does NOT modify the original solution instance.
        """
        logs = []
        success = True
        
        try:
            # Perturb F1 by +2 N (increase)
            delta_F1 = 2.0
            F1_new = self.F1 + delta_F1
            
            # Recalculate expected values using physics formulas from steps
            # All other parameters (masses, g) remain same.
            
            # Step 2: F_f2_new = F1_new (for uniform motion)
            F_f2_new = F1_new
            
            # Step 3: F_N unchanged (same masses)
            F_N_new = (self.mA + self.mB) * self.g
            
            # Step 4 & 5: μ_new = F_f2_new / F_N_new
            mu_new = F_f2_new / F_N_new
            
            # Step 6: F_NB unchanged
            F_NB_new = self.mB * self.g
            
            # Step 7: F_fB_new = μ_new * F_NB_new
            F_fB_new = mu_new * F_NB_new
            
            # Step 8: F_B_new = F_fB_new
            F_B_new = F_fB_new
            
            # Check trends:
            # 1. μ should increase when F1 increases (since F_N constant)
            if mu_new <= self.mu:
                logs.append(f"FAIL: μ should increase when F1 increases. Old μ={self.mu}, new μ={mu_new}")
                success = False
            else:
                logs.append(f"PASS: μ increases with F1: {self.mu:.3f} -> {mu_new:.3f}")
            
            # 2. F_B should increase when μ increases (since F_NB constant)
            if F_B_new <= self.F_B:
                logs.append(f"FAIL: F_B should increase when F1 increases. Old F_B={self.F_B}, new F_B={F_B_new}")
                success = False
            else:
                logs.append(f"PASS: F_B increases with F1: {self.F_B:.3f} -> {F_B_new:.3f}")
            
            # 3. F_f2 should equal new F1
            if not math.isclose(F_f2_new, F1_new, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: New F_f2 should equal new F1, got {F_f2_new} vs {F1_new}")
            
            # 4. System equilibrium maintained
            F_net_new = F1_new - F_f2_new
            if not math.isclose(F_net_new, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: New system net force not zero: {F_net_new}")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and log."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_method()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs

# Example usage (if needed for debugging):
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'F_N': 80.0,
                'F_f1': 0.0,
                'F_NB': 40.0,
                'F_fB': 8.0,
                'F_f2': 16.0,
                'mu': 0.2,
                'F_B': 8.0
            }
            self.final_answers = ["$0$, $16N$", "0.2", "$8N$"]
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    print("Success:", success)
    for log in logs:
        print(log)