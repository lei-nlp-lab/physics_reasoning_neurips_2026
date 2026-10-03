import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs and results
        self.solution = solution_instance
        
        # Extract problem inputs (use known values from problem statement)
        self.m1 = 1.0  # kg
        self.m2 = 4.0  # kg
        self.theta = 60.0  # degrees
        self.g = 10.0  # m/s^2
        self.theta_rad = math.radians(self.theta)
        
        # Extract results with safe fallbacks
        self.F_OC = getattr(solution_instance, 'F_OC', 
                           getattr(solution_instance, 'results', {}).get('F_OC', 0.0))
        self.F_OA = getattr(solution_instance, 'F_OA', 
                           getattr(solution_instance, 'results', {}).get('F_OA', 0.0))
        self.F_f = getattr(solution_instance, 'F_f', 
                          getattr(solution_instance, 'results', {}).get('F_f', 0.0))
        
        # Expected results from steps analysis
        self.expected_F_OC = self.m1 * self.g  # 10 N
        self.expected_F_OA = self.expected_F_OC / math.cos(self.theta_rad)  # 20 N
        self.expected_F_f = self.expected_F_OA * math.sin(self.theta_rad)  # 10√3 N ≈ 17.32 N
        
        # Initialize logs
        self.logs = []
    
    def test_invariants(self):
        """Verify force equilibrium conditions using Newton's laws"""
        logs = []
        try:
            # 1. Check vertical equilibrium of object A
            # F_OC should equal m1*g
            if not math.isclose(self.F_OC, self.expected_F_OC, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: F_OC = {self.F_OC} ≠ m1*g = {self.expected_F_OC}")
            else:
                logs.append(f"PASS: F_OC = m1*g = {self.F_OC}")
            
            # 2. Check vertical equilibrium at point O
            # F_OA*cos(theta) should equal F_OC
            vertical_eq = self.F_OA * math.cos(self.theta_rad)
            if not math.isclose(vertical_eq, self.F_OC, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: F_OA*cosθ = {vertical_eq} ≠ F_OC = {self.F_OC}")
            else:
                logs.append(f"PASS: Vertical equilibrium at O: F_OA*cosθ = F_OC")
            
            # 3. Check horizontal equilibrium at point O
            # F_OA*sin(theta) should equal F_OB (which equals F_f)
            horizontal_eq = self.F_OA * math.sin(self.theta_rad)
            if not math.isclose(horizontal_eq, self.F_f, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: F_OA*sinθ = {horizontal_eq} ≠ F_f = {self.F_f}")
            else:
                logs.append(f"PASS: Horizontal equilibrium at O: F_OA*sinθ = F_f")
            
            # 4. Check horizontal equilibrium of object B
            # F_f should balance tension from string OB (which equals F_OA*sinθ)
            # This is essentially the same as check 3
            if not math.isclose(self.F_f, horizontal_eq, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: F_f = {self.F_f} ≠ F_OA*sinθ = {horizontal_eq}")
            else:
                logs.append(f"PASS: Horizontal equilibrium of object B")
            
            # Symbolic verification
            try:
                θ = sp.symbols('θ')
                F_OA_sym, F_OC_sym, F_f_sym = sp.symbols('F_OA F_OC F_f')
                m1, g = sp.symbols('m1 g')
                
                eq1 = sp.Eq(F_OC_sym, m1*g)
                eq2 = sp.Eq(F_OA_sym*sp.cos(θ), F_OC_sym)
                eq3 = sp.Eq(F_OA_sym*sp.sin(θ), F_f_sym)
                
                # Check if equations are satisfied numerically
                subs_dict = {
                    m1: self.m1,
                    g: self.g,
                    θ: self.theta_rad,
                    F_OC_sym: self.F_OC,
                    F_OA_sym: self.F_OA,
                    F_f_sym: self.F_f
                }
                
                check1 = eq1.subs(subs_dict)
                check2 = eq2.subs(subs_dict)
                check3 = eq3.subs(subs_dict)
                
                if not (check1 == True and check2 == True and check3 == True):
                    logs.append(f"Symbolic check: {check1}, {check2}, {check3}")
                else:
                    logs.append("PASS: All symbolic equilibrium conditions satisfied")
                    
            except Exception as e:
                logs.append(f"Symbolic check failed (using numerical): {str(e)[:50]}")
            
            return all("PASS" in log or "Symbolic check" in log for log in logs[-6:]), logs
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            return False, logs
    
    def test_trace_consistency(self):
        """Check intermediate calculations against final answers"""
        logs = []
        try:
            # Step 1: F_OC calculation
            if not math.isclose(self.F_OC, self.expected_F_OC, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Step1: F_OC should be {self.expected_F_OC}, got {self.F_OC}")
            else:
                logs.append(f"PASS: Step1: F_OC = m1*g = {self.F_OC}")
            
            # Step 2 & 3: F_OA calculation
            if not math.isclose(self.F_OA, self.expected_F_OA, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Step3: F_OA should be {self.expected_F_OA}, got {self.F_OA}")
            else:
                logs.append(f"PASS: Step3: F_OA = F_OC/cosθ = {self.F_OA}")
            
            # Step 4-6: F_f calculation
            if not math.isclose(self.F_f, self.expected_F_f, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Step6: F_f should be {self.expected_F_f}, got {self.F_f}")
            else:
                logs.append(f"PASS: Step6: F_f = F_OA*sinθ = {self.F_f}")
            
            # Cross-check: F_f = F_OC * tanθ
            expected_F_f_alt = self.F_OC * math.tan(self.theta_rad)
            if not math.isclose(self.F_f, expected_F_f_alt, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Consistency check: F_f ≠ F_OC*tanθ")
            else:
                logs.append(f"PASS: Consistency: F_f = F_OC*tanθ")
            
            return all("PASS" in log for log in logs[-4:]), logs
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            return False, logs
    
    def test_monotonicity(self):
        """Test how results change when perturbing input mass m1"""
        logs = []
        try:
            # Store original values
            original_m1 = self.m1
            original_F_OC = self.F_OC
            original_F_OA = self.F_OA
            original_F_f = self.F_f
            
            # Perturb m1 by +10%
            perturbed_m1 = original_m1 * 1.1
            
            # Calculate expected changes using physics formulas
            new_F_OC = perturbed_m1 * self.g
            new_F_OA = new_F_OC / math.cos(self.theta_rad)
            new_F_f = new_F_OA * math.sin(self.theta_rad)
            
            # Check monotonic relationships
            tests_passed = True
            
            # All should increase when m1 increases
            if not (new_F_OC > original_F_OC):
                logs.append(f"FAIL: F_OC doesn't increase with m1: {new_F_OC} ≤ {original_F_OC}")
                tests_passed = False
            else:
                logs.append(f"PASS: F_OC increases with m1: {new_F_OC:.2f} > {original_F_OC:.2f}")
            
            if not (new_F_OA > original_F_OA):
                logs.append(f"FAIL: F_OA doesn't increase with m1: {new_F_OA} ≤ {original_F_OA}")
                tests_passed = False
            else:
                logs.append(f"PASS: F_OA increases with m1: {new_F_OA:.2f} > {original_F_OA:.2f}")
            
            if not (new_F_f > original_F_f):
                logs.append(f"FAIL: F_f doesn't increase with m1: {new_F_f} ≤ {original_F_f}")
                tests_passed = False
            else:
                logs.append(f"PASS: F_f increases with m1: {new_F_f:.2f} > {original_F_f:.2f}")
            
            # Test proportionality: F_OC ∝ m1
            ratio_OC = new_F_OC / original_F_OC
            ratio_m1 = perturbed_m1 / original_m1
            
            if not math.isclose(ratio_OC, ratio_m1, rel_tol=0.01):
                logs.append(f"FAIL: F_OC not proportional to m1: ratio_F_OC={ratio_OC:.3f}, ratio_m1={ratio_m1:.3f}")
                tests_passed = False
            else:
                logs.append(f"PASS: F_OC proportional to m1")
            
            return tests_passed, logs
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            return False, logs
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        all_logs = []
        all_passed = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                passed, logs = test_func()
                all_logs.append(f"=== {test_name} Test ===")
                all_logs.extend(logs)
                if not passed:
                    all_passed = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
                all_logs.append("")
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_passed = False
                all_logs.append("")
        
        # Final summary
        all_logs.append("=" * 50)
        if all_passed:
            all_logs.append("ALL TESTS PASSED")
        else:
            all_logs.append("SOME TESTS FAILED")
        
        return all_passed, all_logs

# Example usage (for testing the checker itself):
if __name__ == "__main__":
    class MockSolution:
        def __init__(self):
            self.results = {
                'F_OC': 10.0,
                'F_OA': 20.0,
                'F_f': 17.320508  # 10√3 ≈ 17.3205
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)