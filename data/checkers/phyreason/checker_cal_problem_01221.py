import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Extract inputs
        self.m_A = getattr(solution_instance, 'm_A', 2.0)
        self.m_B = getattr(solution_instance, 'm_B', 3.0)
        self.k = getattr(solution_instance, 'k', 400.0)
        self.F = getattr(solution_instance, 'F', 60.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.F_1 = results.get('F_1', 0.0)
        self.a = results.get('a', 0.0)
        self.x_2 = results.get('x_2', 0.0)
        self.s = results.get('s', 0.0)
        self.x_1 = results.get('x_1', 0.0)
        
        self.logs = []
        
    def log_test(self, test_name, passed, message, details=""):
        """
        Log test results with details.
        """
        status = "PASS" if passed else "FAIL"
        entry = f"{test_name}: {status} - {message}"
        if details:
            entry += f" | {details}"
        self.logs.append(entry)
        return passed
    
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Fuzzy numerical comparison with tolerance.
        """
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_step1_equilibrium(self):
        """
        Check step 1: F_1 = m_B * g when block B is about to leave ground.
        """
        expected = self.m_B * self.g
        passed = self.fuzzy_equal(self.F_1, expected)
        return self.log_test(
            "Step1_Equilibrium", 
            passed, 
            f"F_1 = {self.F_1} should equal m_B*g = {expected}",
            f"F_1={self.F_1}, expected={expected}"
        )
    
    def test_step2_newton_law(self):
        """
        Check step 2: a = (F - F_1 - m_A*g) / m_A
        """
        numerator = self.F - self.F_1 - self.m_A * self.g
        expected = numerator / self.m_A
        passed = self.fuzzy_equal(self.a, expected)
        return self.log_test(
            "Step2_NewtonLaw", 
            passed, 
            f"a = {self.a} should equal (F-F_1-m_A*g)/m_A = {expected}",
            f"a={self.a}, expected={expected}, numerator={numerator}"
        )
    
    def test_step3_initial_compression(self):
        """
        Check step 3: x_1 = (m_A * g) / k (initial spring compression)
        """
        expected = self.m_A * self.g / self.k
        passed = self.fuzzy_equal(self.x_1, expected)
        return self.log_test(
            "Step3_InitialCompression", 
            passed, 
            f"x_1 = {self.x_1} should equal m_A*g/k = {expected}",
            f"x_1={self.x_1}, expected={expected}"
        )
    
    def test_step4_spring_extension(self):
        """
        Check step 4: x_2 = (m_B * g) / k (spring extension when B leaves ground)
        """
        expected = self.m_B * self.g / self.k
        passed = self.fuzzy_equal(self.x_2, expected)
        return self.log_test(
            "Step4_SpringExtension", 
            passed, 
            f"x_2 = {self.x_2} should equal m_B*g/k = {expected}",
            f"x_2={self.x_2}, expected={expected}"
        )
    
    def test_step5_displacement(self):
        """
        Check step 5: s = x_1 + x_2
        """
        expected = self.x_1 + self.x_2
        passed = self.fuzzy_equal(self.s, expected)
        return self.log_test(
            "Step5_Displacement", 
            passed, 
            f"s = {self.s} should equal x_1+x_2 = {expected}",
            f"s={self.s}, expected={expected}"
        )
    
    def test_invariants(self):
        """
        Test fundamental physics invariants using symbolic and numerical methods.
        """
        passed_all = True
        
        # 1. Hooke's Law: F_1 = k * x_2 (spring force when B leaves ground)
        hooke_expected = self.k * self.x_2
        hooke_passed = self.fuzzy_equal(self.F_1, hooke_expected)
        passed_all &= hooke_passed
        self.log_test(
            "Invariant_HookesLaw", 
            hooke_passed, 
            f"F_1 = {self.F_1} should equal k*x_2 = {hooke_expected}",
            f"F_1={self.F_1}, k*x_2={hooke_expected}"
        )
        
        # 2. Newton's Second Law for block A at the critical moment
        net_force = self.F - self.F_1 - self.m_A * self.g
        newton_expected = net_force
        newton_actual = self.m_A * self.a
        newton_passed = self.fuzzy_equal(newton_actual, newton_expected)
        passed_all &= newton_passed
        self.log_test(
            "Invariant_NewtonsSecondLaw", 
            newton_passed, 
            f"m_A*a = {newton_actual} should equal F_net = {newton_expected}",
            f"m_A*a={newton_actual}, F-F_1-m_A*g={newton_expected}"
        )
        
        # 3. Initial equilibrium: m_A*g = k*x_1
        initial_equilibrium = self.m_A * self.g
        initial_actual = self.k * self.x_1
        initial_passed = self.fuzzy_equal(initial_actual, initial_equilibrium)
        passed_all &= initial_passed
        self.log_test(
            "Invariant_InitialEquilibrium", 
            initial_passed, 
            f"k*x_1 = {initial_actual} should equal m_A*g = {initial_equilibrium}",
            f"k*x_1={initial_actual}, m_A*g={initial_equilibrium}"
        )
        
        return passed_all
    
    def test_trace_consistency(self):
        """
        Cross-check that all intermediate results are consistent.
        """
        passed_all = True
        
        # Check that F_1 from step1 matches F_1 used in step2
        # (Already tested in invariants, but double-check)
        step1_F1 = self.m_B * self.g
        step2_F1_used = self.F_1
        consistency1 = self.fuzzy_equal(step1_F1, step2_F1_used)
        passed_all &= consistency1
        self.log_test(
            "Trace_Consistency_F1", 
            consistency1, 
            f"Step1 F1 ({step1_F1}) should match Step2 used F1 ({step2_F1_used})",
            f"step1_F1={step1_F1}, step2_F1_used={step2_F1_used}"
        )
        
        # Check that x_1 and x_2 from steps 3-4 sum to s
        sum_x = self.x_1 + self.x_2
        consistency2 = self.fuzzy_equal(sum_x, self.s)
        passed_all &= consistency2
        self.log_test(
            "Trace_Consistency_Displacement", 
            consistency2, 
            f"x_1+x_2 = {sum_x} should equal s = {self.s}",
            f"sum={sum_x}, s={self.s}"
        )
        
        # Verify that using x_2 in Hooke's law gives F_1
        hooke_check = self.k * self.x_2
        consistency3 = self.fuzzy_equal(hooke_check, self.F_1)
        passed_all &= consistency3
        self.log_test(
            "Trace_Consistency_Hooke", 
            consistency3, 
            f"k*x_2 = {hooke_check} should equal F_1 = {self.F_1}",
            f"k*x_2={hooke_check}, F_1={self.F_1}"
        )
        
        return passed_all
    
    def test_monotonicity(self):
        """
        Perturb input parameters and check expected trends.
        """
        passed_all = True
        
        # Store original values
        orig_m_A = self.m_A
        orig_results = {
            'F_1': self.F_1,
            'a': self.a,
            'x_1': self.x_1,
            'x_2': self.x_2,
            's': self.s
        }
        
        # Test 1: Increase m_A by 10%
        perturb_m_A = orig_m_A * 1.1
        self.m_A = perturb_m_A
        
        # Recalculate expected changes using formulas
        new_x1 = self.m_A * self.g / self.k
        new_F1 = self.m_B * self.g  # unchanged by m_A
        new_a = (self.F - new_F1 - self.m_A * self.g) / self.m_A
        new_x2 = self.m_B * self.g / self.k  # unchanged by m_A
        new_s = new_x1 + new_x2
        
        # Check trends
        # x1 should increase with m_A
        if not (new_x1 > orig_results['x_1']):
            passed_all = False
            self.log_test(
                "Monotonicity_x1_vs_mA", False,
                f"Increasing m_A should increase x_1",
                f"original x1={orig_results['x_1']}, new x1={new_x1}"
            )
        else:
            self.log_test(
                "Monotonicity_x1_vs_mA", True,
                f"x_1 increases with m_A as expected",
                f"original x1={orig_results['x_1']}, new x1={new_x1}"
            )
        
        # a should decrease with m_A (since denominator increases and numerator decreases)
        if not (new_a < orig_results['a']):
            passed_all = False
            self.log_test(
                "Monotonicity_a_vs_mA", False,
                f"Increasing m_A should decrease acceleration a",
                f"original a={orig_results['a']}, new a={new_a}"
            )
        else:
            self.log_test(
                "Monotonicity_a_vs_mA", True,
                f"a decreases with m_A as expected",
                f"original a={orig_results['a']}, new a={new_a}"
            )
        
        # Restore original values
        self.m_A = orig_m_A
        
        # Test 2: Increase k by 20%
        orig_k = self.k
        perturb_k = orig_k * 1.2
        self.k = perturb_k
        
        # Recalculate
        new_x1 = self.m_A * self.g / self.k
        new_x2 = self.m_B * self.g / self.k
        new_s = new_x1 + new_x2
        
        # Both x1 and x2 should decrease with increasing k
        if not (new_x1 < orig_results['x_1']):
            passed_all = False
            self.log_test(
                "Monotonicity_x1_vs_k", False,
                f"Increasing k should decrease x_1",
                f"original x1={orig_results['x_1']}, new x1={new_x1}"
            )
        else:
            self.log_test(
                "Monotonicity_x1_vs_k", True,
                f"x_1 decreases with k as expected",
                f"original x1={orig_results['x_1']}, new x1={new_x1}"
            )
        
        if not (new_x2 < orig_results['x_2']):
            passed_all = False
            self.log_test(
                "Monotonicity_x2_vs_k", False,
                f"Increasing k should decrease x_2",
                f"original x2={orig_results['x_2']}, new x2={new_x2}"
            )
        else:
            self.log_test(
                "Monotonicity_x2_vs_k", True,
                f"x_2 decreases with k as expected",
                f"original x2={orig_results['x_2']}, new x2={new_x2}"
            )
        
        # Restore original k
        self.k = orig_k
        
        return passed_all
    
    def run_all_tests(self):
        """
        Execute all tests and return success status with logs.
        """
        self.logs = []  # Reset logs
        
        try:
            # Run individual step tests
            step_tests = [
                self.test_step1_equilibrium(),
                self.test_step2_newton_law(),
                self.test_step3_initial_compression(),
                self.test_step4_spring_extension(),
                self.test_step5_displacement()
            ]
            
            # Run invariant tests
            invariant_ok = self.test_invariants()
            
            # Run trace consistency
            trace_ok = self.test_trace_consistency()
            
            # Run monotonicity tests
            monotonicity_ok = self.test_monotonicity()
            
            # Overall success
            all_passed = all(step_tests) and invariant_ok and trace_ok and monotonicity_ok
            
            # Final summary log
            self.logs.append("="*50)
            self.logs.append(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
            
            return all_passed, self.logs
            
        except Exception as e:
            error_msg = f"Exception during testing: {str(e)}"
            self.logs.append(error_msg)
            self.logs.append(traceback.format_exc())
            return False, self.logs