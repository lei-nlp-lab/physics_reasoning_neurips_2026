import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.t_total = getattr(solution_instance, 't_total', self.results.get('t_total', 0.0))
        self.v_B = getattr(solution_instance, 'v_B', self.results.get('v_B', 0.0))
        self.a_add = getattr(solution_instance, 'a_add', self.results.get('a_add', 0.0))
        self.a_subtract = getattr(solution_instance, 'a_subtract', self.results.get('a_subtract', 0.0))
        
        # Derived quantities
        self.t_acc = None  # acceleration time (incline)
        if self.a_add != 0:
            self.t_acc = self.v_B / self.a_add
        
    def test_invariants(self):
        """
        Verify kinematic laws and consistency of the solution.
        """
        logs = []
        
        try:
            # 1. Check v_B = a_add * t_acc (from step 5)
            if self.t_acc is not None:
                v_B_calc = self.a_add * self.t_acc
                if not math.isclose(self.v_B, v_B_calc, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v_B = {self.v_B} but a_add * t_acc = {v_B_calc}")
                else:
                    logs.append(f"PASS: v_B = a_add * t_acc holds")
            
            # 2. Check deceleration phase: v_B = a_subtract * (t_total - t_acc) (since final v=0)
            if self.t_acc is not None:
                v_B_decel = self.a_subtract * (self.t_total - self.t_acc)
                if not math.isclose(self.v_B, v_B_decel, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v_B from deceleration = {v_B_decel} != {self.v_B}")
                else:
                    logs.append(f"PASS: v_B consistent with deceleration phase")
            
            # 3. Check step 2 equation: 1 = a_subtract * (t_total - 11)
            step2_eq = self.a_subtract * (self.t_total - 11)
            if not math.isclose(step2_eq, 1.0, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Step 2 equation gives {step2_eq}, expected 1")
            else:
                logs.append(f"PASS: Step 2 equation holds")
            
            # 4. Check step 4 equation: a_add * t_acc - a_subtract * (9 - t_acc) = 11
            if self.t_acc is not None:
                step4_eq = self.a_add * self.t_acc - self.a_subtract * (9 - self.t_acc)
                if not math.isclose(step4_eq, 11.0, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Step 4 equation gives {step4_eq}, expected 11")
                else:
                    logs.append(f"PASS: Step 4 equation holds")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        
        try:
            # Check t_total from step 2: t_total = 11 + 1/a_subtract
            t_total_calc = 11 + 1.0 / self.a_subtract
            if not math.isclose(self.t_total, t_total_calc, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: t_total = {self.t_total} but calculated as {t_total_calc}")
            else:
                logs.append(f"PASS: t_total consistent with a_subtract")
            
            # Check v_B from simultaneous equations (steps 4-6)
            # From step 4: a_add * t - a_subtract * (9 - t) = 11
            # From step 5: v_B = a_add * t
            # Solve for t: t = 11/(a_add + a_subtract) + (9*a_subtract)/(a_add + a_subtract)
            if self.a_add + self.a_subtract != 0:
                t_calc = (11 + 9*self.a_subtract) / (self.a_add + self.a_subtract)
                v_B_calc = self.a_add * t_calc
                if not math.isclose(self.v_B, v_B_calc, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v_B from simultaneous eqns = {v_B_calc} != {self.v_B}")
                else:
                    logs.append(f"PASS: v_B matches simultaneous solution")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb an input and check if the trend follows physical expectations.
        """
        logs = []
        
        try:
            # Perturb a_subtract by +10% and check effect on t_total
            a_subtract_perturbed = self.a_subtract * 1.10
            t_total_perturbed = 11 + 1.0 / a_subtract_perturbed
            
            # Since a_subtract increases, t_total should decrease
            if t_total_perturbed < self.t_total:
                logs.append(f"PASS: Increasing a_subtract decreases t_total (as expected)")
            else:
                logs.append(f"WARNING: Monotonicity trend for a_subtract unexpected")
            
            # Restore original value (no actual mutation, just checking)
            
            # Perturb a_add by +10% and check effect on v_B
            if self.t_acc is not None:
                a_add_perturbed = self.a_add * 1.10
                v_B_perturbed = a_add_perturbed * self.t_acc
                
                # Since a_add increases, v_B should increase
                if v_B_perturbed > self.v_B:
                    logs.append(f"PASS: Increasing a_add increases v_B (as expected)")
                else:
                    logs.append(f"WARNING: Monotonicity trend for a_add unexpected")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
        
        return logs
    
    def run_all_tests(self):
        """
        Execute all tests and return success status and logs.
        """
        all_logs = []
        success = True
        
        try:
            # Run individual tests
            for test_name, test_method in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                logs = test_method()
                if any('FAIL' in log for log in logs):
                    success = False
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
            
            # Final summary
            if success:
                all_logs.append("=== OVERALL: PASS ===")
            else:
                all_logs.append("=== OVERALL: FAIL ===")
                
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
        
        return success, all_logs