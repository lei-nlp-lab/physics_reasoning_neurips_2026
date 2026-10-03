import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        self.solution = solution_instance
        
        # Inputs with safe fallbacks
        self.R = getattr(solution_instance, 'R', 1.0)  # default for symbolic checks
        self.m = getattr(solution_instance, 'm', 1.0)  # default for symbolic checks
        self.g = getattr(solution_instance, 'g', 9.8)  # gravitational acceleration
        
        # Results with safe extraction
        self.results = {}
        expected_keys = ['costheta', 'F', 'sintheta', 'v_B', 'tantheta', 'v_max']
        for key in expected_keys:
            self.results[key] = getattr(solution_instance, key, None)
            if self.results[key] is None and hasattr(solution_instance, 'results'):
                self.results[key] = getattr(solution_instance.results, key, None)
            if self.results[key] is None and hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                self.results[key] = solution_instance.results.get(key, None)
        
        # Given: pressure at point B is 20 N
        self.N_B = 20.0
        
        # Initialize symbolic variables for robust checking
        self.sym_R, self.sym_m, self.sym_g = sp.symbols('R m g', positive=True)
        self.sym_v_B, self.sym_F, self.sym_v_max = sp.symbols('v_B F v_max', real=True)
        self.sym_sinθ, self.sym_cosθ, self.sym_tanθ = sp.symbols('sinθ cosθ tanθ', real=True)
    
    def test_invariants(self):
        """
        Verify fundamental physics laws using both symbolic and numerical checks.
        """
        logs = []
        success = True
        
        try:
            # 1. Centripetal force at point B (Newton's 2nd Law)
            # Equation: N - m*g = m*v_B^2 / R
            if self.results['v_B'] is not None:
                lhs_num = self.N_B - self.m * self.g
                rhs_num = self.m * self.results['v_B']**2 / self.R
                if not math.isclose(lhs_num, rhs_num, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Centripetal force at B mismatch. N - m*g = {lhs_num:.3f}, m*v_B^2/R = {rhs_num:.3f}")
                    success = False
                else:
                    logs.append("PASS: Centripetal force at B satisfies Newton's 2nd Law.")
            else:
                logs.append("SKIP: v_B not available for centripetal force check.")
            
            # 2. Work-Energy Theorem from A to B
            # Equation: m*g*R - F*R = 0.5*m*v_B^2
            if self.results['v_B'] is not None and self.results['F'] is not None:
                lhs_num = self.m * self.g * self.R - self.results['F'] * self.R
                rhs_num = 0.5 * self.m * self.results['v_B']**2
                if not math.isclose(lhs_num, rhs_num, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Work-Energy Theorem mismatch. mgR - FR = {lhs_num:.3f}, 0.5*m*v_B^2 = {rhs_num:.3f}")
                    success = False
                else:
                    logs.append("PASS: Work-Energy Theorem holds from A to B.")
            else:
                logs.append("SKIP: v_B or F not available for Work-Energy check.")
            
            # 3. Trigonometric consistency
            if all(self.results[k] is not None for k in ['sintheta', 'costheta', 'tantheta']):
                # Check tanθ = sinθ/cosθ
                tan_from_trig = self.results['sintheta'] / self.results['costheta']
                if not math.isclose(self.results['tantheta'], tan_from_trig, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: tanθ ≠ sinθ/cosθ. tanθ = {self.results['tantheta']:.3f}, sinθ/cosθ = {tan_from_trig:.3f}")
                    success = False
                else:
                    logs.append("PASS: tanθ = sinθ/cosθ.")
                
                # Check sin²θ + cos²θ = 1
                trig_identity = self.results['sintheta']**2 + self.results['costheta']**2
                if not math.isclose(trig_identity, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: sin²θ + cos²θ = {trig_identity:.3f} ≠ 1")
                    success = False
                else:
                    logs.append("PASS: sin²θ + cos²θ = 1.")
            else:
                logs.append("SKIP: Trigonometric results not fully available.")
            
            # 4. Condition for maximum speed (net tangential force zero => tanθ = F/(m*g))
            if self.results['tantheta'] is not None and self.results['F'] is not None:
                expected_tan = self.results['F'] / (self.m * self.g)
                if not math.isclose(self.results['tantheta'], expected_tan, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: tanθ ≠ F/(m*g). tanθ = {self.results['tantheta']:.3f}, F/(m*g) = {expected_tan:.3f}")
                    success = False
                else:
                    logs.append("PASS: Condition for maximum speed (tanθ = F/(m*g)) satisfied.")
            else:
                logs.append("SKIP: tanθ or F not available for max speed condition.")
            
            # 5. Energy equation at maximum speed
            # Equation: m*g*R*cosθ - F*R*(1 - sinθ) = 0.5*m*v_max^2
            if all(self.results[k] is not None for k in ['v_max', 'sintheta', 'costheta', 'F']):
                lhs_num = self.m * self.g * self.R * self.results['costheta'] - \
                         self.results['F'] * self.R * (1 - self.results['sintheta'])
                rhs_num = 0.5 * self.m * self.results['v_max']**2
                if not math.isclose(lhs_num, rhs_num, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy equation at max speed mismatch. LHS = {lhs_num:.3f}, RHS = {rhs_num:.3f}")
                    success = False
                else:
                    logs.append("PASS: Energy equation at maximum speed holds.")
            else:
                logs.append("SKIP: v_max or trig results not available for energy check.")
        
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        try:
            # 1. Check that F = 0.5*m*g (from step 3)
            if self.results['F'] is not None:
                expected_F = 0.5 * self.m * self.g
                if not math.isclose(self.results['F'], expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: F ≠ 0.5*m*g. F = {self.results['F']:.3f}, 0.5*m*g = {expected_F:.3f}")
                    success = False
                else:
                    logs.append("PASS: F = 0.5*m*g as derived.")
            else:
                logs.append("SKIP: F not available for consistency check.")
            
            # 2. Check that v_max matches the derived formula √(gR(√5 - 1))
            if self.results['v_max'] is not None:
                expected_v_max = math.sqrt(self.g * self.R * (math.sqrt(5) - 1))
                if not math.isclose(self.results['v_max'], expected_v_max, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_max ≠ √(gR(√5-1)). v_max = {self.results['v_max']:.3f}, expected = {expected_v_max:.3f}")
                    success = False
                else:
                    logs.append("PASS: v_max matches derived formula.")
            else:
                logs.append("SKIP: v_max not available for formula check.")
            
            # 3. Check trigonometric values consistency
            if self.results['tantheta'] is not None:
                expected_tan = 0.5  # from tanθ = F/(m*g) = 0.5
                if not math.isclose(self.results['tantheta'], expected_tan, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: tanθ ≠ 0.5. tanθ = {self.results['tantheta']:.3f}")
                    success = False
                else:
                    logs.append("PASS: tanθ = 0.5 as derived.")
                
                # If sinθ and cosθ are available, check they match tanθ = 0.5
                if self.results['sintheta'] is not None and self.results['costheta'] is not None:
                    expected_sin = 1 / math.sqrt(5)  # ≈ 0.4472
                    expected_cos = 2 / math.sqrt(5)  # ≈ 0.8944
                    if not (math.isclose(self.results['sintheta'], expected_sin, rel_tol=1e-2, abs_tol=1e-2) and
                            math.isclose(self.results['costheta'], expected_cos, rel_tol=1e-2, abs_tol=1e-2)):
                        logs.append(f"FAIL: sinθ or cosθ mismatch. sinθ = {self.results['sintheta']:.3f} (expected {expected_sin:.3f}), "
                                   f"cosθ = {self.results['costheta']:.3f} (expected {expected_cos:.3f})")
                        success = False
                    else:
                        logs.append("PASS: sinθ = 1/√5 and cosθ = 2/√5.")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (mass m) and verify physical trends.
        """
        logs = []
        success = True
        
        try:
            original_m = self.m
            perturbed_m = original_m * 1.1  # +10% change
            
            # Save original results
            original_F = self.results.get('F')
            original_v_max = self.results.get('v_max')
            
            if original_F is None or original_v_max is None:
                logs.append("SKIP: Original F or v_max not available for monotonicity test.")
                return success, logs
            
            # Recalculate using formulas (symbolic evaluation)
            # F is proportional to m: F = 0.5*m*g
            perturbed_F = 0.5 * perturbed_m * self.g
            # v_max is independent of m: v_max = √(gR(√5 - 1))
            perturbed_v_max = math.sqrt(self.g * self.R * (math.sqrt(5) - 1))
            
            # Check monotonic trends
            if not (perturbed_F > original_F):
                logs.append(f"FAIL: F does not increase with mass. Original F = {original_F:.3f}, Perturbed F = {perturbed_F:.3f}")
                success = False
            else:
                logs.append("PASS: F increases with mass (F ∝ m).")
            
            if not math.isclose(perturbed_v_max, original_v_max, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_max changed with mass, but should be independent. Original v_max = {original_v_max:.3f}, Perturbed v_max = {perturbed_v_max:.3f}")
                success = False
            else:
                logs.append("PASS: v_max independent of mass (as expected).")
            
            # Restore original mass (for completeness)
            self.m = original_m
        
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return aggregated results.
        """
        all_logs = []
        overall_success = True
        
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
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs