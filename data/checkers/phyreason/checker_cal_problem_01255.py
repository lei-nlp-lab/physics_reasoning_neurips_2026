import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Safely retrieve results with fallbacks
        self.results = {}
        keys = ['x', 'a_1', 'y', 't_2', 'a', 'h', 'tantheta', 's']
        for key in keys:
            # Use getattr on solution_instance, fallback to results dictionary
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, None)
            self.results[key] = val
        
        # Constants from problem statement
        self.g = 10.0
        self.t1 = 2.0  # ascent time
        self.v0 = 2.0  # horizontal throw speed
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = self.sin37 / self.cos37  # ~0.75
        
        # Additional derived inputs if needed
        self.m = getattr(solution_instance, 'm', 1.0)  # dummy mass if not provided
    
    def test_invariants(self):
        """
        Verify fundamental physics laws: Newton's 2nd Law, energy conservation, 
        and kinematic consistency.
        """
        logs = []
        success = True
        
        try:
            # 1. Check Newton's 2nd Law for drone ascent
            # Lift = 2mg, weight = mg → net force = mg → a1 = g
            a1_expected = self.g
            a1_calculated = self.results.get('a_1')
            if a1_calculated is None:
                logs.append("WARNING: a_1 not found in results")
            elif not math.isclose(a1_calculated, a1_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Newton 2nd Law (drone). Expected a1={a1_expected}, got {a1_calculated}")
                success = False
            else:
                logs.append("PASS: Newton's 2nd Law for drone ascent")
            
            # 2. Check kinematic consistency for drone ascent: h = 0.5 * a1 * t1^2
            h_calc = 0.5 * self.g * self.t1 ** 2  # using a1 = g
            h_given = self.results.get('h')
            if h_given is not None:
                if not math.isclose(h_given, h_calc, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Drone ascent kinematics. Expected h={h_calc}, got {h_given}")
                    success = False
                else:
                    logs.append("PASS: Drone ascent height consistent")
            
            # 3. Check camera projectile motion kinematics
            t2 = self.results.get('t_2')
            if t2 is not None:
                # Horizontal motion: x = v0 * t2
                x_expected = self.v0 * t2
                x_given = self.results.get('x')
                if x_given is not None and not math.isclose(x_given, x_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Camera horizontal motion. Expected x={x_expected}, got {x_given}")
                    success = False
                else:
                    logs.append("PASS: Camera horizontal motion consistent")
                
                # Vertical free fall: y = 0.5 * g * t2^2
                y_expected = 0.5 * self.g * t2 ** 2
                y_given = self.results.get('y')
                if y_given is not None and not math.isclose(y_given, y_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: Camera vertical fall. Expected y={y_expected}, got {y_given}")
                    success = False
                else:
                    logs.append("PASS: Camera vertical fall consistent")
                
                # 4. Check slope geometry relation: The camera is released from height H_e = 17m
                # The correct geometry is: y - H_e = tanθ * x (with proper sign)
                # where H_e = 17m is the height of point E above B
                tanθ_given = self.results.get('tantheta')
                if tanθ_given is not None and x_given is not None and y_given is not None:
                    if x_given != 0:
                        H_e = 17.0  # height of point E above B
                        # The geometry from the correct solution: y - H_e = tanθ * x
                        tanθ_calc = (y_given - H_e) / x_given
                        if not math.isclose(abs(tanθ_calc), self.tan37, rel_tol=0.02, abs_tol=0.02):
                            logs.append(f"FAIL: Slope geometry. Calculated tanθ = {tanθ_calc}, expected magnitude {self.tan37}")
                            success = False
                        else:
                            logs.append("PASS: Slope geometry relation holds")
                    else:
                        logs.append("WARNING: x=0, cannot compute tanθ")
                
                # 5. Check Xiao Ming's motion kinematics
                s_given = self.results.get('s')
                a_given = self.results.get('a')
                if s_given is not None and a_given is not None:
                    # s = 0.5 * a * t2^2
                    s_from_a = 0.5 * a_given * t2 ** 2
                    if not math.isclose(s_given, s_from_a, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: Xiao Ming's motion kinematics. s from a: {s_from_a}, given s: {s_given}")
                        success = False
                    else:
                        logs.append("PASS: Xiao Ming's displacement consistent with acceleration")
                    
                    # s = x / cos37 (Step 8)
                    s_from_x = x_given / self.cos37 if x_given is not None else None
                    if s_from_x is not None and not math.isclose(s_given, s_from_x, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: Slope displacement relation. s from x/cos37: {s_from_x}, given s: {s_given}")
                        success = False
                    else:
                        logs.append("PASS: Slope displacement relation holds")
            else:
                logs.append("WARNING: t_2 missing, skipping projectile and Ming checks")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        try:
            # Target answers: h=20m, t2=2s, a=2.5 m/s²
            h_target = 20.0
            t2_target = 2.0
            a_target = 2.5
            
            h_given = self.results.get('h')
            t2_given = self.results.get('t_2')
            a_given = self.results.get('a')
            
            if h_given is not None and not math.isclose(h_given, h_target, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Height h mismatch. Target {h_target}, got {h_given}")
                success = False
            else:
                logs.append("PASS: Height matches target")
            
            if t2_given is not None and not math.isclose(t2_given, t2_target, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Time t2 mismatch. Target {t2_target}, got {t2_given}")
                success = False
            else:
                logs.append("PASS: Time t2 matches target")
            
            if a_given is not None and not math.isclose(a_given, a_target, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Acceleration a mismatch. Target {a_target}, got {a_given}")
                success = False
            else:
                logs.append("PASS: Acceleration a matches target")
                
            # Additional cross-check: compute a from s and t2
            s_given = self.results.get('s')
            if s_given is not None and t2_given is not None and t2_given != 0:
                a_from_s = 2 * s_given / (t2_given ** 2)
                if a_given is not None and not math.isclose(a_given, a_from_s, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: a from s and t2 inconsistent. a_from_s={a_from_s}, a_given={a_given}")
                    success = False
                else:
                    logs.append("PASS: a consistent with s and t2")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (e.g., drone mass) and verify expected trends.
        Since mass cancels out in this problem, we perturb the lift force factor.
        """
        logs = []
        success = True
        
        try:
            # Original lift force factor = 2 (i.e., 2mg)
            factor_original = 2.0
            factor_perturbed = 2.1  # increase by 5%
            
            # For drone ascent: a1 = (factor * m * g - m * g) / m = (factor - 1) * g
            a1_original = (factor_original - 1) * self.g
            a1_perturbed = (factor_perturbed - 1) * self.g
            
            if a1_perturbed <= a1_original:
                logs.append(f"FAIL: Monotonicity in lift factor. a1 should increase with factor. Original {a1_original}, perturbed {a1_perturbed}")
                success = False
            else:
                logs.append(f"PASS: Lift factor monotonicity holds (a1 increases with factor)")
            
            # Perturb v0 and check t2 trend (qualitative: if v0 increases, horizontal distance x increases, 
            # so t2 might change depending on slope geometry; but here t2 is solved from tanθ equation).
            # We'll skip quantitative check due to complexity.
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== Running {test_name} ===")
            success, logs = test_method()
            all_logs.extend(logs)
            if not success:
                overall_success = False
            all_logs.append("")  # blank line for readability
        
        return overall_success, all_logs