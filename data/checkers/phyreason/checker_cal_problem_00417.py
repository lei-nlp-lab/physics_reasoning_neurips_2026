import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.s3 = 2.5  # displacement in 3rd second (given)
        self.s7 = 2.9  # displacement in 7th second (given)
        self.T = 1.0   # time interval (1 second)
        
        # Extract results safely
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', None))
        self.v0 = getattr(solution_instance, 'v0', solution_instance.results.get('v₀', None))
        self.v6 = getattr(solution_instance, 'v6', solution_instance.results.get('v₆', None))
        self.barv = getattr(solution_instance, 'barv', solution_instance.results.get('barv', None))
        
        self.logs = []
        
    def test_step1_acceleration(self):
        """Check Step 1: a = (s7 - s3) / (4 * T^2)"""
        try:
            # Use sympy for symbolic formula
            s3, s7, T, a = sp.symbols('s3 s7 T a')
            formula = sp.Eq(a, (s7 - s3) / (4 * T**2))
            
            # Numerical check
            a_calc = (self.s7 - self.s3) / (4 * self.T**2)
            
            if self.a is None:
                self.logs.append("FAIL Step1: Acceleration 'a' not found in results")
                return False
                
            if math.isclose(self.a, a_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append("PASS Step1: Acceleration formula verified")
                return True
            else:
                self.logs.append(f"FAIL Step1: a={self.a}, calculated={a_calc}")
                return False
        except Exception as e:
            self.logs.append(f"ERROR Step1: {str(e)}")
            return False
    
    def test_step2_initial_velocity(self):
        """Check Step 2: v0 from displacement in 3rd second"""
        try:
            # Symbolic: displacement from t=2 to t=3
            t2, t3 = 2.0, 3.0
            # s3 = v0*(t3-t2) + 0.5*a*(t3^2 - t2^2)
            v0_calc = self.s3 - 0.5 * self.a * (t3**2 - t2**2)
            
            if self.v0 is None:
                self.logs.append("FAIL Step2: Initial velocity 'v0' not found in results")
                return False
                
            if math.isclose(self.v0, v0_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.logs.append("PASS Step2: Initial velocity verified")
                return True
            else:
                self.logs.append(f"FAIL Step2: v0={self.v0}, calculated={v0_calc}")
                return False
        except Exception as e:
            self.logs.append(f"ERROR Step2: {str(e)}")
            return False
    
    def test_step3_velocity_and_average(self):
        """Check Step 3: v6 and average velocity"""
        try:
            # Check v6 = v0 + a*6
            v6_calc = self.v0 + self.a * 6.0
            
            if self.v6 is None:
                self.logs.append("FAIL Step3: Velocity v6 not found in results")
                return False
                
            v6_ok = math.isclose(self.v6, v6_calc, rel_tol=1e-2, abs_tol=1e-2)
            
            # Check average velocity: (v0 + v6)/2
            barv_calc = (self.v0 + self.v6) / 2.0
            
            if self.barv is None:
                self.logs.append("FAIL Step3: Average velocity not found in results")
                return False
                
            barv_ok = math.isclose(self.barv, barv_calc, rel_tol=1e-2, abs_tol=1e-2)
            
            # Alternative check: average velocity = total displacement / time
            # displacement in first 6s: x6 = v0*6 + 0.5*a*36
            x6 = self.v0 * 6.0 + 0.5 * self.a * 36.0
            barv_alt = x6 / 6.0
            barv_alt_ok = math.isclose(self.barv, barv_alt, rel_tol=1e-2, abs_tol=1e-2)
            
            if v6_ok and barv_ok:
                self.logs.append("PASS Step3: v6 and average velocity verified")
                if barv_alt_ok:
                    self.logs.append("PASS Step3: Alternative average velocity check")
                return True
            else:
                msg = f"FAIL Step3: v6={self.v6}, calc={v6_calc}; barv={self.barv}, calc={barv_calc}"
                if not barv_alt_ok:
                    msg += f", alt={barv_alt}"
                self.logs.append(msg)
                return False
        except Exception as e:
            self.logs.append(f"ERROR Step3: {str(e)}")
            return False
    
    def test_displacement_formula_consistency(self):
        """Check displacement formulas for nth second"""
        try:
            # Formula for displacement in nth second: s_n = v0 + (a/2)*(2n-1)
            # For n=3: s3 = v0 + (a/2)*5
            s3_calc = self.v0 + (self.a / 2.0) * 5.0
            s3_ok = math.isclose(self.s3, s3_calc, rel_tol=1e-2, abs_tol=1e-2)
            
            # For n=7: s7 = v0 + (a/2)*13
            s7_calc = self.v0 + (self.a / 2.0) * 13.0
            s7_ok = math.isclose(self.s7, s7_calc, rel_tol=1e-2, abs_tol=1e-2)
            
            if s3_ok and s7_ok:
                self.logs.append("PASS: Displacement formulas for nth second verified")
                return True
            else:
                msg = f"FAIL: s3={self.s3}, calc={s3_calc}; s7={self.s7}, calc={s7_calc}"
                self.logs.append(msg)
                return False
        except Exception as e:
            self.logs.append(f"ERROR displacement consistency: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # From KB: The target answers are acceleration and average velocity
            target_accel = 0.1
            target_barv = 2.55
            
            accel_ok = math.isclose(self.a, target_accel, rel_tol=1e-2, abs_tol=1e-2)
            barv_ok = math.isclose(self.barv, target_barv, rel_tol=1e-2, abs_tol=1e-2)
            
            if accel_ok and barv_ok:
                self.logs.append("PASS: Results match target answers")
                return True
            else:
                msg = f"FAIL: a={self.a}, target={target_accel}; barv={self.barv}, target={target_barv}"
                self.logs.append(msg)
                return False
        except Exception as e:
            self.logs.append(f"ERROR trace consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb input and check trend"""
        try:
            # Store original values
            original_s3 = self.s3
            
            # Perturb s3 by +10%
            perturbed_s3 = self.s3 * 1.1
            
            # Recalculate using formulas from steps
            # Step 1: a = (s7 - s3) / 4
            perturbed_a = (self.s7 - perturbed_s3) / 4.0
            
            # Step 2: v0 from s3 = v0 + 0.5*a*5
            perturbed_v0 = perturbed_s3 - 0.5 * perturbed_a * 5.0
            
            # Step 3: v6 and average velocity
            perturbed_v6 = perturbed_v0 + perturbed_a * 6.0
            perturbed_barv = (perturbed_v0 + perturbed_v6) / 2.0
            
            # Check trends:
            # If s3 increases, v0 should increase (positive correlation)
            # If s3 increases, a should decrease (negative correlation)
            # The effect on barv depends on the balance
            
            s3_increase = perturbed_s3 > original_s3
            v0_increase = perturbed_v0 > self.v0
            a_decrease = perturbed_a < self.a
            barv_increase = perturbed_barv > self.barv
            
            # Corrected logic: For negative correlation between s3 and a,
            # if s3 increases (True), a should decrease (True), so both are True.
            # The relationship is correct when both have the same truth value
            # (both True or both False) for negative correlation.
            
            trend_consistent = (
                (s3_increase == v0_increase) and  # v0 follows s3 (positive correlation)
                (s3_increase == a_decrease) and   # a moves opposite to s3 (negative correlation means same truth value)
                (s3_increase == barv_increase)    # barv follows s3 (positive correlation)
            )
            
            if trend_consistent:
                self.logs.append("PASS: Monotonicity test - trends are physically consistent")
                return True
            else:
                msg = (f"FAIL Monotonicity: s3↑={s3_increase}, v0↑={v0_increase}, "
                       f"a↓={a_decrease}, barv↑={barv_increase}")
                self.logs.append(msg)
                return False
                
        except Exception as e:
            self.logs.append(f"ERROR monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all checks and return success status with logs"""
        self.logs = []
        
        tests = [
            ("Step1 Acceleration", self.test_step1_acceleration),
            ("Step2 Initial Velocity", self.test_step2_initial_velocity),
            ("Step3 Velocity & Average", self.test_step3_velocity_and_average),
            ("Displacement Consistency", self.test_displacement_formula_consistency),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            # Skip if required values are None
            if test_name.startswith("Step") and self.a is None:
                self.logs.append(f"SKIP {test_name}: Missing required values")
                all_passed = False
                continue
                
            passed = test_func()
            if not passed:
                all_passed = False
        
        return all_passed, self.logs