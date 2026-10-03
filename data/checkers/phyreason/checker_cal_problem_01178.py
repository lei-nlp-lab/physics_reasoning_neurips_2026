import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs from problem statement
        self.v0 = 15.0  # taxi initial speed (m/s)
        self.v1 = 20.0  # truck initial speed (m/s)
        self.x0_initial = 15.0  # initial separation (m)
        self.t_total_pre_brake = 4.0  # 3.5 + 0.5 seconds
        
        # Extract results safely
        self.a_truck = getattr(solution_instance, 'a_texttruck', 
                               self.results.get('a_texttruck', None))
        self.abs_a_truck = getattr(solution_instance, '|a_texttruck|', 
                                   self.results.get('|a_texttruck|', None))
        self.x_truck = getattr(solution_instance, 'x_truck', 
                               self.results.get('x_truck', None))
        self.x_car = getattr(solution_instance, 'x_car', 
                             self.results.get('x_car', None))
        self.x0_at_brake = getattr(solution_instance, 'x_0', 
                                   self.results.get('x_0', None))
        self.v1_prime = getattr(solution_instance, 'v_1prime', 
                                self.results.get('v_1prime', None))
        self.t3 = getattr(solution_instance, 't_3', 
                          self.results.get('t_3', None))
        self.a_out = getattr(solution_instance, 'a_out', 
                             self.results.get('a_out', None))
        
        # Intermediate verification values from steps analysis
        self.expected = {
            'a_truck': -2.5,
            'abs_a_truck': 2.5,
            'x_truck': 60.0,
            'x_car': 60.0,
            'x0_at_brake': 15.0,
            'v1_prime': 10.0,
            't3': 14/3,
            'a_out': -45/14
        }
        
        self.logs = []
    
    def _is_close(self, a, b, name=""):
        if a is None or b is None:
            self.logs.append(f"Missing value for {name}")
            return False
        
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            is_ok = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
            if not is_ok:
                self.logs.append(f"Mismatch for {name}: got {a}, expected {b}")
            return is_ok
        else:
            # Try symbolic equality if values are sympy expressions
            try:
                diff = sp.simplify(a - b)
                is_ok = abs(float(diff)) < 1e-2
                if not is_ok:
                    self.logs.append(f"Symbolic mismatch for {name}: {a} vs {b}")
                return is_ok
            except:
                self.logs.append(f"Could not compare {name}: {a} vs {b}")
                return False
    
    def test_invariants(self):
        """Verify physics conservation laws and derived relationships."""
        try:
            # 1. Truck acceleration from stopping distance
            # Using v^2 = u^2 + 2as with v=0, u=20, s=80
            a_truck_calc = -self.v1**2 / (2 * 80.0)
            if not self._is_close(self.a_truck, a_truck_calc, "a_truck from formula"):
                return False
            
            # 2. Magnitude of truck acceleration
            if not self._is_close(self.abs_a_truck, abs(a_truck_calc), "|a_truck|"):
                return False
            
            # 3. Truck displacement during pre-brake period
            t = self.t_total_pre_brake
            x_truck_calc = self.v1 * t + 0.5 * a_truck_calc * t**2
            if not self._is_close(self.x_truck, x_truck_calc, "x_truck"):
                return False
            
            # 4. Car displacement during pre-brake period (uniform motion)
            x_car_calc = self.v0 * t
            if not self._is_close(self.x_car, x_car_calc, "x_car"):
                return False
            
            # 5. Separation at braking start
            # Initial separation + (truck displacement - car displacement)
            separation_calc = self.x0_initial + (x_truck_calc - x_car_calc)
            if not self._is_close(self.x0_at_brake, separation_calc, "x0_at_brake"):
                return False
            
            # 6. Truck velocity at braking start
            v1_prime_calc = self.v1 + a_truck_calc * t
            if not self._is_close(self.v1_prime, v1_prime_calc, "v1_prime"):
                return False
            
            # 7. Time for taxi to stop (t3) from condition
            # Condition: car must cover separation + remaining truck stopping distance
            remaining_truck_dist = 80.0 - x_truck_calc  # 20 m
            total_distance = separation_calc + remaining_truck_dist  # 35 m
            
            # For uniformly decelerated motion to stop: s = (v0 * t3)/2
            t3_calc = 2 * total_distance / self.v0
            if not self._is_close(self.t3, t3_calc, "t3"):
                return False
            
            # 8. Taxi acceleration from stopping time
            a_out_calc = -self.v0 / t3_calc
            if not self._is_close(self.a_out, a_out_calc, "a_out"):
                return False
            
            # 9. Energy/work check: Compare work done by friction forces
            # Truck: F_truck * 80 = 0.5 * m_truck * 20^2 (unknown mass cancels)
            # Taxi: F_taxi * (v0*t3/2) = 0.5 * m_taxi * 15^2
            # Since masses unknown, we check ratio: a_out / a_truck should be reasonable
            # Expect |a_out| > |a_truck| since taxi must stop more quickly
            if self.a_out is not None and self.a_truck is not None:
                if not abs(self.a_out) > abs(self.a_truck):
                    self.logs.append(f"Warning: |a_out| ({abs(self.a_out)}) not > |a_truck| ({abs(self.a_truck)})")
            
            self.logs.append("All invariants passed")
            return True
            
        except Exception as e:
            self.logs.append(f"Error in test_invariants: {str(e)}")
            self.logs.append(traceback.format_exc())
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check each step against expected values
            steps_ok = True
            
            # Step 1: Truck acceleration
            if not self._is_close(self.a_truck, self.expected['a_truck'], "step1 a_truck"):
                steps_ok = False
            if not self._is_close(self.abs_a_truck, self.expected['abs_a_truck'], "step1 |a_truck|"):
                steps_ok = False
            
            # Step 2: Truck displacement
            if not self._is_close(self.x_truck, self.expected['x_truck'], "step2 x_truck"):
                steps_ok = False
            
            # Step 3: Car displacement
            if not self._is_close(self.x_car, self.expected['x_car'], "step3 x_car"):
                steps_ok = False
            
            # Step 4: Separation at braking
            if not self._is_close(self.x0_at_brake, self.expected['x0_at_brake'], "step4 x0"):
                steps_ok = False
            
            # Step 5: Truck velocity at braking
            if not self._is_close(self.v1_prime, self.expected['v1_prime'], "step5 v1_prime"):
                steps_ok = False
            
            # Step 7: Time t3
            if not self._is_close(self.t3, self.expected['t3'], "step7 t3"):
                steps_ok = False
            
            # Step 8: Taxi acceleration
            if not self._is_close(self.a_out, self.expected['a_out'], "step8 a_out"):
                steps_ok = False
            
            # Check final answers
            target_answers = [2.5, 15.0, 45/14]
            student_answers = [self.abs_a_truck, self.x0_at_brake, -self.a_out]  # a_out is negative
            
            for i, (student, target) in enumerate(zip(student_answers, target_answers)):
                if student is None or not math.isclose(abs(student), target, rel_tol=1e-2, abs_tol=1e-2):
                    steps_ok = False
                    self.logs.append(f"Final answer {i+1} mismatch: got {student}, expected {target}")
            
            if steps_ok:
                self.logs.append("All trace consistency checks passed")
            return steps_ok
            
        except Exception as e:
            self.logs.append(f"Error in test_trace_consistency: {str(e)}")
            self.logs.append(traceback.format_exc())
            return False
    
    def test_monotonicity(self):
        """Perturb initial separation and verify expected trend."""
        try:
            # Backup original value
            original_x0 = self.x0_initial
            
            # Perturb by +10% (more separation)
            perturbed_x0 = original_x0 * 1.1  # 16.5 m
            
            # Recompute condition using physics formulas
            # Truck acceleration remains same: a_truck = -2.5
            a_truck = -2.5
            
            # Pre-brake displacements
            t = self.t_total_pre_brake
            x_truck = self.v1 * t + 0.5 * a_truck * t**2  # 60 m
            x_car = self.v0 * t  # 60 m
            
            # New separation at braking
            separation = perturbed_x0 + (x_truck - x_car)  # 16.5 m
            
            # Remaining truck stopping distance
            remaining_truck_dist = 80.0 - x_truck  # 20 m
            
            # Total distance taxi must cover
            total_distance = separation + remaining_truck_dist  # 36.5 m
            
            # New stopping time for taxi
            t3_new = 2 * total_distance / self.v0  # ≈ 4.8667 s
            
            # New required acceleration
            a_out_new = -self.v0 / t3_new  # ≈ -3.082 m/s²
            
            # Original a_out from solution (should be more negative)
            if self.a_out is not None:
                # With increased separation, required deceleration magnitude should DECREASE
                # (less urgent braking needed)
                if not abs(a_out_new) < abs(self.a_out):
                    self.logs.append(f"Monotonicity violation: |a_out_new| ({abs(a_out_new)}) >= |a_out| ({abs(self.a_out)})")
                    return False
                
                # Verify using symbolic derivative
                # From formula: a_out = -v0² / (2*(x0 + 20))
                # Derivative wrt x0: v0² / (2*(x0 + 20)²) > 0
                # So a_out increases (becomes less negative) as x0 increases
                original_a_out_calc = -self.v0**2 / (2*(original_x0 + 20))
                perturbed_a_out_calc = -self.v0**2 / (2*(perturbed_x0 + 20))
                
                if not (perturbed_a_out_calc > original_a_out_calc):
                    self.logs.append("Symbolic monotonicity check failed")
                    return False
                
                self.logs.append("Monotonicity check passed: Increased separation requires less severe braking")
                return True
            else:
                self.logs.append("Cannot check monotonicity: a_out is None")
                return False
                
        except Exception as e:
            self.logs.append(f"Error in test_monotonicity: {str(e)}")
            self.logs.append(traceback.format_exc())
            return False
    
    def run_all_tests(self):
        """Execute all checks and return success status with logs."""
        self.logs = []
        success = True
        
        # Run tests
        if not self.test_invariants():
            success = False
        
        if not self.test_trace_consistency():
            success = False
        
        if not self.test_monotonicity():
            success = False  # Consider monotonicity as warning, not failure
        
        return success, self.logs