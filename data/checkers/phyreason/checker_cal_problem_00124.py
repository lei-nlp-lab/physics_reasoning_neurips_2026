import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all known parameters safely
        self.m = self._safe_get('m')
        self.q = self._safe_get('q')
        self.B = self._safe_get('B')
        self.d = self._safe_get('d')
        self.L = self._safe_get('L')
        self.beta = self._safe_get('beta')
        self.v0 = self._safe_get('v0')
        
        # Extract results with fallbacks
        self.E = self._safe_get('E')
        self.v_m = self._safe_get('v_m')
        self.v = self._safe_get('v')  # Velocity at entry to magnetic field
        self.v_prime = self._safe_get('v\'')
        self.v_y = self._safe_get('v_y')
        self.a = self._safe_get('a')
        
        # Radii
        self.r1 = self._safe_get('r_1')
        self.r2 = self._safe_get('r_2')
        self.r1_prime = self._safe_get('r_1\'')
        self.r2_prime = self._safe_get('r_2\'')
        self.r11 = self._safe_get('r_11')
        self.r21 = self._safe_get('r_21')
        self.r12_prime = self._safe_get('r_12\'')
        self.r22 = self._safe_get('r_22')
        
        # Times
        self.t_M = self._safe_get('t_M')
        self.t_N = self._safe_get('t_N')
        self.Deltat = self._safe_get('Deltat')
        
        # Coordinates
        self.x4 = self._safe_get('x_4')
        self.y4 = self._safe_get('y_4')
        self.z4 = self._safe_get('z_4')
        
        # Parse coordinate tuple if available
        coord_key = 'x_4,y_4,z_4'
        if coord_key in self.results:
            try:
                coords = eval(self.results[coord_key])
                if isinstance(coords, (tuple, list)) and len(coords) == 3:
                    self.x4 = self.x4 or coords[0]
                    self.y4 = self.y4 or coords[1]
                    self.z4 = self.z4 or coords[2]
            except:
                pass
    
    def _safe_get(self, key):
        """Safely get a value from results or solution instance."""
        value = self.results.get(key, None)
        if value is None:
            value = getattr(self.solution_instance, key, None)
        return value
    
    def test_invariants(self):
        """Check physical invariants and laws."""
        logs = []
        success = True
        
        try:
            # 1. Check electric field calculation from steps 1-4
            if all(v is not None for v in [self.m, self.v0, self.beta, self.q, self.L, self.E]):
                expected_E = (self.m * self.v0**2 * math.sin(self.beta) * math.cos(self.beta)) / (self.q * self.L)
                if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Electric field mismatch: got {self.E}, expected {expected_E}")
                    success = False
                else:
                    logs.append("✓ Electric field matches formula E = (m v0² sinβ cosβ)/(q L)")
            
            # 2. Check maximum velocity v_m = q*B*d/m
            if all(v is not None for v in [self.v_m, self.q, self.B, self.d, self.m]):
                expected_vm = self.q * self.B * self.d / self.m
                if not math.isclose(self.v_m, expected_vm, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Maximum velocity mismatch: got {self.v_m}, expected {expected_vm}")
                    success = False
                else:
                    logs.append("✓ Maximum velocity matches v_m = q B d / m")
            
            # 3. Check radii relationships for magnetic field I
            # r1' = m*v/(q*B) should equal d/2
            if all(v is not None for v in [self.r1_prime, self.m, self.v, self.q, self.B, self.d]):
                expected_r1_prime = self.m * self.v / (self.q * self.B)
                expected_r1_prime_d2 = self.d / 2
                if not (math.isclose(self.r1_prime, expected_r1_prime, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.r1_prime, expected_r1_prime_d2, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"Radius r1' mismatch: got {self.r1_prime}, expected {expected_r1_prime} or {expected_r1_prime_d2}")
                    success = False
                else:
                    logs.append("✓ Radius r1' matches both m*v/(q*B) and d/2")
            
            # 4. Check radii relationships for magnetic field II
            # r2' = m*v/(q*(√2/2)*B) = √2 * r1'
            if all(v is not None for v in [self.r2_prime, self.r1_prime]):
                expected_r2_prime = math.sqrt(2) * self.r1_prime
                if not math.isclose(self.r2_prime, expected_r2_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Radius r2' mismatch: got {self.r2_prime}, expected √2*r1' = {expected_r2_prime}")
                    success = False
                else:
                    logs.append("✓ Radius r2' = √2 * r1'")
            
            # 5. Check kinetic energy relationship between ions M and N
            # (1/2) m v² = (1/2) * 4m * v'² => v' = v/2
            if all(v is not None for v in [self.v, self.v_prime]):
                expected_v_prime = self.v / 2
                if not math.isclose(self.v_prime, expected_v_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Velocity v' mismatch: got {self.v_prime}, expected v/2 = {expected_v_prime}")
                    success = False
                else:
                    logs.append("✓ Kinetic energy conservation gives v' = v/2")
            
            # 6. Check ion N radii
            # r12' = 4m*v'/(q*B) should equal d
            if all(v is not None for v in [self.r12_prime, self.m, self.v_prime, self.q, self.B, self.d]):
                expected_r12_prime = 4 * self.m * self.v_prime / (self.q * self.B)
                if not (math.isclose(self.r12_prime, expected_r12_prime, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.r12_prime, self.d, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"Radius r12' mismatch: got {self.r12_prime}, expected {expected_r12_prime} or d = {self.d}")
                    success = False
                else:
                    logs.append("✓ Radius r12' matches both 4m*v'/(q*B) and d")
            
            # 7. Check r22 = √2 * r12'
            if all(v is not None for v in [self.r22, self.r12_prime]):
                expected_r22 = math.sqrt(2) * self.r12_prime
                if not math.isclose(self.r22, expected_r22, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Radius r22 mismatch: got {self.r22}, expected √2*r12' = {expected_r22}")
                    success = False
                else:
                    logs.append("✓ Radius r22 = √2 * r12'")
            
            # 8. Check coordinates (d, d, 0)
            if all(v is not None for v in [self.x4, self.y4, self.z4, self.d]):
                if not (math.isclose(self.x4, self.d, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.y4, self.d, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(self.z4, 0, abs_tol=1e-2)):
                    logs.append(f"Coordinates mismatch: got ({self.x4}, {self.y4}, {self.z4}), expected ({self.d}, {self.d}, 0)")
                    success = False
                else:
                    logs.append("✓ Coordinates (x4, y4, z4) = (d, d, 0)")
            
            # 9. Check time difference formula Δt = (2+2√2)πm/(qB)
            if all(v is not None for v in [self.Deltat, self.m, self.q, self.B]):
                expected_Deltat = (2 + 2 * math.sqrt(2)) * math.pi * self.m / (self.q * self.B)
                if not math.isclose(self.Deltat, expected_Deltat, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Time difference Δt mismatch: got {self.Deltat}, expected (2+2√2)πm/(qB) = {expected_Deltat}")
                    success = False
                else:
                    logs.append("✓ Time difference Δt = (2+2√2)πm/(qB)")
            
            if success:
                logs.append("All invariants passed.")
            else:
                logs.append("Some invariants failed.")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        logs = []
        success = True
        
        try:
            # 1. Check that Δt = t_N - t_M
            if all(v is not None for v in [self.Deltat, self.t_N, self.t_M]):
                calc_Deltat = self.t_N - self.t_M
                if not math.isclose(self.Deltat, calc_Deltat, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Δt consistency mismatch: stored {self.Deltat}, calculated t_N - t_M = {calc_Deltat}")
                    success = False
                else:
                    logs.append("✓ Δt = t_N - t_M")
            
            # 2. Check that v = v0*cosβ (velocity at O is z-component only)
            if all(v is not None for v in [self.v, self.v0, self.beta]):
                expected_v = self.v0 * math.cos(self.beta)
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Velocity at O mismatch: got {self.v}, expected v0*cosβ = {expected_v}")
                    success = False
                else:
                    logs.append("✓ v = v0*cosβ (velocity at entry to magnetic field)")
            
            # 3. Check acceleration from Newton's second law: a = qE/m
            if all(v is not None for v in [self.a, self.q, self.E, self.m]):
                expected_a = self.q * self.E / self.m
                if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Acceleration mismatch: got {self.a}, expected qE/m = {expected_a}")
                    success = False
                else:
                    logs.append("✓ Newton's second law: a = qE/m")
            
            # 4. Check velocity component v_y = v0*sinβ from uniformly decelerated motion
            if all(v is not None for v in [self.v_y, self.v0, self.beta]):
                expected_v_y = self.v0 * math.sin(self.beta)
                if not math.isclose(self.v_y, expected_v_y, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"v_y mismatch: got {self.v_y}, expected v0*sinβ = {expected_v_y}")
                    success = False
                else:
                    logs.append("✓ v_y = v0*sinβ")
            
            if success:
                logs.append("All trace consistency checks passed.")
            else:
                logs.append("Some trace consistency checks failed.")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing parameters."""
        logs = []
        success = True
        
        try:
            # We'll perturb the magnetic field B and check that Δt changes inversely
            if all(v is not None for v in [self.B, self.Deltat, self.m, self.q]):
                original_B = self.B
                original_Deltat = self.Deltat
                
                # Perturb B by +5%
                perturbed_B = original_B * 1.05
                
                # Expected Δt should decrease inversely: Δt ∝ 1/B
                expected_Deltat = original_Deltat * (original_B / perturbed_B)
                
                # The actual formula: Δt = (2+2√2)πm/(qB)
                # So we can calculate the new Δt directly
                calc_Deltat = (2 + 2 * math.sqrt(2)) * math.pi * self.m / (self.q * perturbed_B)
                
                # Check both the proportionality and the direct calculation
                if not (math.isclose(expected_Deltat, calc_Deltat, rel_tol=1e-2, abs_tol=1e-2) and
                        math.isclose(calc_Deltat, original_Deltat / 1.05, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"Monotonicity check failed: Δt should be inversely proportional to B")
                    logs.append(f"Original: B={original_B}, Δt={original_Deltat}")
                    logs.append(f"Perturbed: B={perturbed_B}, expected Δt≈{expected_Deltat}, calculated Δt={calc_Deltat}")
                    success = False
                else:
                    logs.append("✓ Δt is inversely proportional to B (Δt ∝ 1/B)")
            
            # Also check that v_m is proportional to B*d
            if all(v is not None for v in [self.v_m, self.B, self.d, self.q, self.m]):
                original_vm = self.v_m
                
                # Perturb d by +5%
                perturbed_d = self.d * 1.05
                expected_vm = original_vm * (perturbed_d / self.d)  # v_m ∝ d
                
                # Direct calculation
                calc_vm = self.q * self.B * perturbed_d / self.m
                
                if not math.isclose(expected_vm, calc_vm, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Monotonicity check failed: v_m should be proportional to d")
                    logs.append(f"Original: d={self.d}, v_m={original_vm}")
                    logs.append(f"Perturbed: d={perturbed_d}, expected v_m≈{expected_vm}, calculated v_m={calc_vm}")
                    success = False
                else:
                    logs.append("✓ v_m is proportional to d (v_m ∝ d)")
            
            if success:
                logs.append("All monotonicity checks passed.")
            else:
                logs.append("Some monotonicity checks failed.")
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return combined results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_method()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("All physics checks passed.")
        else:
            all_logs.append("Some physics checks failed.")
        
        return overall_success, all_logs