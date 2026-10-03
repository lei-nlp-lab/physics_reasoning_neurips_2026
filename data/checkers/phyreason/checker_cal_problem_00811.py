import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the checker with safe attribute access.
        """
        # Extract inputs safely
        self.v = getattr(solution_instance, 'v', 6.0)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.T = float(self.results.get('T', 0.0))
        self.Deltat = float(self.results.get('Deltat', 0.0))
        self.Deltax = float(self.results.get('Deltax', 0.0))
        
        # Direction may be string or numeric representation
        direction_val = self.results.get('direction', '')
        if isinstance(direction_val, (int, float)):
            # Convert numeric to string representation
            if direction_val > 0:
                self.direction = '+y'
            elif direction_val < 0:
                self.direction = '-y'
            else:
                self.direction = '0'
        else:
            self.direction = str(direction_val).lower()
        
        # Derived quantities
        self.lambda_wave = self.v * self.T  # λ = v * T
        
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental wave physics invariants.
        """
        logs = []
        success = True
        
        try:
            # 1. Check v = λ / T (from step 1)
            if self.T > 0:
                lambda_from_T = self.v * self.T
                if not math.isclose(self.lambda_wave, lambda_from_T, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant violation: λ from v*T ({lambda_from_T}) doesn't match stored λ ({self.lambda_wave})")
                    success = False
            
            # 2. Numerical check of wave relationship
            if self.T > 0 and self.lambda_wave > 0:
                v_calc = self.lambda_wave / self.T
                if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant violation: v = λ/T fails: {self.v} vs {v_calc}")
                    success = False
            
            # 3. Check step 2: Δt = Δx / v, where Δx = λ/4 (crest to particle P at t=0)
            if self.T > 0 and self.Deltat > 0:
                distance_to_crest = self.lambda_wave / 4.0
                expected_dt = distance_to_crest / self.v
                if not math.isclose(self.Deltat, expected_dt, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Invariant violation: Δt should be ~{expected_dt:.4f}, got {self.Deltat}")
                    success = False
            
            # 4. Check step 3: Δx = v * 1.1s = 6.6 m
            delta_t_total = 1.1
            expected_dx = self.v * delta_t_total
            if not math.isclose(self.Deltax, expected_dx, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Invariant violation: Δx should be {expected_dx}, got {self.Deltax}")
                success = False
            
        except Exception as e:
            logs.append(f"Invariant test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results for consistency.
        """
        logs = []
        success = True
        
        try:
            # 1. Step 1: T = λ / v
            if self.T > 0 and self.lambda_wave > 0:
                T_calc = self.lambda_wave / self.v
                if not math.isclose(self.T, T_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace inconsistency: T mismatch: {self.T} vs {T_calc}")
                    success = False
            
            # 2. Step 2: Δt = (λ/4) / v (crest to particle P)
            if self.T > 0 and self.Deltat > 0:
                distance_to_crest = self.lambda_wave / 4.0
                expected_dt = distance_to_crest / self.v
                if not math.isclose(self.Deltat, expected_dt, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace inconsistency: Δt should be ~{expected_dt:.4f}, got {self.Deltat}")
                    success = False
            
            # 3. Step 3: Δx = v * 1.1s = 6.6 m
            expected_dx = self.v * 1.1
            if not math.isclose(self.Deltax, expected_dx, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace inconsistency: Δx should be {expected_dx}, got {self.Deltax}")
                success = False
            
            # 4. Step 4: Decomposition of 6.6 m into 2λ + 1.8 m
            if self.lambda_wave > 0:
                num_wavelengths = int(self.Deltax // self.lambda_wave)
                remainder = self.Deltax - num_wavelengths * self.lambda_wave
                if not (num_wavelengths == 2 and math.isclose(remainder, 1.8, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"Trace inconsistency: Δx={self.Deltax} should be 2λ + 1.8, got {num_wavelengths}λ + {remainder}")
                    success = False
            
            # 5. Step 6: Direction should be positive y
            expected_directions = ['+y', 'positive y', 'upward', 'up', 'positive']
            direction_lower = self.direction.lower()
            if not any(exp_dir in direction_lower for exp_dir in expected_directions):
                logs.append(f"Trace inconsistency: Expected +y direction, got '{self.direction}'")
                success = False
                
        except Exception as e:
            logs.append(f"Trace consistency test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb wave speed v and check monotonic trends.
        """
        logs = []
        success = True
        
        original_v = self.v
        original_T = self.T
        original_Deltat = self.Deltat
        
        try:
            # Perturbation factor
            perturbation = 1.1  # 10% increase
            
            # Test 1: Increase v by 10%, T should decrease
            v_perturbed = original_v * perturbation
            T_perturbed = self.lambda_wave / v_perturbed  # λ fixed from initial waveform
            
            if not (T_perturbed < original_T):
                logs.append(f"Monotonicity violation: T should decrease when v increases. "
                          f"Original T={original_T}, perturbed T={T_perturbed}")
                success = False
            else:
                logs.append(f"Monotonicity OK: T decreases from {original_T:.3f} to {T_perturbed:.3f} when v increases")
            
            # Test 2: Increase v by 10%, Δt should decrease (distance to crest fixed at λ/4)
            distance_to_crest = self.lambda_wave / 4.0
            Deltat_perturbed = distance_to_crest / v_perturbed
            
            if not (Deltat_perturbed < original_Deltat):
                logs.append(f"Monotonicity violation: Δt should decrease when v increases. "
                          f"Original Δt={original_Deltat}, perturbed Δt={Deltat_perturbed}")
                success = False
            else:
                logs.append(f"Monotonicity OK: Δt decreases from {original_Deltat:.3f} to {Deltat_perturbed:.3f} when v increases")
                
            # Restore original values
            self.v = original_v
            self.T = original_T
            self.Deltat = original_Deltat
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}\n{traceback.format_exc()}")
            success = False
            # Restore on error
            self.v = original_v
            self.T = original_T
            self.Deltat = original_Deltat
            
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and consolidated logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Wave Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                all_logs.append(f"Test {test_name} crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        all_logs.insert(0, f"Overall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs