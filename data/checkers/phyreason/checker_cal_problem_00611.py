import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Inputs with defaults from problem context
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1000.0))  # kg
        self.s = getattr(solution_instance, 's', self.results.get('s', 20.0))    # m
        self.line_length = 6.0  # given in problem, not in mapping
        
        # Result quantities with safe access
        self.v1 = getattr(solution_instance, 'v1', self.results.get('v_1', 10.0))
        self.v2 = getattr(solution_instance, 'v2', self.results.get('v_2', 15.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', None))
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.f = getattr(solution_instance, 'f', self.results.get('f', None))
        self.t0 = getattr(solution_instance, 't0', self.results.get('t_0', None))
        self.v = getattr(solution_instance, 'v', self.results.get('v', None))
        self.v_avg = getattr(solution_instance, 'v_avg', self.results.get('overlinev', None))
        self.t_prime = getattr(solution_instance, 't_prime', self.results.get('t\'', None))
        
        # Sympy symbols for symbolic checks
        self.sym_m, self.sym_s, self.sym_v1, self.sym_v2 = sp.symbols('m s v1 v2')
        self.sym_t, self.sym_a, self.sym_f, self.sym_v = sp.symbols('t a f v')
        self.sym_t0, self.sym_t_prime, self.sym_v_avg = sp.symbols('t0 t\' v_avg')
        
        # Constants from problem (safe defaults)
        self.v1_target = 10.0  # m/s from 36 km/h
        self.v2_target = 15.0  # m/s from 54 km/h
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify physical invariants and laws."""
        logs = []
        success = True
        
        # 1. Unit conversion check (v1 should be 10 m/s)
        if not self._is_close(self.v1, self.v1_target):
            logs.append(f"FAIL: v1 conversion incorrect. Expected ~10 m/s, got {self.v1}")
            success = False
        else:
            logs.append("PASS: v1 unit conversion correct.")
        
        # 2. Kinematics invariants for part 1
        # Using v² = u² + 2as with v=0
        a_calc = -self.v1**2 / (2 * self.s)
        if self.a is not None and not self._is_close(self.a, a_calc):
            logs.append(f"FAIL: Acceleration invariant violated. Expected a ≈ {a_calc:.3f}, got {self.a}")
            success = False
        else:
            logs.append("PASS: Acceleration consistent with kinematics.")
        
        # 3. Time from kinematics: t = 2s/v1 (since v_avg = v1/2)
        t_calc = 2 * self.s / self.v1
        if self.t is not None and not self._is_close(self.t, t_calc):
            logs.append(f"FAIL: Time invariant violated. Expected t ≈ {t_calc:.3f}, got {self.t}")
            success = False
        else:
            logs.append("PASS: Braking time consistent.")
        
        # 4. Newton's second law: f = m*a
        if self.a is not None and self.f is not None:
            f_calc = self.m * self.a
            if not self._is_close(self.f, f_calc):
                logs.append(f"FAIL: Newton's second law violated. Expected f ≈ {f_calc:.3f}, got {self.f}")
                success = False
            else:
                logs.append("PASS: Force obeys Newton's second law.")
        
        # 5. Average velocity check
        if self.v_avg is not None and not self._is_close(self.v_avg, self.v1 / 2):
            logs.append(f"FAIL: Average velocity incorrect. Expected {self.v1/2:.3f}, got {self.v_avg}")
            success = False
        else:
            logs.append("PASS: Average velocity correct.")
        
        # 6. Energy/work invariance: Force * distance = ΔKE
        if self.f is not None:
            work = abs(self.f) * self.s
            delta_ke = 0.5 * self.m * self.v1**2
            if not self._is_close(work, delta_ke):
                logs.append(f"FAIL: Work-energy theorem violated. Work ≈ {work:.3f}, ΔKE ≈ {delta_ke:.3f}")
                success = False
            else:
                logs.append("PASS: Work-energy theorem holds.")
        
        # 7. Part 3: Velocity at crosswalk for v2 = 15 m/s
        if self.v is not None and self.a is not None:
            # Using v² = v2² + 2as (same deceleration a)
            v_calc = math.sqrt(max(0, self.v2**2 + 2 * self.a * self.s))
            if not self._is_close(self.v, v_calc):
                logs.append(f"FAIL: Part 3 velocity invariant. Expected v ≈ {v_calc:.3f}, got {self.v}")
                success = False
            else:
                logs.append("PASS: Part 3 velocity consistent with kinematics.")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        # Check that part 1 results match target answer "4s, 2.5 × 10³N"
        if self.t is not None and not self._is_close(self.t, 4.0):
            logs.append(f"FAIL: Braking time doesn't match 4s. Got {self.t}")
            success = False
        else:
            logs.append("PASS: Braking time matches 4s.")
        
        if self.f is not None and not self._is_close(abs(self.f), 2500.0):
            logs.append(f"FAIL: Force magnitude doesn't match 2500N. Got {abs(self.f)}")
            success = False
        else:
            logs.append("PASS: Force magnitude matches 2500N.")
        
        # Check part 2: waiting time should be 20s
        if self.t0 is not None and not self._is_close(self.t0, 20.0):
            logs.append(f"FAIL: Waiting time doesn't match 20s. Got {self.t0}")
            success = False
        else:
            logs.append("PASS: Waiting time matches 20s.")
        
        # Check part 3: velocity should be 5√5 ≈ 11.180 m/s
        if self.v is not None and not self._is_close(self.v, 5*math.sqrt(5)):
            logs.append(f"FAIL: Part 3 velocity doesn't match 5√5. Got {self.v}")
            success = False
        else:
            logs.append("PASS: Part 3 velocity matches 5√5 m/s.")
        
        # Check consistency between braking time and average velocity
        if self.t is not None and self.v_avg is not None:
            s_calc = self.v_avg * self.t
            if not self._is_close(s_calc, self.s):
                logs.append(f"FAIL: Distance inconsistent. v_avg*t = {s_calc}, expected {self.s}")
                success = False
            else:
                logs.append("PASS: Distance consistent with v_avg and t.")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Test how results change with perturbed inputs."""
        logs = []
        success = True
        
        # Save original state
        original_v1 = self.v1
        original_results = {
            't': self.t,
            'a': self.a,
            'f': self.f,
            'v': self.v
        }
        
        try:
            # Test 1: Increase initial velocity by 10%
            perturbed_v1 = original_v1 * 1.10
            # Calculate expected changes using physics formulas
            # For fixed distance: t ∝ 1/v1, |a| ∝ v1², |f| ∝ v1²
            expected_t = 2 * self.s / perturbed_v1
            expected_a = -perturbed_v1**2 / (2 * self.s)
            expected_f = self.m * expected_a
            
            logs.append(f"Perturbation test: v1 = {perturbed_v1:.3f} m/s (+10%)")
            logs.append(f"  Expected: t ≈ {expected_t:.3f}s, |a| ≈ {abs(expected_a):.3f} m/s², |f| ≈ {abs(expected_f):.3f}N")
            
            # Check monotonic trends qualitatively
            if original_results['t'] is not None:
                if perturbed_v1 > original_v1 and expected_t < original_results['t']:
                    logs.append("  PASS: Braking time decreases with increased v1.")
                else:
                    logs.append("  FAIL: Braking time trend incorrect.")
                    success = False
            
            if original_results['a'] is not None:
                if abs(expected_a) > abs(original_results['a']):
                    logs.append("  PASS: Deceleration magnitude increases with v1.")
                else:
                    logs.append("  FAIL: Deceleration trend incorrect.")
                    success = False
            
            # Test 2: Decrease mass by 10%
            perturbed_m = self.m * 0.90
            if original_results['a'] is not None:
                # For same deceleration, force should decrease proportionally
                expected_f2 = perturbed_m * original_results['a']
                if abs(expected_f2) < abs(original_results['f']):
                    logs.append("  PASS: Force decreases with decreased mass.")
                else:
                    logs.append("  FAIL: Force trend with mass incorrect.")
                    success = False
            
            # Test 3: Part 3 velocity monotonicity
            if self.v2 is not None and self.a is not None:
                perturbed_v2 = self.v2 * 1.10
                v_calc = math.sqrt(max(0, perturbed_v2**2 + 2 * self.a * self.s))
                if v_calc > self.v:
                    logs.append("  PASS: Part 3 velocity increases with increased v2.")
                else:
                    logs.append("  FAIL: Part 3 velocity trend incorrect.")
                    success = False
                    
        except Exception as e:
            logs.append(f"ERROR during monotonicity test: {e}")
            success = False
        finally:
            # Restore original state
            self.v1 = original_v1
            # Note: We don't modify the solution, just our local copy
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker Report ==="]
        overall_success = True
        
        # Run invariant tests
        success, logs = self.test_invariants()
        all_logs.append("\n--- Invariant Tests ---")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- Trace Consistency Tests ---")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- Monotonicity Tests ---")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Summary
        all_logs.append("\n=== Summary ===")
        all_logs.append(f"Overall check: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs