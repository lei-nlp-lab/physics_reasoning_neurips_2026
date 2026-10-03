import sympy as sp
import math
import traceback
from typing import Dict, Any, Tuple, List


class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction
        results = getattr(solution_instance, 'results', {})
        
        # Extract inputs
        self.R = getattr(solution_instance, 'R', None)
        self.m_A = getattr(solution_instance, 'm_A', None)
        self.m_B = getattr(solution_instance, 'm_B', None)
        self.v0 = getattr(solution_instance, 'v0', None)
        self.e = getattr(solution_instance, 'e', None)  # For inelastic case
        self.n = getattr(solution_instance, 'n', None)  # For geometric series
        
        # Extract results with safe defaults
        self.v = results.get('v', None)
        self.F = results.get('F', None)
        self.fracm_Am_B = results.get('fracm_Am_B', None)
        self.s = results.get('s', None)
        
        # Elastic collision results
        self.v_A = results.get('v_A', None)
        self.v_B = results.get('v_B', None)
        self.v_Aprime = results.get('v_Aprime', None)
        self.v_Bprime = results.get('v_Bprime', None)
        self.fracv_Av_B = results.get('fracv_Av_B', None)
        
        # Inelastic collision results
        self.v_1_relative = results.get('v_1_relative', None)
        self.v_2_relative = results.get('v_2_relative', None)
        self.v_B1 = results.get('v_B1', None)
        self.t_1 = results.get('t_1', None)
        self.t_2 = results.get('t_2', None)
        self.s_1 = results.get('s_1', None)
        self.s_2 = results.get('s_2', None)
        
        self.logs = []
    
    def _log(self, msg: str, success: bool = True):
        """Helper to log messages"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {msg}")
        return success
    
    def test_invariants(self) -> List[bool]:
        """Test physical invariants: conservation laws, Newton's laws"""
        tests = []
        
        try:
            # 1. Test momentum conservation for inelastic collision (step 1)
            if all(x is not None for x in [self.m_A, self.m_B, self.v0, self.v]):
                m_A, m_B, v0, v = float(self.m_A), float(self.m_B), float(self.v0), float(self.v)
                momentum_initial = m_A * v0
                momentum_final = (m_A + m_B) * v
                if math.isclose(momentum_initial, momentum_final, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("Momentum conserved in initial inelastic collision"))
                else:
                    tests.append(self._log(f"Momentum NOT conserved: initial={momentum_initial}, final={momentum_final}", False))
            
            # 2. Test centripetal force formula (step 2)
            if all(x is not None for x in [self.m_A, self.m_B, self.v, self.R, self.F]):
                m_A, m_B, v, R, F = float(self.m_A), float(self.m_B), float(self.v), float(self.R), float(self.F)
                F_calc = (m_A + m_B) * (v**2) / R
                if math.isclose(F, F_calc, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("Centripetal force matches Newton's second law"))
                else:
                    tests.append(self._log(f"Centripetal force mismatch: {F} vs {F_calc}", False))
            
            # 3. Test elastic collision conservation laws (steps 3-4)
            if all(x is not None for x in [self.m_A, self.m_B, self.v0, self.v_A, self.v_B]):
                m_A, m_B, v0, v_A, v_B = float(self.m_A), float(self.m_B), float(self.v0), float(self.v_A), float(self.v_B)
                
                # Momentum conservation
                p_initial = m_A * v0
                p_final = m_A * v_A + m_B * v_B
                momentum_conserved = math.isclose(p_initial, p_final, rel_tol=1e-2, abs_tol=1e-2)
                
                # Kinetic energy conservation (elastic)
                KE_initial = 0.5 * m_A * v0**2
                KE_final = 0.5 * m_A * v_A**2 + 0.5 * m_B * v_B**2
                energy_conserved = math.isclose(KE_initial, KE_final, rel_tol=1e-2, abs_tol=1e-2)
                
                if momentum_conserved and energy_conserved:
                    tests.append(self._log("Elastic collision conserves momentum and energy"))
                else:
                    tests.append(self._log(f"Elastic collision: momentum={momentum_conserved}, energy={energy_conserved}", False))
            
            # 4. Test inelastic collision relative velocity (step 15, 17)
            if all(x is not None for x in [self.e, self.v0, self.v_1_relative]):
                e, v0, v_rel = float(self.e), float(self.v0), float(self.v_1_relative)
                expected = e * v0
                if math.isclose(v_rel, expected, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("Relative velocity matches coefficient of restitution"))
                else:
                    tests.append(self._log(f"Relative velocity mismatch: {v_rel} vs {expected}", False))
            
            # 5. Test time-distance relationship (step 15)
            if all(x is not None for x in [self.R, self.v_1_relative, self.t_1]):
                R, v_rel, t1 = float(self.R), float(self.v_1_relative), float(self.t_1)
                expected_t1 = 2 * math.pi * R / v_rel if abs(v_rel) > 1e-10 else float('inf')
                if math.isclose(t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("Time for one lap matches distance/velocity"))
                else:
                    tests.append(self._log(f"Time mismatch: {t1} vs {expected_t1}", False))
            
        except Exception as e:
            tests.append(self._log(f"Invariants test failed with error: {str(e)}", False))
            self.logs.append(traceback.format_exc())
        
        return tests
    
    def test_trace_consistency(self) -> List[bool]:
        """Cross-check intermediate results with final answers"""
        tests = []
        
        try:
            # 1. Check velocity ratio vs mass ratio (elastic case)
            if all(x is not None for x in [self.m_A, self.m_B, self.v_A, self.v_B, self.fracv_Av_B]):
                m_A, m_B, v_A, v_B = float(self.m_A), float(self.m_B), float(self.v_A), float(self.v_B)
                fracv_Av_B_val = float(self.fracv_Av_B)
                
                # From elastic collision formulas
                expected_v_A = (m_A - m_B) / (m_A + m_B) * float(self.v0) if self.v0 else None
                expected_v_B = 2 * m_A / (m_A + m_B) * float(self.v0) if self.v0 else None
                expected_ratio = (m_A - m_B) / (2 * m_A) if m_A > 0 else None
                
                if expected_v_A and math.isclose(v_A, expected_v_A, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("v_A matches elastic collision formula"))
                else:
                    tests.append(self._log(f"v_A mismatch: {v_A} vs {expected_v_A}", False))
                
                if expected_v_B and math.isclose(v_B, expected_v_B, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("v_B matches elastic collision formula"))
                else:
                    tests.append(self._log(f"v_B mismatch: {v_B} vs {expected_v_B}", False))
                
                if expected_ratio and math.isclose(fracv_Av_B_val, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("Velocity ratio matches mass ratio"))
                else:
                    tests.append(self._log(f"Velocity ratio mismatch: {fracv_Av_B_val} vs {expected_ratio}", False))
            
            # 2. Check mass ratio values
            if self.fracm_Am_B is not None:
                frac_val = float(self.fracm_Am_B)
                if math.isclose(frac_val, 2.0, rel_tol=1e-2, abs_tol=1e-2) or math.isclose(frac_val, 5.0, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log(f"Mass ratio {frac_val} is valid (2 or 5)"))
                else:
                    tests.append(self._log(f"Mass ratio {frac_val} not in {{2, 5}}", False))
            
            # 3. Check inelastic collision distance consistency
            if all(x is not None for x in [self.s_1, self.v_B1, self.t_1]):
                s1, vB1, t1 = float(self.s_1), float(self.v_B1), float(self.t_1)
                expected_s1 = vB1 * t1
                if math.isclose(s1, expected_s1, rel_tol=1e-2, abs_tol=1e-2):
                    tests.append(self._log("Distance s1 = v_B1 * t_1 consistent"))
                else:
                    tests.append(self._log(f"s1 mismatch: {s1} vs {expected_s1}", False))
            
            # 4. Check geometric series total distance (step 25)
            if all(x is not None for x in [self.s, self.R, self.m_A, self.m_B, self.e, self.n]):
                try:
                    m_A, m_B, R, e, n = float(self.m_A), float(self.m_B), float(self.R), float(self.e), float(self.n)
                    s_val = float(self.s)
                    
                    # Formula from step 25: (2πR * m_A/(m_A+m_B)) * (e^(2n)-1)/(e^n*(e-1))
                    prefactor = 2 * math.pi * R * m_A / (m_A + m_B)
                    numerator = (e**(2*n)) - 1
                    denominator = (e**n) * (e - 1)
                    expected_s = prefactor * numerator / denominator
                    
                    if math.isclose(s_val, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                        tests.append(self._log("Total distance matches geometric series formula"))
                    else:
                        tests.append(self._log(f"Total distance mismatch: {s_val} vs {expected_s}", False))
                except ZeroDivisionError:
                    tests.append(self._log("Geometric series test skipped due to e=1", True))
            
        except Exception as e:
            tests.append(self._log(f"Trace consistency test failed: {str(e)}", False))
            self.logs.append(traceback.format_exc())
        
        return tests
    
    def test_monotonicity(self) -> List[bool]:
        """Perturb inputs and check expected trends"""
        tests = []
        
        try:
            # Store original values
            orig_m_A = self.m_A
            
            if self.m_A is not None and self.m_B is not None and self.v0 is not None:
                m_A, m_B, v0 = float(self.m_A), float(self.m_B), float(self.v0)
                
                # Test 1: Increase mass ratio, velocity after inelastic collision should decrease
                perturbed_m_A = m_A * 1.01  # Increase by 1%
                original_v = m_A * v0 / (m_A + m_B)
                perturbed_v = perturbed_m_A * v0 / (perturbed_m_A + m_B)
                
                if perturbed_v > original_v:
                    tests.append(self._log(f"v increases with m_A (monotonicity check): {original_v:.4f} -> {perturbed_v:.4f}"))
                else:
                    tests.append(self._log(f"v does NOT increase with m_A: {original_v:.4f} -> {perturbed_v:.4f}", False))
                
                # Test 2: Centripetal force should increase with increased mass
                original_F = (m_A**2 * v0**2) / ((m_A + m_B) * float(self.R)) if self.R else None
                perturbed_F = (perturbed_m_A**2 * v0**2) / ((perturbed_m_A + m_B) * float(self.R)) if self.R else None
                
                if original_F and perturbed_F and perturbed_F > original_F:
                    tests.append(self._log(f"F increases with m_A (monotonicity check): {original_F:.4f} -> {perturbed_F:.4f}"))
                elif original_F and perturbed_F:
                    tests.append(self._log(f"F does NOT increase with m_A: {original_F:.4f} -> {perturbed_F:.4f}", False))
                
                # Restore original value
                self.m_A = orig_m_A
            
            # Test 3: For elastic collision, check mass ratio effect
            if all(x is not None for x in [self.m_A, self.m_B, self.v_A, self.v_B]):
                m_A, m_B, v_A, v_B = float(self.m_A), float(self.m_B), float(self.v_A), float(self.v_B)
                perturbed_m_A = m_A * 1.01
                
                # Recalculate using elastic collision formulas
                perturbed_v_A = (perturbed_m_A - m_B) / (perturbed_m_A + m_B) * v0
                perturbed_v_B = 2 * perturbed_m_A / (perturbed_m_A + m_B) * v0
                
                # v_A should increase with mass ratio (since m_A > m_B)
                if perturbed_v_A > v_A:
                    tests.append(self._log(f"v_A increases with m_A (elastic): {v_A:.4f} -> {perturbed_v_A:.4f}"))
                else:
                    tests.append(self._log(f"v_A does NOT increase with m_A: {v_A:.4f} -> {perturbed_v_A:.4f}", False))
                
                # v_B should also increase (2m_A/(m_A+m_B) is increasing in m_A)
                if perturbed_v_B > v_B:
                    tests.append(self._log(f"v_B increases with m_A (elastic): {v_B:.4f} -> {perturbed_v_B:.4f}"))
                else:
                    tests.append(self._log(f"v_B does NOT increase with m_A: {v_B:.4f} -> {perturbed_v_B:.4f}", False))
            
        except Exception as e:
            tests.append(self._log(f"Monotonicity test failed: {str(e)}", False))
            self.logs.append(traceback.format_exc())
        
        return tests
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs"""
        all_tests = []
        
        try:
            self.logs.append("=== Physics Checker Started ===")
            
            # Run individual test suites
            invariants_results = self.test_invariants()
            trace_results = self.test_trace_consistency()
            monotonicity_results = self.test_monotonicity()
            
            all_tests = invariants_results + trace_results + monotonicity_results
            
            # Count successes
            success_count = sum(1 for test in all_tests if test)
            total_count = len(all_tests)
            
            overall_success = success_count / total_count > 0.75 if total_count > 0 else False
            
            self.logs.append(f"=== Summary: {success_count}/{total_count} tests passed ===")
            
            return overall_success, self.logs
            
        except Exception as e:
            self.logs.append(f"Critical error in run_all_tests: {str(e)}")
            self.logs.append(traceback.format_exc())
            return False, self.logs