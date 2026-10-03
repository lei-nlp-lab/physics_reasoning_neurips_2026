import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallback
        self.v_total = self.results.get('v_total', 0.0)
        self.M = self.results.get('M', 0.0)
        self.a = self.results.get('a', 0.0)
        self.t_1 = self.results.get('t_1', 0.0)
        self.x = self.results.get('x', 0.0)
        self.t_2 = self.results.get('t_2', 0.0)
        self.t = self.results.get('t', 0.0)
        
        # Problem constants
        self.L_AB = 10.0          # m
        self.v_belt = 1.0         # m/s
        self.m_block = 0.04       # kg
        self.m_pellet = 0.01      # kg
        self.v0_pellet = 35.0     # m/s
        self.mu = 0.3             # dimensionless
        self.g = 10.0             # m/s²
        
        # Target answers
        self.target_v_total = 7.0  # m/s
        self.target_t = 4.0        # s
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def _is_close(self, a, b):
        """Robust fuzzy comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        """Verify physics conservation laws."""
        logs = []
        
        try:
            # 1. Conservation of momentum during collision
            initial_momentum = self.m_pellet * self.v0_pellet
            final_momentum = self.M * self.v_total
            if not self._is_close(initial_momentum, final_momentum):
                logs.append(f"ERROR: Momentum not conserved. Initial: {initial_momentum}, Final: {final_momentum}")
            else:
                logs.append("INFO: Momentum conservation verified.")
                
            # 2. Newton's Second Law for friction
            expected_friction_force = self.mu * self.M * self.g
            calculated_force = self.M * self.a
            if not self._is_close(expected_friction_force, calculated_force):
                logs.append(f"ERROR: Newton's 2nd Law mismatch. F_friction: {expected_friction_force}, Ma: {calculated_force}")
            else:
                logs.append("INFO: Newton's 2nd Law (friction) verified.")
                
            # 3. Time consistency
            if not self._is_close(self.t, self.t_1 + self.t_2):
                logs.append(f"ERROR: Time sum inconsistent. t: {self.t}, t1+t2: {self.t_1 + self.t_2}")
            else:
                logs.append("INFO: Time summation verified.")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        try:
            # Step 1: v_total from momentum conservation
            v_total_calc = (self.m_pellet * self.v0_pellet) / (self.m_pellet + self.m_block)
            if not self._is_close(self.v_total, v_total_calc):
                logs.append(f"ERROR: v_total mismatch. Result: {self.v_total}, Calculated: {v_total_calc}")
            else:
                logs.append(f"INFO: v_total consistent: {self.v_total}")
                
            # Step 2: Combined mass
            M_calc = self.m_pellet + self.m_block
            if not self._is_close(self.M, M_calc):
                logs.append(f"ERROR: M mismatch. Result: {self.M}, Calculated: {M_calc}")
            else:
                logs.append(f"INFO: M consistent: {self.M}")
                
            # Step 3: Acceleration from friction
            a_calc = self.mu * self.g
            if not self._is_close(self.a, a_calc):
                logs.append(f"ERROR: a mismatch. Result: {self.a}, Calculated: {a_calc}")
            else:
                logs.append(f"INFO: a consistent: {self.a}")
                
            # Step 4: Time to decelerate to belt speed
            if self.a > 0:
                t1_calc = (self.v_total - self.v_belt) / self.a
                if not self._is_close(self.t_1, t1_calc):
                    logs.append(f"ERROR: t1 mismatch. Result: {self.t_1}, Calculated: {t1_calc}")
                else:
                    logs.append(f"INFO: t1 consistent: {self.t_1}")
                    
            # Step 5: Displacement during deceleration
            x_calc = ((self.v_total + self.v_belt) / 2) * self.t_1
            if not self._is_close(self.x, x_calc):
                logs.append(f"ERROR: x mismatch. Result: {self.x}, Calculated: {x_calc}")
            else:
                logs.append(f"INFO: x consistent: {self.x}")
                
            # Step 6: Time of constant velocity motion
            if self.v_belt > 0:
                t2_calc = (self.L_AB - self.x) / self.v_belt
                if not self._is_close(self.t_2, t2_calc):
                    logs.append(f"ERROR: t2 mismatch. Result: {self.t_2}, Calculated: {t2_calc}")
                else:
                    logs.append(f"INFO: t2 consistent: {self.t_2}")
                    
            # Step 7: Total time
            t_calc = self.t_1 + self.t_2
            if not self._is_close(self.t, t_calc):
                logs.append(f"ERROR: t mismatch. Result: {self.t}, Calculated: {t_calc}")
            else:
                logs.append(f"INFO: t consistent: {self.t}")
                
            # Check against target answers
            if not self._is_close(self.v_total, self.target_v_total):
                logs.append(f"ERROR: v_total far from target. Result: {self.v_total}, Target: {self.target_v_total}")
            else:
                logs.append(f"INFO: v_total matches target: {self.v_total}")
                
            if not self._is_close(self.t, self.target_t):
                logs.append(f"ERROR: t far from target. Result: {self.t}, Target: {self.target_t}")
            else:
                logs.append(f"INFO: t matches target: {self.t}")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs

    def test_monotonicity(self):
        """Perturb inputs and check physical trends."""
        logs = []
        
        try:
            # Store original values
            original_m_pellet = self.m_pellet
            original_mu = self.mu
            original_t = self.t
            
            # Helper function to recalculate t
            def recalc_t(m_pellet=None, mu=None):
                m_pellet = m_pellet if m_pellet is not None else original_m_pellet
                mu = mu if mu is not None else original_mu
                
                M_temp = m_pellet + self.m_block
                v_total_temp = (m_pellet * self.v0_pellet) / M_temp
                a_temp = mu * self.g
                t1_temp = (v_total_temp - self.v_belt) / a_temp
                x_temp = ((v_total_temp + self.v_belt) / 2) * t1_temp
                t2_temp = (self.L_AB - x_temp) / self.v_belt
                return t1_temp + t2_temp
            
            # Test 1: Increase pellet mass (should affect v_total and timing)
            perturbed_m = original_m_pellet * 1.01  # +1%
            t_perturbed = recalc_t(m_pellet=perturbed_m)
            
            if abs(t_perturbed - original_t) < self.abs_tol:
                logs.append("WARNING: t insensitive to pellet mass perturbation")
            else:
                logs.append(f"INFO: t changes from {original_t} to {t_perturbed} with +1% m_pellet")
                
            # Test 2: Increase friction coefficient (should decrease t1, increase t2)
            perturbed_mu = original_mu * 1.01  # +1%
            t_perturbed = recalc_t(mu=perturbed_mu)
            
            if abs(t_perturbed - original_t) < self.abs_tol:
                logs.append("WARNING: t insensitive to friction perturbation")
            else:
                logs.append(f"INFO: t changes from {original_t} to {t_perturbed} with +1% mu")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            
        return logs

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        all_logs = []
        
        # Run individual tests
        all_logs.append("=== PhysicsChecker Starting ===")
        
        invariants_logs = self.test_invariants()
        all_logs.append("\n--- test_invariants ---")
        all_logs.extend(invariants_logs)
        
        trace_logs = self.test_trace_consistency()
        all_logs.append("\n--- test_trace_consistency ---")
        all_logs.extend(trace_logs)
        
        monotonicity_logs = self.test_monotonicity()
        all_logs.append("\n--- test_monotonicity ---")
        all_logs.extend(monotonicity_logs)
        
        # Determine success
        error_count = sum(1 for log in all_logs if log.startswith("ERROR"))
        success = (error_count == 0)
        
        all_logs.append(f"\n=== Summary: {'PASS' if success else 'FAIL'} with {error_count} error(s) ===")
        
        return success, all_logs