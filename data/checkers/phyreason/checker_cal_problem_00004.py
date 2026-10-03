import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance (no direct attribute access)
        self.U = getattr(solution_instance, 'U', 380.0)
        self.R = getattr(solution_instance, 'R', 0.25)
        self.v = getattr(solution_instance, 'v', 2.0)
        self.I = getattr(solution_instance, 'I', 40.0)
        self.m1 = getattr(solution_instance, 'm1', 100.0)
        self.m2 = getattr(solution_instance, 'm2', 1200.0)
        self.m0 = getattr(solution_instance, 'm0', 40.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safely extract results (with fallback to 0.0)
        results = getattr(solution_instance, 'results', {})
        self.T1 = results.get('T_1', 0.0)
        self.T2 = results.get('T_2', 0.0)
        self.sintheta = results.get('sintheta', 0.0)
        self.k = results.get('k', 0.0)
        self.a = results.get('a', 0.0)
        self.L = results.get('L', 0.0)
        
        # Store logs for all tests
        self.logs = []
        
    def _log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify energy conservation, Newton's 1st/2nd laws based on KB."""
        try:
            # 1. Energy conservation in step1: UI = I^2 R + T1 v
            lhs_energy = self.U * self.I
            rhs_energy = self.I**2 * self.R + self.T1 * self.v
            energy_check = math.isclose(lhs_energy, rhs_energy, rel_tol=1e-2, abs_tol=1e-2)
            if not energy_check:
                self._log(f"Energy conservation failed: UI={lhs_energy:.2f}, I²R+T1v={rhs_energy:.2f}")
            
            # 2. Newton's 1st law for counterweight (step2): T2 = m0 * g
            T2_expected = self.m0 * self.g
            T2_check = math.isclose(self.T2, T2_expected, rel_tol=1e-2, abs_tol=1e-2)
            if not T2_check:
                self._log(f"Newton's 1st law (counterweight) failed: T2={self.T2:.2f}, m0*g={T2_expected:.2f}")
            
            # 3. Newton's 1st law for trolley with motor on (uniform uphill motion)
            # Forces parallel to incline: T1 + T2 = (m1+m2)g sinθ + k(m1+m2)g
            total_mass = self.m1 + self.m2
            lhs_force = self.T1 + self.T2
            rhs_force = total_mass * self.g * self.sintheta + self.k * total_mass * self.g
            force_check = math.isclose(lhs_force, rhs_force, rel_tol=1e-2, abs_tol=1e-2)
            if not force_check:
                self._log(f"Newton's 1st law (trolley) failed: T1+T2={lhs_force:.2f}, (m1+m2)g sinθ + k(m1+m2)g={rhs_force:.2f}")
            
            # 4. Newton's 2nd law after motor off (step6)
            # (m1+m2)g sinθ + k(m1+m2)g - m0*g = (m1+m2+m0)*a
            lhs_accel = total_mass * self.g * self.sintheta + self.k * total_mass * self.g - self.m0 * self.g
            rhs_accel = (total_mass + self.m0) * self.a
            accel_check = math.isclose(lhs_accel, rhs_accel, rel_tol=1e-2, abs_tol=1e-2)
            if not accel_check:
                self._log(f"Newton's 2nd law failed: left={lhs_accel:.2f}, right={rhs_accel:.2f}")
            
            return energy_check and T2_check and force_check and accel_check
            
        except Exception as e:
            self._log(f"test_invariants error: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # 1. Check step5 results (sinθ and k) satisfy both equations simultaneously
            total_mass = self.m1 + self.m2
            
            # Equation from uniform motion with motor: T1 + T2 = (m1+m2)g sinθ + k(m1+m2)g
            eq1_lhs = self.T1 + self.T2
            eq1_rhs = total_mass * self.g * self.sintheta + self.k * total_mass * self.g
            eq1_check = math.isclose(eq1_lhs, eq1_rhs, rel_tol=1e-2, abs_tol=1e-2)
            
            # Equation from deceleration after motor off: (m1+m2)g sinθ + k(m1+m2)g - m0*g = (m1+m2+m0)*a
            eq2_lhs = total_mass * self.g * self.sintheta + self.k * total_mass * self.g - self.m0 * self.g
            eq2_rhs = (total_mass + self.m0) * self.a
            eq2_check = math.isclose(eq2_lhs, eq2_rhs, rel_tol=1e-2, abs_tol=1e-2)
            
            if not (eq1_check and eq2_check):
                self._log(f"Trace consistency failed: eq1={eq1_check}, eq2={eq2_check}")
            
            # 2. Check kinematic equation in step7: v² = 2aL
            lhs_kinematic = self.v**2
            rhs_kinematic = 2 * self.a * self.L
            kinematic_check = math.isclose(lhs_kinematic, rhs_kinematic, rel_tol=1e-2, abs_tol=1e-2)
            if not kinematic_check:
                self._log(f"Kinematic equation failed: v²={lhs_kinematic:.2f}, 2aL={rhs_kinematic:.2f}")
            
            return eq1_check and eq2_check and kinematic_check
            
        except Exception as e:
            self._log(f"test_trace_consistency error: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb an input (m0) and check logical trend, then restore."""
        try:
            # Perturb m0 by +10% and check L changes appropriately
            original_m0 = self.m0
            perturbed_m0 = original_m0 * 1.1  # Increase by 10%
            
            # Recalculate intermediate quantities using symbolic relationships
            # 1. Recalculate T2 (Newton's 1st law for counterweight)
            T2_perturbed = perturbed_m0 * self.g
            
            # 2. Keep T1 unchanged (from energy conservation, depends on motor parameters)
            T1 = self.T1
            
            # 3. Solve for new sintheta and k from two equations:
            #    Equation A: T1 + T2_perturbed = (m1+m2)g sinθ + k(m1+m2)g
            #    Equation B: (m1+m2)g sinθ + k(m1+m2)g - perturbed_m0*g = (m1+m2+perturbed_m0)*a
            #    But we don't know the new a. However, we can use the original sintheta and k
            #    to compute what the new acceleration would be, then check L trend.
            
            # Instead, we directly test physical intuition: increasing counterweight mass
            # should reduce the deceleration after motor off (since counterweight provides
            # more upward pull), thus increasing stopping distance L.
            
            # Using original sintheta and k (assume they remain valid for small perturbation)
            total_mass = self.m1 + self.m2
            a_perturbed = (total_mass * self.g * self.sintheta + self.k * total_mass * self.g - perturbed_m0 * self.g) / (total_mass + perturbed_m0)
            
            # New stopping distance
            L_perturbed = self.v**2 / (2 * abs(a_perturbed))
            
            # Check monotonicity: larger m0 -> larger L (if a decreases in magnitude)
            # Since original a is positive in the equation (deceleration magnitude),
            # we compare |a| values
            a_original_mag = abs(self.a)
            a_perturbed_mag = abs(a_perturbed)
            
            if a_perturbed_mag < a_original_mag:
                expected_L_increase = L_perturbed > self.L
                if not expected_L_increase:
                    self._log(f"Monotonicity failed: m0↑ ({original_m0:.1f}→{perturbed_m0:.1f}) should increase L, but L_perturbed={L_perturbed:.3f} ≤ original L={self.L:.3f}")
                    return False
            else:
                self._log(f"Monotonicity check: a magnitude didn't decrease as expected")
                return False
                
            return True
            
        except Exception as e:
            self._log(f"test_monotonicity error: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        success = test1 and test2 and test3
        
        summary = f"Tests: invariants={test1}, trace={test2}, monotonicity={test3}"
        self._log(summary)
        
        return success, self.logs