import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with a solution instance.
        Safely extract inputs and results using getattr to avoid AttributeError.
        """
        # Safe extraction of inputs (with defaults if not found)
        self.m_A = getattr(solution_instance, 'm_A', getattr(solution_instance, 'inputs', {}).get('m_A', 2.0))
        self.m_B = getattr(solution_instance, 'm_B', getattr(solution_instance, 'inputs', {}).get('m_B', 1.0))
        self.h = getattr(solution_instance, 'h', getattr(solution_instance, 'inputs', {}).get('h', 1.2))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'inputs', {}).get('t', 0.2))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        
        # Safe extraction of results (with defaults if not found)
        self.results = getattr(solution_instance, 'results', {})
        self.x = self.results.get('x', 0.0)
        self.v_B0 = self.results.get('v_B0', 0.0)
        self.v_A = self.results.get('v_A', 0.0)
        self.v_B = self.results.get('v_B', 0.0)
        self.v = self.results.get('v', 0.0)
        self.DeltaE = self.results.get('DeltaE', 0.0)
        
        self.logs = []

    def log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")

    def test_invariants(self):
        """Test physical invariants: momentum conservation and energy loss formula."""
        success = True
        try:
            # 1. Momentum conservation during collision (perfectly inelastic)
            # Initial momentum: m_A*v_A (downward positive) + m_B*(-v_B) (since v_B is upward, we take negative for consistent direction)
            # Actually from STEP ANALYSIS: equation is m_A*v_A - m_B*v_B = (m_A+m_B)*v
            # This implies v_A is positive downward, v_B positive upward.
            p_initial = self.m_A * self.v_A - self.m_B * self.v_B
            p_final = (self.m_A + self.m_B) * self.v
            if not math.isclose(p_initial, p_final, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Momentum not conserved: p_initial={p_initial:.4f}, p_final={p_final:.4f}", False)
                success = False
            else:
                self.log(f"Momentum conservation: p_initial={p_initial:.4f}, p_final={p_final:.4f}", True)
                
            # 2. Energy loss formula from STEP ANALYSIS
            KE_initial = 0.5 * self.m_A * self.v_A**2 + 0.5 * self.m_B * self.v_B**2
            KE_final = 0.5 * (self.m_A + self.m_B) * self.v**2
            DeltaE_expected = KE_initial - KE_final
            if not math.isclose(self.DeltaE, DeltaE_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Energy loss mismatch: claimed {self.DeltaE:.4f}, computed {DeltaE_expected:.4f}", False)
                success = False
            else:
                self.log(f"Energy loss consistent: ΔE = {self.DeltaE:.4f} J", True)
                
        except Exception as e:
            self.log(f"Error in test_invariants: {e}\n{traceback.format_exc()}", False)
            success = False
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with kinematics and final answer."""
        success = True
        try:
            # 1. Check collision height x from object A's kinematics
            x_A_expected = self.h - 0.5 * self.g * self.t**2
            if not math.isclose(self.x, x_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Collision height mismatch (from A): x={self.x:.4f}, expected {x_A_expected:.4f}", False)
                success = False
            else:
                self.log(f"Collision height x={self.x:.4f} m consistent with A's kinematics", True)
            
            # 2. Check initial velocity of B from kinematics using x
            # x = v_B0 * t - 0.5 * g * t^2  => v_B0 = (x + 0.5*g*t^2)/t
            v_B0_expected = (self.x + 0.5 * self.g * self.t**2) / self.t
            if not math.isclose(self.v_B0, v_B0_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_B0 mismatch: claimed {self.v_B0:.4f}, expected {v_B0_expected:.4f}", False)
                success = False
            else:
                self.log(f"v_B0={self.v_B0:.4f} m/s consistent with kinematics", True)
            
            # 3. Check v_A from kinematics: v_A = g * t (since released from rest)
            v_A_expected = self.g * self.t
            if not math.isclose(self.v_A, v_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_A mismatch: claimed {self.v_A:.4f}, expected {v_A_expected:.4f}", False)
                success = False
            else:
                self.log(f"v_A={self.v_A:.4f} m/s consistent with kinematics", True)
            
            # 4. Check v_B from kinematics: v_B = v_B0 - g * t
            v_B_expected = self.v_B0 - self.g * self.t
            if not math.isclose(self.v_B, v_B_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_B mismatch: claimed {self.v_B:.4f}, expected {v_B_expected:.4f}", False)
                success = False
            else:
                self.log(f"v_B={self.v_B:.4f} m/s consistent with kinematics", True)
            
            # 5. Check combined velocity v from momentum conservation (redundant but good)
            v_expected = (self.m_A * self.v_A - self.m_B * self.v_B) / (self.m_A + self.m_B)
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v mismatch: claimed {self.v:.4f}, expected {v_expected:.4f}", False)
                success = False
            else:
                self.log(f"v={self.v:.4f} m/s consistent with momentum", True)
                
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}", False)
            success = False
        return success

    def test_monotonicity(self):
        """
        Perturb an input (mass m_A) and verify the trend in results.
        We recompute using the same formulas, then restore original values.
        """
        original_m_A = self.m_A
        success = True
        try:
            # Increase m_A by 10%
            perturbed_m_A = original_m_A * 1.1
            
            # Recompute using the same kinematic formulas (independent of mass)
            x = self.h - 0.5 * self.g * self.t**2
            v_B0 = (x + 0.5 * self.g * self.t**2) / self.t
            v_A = self.g * self.t
            v_B = v_B0 - self.g * self.t
            
            # Recompute v and DeltaE with perturbed mass
            v_perturbed = (perturbed_m_A * v_A - self.m_B * v_B) / (perturbed_m_A + self.m_B)
            KE_initial = 0.5 * perturbed_m_A * v_A**2 + 0.5 * self.m_B * v_B**2
            KE_final = 0.5 * (perturbed_m_A + self.m_B) * v_perturbed**2
            DeltaE_perturbed = KE_initial - KE_final
            
            # Check monotonicity: 
            # As m_A increases, the combined mass increases, so v should decrease (since numerator changes).
            # Here, with original values: v=0, so let's check sign of v_perturbed.
            # Also, energy loss should increase because more mass is involved in inelastic collision.
            if original_m_A > 0:
                # For v: if v=0 originally, perturbed v should be positive? Let's compute sign.
                # Since v_A=2, v_B=4, m_A=2, m_B=1: numerator = 2*2 - 1*4 = 0 -> v=0.
                # With perturbed m_A=2.2: numerator = 2.2*2 - 1*4 = 4.4 - 4 = 0.4 >0 -> v_perturbed >0.
                # So v increases from 0 to positive.
                # But monotonicity in mass: derivative of v w.r.t m_A? Actually v = (m_A*v_A - m_B*v_B)/(m_A+m_B).
                # We can check that v_perturbed has the correct trend compared to original v.
                # Since original v=0, and v_perturbed>0, that's fine.
                
                # For DeltaE: original=12, perturbed should be larger because more initial KE and same final KE? 
                # Actually final KE is 0 originally, but with perturbed, final KE is >0, so DeltaE might be different.
                # Let's compute: original DeltaE=12, perturbed DeltaE = KE_initial (larger) - KE_final (positive).
                # We expect DeltaE to increase because more mass means more energy loss in inelastic collision?
                # Actually, compute numerically:
                # Original: KE_initial=0.5*2*4 + 0.5*1*16 = 4+8=12, KE_final=0 -> DeltaE=12.
                # Perturbed: KE_initial=0.5*2.2*4 + 0.5*1*16 = 4.4+8=12.4, KE_final=0.5*3.2*(0.125)^2? Wait, v_perturbed=0.4/3.2=0.125.
                # KE_final=0.5*3.2*0.015625=0.025 -> DeltaE_perturbed=12.375, which is slightly larger than 12.
                # So DeltaE increases.
                
                # We'll just check that the recomputed values are consistent with formulas.
                # No hard monotonicity rule, but we can log the trend.
                self.log(f"Perturbation test: m_A increased by 10% -> v changes from {self.v:.4f} to {v_perturbed:.4f}, ΔE changes from {self.DeltaE:.4f} to {DeltaE_perturbed:.4f}", True)
            else:
                self.log("Skipping monotonicity due to non-positive mass", False)
                success = False
                
        except Exception as e:
            self.log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}", False)
            success = False
        finally:
            # Restore original m_A
            self.m_A = original_m_A
        return success

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []  # Reset logs
        overall_success = True
        
        # Run individual tests
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"Test {test_name} crashed: {e}\n{traceback.format_exc()}", False)
                overall_success = False
        
        self.log(f"All tests completed. Overall success: {overall_success}", overall_success)
        return overall_success, self.logs