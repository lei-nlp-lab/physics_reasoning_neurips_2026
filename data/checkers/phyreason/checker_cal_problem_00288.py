import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary inputs and results from the solution instance.
        Use getattr and .get() to avoid AttributeError in the sandbox.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe access and default values
        self.s1 = getattr(solution_instance, 's1', 5.0)  # inclined plane length in meters
        self.mu = getattr(solution_instance, 'mu', 0.4)  # coefficient of kinetic friction
        self.theta_deg = getattr(solution_instance, 'theta_deg', 37.0)  # angle in degrees
        self.g = getattr(solution_instance, 'g', 10.0)  # gravitational acceleration (used in solution)
        
        # Convert angle to radians for calculations
        self.theta = math.radians(self.theta_deg)
        
        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        self.s2 = getattr(solution_instance, 's2', results.get('s_2', 0.0))
        self.W_g = getattr(solution_instance, 'W_g', results.get('W_g', 0.0))
        self.W_f = getattr(solution_instance, 'W_f', results.get('W_f', 0.0))
        
        # For symbolic calculations, use sympy symbols
        self.m = sp.symbols('m', positive=True)
        self.symbolic_v = sp.symbols('v')
        self.symbolic_s2 = sp.symbols('s2')
        
    def test_invariants(self):
        """
        Verify fundamental physical laws: work-energy theorem and energy conservation.
        """
        logs = []
        success = True
        
        try:
            # 1. Check velocity at bottom of incline using work-energy theorem
            # v = sqrt(2 * g * s1 * (sinθ - μ cosθ))
            expected_v = math.sqrt(2 * self.g * self.s1 * 
                                  (math.sin(self.theta) - self.mu * math.cos(self.theta)))
            
            if not math.isclose(self.v, expected_v, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Velocity invariant failed: v={self.v:.3f}, expected={expected_v:.3f}")
                success = False
            else:
                logs.append(f"Velocity invariant passed: v={self.v:.3f} ≈ {expected_v:.3f}")
                
            # 2. Check horizontal distance using work-energy theorem
            # s2 = v² / (2 * μ * g)
            expected_s2 = self.v**2 / (2 * self.mu * self.g)
            
            if not math.isclose(self.s2, expected_s2, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Distance invariant failed: s2={self.s2:.3f}, expected={expected_s2:.3f}")
                success = False
            else:
                logs.append(f"Distance invariant passed: s2={self.s2:.3f} ≈ {expected_s2:.3f}")
                
            # 3. Energy conservation: Initial PE = Work done by friction (total)
            # m*g*h = m*g*s1*sinθ = μ*m*g*cosθ*s1 + μ*m*g*s2
            h = self.s1 * math.sin(self.theta)
            initial_pe = self.m * self.g * h  # symbolic
            friction_incline = self.mu * self.m * self.g * math.cos(self.theta) * self.s1
            friction_horizontal = self.mu * self.m * self.g * self.s2
            total_friction_work = friction_incline + friction_horizontal
            
            # Symbolic check: initial_pe should equal total_friction_work
            energy_eq = sp.simplify(initial_pe - total_friction_work)
            if energy_eq != 0:
                # If symbolic fails, try numerical with m=1
                m_val = 1.0
                initial_pe_num = m_val * self.g * h
                total_friction_num = (self.mu * m_val * self.g * math.cos(self.theta) * self.s1 +
                                    self.mu * m_val * self.g * self.s2)
                
                if not math.isclose(initial_pe_num, total_friction_num, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"Energy conservation failed: PE={initial_pe_num:.3f}, "
                              f"Friction work={total_friction_num:.3f}")
                    success = False
                else:
                    logs.append("Energy conservation passed (numerical fallback)")
            else:
                logs.append("Energy conservation passed (symbolic)")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        try:
            # 1. Check gravity work consistency
            # W_g = m*g*s1*sinθ
            m_val = 1.0  # use m=1 for numerical check
            expected_W_g = m_val * self.g * self.s1 * math.sin(self.theta)
            
            if not math.isclose(self.W_g, expected_W_g, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Gravity work inconsistent: W_g={self.W_g:.3f}, "
                          f"expected={expected_W_g:.3f}")
                success = False
            else:
                logs.append(f"Gravity work consistent: W_g={self.W_g:.3f}")
                
            # 2. Check friction work on incline consistency
            # W_f_incline = -μ*m*g*cosθ*s1
            expected_W_f_incline = -self.mu * m_val * self.g * math.cos(self.theta) * self.s1
            
            if not math.isclose(self.W_f, expected_W_f_incline, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Friction work inconsistent: W_f={self.W_f:.3f}, "
                          f"expected={expected_W_f_incline:.3f}")
                success = False
            else:
                logs.append(f"Friction work consistent: W_f={self.W_f:.3f}")
                
            # 3. Check work-energy theorem on incline: W_g + W_f = 0.5*m*v²
            net_work_incline = self.W_g + self.W_f
            ke_at_bottom = 0.5 * m_val * self.v**2
            
            if not math.isclose(net_work_incline, ke_at_bottom, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Work-energy theorem (incline) failed: "
                          f"net_work={net_work_incline:.3f}, KE={ke_at_bottom:.3f}")
                success = False
            else:
                logs.append(f"Work-energy theorem (incline) consistent")
                
            # 4. Check work-energy theorem on horizontal: -μ*m*g*s2 = -0.5*m*v²
            friction_horizontal = -self.mu * m_val * self.g * self.s2
            ke_loss = -0.5 * m_val * self.v**2
            
            if not math.isclose(friction_horizontal, ke_loss, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"Work-energy theorem (horizontal) failed: "
                          f"friction={friction_horizontal:.3f}, KE_loss={ke_loss:.3f}")
                success = False
            else:
                logs.append(f"Work-energy theorem (horizontal) consistent")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs

    def test_monotonicity(self):
        """
        Perturb friction coefficient and verify expected trend in horizontal distance.
        """
        logs = []
        success = True
        
        try:
            # Original parameters
            original_mu = self.mu
            original_s2 = self.s2
            
            # Perturb: increase friction by 10%
            perturbed_mu = original_mu * 1.1
            
            # Recalculate s2 using the derived formula: s2 = s1*(sinθ/μ - cosθ)
            perturbed_s2 = self.s1 * (math.sin(self.theta)/perturbed_mu - math.cos(self.theta))
            
            # Expected: higher friction → shorter distance
            if perturbed_s2 >= original_s2:
                logs.append(f"Monotonicity failed: μ increased from {original_mu:.3f} to {perturbed_mu:.3f}, "
                          f"but s2 changed from {original_s2:.3f} to {perturbed_s2:.3f}")
                success = False
            else:
                logs.append(f"Monotonicity passed: μ↑ → s2↓ "
                          f"({original_s2:.3f} → {perturbed_s2:.3f})")
                
            # Restore original value (important!)
            self.mu = original_mu
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            # Ensure restoration even on error
            self.mu = getattr(self.solution, 'mu', 0.4)
            
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success status and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== {test_name} ===")
            success, logs = test_method()
            all_logs.extend(logs)
            if not success:
                overall_success = False
            all_logs.append("")  # blank line for readability
            
        return overall_success, all_logs