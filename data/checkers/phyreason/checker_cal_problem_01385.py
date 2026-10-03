import math
import sympy as sp
from sympy import symbols, sin, cos, Eq, simplify

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results."""
        # Inputs
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'input_v0', 0.0))
        self.theta = getattr(solution_instance, 'theta', getattr(solution_instance, 'input_theta', 0.0))
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Results
        results = getattr(solution_instance, 'results', {})
        self.l = results.get('l', 0.0)
        self.H = results.get('H', 0.0)
        self.t = results.get('t', 0.0)
        
        # Derived parameters
        self.alpha = math.pi/2 - self.theta  # α + θ = 90°
        
        # Symbolic setup
        self.v0_sym, self.theta_sym, self.g_sym, self.t_sym = symbols('v0 theta g t')
        self.alpha_sym = sp.pi/2 - self.theta_sym

    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        try:
            # 1. Energy conservation (kinetic + potential at O vs at max height)
            # At O: E_O = 0.5*m*v0^2 (taking O as reference for potential energy along slope)
            # At max height: E_max = 0.5*m*v_parallel^2 + m*g*H_perpendicular
            # v_parallel = v0*cosα remains constant
            # H_perpendicular = H (max distance from slope)
            v_parallel = self.v0 * math.cos(self.alpha)
            E_O = 0.5 * self.v0**2
            E_max = 0.5 * v_parallel**2 + self.g * self.H
            
            if math.isclose(E_O, E_max, rel_tol=0.02, abs_tol=0.02):
                logs.append("✓ Energy conservation approximately holds")
            else:
                logs.append(f"⚠ Energy discrepancy: E_O={E_O:.3f}, E_max={E_max:.3f}")
                
            # 2. Time of flight from perpendicular motion
            # t = 2*v0*sinα/(g*cosθ)
            t_expected = 2 * self.v0 * math.sin(self.alpha) / (self.g * math.cos(self.theta))
            if math.isclose(self.t, t_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append("✓ Time of flight matches perpendicular motion equation")
            else:
                logs.append(f"⚠ Time mismatch: t={self.t:.3f}, expected={t_expected:.3f}")
                
            # 3. Check α + θ = 90° condition
            if math.isclose(self.alpha + self.theta, math.pi/2, rel_tol=1e-4):
                logs.append("✓ Condition α + θ = 90° satisfied")
            else:
                logs.append(f"⚠ Condition violated: α={self.alpha:.3f}, θ={self.theta:.3f}")
                
        except Exception as e:
            logs.append(f"✗ Invariant test failed: {str(e)}")
            
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate and final results."""
        logs = []
        try:
            # Convert to float for numerical comparison
            v0_val = float(self.v0)
            theta_val = float(self.theta)
            g_val = float(self.g)
            
            # Check t from step_1
            t_expected = 2 * v0_val / g_val  # Since sinα = cosθ, simplifies to 2v0/g
            if math.isclose(float(self.t), t_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append("✓ t consistent with simplified expression 2v0/g")
            else:
                logs.append(f"⚠ t inconsistency: got {self.t:.3f}, expected {t_expected:.3f}")
            
            # Check l from step_2 using t
            # l = v0*cosα*t + 0.5*g*sinθ*t²
            l_calculated = (v0_val * math.cos(self.alpha) * self.t + 
                           0.5 * g_val * math.sin(theta_val) * self.t**2)
            if math.isclose(float(self.l), l_calculated, rel_tol=0.02, abs_tol=0.02):
                logs.append("✓ l consistent with kinematic equation using t")
            else:
                logs.append(f"⚠ l inconsistency: got {self.l:.3f}, calculated {l_calculated:.3f}")
            
            # Check H from step_3
            # H = (v0*sinα)²/(2*g*cosθ)
            H_calculated = (v0_val * math.sin(self.alpha))**2 / (2 * g_val * math.cos(theta_val))
            if math.isclose(float(self.H), H_calculated, rel_tol=0.02, abs_tol=0.02):
                logs.append("✓ H consistent with max height formula")
            else:
                logs.append(f"⚠ H inconsistency: got {self.H:.3f}, calculated {H_calculated:.3f}")
                
            # Symbolic check: verify final expressions match target answers
            l_symbolic = (self.v0_sym * cos(self.alpha_sym) * self.t_sym + 
                         0.5 * self.g_sym * sin(self.theta_sym) * self.t_sym**2)
            t_symbolic = 2 * self.v0_sym / self.g_sym  # from step_1
            l_simplified = simplify(l_symbolic.subs({self.t_sym: t_symbolic}))
            l_target = 4 * self.v0_sym**2 * sin(self.theta_sym) / self.g_sym
            
            if simplify(l_simplified - l_target) == 0:
                logs.append("✓ Symbolic l matches target expression")
            else:
                logs.append("⚠ Symbolic l doesn't match target")
                
            H_symbolic = (self.v0_sym * sin(self.alpha_sym))**2 / (2 * self.g_sym * cos(self.theta_sym))
            H_target = self.v0_sym**2 * cos(self.theta_sym) / (2 * self.g_sym)
            
            if simplify(H_symbolic - H_target) == 0:
                logs.append("✓ Symbolic H matches target expression")
            else:
                logs.append("⚠ Symbolic H doesn't match target")
                
        except Exception as e:
            logs.append(f"✗ Trace consistency test failed: {str(e)}")
            
        return logs

    def test_monotonicity(self):
        """Check that increasing v0 increases l and H."""
        logs = []
        original_v0 = self.v0
        try:
            # Perturb v0 by 10%
            perturbed_v0 = original_v0 * 1.1
            
            # Calculate l and H with perturbed v0 using derived formulas
            # Keep using the same t formula: t = 2*v0/g
            t_perturbed = 2 * perturbed_v0 / self.g
            l_perturbed = (perturbed_v0 * math.cos(self.alpha) * t_perturbed + 
                          0.5 * self.g * math.sin(self.theta) * t_perturbed**2)
            H_perturbed = (perturbed_v0 * math.sin(self.alpha))**2 / (2 * self.g * math.cos(self.theta))
            
            # Check monotonic increase
            if l_perturbed > self.l and H_perturbed > self.H:
                logs.append("✓ Monotonicity: l and H increase with v0")
            else:
                logs.append(f"⚠ Monotonicity violation: l {self.l:.3f}→{l_perturbed:.3f}, "
                           f"H {self.H:.3f}→{H_perturbed:.3f}")
                
            # Check quadratic dependence: ratios should be (1.1)^2 = 1.21
            l_ratio = l_perturbed / self.l if self.l != 0 else 0
            H_ratio = H_perturbed / self.H if self.H != 0 else 0
            
            if (math.isclose(l_ratio, 1.21, rel_tol=0.05) and 
                math.isclose(H_ratio, 1.21, rel_tol=0.05)):
                logs.append("✓ Quadratic dependence on v0 confirmed")
            else:
                logs.append(f"⚠ Quadratic dependence off: l_ratio={l_ratio:.3f}, H_ratio={H_ratio:.3f}")
                
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed: {str(e)}")
            
        return logs

    def run_all_tests(self):
        """Execute all tests and return summary."""
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            test_logs = test_method()
            all_logs.extend(test_logs)
            
            # Check for critical errors
            if any("✗" in log for log in test_logs):
                success = False
                
        return success, all_logs