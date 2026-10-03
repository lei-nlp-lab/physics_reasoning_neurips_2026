import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs
        self.m = getattr(solution_instance, 'm', 0.1)  # kg
        self.l = getattr(solution_instance, 'l', 0.2)  # m (rod length and rail separation)
        self.E = getattr(solution_instance, 'E', 15.0)  # V (EMF)
        self.r = getattr(solution_instance, 'r', 1.0)   # Ω (internal resistance)
        self.R = getattr(solution_instance, 'R', 1.0)   # Ω (rod resistance)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s^2

        # Safely extract results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.mu = results.get('mu', None)
        self.F = results.get('F', None)
        self.I = results.get('I', None)
        self.B = results.get('B', None)
        self.v = results.get('v', None)
        self.a = results.get('a', None)

        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def test_invariants(self):
        """Verify physical invariants: equilibrium conditions, Ohm's Law, Ampere force."""
        try:
            # 1. CORRECTED Equilibrium at θ=37° and θ=90°
            # At θ=90°: μF = mg
            # At θ=37°: F*cos37 = mg*sin37 + μ*(mg*cos37 + F*sin37)
            if self.mu is not None and self.F is not None:
                # Check θ=90° condition: μF = mg
                left_90 = self.mu * self.F
                right_90 = self.m * self.g
                if not math.isclose(left_90, right_90, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: θ=90° equilibrium violated. μF={left_90}, mg={right_90}")
                    return False
                
                # Check θ=37° condition
                N2 = self.m * self.g * self.cos37 + self.F * self.sin37
                f2 = self.mu * N2
                left_37 = self.F * self.cos37
                right_37 = self.m * self.g * self.sin37 + f2
                if not math.isclose(left_37, right_37, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: θ=37° equilibrium violated. Fcos37={left_37}, mg sin37 + f2={right_37}")
                    return False
                    
                self.log("PASS: Equilibrium conditions satisfied.")
            else:
                self.log("SKIP: Missing mu or F for equilibrium check.")

            # 2. Ohm's Law: I = E / (r + R)
            if self.I is not None:
                I_expected = self.E / (self.r + self.R)
                if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Ohm's Law violated. I={self.I}, expected={I_expected}")
                    return False
                self.log("PASS: Ohm's Law satisfied.")

            # 3. Ampere force: F = B * I * l
            if all(x is not None for x in [self.F, self.B, self.I]):
                F_expected = self.B * self.I * self.l
                if not math.isclose(self.F, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Ampere force formula violated. F={self.F}, expected={F_expected}")
                    return False
                self.log("PASS: Ampere force formula satisfied.")

            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Target answers: B=4/3 T, v=3 m/s, a=6.07 m/s^2
            # Check B magnitude (ignore direction)
            if self.B is not None:
                if not math.isclose(self.B, 4/3, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: B={self.B} not close to 4/3 T.")
                    return False
                self.log(f"PASS: B={self.B:.3f} matches 4/3 T.")

            # Check v
            if self.v is not None:
                if not math.isclose(self.v, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: v={self.v} not close to 3 m/s.")
                    return False
                self.log(f"PASS: v={self.v:.3f} matches 3 m/s.")

            # Check a
            if self.a is not None:
                if not math.isclose(self.a, 6.07, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: a={self.a} not close to 6.07 m/s^2.")
                    return False
                self.log(f"PASS: a={self.a:.3f} matches 6.07 m/s^2.")

            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb mass m and check expected trends in F and mu."""
        try:
            if self.mu is None or self.F is None:
                self.log("SKIP: Missing mu or F for monotonicity test.")
                return True

            # Save original values
            m_original = self.m
            F_original = self.F
            mu_original = self.mu
            
            # Perturb mass by +10%
            self.m = m_original * 1.10
            
            # CORRECTED: From θ=90°: μF = mg, so F_new = (m_new * g) / μ
            # But μ might change? Actually from the equilibrium equations,
            # μ = (mg)/F from θ=90°, and also must satisfy θ=37° condition.
            # When m changes, both F and μ could change.
            # Instead, solve the equilibrium equations with new m:
            # Equation 1: μ_new * F_new = m_new * g
            # Equation 2: F_new * cos37 = m_new*g*sin37 + μ_new*(m_new*g*cos37 + F_new*sin37)
            # Solve for F_new and μ_new
            
            # Let's solve numerically
            g = self.g
            sin37 = self.sin37
            cos37 = self.cos37
            
            # From eq1: μ_new = (m_new * g) / F_new
            # Substitute into eq2:
            # F_new*cos37 = m_new*g*sin37 + ((m_new*g)/F_new)*(m_new*g*cos37 + F_new*sin37)
            # Multiply by F_new:
            # F_new^2*cos37 = m_new*g*sin37*F_new + m_new*g*(m_new*g*cos37 + F_new*sin37)
            # F_new^2*cos37 = m_new*g*sin37*F_new + (m_new*g)^2*cos37 + m_new*g*F_new*sin37
            # F_new^2*cos37 = 2*m_new*g*sin37*F_new + (m_new*g)^2*cos37
            # Rearranged: cos37*F_new^2 - 2*m_new*g*sin37*F_new - (m_new*g)^2*cos37 = 0
            
            a_coeff = cos37
            b_coeff = -2 * self.m * g * sin37
            c_coeff = -(self.m * g)**2 * cos37
            
            # Solve quadratic for F_new (positive root)
            discriminant = b_coeff**2 - 4*a_coeff*c_coeff
            if discriminant < 0:
                self.log(f"FAIL: No real solution for F_new with perturbed mass.")
                self.m = m_original
                return False
                
            F_new1 = (-b_coeff + math.sqrt(discriminant)) / (2*a_coeff)
            F_new2 = (-b_coeff - math.sqrt(discriminant)) / (2*a_coeff)
            F_new = max(F_new1, F_new2)  # Take positive solution
            
            # Compute μ_new from eq1
            mu_new = (self.m * g) / F_new
            
            # Check monotonicity: if m increases, F should increase
            if not (F_new > F_original):
                self.log(f"FAIL: F did not increase with mass. F_old={F_original}, F_new={F_new}")
                self.m = m_original
                return False

            # μ should remain approximately constant (check within 1%)
            if not math.isclose(mu_new, mu_original, rel_tol=0.01, abs_tol=1e-3):
                self.log(f"FAIL: mu changed significantly with mass. mu_old={mu_original}, mu_new={mu_new}")
                self.m = m_original
                return False

            self.log(f"PASS: Monotonicity test passed. F increased from {F_original:.3f} to {F_new:.3f}, mu nearly constant.")
            # Restore original mass
            self.m = m_original
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            self.m = m_original  # restore on error
            return False

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        self.logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test in tests:
            self.log(f"\n--- {name} ---")
            if not test():
                success = False
                self.log(f"{name} failed.")
            else:
                self.log(f"{name} passed.")

        return success, self.logs