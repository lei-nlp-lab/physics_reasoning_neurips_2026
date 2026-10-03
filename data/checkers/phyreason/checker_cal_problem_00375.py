import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access."""
        self.logs = []
        
        # Extract inputs (with defaults if not present)
        self.m = getattr(solution_instance, 'm', 19.0)      # mass of Team A stone (kg)
        self.M = getattr(solution_instance, 'M', 20.0)      # mass of Team B stone (kg)
        self.V01 = getattr(solution_instance, 'V01', 3.0)   # initial speed of Team A (m/s)
        self.V02 = getattr(solution_instance, 'V02', math.sqrt(34))  # initial speed of Team B (m/s)
        self.g = getattr(solution_instance, 'g', 10.0)      # gravitational acceleration (m/s²)
        self.r = getattr(solution_instance, 'r', 0.6)       # inner circle radius (m)
        self.R = getattr(solution_instance, 'R', 1.8)       # outer circle radius (m)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        
        # Map expected keys to attributes (with safe defaults)
        self.mu = results.get('mu', 0.0)                     # coefficient of friction
        self.S_A0 = 30.0  # Derived from geometry: given mu=0.015, V01=3 → S_A0 = V01²/(2*g*mu) = 30 m
        
        self.v1 = results.get('v_1', 0.0)                    # Team B speed at O before collision (m/s)
        self.V_prime = results.get("V'", 0.0)                # Team A velocity after collision (m/s)
        self.V2_dprime = results.get("V_2''", 0.0)           # Team B velocity after collision (m/s)
        self.S1 = results.get('S_1', 0.0)                    # Team A stopping distance after collision (m)
        self.S2 = results.get('S_2', 0.0)                    # Team B stopping distance after collision (m)
        self.W_f = results.get('W_f', 0.0)                   # Work by friction on Team A stone (J)
        
        # For symbolic checks
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'M': sp.symbols('M', positive=True),
            'V01': sp.symbols('V01', positive=True),
            'V02': sp.symbols('V02', positive=True),
            'mu': sp.symbols('mu', positive=True),
            'g': sp.symbols('g', positive=True),
            'S_A0': sp.symbols('S_A0', positive=True),
            'v1': sp.symbols('v1', positive=True),
            'V_prime': sp.symbols('V_prime', real=True),
            'V2_dprime': sp.symbols('V2_dprime', real=True),
            'S1': sp.symbols('S1', positive=True),
            'S2': sp.symbols('S2', positive=True)
        }
        
    def _log(self, msg, success):
        """Log test result."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify conservation laws and work-energy theorem."""
        checks = []
        
        try:
            # 1. Check work-energy for Team A stone (pre-collision)
            # Should have: 0.5*m*V01² = mu*m*g*S_A0 (since comes to rest at O)
            ke_initial = 0.5 * self.m * self.V01**2
            work_friction = self.mu * self.m * self.g * self.S_A0
            check1 = self._is_close(ke_initial, work_friction)
            checks.append(self._log(f"Work-energy for Team A stone: KE_initial={ke_initial:.3f} J, "
                                   f"W_friction={work_friction:.3f} J", check1))
            
            # 2. Check work-energy for Team B stone (pre-collision)
            # Should have: 0.5*M*V02² - mu*M*g*S_A0 = 0.5*M*v1²
            ke_initial_B = 0.5 * self.M * self.V02**2
            work_friction_B = self.mu * self.M * self.g * self.S_A0
            ke_at_O = 0.5 * self.M * self.v1**2
            check2 = self._is_close(ke_initial_B - work_friction_B, ke_at_O)
            checks.append(self._log(f"Work-energy for Team B stone: ΔKE={ke_initial_B-work_friction_B:.3f} J, "
                                   f"KE_at_O={ke_at_O:.3f} J", check2))
            
            # 3. Check momentum conservation in collision (no external forces during collision)
            # m*V' + M*V2'' = M*v1 (since Team A stone initially at rest)
            momentum_before = self.M * self.v1
            momentum_after = self.m * self.V_prime + self.M * self.V2_dprime
            check3 = self._is_close(momentum_before, momentum_after)
            checks.append(self._log(f"Momentum conservation: before={momentum_before:.3f} kg·m/s, "
                                   f"after={momentum_after:.3f} kg·m/s", check3))
            
            # 4. Check kinetic energy conservation in elastic collision
            ke_before = 0.5 * self.M * self.v1**2
            ke_after = 0.5 * self.m * self.V_prime**2 + 0.5 * self.M * self.V2_dprime**2
            check4 = self._is_close(ke_before, ke_after)
            checks.append(self._log(f"KE conservation in collision: before={ke_before:.3f} J, "
                                   f"after={ke_after:.3f} J", check4))
            
            # 5. Check post-collision work-energy for Team A stone
            # Should have: 0.5*m*V'² = mu*m*g*S1
            ke_post_A = 0.5 * self.m * self.V_prime**2
            work_fric_A = self.mu * self.m * self.g * self.S1
            check5 = self._is_close(ke_post_A, work_fric_A)
            checks.append(self._log(f"Post-collision work-energy Team A: KE={ke_post_A:.3f} J, "
                                   f"W_friction={work_fric_A:.3f} J", check5))
            
            # 6. Check post-collision work-energy for Team B stone
            ke_post_B = 0.5 * self.M * self.V2_dprime**2
            work_fric_B = self.mu * self.M * self.g * self.S2
            check6 = self._is_close(ke_post_B, work_fric_B)
            checks.append(self._log(f"Post-collision work-energy Team B: KE={ke_post_B:.3f} J, "
                                   f"W_friction={work_fric_B:.3f} J", check6))
            
            return all(checks)
            
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        checks = []
        
        try:
            # 1. Check mu calculation from step 2
            mu_calc = self.V01**2 / (2 * self.g * self.S_A0)
            check1 = self._is_close(self.mu, mu_calc)
            checks.append(self._log(f"Friction coefficient: calculated={mu_calc:.4f}, "
                                   f"reported={self.mu:.4f}", check1))
            
            # 2. Check v1 calculation from step 3
            v1_calc = math.sqrt(self.V02**2 - 2 * self.g * self.mu * self.S_A0)
            check2 = self._is_close(self.v1, v1_calc)
            checks.append(self._log(f"Team B speed at O: calculated={v1_calc:.3f} m/s, "
                                   f"reported={self.v1:.3f} m/s", check2))
            
            # 3. Check elastic collision formulas (solve for V' and V2'')
            # Equations: 
            #   m*V' + M*V2'' = M*v1
            #   m*V'^2 + M*V2''^2 = M*v1^2
            # Solutions for elastic collision with m2 initially at rest:
            #   V' = (m - M)/(m + M) * 0 + (2*M)/(m + M) * v1 = (2*M)/(m + M) * v1
            #   V2'' = (2*m)/(m + M) * 0 + (M - m)/(m + M) * v1 = (M - m)/(m + M) * v1
            V_prime_calc = (2 * self.M * self.v1) / (self.m + self.M)
            V2_dprime_calc = ((self.M - self.m) * self.v1) / (self.m + self.M)
            
            check3 = self._is_close(self.V_prime, V_prime_calc)
            check4 = self._is_close(self.V2_dprime, V2_dprime_calc)
            checks.append(self._log(f"Team A post-collision velocity: calculated={V_prime_calc:.3f} m/s, "
                                   f"reported={self.V_prime:.3f} m/s", check3))
            checks.append(self._log(f"Team B post-collision velocity: calculated={V2_dprime_calc:.3f} m/s, "
                                   f"reported={self.V2_dprime:.3f} m/s", check4))
            
            # 4. Check stopping distances
            S1_calc = self.V_prime**2 / (2 * self.mu * self.g)
            S2_calc = self.V2_dprime**2 / (2 * self.mu * self.g)
            check5 = self._is_close(self.S1, S1_calc)
            check6 = self._is_close(self.S2, S2_calc)
            checks.append(self._log(f"Team A stopping distance: calculated={S1_calc:.3f} m, "
                                   f"reported={self.S1:.3f} m", check5))
            checks.append(self._log(f"Team B stopping distance: calculated={S2_calc:.3f} m, "
                                   f"reported={self.S2:.3f} m", check6))
            
            # 5. Check final outcome
            # Team B wins if: S1 > r (Team A leaves inner circle) OR (S2 < R and S2 > r) ?
            # From problem: "knock your opponent's curling stone out of the larger circle"
            # Team A stone should be knocked out of outer circle (S1 > R) or at least inner circle?
            # Actually, step 10 says "Team B wins" with given distances.
            # With reported values: S1=86.7 m > R=1.8 m, S2=0.056 m < r=0.6 m
            # So Team A stone goes far beyond outer circle, Team B stone stays inside inner circle
            team_b_wins = (self.S1 > self.R) and (self.S2 < self.r)
            check7 = team_b_wins
            checks.append(self._log(f"Game outcome: Team B wins = {team_b_wins} "
                                   f"(S1={self.S1:.3f} m > R={self.R} m, "
                                   f"S2={self.S2:.3f} m < r={self.r} m)", check7))
            
            return all(checks)
            
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check trend consistency."""
        original_m = self.m
        perturbed_m = original_m * 1.1  # Increase by 10%
        
        try:
            # Keep original results
            original_mu = self.mu
            original_v1 = self.v1
            
            # Recalculate with perturbed mass
            # 1. Friction coefficient should remain the same (independent of mass)
            mu_perturbed = self.V01**2 / (2 * self.g * self.S_A0)
            
            # 2. Team B speed at O before collision should remain the same (independent of Team A mass)
            v1_perturbed = math.sqrt(self.V02**2 - 2 * self.g * mu_perturbed * self.S_A0)
            
            # 3. Post-collision velocities with perturbed m
            V_prime_perturbed = (2 * self.M * v1_perturbed) / (perturbed_m + self.M)
            V2_dprime_perturbed = ((self.M - perturbed_m) * v1_perturbed) / (perturbed_m + self.M)
            
            # 4. Stopping distances
            S1_perturbed = V_prime_perturbed**2 / (2 * mu_perturbed * self.g)
            S2_perturbed = V2_dprime_perturbed**2 / (2 * mu_perturbed * self.g)
            
            # Check monotonic trends:
            # If Team A mass increases:
            # - V' should decrease (since denominator increases)
            # - V2'' magnitude should decrease (M-m becomes smaller positive)
            # - S1 should decrease (since V' decreases)
            # - S2 should decrease (since V2'' decreases)
            
            trends_ok = True
            msg_parts = []
            
            if not (V_prime_perturbed < self.V_prime):
                trends_ok = False
                msg_parts.append(f"V' should decrease with increased mass (was {self.V_prime:.3f}, now {V_prime_perturbed:.3f})")
            
            if not (abs(V2_dprime_perturbed) < abs(self.V2_dprime)):
                trends_ok = False
                msg_parts.append(f"|V2''| should decrease with increased mass (was {abs(self.V2_dprime):.3f}, now {abs(V2_dprime_perturbed):.3f})")
            
            if not (S1_perturbed < self.S1):
                trends_ok = False
                msg_parts.append(f"S1 should decrease with increased mass (was {self.S1:.3f}, now {S1_perturbed:.3f})")
            
            if not (S2_perturbed < self.S2):
                trends_ok = False
                msg_parts.append(f"S2 should decrease with increased mass (was {self.S2:.3f}, now {S2_perturbed:.3f})")
            
            success = trends_ok
            msg = "Monotonicity test: " + ("All trends correct" if success else "; ".join(msg_parts))
            
            # Restore original mass
            self.m = original_m
            
            return self._log(msg, success)
            
        except Exception as e:
            # Restore original mass on error
            self.m = original_m
            self.logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        tests = [
            ("Invariants test", self.test_invariants),
            ("Trace consistency test", self.test_trace_consistency),
            ("Monotonicity test", self.test_monotonicity)
        ]
        
        results = []
        for test_name, test_func in tests:
            self.logs.append(f"=== Running {test_name} ===")
            try:
                results.append(test_func())
            except Exception as e:
                self.logs.append(f"CRASH in {test_name}: {traceback.format_exc()}")
                results.append(False)
        
        success = all(results)
        self.logs.append(f"=== Overall result: {'PASS' if success else 'FAIL'} ===")
        
        return success, self.logs