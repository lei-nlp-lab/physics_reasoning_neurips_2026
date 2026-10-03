import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        All attribute accesses are protected with getattr and dict.get.
        """
        # Extract problem inputs
        self.m = getattr(solution_instance, 'm', 2.0)  # kg
        self.F_pull = getattr(solution_instance, 'F_pull', 16.0)  # N
        self.mu_k = getattr(solution_instance, 'mu_k', 0.2)
        self.t1 = getattr(solution_instance, 't1', 2.0)  # s (time force is applied)
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s^2
        
        # Extract results using safe access pattern
        results = getattr(solution_instance, 'results', {})
        self.a1 = results.get('a_1', 0.0)  # acceleration during pulling
        self.v1 = results.get('v_1', 0.0)  # velocity at t=2s
        self.Ek = results.get('E_K', 0.0)  # kinetic energy at t=2s
        self.a2 = results.get('a_2', 0.0)  # acceleration after force removed (deceleration)
        self.x = results.get('x', 0.0)  # distance slid after force removed
        
        # Store symbolic variables for invariant checks
        self._init_symbols()
        
    def _init_symbols(self):
        """Initialize SymPy symbols for physics laws."""
        self.m_sym, self.F_pull_sym, self.mu_k_sym, self.t1_sym, self.g_sym = sp.symbols('m F_pull mu_k t1 g')
        self.a1_sym, self.v1_sym, self.Ek_sym, self.a2_sym, self.x_sym = sp.symbols('a1 v1 Ek a2 x')
        
    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Newton's 2nd Law during pulling phase
        2. Newton's 2nd Law after force removal
        3. Kinetic energy formula
        4. Energy conservation during deceleration
        """
        logs = []
        success = True
        
        try:
            # 1. Newton's 2nd Law during pulling (0-2s)
            # Net force = F_pull - friction = m * a1
            F_friction = self.mu_k * self.m * self.g
            net_force = self.F_pull - F_friction
            expected_a1 = net_force / self.m if self.m != 0 else 0
            
            if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd Law (pulling phase). a1={self.a1}, expected={expected_a1}")
                success = False
            else:
                logs.append(f"PASS: Newton's 2nd Law (pulling phase) satisfied: {self.a1:.2f} m/s²")
                
            # 2. Newton's 2nd Law after force removal
            # Net force = -friction = m * a2 (negative)
            expected_a2 = -self.mu_k * self.g
            
            if not math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd Law (coasting phase). a2={self.a2}, expected={expected_a2}")
                success = False
            else:
                logs.append(f"PASS: Newton's 2nd Law (coasting phase) satisfied: {self.a2:.2f} m/s²")
                
            # 3. Kinetic energy formula check
            expected_Ek = 0.5 * self.m * (self.v1 ** 2)
            
            if not math.isclose(self.Ek, expected_Ek, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Kinetic energy formula. Ek={self.Ek}, expected={expected_Ek}")
                success = False
            else:
                logs.append(f"PASS: Kinetic energy formula satisfied: {self.Ek:.2f} J")
                
            # 4. Energy conservation during deceleration (work-energy theorem)
            # Work done by friction = ΔKE = 0 - Ek = -Ek
            # Also work = -friction * x = -mu_k * m * g * x
            work_friction = -self.mu_k * self.m * self.g * self.x
            
            if not math.isclose(work_friction, -self.Ek, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation during deceleration. Friction work={work_friction:.2f} J, ΔKE={-self.Ek:.2f} J")
                success = False
            else:
                logs.append(f"PASS: Energy conservation during deceleration satisfied")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer.
        Based on STEPS ANALYSIS from KB.
        """
        logs = []
        success = True
        
        try:
            # Step 3: a1 = (F_pull - μmg)/m
            expected_a1_step = (self.F_pull - self.mu_k * self.m * self.g) / self.m
            if not math.isclose(self.a1, expected_a1_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 3 consistency. a1={self.a1}, step formula gives={expected_a1_step}")
                success = False
            else:
                logs.append(f"PASS: Step 3 (a1 calculation) consistent")
                
            # Step 4: v1 = a1 * t1 (starting from rest)
            expected_v1_step = self.a1 * self.t1
            if not math.isclose(self.v1, expected_v1_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 4 consistency. v1={self.v1}, step formula gives={expected_v1_step}")
                success = False
            else:
                logs.append(f"PASS: Step 4 (v1 calculation) consistent")
                
            # Step 5: Ek = 0.5 * m * v1²
            expected_Ek_step = 0.5 * self.m * (self.v1 ** 2)
            if not math.isclose(self.Ek, expected_Ek_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 5 consistency. Ek={self.Ek}, step formula gives={expected_Ek_step}")
                success = False
            else:
                logs.append(f"PASS: Step 5 (Ek calculation) consistent")
                
            # Step 6: a2 = -μ*g (negative)
            expected_a2_step = -self.mu_k * self.g
            if not math.isclose(self.a2, expected_a2_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 6 consistency. a2={self.a2}, step formula gives={expected_a2_step}")
                success = False
            else:
                logs.append(f"PASS: Step 6 (a2 calculation) consistent")
                
            # Step 8: x = v1²/(2*|a2|) using kinematic equation v² = u² + 2as with v=0
            expected_x_step = (self.v1 ** 2) / (2 * abs(self.a2))
            if not math.isclose(self.x, expected_x_step, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 8 consistency. x={self.x}, step formula gives={expected_x_step}")
                success = False
            else:
                logs.append(f"PASS: Step 8 (x calculation) consistent")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb input mass and verify expected trends.
        We'll increase mass by 10%, recalculate using physics formulas,
        check trends, then restore original values.
        """
        logs = []
        success = True
        
        try:
            # Store original values
            orig_m = self.m
            orig_a1 = self.a1
            orig_v1 = self.v1
            orig_Ek = self.Ek
            orig_a2 = self.a2
            orig_x = self.x
            
            # Perturb mass: increase by 10%
            self.m = orig_m * 1.10
            
            # Recalculate using physics formulas (not the solver's code)
            # 1. New a1 during pulling
            new_a1 = (self.F_pull - self.mu_k * self.m * self.g) / self.m
            # 2. New v1 at t=2s
            new_v1 = new_a1 * self.t1
            # 3. New Ek
            new_Ek = 0.5 * self.m * (new_v1 ** 2)
            # 4. New a2 (should be same as before, independent of mass)
            new_a2 = -self.mu_k * self.g
            # 5. New x
            new_x = (new_v1 ** 2) / (2 * abs(new_a2))
            
            # Check expected trends:
            # a1 should DECREASE with increased mass (since a1 = F_pull/m - μ*g)
            if new_a1 >= orig_a1:
                logs.append(f"FAIL: Monotonicity: a1 should decrease with increased mass. orig={orig_a1:.2f}, new={new_a1:.2f}")
                success = False
            else:
                logs.append(f"PASS: a1 decreases with increased mass as expected")
                
            # v1 should DECREASE (since a1 decreased and time fixed)
            if new_v1 >= orig_v1:
                logs.append(f"FAIL: Monotonicity: v1 should decrease with increased mass. orig={orig_v1:.2f}, new={new_v1:.2f}")
                success = False
            else:
                logs.append(f"PASS: v1 decreases with increased mass as expected")
                
            # Ek might increase or decrease (depends on m*v²). Let's check formula: Ek = 0.5*m*v²
            # Since v decreases but m increases, we need to check derivative.
            # For this specific perturbation (10% mass increase), we expect:
            # v1_new ≈ (F_pull/(1.1m) - μ*g)*t = (F_pull*t/(1.1m) - μ*g*t)
            # Ek_new = 0.5*(1.1m)*v_new²
            # We'll compute numerically and check if monotonicity holds.
            # Actually, let's use symbolic check:
            m_sym = sp.symbols('m', positive=True)
            a1_expr = (self.F_pull - self.mu_k * m_sym * self.g) / m_sym
            v1_expr = a1_expr * self.t1
            Ek_expr = 0.5 * m_sym * (v1_expr ** 2)
            # Compute derivative d(Ek)/dm
            dEk_dm = sp.diff(Ek_expr, m_sym)
            # Evaluate at original m
            dEk_dm_val = float(dEk_dm.subs(m_sym, orig_m))
            # If derivative is negative, Ek should decrease with mass increase
            # If positive, Ek should increase
            expected_trend = "increase" if dEk_dm_val > 0 else "decrease"
            actual_trend = "increase" if new_Ek > orig_Ek else "decrease"
            
            if expected_trend != actual_trend:
                logs.append(f"FAIL: Monotonicity: Ek should {expected_trend} with mass. orig={orig_Ek:.2f}, new={new_Ek:.2f}, dEk/dm={dEk_dm_val:.2f}")
                success = False
            else:
                logs.append(f"PASS: Ek {actual_trend}s with increased mass as expected (dEk/dm={dEk_dm_val:.2f})")
                
            # a2 should remain CONSTANT (independent of mass)
            if not math.isclose(new_a2, orig_a2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Monotonicity: a2 should be constant. orig={orig_a2:.2f}, new={new_a2:.2f}")
                success = False
            else:
                logs.append(f"PASS: a2 constant with mass change as expected")
                
            # x: v1 decreased, so x should decrease (since x = v1²/(2*μ*g))
            if new_x >= orig_x:
                logs.append(f"FAIL: Monotonicity: x should decrease with increased mass. orig={orig_x:.2f}, new={new_x:.2f}")
                success = False
            else:
                logs.append(f"PASS: x decreases with increased mass as expected")
                
            # Restore original values
            self.m = orig_m
            self.a1 = orig_a1
            self.v1 = orig_v1
            self.Ek = orig_Ek
            self.a2 = orig_a2
            self.x = orig_x
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            success = False
            # Try to restore original values on error
            try:
                self.m = orig_m
            except:
                pass
                
        return success, logs
    
    def run_all_tests(self):
        """Run all physics checks and return aggregated results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_method()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
                
        return overall_success, all_logs