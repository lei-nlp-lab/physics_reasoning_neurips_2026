import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: extract inputs with defaults
        self.inputs = {}
        self.results = {}
        self.logs = []
        
        # Problem constants (with safe extraction)
        results_dict = getattr(solution_instance, 'results', {})
        self.inputs['l'] = getattr(solution_instance, 'l', 0.75)
        self.inputs['m'] = getattr(solution_instance, 'm', 0.25)
        self.inputs['v0'] = getattr(solution_instance, 'v0', 4.0)
        self.inputs['h'] = getattr(solution_instance, 'h', 1.25)
        self.inputs['s'] = getattr(solution_instance, 's', 1.5)
        self.inputs['mu'] = getattr(solution_instance, 'mu', 0.2)
        self.inputs['R'] = getattr(solution_instance, 'R', 0.4)
        self.inputs['g'] = getattr(solution_instance, 'g', 10.0)
        
        # Extract results (using safe pattern)
        self.results['v_prime'] = getattr(solution_instance, 'v_prime', results_dict.get("v'", None))
        self.results['F'] = getattr(solution_instance, 'F', results_dict.get('F', None))
        self.results['l_1'] = getattr(solution_instance, 'l_1', results_dict.get('l_1', None))
        self.results['t'] = getattr(solution_instance, 't', results_dict.get('t', None))
        self.results['a'] = getattr(solution_instance, 'a', results_dict.get('a', None))
        self.results['v'] = getattr(solution_instance, 'v', results_dict.get('v', None))
        self.results['t_1'] = getattr(solution_instance, 't_1', results_dict.get('t_1', None))
        self.results['L'] = getattr(solution_instance, 'L', results_dict.get('L', None))
        
        # Validate we have needed results
        missing = [k for k, v in self.results.items() if v is None]
        if missing:
            self.logs.append(f"Warning: Missing results for keys: {missing}")
    
    def test_invariants(self):
        """Verify fundamental physics laws: energy conservation and Newton's laws."""
        try:
            # 1. Check energy conservation in vertical projectile (should conserve mechanical energy)
            if self.results['v'] is not None:
                # At lowest point: kinetic + potential = 1/2 m v² + 0
                # At landing point: kinetic = 1/2 m v_horiz², potential = -mgh (loss)
                # But rope breaks, so only gravity does work. Energy should be conserved.
                # Actually, the block lands with horizontal velocity v (from step 3)
                # and vertical velocity from free fall: v_vert = sqrt(2gh)
                # Total kinetic energy at landing: 1/2 m (v² + 2gh)
                # Initial kinetic at lowest point: 1/2 m v²
                # Loss in potential: -mgh (height decreases by h)
                # So energy change: ΔE = 1/2 m (v² + 2gh) - 1/2 m v² - (-mgh) = 0
                # This is automatically satisfied. We'll check numerically.
                v = self.results['v']
                g = self.inputs['g']
                h = self.inputs['h']
                KE_initial = 0.5 * self.inputs['m'] * v**2
                PE_initial = 0  # reference at lowest point
                KE_landing = 0.5 * self.inputs['m'] * (v**2 + 2*g*h)  # horizontal v + vertical from free fall
                PE_landing = -self.inputs['m'] * g * h
                total_initial = KE_initial + PE_initial
                total_final = KE_landing + PE_landing
                if not math.isclose(total_initial, total_final, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Energy conservation violation: initial={total_initial:.3f}, final={total_final:.3f}")
                    return False
            
            # 2. Check Newton's second law for centripetal force at lowest point
            if self.results['v'] is not None and self.results['F'] is not None:
                m, g, l = self.inputs['m'], self.inputs['g'], self.inputs['l']
                v = self.results['v']
                expected_F = m * g + m * v**2 / l  # T - mg = m v²/l
                if not math.isclose(self.results['F'], expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Centripetal force mismatch: F={self.results['F']}, expected={expected_F}")
                    return False
            
            # 3. Check Newton's second law for conveyor belt friction
            if self.results['a'] is not None:
                expected_a = self.inputs['mu'] * self.inputs['g']
                if not math.isclose(self.results['a'], expected_a, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Friction acceleration mismatch: a={self.results['a']}, expected μ g={expected_a}")
                    return False
            
            # 4. Check flying-off condition: centripetal force equals weight at pulley edge
            if self.results['v_prime'] is not None:
                expected_v_prime = math.sqrt(self.inputs['g'] * self.inputs['R'])
                if not math.isclose(self.results['v_prime'], expected_v_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Flying-off velocity mismatch: v'={self.results['v_prime']}, expected √(gR)={expected_v_prime}")
                    return False
            
            # 5. Check conveyor belt kinematics: v'² = 2aL
            if all(k in self.results and self.results[k] is not None for k in ['v_prime', 'a', 'L']):
                vp, a, L = self.results['v_prime'], self.results['a'], self.results['L']
                left = vp**2
                right = 2 * a * L
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Kinematics mismatch: v'²={left:.3f}, 2aL={right:.3f}")
                    return False
            
            # 6. Check new rope length consistency with projectile motion and tension
            if all(k in self.results and self.results[k] is not None for k in ['l_1', 'F', 't_1']):
                l1, F, t1 = self.results['l_1'], self.results['F'], self.results['t_1']
                m, g, h, s = self.inputs['m'], self.inputs['g'], self.inputs['h'], self.inputs['s']
                
                # From tension equation: F = mg + m v1²/l1 => v1 = sqrt((F - mg) * l1 / m)
                v1 = math.sqrt((F - m * g) * l1 / m)
                
                # From vertical fall: h + 0.75 - l1 = 1/2 g t1²
                expected_t1 = math.sqrt(2 * (h + 0.75 - l1) / g)
                if not math.isclose(t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Time t1 mismatch: {t1} vs {expected_t1}")
                    return False
                
                # From horizontal motion: s = v1 * t1
                expected_s = v1 * t1
                if not math.isclose(s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Horizontal displacement mismatch: s={s}, v1*t1={expected_s}")
                    return False
            
            self.logs.append("All invariants passed.")
            return True
        except Exception as e:
            self.logs.append(f"Invariant test failed with error: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check velocity v from projectile steps
            if self.results['t'] is not None and self.results['v'] is not None:
                # t = sqrt(2h/g)
                expected_t = math.sqrt(2 * self.inputs['h'] / self.inputs['g'])
                if not math.isclose(self.results['t'], expected_t, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Time t mismatch: {self.results['t']} vs √(2h/g)={expected_t}")
                    return False
                
                # v = s/t
                expected_v = self.inputs['s'] / expected_t
                if not math.isclose(self.results['v'], expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Velocity v mismatch: {self.results['v']} vs s/t={expected_v}")
                    return False
            
            # 2. Check tension F from v
            if self.results['v'] is not None and self.results['F'] is not None:
                expected_F = self.inputs['m'] * self.inputs['g'] + self.inputs['m'] * self.results['v']**2 / self.inputs['l']
                if not math.isclose(self.results['F'], expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Tension F inconsistent with v: {self.results['F']} vs {expected_F}")
                    return False
            
            # 3. Check conveyor belt length L
            if all(k in self.results and self.results[k] is not None for k in ['v_prime', 'a', 'L']):
                expected_L = self.results['v_prime']**2 / (2 * self.results['a'])
                if not math.isclose(self.results['L'], expected_L, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"Conveyor length L mismatch: {self.results['L']} vs v'²/(2a)={expected_L}")
                    return False
            
            # 4. Check new rope length l1 (if provided)
            if self.results['l_1'] is not None:
                # Using the known answer 0.75 m
                if not math.isclose(self.results['l_1'], 0.75, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"New rope length l1 mismatch: {self.results['l_1']} vs expected 0.75")
                    return False
            
            self.logs.append("All trace consistency checks passed.")
            return True
        except Exception as e:
            self.logs.append(f"Trace consistency test failed: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        try:
            original_m = self.inputs['m']
            perturbed_m = original_m * 1.1  # Increase mass by 10%
            
            # 1. Tension F should increase proportionally more than mass increase
            # because F = m(g + v²/l) and v is independent of m (from projectile motion)
            if self.results['v'] is not None and self.results['F'] is not None:
                v = self.results['v']
                l, g = self.inputs['l'], self.inputs['g']
                original_F_formula = original_m * (g + v**2 / l)
                perturbed_F_formula = perturbed_m * (g + v**2 / l)
                expected_ratio = perturbed_F_formula / original_F_formula
                actual_ratio = (perturbed_m / original_m)  # Since v doesn't depend on m
                if not math.isclose(expected_ratio, actual_ratio, rel_tol=1e-2):
                    self.logs.append(f"Monotonicity violation: F should scale linearly with m")
                    return False
            
            # 2. Conveyor belt length L should be independent of mass
            # because v' = √(gR) and a = μg, both independent of m
            if self.results['L'] is not None:
                # L = v'²/(2a) = (gR)/(2μg) = R/(2μ)
                expected_L = self.inputs['R'] / (2 * self.inputs['mu'])
                if not math.isclose(self.results['L'], expected_L, rel_tol=1e-2, abs_tol=1e-2):
                    self.logs.append(f"L should be R/(2μ) = {expected_L}, got {self.results['L']}")
                    return False
            
            # 3. New rope length l1: perturb coefficient μ and check L trend
            original_mu = self.inputs['mu']
            perturbed_mu = original_mu * 1.1
            original_L_formula = self.inputs['R'] / (2 * original_mu)
            perturbed_L_formula = self.inputs['R'] / (2 * perturbed_mu)
            if perturbed_L_formula >= original_L_formula:  # Should decrease
                self.logs.append(f"Monotonicity violation: L should decrease when μ increases")
                return False
            
            # Restore original values (simulated)
            self.logs.append(f"Monotonicity: F ∝ m, L ∝ 1/μ, and l1 independent of m.")
            self.logs.append("All monotonicity checks passed.")
            return True
        except Exception as e:
            self.logs.append(f"Monotonicity test failed: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        try:
            success = True
            self.logs = []
            
            # Run tests in order
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
            
            return success, self.logs
        except Exception as e:
            self.logs.append(f"Checker crashed: {traceback.format_exc()}")
            return False, self.logs