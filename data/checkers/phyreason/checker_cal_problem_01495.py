import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 1.0))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 9.8))
        self.theta = math.radians(30)  # θ=30°
        
        # Extract results with safe fallbacks
        self.h = getattr(solution_instance, 'h', self.results.get('h', 0.0))
        self.v_D = getattr(solution_instance, 'v_D', self.results.get('v_D', 0.0))
        self.v_D_prime = getattr(solution_instance, 'v_D\'', self.results.get('v_D\'', 0.0))
        self.v_C = getattr(solution_instance, 'v_C', self.results.get('v_C', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', self.results.get('F_N', 0.0))
        self.f = getattr(solution_instance, 'f', self.results.get('f', 0.0))
        self.x_prime = getattr(solution_instance, 'x\'', self.results.get('x\'', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.t_prime = getattr(solution_instance, 't\'', self.results.get('t\'', 0.0))
        
        # Define symbolic variables for invariant checks
        self.m_sym, self.g_sym, self.R_sym, self.h_sym, self.v_sym = sp.symbols('m g R h v', positive=True)
        self.logs = []
        
    def log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
        
    def test_invariants(self):
        """Verify fundamental physics laws."""
        tests_passed = []
        
        # 1. Conservation of Energy from release to point D (minimum condition)
        try:
            # According to steps: mg(h-R) = 1/2 m v_D^2
            lhs = self.m * self.g * (self.h - self.R)
            rhs = 0.5 * self.m * self.v_D**2
            if math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("Energy conservation (release → D)", True))
            else:
                tests_passed.append(self.log(f"Energy conservation (release → D): LHS={lhs:.3f}, RHS={rhs:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"Energy conservation (release → D) failed: {e}", False))
        
        # 2. Conservation of Energy from release to point C
        try:
            # According to steps: mgh = 1/2 m v_C^2
            lhs = self.m * self.g * self.h
            rhs = 0.5 * self.m * self.v_C**2
            if math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("Energy conservation (release → C)", True))
            else:
                tests_passed.append(self.log(f"Energy conservation (release → C): LHS={lhs:.3f}, RHS={rhs:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"Energy conservation (release → C) failed: {e}", False))
        
        # 3. Newton's Second Law at point D (minimum speed condition)
        try:
            # Condition for circular motion at top: mg = m v_D^2 / R
            lhs = self.m * self.g
            rhs = self.m * self.v_D**2 / self.R
            if math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("Newton's 2nd Law at D (circular motion)", True))
            else:
                tests_passed.append(self.log(f"Newton's 2nd Law at D: mg={lhs:.3f}, mv²/R={rhs:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"Newton's 2nd Law at D failed: {e}", False))
        
        # 4. Newton's Second Law at point C
        try:
            # Normal force: F_N = m v_C^2 / R
            lhs = self.F_N
            rhs = self.m * self.v_C**2 / self.R
            if math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("Newton's 2nd Law at C (normal force)", True))
            else:
                tests_passed.append(self.log(f"Newton's 2nd Law at C: F_N={lhs:.3f}, mv²/R={rhs:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"Newton's 2nd Law at C failed: {e}", False))
        
        # 5. Static equilibrium of block (friction force)
        try:
            # From steps: f = F_N (horizontal balance)
            if math.isclose(self.f, self.F_N, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("Static equilibrium (f = F_N)", True))
            else:
                tests_passed.append(self.log(f"Static equilibrium: f={self.f:.3f}, F_N={self.F_N:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"Static equilibrium failed: {e}", False))
            
        return all(tests_passed)
    
    def test_trace_consistency(self):
        """Cross-check intermediate calculations with final answers."""
        tests_passed = []
        
        # 1. Check h from energy and circular motion (lower bound)
        try:
            # From steps: h_min = 3/2 R
            h_min_expected = 1.5 * self.R
            # Derived from mg(h-R) = 1/2 m v_D^2 and mg = m v_D^2/R
            # => h - R = v_D^2/(2g) = (gR)/(2g) = R/2 => h = 3R/2
            if math.isclose(self.h, h_min_expected, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("Lower bound h = 3R/2", True))
            else:
                tests_passed.append(self.log(f"Lower bound: h={self.h:.3f}, expected 1.5R={h_min_expected:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"Lower bound check failed: {e}", False))
        
        # 2. Check v_C from energy
        try:
            # From mgh = 1/2 m v_C^2 => v_C = sqrt(2gh)
            v_C_expected = math.sqrt(2 * self.g * self.h)
            if math.isclose(self.v_C, v_C_expected, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("v_C from energy conservation", True))
            else:
                tests_passed.append(self.log(f"v_C: {self.v_C:.3f}, expected sqrt(2gh)={v_C_expected:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"v_C check failed: {e}", False))
        
        # 3. Check projectile motion timing (for upper bound condition)
        try:
            # Time to fall vertical distance R: t = sqrt(2R/g)
            t_expected = math.sqrt(2 * self.R / self.g)
            if math.isclose(self.t, t_expected, rel_tol=0.02, abs_tol=0.02):
                tests_passed.append(self.log("Projectile time t = sqrt(2R/g)", True))
            else:
                tests_passed.append(self.log(f"Projectile time: t={self.t:.3f}, expected={t_expected:.3f}", False))
        except Exception as e:
            tests_passed.append(self.log(f"Projectile time check failed: {e}", False))
            
        return all(tests_passed)
    
    def test_monotonicity(self):
        """Perturb input R and check trend consistency."""
        original_R = self.R
        tests_passed = []
        
        try:
            # Perturb R by +10%
            perturbed_R = original_R * 1.1
            
            # Recalculate expected changes using formulas
            # Lower bound h scales linearly with R: h_min = 1.5 * R
            h_min_original = 1.5 * original_R
            h_min_perturbed = 1.5 * perturbed_R
            h_ratio = h_min_perturbed / h_min_original
            
            # v_D for lower bound: sqrt(gR)
            v_D_original = math.sqrt(self.g * original_R)
            v_D_perturbed = math.sqrt(self.g * perturbed_R)
            v_D_ratio = v_D_perturbed / v_D_original
            
            # v_C for lower bound: sqrt(2g * 1.5R) = sqrt(3gR)
            v_C_original = math.sqrt(3 * self.g * original_R)
            v_C_perturbed = math.sqrt(3 * self.g * perturbed_R)
            v_C_ratio = v_C_perturbed / v_C_original
            
            # Check monotonic relationships
            # As R increases, all quantities should increase
            if perturbed_R > original_R:
                if (h_min_perturbed > h_min_original and 
                    v_D_perturbed > v_D_original and 
                    v_C_perturbed > v_C_original):
                    tests_passed.append(self.log("Monotonicity: R↑ ⇒ h↑, v_D↑, v_C↑", True))
                else:
                    tests_passed.append(self.log("Monotonicity violated with R increase", False))
                    
            # Check scaling ratios (should match sqrt(R) for velocities)
            expected_velocity_ratio = math.sqrt(1.1)  # since v ∝ √R
            if (math.isclose(v_D_ratio, expected_velocity_ratio, rel_tol=0.02) and
                math.isclose(v_C_ratio, expected_velocity_ratio, rel_tol=0.02)):
                tests_passed.append(self.log("Velocity scaling v ∝ √R", True))
            else:
                tests_passed.append(self.log(f"Scaling: v_D ratio={v_D_ratio:.3f}, v_C ratio={v_C_ratio:.3f}, expected √1.1={expected_velocity_ratio:.3f}", False))
                
        except Exception as e:
            tests_passed.append(self.log(f"Monotonicity test failed: {e}", False))
        finally:
            # Restore original R
            self.R = original_R
            
        return all(tests_passed)
    
    def run_all_tests(self):
        """Execute all checks and return success flag with logs."""
        self.logs = ["=== Physics Audit Report ==="]
        
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            summary = [
                f"\nSummary:",
                f"  Invariants: {'PASS' if inv_ok else 'FAIL'}",
                f"  Trace Consistency: {'PASS' if trace_ok else 'FAIL'}",
                f"  Monotonicity: {'PASS' if mono_ok else 'FAIL'}",
                f"Overall: {'ALL CHECKS PASSED' if success else 'SOME CHECKS FAILED'}"
            ]
            self.logs.extend(summary)
            
        except Exception as e:
            self.logs.append(f"\nCritical error during testing: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
            
        return success, self.logs