import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Extract inputs and results safely."""
        self.solution = solution_instance
        
        # Extract inputs with fallbacks
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', None))
        self.l = getattr(solution_instance, 'l', solution_instance.results.get('l', None))
        self.k = getattr(solution_instance, 'k', solution_instance.results.get('k', None))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', None))
        
        # Extract results with fallbacks
        self.t0 = getattr(solution_instance, 't_0', solution_instance.results.get('t_0', None))
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', None))
        self.F_A = getattr(solution_instance, 'F_A', solution_instance.results.get('F_A', None))
        
        self.logs = []
        
    def _compare(self, a, b, name="", rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with fallback to symbolic."""
        try:
            # Try numerical comparison
            a_num = float(a) if not isinstance(a, (int, float)) else a
            b_num = float(b) if not isinstance(b, (int, float)) else b
            if math.isclose(a_num, b_num, rel_tol=rel_tol, abs_tol=abs_tol):
                self.logs.append(f"✓ {name}: {a} ≈ {b}")
                return True
            else:
                self.logs.append(f"✗ {name}: {a} ≠ {b} (numerical diff: {abs(a_num - b_num)})")
                return False
        except (TypeError, ValueError):
            # Fallback to symbolic simplification
            diff = sp.simplify(a - b)
            if diff == 0:
                self.logs.append(f"✓ {name}: {a} ≡ {b} (symbolic match)")
                return True
            else:
                self.logs.append(f"✗ {name}: {a} ≠ {b} (symbolic diff: {diff})")
                return False
    
    def test_invariants(self):
        """Verify Faraday's Law, Ohm's Law, and Force Balance."""
        success = True
        
        # Faraday's Law: E = (k*l^2)/2
        E_expected = (self.k * self.l**2) / 2
        if not self._compare(self.E, E_expected, "Faraday's Law: E = k*l²/2"):
            success = False
        
        # Ohm's Law: I = E/R
        I_expected = self.E / self.R
        # From Faraday's Law and steps: I = (k*l^2)/(2R)
        I_from_steps = (self.k * self.l**2) / (2 * self.R)
        if not self._compare(I_expected, I_from_steps, "Ohm's Law consistency"):
            success = False
        
        # Ampere Force at general time t: F_A = (k² * l³ * t) / (2R)
        # At t = t0, should equal mg
        t = sp.Symbol('t')
        F_A_general = (self.k**2 * self.l**3 * t) / (2 * self.R)
        
        # Check force balance at t0: mg = F_A(t0)
        mg = self.m * self.g
        F_A_at_t0 = F_A_general.subs(t, self.t0)
        if not self._compare(mg, F_A_at_t0, "Force balance mg = F_A(t₀)"):
            success = False
        
        # Newton's 2nd Law consistency
        # When loop begins to move, net force = 0
        # So F_A(t0) - mg = 0
        net_force = F_A_at_t0 - mg
        if not self._compare(net_force, 0, "Newton's 2nd Law at t₀"):
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # From steps: t0 = (2*m*g*R)/(k²*l³)
        t0_expected = (2 * self.m * self.g * self.R) / (self.k**2 * self.l**3)
        if not self._compare(self.t0, t0_expected, "t₀ from derived formula"):
            success = False
        
        # Check that substituting t0 into F_A gives mg
        F_A_from_t0 = (self.k**2 * self.l**3 * self.t0) / (2 * self.R)
        if not self._compare(F_A_from_t0, self.m * self.g, "F_A(t₀) = mg"):
            success = False
        
        # E should be constant (independent of time)
        # Derivative of E should be 0
        try:
            E_expr = (self.k * self.l**2) / 2
            if hasattr(E_expr, 'diff'):
                dE_dt = sp.diff(E_expr, sp.Symbol('t'))
                if not self._compare(dE_dt, 0, "E is time-independent"):
                    success = False
        except:
            pass  # Skip if symbolic differentiation fails
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends."""
        success = True
        
        # Store original values
        orig_m = self.m
        orig_R = self.R
        orig_l = self.l
        orig_k = self.k
        orig_g = self.g
        
        # Test 1: Increase mass → t0 should increase
        self.m = orig_m * 1.1  # +10%
        t0_perturbed_m = (2 * self.m * self.g * self.R) / (self.k**2 * self.l**3)
        if not (float(t0_perturbed_m) > float(self.t0)):
            self.logs.append(f"✗ Monotonicity: Increasing m should increase t₀")
            success = False
        else:
            self.logs.append(f"✓ Monotonicity: t₀ increases with m")
        
        # Restore m
        self.m = orig_m
        
        # Test 2: Increase k → t0 should decrease
        self.k = orig_k * 1.1  # +10%
        t0_perturbed_k = (2 * self.m * self.g * self.R) / (self.k**2 * self.l**3)
        if not (float(t0_perturbed_k) < float(self.t0)):
            self.logs.append(f"✗ Monotonicity: Increasing k should decrease t₀")
            success = False
        else:
            self.logs.append(f"✓ Monotonicity: t₀ decreases with k")
        
        # Restore k
        self.k = orig_k
        
        # Test 3: Increase l → t0 should decrease (cubic in denominator)
        self.l = orig_l * 1.1  # +10%
        t0_perturbed_l = (2 * self.m * self.g * self.R) / (self.k**2 * self.l**3)
        if not (float(t0_perturbed_l) < float(self.t0)):
            self.logs.append(f"✗ Monotonicity: Increasing l should decrease t₀")
            success = False
        else:
            self.logs.append(f"✓ Monotonicity: t₀ decreases with l")
        
        # Restore l
        self.l = orig_l
        
        # Test 4: Increase R → t0 should increase
        self.R = orig_R * 1.1  # +10%
        t0_perturbed_R = (2 * self.m * self.g * self.R) / (self.k**2 * self.l**3)
        if not (float(t0_perturbed_R) > float(self.t0)):
            self.logs.append(f"✗ Monotonicity: Increasing R should increase t₀")
            success = False
        else:
            self.logs.append(f"✓ Monotonicity: t₀ increases with R")
        
        # Restore R
        self.R = orig_R
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} Test ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"✗ {test_name} failed with error: {e}")
                self.logs.append(traceback.format_exc())
                overall_success = False
        
        return overall_success, self.logs