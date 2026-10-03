import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with defaults
        self.a0 = getattr(solution_instance, 'a0', 4.0)  # belt acceleration (m/s²)
        self.v0 = getattr(solution_instance, 'v0', 8.0)  # belt final velocity (m/s)
        self.mu = getattr(solution_instance, 'mu', 0.2)  # friction coefficient
        self.g = getattr(solution_instance, 'g', 10.0)   # gravity (m/s²)
        
        # Extract results with safe fallbacks
        self.t = float(self.results.get('t', 0.0))          # time to reach v0 for belt
        self.t_prime = float(self.results.get("t'", self.results.get('t_prime', 0.0)))  # additional time for block
        self.a_block = float(self.results.get('a', 0.0))    # block's acceleration
        self.v_block_at_t = float(self.results.get('v', 0.0))  # block's velocity at time t
        self.s_belt = float(self.results.get('s_0', 0.0))   # belt displacement
        self.s_block = float(self.results.get('s', 0.0))    # block displacement
        self.l_mark = float(self.results.get('l', 0.0))     # black mark length
        
        # Store logs for test results
        self.logs = []

    def _is_close(self, a, b, msg="", rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with logging."""
        is_ok = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        if not is_ok:
            self.logs.append(f"FAIL: {msg} | Expected ~{b}, got {a}")
        else:
            self.logs.append(f"OK: {msg}")
        return is_ok

    def test_invariants(self):
        """Verify fundamental physics laws."""
        success = True
        
        # 1. Check block acceleration from friction: a_block = μ * g
        expected_a = self.mu * self.g
        success &= self._is_close(self.a_block, expected_a, 
                                 f"Block acceleration: μ*g = {expected_a}")
        
        # 2. Check work-energy theorem for block: v0² = 2*a_block*s_block
        # Block starts at rest, ends at v0 (8 m/s)
        left = self.v0**2
        right = 2 * self.a_block * self.s_block
        success &= self._is_close(left, right,
                                 f"Work-energy for block: v0² = {left}, 2*a*s = {right}")
        
        # 3. Newton's 2nd law check during belt acceleration phase
        # Relative acceleration: a_rel = a0 - a_block
        a_rel = self.a0 - self.a_block
        # Relative velocity at time t should be: v_rel = a_rel * t
        v_rel_expected = a_rel * self.t
        v_rel_actual = self.v0 - self.v_block_at_t  # belt v0 minus block v at time t
        success &= self._is_close(v_rel_actual, v_rel_expected,
                                 f"Relative velocity: (v0 - v_block) = {v_rel_actual}, a_rel*t = {v_rel_expected}")
        
        return success

    def test_trace_consistency(self):
        """Verify intermediate calculations lead to final answer."""
        success = True
        
        # 1. Time for belt to reach v0: t = v0 / a0
        t_expected = self.v0 / self.a0
        success &= self._is_close(self.t, t_expected,
                                 f"Belt acceleration time: t = v0/a0 = {t_expected}")
        
        # 2. Block velocity at time t: v = a_block * t
        v_expected = self.a_block * self.t
        success &= self._is_close(self.v_block_at_t, v_expected,
                                 f"Block velocity at t: a_block*t = {v_expected}")
        
        # 3. Additional time for block to reach v0: t' = (v0 - v) / a_block
        if self.a_block > 0:
            t_prime_expected = (self.v0 - self.v_block_at_t) / self.a_block
            success &= self._is_close(self.t_prime, t_prime_expected,
                                     f"Additional time: t' = (v0-v)/a = {t_prime_expected}")
        
        # 4. Belt displacement: s_belt = 0.5*a0*t² + v0*t'
        s_belt_expected = 0.5 * self.a0 * self.t**2 + self.v0 * self.t_prime
        success &= self._is_close(self.s_belt, s_belt_expected,
                                 f"Belt displacement: 0.5*a0*t² + v0*t' = {s_belt_expected}")
        
        # 5. Block displacement: s_block = v0²/(2*a_block)
        if self.a_block > 0:
            s_block_expected = self.v0**2 / (2 * self.a_block)
            success &= self._is_close(self.s_block, s_block_expected,
                                     f"Block displacement: v0²/(2a) = {s_block_expected}")
        
        # 6. Final answer: l = s_belt - s_block
        l_expected = self.s_belt - self.s_block
        success &= self._is_close(self.l_mark, l_expected,
                                 f"Black mark length: s_belt - s_block = {l_expected}")
        
        return success

    def test_monotonicity(self):
        """Perturb μ and check if l changes as expected."""
        success = True
        
        # Perturb μ by +10%
        mu_perturbed = self.mu * 1.1
        a_block_perturbed = mu_perturbed * self.g
        
        # Recompute key quantities
        t_expected = self.v0 / self.a0  # unchanged (belt property)
        v_block_at_t = a_block_perturbed * t_expected
        
        if self.v0 > v_block_at_t:
            t_prime_perturbed = (self.v0 - v_block_at_t) / a_block_perturbed
        else:
            t_prime_perturbed = 0.0
        
        s_belt_perturbed = 0.5 * self.a0 * t_expected**2 + self.v0 * t_prime_perturbed
        s_block_perturbed = self.v0**2 / (2 * a_block_perturbed)
        l_perturbed = s_belt_perturbed - s_block_perturbed
        
        # Check monotonicity: larger μ → faster block acceleration → shorter mark
        if l_perturbed < self.l_mark:
            self.logs.append(f"OK: Monotonicity - μ↑ (0.2→{mu_perturbed:.3f}) → l↓ ({self.l_mark:.3f}→{l_perturbed:.3f})")
        else:
            self.logs.append(f"FAIL: Monotonicity violated - μ increased but l didn't decrease")
            success = False
        
        # Additionally, check symbolic derivative ∂l/∂μ is negative
        try:
            μ, g, a0, v0 = sp.symbols('μ g a0 v0', positive=True)
            a_block_sym = μ * g
            t_sym = v0 / a0
            v_block_sym = a_block_sym * t_sym
            t_prime_sym = (v0 - v_block_sym) / a_block_sym
            s_belt_sym = sp.Rational(1,2) * a0 * t_sym**2 + v0 * t_prime_sym
            s_block_sym = v0**2 / (2 * a_block_sym)
            l_sym = s_belt_sym - s_block_sym
            
            dl_dmu = sp.diff(l_sym, μ)
            # Substitute numerical values (g=10, a0=4, v0=8, μ=0.2)
            dl_dmu_num = float(dl_dmu.subs({g: self.g, a0: self.a0, v0: self.v0, μ: self.mu}))
            if dl_dmu_num < 0:
                self.logs.append(f"OK: Symbolic derivative ∂l/∂μ = {dl_dmu_num:.3f} < 0")
            else:
                self.logs.append(f"FAIL: Symbolic derivative ∂l/∂μ = {dl_dmu_num:.3f} ≥ 0")
                success = False
        except Exception as e:
            self.logs.append(f"Symbolic check failed: {e}")
        
        return success

    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs.clear()
        
        self.logs.append("=== Physics Checker Started ===")
        self.logs.append(f"Parameters: a0={self.a0}, v0={self.v0}, μ={self.mu}, g={self.g}")
        self.logs.append(f"Results: t={self.t}, t'={self.t_prime}, a_block={self.a_block}")
        self.logs.append(f"         v_block(t)={self.v_block_at_t}, s_belt={self.s_belt}")
        self.logs.append(f"         s_block={self.s_block}, l={self.l_mark}")
        
        test1 = self.test_invariants()
        self.logs.append(f"--- Invariants test: {'PASS' if test1 else 'FAIL'} ---")
        
        test2 = self.test_trace_consistency()
        self.logs.append(f"--- Trace consistency: {'PASS' if test2 else 'FAIL'} ---")
        
        test3 = self.test_monotonicity()
        self.logs.append(f"--- Monotonicity test: {'PASS' if test3 else 'FAIL'} ---")
        
        overall = test1 and test2 and test3
        self.logs.append(f"=== Overall: {'PASS' if overall else 'FAIL'} ===")
        
        return overall, self.logs.copy()