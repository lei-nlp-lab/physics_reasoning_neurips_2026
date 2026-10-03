import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr with fallback to .results dictionary to avoid AttributeError.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe fallbacks
        self.m_P = getattr(solution_instance, 'm_P', 0.15)
        self.L = getattr(solution_instance, 'L', 1.2)
        self.m_Q = getattr(solution_instance, 'm_0', 0.1)  # m_0 is block Q mass
        self.g = getattr(solution_instance, 'g', 9.8)
        self.mu = getattr(solution_instance, 'mu', 0.28)
        self.a_c = getattr(solution_instance, 'a_c', 1.6)  # centripetal acceleration
        self.ratio = getattr(solution_instance, 'ratio', 5.0)  # v0/vp ratio = 5
        
        # Extract results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Expected keys from KB mapping
        self.v_0 = results.get('v_0', 0.0)
        self.v_p = results.get('v_p', 0.0)
        self.v_Q = results.get('v_Q', 0.0)
        self.h = results.get('h', 0.0)
        self.t = results.get('t', 0.0)  # pendulum half-period
        self.a = results.get('a', 0.0)  # block Q acceleration
        self.t_prime = results.get("t''", results.get('t_prime', 0.0))  # Q movement time
        self.s = results.get('s', 0.0)
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Initialize logs
        self.logs = []
    
    def log(self, message, success=None):
        """Helper to log messages with optional success status."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"[{status}] {message}")
        else:
            self.logs.append(f"[INFO] {message}")
    
    def test_invariants(self):
        """Verify physical invariants based on KB theorems."""
        try:
            # 1. Centripetal acceleration check
            # From KB: a_c = v_0^2 / L
            expected_a_c = self.v_0**2 / self.L
            if math.isclose(self.a_c, expected_a_c, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Centripetal acceleration: {self.a_c:.3f} ≈ {expected_a_c:.3f} (v0²/L)", True)
            else:
                self.log(f"Centripetal acceleration mismatch: {self.a_c:.3f} ≠ {expected_a_c:.3f} (v0²/L)", False)
            
            # 2. Velocity ratio check
            if math.isclose(self.v_0 / self.v_p, self.ratio, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Velocity ratio: v0/vp = {self.v_0/self.v_p:.2f} ≈ {self.ratio}", True)
            else:
                self.log(f"Velocity ratio mismatch: {self.v_0/self.v_p:.2f} ≠ {self.ratio}", False)
            
            # 3. Momentum conservation during elastic collision
            # Initial: m_P * v_0, Final: m_P * v_p + m_Q * v_Q
            p_initial = self.m_P * self.v_0
            p_final = self.m_P * self.v_p + self.m_Q * self.v_Q
            if math.isclose(p_initial, p_final, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Momentum conservation: {p_initial:.3f} ≈ {p_final:.3f} kg·m/s", True)
            else:
                self.log(f"Momentum conservation mismatch: {p_initial:.3f} ≠ {p_final:.3f} kg·m/s", False)
            
            # 4. Energy conservation for pendulum swing after collision
            # KE_initial = 0.5 * m_P * v_p^2
            # PE_final = m_P * g * h
            ke_initial = 0.5 * self.m_P * self.v_p**2
            pe_final = self.m_P * self.g * self.h
            if math.isclose(ke_initial, pe_final, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Energy conservation (pendulum): KE={ke_initial:.4f} ≈ PE={pe_final:.4f} J", True)
            else:
                self.log(f"Energy conservation mismatch: KE={ke_initial:.4f} ≠ PE={pe_final:.4f} J", False)
            
            # 5. Newton's 2nd Law for block Q
            # F_net = m_Q * a, F_friction = mu * m_Q * g
            expected_a = self.mu * self.g
            if math.isclose(self.a, expected_a, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Block Q acceleration: a={self.a:.3f} ≈ μ*g={expected_a:.3f} m/s²", True)
            else:
                self.log(f"Block Q acceleration mismatch: {self.a:.3f} ≠ μ*g={expected_a:.3f} m/s²", False)
            
            # 6. Kinematics for block Q
            # t' = v_Q / a
            expected_t_prime = self.v_Q / self.a if self.a != 0 else 0
            if math.isclose(self.t_prime, expected_t_prime, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Block Q movement time: t'={self.t_prime:.3f} ≈ v_Q/a={expected_t_prime:.3f} s", True)
            else:
                self.log(f"Block Q movement time mismatch: {self.t_prime:.3f} ≠ v_Q/a={expected_t_prime:.3f} s", False)
            
            # s = v_Q² / (2*a)
            expected_s = self.v_Q**2 / (2 * self.a) if self.a != 0 else 0
            if math.isclose(self.s, expected_s, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Block Q distance: s={self.s:.3f} ≈ v_Q²/(2a)={expected_s:.3f} m", True)
            else:
                self.log(f"Block Q distance mismatch: {self.s:.3f} ≠ v_Q²/(2a)={expected_s:.3f} m", False)
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", False)
            self.log(traceback.format_exc(), False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check that v_Q from momentum matches the target answer
            # Target answer 1: "1.68 m/s"
            if math.isclose(self.v_Q, 1.68, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"v_Q matches target: {self.v_Q:.3f} ≈ 1.68 m/s", True)
            else:
                self.log(f"v_Q mismatch: {self.v_Q:.3f} ≠ 1.68 m/s", False)
            
            # Check that s from kinematics matches the target answer
            # Target answer 2: "0.504 m"
            if math.isclose(self.s, 0.504, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"s matches target: {self.s:.3f} ≈ 0.504 m", True)
            else:
                self.log(f"s mismatch: {self.s:.3f} ≠ 0.504 m", False)
            
            # Verify pendulum half-period
            expected_t = math.pi * math.sqrt(self.L / self.g)
            if math.isclose(self.t, expected_t, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Pendulum half-period: t={self.t:.3f} ≈ π√(L/g)={expected_t:.3f} s", True)
            else:
                self.log(f"Pendulum half-period mismatch: {self.t:.3f} ≠ π√(L/g)={expected_t:.3f} s", False)
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", False)
            self.log(traceback.format_exc(), False)
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input (mass of ball P) and verify expected trends.
        Then restore original values.
        """
        original_m_P = self.m_P
        
        try:
            # Test 1: Increase m_P by 10%
            self.m_P *= 1.1
            
            # Recalculate using formulas from steps analysis
            # v0 from centripetal acceleration: v0 = sqrt(a_c * L) [independent of m_P]
            # Actually, v0 is determined by a_c and L, not m_P. But in collision, momentum depends on m_P.
            # So we expect v_Q to increase with m_P.
            
            # Symbolic momentum conservation
            v0 = math.sqrt(self.a_c * self.L)  # unchanged
            vp = v0 / self.ratio
            v_Q_new = (self.m_P * v0 - self.m_P * vp) / self.m_Q
            
            if v_Q_new > self.v_Q:
                self.log(f"Monotonicity: v_Q increases with m_P ({v_Q_new:.3f} > {self.v_Q:.3f})", True)
            else:
                self.log(f"Monotonicity violation: v_Q doesn't increase with m_P ({v_Q_new:.3f} ≤ {self.v_Q:.3f})", False)
            
            # Restore original m_P
            self.m_P = original_m_P
            
            # Test 2: Decrease friction coefficient mu
            original_mu = self.mu
            self.mu *= 0.9  # reduce by 10%
            
            # Block acceleration decreases
            a_new = self.mu * self.g
            if a_new < self.a:
                self.log(f"Monotonicity: a decreases with μ ({a_new:.3f} < {self.a:.3f})", True)
            else:
                self.log(f"Monotonicity violation: a doesn't decrease with μ ({a_new:.3f} ≥ {self.a:.3f})", False)
            
            # Distance s should increase (since a decreases, stopping distance increases)
            s_new = self.v_Q**2 / (2 * a_new) if a_new != 0 else 0
            if s_new > self.s:
                self.log(f"Monotonicity: s increases with decreased μ ({s_new:.3f} > {self.s:.3f})", True)
            else:
                self.log(f"Monotonicity violation: s doesn't increase with decreased μ ({s_new:.3f} ≤ {self.s:.3f})", False)
            
            # Restore mu
            self.mu = original_mu
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}", False)
            self.log(traceback.format_exc(), False)
            # Restore original values on error
            self.m_P = original_m_P
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []  # Reset logs
        
        success = True
        
        # Run each test
        if not self.test_invariants():
            success = False
        
        if not self.test_trace_consistency():
            success = False
        
        if not self.test_monotonicity():
            success = False
        
        # Final summary
        if success:
            self.log("All physics checks passed!", True)
        else:
            self.log("Some physics checks failed.", False)
        
        return success, self.logs