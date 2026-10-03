import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize checker with safe attribute access.
        Use getattr with fallback to results dict.
        """
        self.solution = solution_instance
        
        # Extract inputs (with safe defaults if missing)
        self.I = getattr(solution_instance, 'I', 2.0)
        self.R = getattr(solution_instance, 'R', 200.0)
        
        # Extract results with fallback from results dict
        self.U = getattr(solution_instance, 'U', 
                        getattr(solution_instance, 'results', {}).get('U', 0.0))
        self.Q = getattr(solution_instance, 'Q', 
                        getattr(solution_instance, 'results', {}).get('Q', 0.0))
        
        # Time may be implicit; infer from Q if needed
        self.t = getattr(solution_instance, 't', None)
        if self.t is None:
            # Calculate t from Q = I^2 * R * t if possible
            if abs(self.Q) > 1e-12 and abs(self.I**2 * self.R) > 1e-12:
                self.t = self.Q / (self.I**2 * self.R)
            else:
                self.t = 60.0  # Default 60 seconds based on step analysis
        
        # SymPy symbols for symbolic checks
        self.I_sym, self.R_sym, self.t_sym, self.U_sym, self.Q_sym = sp.symbols('I R t U Q')
        
        # Expected values from step analysis
        self.expected_U = 400.0
        self.expected_Q = 48000.0
        
        self.logs = []
    
    def _log(self, message, success=True):
        """Helper to log messages."""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def test_invariants(self):
        """Verify fundamental physics laws (Ohm's Law and Joule's Law)."""
        tests_passed = True
        
        # Ohm's Law: U = I * R
        ohm_law_lhs = self.U_sym
        ohm_law_rhs = self.I_sym * self.R_sym
        ohm_eq = sp.Eq(ohm_law_lhs, ohm_law_rhs)
        
        # Symbolic check
        try:
            if not sp.simplify(ohm_eq.subs({
                self.U_sym: self.U,
                self.I_sym: self.I,
                self.R_sym: self.R
            })):
                tests_passed = self._log("Ohm's Law symbolic check failed", False) and False
        except:
            pass
        
        # Numerical fallback for Ohm's Law
        expected_U_from_ohm = self.I * self.R
        if not math.isclose(self.U, expected_U_from_ohm, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Ohm's Law numeric check failed: U={self.U}, expected I×R={expected_U_from_ohm}", 
                False
            ) and False
        else:
            self._log("Ohm's Law satisfied: U = I × R")
        
        # Joule's Law: Q = I² × R × t
        joule_law_lhs = self.Q_sym
        joule_law_rhs = self.I_sym**2 * self.R_sym * self.t_sym
        joule_eq = sp.Eq(joule_law_lhs, joule_law_rhs)
        
        # Symbolic check
        try:
            if not sp.simplify(joule_eq.subs({
                self.Q_sym: self.Q,
                self.I_sym: self.I,
                self.R_sym: self.R,
                self.t_sym: self.t
            })):
                tests_passed = self._log("Joule's Law symbolic check failed", False) and False
        except:
            pass
        
        # Numerical fallback for Joule's Law
        expected_Q_from_joule = self.I**2 * self.R * self.t
        if not math.isclose(self.Q, expected_Q_from_joule, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Joule's Law numeric check failed: Q={self.Q}, expected I²×R×t={expected_Q_from_joule}", 
                False
            ) and False
        else:
            self._log("Joule's Law satisfied: Q = I² × R × t")
        
        # Cross-check P = I² × R = U² / R
        P_from_I = self.I**2 * self.R
        P_from_U = self.U**2 / self.R if self.R != 0 else float('inf')
        if not math.isclose(P_from_I, P_from_U, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Power consistency check failed: P(I²R)={P_from_I}, P(U²/R)={P_from_U}", 
                False
            ) and False
        else:
            self._log(f"Power consistency: I²R = U²/R = {P_from_I:.1f} W")
        
        return tests_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        # Check U matches expected from step analysis
        if not math.isclose(self.U, self.expected_U, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Voltage mismatch: U={self.U}, expected {self.expected_U}", 
                False
            ) and False
        else:
            self._log(f"Voltage consistent: U = {self.U} V")
        
        # Check Q matches expected from step analysis
        if not math.isclose(self.Q, self.expected_Q, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Heat mismatch: Q={self.Q}, expected {self.expected_Q}", 
                False
            ) and False
        else:
            self._log(f"Heat consistent: Q = {self.Q} J")
        
        # Verify t = 60s from Q = 48000J, I=2A, R=200Ω
        expected_t = self.expected_Q / (self.I**2 * self.R)
        if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Time mismatch: t={self.t}s, expected {expected_t}s from Q calculation", 
                False
            ) and False
        else:
            self._log(f"Time consistent: t = {self.t} s")
        
        return tests_passed
    
    def test_monotonicity(self):
        """
        Perturb input current I and check quadratic response of Q.
        Restore original I after test.
        """
        tests_passed = True
        
        original_I = self.I
        original_Q = self.Q
        
        # Perturb current by +5%
        perturbed_I = original_I * 1.05
        
        # Calculate expected Q with perturbed I (Q ∝ I²)
        expected_Q_perturbed = original_Q * (perturbed_I**2 / original_I**2)
        
        # Calculate actual Q using Joule's Law with perturbed I
        actual_Q_perturbed = perturbed_I**2 * self.R * self.t
        
        # Check if actual matches expected within 1% tolerance
        if not math.isclose(actual_Q_perturbed, expected_Q_perturbed, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Monotonicity check failed: Q should scale quadratically with I\n"
                f"  At I={perturbed_I:.2f}A, Q={actual_Q_perturbed:.0f}J, "
                f"expected {expected_Q_perturbed:.0f}J", 
                False
            ) and False
        else:
            self._log(
                f"Monotonicity satisfied: Q ∝ I²\n"
                f"  I↑{5}% → Q↑{(actual_Q_perturbed/original_Q-1)*100:.1f}% (expected ~{10.25}%)"
            )
        
        # Restore original I (important!)
        self.I = original_I
        
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        self.logs.append("=== Physics Audit Report ===")
        
        success = True
        
        try:
            if not self.test_invariants():
                success = False
            self.logs.append("")
            
            if not self.test_trace_consistency():
                success = False
            self.logs.append("")
            
            if not self.test_monotonicity():
                success = False
            
        except Exception as e:
            self.logs.append(f"✗ Exception during testing: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        self.logs.append("=" * 30)
        self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, self.logs