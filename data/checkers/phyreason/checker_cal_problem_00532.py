import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        Extracts all required inputs and results from the solution instance.
        """
        self.solution = solution_instance
        # Fixed inputs from problem statement
        self.E = 4.5          # Electromotive force (V)
        self.R = 4.0          # External resistance (Ω)
        self.U_terminal = 4.0 # Terminal voltage when R=4.0Ω (V)
        self.R1 = 4.0         # Parallel resistor (Ω) - from steps analysis
        self.R2 = 9.0         # Resistor for heat calculation (Ω) - from steps analysis
        self.t = 30.0         # Time for heat calculation (s)
        
        # Extract results with safe fallbacks
        self.r = self._safe_get('r')
        self.U1 = self._safe_get('U_1')
        self.Q = self._safe_get('Q')
        self.I = self._safe_get('I')
        self.I2 = self._safe_get('I_2')
        self.R_ext = self._safe_get('R_ext')
        
        # Initialize symbolic variables for symbolic checks
        self.E_sym = sp.Symbol('E')
        self.R_sym = sp.Symbol('R')
        self.r_sym = sp.Symbol('r')
        self.I_sym = sp.Symbol('I')
        self.U_sym = sp.Symbol('U')
        self.Q_sym = sp.Symbol('Q')
        self.t_sym = sp.Symbol('t')
        self.R2_sym = sp.Symbol('R2')
        
    def _safe_get(self, key):
        """Safe extraction of attributes from solution instance."""
        return getattr(self.solution, key, getattr(self.solution, 'results', {}).get(key, 0.0))
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self):
        """Test fundamental physics invariants using Ohm's and Joule's laws."""
        logs = []
        success = True
        
        try:
            # 1. Ohm's law for closed circuit: E = I*(R + r)
            lhs1 = self.E
            rhs1 = self.I * (self.R + self.r)
            if not self._is_close(lhs1, rhs1):
                logs.append(f"FAIL: Ohm's law (E = I*(R+r)): {lhs1:.3f} ≠ {rhs1:.3f}")
                success = False
            else:
                logs.append(f"PASS: Ohm's law (E = I*(R+r)): {lhs1:.3f} ≈ {rhs1:.3f}")
            
            # 2. Terminal voltage relationship: U = I*R
            lhs2 = self.U_terminal
            rhs2 = self.I * self.R
            if not self._is_close(lhs2, rhs2):
                logs.append(f"FAIL: Terminal voltage (U = I*R): {lhs2:.3f} ≠ {rhs2:.3f}")
                success = False
            else:
                logs.append(f"PASS: Terminal voltage (U = I*R): {lhs2:.3f} ≈ {rhs2:.3f}")
            
            # 3. Joule's law: Q = I2² * R2 * t
            lhs3 = self.Q
            rhs3 = self.I2**2 * self.R2 * self.t
            if not self._is_close(lhs3, rhs3):
                logs.append(f"FAIL: Joule's law (Q = I2²*R2*t): {lhs3:.3f} ≠ {rhs3:.3f}")
                success = False
            else:
                logs.append(f"PASS: Joule's law (Q = I2²*R2*t): {lhs3:.3f} ≈ {rhs3:.3f}")
            
            # 4. Parallel resistance formula
            expected_R_ext = (self.R * self.R1) / (self.R + self.R1)
            if not self._is_close(self.R_ext, expected_R_ext):
                logs.append(f"FAIL: Parallel resistance: {self.R_ext:.3f} ≠ {expected_R_ext:.3f}")
                success = False
            else:
                logs.append(f"PASS: Parallel resistance: {self.R_ext:.3f} ≈ {expected_R_ext:.3f}")
            
            # 5. Ohm's law for parallel circuit: U1 = I1 * R_ext
            I1 = self.E / (self.R_ext + self.r)
            lhs5 = self.U1
            rhs5 = I1 * self.R_ext
            if not self._is_close(lhs5, rhs5):
                logs.append(f"FAIL: U1 = I1*R_ext: {lhs5:.3f} ≠ {rhs5:.3f}")
                success = False
            else:
                logs.append(f"PASS: U1 = I1*R_ext: {lhs5:.3f} ≈ {rhs5:.3f}")
            
            # 6. Symbolic Ohm's law check (extra robustness)
            eq_symbolic = sp.Eq(self.E_sym, self.I_sym * (self.R_sym + self.r_sym))
            lhs_sym = eq_symbolic.lhs.subs({self.E_sym: self.E})
            rhs_sym = eq_symbolic.rhs.subs({
                self.I_sym: self.I,
                self.R_sym: self.R,
                self.r_sym: self.r
            })
            if not self._is_close(float(lhs_sym), float(rhs_sym)):
                logs.append(f"FAIL: Symbolic Ohm's law: {float(lhs_sym):.3f} ≠ {float(rhs_sym):.3f}")
                success = False
            else:
                logs.append(f"PASS: Symbolic Ohm's law check")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check internal resistance calculation
            r_calculated = (self.E - self.U_terminal) / self.I
            if not self._is_close(self.r, r_calculated):
                logs.append(f"FAIL: r mismatch: {self.r:.3f} ≠ {r_calculated:.3f}")
                success = False
            else:
                logs.append(f"PASS: r consistent: {self.r:.3f} ≈ {r_calculated:.3f}")
            
            # Check U1 consistency with circuit
            I1_calculated = self.E / (self.R_ext + self.r)
            U1_calculated = I1_calculated * self.R_ext
            if not self._is_close(self.U1, U1_calculated):
                logs.append(f"FAIL: U1 mismatch: {self.U1:.3f} ≠ {U1_calculated:.3f}")
                success = False
            else:
                logs.append(f"PASS: U1 consistent: {self.U1:.3f} ≈ {U1_calculated:.3f}")
            
            # Check I2 consistency with total circuit resistance
            I2_calculated = self.E / (self.R + self.R2 + self.r)
            if not self._is_close(self.I2, I2_calculated):
                logs.append(f"FAIL: I2 mismatch: {self.I2:.3f} ≠ {I2_calculated:.3f}")
                success = False
            else:
                logs.append(f"PASS: I2 consistent: {self.I2:.3f} ≈ {I2_calculated:.3f}")
            
            # Check Q consistency with I2
            Q_calculated = self.I2**2 * self.R2 * self.t
            if not self._is_close(self.Q, Q_calculated):
                logs.append(f"FAIL: Q mismatch: {self.Q:.3f} ≠ {Q_calculated:.3f}")
                success = False
            else:
                logs.append(f"PASS: Q consistent: {self.Q:.3f} ≈ {Q_calculated:.3f}")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic relationships by perturbing inputs."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase external resistance -> terminal voltage increases
            R_perturbed = self.R * 1.1  # +10%
            I_perturbed = self.E / (R_perturbed + self.r)
            U_perturbed = I_perturbed * R_perturbed
            
            if U_perturbed <= self.U_terminal:
                logs.append(f"FAIL: Terminal voltage monotonicity: U({R_perturbed:.2f}) = {U_perturbed:.3f} ≤ {self.U_terminal:.3f}")
                success = False
            else:
                logs.append(f"PASS: Terminal voltage increases with R: {U_perturbed:.3f} > {self.U_terminal:.3f}")
            
            # Test 2: Increase R2 -> I2 decreases
            R2_perturbed = self.R2 * 1.1
            I2_perturbed = self.E / (self.R + R2_perturbed + self.r)
            
            if I2_perturbed >= self.I2:
                logs.append(f"FAIL: I2 monotonicity: I2({R2_perturbed:.2f}) = {I2_perturbed:.3f} ≥ {self.I2:.3f}")
                success = False
            else:
                logs.append(f"PASS: I2 decreases with R2: {I2_perturbed:.3f} < {self.I2:.3f}")
            
            # Test 3: Increase I2 -> Q increases quadratically
            Q_original = self.I2**2 * self.R2 * self.t
            I2_perturbed2 = self.I2 * 1.1
            Q_perturbed = I2_perturbed2**2 * self.R2 * self.t
            expected_increase = (1.1**2 - 1) * 100  # ~21% increase
            
            if Q_perturbed <= Q_original:
                logs.append(f"FAIL: Q monotonicity: Q({I2_perturbed2:.3f}) = {Q_perturbed:.3f} ≤ {Q_original:.3f}")
                success = False
            else:
                actual_increase = ((Q_perturbed - Q_original) / Q_original) * 100
                logs.append(f"PASS: Q increases quadratically with I2: +{actual_increase:.1f}% (expected ~{expected_increase:.1f}%)")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run invariant tests
        success, logs = self.test_invariants()
        all_logs.append("=== INVARIANT TESTS ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== TRACE CONSISTENCY TESTS ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== MONOTONICITY TESTS ===")
        all_logs.extend(logs)
        overall_success = overall_success and success
        
        # Summary
        all_logs.append(f"\n=== OVERALL RESULT: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs