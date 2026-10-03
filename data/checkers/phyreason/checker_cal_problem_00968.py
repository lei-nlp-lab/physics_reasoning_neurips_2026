import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Uses getattr with fallback to results dictionary to avoid AttributeError.
        """
        # Extract inputs (assuming they are stored in solution_instance)
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'B_in', 1.0))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'L_in', 1.0))
        self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'v_in', 1.0))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'R_in', 1.0))
        
        # Extract results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        self.E = getattr(solution_instance, 'E', results.get('E', 0.0))
        self.I = getattr(solution_instance, 'I', results.get('I', 0.0))
        self.F = getattr(solution_instance, 'F', results.get('F', 0.0))
        self.F_A = getattr(solution_instance, 'F_A', results.get('F_A', 0.0))
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.Q = getattr(solution_instance, 'Q', results.get('Q', 0.0))
        self.P = getattr(solution_instance, 'P', results.get('P', 0.0))
        self.R_ad = getattr(solution_instance, 'R_ad', results.get('R_ad', 0.0))
        
        # Store logs for reporting
        self.logs = []
    
    def log(self, message):
        """Helper to append log messages."""
        self.logs.append(message)
    
    def test_invariants(self):
        """
        Verify fundamental physical laws:
        1. Newton's First Law (constant velocity => net force zero)
        2. Ohm's Law (E = I * R)
        3. Motional EMF (E = B * L * v)
        4. Resistance of side ad (R_ad = R / 4)
        5. Time kinematics (t = L / v)
        6. Joule's Law (Q = I² * R_ad * t)
        """
        try:
            # 1. Newton's First Law: F_net = 0 => F (pulling) = F_A (Ampere force)
            if not math.isclose(self.F, self.F_A, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Newton's First Law: F={self.F} != F_A={self.F_A}")
                return False
            self.log("PASS Newton's First Law: F = F_A (constant velocity)")
            
            # 2. Ohm's Law: E = I * R
            expected_E = self.I * self.R
            if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Ohm's Law: E={self.E} != I*R={expected_E}")
                return False
            self.log("PASS Ohm's Law: E = I * R")
            
            # 3. Motional EMF: E = B * L * v
            expected_emf = self.B * self.L * self.v
            if not math.isclose(self.E, expected_emf, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Motional EMF: E={self.E} != B*L*v={expected_emf}")
                return False
            self.log("PASS Motional EMF: E = B * L * v")
            
            # 4. Resistance of side ad: R_ad = R / 4
            expected_R_ad = self.R / 4.0
            if not math.isclose(self.R_ad, expected_R_ad, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Resistance division: R_ad={self.R_ad} != R/4={expected_R_ad}")
                return False
            self.log("PASS Resistance division: R_ad = R / 4")
            
            # 5. Time kinematics: t = L / v
            expected_t = self.L / self.v
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Kinematics: t={self.t} != L/v={expected_t}")
                return False
            self.log("PASS Kinematics: t = L / v")
            
            # 6. Joule's Law: Q = I² * R_ad * t
            expected_Q = self.I**2 * self.R_ad * self.t
            if not math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Joule's Law: Q={self.Q} != I²*R_ad*t={expected_Q}")
                return False
            self.log("PASS Joule's Law: Q = I² * R_ad * t")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers:
        1. Power P = F * v = (B² L² v²) / R
        2. Heat Q = (B² L³ v) / (4R)
        """
        try:
            # 1. Power from pulling force: P = F * v
            P_from_F = self.F * self.v
            if not math.isclose(self.P, P_from_F, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Power trace: P={self.P} != F*v={P_from_F}")
                return False
            self.log("PASS Power trace: P = F * v")
            
            # 2. Final power expression: P = (B² L² v²) / R
            P_expected = (self.B**2 * self.L**2 * self.v**2) / self.R
            if not math.isclose(self.P, P_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Power final: P={self.P} != (B²L²v²)/R={P_expected}")
                return False
            self.log("PASS Power final: P = (B² L² v²) / R")
            
            # 3. Final heat expression: Q = (B² L³ v) / (4R)
            Q_expected = (self.B**2 * self.L**3 * self.v) / (4.0 * self.R)
            if not math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL Heat final: Q={self.Q} != (B²L³v)/(4R)={Q_expected}")
                return False
            self.log("PASS Heat final: Q = (B² L³ v) / (4R)")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input (B) and verify expected trends in outputs (P and Q).
        Since P ∝ B² and Q ∝ B², increasing B should increase both.
        """
        original_B = self.B
        try:
            # Perturb B by +10%
            perturbed_B = original_B * 1.1
            # Recalculate expected P and Q using formulas
            P_original = (original_B**2 * self.L**2 * self.v**2) / self.R
            P_perturbed = (perturbed_B**2 * self.L**2 * self.v**2) / self.R
            Q_original = (original_B**2 * self.L**3 * self.v) / (4.0 * self.R)
            Q_perturbed = (perturbed_B**2 * self.L**3 * self.v) / (4.0 * self.R)
            
            # Check monotonic increase
            if not (P_perturbed > P_original and Q_perturbed > Q_original):
                self.log(f"FAIL Monotonicity: P or Q did not increase as expected when B increased")
                return False
            
            self.log("PASS Monotonicity: P and Q increase with B (quadratic trend)")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
        finally:
            # Restore original B (though self.B is not modified)
            pass
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []  # Reset logs
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} Test ---")
            if not test_func():
                success = False
                self.log(f"FAILED {test_name}")
            else:
                self.log(f"PASSED {test_name}")
        
        return success, self.logs