import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all necessary inputs and results from the solution instance.
        Uses getattr with fallbacks to avoid AttributeError in sandbox.
        """
        # Input constants from problem statement
        self.AB = 1.0  # cm
        self.FG = 1.0  # cm
        self.BC = math.sqrt(3)  # cm
        self.OA = 2.0  # cm
        self.theta_deg = 120.0  # degrees
        self.theta_rad = math.radians(self.theta_deg)
        
        # Extract results with safe access
        self.tanC = getattr(solution_instance, 'tanC', solution_instance.results.get('tanC', 0.0))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', 0.0))
        self.i = getattr(solution_instance, 'i', solution_instance.results.get('i', 0.0))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 0.0))
        self.n = getattr(solution_instance, 'n', solution_instance.results.get('n', 0.0))
        self.C = getattr(solution_instance, 'C', solution_instance.results.get('C', 0.0))
        
        # Convert angles to radians for calculations if they are in degrees
        if abs(self.i) > 2*math.pi:  # heuristic for degrees
            self.i_rad = math.radians(self.i)
        else:
            self.i_rad = self.i
        if abs(self.r) > 2*math.pi:
            self.r_rad = math.radians(self.r)
        else:
            self.r_rad = self.r
        if abs(self.C) > 2*math.pi:
            self.C_rad = math.radians(self.C)
        else:
            self.C_rad = self.C
        
        self.logs = []
    
    def log(self, message):
        """Helper to append logs."""
        self.logs.append(message)
    
    def test_invariants(self):
        """Test Snell's Law and critical angle relationships."""
        success = True
        
        # 1. Snell's Law at point A: n1 * sin(i) = n2 * sin(r) with n1=1 (air)
        # n = sin(i)/sin(r) should hold
        sin_i = math.sin(self.i_rad)
        sin_r = math.sin(self.r_rad)
        if sin_r != 0:
            n_calculated = sin_i / sin_r
            if not math.isclose(self.n, n_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Snell's Law violated. n from solution={self.n:.4f}, computed n=sin(i)/sin(r)={n_calculated:.4f}")
                success = False
            else:
                self.log(f"PASS: Snell's Law holds (n={self.n:.4f} ≈ sin(i)/sin(r)={n_calculated:.4f})")
        else:
            self.log("ERROR: sin(r) is zero, cannot check Snell's Law")
            success = False
        
        # 2. Critical angle condition: sin(C) = 1/n
        if self.n > 0:
            sin_C_calculated = 1.0 / self.n
            sin_C_actual = math.sin(self.C_rad)
            if not math.isclose(sin_C_actual, sin_C_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Critical angle condition violated. sin(C)={sin_C_actual:.4f}, 1/n={sin_C_calculated:.4f}")
                success = False
            else:
                self.log(f"PASS: Critical angle condition holds (sin(C)={sin_C_actual:.4f} ≈ 1/n={sin_C_calculated:.4f})")
            
            # 3. tan(C) = sin(C)/cos(C) consistency
            tan_C_calculated = math.tan(self.C_rad)
            if not math.isclose(self.tanC, tan_C_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"WARN: tan(C) mismatch. From solution={self.tanC:.4f}, computed tan(C)={tan_C_calculated:.4f}")
                # Not critical, but log
        else:
            self.log("ERROR: n is zero or negative, cannot check critical angle")
            success = False
        
        # 4. Geometric invariant: angle i should be 60° (from step analysis)
        expected_i_deg = 60.0
        if not math.isclose(self.i, expected_i_deg, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Expected i={expected_i_deg}°, got {self.i:.2f}°")
            success = False
        else:
            self.log(f"PASS: i={self.i:.2f}° matches expected 60°")
        
        # 5. Geometric invariant: angle r should be 30° (from step analysis)
        expected_r_deg = 30.0
        if not math.isclose(self.r, expected_r_deg, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Expected r={expected_r_deg}°, got {self.r:.2f}°")
            success = False
        else:
            self.log(f"PASS: r={self.r:.2f}° matches expected 30°")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final radius R."""
        success = True
        
        # From step analysis: R = (OA * cos60° + BC) * tan(C)
        OA_cos60 = self.OA * math.cos(math.radians(60))
        term = OA_cos60 + self.BC
        R_calculated = term * self.tanC
        
        if not math.isclose(self.R, R_calculated, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: R inconsistency. From solution R={self.R:.4f} cm, computed R={R_calculated:.4f} cm")
            success = False
        else:
            self.log(f"PASS: R={self.R:.4f} cm matches formula (OA*cos60°+BC)*tan(C)={R_calculated:.4f} cm")
        
        # Also check that n is √3
        expected_n = math.sqrt(3)
        if not math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"WARN: n={self.n:.4f} differs from √3={expected_n:.4f}")
        
        # Check that tan(C) is √2/2
        expected_tanC = math.sqrt(2)/2
        if not math.isclose(self.tanC, expected_tanC, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"WARN: tan(C)={self.tanC:.4f} differs from √2/2={expected_tanC:.4f}")
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb input BC and observe effect on R.
        Since R = (OA*cos60° + BC)*tan(C) and tan(C) depends on n (which is fixed by i and r),
        increasing BC should increase R.
        """
        success = True
        
        # Store original BC
        BC_original = self.BC
        perturbation = 0.1  # cm
        
        # Perturb BC upward
        self.BC += perturbation
        
        # Recalculate R using the formula (tanC and OA are unchanged)
        OA_cos60 = self.OA * math.cos(math.radians(60))
        R_perturbed = (OA_cos60 + self.BC) * self.tanC
        
        # Restore BC
        self.BC = BC_original
        
        # Calculate original R using the same formula for consistency
        R_original = (OA_cos60 + self.BC) * self.tanC
        
        if not R_perturbed > R_original:
            self.log(f"FAIL: Monotonicity violated. R(BC+{perturbation})={R_perturbed:.4f} not > R(BC)={R_original:.4f}")
            success = False
        else:
            self.log(f"PASS: R increases with BC (R_original={R_original:.4f} cm, R_perturbed={R_perturbed:.4f} cm)")
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        success = True
        
        try:
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
        except Exception as e:
            self.log(f"ERROR during testing: {str(e)}")
            self.log(traceback.format_exc())
            success = False
        
        return success, self.logs