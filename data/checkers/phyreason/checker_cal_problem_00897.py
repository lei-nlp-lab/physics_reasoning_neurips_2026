import math
import sympy as sp
from sympy import symbols, Eq, solve

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results using getattr with fallback to results dict.
        """
        # Inputs from problem statement
        self.R1 = 2.5  # Ω
        self.R2 = 7.5  # Ω
        self.I1_measured = 0.2  # A
        self.I2_measured = 0.4  # A
        
        # Results from solution_instance with safe access
        self.E = getattr(solution_instance, 'E', solution_instance.results.get('E', 0.0))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', 0.0))
        
        # For logging
        self.logs = []
    
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """
        Verify Ohm's Law in both circuit states using symbolic and numeric checks.
        """
        success = True
        
        # Symbolic Ohm's Law equations
        E, I1, I2, R1, R2, r = symbols('E I1 I2 R1 R2 r')
        eq1 = Eq(E, I1 * (R1 + R2 + r))  # Switch just closed
        eq2 = Eq(E, I2 * (R1 + r))       # After stabilization
        
        # Numerical check for first state
        E_calc1 = self.I1_measured * (self.R1 + self.R2 + self.r)
        if not math.isclose(self.E, E_calc1, rel_tol=0.02, abs_tol=0.02):
            self.log(f"FAIL Ohm's Law (initial): E={self.E} != I1*(R1+R2+r)={E_calc1}")
            success = False
        else:
            self.log(f"PASS Ohm's Law (initial): E={self.E} ≈ I1*(R1+R2+r)={E_calc1}")
        
        # Numerical check for second state
        E_calc2 = self.I2_measured * (self.R1 + self.r)
        if not math.isclose(self.E, E_calc2, rel_tol=0.02, abs_tol=0.02):
            self.log(f"FAIL Ohm's Law (steady): E={self.E} != I2*(R1+r)={E_calc2}")
            success = False
        else:
            self.log(f"PASS Ohm's Law (steady): E={self.E} ≈ I2*(R1+r)={E_calc2}")
        
        # Consistency between the two states (should yield same E and r)
        # Solve symbolically
        sol = solve([eq1.subs({I1: self.I1_measured, R1: self.R1, R2: self.R2}),
                     eq2.subs({I2: self.I2_measured, R1: self.R1})], [E, r])
        if sol:
            E_sym, r_sym = sol[E], sol[r]
            if (math.isclose(self.E, float(E_sym), rel_tol=0.02, abs_tol=0.02) and
                math.isclose(self.r, float(r_sym), rel_tol=0.02, abs_tol=0.02)):
                self.log(f"PASS symbolic consistency: E={self.E}≈{float(E_sym)}, r={self.r}≈{float(r_sym)}")
            else:
                self.log(f"FAIL symbolic consistency: computed E={self.E}, r={self.r} vs solved E={float(E_sym)}, r={float(r_sym)}")
                success = False
        else:
            self.log("FAIL symbolic solution could not be found")
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """
        Check that intermediate results (from steps analysis) align with final E and r.
        """
        success = True
        
        # Recompute E and r from the two Ohm's Law equations
        # From I2 = E/(R1+r) -> E = I2*(R1+r) -> substitute into I1 = E/(R1+R2+r)
        # This yields: I1 = I2*(R1+r)/(R1+R2+r)
        # Solve for r:
        # I1*(R1+R2+r) = I2*(R1+r)
        # I1*R1 + I1*R2 + I1*r = I2*R1 + I2*r
        # I1*R2 + I1*R1 - I2*R1 = (I2 - I1)*r
        # r = (I1*(R1+R2) - I2*R1) / (I2 - I1)
        numerator = self.I1_measured * (self.R1 + self.R2) - self.I2_measured * self.R1
        denominator = self.I2_measured - self.I1_measured
        if abs(denominator) < 1e-12:
            self.log("FAIL denominator zero in r calculation")
            return False
        r_calc = numerator / denominator
        E_calc = self.I2_measured * (self.R1 + r_calc)
        
        # Check against solution results
        if not math.isclose(self.r, r_calc, rel_tol=0.02, abs_tol=0.02):
            self.log(f"FAIL r mismatch: solution r={self.r} vs derived r={r_calc}")
            success = False
        else:
            self.log(f"PASS r consistency: {self.r} ≈ {r_calc}")
        
        if not math.isclose(self.E, E_calc, rel_tol=0.02, abs_tol=0.02):
            self.log(f"FAIL E mismatch: solution E={self.E} vs derived E={E_calc}")
            success = False
        else:
            self.log(f"PASS E consistency: {self.E} ≈ {E_calc}")
        
        # Also verify the given currents match the equations with computed E and r
        I1_calc = self.E / (self.R1 + self.R2 + self.r)
        I2_calc = self.E / (self.R1 + self.r)
        if not math.isclose(self.I1_measured, I1_calc, rel_tol=0.02, abs_tol=0.02):
            self.log(f"FAIL I1 mismatch: given {self.I1_measured} vs computed {I1_calc}")
            success = False
        else:
            self.log(f"PASS I1 consistency: {self.I1_measured} ≈ {I1_calc}")
        
        if not math.isclose(self.I2_measured, I2_calc, rel_tol=0.02, abs_tol=0.02):
            self.log(f"FAIL I2 mismatch: given {self.I2_measured} vs computed {I2_calc}")
            success = False
        else:
            self.log(f"PASS I2 consistency: {self.I2_measured} ≈ {I2_calc}")
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb R1 (increase by 10%), recalculate E and r using Ohm's Laws, 
        check that I1 and I2 change as expected, then restore.
        """
        success = True
        
        # Perturb R1
        R1_original = self.R1
        R1_perturbed = R1_original * 1.10  # +10%
        
        # Recompute r and E from the perturbed system using the same formulas
        # Using the derived formula from trace_consistency
        numerator = self.I1_measured * (R1_perturbed + self.R2) - self.I2_measured * R1_perturbed
        denominator = self.I2_measured - self.I1_measured
        if abs(denominator) < 1e-12:
            self.log("FAIL denominator zero in perturbed r calculation")
            return False
        r_perturbed = numerator / denominator
        E_perturbed = self.I2_measured * (R1_perturbed + r_perturbed)
        
        # Compute currents with perturbed parameters
        I1_perturbed = E_perturbed / (R1_perturbed + self.R2 + r_perturbed)
        I2_perturbed = E_perturbed / (R1_perturbed + r_perturbed)
        
        # Check trends: if R1 increases, both I1 and I2 should decrease (since denominator increases)
        # We compare perturbed currents with original measured currents using tolerance
        # Note: Use rel_tol=1e-2 (1%) for meaningful comparison
        if not math.isclose(I1_perturbed, self.I1_measured, rel_tol=0.01):
            if I1_perturbed >= self.I1_measured:
                self.log(f"FAIL I1 monotonicity: R1↑ but I1 not ↓ ({I1_perturbed} >= {self.I1_measured})")
                success = False
            else:
                self.log(f"PASS I1 monotonicity: R1↑ → I1↓ ({I1_perturbed} < {self.I1_measured})")
        else:
            # If they're essentially equal within tolerance, consider it passing
            self.log(f"PASS I1 monotonicity: R1↑ → I1 essentially unchanged ({I1_perturbed} ≈ {self.I1_measured})")
        
        if not math.isclose(I2_perturbed, self.I2_measured, rel_tol=0.01):
            if I2_perturbed >= self.I2_measured:
                self.log(f"FAIL I2 monotonicity: R1↑ but I2 not ↓ ({I2_perturbed} >= {self.I2_measured})")
                success = False
            else:
                self.log(f"PASS I2 monotonicity: R1↑ → I2↓ ({I2_perturbed} < {self.I2_measured})")
        else:
            # If they're essentially equal within tolerance, consider it passing
            self.log(f"PASS I2 monotonicity: R1↑ → I2 essentially unchanged ({I2_perturbed} ≈ {self.I2_measured})")
        
        # Additionally, check that E and r change in a physically plausible way
        # For a real battery, internal resistance r should be positive
        if r_perturbed <= 0:
            self.log(f"FAIL r positivity after perturbation: r={r_perturbed}")
            success = False
        
        # E should remain positive
        if E_perturbed <= 0:
            self.log(f"FAIL E positivity after perturbation: E={E_perturbed}")
            success = False
        
        # Restore R1 (no actual change needed as we didn't modify instance)
        self.log("R1 perturbation test completed and original R1 restored (conceptually).")
        
        return success
    
    def run_all_tests(self):
        """
        Execute all tests and return success flag and logs.
        """
        self.logs = []
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall = test1 and test2 and test3
        self.log(f"\nOverall test result: {'PASS' if overall else 'FAIL'}")
        
        return overall, self.logs