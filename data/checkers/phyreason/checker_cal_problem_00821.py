import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all needed attributes using getattr with fallbacks
        self.solution = solution_instance
        # Extract input parameters
        self.k = getattr(solution_instance, 'k', getattr(solution_instance, 'inputs', {}).get('k', 1.5e5))
        self.M = getattr(solution_instance, 'M', getattr(solution_instance, 'inputs', {}).get('M', 600.0))
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 70.0))
        self.f = getattr(solution_instance, 'f', getattr(solution_instance, 'inputs', {}).get('f', 2.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 9.8))
        
        # Extract results from solution_instance with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.l = self.results.get('l', 0.0)  # compression length in meters
        self.n = self.results.get('n', 0.0)  # number of people
        
        # Store logs for output
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify fundamental physical laws: frequency formula and static equilibrium."""
        success = True
        
        # 1. Check frequency formula: f = (1/(2π)) * sqrt(g/l)
        try:
            # Numerical check with tolerance
            lhs = self.f
            rhs = 1/(2*math.pi) * math.sqrt(self.g/self.l) if self.l > 0 else 0
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Frequency formula mismatch. f={lhs}, calculated={rhs}")
                success = False
            else:
                self.log(f"PASS: Frequency formula validated (f={lhs} Hz, l={self.l} m)")
        except Exception as e:
            self.log(f"ERROR in frequency check: {traceback.format_exc()}")
            success = False
        
        # 2. Check static equilibrium with Hooke's Law: k*l = (M + n*m)*g
        try:
            # Total mass including passengers
            total_mass = self.M + self.n * self.m
            spring_force = self.k * self.l
            weight_force = total_mass * self.g
            
            if not math.isclose(spring_force, weight_force, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Static equilibrium mismatch. k*l={spring_force}, (M+n*m)g={weight_force}")
                success = False
            else:
                self.log(f"PASS: Static equilibrium validated (k*l = {spring_force} N, total weight = {weight_force} N)")
        except Exception as e:
            self.log(f"ERROR in equilibrium check: {traceback.format_exc()}")
            success = False
        
        # 3. Additional symbolic check using sympy (for completeness)
        try:
            l_sym, f_sym, g_sym = sp.symbols('l f g', positive=True)
            freq_eq = sp.Eq(f_sym, 1/(2*sp.pi) * sp.sqrt(g_sym/l_sym))
            self.log(f"Symbolic frequency formula: {freq_eq}")
            
            # Substitute numerical values
            lhs_val = freq_eq.lhs.subs({f_sym: self.f})
            rhs_val = freq_eq.rhs.subs({g_sym: self.g, l_sym: self.l})
            if not math.isclose(float(lhs_val), float(rhs_val), rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Symbolic check mismatch: LHS={float(lhs_val)}, RHS={float(rhs_val)}")
        except Exception as e:
            self.log(f"Symbolic check note: {e}")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate and final results with the step-by-step analysis."""
        success = True
        
        # Step 1: Check compression length l from frequency formula
        try:
            expected_l = self.g / (4 * math.pi**2 * self.f**2)
            if not math.isclose(self.l, expected_l, rel_tol=1e-2, abs_tol=1e-3):
                self.log(f"FAIL: Compression length mismatch. Got l={self.l} m, expected l={expected_l} m")
                success = False
            else:
                self.log(f"PASS: Compression length l={self.l} m matches formula l=g/(4π²f²)")
        except Exception as e:
            self.log(f"ERROR in step 1 consistency: {traceback.format_exc()}")
            success = False
        
        # Step 2: Check number of passengers n from equilibrium
        try:
            expected_n = (self.k * self.l - self.M * self.g) / (self.m * self.g)
            # n should be approximately integer, but check numerically
            if not math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Passenger count mismatch. Got n={self.n}, expected n≈{expected_n}")
                success = False
            else:
                self.log(f"PASS: Passenger count n={self.n} matches formula n=(k*l - M*g)/(m*g)")
        except Exception as e:
            self.log(f"ERROR in step 2 consistency: {traceback.format_exc()}")
            success = False
        
        # Check that n is reasonable (non-negative, integer-like)
        if success and self.n < 0:
            self.log(f"WARNING: Negative passenger count n={self.n}")
        if success and abs(self.n - round(self.n)) > 0.1:
            self.log(f"WARNING: Passenger count n={self.n} is not near integer")
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        success = True
        
        # Test 1: Increase spring constant k -> compression length l should decrease (for fixed frequency)
        try:
            original_k = self.k
            perturbed_k = original_k * 1.1  # Increase by 10%
            
            # Recalculate l using frequency formula (independent of k)
            # Actually, from f = (1/(2π)) * sqrt(g/l), l depends only on f and g, not k.
            # So l should remain constant when k changes (for fixed f).
            # But for monotonicity with respect to k in the equilibrium equation:
            # n = (k*l - M*g)/(m*g), with l fixed by frequency, n increases with k.
            
            # Calculate original and perturbed n (with fixed l from frequency formula)
            original_n = (original_k * self.l - self.M * self.g) / (self.m * self.g)
            perturbed_n = (perturbed_k * self.l - self.M * self.g) / (self.m * self.g)
            
            if perturbed_n <= original_n:
                self.log(f"FAIL: Increasing k from {original_k} to {perturbed_k} should increase n, but got n: {original_n} -> {perturbed_n}")
                success = False
            else:
                self.log(f"PASS: Increasing k increases n as expected ({original_n} -> {perturbed_n})")
        except Exception as e:
            self.log(f"ERROR in k monotonicity check: {traceback.format_exc()}")
            success = False
        
        # Test 2: Increase car mass M -> n should decrease (for fixed k and l)
        try:
            original_M = self.M
            perturbed_M = original_M * 1.1  # Increase by 10%
            
            original_n = (self.k * self.l - original_M * self.g) / (self.m * self.g)
            perturbed_n = (self.k * self.l - perturbed_M * self.g) / (self.m * self.g)
            
            if perturbed_n >= original_n:
                self.log(f"FAIL: Increasing M from {original_M} to {perturbed_M} should decrease n, but got n: {original_n} -> {perturbed_n}")
                success = False
            else:
                self.log(f"PASS: Increasing M decreases n as expected ({original_n} -> {perturbed_n})")
        except Exception as e:
            self.log(f"ERROR in M monotonicity check: {traceback.format_exc()}")
            success = False
        
        # Test 3: Increase frequency f -> l should decrease (from l = g/(4π²f²))
        try:
            original_f = self.f
            perturbed_f = original_f * 1.1  # Increase by 10%
            
            original_l = self.g / (4 * math.pi**2 * original_f**2)
            perturbed_l = self.g / (4 * math.pi**2 * perturbed_f**2)
            
            if perturbed_l >= original_l:
                self.log(f"FAIL: Increasing f from {original_f} to {perturbed_f} should decrease l, but got l: {original_l} -> {perturbed_l}")
                success = False
            else:
                self.log(f"PASS: Increasing f decreases l as expected ({original_l} -> {perturbed_l})")
        except Exception as e:
            self.log(f"ERROR in f monotonicity check: {traceback.format_exc()}")
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Final verdict
        overall_success = test1 and test2 and test3
        
        # Check against expected answers (allow some tolerance)
        expected_l_cm = 6.3
        expected_n = 5
        
        try:
            # Convert l to cm for comparison
            l_cm = self.l * 100
            if not math.isclose(l_cm, expected_l_cm, rel_tol=1e-2, abs_tol=0.1):
                self.log(f"FAIL: l={l_cm} cm differs from expected {expected_l_cm} cm")
                overall_success = False
            else:
                self.log(f"PASS: l={l_cm:.1f} cm matches expected {expected_l_cm} cm")
        except Exception as e:
            self.log(f"ERROR in l comparison: {traceback.format_exc()}")
            overall_success = False
        
        try:
            if not math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=0.5):
                self.log(f"FAIL: n={self.n} differs from expected {expected_n}")
                overall_success = False
            else:
                self.log(f"PASS: n={self.n} matches expected {expected_n}")
        except Exception as e:
            self.log(f"ERROR in n comparison: {traceback.format_exc()}")
            overall_success = False
        
        self.log(f"Overall physics audit: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs