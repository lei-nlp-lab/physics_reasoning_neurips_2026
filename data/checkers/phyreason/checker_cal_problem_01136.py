import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get() fallback
        self.solution = solution_instance
        self.logs = []
        
        # Extract inputs safely
        self.l = getattr(solution_instance, 'l', 0.50)
        self.R = getattr(solution_instance, 'R', 0.20)
        self.B = getattr(solution_instance, 'B', 0.40)
        self.v = getattr(solution_instance, 'v', 4.0)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.E = results.get('E', 0.0)
        self.I = results.get('I', 0.0)
        self.F = results.get('F', 0.0)
        self.F_ext = results.get('F_ext', results.get('F_{ext}', 0.0))
        
        # Expected values from steps analysis
        self.expected = {
            'E': 0.80,
            'I': 4.0,
            'F': 0.80,
            'F_ext': 0.80
        }
        
        # Define symbols for symbolic checks
        self.symbols = {
            'B': sp.Symbol('B'),
            'l': sp.Symbol('l'),
            'v': sp.Symbol('v'),
            'E': sp.Symbol('E'),
            'R': sp.Symbol('R'),
            'I': sp.Symbol('I'),
            'F': sp.Symbol('F'),
            'F_ext': sp.Symbol('F_ext')
        }
    
    def fuzzy_equal(self, a, b, msg=""):
        """Robust fuzzy comparison with physics tolerance."""
        try:
            a_float = float(a)
            b_float = float(b)
            is_close = math.isclose(a_float, b_float, rel_tol=1e-2, abs_tol=1e-2)
            if not is_close:
                self.logs.append(f"FAIL {msg}: {a_float} ≠ {b_float}")
            return is_close
        except (TypeError, ValueError):
            self.logs.append(f"ERROR {msg}: Cannot convert to float")
            return False
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Motional EMF: E = B * l * v
            expected_E = self.B * self.l * self.v
            emf_check = self.fuzzy_equal(self.E, expected_E, 
                f"EMF: E = B*l*v = {self.B}*{self.l}*{self.v} = {expected_E}")
            if emf_check:
                self.logs.append("PASS: Motional EMF formula E = B*l*v holds")
            
            # 2. Ohm's Law: I = E / R
            expected_I = self.E / self.R
            ohm_check = self.fuzzy_equal(self.I, expected_I,
                f"Ohm's Law: I = E/R = {self.E}/{self.R} = {expected_I}")
            if ohm_check:
                self.logs.append("PASS: Ohm's Law I = E/R holds")
            
            # 3. Ampere Force: F = B * I * l
            expected_F = self.B * self.I * self.l
            ampere_check = self.fuzzy_equal(self.F, expected_F,
                f"Ampere Force: F = B*I*l = {self.B}*{self.I}*{self.l} = {expected_F}")
            if ampere_check:
                self.logs.append("PASS: Ampere Force formula F = B*I*l holds")
            
            # 4. Newton's First Law: F_ext = F (net force zero for constant velocity)
            newton_check = self.fuzzy_equal(self.F_ext, self.F,
                f"Newton's 1st Law: F_ext = F = {self.F}")
            if newton_check:
                self.logs.append("PASS: Newton's 1st Law (F_ext = F) holds for constant velocity")
            
            # 5. Energy conservation: Power input = Power dissipated
            # Power from external force: P_ext = F_ext * v
            # Power dissipated in resistor: P_res = I^2 * R
            P_ext = self.F_ext * self.v
            P_res = self.I**2 * self.R
            energy_check = self.fuzzy_equal(P_ext, P_res,
                f"Energy conservation: F_ext*v = I^2*R => {P_ext} = {P_res}")
            if energy_check:
                self.logs.append("PASS: Energy conservation (power balance) holds")
            
            return all([emf_check, ohm_check, ampere_check, newton_check, energy_check])
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check each step against expected values from steps analysis
            checks = []
            
            # Step 1: EMF
            checks.append(self.fuzzy_equal(self.E, self.expected['E'],
                f"Step 1 EMF: got {self.E}, expected {self.expected['E']}V"))
            
            # Step 2: Current
            checks.append(self.fuzzy_equal(self.I, self.expected['I'],
                f"Step 2 Current: got {self.I}, expected {self.expected['I']}A"))
            
            # Step 3: Ampere force
            checks.append(self.fuzzy_equal(self.F, self.expected['F'],
                f"Step 3 Force: got {self.F}, expected {self.expected['F']}N"))
            
            # Step 4: External force (should equal Ampere force)
            checks.append(self.fuzzy_equal(self.F_ext, self.expected['F_ext'],
                f"Step 4 External force: got {self.F_ext}, expected {self.expected['F_ext']}N"))
            
            # Check internal consistency: I from E/R should match reported I
            I_from_E = self.E / self.R
            checks.append(self.fuzzy_equal(self.I, I_from_E,
                f"Internal I consistency: reported {self.I} vs calculated {I_from_E}"))
            
            # Check force direction logic: F_ext must equal F (no other forces)
            force_dir_check = self.fuzzy_equal(self.F_ext, self.F,
                f"Force direction: F_ext={self.F_ext} should equal F={self.F}")
            checks.append(force_dir_check)
            
            if all(checks):
                self.logs.append("PASS: All trace consistency checks passed")
            return all(checks)
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Test trend when perturbing input parameters."""
        try:
            original_v = self.v
            perturbations_passed = []
            
            # Perturbation 1: Increase velocity by 10%
            self.v *= 1.1
            new_E = self.B * self.l * self.v
            new_I = new_E / self.R
            new_F = self.B * new_I * self.l
            
            # Check monotonicity: E, I, F should all increase when v increases
            if new_E > self.E and new_I > self.I and new_F > self.F:
                self.logs.append("PASS: Increasing v increases E, I, and F")
                perturbations_passed.append(True)
            else:
                self.logs.append(f"FAIL: Monotonicity violated for v increase. "
                               f"Old: E={self.E}, I={self.I}, F={self.F}; "
                               f"New: E={new_E}, I={new_I}, F={new_F}")
                perturbations_passed.append(False)
            
            # Restore original v
            self.v = original_v
            
            # Perturbation 2: Decrease magnetic field by 20%
            original_B = self.B
            self.B *= 0.8
            new_E = self.B * self.l * self.v
            new_I = new_E / self.R
            new_F = self.B * new_I * self.l
            
            # All quantities should decrease when B decreases
            if new_E < self.E and new_I < self.I and new_F < self.F:
                self.logs.append("PASS: Decreasing B decreases E, I, and F")
                perturbations_passed.append(True)
            else:
                self.logs.append(f"FAIL: Monotonicity violated for B decrease. "
                               f"Old: E={self.E}, I={self.I}, F={self.F}; "
                               f"New: E={new_E}, I={new_I}, F={new_F}")
                perturbations_passed.append(False)
            
            # Restore original B
            self.B = original_B
            
            # Perturbation 3: Increase resistance by 50%
            original_R = self.R
            self.R *= 1.5
            # E unchanged (depends only on B, l, v)
            new_I = self.E / self.R
            new_F = self.B * new_I * self.l
            
            # I and F should decrease when R increases (E constant)
            if new_I < self.I and new_F < self.F:
                self.logs.append("PASS: Increasing R decreases I and F (E constant)")
                perturbations_passed.append(True)
            else:
                self.logs.append(f"FAIL: Monotonicity violated for R increase. "
                               f"Old: I={self.I}, F={self.F}; "
                               f"New: I={new_I}, F={new_F}")
                perturbations_passed.append(False)
            
            # Restore original R
            self.R = original_R
            
            return all(perturbations_passed)
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        success = True
        
        try:
            self.logs.append("=" * 50)
            self.logs.append("PhysicsChecker: Starting comprehensive audit")
            self.logs.append(f"Inputs: l={self.l}m, R={self.R}Ω, B={self.B}T, v={self.v}m/s")
            self.logs.append(f"Results: E={self.E}V, I={self.I}A, F={self.F}N, F_ext={self.F_ext}N")
            
            # Run all tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            success = test1 and test2 and test3
            
            # Summary
            self.logs.append("=" * 50)
            self.logs.append(f"Test Results: Invariants={test1}, "
                           f"Trace={test2}, Monotonicity={test3}")
            self.logs.append(f"OVERALL: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        return success, self.logs