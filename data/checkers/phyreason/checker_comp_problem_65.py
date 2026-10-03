import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all inputs and results from the solution instance.
        Uses getattr and .get() to avoid AttributeError.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants
        self.mu0 = 4e-7 * math.pi  # Magnetic constant
        
        # Input parameters (with safe defaults)
        self.r = getattr(solution_instance, 'r', self.results.get('r', 1.0))
        self.R = getattr(solution_instance, 'R', self.results.get('R', 1.0))
        self.E = getattr(solution_instance, 'E', self.results.get('E', 1.0))  # Electromotive force
        self.l = getattr(solution_instance, 'l', self.results.get('l', 1.0))  # Solenoid length
        self.A = getattr(solution_instance, 'A', self.results.get('A', 1.0))  # Cross-sectional area
        self.N = getattr(solution_instance, 'N', self.results.get('N', 1))    # Number of turns
        self.I = getattr(solution_instance, 'I', self.results.get('I', 1.0))  # Solenoid current
        
        # Computed results (with safe defaults)
        self.B_in = getattr(solution_instance, 'B_in', self.results.get('B_textin', None))
        self.Phi = getattr(solution_instance, 'Phi', self.results.get('Phi', None))
        self.J = getattr(solution_instance, 'J', self.results.get('J', None))
        self.Br = getattr(solution_instance, 'Br', self.results.get('Br', None))
        self.B_loop = getattr(solution_instance, 'B_loop', self.results.get('B_textloop', None))
        self.F1 = getattr(solution_instance, 'F1', self.results.get('F_1', None))
        self.F2 = getattr(solution_instance, 'F2', self.results.get('F_2', None))
        self.vecF2 = getattr(solution_instance, 'vecF2', self.results.get('vecF_2', None))
        self.Q_m = getattr(solution_instance, 'Q_m', self.results.get('Q_m', None))
        self.m = getattr(solution_instance, 'm', self.results.get('m', None))  # Magnetic dipole moment
        
        # Additional intermediate results if available
        self.Phi_in = getattr(solution_instance, 'Phi_in', self.results.get('Phi_in', None))
        self.B_loopz = getattr(solution_instance, 'B_loopz', self.results.get('B_loopz', None))
        self.deltaPhi_12 = getattr(solution_instance, 'deltaPhi_12', self.results.get('deltaPhi_12', None))
        self.deltaE_field = getattr(solution_instance, 'deltaE_field', self.results.get('deltaE_field', None))
        self.deltaE_total = getattr(solution_instance, 'deltaE_total', self.results.get('deltaE_total', None))
        self.V_ind_loop = getattr(solution_instance, 'V_ind_loop', self.results.get('V_textindtextloop', None))
        self.V_ind_solenoid = getattr(solution_instance, 'V_ind_solenoid', self.results.get('V_textindtextsolenoid', None))
        
        # For symbolic checks
        self.symbols = {
            'r': sp.symbols('r', positive=True),
            'R': sp.symbols('R', positive=True),
            'E': sp.symbols('E', positive=True),
            'l': sp.symbols('l', positive=True),
            'A': sp.symbols('A', positive=True),
            'N': sp.symbols('N', positive=True, integer=True),
            'I': sp.symbols('I', positive=True),
            'mu0': sp.symbols('mu0', positive=True)
        }
        
        # Expected formula for F1 (from target answer)
        self.expected_F1_expr = self.symbols['mu0'] * self.symbols['N'] * self.symbols['I'] * \
                                self.symbols['A'] * self.symbols['E'] / (2 * self.symbols['l'] * \
                                self.symbols['R'] * self.symbols['r'])
    
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental physical invariants:
        1. Magnetic field inside solenoid.
        2. Loop current from Ohm's law.
        3. Newton's third law (F2 = -F1).
        4. Energy conservation (if data available).
        """
        logs = []
        success = True
        
        # 1. Magnetic field inside solenoid: B_in = mu0 * n * I, n = N/l
        n = self.N / self.l
        B_in_expected = self.mu0 * n * self.I
        if self.B_in is not None:
            if not self.fuzzy_equal(self.B_in, B_in_expected):
                logs.append(f"Invariant violated: B_in = {self.B_in}, expected {B_in_expected}")
                success = False
            else:
                logs.append("Magnetic field inside solenoid: OK")
        else:
            logs.append("Skipping B_in check (not provided)")
        
        # 2. Loop current: J = E / R
        J_expected = self.E / self.R
        if self.J is not None:
            if not self.fuzzy_equal(self.J, J_expected):
                logs.append(f"Invariant violated: J = {self.J}, expected {J_expected}")
                success = False
            else:
                logs.append("Loop current J: OK")
        else:
            logs.append("Skipping J check (not provided)")
        
        # 3. Newton's third law: F2 = -F1 (if both provided)
        if self.F1 is not None and self.F2 is not None:
            if not self.fuzzy_equal(self.F2, -self.F1):
                logs.append(f"Newton's third law violated: F1 = {self.F1}, F2 = {self.F2} (expected F2 = -F1)")
                success = False
            else:
                logs.append("Newton's third law (F2 = -F1): OK")
        else:
            logs.append("Skipping Newton's third law check (F1 or F2 missing)")
        
        # 4. Energy conservation (if relevant data available)
        # From step 18: deltaE_total = -deltaE_field
        if self.deltaE_total is not None and self.deltaE_field is not None:
            if not self.fuzzy_equal(self.deltaE_total, -self.deltaE_field):
                logs.append(f"Energy conservation violated: deltaE_total = {self.deltaE_total}, deltaE_field = {self.deltaE_field}")
                success = False
            else:
                logs.append("Energy conservation (deltaE_total = -deltaE_field): OK")
        else:
            logs.append("Skipping energy conservation check (data missing)")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        success = True
        
        # Check that F1 matches the expected formula numerically
        if self.F1 is not None:
            F1_expected = (self.mu0 * self.N * self.I * self.A * self.E) / (2 * self.l * self.R * self.r)
            if not self.fuzzy_equal(self.F1, F1_expected):
                logs.append(f"F1 mismatch: computed {self.F1}, expected {F1_expected}")
                success = False
            else:
                logs.append("F1 matches expected formula: OK")
            
            # If intermediate results are available, check consistency
            # For example, check B_loop and Q_m if available
            if self.B_loop is not None and self.Q_m is not None:
                # From step 8: F1 = Q_m * B_loop
                F1_from_Qm_Bloop = self.Q_m * self.B_loop
                if not self.fuzzy_equal(self.F1, F1_from_Qm_Bloop):
                    logs.append(f"F1 inconsistent with Q_m and B_loop: {self.F1} vs {F1_from_Qm_Bloop}")
                    success = False
                else:
                    logs.append("F1 consistent with Q_m * B_loop: OK")
            
            # Check magnetic flux from solenoid (step 2)
            if self.Phi is not None and self.B_in is not None:
                Phi_expected = self.B_in * self.A
                if not self.fuzzy_equal(self.Phi, Phi_expected):
                    logs.append(f"Phi mismatch: {self.Phi} vs {Phi_expected}")
                    success = False
                else:
                    logs.append("Magnetic flux Phi consistent: OK")
            
            # Check B(r) from step 3 (approximation)
            if self.Br is not None:
                # B(r) = Phi / (4π r^2) = (mu0 * N * I * A) / (4π * l * r^2)
                Br_expected = (self.mu0 * self.N * self.I * self.A) / (4 * math.pi * self.l * self.r**2)
                if not self.fuzzy_equal(self.Br, Br_expected):
                    logs.append(f"B(r) mismatch: {self.Br} vs {Br_expected}")
                    success = False
                else:
                    logs.append("B(r) consistent: OK")
        else:
            logs.append("Skipping trace consistency (F1 missing)")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb an input (e.g., current I) and check if F1 changes as expected.
        We do not modify the original solution; instead, we recompute using the formula.
        """
        logs = []
        success = True
        
        # Perturb I by +10%
        I_original = self.I
        I_perturbed = I_original * 1.1
        
        # Compute F1 for both using the formula
        F1_original = (self.mu0 * self.N * I_original * self.A * self.E) / (2 * self.l * self.R * self.r)
        F1_perturbed = (self.mu0 * self.N * I_perturbed * self.A * self.E) / (2 * self.l * self.R * self.r)
        
        # Since F1 is proportional to I, F1 should increase when I increases
        if F1_perturbed <= F1_original:
            logs.append(f"Monotonicity violated: F1({I_original}) = {F1_original}, F1({I_perturbed}) = {F1_perturbed}")
            success = False
        else:
            logs.append(f"Monotonicity OK: F1 increases with I (from {F1_original} to {F1_perturbed})")
        
        # Additionally, check that F1 is proportional to other variables
        # Perturb A (area) and check
        A_original = self.A
        A_perturbed = A_original * 1.1
        F1_A_original = (self.mu0 * self.N * self.I * A_original * self.E) / (2 * self.l * self.R * self.r)
        F1_A_perturbed = (self.mu0 * self.N * self.I * A_perturbed * self.E) / (2 * self.l * self.R * self.r)
        if F1_A_perturbed <= F1_A_original:
            logs.append(f"Monotonicity violated for A: F1({A_original}) = {F1_A_original}, F1({A_perturbed}) = {F1_A_perturbed}")
            success = False
        else:
            logs.append(f"Monotonicity for A: OK")
        
        # Perturb r (radius) and check: F1 is inversely proportional to r
        r_original = self.r
        r_perturbed = r_original * 1.1
        F1_r_original = (self.mu0 * self.N * self.I * self.A * self.E) / (2 * self.l * self.R * r_original)
        F1_r_perturbed = (self.mu0 * self.N * self.I * self.A * self.E) / (2 * self.l * self.R * r_perturbed)
        if F1_r_perturbed >= F1_r_original:
            logs.append(f"Monotonicity violated for r: F1({r_original}) = {F1_r_original}, F1({r_perturbed}) = {F1_r_perturbed}")
            success = False
        else:
            logs.append(f"Monotonicity for r: OK (F1 decreases as r increases)")
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        logs = ["=== PhysicsChecker Report ==="]
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            logs.append(f"\n--- {test_name} ---")
            try:
                success, test_logs = test_func()
                logs.extend(test_logs)
                if not success:
                    overall_success = False
                    logs.append(f"FAILED: {test_name}")
                else:
                    logs.append(f"PASSED: {test_name}")
            except Exception as e:
                logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, logs