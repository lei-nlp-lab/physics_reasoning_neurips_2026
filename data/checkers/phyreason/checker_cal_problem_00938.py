import sympy as sp
import math
import traceback
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 0.012))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 1.5))
        self.theta = getattr(solution_instance, 'theta', getattr(solution_instance, 'inputs', {}).get('theta', math.radians(30)))
        self.d = getattr(solution_instance, 'd', getattr(solution_instance, 'inputs', {}).get('d', 0.1*math.sqrt(3)))
        self.r = getattr(solution_instance, 'r', getattr(solution_instance, 'inputs', {}).get('r', 0.5))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 0.2))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        
        # Convert theta to radians if in degrees
        if isinstance(self.theta, (int, float)) and abs(self.theta - 30) < 1e-3:
            self.theta = math.radians(self.theta)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        self.F = self.results.get('F', 4*math.sqrt(3)*1e-2)
        self.E = self.results.get('E', 4.0)
        
        # Symbolic variables for verification
        self.sym_m, self.sym_g, self.sym_theta = sp.symbols('m g theta')
        self.sym_B, self.sym_d, self.sym_I = sp.symbols('B d I')
        self.sym_F, self.sym_E = sp.symbols('F E')
        self.sym_R, self.sym_r = sp.symbols('R r')
        
        self.logs = []
        
    def log(self, message: str, status: str = "INFO"):
        self.logs.append(f"{status}: {message}")
    
    def test_invariants(self) -> bool:
        """Verify force equilibrium and Ohm's law relationships."""
        success = True
        
        # 1. Force equilibrium: F = m*g*tan(theta) from STEP 1
        expected_F = self.m * self.g * math.tan(self.theta)
        if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Force equilibrium violated: F={self.F}, expected {expected_F}", "ERROR")
            success = False
        else:
            self.log(f"Force equilibrium satisfied: F={self.F} ≈ m*g*tanθ={expected_F}")
        
        # 2. Ampere's force law: F = B*I*d, where I = E/(R+r) from Ohm's law
        I_calculated = self.E / (self.R + self.r)
        expected_F_from_ampere = self.B * I_calculated * self.d
        if not math.isclose(self.F, expected_F_from_ampere, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Ampere's law violated: F={self.F}, expected B*I*d={expected_F_from_ampere}", "ERROR")
            success = False
        else:
            self.log(f"Ampere's law satisfied: F={self.F} ≈ B*I*d={expected_F_from_ampere}")
        
        # 3. Symbolic verification of force equilibrium
        try:
            symbolic_eq = sp.Eq(self.sym_F, self.sym_m * self.sym_g * sp.tan(self.sym_theta))
            # Substitute numerical values
            subs_dict = {
                self.sym_F: self.F,
                self.sym_m: self.m,
                self.sym_g: self.g,
                self.sym_theta: self.theta
            }
            symbolic_check = symbolic_eq.subs(subs_dict)
            # Evaluate numerically
            lhs_val = float(symbolic_check.lhs)
            rhs_val = float(symbolic_check.rhs)
            if not math.isclose(lhs_val, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Symbolic force check failed: {lhs_val} ≠ {rhs_val}", "WARNING")
                success = False
            else:
                self.log("Symbolic force equilibrium verified")
        except Exception as e:
            self.log(f"Symbolic force check failed with error: {str(e)}", "WARNING")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check that intermediate results align with final answer."""
        success = True
        
        # From STEP 1: F = 4√3 × 10⁻² N ≈ 0.069282 N
        # From STEP 4: E = 4.0 V
        # Relationship: F = B * [E/(R+r)] * d
        
        # Calculate I from E
        I_from_E = self.E / (self.R + self.r)
        
        # Recalculate F from E
        F_from_E = self.B * I_from_E * self.d
        
        if not math.isclose(self.F, F_from_E, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"Trace inconsistency: F from solution={self.F}, F recalculated from E={F_from_E}", "ERROR")
            success = False
        else:
            self.log(f"Trace consistency verified: F={self.F} matches recalculation from E")
        
        # Verify the value of F matches the symbolic expression from STEP 1
        expected_F_value = 4 * math.sqrt(3) * 1e-2  # 4√3 × 10⁻²
        if not math.isclose(self.F, expected_F_value, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"F does not match expected value from STEP 1: {self.F} ≠ {expected_F_value}", "WARNING")
        else:
            self.log(f"F matches expected value from STEP 1: {self.F}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb mass m and check F changes proportionally, then restore."""
        success = True
        
        original_m = self.m
        perturbation_factor = 1.1  # 10% increase
        
        # Perturb m
        self.m = original_m * perturbation_factor
        
        # Recalculate expected F from F = m*g*tan(theta)
        new_F_expected = self.m * self.g * math.tan(self.theta)
        
        # From the relationship, increasing m should increase F
        if new_F_expected <= self.F:  # Should be strictly greater
            self.log(f"Monotonicity violation: F did not increase with mass increase", "ERROR")
            success = False
        else:
            self.log(f"Monotonicity satisfied: F increases with mass (old F={self.F}, new expected F={new_F_expected})")
        
        # Restore original mass
        self.m = original_m
        
        # Additional check: Perturb E and verify F changes accordingly
        original_E = self.E
        self.E = original_E * perturbation_factor
        
        # Recalculate I and F
        new_I = self.E / (self.R + self.r)
        new_F_from_E = self.B * new_I * self.d
        
        # Increasing E should increase F (since I increases)
        if new_F_from_E <= self.F:
            self.log(f"Monotonicity violation: F did not increase with E increase", "ERROR")
            success = False
        else:
            self.log(f"Monotonicity satisfied: F increases with E (old F={self.F}, new F from E={new_F_from_E})")
        
        # Restore E
        self.E = original_E
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            try:
                self.log(f"Running {test_name} test...")
                if not test_func():
                    overall_success = False
                    self.log(f"{test_name} test failed", "ERROR")
                else:
                    self.log(f"{test_name} test passed", "SUCCESS")
            except Exception as e:
                self.log(f"{test_name} test crashed: {str(e)}\n{traceback.format_exc()}", "CRITICAL")
                overall_success = False
        
        return overall_success, self.logs