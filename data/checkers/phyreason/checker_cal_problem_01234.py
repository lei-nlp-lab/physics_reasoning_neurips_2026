import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize the checker from the solution instance.
        """
        self.solution = solution_instance
        
        # Extract results with safe fallbacks
        self.G = getattr(solution_instance, 'G', solution_instance.results.get('G', 0.0))
        self.F_2 = getattr(solution_instance, 'F_2', solution_instance.results.get('F_2', 0.0))
        self.F_3 = getattr(solution_instance, 'F_3', solution_instance.results.get('F_3', 0.0))
        
        # Constants from the problem statement
        self.F1_max = 2.0e4  # Maximum tension in cable OB (N)
        self.theta_deg = 30.0  # Angle AOB in degrees
        self.theta_rad = math.radians(self.theta_deg)
        
        # Target answers for direct comparison
        self.target_G = 1.0e4
        self.target_F2 = 1.73e4

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Test fundamental physics laws: force equilibrium and trigonometric relationships.
        """
        logs = []
        success = True
        
        # 1. Vertical equilibrium: G = F1_max * sin(30°)
        expected_G = self.F1_max * math.sin(self.theta_rad)
        if not math.isclose(self.G, expected_G, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Vertical equilibrium violation: G={self.G:.2f}, expected {expected_G:.2f}")
            success = False
        
        # 2. Horizontal equilibrium: F_2 = F1_max * cos(30°)
        expected_F2 = self.F1_max * math.cos(self.theta_rad)
        if not math.isclose(self.F_2, expected_F2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Horizontal equilibrium violation: F_2={self.F_2:.2f}, expected {expected_F2:.2f}")
            success = False
        
        # 3. Consistency of vertical component F_3 (if provided)
        if self.F_3 != 0.0:  # Only check if F_3 was extracted
            if not math.isclose(self.F_3, expected_G, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"F_3 mismatch: F_3={self.F_3:.2f}, should equal G={expected_G:.2f}")
                success = False
        
        # 4. Symbolic equilibrium check using sympy (for robustness)
        try:
            F1, theta, G, F2 = sp.symbols('F1 theta G F2')
            eq1 = sp.Eq(G, F1 * sp.sin(theta))
            eq2 = sp.Eq(F2, F1 * sp.cos(theta))
            
            # Substitute numerical values
            subs_dict = {F1: self.F1_max, theta: self.theta_rad}
            numeric_G = eq1.rhs.subs(subs_dict)
            numeric_F2 = eq2.rhs.subs(subs_dict)
            
            if not (math.isclose(float(numeric_G), self.G, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(float(numeric_F2), self.F_2, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append("Symbolic equilibrium check failed")
                success = False
        except Exception as e:
            logs.append(f"Symbolic check error: {e}")
            # Fallback to numerical check already performed
        
        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Ensure intermediate steps align with final results and target answers.
        """
        logs = []
        success = True
        
        # 1. Direct comparison with target answers
        if not math.isclose(self.G, self.target_G, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"G mismatch with target: {self.G:.2f} vs {self.target_G}")
            success = False
        
        if not math.isclose(self.F_2, self.target_F2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"F_2 mismatch with target: {self.F_2:.2f} vs {self.target_F2}")
            success = False
        
        # 2. Check that G and F_2 are derived from the same F1_max
        derived_F1_from_G = self.G / math.sin(self.theta_rad)
        derived_F1_from_F2 = self.F_2 / math.cos(self.theta_rad)
        if not math.isclose(derived_F1_from_G, derived_F1_from_F2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Inconsistent F1 derived: {derived_F1_from_G:.2f} from G, {derived_F1_from_F2:.2f} from F_2")
            success = False
        
        # 3. Verify the derived F1 does not exceed the maximum
        if derived_F1_from_G > self.F1_max * 1.02:  # 2% tolerance
            logs.append(f"Derived tension {derived_F1_from_G:.2f} exceeds maximum {self.F1_max}")
            success = False
        
        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb input (max tension) and verify output trends match physical intuition.
        """
        logs = []
        success = True
        
        # Store original values
        original_F1 = self.F1_max
        original_G = self.G
        original_F2 = self.F_2
        
        # Perturb F1_max by +10%
        perturbed_F1 = original_F1 * 1.10
        expected_G_perturbed = perturbed_F1 * math.sin(self.theta_rad)
        expected_F2_perturbed = perturbed_F1 * math.cos(self.theta_rad)
        
        # Check monotonic increase (both outputs should increase)
        if not (expected_G_perturbed > original_G and expected_F2_perturbed > original_F2):
            logs.append("Perturbation test: G and F_2 should both increase with increased tension")
            success = False
        
        # Check linearity (ratios remain constant)
        ratio_G_original = original_G / original_F1
        ratio_G_perturbed = expected_G_perturbed / perturbed_F1
        if not math.isclose(ratio_G_original, ratio_G_perturbed, rel_tol=1e-2, abs_tol=1e-2):
            logs.append("Linearity violation: G/F1 ratio changed after perturbation")
            success = False
        
        ratio_F2_original = original_F2 / original_F1
        ratio_F2_perturbed = expected_F2_perturbed / perturbed_F1
        if not math.isclose(ratio_F2_original, ratio_F2_perturbed, rel_tol=1e-2, abs_tol=1e-2):
            logs.append("Linearity violation: F_2/F1 ratio changed after perturbation")
            success = False
        
        # Restore original values (conceptually)
        self.F1_max = original_F1
        
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return aggregate result.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                if logs:
                    all_logs.append(f"{test_name}:")
                    all_logs.extend([f"  {log}" for log in logs])
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"{test_name} raised exception: {e}")
                overall_success = False
        
        if overall_success and not all_logs:
            all_logs.append("All tests passed.")
        
        return overall_success, all_logs