import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallbacks
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', 0.0))
        self.delta_l = getattr(solution_instance, 'delta_l', solution_instance.results.get('delta_l', 0.0))
        self.f_min = getattr(solution_instance, 'f_min', solution_instance.results.get('f_min', 0.0))
        
        # Constants from problem statement
        self.d = 12.0  # distance between sources (m)
        self.x = 5.0   # distance from S1 to observer (m)
        
        # Symbolic variables for laws
        self.lambda_val = sp.symbols('lambda_val', real=True)
        self.delta_phi = sp.symbols('delta_phi', real=True)
        self.delta_L = sp.symbols('delta_L', real=True)
        self.m = sp.symbols('m', integer=True)
        self.I = sp.symbols('I', real=True)
        self.I_max = sp.symbols('I_max', real=True)

    def test_invariants(self) -> List[str]:
        logs = []
        
        # Check 1: Wave equation v = f * lambda (numerical consistency)
        if self.f > 0:
            lambda_calc = self.v / self.f
            # Using destructive interference condition: delta_L = (2m+1)*lambda/2 for m=0
            lambda_from_delta = 2 * self.delta_l  # for first minimum (m=0)
            if not math.isclose(lambda_calc, lambda_from_delta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Wave equation inconsistency: v/f = {lambda_calc}, but from delta_L got lambda = {lambda_from_delta}")
        
        # Check 2: Destructive interference condition (symbolic + numerical)
        # Symbolic: delta_L should equal (2m+1)*lambda/2 for some integer m
        m_val = 0  # first minimum
        expected_delta_l = (2*m_val + 1) * self.v / (2 * self.f_min)
        if not math.isclose(self.delta_l, expected_delta_l, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Destructive interference failure: delta_L = {self.delta_l}, expected (2m+1)v/(2f) = {expected_delta_l} for m=0")
        
        # Check 3: Path difference consistency with geometry
        # Two possible positions: between sources (2m) or outside (12m)
        valid_deltas = [2.0, 12.0]
        if not any(math.isclose(self.delta_l, d, rel_tol=1e-2, abs_tol=1e-2) for d in valid_deltas):
            logs.append(f"Path difference {self.delta_l}m not consistent with geometry (expected 2m or 12m)")
        
        return logs

    def test_trace_consistency(self) -> List[str]:
        logs = []
        
        # Check step_4 formula: f = (2n+1)v/(2*delta_L)
        # For n=0, this becomes f_min = v/(2*delta_L)
        expected_f_min = self.v / (2 * self.delta_l)
        if not math.isclose(self.f_min, expected_f_min, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace inconsistency: f_min = {self.f_min}, expected v/(2ΔL) = {expected_f_min}")
        
        # Check step_5: minimum frequency for n=0 should be 85Hz if delta_L=2m
        if math.isclose(self.delta_l, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            expected_85 = 85.0
            if not math.isclose(self.f_min, expected_85, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Minimum frequency mismatch: f_min = {self.f_min}, expected 85Hz for ΔL=2m")
        
        # Check that f and f_min are related: f = (2n+1)*f_min for some integer n
        if self.f > 0 and self.f_min > 0:
            ratio = self.f / self.f_min
            # ratio should be approximately an odd integer (2n+1)
            nearest_odd = round((ratio - 1) / 2)
            expected_ratio = 2*nearest_odd + 1
            if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Frequency relationship: f/f_min = {ratio}, not an odd integer")
        
        return logs

    def test_monotonicity(self) -> List[str]:
        logs = []
        
        # Perturb distance between sources (d) and check frequency trend
        original_d = self.d
        perturbation = 1.0  # 1 meter increase
        
        # Store original results
        original_delta_l = self.delta_l
        original_f_min = self.f_min
        
        # Case 1: If observer is between sources (delta_L = |x - (d-x)|)
        if math.isclose(original_delta_l, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            # New path difference with increased d
            new_d = original_d + perturbation
            new_delta_l = abs(self.x - (new_d - self.x))
            new_f_min = self.v / (2 * new_delta_l)
            
            # As d increases, delta_L increases (if x < d/2), so f_min should decrease
            if new_delta_l > original_delta_l and new_f_min >= original_f_min:
                logs.append(f"Monotonicity violation: d↑ → ΔL↑ but f_min didn't decrease (was {original_f_min}, became {new_f_min})")
            elif new_delta_l < original_delta_l and new_f_min <= original_f_min:
                logs.append(f"Monotonicity violation: d↓ → ΔL↓ but f_min didn't increase (was {original_f_min}, became {new_f_min})")
        
        # Case 2: If observer is outside sources (delta_L = d)
        elif math.isclose(original_delta_l, 12.0, rel_tol=1e-2, abs_tol=1e-2):
            new_d = original_d + perturbation
            new_delta_l = new_d  # delta_L = d for outside case
            new_f_min = self.v / (2 * new_delta_l)
            
            # As d increases, delta_L increases, so f_min should decrease
            if new_f_min >= original_f_min:
                logs.append(f"Monotonicity violation: d↑ → f_min should decrease (was {original_f_min}, became {new_f_min})")
        
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        
        all_logs.extend(self.test_invariants())
        all_logs.extend(self.test_trace_consistency())
        all_logs.extend(self.test_monotonicity())
        
        success = len(all_logs) == 0
        return success, all_logs